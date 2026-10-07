// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "cuda_block.h"

#include <sonicboom/model/exec.h>
#include <sonicboom/nn/cuda_elementwise.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include "../nn/cuda_resident.h"
#include "../quant/cuda_resident.h"

#include <cuda_runtime.h>

#include <utility>
#include <vector>

// Device-resident single-token forward (Phase 6a Option A).
//
// MIRRORS run_block_core in core/src/model/exec.cpp: the same per-block
// attention / gated-FFN / per-layer-gate chain, but with every activation kept
// in device buffers and chained through the device-resident *_dev kernels (no
// per-op H2D/D2H). The host only synchronizes once at the end (the final hidden
// state). The arithmetic is bit-identical to the existing CUDA path: the same
// kernels run, only the intermediate host round-trips are removed (an f32
// memcpy round-trip is lossless), except the rms_norm `inv` which is now
// computed by a one-thread device kernel with the identical IEEE expression.
//
// The per-layer gate's embed_per_layer is still invoked via the host reference
// (bit-identical); hoisting its per-layer token-embd dequant to once-per-token
// + a device-side combine is a follow-up.

namespace sonicboom::model {

namespace {

// One device buffer per named activation, allocated once to the model's max
// sizes and reused across blocks/tokens.
struct DevBuf {
  float* inp = nullptr;         // d
  float* h = nullptr;           // d
  float* q = nullptr;           // n_q * head_dim (max)
  float* k = nullptr;           // n_kv * head_dim (max)
  float* v = nullptr;           // n_kv * head_dim (max)
  float* attn = nullptr;        // n_q * head_dim (max)
  float* attn_proj = nullptr;   // d
  float* o = nullptr;           // d
  float* x1 = nullptr;          // d
  float* f = nullptr;           // d
  float* up = nullptr;          // ff
  float* gate = nullptr;        // ff
  float* ffn_hidden = nullptr;  // ff
  float* ffn_out = nullptr;     // d
  float* f2 = nullptr;          // d
  float* x2 = nullptr;          // d
  float* inp_ple = nullptr;     // per
  float* g0 = nullptr;          // per
  float* g1 = nullptr;          // d
  float* g2 = nullptr;          // d
  float* x3 = nullptr;          // d
  float* out = nullptr;         // d
  float* donor_swa = nullptr;   // n_kv * head_dim (max)
  float* donor_global = nullptr;// n_kv * head_dim (max)

  int d = 0, ff = 0, per = 0, qw = 0, kvw = 0;  // dims the buffers were sized for
};

DevBuf g_buf;

void free_buf(DevBuf& b) {
  float* ps[] = {b.inp,          b.h,          b.q,          b.k,
                 b.v,            b.attn,       b.attn_proj,  b.o,
                 b.x1,           b.f,          b.up,         b.gate,
                 b.ffn_hidden,   b.ffn_out,    b.f2,         b.x2,
                 b.inp_ple,      b.g0,         b.g1,         b.g2,
                 b.x3,           b.out,        b.donor_swa,  b.donor_global};
  for (float* p : ps)
    if (p) cudaFree(p);
  b = DevBuf{};
}

bool alloc_float(float*& p, int n) {
  return cudaMalloc(reinterpret_cast<void**>(&p), (size_t)n * sizeof(float)) == cudaSuccess;
}

bool ensure_buffers(const Gemma4Model& m) {
  const int d = int(m.config.embedding_length);
  const int ff = int(m.config.feed_forward_length);
  const int per = int(m.config.per_layer_input);
  const int qw = int(m.config.head_count) * int(m.config.head_dim_global);
  const int kvw = int(m.config.head_count_kv) * int(m.config.head_dim_global);
  if (g_buf.d == d && g_buf.ff == ff && g_buf.per == per && g_buf.qw == qw &&
      g_buf.kvw == kvw)
    return true;
  free_buf(g_buf);
  g_buf.d = d;
  g_buf.ff = ff;
  g_buf.per = per;
  g_buf.qw = qw;
  g_buf.kvw = kvw;
  return alloc_float(g_buf.inp, d) && alloc_float(g_buf.h, d) &&
         alloc_float(g_buf.q, qw) && alloc_float(g_buf.k, kvw) &&
         alloc_float(g_buf.v, kvw) && alloc_float(g_buf.attn, qw) &&
         alloc_float(g_buf.attn_proj, d) && alloc_float(g_buf.o, d) &&
         alloc_float(g_buf.x1, d) && alloc_float(g_buf.f, d) &&
         alloc_float(g_buf.up, ff) && alloc_float(g_buf.gate, ff) &&
         alloc_float(g_buf.ffn_hidden, ff) && alloc_float(g_buf.ffn_out, d) &&
         alloc_float(g_buf.f2, d) && alloc_float(g_buf.x2, d) &&
         alloc_float(g_buf.inp_ple, per) && alloc_float(g_buf.g0, per) &&
         alloc_float(g_buf.g1, d) && alloc_float(g_buf.g2, d) &&
         alloc_float(g_buf.x3, d) && alloc_float(g_buf.out, d) &&
         alloc_float(g_buf.donor_swa, kvw) && alloc_float(g_buf.donor_global, kvw);
}

} // namespace

bool cuda_forward_resident(const Gemma4Model& m, uint64_t token_id, uint64_t pos,
                           std::span<float> out) {
  if (!nn::cuda::available() || !quant::cuda::available()) return false;
  if (m.weights.blocks.empty()) return false;
  const int d = int(m.config.embedding_length);
  if (out.size() < (size_t)d) return false;
  if (!ensure_buffers(m)) return false;

  // Token embedding (host reference) -> device residual stream.
  std::vector<float> inpL(d);
  if (!embed_token(m, token_id, inpL)) return false;
  if (cudaMemcpy(g_buf.inp, inpL.data(), d * sizeof(float), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return false;

  const uint32_t n_kv = m.config.n_layer_kv();
  const uint32_t donor_swa_layer = n_kv - 2;
  const uint32_t donor_global_layer = n_kv - 1;
  const float eps = m.config.rms_epsilon;

  // Slice 1: the per-layer embed combines on the host (bit-identical); only the
  // result is uploaded.
  std::vector<float> inp_ple_host(m.config.per_layer_input);

  for (uint32_t l = 0; l < m.config.block_count; ++l) {
    const LayerConfig& lc = m.plan[l];
    const BlockWeights& bw = m.weights.blocks[l];
    const int n_q = int(lc.n_heads_q);
    const int n_kvh = int(lc.n_heads_kv);
    const int hd = int(lc.head_dim);
    const int qw = n_q * hd;
    const int kvw = n_kvh * hd;
    const int ff = int(m.config.feed_forward_length);
    const int per = int(m.config.per_layer_input);
    const bool is_kv = m.config.has_kv(l);
    const std::span<const float> freq_factors =
        lc.attention == AttentionKind::Global ? m.weights.rope_freqs
                                              : std::span<const float>{};

    // --- attention -----------------------------------------------------------
    if (!nn::cuda::rms_norm_dev(bw.attn_norm, g_buf.inp, eps, g_buf.h, d)) return false;
    if (!quant::cuda::matvec_f32_dev(bw.attn_q, g_buf.h, g_buf.q)) return false;
    if (!nn::cuda::rms_norm_heads_dev(bw.attn_q_norm, g_buf.q, eps, g_buf.q, hd, n_q))
      return false;
    if (!nn::cuda::rope_neox_heads_dev(g_buf.q, hd, n_q, pos, lc.rope_base, 1.0f,
                                       freq_factors))
      return false;

    if (is_kv) {
      if (!quant::cuda::matvec_f32_dev(bw.attn_k, g_buf.h, g_buf.k)) return false;
      if (!quant::cuda::matvec_f32_dev(bw.attn_v, g_buf.h, g_buf.v)) return false;
      if (!nn::cuda::rms_norm_heads_dev(bw.attn_k_norm, g_buf.k, eps, g_buf.k, hd, n_kvh))
        return false;
      if (!nn::cuda::rms_norm_heads_dev(std::span<const float>{}, g_buf.v, eps, g_buf.v,
                                        hd, n_kvh))
        return false;
      if (!nn::cuda::cast_fp16_dev(g_buf.v, kvw)) return false;
      if (!nn::cuda::rope_neox_heads_dev(g_buf.k, hd, n_kvh, pos, lc.rope_base, 1.0f,
                                         freq_factors))
        return false;
      if (!nn::cuda::gqa_broadcast_dev(g_buf.v, g_buf.attn, n_q, n_kvh, hd)) return false;
      // Retain the fp16-normalized V of the two donor layers for the shared layers.
      if (l == donor_swa_layer) {
        if (cudaMemcpy(g_buf.donor_swa, g_buf.v, kvw * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess)
          return false;
      } else if (l == donor_global_layer) {
        if (cudaMemcpy(g_buf.donor_global, g_buf.v, kvw * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess)
          return false;
      }
    } else {
      const float* donor =
          m.config.sliding_window_pattern[l] ? g_buf.donor_swa : g_buf.donor_global;
      if (!nn::cuda::gqa_broadcast_dev(donor, g_buf.attn, n_q, n_kvh, hd)) return false;
    }
    if (!quant::cuda::matvec_f32_dev(bw.attn_output, g_buf.attn, g_buf.attn_proj))
      return false;
    if (!nn::cuda::rms_norm_dev(bw.post_attention_norm, g_buf.attn_proj, eps, g_buf.o, d))
      return false;
    if (!nn::cuda::add_dev(g_buf.o, g_buf.inp, g_buf.x1, d)) return false;

    // --- gated feed-forward ---------------------------------------------------
    if (!nn::cuda::rms_norm_dev(bw.ffn_norm, g_buf.x1, eps, g_buf.f, d)) return false;
    if (!quant::cuda::matvec_f32_dev(bw.ffn_up, g_buf.f, g_buf.up)) return false;
    if (!quant::cuda::matvec_f32_dev(bw.ffn_gate, g_buf.f, g_buf.gate)) return false;
    if (!nn::cuda::gelu_fp16_dev(g_buf.gate, g_buf.gate, ff)) return false;
    if (!nn::cuda::mul_dev(g_buf.gate, g_buf.up, g_buf.ffn_hidden, ff)) return false;
    if (!quant::cuda::matvec_f32_dev(bw.ffn_down, g_buf.ffn_hidden, g_buf.ffn_out))
      return false;
    if (!nn::cuda::rms_norm_dev(bw.post_ffw_norm, g_buf.ffn_out, eps, g_buf.f2, d))
      return false;
    if (!nn::cuda::add_dev(g_buf.f2, g_buf.x1, g_buf.x2, d)) return false;

    // --- per-layer gate --------------------------------------------------------
    if (!embed_per_layer(m, l, token_id, inp_ple_host, true)) return false;
    if (cudaMemcpy(g_buf.inp_ple, inp_ple_host.data(), per * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    if (!nn::cuda::matvec_f32_dev(bw.inp_gate, d, g_buf.x2, g_buf.g0, per)) return false;
    if (!nn::cuda::gelu_fp16_dev(g_buf.g0, g_buf.g0, per)) return false;
    if (!nn::cuda::mul_dev(g_buf.g0, g_buf.inp_ple, g_buf.g0, per)) return false;
    if (!nn::cuda::matvec_f32_dev(bw.proj, per, g_buf.g0, g_buf.g1, d)) return false;
    if (!nn::cuda::rms_norm_dev(bw.post_norm, g_buf.g1, eps, g_buf.g2, d)) return false;
    if (!nn::cuda::add_dev(g_buf.x2, g_buf.g2, g_buf.x3, d)) return false;
    if (!bw.layer_output_scale.empty()) {
      if (!nn::cuda::scale_dev(g_buf.x3, bw.layer_output_scale[0], g_buf.out, d))
        return false;
    } else {
      if (cudaMemcpy(g_buf.out, g_buf.x3, d * sizeof(float), cudaMemcpyDeviceToDevice) !=
          cudaSuccess)
        return false;
    }

    // Residual stream: this block's out becomes the next block's input.
    std::swap(g_buf.inp, g_buf.out);
  }

  // The final residual is the last block's l_out.
  if (cudaMemcpy(out.data(), g_buf.inp, d * sizeof(float), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return false;
  return true;
}

void cuda_resident_clear() { free_buf(g_buf); }

} // namespace sonicboom::model
