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

#include <sonicboom/model/exec.h>
#include <sonicboom/model/plan.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <sonicboom/dtype.h>
#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/attention.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/embedding.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/quant/dequant.h>
#include <sonicboom/quant/quantized_matmul.h>
#ifdef SONICBOOM_USE_CUDA
#include <sonicboom/nn/cuda_elementwise.h>
#include "cuda_block.h"
#endif

namespace sonicboom::model {

// --- temporary profiling (remove after measuring) ---------------------------
namespace {
double prof_now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
struct ExecProf {
  long n_block = 0;
  long n_embed = 0;
  double t_block = 0.0;    // run_block_core total (includes everything below)
  double t_embed_pl = 0.0; // embed_per_layer total (nested inside t_block)
  double t_attn = 0.0;     // attention section (nested inside t_block)
  double t_ffn = 0.0;      // gated feed-forward section
  double t_gate = 0.0;     // per-layer gate section (incl. embed_per_layer)
  ~ExecProf() {
    if (n_block)
      std::fprintf(stderr,
                   "[exec-prof] %ld blocks: block %.2f ms | attn %.2f ffn %.2f "
                   "gate %.2f (embed_per_layer %.2f) ms\n",
                   n_block, t_block, t_attn, t_ffn, t_gate, t_embed_pl);
  }
} g_prof;
struct ScopeTimer {
  double* sink;
  double t0 = prof_now_ms();
  explicit ScopeTimer(double* s) : sink(s) {}
  ~ScopeTimer() { *sink += prof_now_ms() - t0; }
};

// Per-op dispatch: route to the CUDA kernel when `use_cuda` is set (Cuda backend
// active AND a device present), else fall through to the CPU reference. Each
// nn::cuda::* mirrors the matching nn::* signature and reproduces its arithmetic
// to fp32 tolerance (the CPU reference accumulates mean/matvec in double while
// the kernel accumulates in fp32 — the same relaxation the K-quant CUDA gemv
// makes). These are file-local: the model layer is the only consumer.
#ifdef SONICBOOM_USE_CUDA
static bool rms_norm_d(bool use_cuda, std::span<const float> x,
                       std::span<const float> w, float eps, std::span<float> y) {
  if (use_cuda) return nn::cuda::rms_norm(x, w, eps, y);
  return nn::rms_norm(x, w, eps, y);
}
static bool gelu_fp16_d(bool use_cuda, std::span<const float> x,
                        std::span<float> y) {
  if (use_cuda) return nn::cuda::gelu_fp16(x, y);
  return nn::gelu_fp16(x, y);
}
static bool mul_d(bool use_cuda, std::span<const float> a, std::span<const float> b,
                  std::span<float> y) {
  if (use_cuda) return nn::cuda::mul(a, b, y);
  return nn::mul(a, b, y);
}
static bool add_d(bool use_cuda, std::span<const float> a, std::span<const float> b,
                  std::span<float> y) {
  if (use_cuda) return nn::cuda::add(a, b, y);
  return nn::add(a, b, y);
}
static bool scale_d(bool use_cuda, std::span<const float> x, float s,
                    std::span<float> y) {
  if (use_cuda) return nn::cuda::scale(x, s, y);
  return nn::scale(x, s, y);
}
static bool rope_heads_d(bool use_cuda, std::span<float> x, uint64_t head_dim,
                         uint64_t pos, float base,
                         std::span<const float> ff) {
  if (use_cuda) return nn::cuda::rope_neox_heads(x, head_dim, pos, base, 1.0f, ff);
  // CPU: the per-head rotation, split across the thread pool by rope_neox_heads
  // (trig-heavy RoPE is big enough to amortise the handshake, unlike the trivial
  // elementwise ops).
  return nn::rope_neox_heads(x, head_dim, pos, base, 1.0f, ff);
}
static bool matvec_f32_d(bool use_cuda, std::span<const float> W, uint64_t cols,
                         std::span<const float> x, std::span<float> y) {
  if (use_cuda) return nn::cuda::matvec_f32(W, cols, x, y);
  return nn::matvec_f32(W, cols, x, y);
}
static bool matvec_bf16_d(bool use_cuda, std::span<const uint16_t> W, uint64_t cols,
                          std::span<const float> x, std::span<float> y) {
  if (use_cuda) return nn::cuda::matvec_bf16(W, cols, x, y);
  return nn::matvec_bf16(W, cols, x, y);
}
#else
static bool rms_norm_d(bool, std::span<const float> x, std::span<const float> w,
                       float eps, std::span<float> y) {
  return nn::rms_norm(x, w, eps, y);
}
static bool gelu_fp16_d(bool, std::span<const float> x, std::span<float> y) {
  return nn::gelu_fp16(x, y);
}
static bool mul_d(bool, std::span<const float> a, std::span<const float> b,
                  std::span<float> y) {
  return nn::mul(a, b, y);
}
static bool add_d(bool, std::span<const float> a, std::span<const float> b,
                  std::span<float> y) {
  return nn::add(a, b, y);
}
static bool scale_d(bool, std::span<const float> x, float s, std::span<float> y) {
  return nn::scale(x, s, y);
}
static bool rope_heads_d(bool, std::span<float> x, uint64_t head_dim, uint64_t pos,
                         float base, std::span<const float> ff) {
  return nn::rope_neox_heads(x, head_dim, pos, base, 1.0f, ff);
}
static bool matvec_f32_d(bool, std::span<const float> W, uint64_t cols,
                         std::span<const float> x, std::span<float> y) {
  return nn::matvec_f32(W, cols, x, y);
}
static bool matvec_bf16_d(bool, std::span<const uint16_t> W, uint64_t cols,
                          std::span<const float> x, std::span<float> y) {
  return nn::matvec_bf16(W, cols, x, y);
}
#endif
} // namespace

bool embed_token(const Gemma4Model& m, uint64_t token_id, std::span<float> out) {
  if (m.weights.blocks.empty()) return false;
  const uint32_t d = m.config.embedding_length;
  if (out.size() < d) return false;
  if (!nn::embedding_f32(m.weights.token_embd, token_id, out)) return false;
  // inpL = tok_embd * sqrt(n_embd).
  return nn::scale(out, std::sqrt(float(d)), out);
}

bool embed_per_layer(const Gemma4Model& m, uint32_t layer, uint64_t token_id,
                     std::span<float> out, bool use_cuda) {
  ScopeTimer _t(&g_prof.t_embed_pl);
  ++g_prof.n_embed;
  if (m.weights.blocks.empty() || layer >= m.config.block_count) return false;
  const uint32_t d = m.config.embedding_length;
  const uint32_t per = m.config.per_layer_input;
  if (out.size() < per) return false;

  // inpL = token_embd[token] * sqrt(d).
  std::vector<float> inpL(d);
  if (!embed_token(m, token_id, inpL)) return false;

  // proj = per_layer_model_proj[:, layer*per .. +per) @ inpL, bf16 [d, n_layer*per]
  // (d contiguous). Column (layer*per + j) lives at offset (layer*per + j)*d.
  //
  // llama.cpp packs the f32 activation to bf16 (the weight's vec_dot_type) before
  // the dot product, so each input element is rounded through bf16 here — a
  // full-f32 input would not reproduce the oracle's rounding (Phase 5D finding).
  // The dot accumulates in double (ggml's ggml_float), with each bf16 product
  // rounded to f32 first, matching scalar ggml_vec_dot_bf16; the projection
  // scale is applied to the f32-cast dot, exactly as ggml_scale does.
  //
  // The bf16 rounding of inpL is hoisted out of the column loop (it was recomputed
  // per*col times), and the `per` columns are dotted in parallel via the process
  // thread pool. Each column writes only its own proj[j] (double accumulation is
  // per-column), so the result is deterministic and identical to the serial path.
  const std::span<const uint16_t> M = m.weights.per_layer_model_proj;
  const float proj_scale = 1.0f / std::sqrt(float(d));
  std::vector<float> inpL_bf16(d);
  for (uint32_t i = 0; i < d; ++i)
    inpL_bf16[i] = bf16_to_f32(f32_to_bf16(inpL[i]));

  std::vector<float> proj(per);
  // per_layer_model_proj[:, layer*per .. +per) @ inpL_bf16. The layer's `per`
  // columns are contiguous in the packed bf16 tensor (offset layer*per*d), so it
  // is one span. The projection scale is applied on the host after the matvec
  // (the CPU reference accumulates in double, the CUDA bf16 gemv in fp32).
  const uint64_t layer_base = (uint64_t(layer) * per) * d;
  if (!matvec_bf16_d(use_cuda, M.subspan(layer_base, uint64_t(per) * d), d,
                     inpL_bf16, proj))
    return false;
  for (uint32_t j = 0; j < per; ++j) proj[j] *= proj_scale;

  // proj = rms_norm(proj, eps) * per_layer_proj_norm.
  if (!nn::rms_norm(proj, m.weights.per_layer_proj_norm, m.config.rms_epsilon,
                    proj))
    return false;

  // ple = per_layer_token_embd[token][layer*per .. +per) * sqrt(per).
  const uint64_t total_per = uint64_t(m.config.block_count) * per;
  std::vector<float> row(total_per);
  if (!quant::dequantize_row_f32(m.weights.per_layer_token_embd, token_id,
                                 row.data(), row.size()))
    return false;
  const float ple_scale = std::sqrt(float(per));
  const uint64_t col0 = uint64_t(layer) * per;

  // out = (proj + ple) * (1/sqrt(2)).
  const float combine_scale = 1.0f / std::sqrt(2.0f);
  if (!nn::layer_combine(proj, std::span<const float>(row).subspan(col0, per),
                         ple_scale, combine_scale, out))
    return false;
  return true;
}

// One block given its residual-stream input `inp` (length d). The trace callback
// emits llama.cpp's named intermediates ("<name>-<layer>"). `donor_v` is empty
// for a KV layer (0..n_layer_kv-1, which owns a K/V projection) and non-empty
// (length n_kv*head_dim) for a shared layer, whose single-token attention output
// is the GQA-broadcast donor V. For a KV layer, if `kv_out` is non-empty
// (>= n_kv*head_dim) it receives the fp16-normalized V — exactly the value
// llama.cpp caches — so the forward loop can hand it to the shared layers.
static bool run_block_core(const Gemma4Model& m, uint32_t layer,
                           std::span<const float> inp, uint64_t token_id,
                           uint64_t pos, std::span<const float> donor_v,
                           std::span<float> kv_out, std::span<float> out,
                           const BlockTraceFn& trace,
                           quant::MatmulBackend backend) {
  ScopeTimer _t(&g_prof.t_block);
  ++g_prof.n_block;
  if (m.weights.blocks.empty() || layer >= m.config.block_count) return false;
  const uint32_t d = m.config.embedding_length;
  if (inp.size() < d || out.size() < d) return false;

  const LayerConfig& lc = m.plan[layer];
  const BlockWeights& bw = m.weights.blocks[layer];
  const uint32_t n_q = lc.n_heads_q;
  const uint32_t n_kv = lc.n_heads_kv;
  const uint32_t hd = lc.head_dim;
  const uint32_t ff = m.config.feed_forward_length;
  const uint32_t per = m.config.per_layer_input;
  const float eps = m.config.rms_epsilon;
  const bool is_kv = m.config.has_kv(layer);
  // Route the d/ff/qw/kvw elementwise + dense gate matvecs to CUDA when the Cuda
  // backend is active and a device is present; small per-head loops (hd=256/512,
  // per=256) and SDPA stay on the CPU.
  const bool use_cuda = backend == quant::MatmulBackend::Cuda && quant::cuda_available();

  // Named-stage tracer matching llama.cpp's cb() convention ("<name>-<layer>").
  auto emit = [&](const char* base, std::span<const float> v) {
    if (!trace) return;
    char name[64];
    std::snprintf(name, sizeof name, "%s-%u", base, layer);
    trace(name, v);
  };

  // --- attention -----------------------------------------------------------
  const double _t_attn = prof_now_ms();
  std::vector<float> h(d);
  if (!rms_norm_d(use_cuda, inp, bw.attn_norm, eps, h)) return false;
  emit("attn_norm", h);

  const uint64_t qw = uint64_t(n_q) * hd;
  const uint64_t kvw = uint64_t(n_kv) * hd;
  std::vector<float> q(qw);
  if (!quant::matvec(bw.attn_q, h, q, backend)) return false;
  emit("Qcur", q);

  // Per-head RMSNorm: q with the (scalar-broadcast) weight. Done for both KV and
  // shared layers (Q is always computed; only the K/V side differs).
  if (!nn::rms_norm_heads(q, bw.attn_q_norm, hd, eps)) return false;
  emit("Qcur_normed", q);

  // NEOX RoPE, per head, at absolute position `pos`. Proportional-RoPE
  // freq_factors (rope_freqs) apply only to the global (full-attention) layers.
  const std::span<const float> freq_factors =
      lc.attention == AttentionKind::Global ? m.weights.rope_freqs
                                            : std::span<const float>{};
  // Rotate the whole Q span in one pass (single kernel over qw when on CUDA).
  if (!rope_heads_d(use_cuda, q, hd, pos, lc.rope_base, freq_factors)) return false;
  emit("Qcur_pos", q);

  // Gemma 4 uses scale = 1.0 (Q/K RMSNorm supplies the normalization) and no
  // logit softcapping (softcap = 0). At n_k == 1 the softmax is over a single
  // logit, so the attention output is simply the value vector (GQA-broadcast).
  std::vector<float> attn(qw);
  if (is_kv) {
    std::vector<float> k(kvw), v(kvw);
    if (!quant::matvec(bw.attn_k, h, k, backend)) return false;
    emit("Kcur", k);
    if (!quant::matvec(bw.attn_v, h, v, backend)) return false;
    emit("Vcur", v);
    // Per-head RMSNorm: k with the (scalar-broadcast) weight, v without.
    if (!nn::rms_norm_heads(k, bw.attn_k_norm, hd, eps)) return false;
    emit("Kcur_normed", k);
    if (!nn::rms_norm_heads(v, {}, hd, eps)) return false;
    emit("Vcur_normed", v);
    nn::cast_fp16(v);  // match llama.cpp flash-attn's fp16 value cast
    // Rotate the whole K span in one pass (single kernel over kvw when on CUDA).
    if (!rope_heads_d(use_cuda, k, hd, pos, lc.rope_base, freq_factors)) return false;
    emit("Kcur_pos", k);
    if (!nn::scaled_dot_product_attention(q, 1, k, v, 1, n_q, n_kv, hd, true,
                                          lc.sliding_window, 0.0f, 1.0f, attn))
      return false;
    if (!kv_out.empty()) {
      if (kv_out.size() < kvw) return false;
      std::copy(v.begin(), v.end(), kv_out.begin());  // fp16 V == the KV-cache value
    }
  } else {
    // Shared layer: no K/V projection; attention reads the donor's cached KV. At
    // n_k == 1 the softmax over the single logit is 1, so the output is the
    // GQA-broadcast donor V (K is irrelevant). kv_out is left untouched.
    if (!nn::gqa_broadcast(donor_v, attn, n_q, n_kv, hd)) return false;
  }
  emit("kqv_out", attn);

  std::vector<float> attn_proj(d);
  if (!quant::matvec(bw.attn_output, attn, attn_proj, backend)) return false;
  std::vector<float> o(d);
  if (!rms_norm_d(use_cuda, attn_proj, bw.post_attention_norm, eps, o)) return false;
  emit("attn_post_norm", o);
  std::vector<float> x1(d);
  if (!add_d(use_cuda, o, inp, x1)) return false;
  emit("attn_out", x1);

  // --- gated feed-forward (GELU-tanh, parallel gate) -----------------------
  g_prof.t_attn += prof_now_ms() - _t_attn;
  const double _t_ffn = prof_now_ms();
  std::vector<float> f(d);
  if (!rms_norm_d(use_cuda, x1, bw.ffn_norm, eps, f)) return false;
  emit("ffn_norm", f);
  std::vector<float> up(ff), gate(ff), ffn_hidden(ff);
  if (!quant::matvec(bw.ffn_up, f, up, backend)) return false;
  emit("ffn_up", up);
  if (!quant::matvec(bw.ffn_gate, f, gate, backend)) return false;
  emit("ffn_gate", gate);
  if (!gelu_fp16_d(use_cuda, gate, gate)) return false;  // GGML_GELU_FP16 table path
  if (!mul_d(use_cuda, gate, up, ffn_hidden)) return false;
  emit("ffn_geglu", ffn_hidden);
  std::vector<float> ffn_out(d);
  if (!quant::matvec(bw.ffn_down, ffn_hidden, ffn_out, backend)) return false;
  emit("ffn_out", ffn_out);
  std::vector<float> f2(d);
  if (!rms_norm_d(use_cuda, ffn_out, bw.post_ffw_norm, eps, f2)) return false;
  emit("ffn_post_norm", f2);
  std::vector<float> x2(d);
  if (!add_d(use_cuda, f2, x1, x2)) return false;
  emit("pe_in", x2);

  // --- per-layer gate ------------------------------------------------------
  g_prof.t_ffn += prof_now_ms() - _t_ffn;
  const double _t_gate = prof_now_ms();
  std::vector<float> inp_ple(per);
  if (!embed_per_layer(m, layer, token_id, inp_ple, use_cuda)) return false;
  std::vector<float> g0(per);
  if (!matvec_f32_d(use_cuda, bw.inp_gate, d, x2, g0)) return false;  // [d] -> [per]
  if (!gelu_fp16_d(use_cuda, g0, g0)) return false;  // GGML_GELU_FP16 table path
  if (!nn::mul(g0, inp_ple, g0)) return false;  // * inp_per_layer[l]  (per=256: CPU)
  std::vector<float> g1(d);
  if (!matvec_f32_d(use_cuda, bw.proj, per, g0, g1)) return false;  // [per] -> [d]
  std::vector<float> g2(d);
  if (!rms_norm_d(use_cuda, g1, bw.post_norm, eps, g2)) return false;
  emit("per_layer_embd_out", g2);
  std::vector<float> x3(d);
  if (!add_d(use_cuda, x2, g2, x3)) return false;

  // Whole-block output scale (blk.%d.layer_output_scale.weight). Gemma 4 applies
  // a per-layer scalar here (e.g. 0.061/0.16/0.44 across layers); llama.cpp
  // multiplies it in and then cb()'s "out_scaled" and "l_out" on the *same*
  // tensor (build_cvec is identity for gemma4), so the scale survives only as
  // "l_out". Apply the scale and emit only "l_out" to match the oracle dump.
  if (!bw.layer_output_scale.empty()) {
    if (!scale_d(use_cuda, x3, bw.layer_output_scale[0], out)) return false;
  } else {
    std::copy(x3.begin(), x3.end(), out.begin());
  }
  emit("l_out", out);
  g_prof.t_gate += prof_now_ms() - _t_gate;
  return true;
}

bool run_block(const Gemma4Model& m, uint32_t layer, uint64_t token_id,
               uint64_t pos, std::span<float> out, quant::MatmulBackend backend) {
  return run_block_traced(m, layer, token_id, pos, out, {}, backend);
}

bool run_block_traced(const Gemma4Model& m, uint32_t layer, uint64_t token_id,
                      uint64_t pos, std::span<float> out,
                      const BlockTraceFn& trace, quant::MatmulBackend backend) {
  // A single block in isolation: the residual input is the token embedding
  // (correct only for layer 0; use forward/forward_traced for chaining).
  if (m.weights.blocks.empty() || layer >= m.config.block_count) return false;
  const uint32_t d = m.config.embedding_length;
  if (out.size() < d) return false;
  std::vector<float> inpL(d);
  if (!embed_token(m, token_id, inpL)) return false;
  return run_block_core(m, layer, inpL, token_id, pos, {}, {}, out, trace, backend);
}

bool forward(const Gemma4Model& m, uint64_t token_id, uint64_t pos,
             std::span<float> out, quant::MatmulBackend backend) {
  return forward_traced(m, token_id, pos, out, {}, backend);
}

bool forward_traced(const Gemma4Model& m, uint64_t token_id, uint64_t pos,
                    std::span<float> out, const BlockTraceFn& trace,
                    quant::MatmulBackend backend) {
  if (m.weights.blocks.empty()) return false;
  const uint32_t d = m.config.embedding_length;
  if (out.size() < d) return false;

#ifdef SONICBOOM_USE_CUDA
  // Device-resident fast path (Phase 6a Option A): without a trace and with a
  // present device, run the whole forward with activations resident on the GPU
  // (synchronizes only at the end). The host path below remains the reference
  // for the CPU backend and for trace-based oracle dumps.
  if (!trace && backend == quant::MatmulBackend::Cuda && quant::cuda_available())
    return cuda_forward_resident(m, token_id, pos, out);
#endif

  // Residual stream: token embedding in, then each block's l_out becomes the
  // next block's input.
  std::vector<float> inpL(d);
  if (!embed_token(m, token_id, inpL)) return false;

  // Gemma 4 shared KV: the last two KV layers (n_kv-2 = SWA, n_kv-1 = global)
  // are the KV donors for the 18 shared layers (24..41). Retain their fp16 V as
  // we pass through so the shared layers can read it.
  const uint32_t n_kv = m.config.n_layer_kv();
  const uint32_t donor_swa_layer = n_kv - 2;     // 22
  const uint32_t donor_global_layer = n_kv - 1;  // 23
  std::vector<float> donor_swa, donor_global;
  std::vector<float> next(d);
  for (uint32_t l = 0; l < m.config.block_count; ++l) {
    const uint32_t kvw = uint64_t(m.plan[l].n_heads_kv) * m.plan[l].head_dim;
    if (m.config.has_kv(l)) {
      std::vector<float> kv(kvw);
      if (!run_block_core(m, l, inpL, token_id, pos, {}, kv, next, trace, backend))
        return false;
      if (l == donor_swa_layer)
        donor_swa = std::move(kv);
      else if (l == donor_global_layer)
        donor_global = std::move(kv);
    } else {
      const std::vector<float>& donor =
          m.config.sliding_window_pattern[l] ? donor_swa : donor_global;
      if (donor.empty()) return false;  // donor layer must have run first
      if (!run_block_core(m, l, inpL, token_id, pos, donor, {}, next, trace, backend))
        return false;
    }
    std::swap(inpL, next);
  }

  std::copy(inpL.begin(), inpL.end(), out.begin());

  // Phase 5E lm-head (trace only): emit the post-loop intermediates so the dump
  // validates the whole head against the oracle. h_nextn == result_norm (the
  // post-output-norm hidden; llama.cpp cb()s the same tensor under both names for
  // single-token decode). result_output is the softcapped logits.
  if (trace) {
    std::vector<float> h(d);
    if (!nn::rms_norm(inpL, m.weights.output_norm, m.config.rms_epsilon, h))
      return false;
    trace("h_nextn", h);
    trace("result_norm", h);
    std::vector<float> logits(m.config.vocab_size);
    if (!quant::matvec(m.weights.token_embd, h, logits, backend)) return false;
    const float sc = m.config.final_logit_softcapping;
    const float inv_sc = 1.0f / sc;  // ggml_scale(cur, 1/sc) multiplies by this
    for (float& l : logits) l = sc * std::tanh(l * inv_sc);
    trace("result_output", logits);
  }

  return true;
}

bool lm_head(const Gemma4Model& m, std::span<const float> hidden,
             std::span<float> logits, quant::MatmulBackend backend) {
  if (m.weights.blocks.empty()) return false;
  const uint32_t d = m.config.embedding_length;
  if (hidden.size() < d || logits.size() < m.config.vocab_size) return false;

  // h = rms_norm(hidden, output_norm, eps) — the "result_norm" / "h_nextn" state.
  const bool use_cuda = backend == quant::MatmulBackend::Cuda && quant::cuda_available();
  std::vector<float> h(d);
  if (!rms_norm_d(use_cuda, hidden, m.weights.output_norm, m.config.rms_epsilon, h))
    return false;

  // logit = token_embd @ h (weight-tied output, q3_K; q8_K-fused dot), then the
  // final-logit softcapping: 30 * tanh(logit / 30).
  if (!quant::matvec(m.weights.token_embd, h, logits, backend)) return false;
  const float sc = m.config.final_logit_softcapping;
  const float inv_sc = 1.0f / sc;  // ggml_scale(cur, 1/sc) multiplies by this
  for (size_t i = 0; i < m.config.vocab_size; ++i)
    logits[i] = sc * std::tanh(logits[i] * inv_sc);
  return true;
}

uint64_t generate_reference(const Gemma4Model& m, uint64_t start_token,
                            uint64_t start_pos, std::span<int64_t> tokens,
                            quant::MatmulBackend backend) {
  if (m.weights.blocks.empty() || start_token >= m.config.vocab_size) return 0;
  const uint32_t d = m.config.embedding_length;

  // Scratch reused across steps (no per-token realloc): the final hidden state
  // and the lm-head logits.
  std::vector<float> hidden(d);
  std::vector<float> logits(m.config.vocab_size);

  uint64_t token = start_token;
  uint64_t pos = start_pos;
  uint64_t step = 0;
  for (; step < tokens.size(); ++step) {
    if (!forward(m, token, pos, hidden, backend)) break;
    if (!lm_head(m, hidden, logits, backend)) break;
    const int64_t next = nn::argmax(logits);
    if (next < 0) break;
    tokens[step] = next;
    token = static_cast<uint64_t>(next);
    ++pos;
  }
  return step;
}

uint64_t generate(const Gemma4Model& m, uint64_t start_token, uint64_t start_pos,
                  std::span<int64_t> tokens, quant::MatmulBackend backend) {
  // Phase 6 (M4): the runtime decode path is the spine. The bespoke forward +
  // lm_head loop (generate_reference) is oracle-only; it stays reachable for the
  // f32 scalar reference, which has no spine variant.
  switch (backend) {
    case quant::MatmulBackend::Cuda:
      if (quant::cuda_available())
        return generate_spine_cuda(m, start_token, start_pos, tokens);
      [[fallthrough]];  // no device: run the CPU spine
    case quant::MatmulBackend::CpuQ8K:
      return generate_spine(m, start_token, start_pos, tokens);
    case quant::MatmulBackend::CpuF32:
      return generate_reference(m, start_token, start_pos, tokens, backend);
  }
  return 0;
}

} // namespace sonicboom::model
