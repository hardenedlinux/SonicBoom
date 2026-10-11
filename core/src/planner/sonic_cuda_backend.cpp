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

#include <sonicboom/planner/sonic_cuda_backend.h>

#include <sonicboom/model/exec.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/cuda_elementwise.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include "../nn/cuda_graph.h"
#include "../nn/cuda_resident.h"
#include "../nn/cuda_stream.h"
#include "../quant/cuda_resident.h"
#include "device_allocator.h"
#include "device_pool.h"

#include <nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sonicboom::planner {

namespace {

RuntimeError fail(std::string msg) {
  return RuntimeError(RuntimeErrorCode::BackendFailure, std::move(msg),
                      Phase::Execution);
}

// NVTX markers are opt-in. They stay OFF for normal runs (ctest) and turn ON
// only when SONICBOOM_NVTX is set — the nsys measurement runs that need the
// per-step/per-layer ranges. Emitting them perturbs host-side timing, which
// would bias the tok/s measurement; it no longer affects numerics (the split-K
// gemv is a deterministic two-phase reduction, no float atomicAdd).
bool nvtx_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("SONICBOOM_NVTX");
    return e && *e && std::string_view(e) != "0";
  }();
  return on;
}

bool attr_i(const GraphNodeDesc& n, std::string_view name, int64_t& out) {
  for (const auto& a : n.attributes)
    if (a.name == name && a.kind() == sx::AttrValueKind::Int) {
      out = std::get<int64_t>(a.value);
      return true;
    }
  return false;
}

bool attr_f(const GraphNodeDesc& n, std::string_view name, double& out) {
  for (const auto& a : n.attributes)
    if (a.name == name && a.kind() == sx::AttrValueKind::Float) {
      out = std::get<double>(a.value);
      return true;
    }
  return false;
}

const float* dev(const TensorValue& v) {
  return reinterpret_cast<const float*>(v.device.handle);
}

float* dev_mut(uint64_t handle) {
  return reinterpret_cast<float*>(handle);
}

// The device allocator backing this backend's persistent buffers. Stateless: it
// delegates to the process-wide device pool, so a single shared instance serves
// all backends (mirrors the pool's global free-list).
hal::DeviceAllocator g_alloc;

// HAL StepProgram backed by an instantiated CUDA graph. Owns the GraphStep (and
// therefore the cudaGraphExec_t / stream) and maps replay()/sync()/valid() onto
// it. This is the only place the CUDA graph type appears behind the seam; it is
// a backend-private detail and never crosses into Layer 2 or the planner.
class CudaStepProgram final : public hal::StepProgram {
 public:
  bool replay() override { return graph_.replay(); }
  bool sync() override { return graph_.sync(); }
  bool valid() const override { return graph_.valid(); }

  sonicboom::cuda::GraphStep& graph() { return graph_; }

 private:
  sonicboom::cuda::GraphStep graph_;
};

} // namespace

SonicCudaBackend::SonicCudaBackend(const model::Gemma4Model& model,
                                   const Graph& graph, uint64_t n_ctx)
    : model_(model), graph_(graph), n_ctx_(n_ctx) {
  ensure_caches(n_ctx);
  token_buf_ = g_alloc.allocate(sizeof(float));   // persistent current-token slot
  pos_buf_ = g_alloc.allocate(sizeof(uint64_t));  // persistent position slot
  if (pos_buf_) {
    const uint64_t zero = 0;
    g_alloc.copy_to_device(pos_buf_, &zero, sizeof(uint64_t));  // defensive: pos starts at 0
  }
}

SonicCudaBackend::~SonicCudaBackend() {
  for (const auto& c : caches_) {
    if (c.k_buf) g_alloc.deallocate(c.k_buf);
    if (c.v_buf) g_alloc.deallocate(c.v_buf);
  }
  if (token_buf_) g_alloc.deallocate(token_buf_);
  if (pos_buf_) g_alloc.deallocate(pos_buf_);
}

bool SonicCudaBackend::seed_token(uint64_t token) {
  if (!token_buf_) return false;
  const float f = float(token);
  return g_alloc.copy_to_device(token_buf_, &f, sizeof(float));
}

bool SonicCudaBackend::set_pos(uint64_t pos) {
  if (!pos_buf_) return false;
  return g_alloc.copy_to_device(pos_buf_, &pos, sizeof(uint64_t));
}

bool SonicCudaBackend::read_token(float* host) const {
  if (!token_buf_ || !host) return false;
  return g_alloc.copy_from_device(host, token_buf_, sizeof(float));
}

bool SonicCudaBackend::synchronize() const {
  return g_alloc.synchronize();
}

std::unique_ptr<hal::StepProgram> SonicCudaBackend::record_step(
    std::function<bool()> step_fn) {
  auto prog = std::make_unique<CudaStepProgram>();
  auto& gs = prog->graph();
  if (!gs.begin()) return nullptr;
  bool captured = false;
  {
    sonicboom::cuda::CaptureScope cap;
    sonicboom::cuda::StreamScope ss(gs.stream());
    captured = step_fn();
  }
  if (!captured || !gs.end()) return nullptr;
  return prog;
}

void SonicCudaBackend::ensure_caches(uint64_t n_ctx) {
  const uint32_t n_kv = model_.config.n_layer_kv();
  caches_.clear();
  caches_.resize(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l) {
    const model::LayerConfig& lc = model_.config.layer_config(l);
    KvCacheDev& c = caches_[l];
    c.n_heads_kv = lc.n_heads_kv;
    c.head_dim = lc.head_dim;
    // SWA layers are a fixed-size ring; global layers the full decode length.
    c.n_slots = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx;
    c.bytes = c.n_slots * c.n_heads_kv * c.head_dim * sizeof(float);
    c.k_buf = g_alloc.allocate(c.bytes);
    c.v_buf = g_alloc.allocate(c.bytes);
  }
}

std::expected<std::vector<TensorValue>, RuntimeError> SonicCudaBackend::execute(
    const TaskDesc& task, const std::vector<TensorValue>& in) {
  const auto* compute = task.as_compute();
  if (!compute || compute->graph_nodes.empty())
    return std::unexpected(fail("Sonic CUDA task has no compute node"));
  const auto* node = graph_.find_node(compute->graph_nodes.front());
  if (!node)
    return std::unexpected(fail("Sonic CUDA task references an unknown graph node"));
  if (node->outputs.empty())
    return std::unexpected(fail("Sonic CUDA node has no output"));
  const auto* out_td = graph_.find_tensor(node->outputs.front());
  if (!out_td)
    return std::unexpected(fail("Sonic CUDA node output tensor is unknown"));

  const auto& cfg = model_.config;
  const uint32_t d = cfg.embedding_length;
  const uint32_t per = cfg.per_layer_input;
  const float eps = cfg.rms_epsilon;

  int64_t layer = -1;
  attr_i(*node, "layer", layer);
  int64_t slot_i = 0;
  attr_i(*node, "slot", slot_i);
  const model::WeightSlot slot = static_cast<model::WeightSlot>(slot_i);
  double scale_d = 0.0;
  attr_f(*node, "scale", scale_d);

  const model::BlockWeights& block =
      model_.weights.blocks[layer >= 0 ? uint32_t(layer) : 0];
  const model::LayerConfig& lc = cfg.layer_config(layer >= 0 ? uint32_t(layer) : 0);

  auto fweight = [&](model::WeightSlot s) -> std::span<const float> {
    switch (s) {
      case model::WeightSlot::AttnNorm: return block.attn_norm;
      case model::WeightSlot::FfnNorm: return block.ffn_norm;
      case model::WeightSlot::AttnQNorm: return block.attn_q_norm;
      case model::WeightSlot::AttnKNorm: return block.attn_k_norm;
      case model::WeightSlot::PostAttentionNorm: return block.post_attention_norm;
      case model::WeightSlot::PostFfwNorm: return block.post_ffw_norm;
      case model::WeightSlot::PostNorm: return block.post_norm;
      case model::WeightSlot::InpGate: return block.inp_gate;
      case model::WeightSlot::Proj: return block.proj;
      case model::WeightSlot::PerLayerProjNorm: return model_.weights.per_layer_proj_norm;
      case model::WeightSlot::OutputNorm: return model_.weights.output_norm;
      case model::WeightSlot::RopeFreqs: return model_.weights.rope_freqs;
      default: return {};
    }
  };

  auto qweight = [&](model::WeightSlot s) -> const quant::QuantizedTensor& {
    static const quant::QuantizedTensor kEmpty;
    switch (s) {
      case model::WeightSlot::AttnQ: return block.attn_q;
      case model::WeightSlot::AttnK: return block.attn_k;
      case model::WeightSlot::AttnV: return block.attn_v;
      case model::WeightSlot::AttnOutput: return block.attn_output;
      case model::WeightSlot::FfnGate: return block.ffn_gate;
      case model::WeightSlot::FfnUp: return block.ffn_up;
      case model::WeightSlot::FfnDown: return block.ffn_down;
      case model::WeightSlot::TokenEmbd: return model_.weights.token_embd;
      default: return kEmpty;
    }
  };

  // Allocate the device output buffer from the pool. Device activations never
  // reach this backend on the host, so every input is device-resident except the
  // Int64 `pos` scalar consumed by Rope.
  const uint64_t out_bytes = out_td->size_bytes;
  const uint64_t out_handle = device::acquire(out_bytes);
  if (!out_handle)
    return std::unexpected(fail("device buffer allocation failed"));
  float* out = dev_mut(out_handle);
  const int out_n = int(out_bytes / sizeof(float));

  bool ok = true;
  // NVTX range named after the graph node (its op kind, e.g. "rms_norm",
  // "quantized_matmul"), so nsys attributes each kernel launch to its operator
  // for the per-op launch-count breakdown.
  if (nvtx_enabled()) nvtxRangePushA(node->name.c_str());
  switch (node->op) {
    case OpKind::Scale: {
      ok = nn::cuda::scale_dev(dev(in[0]), float(scale_d), out, out_n);
      break;
    }
    case OpKind::RmsNorm: {
      ok = nn::cuda::rms_norm_dev(fweight(slot), dev(in[0]), eps, out, out_n);
      break;
    }
    case OpKind::RmsNormHeads: {
      std::span<const float> w;
      if (slot == model::WeightSlot::AttnQNorm)
        w = block.attn_q_norm;
      else if (slot == model::WeightSlot::AttnKNorm)
        w = block.attn_k_norm;
      ok = nn::cuda::rms_norm_heads_dev(w, dev(in[0]), eps, out, int(lc.head_dim),
                                        out_n / int(lc.head_dim));
      break;
    }
    case OpKind::Rope: {
      const std::span<const float> ff =
          lc.attention == model::AttentionKind::Global
              ? model_.weights.rope_freqs
              : std::span<const float>{};
      if (!device::device_copy(in[0].device.handle, out_handle, out_bytes)) {
        ok = false;
        break;
      }
      ok = nn::cuda::rope_neox_heads_dev_pos(
          out, int(lc.head_dim), out_n / int(lc.head_dim),
          reinterpret_cast<const uint64_t*>(pos_buf_.id), lc.rope_base, 1.0f, ff);
      break;
    }
    case OpKind::CastFp16: {
      if (!device::device_copy(in[0].device.handle, out_handle, out_bytes)) {
        ok = false;
        break;
      }
      ok = nn::cuda::cast_fp16_dev(out, out_n);
      break;
    }
    case OpKind::Attention: {
      // KV layer: append the fp16-rounded K/V to this layer's device cache at
      // slot `pos % n_slots` (computed in-kernel from pos_buf_), then run
      // incremental decode attention over it. inputs: {q (RoPE'd f32, device),
      // kc (device), vc (device)}; the position comes from the device slot.
      KvCacheDev& cache = caches_[uint32_t(layer)];
      const int row_elems = int(cache.n_heads_kv * cache.head_dim);
      const uint64_t window = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx_;
      if (!cache.k_buf || !cache.v_buf ||
          !nn::cuda::kv_append_dev(dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
                                   dev(in[1]), dev(in[2]),
                                   reinterpret_cast<const uint64_t*>(pos_buf_.id),
                                   cache.n_slots, row_elems)) {
        ok = false;
        break;
      }
      ok = nn::cuda::decode_attention_dev(
          dev(in[0]), dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
          reinterpret_cast<const uint64_t*>(pos_buf_.id), window, cache.n_slots,
          int(lc.n_heads_q), int(cache.n_heads_kv), int(cache.head_dim),
          lc.sliding_window, 1.0f, out);
      break;
    }
    case OpKind::AttentionShared: {
      // Shared layer (24..41): attend over the donor KV layer's cache (no
      // append). inputs: {q (device)}; the position comes from the device slot.
      const uint32_t donor = cfg.kv_donor_layer(uint32_t(layer));
      const KvCacheDev& cache = caches_[donor];
      const uint64_t window = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx_;
      ok = nn::cuda::decode_attention_dev(
          dev(in[0]), dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
          reinterpret_cast<const uint64_t*>(pos_buf_.id), window, cache.n_slots,
          int(lc.n_heads_q), int(cache.n_heads_kv), int(cache.head_dim),
          lc.sliding_window, 1.0f, out);
      break;
    }
    case OpKind::GqaBroadcast: {
      ok = nn::cuda::gqa_broadcast_dev(dev(in[0]), out, int(lc.n_heads_q),
                                       int(lc.n_heads_kv), int(lc.head_dim));
      break;
    }
    case OpKind::QuantizedMatmul: {
      ok = quant::cuda::matvec_f32_dev(qweight(slot), dev(in[0]), out);
      break;
    }
    case OpKind::Quantize: {
      // f32 activation -> q8_1 (quantize-once, T1). The input is a plain f32
      // tensor; its element count is the input byte size / sizeof(float). The
      // output buffer is f32-sized (a superset of the q8_1 bytes), so the q8_1
      // blocks are packed into its front.
      const int n = int(in[0].device.size_bytes / sizeof(float));
      ok = quant::cuda::quantize_q8_1_dev(
          dev(in[0]), reinterpret_cast<quant::cuda::BlockQ8_1*>(out), n / 32);
      break;
    }
    case OpKind::QuantizedMatmulQ8: {
      // K-quant gemv against a pre-quantized q8_1 activation (T1): no internal
      // quantize; blocks_per_row / num_rows come from the weight.
      ok = quant::cuda::gemv_q8_1_dev(
          qweight(slot), reinterpret_cast<const quant::cuda::BlockQ8_1*>(dev(in[0])),
          out);
      break;
    }
    case OpKind::RmsNormQuantize: {
      // rms_norm + q8_1 quantize fused (T3): the norm weight is keyed by `slot`
      // (AttnNorm / FfnNorm / OutputNorm); the output is the q8_1 activation
      // packed into an f32-sized buffer (same superset convention as Quantize).
      ok = quant::cuda::rms_norm_quantize_q8_1_dev(
          fweight(slot), dev(in[0]), eps,
          reinterpret_cast<quant::cuda::BlockQ8_1*>(out), out_n);
      break;
    }
    case OpKind::GeluMulQuantize: {
      // gelu_fp16(gate) * up, q8_1-quantized in one kernel (T5). Inputs are the
      // gate/up projection outputs (device f32); the output is the q8_1
      // activation for the down-projection gemv.
      ok = quant::cuda::gelu_mul_quantize_q8_1_dev(
          dev(in[0]), dev(in[1]),
          reinterpret_cast<quant::cuda::BlockQ8_1*>(out), out_n);
      break;
    }
    case OpKind::MatvecF32: {
      const int cols = int(in[0].device.size_bytes / sizeof(float));
      ok = nn::cuda::matvec_f32_dev(fweight(slot), cols, dev(in[0]), out, out_n);
      break;
    }
    case OpKind::MatvecBf16: {
      const uint64_t layer_base = (uint64_t(layer) * per) * d;
      const std::span<const uint16_t> W =
          model_.weights.per_layer_model_proj.subspan(layer_base, uint64_t(per) * d);
      const int cols = int(in[0].device.size_bytes / sizeof(float));
      // The CPU path rounds the activation through bf16 before the dot; do the
      // same on a scratch copy so the shared `inp` tensor is left untouched.
      const uint64_t xbytes = in[0].device.size_bytes;
      uint64_t xtmp = device::acquire(xbytes);
      if (!xtmp) {
        ok = false;
        break;
      }
      if (!device::device_copy(in[0].device.handle, xtmp, xbytes) ||
          !nn::cuda::cast_bf16_dev(dev_mut(xtmp), cols)) {
        device::release(xtmp, xbytes);
        ok = false;
        break;
      }
      ok = nn::cuda::matvec_bf16_dev(W, cols, reinterpret_cast<const float*>(xtmp),
                                     out, out_n);
      device::release(xtmp, xbytes);
      break;
    }
    case OpKind::GeluFp16: {
      ok = nn::cuda::gelu_fp16_dev(dev(in[0]), out, out_n);
      break;
    }
    case OpKind::Mul: {
      ok = nn::cuda::mul_dev(dev(in[0]), dev(in[1]), out, out_n);
      break;
    }
    case OpKind::Add: {
      ok = nn::cuda::add_dev(dev(in[0]), dev(in[1]), out, out_n);
      break;
    }
    case OpKind::LayerCombine: {
      const uint64_t col = uint64_t(layer) * per;
      const float ple_scale = std::sqrt(float(per));
      const float combine_scale = 1.0f / std::sqrt(2.0f);
      ok = nn::cuda::layer_combine_dev(dev(in[0]), dev(in[1]) + col, ple_scale,
                                       combine_scale, out, int(per));
      break;
    }
    case OpKind::PerLayerInput: {
      // inputs: {inp [d] (device), ple_row [total_per] (device)}; output
      // [total_per]. One full-width bf16 gemv + scale + segmented rms_norm +
      // combine per token (Phase 2.3a). In-place steps are safe: each kernel
      // reads element i before writing it, and the rms reduction is a separate
      // launch before the apply.
      const int total_per = out_n;
      const int cols = int(in[0].device.size_bytes / sizeof(float));  // d
      const uint64_t xbytes = in[0].device.size_bytes;
      uint64_t xtmp = device::acquire(xbytes);
      if (!xtmp) {
        ok = false;
        break;
      }
      if (!device::device_copy(in[0].device.handle, xtmp, xbytes) ||
          !nn::cuda::cast_bf16_dev(dev_mut(xtmp), cols)) {
        device::release(xtmp, xbytes);
        ok = false;
        break;
      }
      ok = nn::cuda::matvec_bf16_dev(model_.weights.per_layer_model_proj, cols,
                                     reinterpret_cast<const float*>(xtmp), out,
                                     total_per);
      device::release(xtmp, xbytes);
      if (!ok) break;
      ok = nn::cuda::scale_dev(out, 1.0f / std::sqrt(float(d)), out, total_per);
      if (!ok) break;
      ok = nn::cuda::rms_norm_heads_dev(model_.weights.per_layer_proj_norm, out,
                                        eps, out, int(per), total_per / int(per));
      if (!ok) break;
      ok = nn::cuda::layer_combine_dev(out, dev(in[1]), std::sqrt(float(per)),
                                       1.0f / std::sqrt(2.0f), out, total_per);
      break;
    }
    case OpKind::Slice: {
      int64_t offset = 0, length = 0;
      attr_i(*node, "offset", offset);
      attr_i(*node, "length", length);
      const uint64_t src = in[0].device.handle + uint64_t(offset) * sizeof(float);
      ok = device::device_copy(src, out_handle, uint64_t(length) * sizeof(float));
      break;
    }
    case OpKind::Embedding: {
      // Token embedding gather on the device: dequantize the token's row of
      // token_embd (Q3_K) / per_layer_token_embd (Q4_K) straight to device f32.
      // The token id is read from the persistent device buffer `token_buf_`
      // (Phase 3.2 B4), not the host graph input, so the next token id never
      // round-trips through the host.
      const float* tok_dev = dev_mut(token_buf_.id);
      if (slot == model::WeightSlot::PerLayerTokenEmbd) {
        // ~1.5 GiB tensor that does not fit resident: pin+map it in place and
        // gather the token's row zero-copy (no per-token H2D).
        ok = quant::cuda::gather_f32_mapped_dev_row(
            model_.weights.per_layer_token_embd, tok_dev, out);
      } else {
        // token_embd is the tied lm_head, already resident (the logits matvec
        // uploads it once): gather straight from the device weight, no H2D.
        ok = quant::cuda::gather_f32_resident_dev_row(model_.weights.token_embd,
                                                      tok_dev, out);
      }
      break;
    }
    case OpKind::Argmax: {
      // Greedy token selection on the device: reduce the [vocab] logits to a
      // single index, so the per-token 1 MiB D2H of the full logits becomes a
      // 4-byte transfer of the winning index. The index is also written to the
      // persistent `token_buf_` slot (Phase 3.2 B4) so the next step's Embedding
      // gather reads it straight from the device.
      ok = nn::cuda::argmax_dev(dev(in[0]),
                                int(in[0].device.size_bytes / sizeof(float)), out);
      if (ok && token_buf_)
        ok = device::device_copy(out_handle, token_buf_.id, sizeof(float));
      break;
    }
    default:
      // Softcap is placed on the CPU SonicBackend (no *_dev kernel); hitting it
      // here means the placement pass routed a host-only op to the device.
      ok = false;
      break;
  }
  if (nvtx_enabled()) nvtxRangePop();

  if (!ok) {
    device::release(out_handle, out_bytes);
    return std::unexpected(fail("Sonic CUDA kernel failed for node '" + node->name +
                                "'"));
  }

  std::vector<TensorValue> res;
  res.push_back(TensorValue::from_device(out_handle, out_bytes));
  return res;
}

bool SonicCudaBackend::prefill(std::span<const uint64_t> tokens, uint64_t start_pos,
                               std::span<float> out_hidden) {
  const auto& cfg = model_.config;
  if (model_.weights.blocks.empty()) return false;
  if (!quant::cuda::available() || !nn::cuda::available()) return false;
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;
  const uint32_t per = cfg.per_layer_input;
  const float eps = cfg.rms_epsilon;
  const uint32_t n = uint32_t(tokens.size());
  if (n == 0 || out_hidden.size() < d) return false;
  if (caches_.size() != cfg.n_layer_kv()) return false;

  // The prompt's positions must land in a straight, wrap-free run of each KV
  // layer's cache (SWA ring n_slots == sliding_window, global n_slots == n_ctx).
  for (uint32_t l = 0; l < cfg.n_layer_kv(); ++l) {
    const KvCacheDev& c = caches_[l];
    if (c.n_slots == 0 || (start_pos % c.n_slots) + n > c.n_slots) return false;
  }

  // Activations are [dim, n] column-major (n innermost) to match the batched
  // f32 matmul layout; Q/K/V are transposed to [n, dim] position-major for the
  // per-position head norm / RoPE / flash attention, then back. Mirrors the CPU
  // SonicBackend::prefill; only the heavy kernels (batched matmul + flash
  // attention) run on the device.
  auto transpose = [&](const float* src, float* dst, uint32_t dim) {
    for (uint32_t c = 0; c < n; ++c)
      for (uint32_t k = 0; k < dim; ++k) dst[c * dim + k] = src[k * n + c];
  };
  auto transpose_back = [&](const float* src, float* dst, uint32_t dim) {
    for (uint32_t c = 0; c < n; ++c)
      for (uint32_t k = 0; k < dim; ++k) dst[k * n + c] = src[c * dim + k];
  };
  auto rms_cols = [&](const float* x, std::span<const float> w, uint32_t dim,
                      float* y) {
    std::vector<float> col(dim);
    for (uint32_t c = 0; c < n; ++c) {
      for (uint32_t k = 0; k < dim; ++k) col[k] = x[k * n + c];
      if (!nn::rms_norm(col, w, eps, col)) return false;
      for (uint32_t k = 0; k < dim; ++k) y[k * n + c] = col[k];
    }
    return true;
  };

  // Batched token embedding: res[k*n+c] = token_embd[tokens[c]][k] * sqrt(d).
  std::vector<float> res(uint64_t(d) * n);
  {
    std::vector<float> col(d);
    for (uint32_t c = 0; c < n; ++c) {
      if (!model::embed_token(model_, tokens[c], col)) return false;
      for (uint32_t k = 0; k < d; ++k) res[k * n + c] = col[k];
    }
  }

  const uint32_t n_kv = cfg.n_layer_kv();
  const uint32_t donor_swa_layer = n_kv - 2;
  const uint32_t donor_global_layer = n_kv - 1;
  std::vector<float> donor_swa_k, donor_swa_v, donor_global_k, donor_global_v;

  std::vector<float> next(uint64_t(d) * n);

  for (uint32_t l = 0; l < cfg.block_count; ++l) {
    const model::LayerConfig& lc = model_.plan[l];
    const model::BlockWeights& bw = model_.weights.blocks[l];
    const uint32_t n_q = lc.n_heads_q;
    const uint32_t n_kvh = lc.n_heads_kv;
    const uint32_t hd = lc.head_dim;
    const uint32_t qw = n_q * hd;
    const uint32_t kvw = n_kvh * hd;
    const bool is_kv = cfg.has_kv(l);
    const std::span<const float> freq =
        lc.attention == model::AttentionKind::Global ? model_.weights.rope_freqs
                                                     : std::span<const float>{};

    // --- attention -----------------------------------------------------------
    std::vector<float> h(uint64_t(d) * n);
    if (!rms_cols(res.data(), bw.attn_norm, d, h.data())) return false;

    std::vector<float> q_col(uint64_t(qw) * n);
    if (!quant::cuda::matmul_f32(bw.attn_q, h, q_col, n)) return false;
    std::vector<float> q(uint64_t(qw) * n);
    transpose(q_col.data(), q.data(), qw);
    for (uint32_t c = 0; c < n; ++c) {
      std::span<float> qc(q.data() + uint64_t(c) * qw, qw);
      if (!nn::rms_norm_heads(qc, bw.attn_q_norm, hd, eps)) return false;
      if (!nn::rope_neox_heads(qc, hd, start_pos + c, lc.rope_base, 1.0f, freq))
        return false;
    }

    std::vector<float> attn(uint64_t(qw) * n);
    if (is_kv) {
      std::vector<float> k_col(uint64_t(kvw) * n), v_col(uint64_t(kvw) * n);
      if (!quant::cuda::matmul_f32(bw.attn_k, h, k_col, n)) return false;
      if (!quant::cuda::matmul_f32(bw.attn_v, h, v_col, n)) return false;

      std::vector<float> k(uint64_t(kvw) * n), v(uint64_t(kvw) * n);
      transpose(k_col.data(), k.data(), kvw);
      transpose(v_col.data(), v.data(), kvw);

      KvCacheDev& cache = caches_[l];
      const size_t row = (size_t)cache.n_heads_kv * cache.head_dim;  // == kvw
      for (uint32_t c = 0; c < n; ++c) {
        std::span<float> kc(k.data() + uint64_t(c) * kvw, kvw);
        std::span<float> vc(v.data() + uint64_t(c) * kvw, kvw);
        if (!nn::rms_norm_heads(kc, bw.attn_k_norm, hd, eps)) return false;
        if (!nn::rms_norm_heads(vc, {}, hd, eps)) return false;
        nn::cast_fp16(vc);  // V is fp16-rounded without RoPE
        if (!nn::rope_neox_heads(kc, hd, start_pos + c, lc.rope_base, 1.0f, freq))
          return false;
        nn::cast_fp16(kc);  // K is fp16-rounded after RoPE (spine cache repr)
        const size_t off = (size_t)((start_pos + c) % cache.n_slots) * row;
        if (!cache.k_buf || !cache.v_buf ||
            !device::upload(kc.data(), cache.k_buf.id + off * sizeof(float),
                            kvw * sizeof(float)) ||
            !device::upload(vc.data(), cache.v_buf.id + off * sizeof(float),
                            kvw * sizeof(float)))
          return false;
      }

      if (!nn::cuda::flash_attention(q, n, k, v, n, n_q, n_kvh, hd,
                                     lc.sliding_window, 1.0f, attn))
        return false;

      if (l == donor_swa_layer) {
        donor_swa_k = std::move(k);
        donor_swa_v = std::move(v);
      } else if (l == donor_global_layer) {
        donor_global_k = std::move(k);
        donor_global_v = std::move(v);
      }
    } else {
      const uint32_t donor = cfg.kv_donor_layer(l);
      const std::vector<float>& dk =
          cfg.sliding_window_pattern[l] ? donor_swa_k : donor_global_k;
      const std::vector<float>& dv =
          cfg.sliding_window_pattern[l] ? donor_swa_v : donor_global_v;
      if (dk.empty() || dv.empty()) return false;
      if (!nn::cuda::flash_attention(q, n, dk, dv, n, n_q, n_kvh, hd,
                                     lc.sliding_window, 1.0f, attn))
        return false;
    }

    std::vector<float> attn_col(uint64_t(qw) * n);
    transpose_back(attn.data(), attn_col.data(), qw);
    std::vector<float> ap(uint64_t(d) * n);
    if (!quant::cuda::matmul_f32(bw.attn_output, attn_col, ap, n)) return false;

    std::vector<float> o(uint64_t(d) * n);
    if (!rms_cols(ap.data(), bw.post_attention_norm, d, o.data())) return false;
    std::vector<float> x1(uint64_t(d) * n);
    if (!nn::add(o, res, x1)) return false;

    // --- gated feed-forward --------------------------------------------------
    std::vector<float> f(uint64_t(d) * n);
    if (!rms_cols(x1.data(), bw.ffn_norm, d, f.data())) return false;
    std::vector<float> up(uint64_t(ff) * n), gate(uint64_t(ff) * n);
    if (!quant::cuda::matmul_f32(bw.ffn_up, f, up, n)) return false;
    if (!quant::cuda::matmul_f32(bw.ffn_gate, f, gate, n)) return false;
    if (!nn::gelu_fp16(gate, gate)) return false;
    if (!nn::mul(gate, up, gate)) return false;
    std::vector<float> ffn_out(uint64_t(d) * n);
    if (!quant::cuda::matmul_f32(bw.ffn_down, gate, ffn_out, n)) return false;
    std::vector<float> f2(uint64_t(d) * n);
    if (!rms_cols(ffn_out.data(), bw.post_ffw_norm, d, f2.data())) return false;
    std::vector<float> x2(uint64_t(d) * n);
    if (!nn::add(f2, x1, x2)) return false;

    // --- per-layer gate (per position; matches the spine's embed_per_layer) --
    std::vector<float> x2_col(d), g0(per), g1(d), g2(d), x3_col(d), ple(per);
    for (uint32_t c = 0; c < n; ++c) {
      for (uint32_t k = 0; k < d; ++k) x2_col[k] = x2[k * n + c];
      if (!model::embed_per_layer(model_, l, tokens[c], ple, false)) return false;
      if (!nn::matvec_f32(bw.inp_gate, d, x2_col, g0)) return false;
      if (!nn::gelu_fp16(g0, g0)) return false;
      if (!nn::mul(g0, ple, g0)) return false;
      if (!nn::matvec_f32(bw.proj, per, g0, g1)) return false;
      if (!nn::rms_norm(g1, bw.post_norm, eps, g2)) return false;
      if (!nn::add(x2_col, g2, x3_col)) return false;
      if (!bw.layer_output_scale.empty() &&
          !nn::scale(x3_col, bw.layer_output_scale[0], x3_col))
        return false;
      for (uint32_t k = 0; k < d; ++k) next[k * n + c] = x3_col[k];
    }

    std::swap(res, next);
  }

  // The final residual column (position n-1) is the post-prefill hidden state.
  for (uint32_t k = 0; k < d; ++k) out_hidden[k] = res[k * n + (n - 1)];
  return true;
}

} // namespace sonicboom::planner
