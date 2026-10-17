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

#include <sonicboom/compiler/GlueCodegen.h>
#include <sonicboom/model/exec.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/cuda_elementwise.h>
#include <sonicboom/planner/plan_record.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include "../nn/cuda_graph.h"
#include "../nn/cuda_resident.h"
#include "../nn/cuda_stream.h"
#include "../quant/cuda_resident.h"
#include "device_allocator.h"
#include "device_pool.h"
#include "glue_jit.h"

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

// The string a node's attribute record carries for `name` (e.g. the region name
// of a `fused_region` node), or false.
bool attr_s(const GraphNodeDesc& n, std::string_view name, std::string_view& out) {
  for (const auto& a : n.attributes)
    if (a.name == name && a.kind() == sx::AttrValueKind::String) {
      out = std::get<std::string>(a.value);
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

  // Async H2D on the graph's stream, ordered before the next replay(). Unlike
  // the device pool's synchronous cudaMemcpy (which lands on the legacy default
  // stream and therefore blocks for every other stream's in-flight work), this
  // never synchronizes the host with the device: the host enqueues staging +
  // replay back-to-back and the GPU serializes them on one stream.
  bool stage_upload(const hal::Buffer& dst, const void* host,
                    uint64_t bytes) override {
    if (dst.id == 0 || (!host && bytes != 0)) return false;
    return cudaMemcpyAsync(reinterpret_cast<void*>(dst.id), host, bytes,
                           cudaMemcpyHostToDevice, graph_.stream()) == cudaSuccess;
  }

  sonicboom::cuda::GraphStep& graph() { return graph_; }

 private:
  sonicboom::cuda::GraphStep graph_;
};

} // namespace

SonicCudaBackend::SonicCudaBackend(const KvCacheSpec& spec, const Graph& graph,
                                   uint64_t n_ctx)
    : spec_(spec), graph_(graph), n_ctx_(n_ctx) {
  ensure_caches(n_ctx);
  token_buf_ = g_alloc.allocate(sizeof(float));   // persistent current-token slot
  pos_buf_ = g_alloc.allocate(sizeof(uint64_t));  // persistent position slot
  // Prefill chunk token-id slot: sized for the largest chunk (the SWA window),
  // so any segment (<= sliding_window tokens) stages into it in place.
  const uint64_t chunk = spec_.chunk;
  if (chunk > 0)
    tokens_buf_ = g_alloc.allocate(chunk * sizeof(uint64_t));
  if (pos_buf_) {
    const uint64_t zero = 0;
    g_alloc.copy_to_device(pos_buf_, &zero, sizeof(uint64_t));  // defensive: pos starts at 0
  }
}

SonicCudaBackend::~SonicCudaBackend() {
  // The caches_ shared_ptr holds a deleter that frees each layer's device
  // buffers; the last owner (decode or prefill backend) frees them exactly once.
  caches_.reset();
  if (token_buf_) g_alloc.deallocate(token_buf_);
  if (pos_buf_) g_alloc.deallocate(pos_buf_);
  if (tokens_buf_) g_alloc.deallocate(tokens_buf_);
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

bool SonicCudaBackend::set_prefill_tokens(std::span<const uint64_t> tokens) {
  if (!tokens_buf_ || tokens.empty()) return false;
  const uint64_t chunk = spec_.chunk;
  if (tokens.size() > chunk) return false;  // a segment never exceeds the window
  return g_alloc.copy_to_device(tokens_buf_, tokens.data(),
                                tokens.size() * sizeof(uint64_t));
}

bool SonicCudaBackend::stage_prefill_tokens(hal::StepProgram& prog,
                                            std::span<const uint64_t> tokens) {
  if (!tokens_buf_ || tokens.empty()) return false;
  const uint64_t chunk = spec_.chunk;
  if (tokens.size() > chunk) return false;  // a segment never exceeds the window
  return prog.stage_upload(tokens_buf_, tokens.data(),
                           tokens.size() * sizeof(uint64_t));
}

bool SonicCudaBackend::stage_pos(hal::StepProgram& prog, uint64_t pos) {
  if (!pos_buf_) return false;
  return prog.stage_upload(pos_buf_, &pos, sizeof(uint64_t));
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
  const uint32_t n_kv = spec_.n_layers;
  auto* raw = new std::vector<KvCacheDev>();
  raw->resize(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l) {
    const KvCacheSpec::Layer& sl = spec_.layers[l];
    KvCacheDev& c = (*raw)[l];
    c.n_heads_kv = sl.kv_heads;
    c.head_dim = sl.head_dim;
    // SWA layers are a fixed-size ring; global layers the full decode length.
    c.n_slots = (sl.window > 0) ? sl.window : n_ctx;
    c.bytes = c.n_slots * c.n_heads_kv * c.head_dim * sizeof(float);
    c.k_buf = g_alloc.allocate(c.bytes);
    c.v_buf = g_alloc.allocate(c.bytes);
  }
  // The deleter frees each layer's device buffers, so whichever backend is the
  // last owner (after share_caches) deallocates exactly once.
  caches_ = std::shared_ptr<std::vector<KvCacheDev>>(
      raw, [](std::vector<KvCacheDev>* v) {
        if (!v) return;
        for (auto& c : *v) {
          if (c.k_buf) g_alloc.deallocate(c.k_buf);
          if (c.v_buf) g_alloc.deallocate(c.v_buf);
        }
        delete v;
      });
}

void SonicCudaBackend::share_caches(const SonicCudaBackend& other) {
  caches_ = other.caches_;
}

void SonicCudaBackend::share_caches(const KvCacheSession& session) {
  if (session.caches)
    caches_ = session.caches;  // adopt the session's cache (prefix reuse)
}

void SonicCudaBackend::publish_cache(KvCacheSession& session) const {
  session.caches = caches_;
  session.n_ctx = n_ctx_;
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

  if (!model_)
    return std::unexpected(fail("Sonic CUDA backend has no bound model"));

  const auto& cfg = model_->config;
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
      model_->weights.blocks[layer >= 0 ? uint32_t(layer) : 0];
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
      case model::WeightSlot::AttnQBias: return block.attn_q_bias;
      case model::WeightSlot::AttnKBias: return block.attn_k_bias;
      case model::WeightSlot::AttnVBias: return block.attn_v_bias;
      case model::WeightSlot::AttnOutputBias: return block.attn_output_bias;
      case model::WeightSlot::InpGate: return block.inp_gate;
      case model::WeightSlot::Proj: return block.proj;
      case model::WeightSlot::PerLayerProjNorm: return model_->weights.per_layer_proj_norm;
      case model::WeightSlot::OutputNorm: return model_->weights.output_norm;
      case model::WeightSlot::RopeFreqs: return model_->weights.rope_freqs;
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
      case model::WeightSlot::TokenEmbd: return model_->weights.token_embd;
      case model::WeightSlot::Output: return model_->weights.output;
      default: return kEmpty;
    }
  };

  // Plan-record annotations (plan_record.h). OpKind names the operation but not
  // the numeric format it ran in: the weight's quant type and the activation's
  // layout are chosen from the node's attributes at dispatch time, so they are
  // recorded here where both are resolved. Both are no-ops with recording off —
  // a thread-local load, and no format-string work at all.
  auto note_weight_quant = [&](const quant::QuantizedTensor& w) {
    if (auto* r = recorder())
      if (const auto* qi = quant::quant_type_info(w.type()))
        r->set_weight_quant(qi->name);
  };
  auto note_activation_quant = [&](const char* layout) {
    if (auto* r = recorder()) r->set_activation_quant(layout);
  };
  // Name the mmq activation's scale layout from the node's `ds` attribute
  // (mmq::DsLayout: D4 = a float d4 for Q3_K weights, DS4 = a half2 ds4 for
  // Q4_K/Q5_K, D4_Q8_1 = the q8_1-compatible pair).
  auto mmq_activation = [](int64_t ds) -> const char* {
    switch (ds) {
      case 1: return "block_q8_1_mmq(ds4)";
      case 2: return "block_q8_1_mmq(d4_q8_1)";
      default: return "block_q8_1_mmq(d4)";
    }
  };

  // The number of positions this execution covers: the node's `batch` (the
  // extent the plan was built at) overridden by the extent bound for this
  // execution. A plan lowered from a symbolic IR carries the planned maximum as
  // its `batch`, so overriding it is what lets one plan serve every segment of
  // its shape class.
  auto batch_attr = [&](int64_t& nb) {
    if (!attr_i(*node, "batch", nb))
      return false;
    if (segment_extent_ != 0)
      nb = int64_t(segment_extent_);
    return true;
  };

  // Allocate the device output buffer from the pool. Device activations never
  // reach this backend on the host, so every input is device-resident except the
  // Int64 `pos` scalar consumed by Rope. A tensor that scales with the plan's
  // dynamic extent occupies its per-unit stride times this execution's extent.
  if (out_td->dynamic_stride_bytes != 0 && segment_extent_ == 0)
    return std::unexpected(fail("dynamic tensor executed without an extent"));
  const uint64_t out_bytes =
      out_td->dynamic_stride_bytes != 0
          ? out_td->dynamic_stride_bytes * segment_extent_ +
                out_td->dynamic_base_bytes
          : out_td->size_bytes;
  const uint64_t out_handle = device::acquire(out_bytes);
  if (!out_handle)
    return std::unexpected(fail("device buffer allocation failed"));
  float* out = dev_mut(out_handle);
  const int out_n = int(out_bytes / sizeof(float));

  bool ok = true;
  // A fused region may keep a producer's output as a second result (the prefill
  // add+norm region writes the residual sum as well as the activation). The
  // first output is `out` above; this is the one the region allocates for the
  // second, and it is returned after `out`.
  uint64_t extra_handle = 0;
  uint64_t extra_bytes = 0;
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
      int64_t nb = 0, dim_i = 0;
      if (batch_attr(nb)) {
        // Row norm over [n, dim]: reuse the per-head norm with a single
        // `dim_i`-wide head per row (position), so each row is normalized
        // independently with the full-width weight.
        attr_i(*node, "dim", dim_i);
        ok = nn::cuda::rms_norm_heads_dev(fweight(slot), dev(in[0]), eps, out,
                                          int(dim_i), int(nb));
      } else {
        ok = nn::cuda::rms_norm_dev(fweight(slot), dev(in[0]), eps, out, out_n);
      }
      break;
    }
    case OpKind::RmsNormHeads: {
      // Per-head (or per-segment) rms_norm. The segment width is the node's
      // `dim` attribute (defaults to the layer's head_dim); the weight is the
      // node's `slot` (empty => unit), so this also covers the model-wide
      // per-layer projection norm (slot == PerLayerProjNorm, dim == per).
      const std::span<const float> w = fweight(slot);
      int64_t dim_i = lc.head_dim;
      attr_i(*node, "dim", dim_i);
      ok = nn::cuda::rms_norm_heads_dev(w, dev(in[0]), eps, out, int(dim_i),
                                        out_n / int(dim_i));
      break;
    }
    case OpKind::Rope: {
      const std::span<const float> ff =
          lc.attention == model::AttentionKind::Global
              ? model_->weights.rope_freqs
              : std::span<const float>{};
      if (!device::device_copy(in[0].device.handle, out_handle, out_bytes)) {
        ok = false;
        break;
      }
      int64_t nb = 0;
      if (batch_attr(nb)) {
        ok = nn::cuda::rope_neox_heads_batched_dev_pos(
            out, int(lc.head_dim), out_n / int(nb) / int(lc.head_dim), int(nb),
            reinterpret_cast<const uint64_t*>(pos_buf_.id), lc.rope_base, 1.0f, ff);
      } else {
        ok = nn::cuda::rope_neox_heads_dev_pos(
            out, int(lc.head_dim), out_n / int(lc.head_dim),
            reinterpret_cast<const uint64_t*>(pos_buf_.id), lc.rope_base, 1.0f, ff);
      }
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
      // Attends over the KV-cache layer named by the node's `kv_source_layer`
      // attribute (identity for a KV layer, the donor for a shared-KV layer).
      // A KV layer appends its fp16-rounded K/V to the cache at slot
      // `pos % n_slots` (computed in-kernel from pos_buf_) before attending
      // (inputs {q, kc, vc}); a shared layer omits K/V and does not append
      // (inputs {q}). The position comes from the device slot.
      int64_t ksrc = layer;
      attr_i(*node, "kv_source_layer", ksrc);
      const uint32_t cache_layer = ksrc >= 0 ? uint32_t(ksrc) : uint32_t(layer);
      KvCacheDev& cache = (*caches_)[cache_layer];
      const int row_elems = int(cache.n_heads_kv * cache.head_dim);
      const uint64_t window = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx_;
      if (in.size() >= 4) {
        if (!cache.k_buf || !cache.v_buf ||
            !nn::cuda::kv_append_dev(dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
                                     dev(in[1]), dev(in[2]),
                                     reinterpret_cast<const uint64_t*>(pos_buf_.id),
                                     cache.n_slots, row_elems)) {
          ok = false;
          break;
        }
      }
      double attn_scale = 1.0;
      attr_f(*node, "scale", attn_scale);
      ok = nn::cuda::decode_attention_dev(
          dev(in[0]), dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
          reinterpret_cast<const uint64_t*>(pos_buf_.id), window, cache.n_slots,
          int(lc.n_heads_q), int(cache.n_heads_kv), int(cache.head_dim),
          lc.sliding_window, float(attn_scale), out);
      break;
    }
    case OpKind::GqaBroadcast: {
      ok = nn::cuda::gqa_broadcast_dev(dev(in[0]), out, int(lc.n_heads_q),
                                       int(lc.n_heads_kv), int(lc.head_dim));
      break;
    }
    case OpKind::QuantizedMatmul: {
      int64_t nb = 0;
      note_weight_quant(qweight(slot));
      if (batch_attr(nb)) {
        ok = quant::cuda::matmul_f32_dev(qweight(slot), dev(in[0]), out, int(nb));
      } else {
        ok = quant::cuda::matvec_f32_dev(qweight(slot), dev(in[0]), out);
      }
      break;
    }
    case OpKind::Quantize: {
      // f32 activation -> q8_1 (quantize-once, T1). The input is a plain f32
      // tensor; its element count is the input byte size / sizeof(float). The
      // output buffer is f32-sized (a superset of the q8_1 bytes), so the q8_1
      // blocks are packed into its front.
      const int n = int(in[0].device.size_bytes / sizeof(float));
      // The 32-element q8_1 layout (no `batch`): the decode side's activation
      // quantizer. The 128-element mmq layout appears under RmsNormQuantize &
      // co. below.
      note_activation_quant("q8_1");
      ok = quant::cuda::quantize_q8_1_dev(
          dev(in[0]), reinterpret_cast<quant::cuda::BlockQ8_1*>(out), n / 32);
      break;
    }
    case OpKind::QuantizedMatmulQ8: {
      // K-quant matmul against a pre-quantized activation: the decode path
      // (no `batch`) consumes a q8_1 (32-elem block) activation via gemv; the
      // prefill path (`batch` present) consumes a block_q8_1_mmq (128-elem
      // block) activation via the int8-tensor-core matmul. No internal
      // quantize; blocks_per_row / num_rows come from the weight.
      int64_t nb = 0;
      note_weight_quant(qweight(slot));
      if (batch_attr(nb)) {
        int64_t ds_i = 0;
        attr_i(*node, "ds", ds_i);
        note_activation_quant(mmq_activation(ds_i));
        ok = quant::cuda::matmul_mmq_q8_dev(
            qweight(slot),
            reinterpret_cast<const quant::cuda::BlockQ8_1Mmq*>(dev(in[0])), out,
            int(nb));
      } else {
        note_activation_quant("q8_1");
        const auto* xq = reinterpret_cast<const quant::cuda::BlockQ8_1*>(dev(in[0]));
        const auto& qw = qweight(slot);
        const bool epilogue = quant::cuda::gemv_bias_epilogue_supported(qw.type());
        // `bias_slot` is set by fuse_bias_into_gemv_pass: the projection's bias
        // (a bound [rows] model weight) was folded into this matmul node by
        // dropping the AddBias node that followed it. `add_input` is set by
        // fuse_residual_add_into_gemv_pass: the residual of `Add(proj, residual)`
        // became this matmul's second activation input when the Add node was
        // dropped. The gemv epilogue adds both to the row's reduced sum (bias
        // first, then the residual) — bit-identical to the separate kernels it
        // replaces, since both add to the stored f32 output in that order. A type
        // without the epilogue runs the gemv and then the plain elementwise
        // kernels over the output (same values, more launches).
        int64_t bias_slot = 0, add_idx = 0;
        const bool has_bias = attr_i(*node, "bias_slot", bias_slot);
        const bool has_add = attr_i(*node, "add_input", add_idx);
        // The epilogue adds one bias element per output row; a bias that is
        // shorter than the row (the AddBias broadcast form) goes through the
        // separate kernel, which reproduces add_bias's i % bias_n indexing.
        const std::span<const float> b =
            has_bias ? fweight(static_cast<model::WeightSlot>(bias_slot))
                     : std::span<const float>{};
        const bool bias_fused = has_bias && b.size() == size_t(out_n) && epilogue;
        const float* add_p = has_add ? static_cast<const float*>(dev(in[add_idx]))
                                     : nullptr;
        const bool add_fused = add_p != nullptr && epilogue;
        if (bias_fused || add_fused) {
          ok = quant::cuda::gemv_q8_1_dev(
              qw, xq, out, bias_fused ? b : std::span<const float>{},
              add_fused ? add_p : nullptr);
          if (ok && has_bias && !bias_fused && !b.empty())
            ok = nn::cuda::add_bias_dev(b, out, out, out_n);
          if (ok && has_add && !add_fused)
            ok = nn::cuda::add_dev(add_p, out, out, out_n);
        } else {
          ok = quant::cuda::gemv_q8_1_dev(qw, xq, out);
          if (ok && has_bias && !b.empty())
            ok = nn::cuda::add_bias_dev(b, out, out, out_n);
          if (ok && has_add) ok = nn::cuda::add_dev(add_p, out, out, out_n);
        }
      }
      break;
    }
    case OpKind::RmsNormQuantize: {
      // rms_norm + quantize fused (T3): the norm weight is keyed by `slot`
      // (AttnNorm / FfnNorm / OutputNorm); the output buffer holds only the
      // quantized activation. The decode path (no `batch`) emits a q8_1
      // (32-elem block) activation — its element count is the f32 input's, not
      // the (smaller) output buffer's; the prefill path (`batch` + `dim` +
      // `ds`) emits a block_q8_1_mmq (128-elem block, D4/DS4 scales) activation.
      int64_t nb = 0, dim_i = 0, ds_i = 0;
      if (batch_attr(nb)) {
        attr_i(*node, "dim", dim_i);
        attr_i(*node, "ds", ds_i);
        note_activation_quant(mmq_activation(ds_i));
        ok = quant::cuda::rms_norm_quantize_mmq_q8_1_dev(
            fweight(slot), dev(in[0]), eps,
            reinterpret_cast<quant::cuda::BlockQ8_1Mmq*>(out), int(dim_i), int(nb),
            int(ds_i));
      } else {
        note_activation_quant("q8_1");
        const int n = int(in[0].device.size_bytes / sizeof(float));
        ok = quant::cuda::rms_norm_quantize_q8_1_dev(
            fweight(slot), dev(in[0]), eps,
            reinterpret_cast<quant::cuda::BlockQ8_1*>(out), n);
      }
      break;
    }
    case OpKind::FusedRegion: {
      // A glue region the compiler lowered to a device kernel (Stage 5,
      // FuseGlueRegionsPass). The region's name is the node's `fused_region`
      // string attribute; the node's operands/attributes are the ones the op it
      // replaced carried, so the same dispatch information is read here.
      //
      // rms_norm_quantize_q8_1 is the decode (rms_norm -> q8_1) region: one
      // row, one 1024-thread block, `slot` naming the norm weight. The generated
      // kernel is byte-identical to quant::cuda::rms_norm_quantize_q8_1_dev, so
      // that call is also the fallback if the kernel cannot be compiled or
      // loaded (a missing/corrupt cache and no MLIR in this process).
      std::string_view region;
      if (!attr_s(*node, "fused_region", region)) {
        ok = false;
        break;
      }
      if (region == "rms_norm_quantize_q8_1") {
        note_activation_quant("q8_1");
        const int n = int(in[0].device.size_bytes / sizeof(float));
        const std::span<const float> w = fweight(slot);
        const void* wd =
            w.empty()
                ? nullptr
                : quant::cuda::upload_bytes_cached(w.data(),
                                                   w.size() * sizeof(float));
        if (!w.empty() && wd == nullptr) {
          ok = false;
          break;
        }
        // The generated kernel's ABI: (x, w, y, n, eps), one block per row.
        void* xarg = const_cast<float*>(dev(in[0]));
        void* warg = const_cast<void*>(wd);
        void* yarg = out;
        int narg = n;
        float earg = eps;
        void* args[] = {&xarg, &warg, &yarg, &narg, &earg};
        ok = glue::launch(region, args, 1, compiler::kGlueBlockThreads,
                          compiler::kGlueDynamicSharedBytes);
        if (!ok)
          ok = quant::cuda::rms_norm_quantize_q8_1_dev(
              w, dev(in[0]), eps, reinterpret_cast<quant::cuda::BlockQ8_1*>(out),
              n);
        break;
      }
      if (region == "add_rms_norm_quantize_mmq_q8_1") {
        // Prefill residual add + batched norm + mmq quantize (Stage 5b):
        // outputs are {mmq q8_1 activation, residual sum}. The sum is a real
        // graph tensor — the next layer's residual add reads it — so the fused
        // node keeps it as its second output and the kernel writes it.
        int64_t nb = 0, dim_i = 0, ds_i = 0;
        if (!batch_attr(nb) || !attr_i(*node, "dim", dim_i) ||
            !attr_i(*node, "ds", ds_i) || node->outputs.size() < 2) {
          ok = false;
          break;
        }
        note_activation_quant(mmq_activation(ds_i));
        const std::span<const float> w = fweight(slot);
        const void* wd =
            w.empty()
                ? nullptr
                : quant::cuda::upload_bytes_cached(w.data(),
                                                   w.size() * sizeof(float));
        if (!w.empty() && wd == nullptr) {
          ok = false;
          break;
        }
        const auto* res_td = graph_.find_tensor(node->outputs[1]);
        if (!res_td) {
          ok = false;
          break;
        }
        extra_bytes = res_td->dynamic_stride_bytes != 0
                          ? res_td->dynamic_stride_bytes * segment_extent_ +
                                res_td->dynamic_base_bytes
                          : res_td->size_bytes;
        extra_handle = device::acquire(extra_bytes);
        if (!extra_handle) {
          ok = false;
          break;
        }
        float* resp = dev_mut(extra_handle);
        // The generated kernel's ABI: (a, b, w, y, res, dim, n, ds, eps), one
        // block per token.
        void* aarg = const_cast<float*>(dev(in[0]));
        void* barg = const_cast<float*>(dev(in[1]));
        void* warg = const_cast<void*>(wd);
        void* yarg = out;
        void* rarg = resp;
        int dimarg = int(dim_i);
        int nbarg = int(nb);
        int dsarg = int(ds_i);
        float earg = eps;
        void* args[] = {&aarg,  &barg, &warg,  &yarg,     &rarg,
                        &dimarg, &nbarg, &dsarg, &earg};
        ok = glue::launch(region, args, int(nb),
                          compiler::kAddRmsNormBlockThreads,
                          compiler::kAddRmsNormSharedBytes);
        if (!ok) {
          ok = nn::cuda::add_dev(dev(in[0]), dev(in[1]), resp,
                                 int(nb * dim_i));
          if (ok)
            ok = quant::cuda::rms_norm_quantize_mmq_q8_1_dev(
                w, resp, eps,
                reinterpret_cast<quant::cuda::BlockQ8_1Mmq*>(out), int(dim_i),
                int(nb), int(ds_i));
        }
        break;
      }
      // An unknown region would mean a pass emitted a node this backend has no
      // kernel for; fail rather than silently produce an unwritten buffer.
      ok = false;
      break;
    }
    case OpKind::GeluMulQuantize: {
      // gelu_fp16(gate) * up, quantized in one kernel (T5). Inputs are the
      // gate/up projection outputs (device f32); the output is the quantized
      // activation for the down-projection matmul. Decode (no `batch`) emits
      // q8_1; prefill (`batch` + `dim` + `ds`) emits block_q8_1_mmq.
      int64_t nb = 0, dim_i = 0, ds_i = 0;
      if (batch_attr(nb)) {
        attr_i(*node, "dim", dim_i);
        attr_i(*node, "ds", ds_i);
        note_activation_quant(mmq_activation(ds_i));
        ok = quant::cuda::gelu_mul_quantize_mmq_q8_1_dev(
            dev(in[0]), dev(in[1]),
            reinterpret_cast<quant::cuda::BlockQ8_1Mmq*>(out), int(dim_i), int(nb),
            int(ds_i));
      } else {
        note_activation_quant("q8_1");
        const int n = int(in[0].device.size_bytes / sizeof(float));
        ok = quant::cuda::gelu_mul_quantize_q8_1_dev(
            dev(in[0]), dev(in[1]),
            reinterpret_cast<quant::cuda::BlockQ8_1*>(out), n);
      }
      break;
    }
    case OpKind::SiluMulQuantize: {
      // silu(gate) * up, quantized in one kernel (T5, SwiGLU). Exactly the
      // Silu counterpart of GeluMulQuantize above. Decode (no `batch`) emits a
      // q8_1 (32-elem block) activation; prefill (`batch` + `dim` + `ds`) emits
      // a block_q8_1_mmq (128-elem block, D4/DS4 scales) activation, the same
      // two flavours the gelu fusion has.
      int64_t nb = 0, dim_i = 0, ds_i = 0;
      if (batch_attr(nb)) {
        attr_i(*node, "dim", dim_i);
        attr_i(*node, "ds", ds_i);
        note_activation_quant(mmq_activation(ds_i));
        ok = quant::cuda::silu_mul_quantize_mmq_q8_1_dev(
            dev(in[0]), dev(in[1]),
            reinterpret_cast<quant::cuda::BlockQ8_1Mmq*>(out), int(dim_i), int(nb),
            int(ds_i));
      } else {
        note_activation_quant("q8_1");
        const int n = int(in[0].device.size_bytes / sizeof(float));
        ok = quant::cuda::silu_mul_quantize_q8_1_dev(
            dev(in[0]), dev(in[1]),
            reinterpret_cast<quant::cuda::BlockQ8_1*>(out), n);
      }
      break;
    }
    case OpKind::MatvecF32: {
      int64_t nb = 0;
      if (batch_attr(nb)) {
        const int n = int(nb);
        const int cols = int(in[0].device.size_bytes / sizeof(float) / uint64_t(n));
        const int rows = out_n / n;
        ok = nn::cuda::matvec_f32_batched_dev(fweight(slot), cols, dev(in[0]), out,
                                              rows, n);
      } else {
        const int cols = int(in[0].device.size_bytes / sizeof(float));
        ok = nn::cuda::matvec_f32_dev(fweight(slot), cols, dev(in[0]), out, out_n);
      }
      break;
    }
    case OpKind::MatvecBf16: {
      // Dense bf16 projection against a slice of per_layer_model_proj, named by
      // `offset` (start row) and `rows` (output width). The default (offset ==
      // layer*per, rows == per) is the per-layer projection; the hoisted
      // full-width projection uses offset == 0, rows == total_per.
      int64_t offset = (layer >= 0 ? int64_t(layer) : 0) * per;
      int64_t rows_i = per;
      attr_i(*node, "offset", offset);
      attr_i(*node, "rows", rows_i);
      const uint32_t rows = uint32_t(rows_i);
      const std::span<const uint16_t> W =
          model_->weights.per_layer_model_proj.subspan(uint64_t(offset) * d,
                                                       uint64_t(rows) * d);
      int64_t nb = 0;
      const bool batched = batch_attr(nb);
      const int n = int(nb);
      const int cols =
          int(in[0].device.size_bytes / sizeof(float) / uint64_t(batched ? n : 1));
      // The CPU path rounds the activation through bf16 before the dot; do the
      // same on a scratch copy so the shared `inp` tensor is left untouched.
      const uint64_t xbytes = in[0].device.size_bytes;
      uint64_t xtmp = device::acquire(xbytes);
      if (!xtmp) {
        ok = false;
        break;
      }
      if (!device::device_copy(in[0].device.handle, xtmp, xbytes) ||
          !nn::cuda::cast_bf16_dev(dev_mut(xtmp), cols * (batched ? n : 1))) {
        device::release(xtmp, xbytes);
        ok = false;
        break;
      }
      if (batched)
        ok = nn::cuda::matvec_bf16_batched_dev(
            W, cols, reinterpret_cast<const float*>(xtmp), out, int(rows), n);
      else
        ok = nn::cuda::matvec_bf16_dev(W, cols, reinterpret_cast<const float*>(xtmp),
                                       out, out_n);
      device::release(xtmp, xbytes);
      break;
    }
    case OpKind::GeluFp16: {
      ok = nn::cuda::gelu_fp16_dev(dev(in[0]), out, out_n);
      break;
    }
    case OpKind::Silu: {
      ok = nn::cuda::silu_dev(dev(in[0]), out, out_n);
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
    case OpKind::AddBias: {
      // A [width] bias broadcast over the rows of the projection output; `out_n`
      // is [width] when decoding and [n, width] when prefilling, so one launch
      // covers both. The bias is a bound model weight (host span, uploaded once).
      const std::span<const float> b = fweight(slot);
      ok = !b.empty() && nn::cuda::add_bias_dev(b, dev(in[0]), out, out_n);
      break;
    }
    case OpKind::Slice: {
      // Sub-span copy. `offset`/`length` name the contiguous span; when `stride`
      // is present it is a strided row slice over `rows` rows of `length`
      // elements each (rows == out_n / length), used to extract a layer's
      // per-layer-embedding column from a [n, total_per] row.
      int64_t offset = 0, length = 0, stride = 0;
      attr_i(*node, "offset", offset);
      attr_i(*node, "length", length);
      attr_i(*node, "stride", stride);
      if (stride > 0) {
        const int len = int(length);
        ok = (len > 0 && out_n % len == 0) &&
             nn::cuda::slice_strided_dev(dev(in[0]), out, out_n / len, len,
                                         int(offset), int(stride));
      } else {
        const uint64_t src = in[0].device.handle + uint64_t(offset) * sizeof(float);
        ok = device::device_copy(src, out_handle, uint64_t(length) * sizeof(float));
      }
      break;
    }
    case OpKind::Embedding: {
      int64_t nb = 0;
      // The gathered table's format, per the `slot` the node names. token_embd
      // (the tied lm_head) is a quantized table; the per-layer table is plain
      // f32 and so has no quant type to record.
      if (slot != model::WeightSlot::PerLayerTokenEmbd)
        note_weight_quant(model_->weights.token_embd);
      if (batch_attr(nb)) {
        // Batched: gather n token rows (prefill), ids read from the persistent
        // device slot tokens_buf_.
        const uint64_t* tokens_dev =
            reinterpret_cast<const uint64_t*>(tokens_buf_.id);
        if (!tokens_dev) { ok = false; break; }
        if (slot == model::WeightSlot::PerLayerTokenEmbd)
          ok = quant::cuda::gather_f32_mapped_dev_batched_ids(
              model_->weights.per_layer_token_embd, tokens_dev, int(nb), out);
        else
          ok = quant::cuda::gather_f32_resident_dev_batched_ids(
              model_->weights.token_embd, tokens_dev, int(nb), out);
      } else {
        // Token embedding gather on the device: dequantize the token's row of
        // token_embd (Q3_K) / per_layer_token_embd (Q4_K) straight to device
        // f32. The token id is read from the persistent device buffer
        // `token_buf_` (Phase 3.2 B4), not the host graph input, so the next
        // token id never round-trips through the host.
        const float* tok_dev = dev_mut(token_buf_.id);
        if (slot == model::WeightSlot::PerLayerTokenEmbd) {
          // ~1.5 GiB tensor that does not fit resident: pin+map it in place and
          // gather the token's row zero-copy (no per-token H2D).
          ok = quant::cuda::gather_f32_mapped_dev_row(
              model_->weights.per_layer_token_embd, tok_dev, out);
        } else {
          // token_embd is the tied lm_head, already resident (the logits matvec
          // uploads it once): gather straight from the device weight, no H2D.
          ok = quant::cuda::gather_f32_resident_dev_row(model_->weights.token_embd,
                                                        tok_dev, out);
        }
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
    // --- inherently-batched prefill ops (Phase 7 / Section C.3) ---------------
    // FlashAttention/KvAppend have no per-token decode twin;
    // they always carry the `batch` attribute and run the chunked *_dev kernels.
    // Activations are [n, dim] position-major (dim innermost). The Int64 graph
    // inputs (tokens, start_pos) are read from the persistent device slots
    // tokens_buf_ / pos_buf_ (staged by the spine before execute/replay), so a
    // whole segment is capturable into a CUDA Graph (no host value baked into a
    // launch argument and no synchronous H2D upload).
    case OpKind::FlashAttention: {
      // Chunked attention over prior cache + current batch (no append — the
      // append is the deferred KvAppend node). The cache read is the layer named
      // by the node's `kv_source_layer` attribute (identity for a KV layer, the
      // donor for a shared-KV layer); a shared layer passes its donor's K/V
      // graph tensors as inputs. inputs: {q, kc/dk, vc/dv, start_pos}. For a
      // single chunk at start_pos == 0 the prior range is empty and this is
      // bit-identical to the old intra-batch flash attention.
      int64_t nb = 0;
      batch_attr(nb);
      int64_t ksrc = layer;
      attr_i(*node, "kv_source_layer", ksrc);
      const uint32_t cache_layer = ksrc >= 0 ? uint32_t(ksrc) : uint32_t(layer);
      KvCacheDev& cache = (*caches_)[cache_layer];
      double attn_scale = 1.0;
      attr_f(*node, "scale", attn_scale);
      ok = nn::cuda::flash_attention_chunk_dev_pos(
          dev(in[0]), dev(in[1]), dev(in[2]), dev_mut(cache.k_buf.id),
          dev_mut(cache.v_buf.id), reinterpret_cast<const uint64_t*>(pos_buf_.id),
          cache.n_slots, int(nb), int(lc.n_heads_q), int(cache.n_heads_kv),
          int(cache.head_dim), lc.sliding_window, float(attn_scale), out);
      break;
    }
    case OpKind::KvAppend: {
      // Deferred cache append: copy the chunk's n fp16-rounded K/V rows into the
      // layer's cache at slots (start_pos + c) % n_slots. inputs: {kc, vc, start_pos}.
      int64_t nb = 0;
      batch_attr(nb);
      KvCacheDev& cache = (*caches_)[uint32_t(layer)];
      const int row_elems = int(cache.n_heads_kv * cache.head_dim);
      if (!cache.k_buf || !cache.v_buf ||
          !nn::cuda::kv_append_batched_dev_pos(
              dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id), dev(in[0]),
              dev(in[1]), reinterpret_cast<const uint64_t*>(pos_buf_.id),
              cache.n_slots, row_elems, int(nb))) {
        ok = false;
        break;
      }
      // The append is a cache side effect; the dummy scalar output is never read,
      // so it is left unwritten (a host write to the device handle would fault).
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
    if (extra_handle) device::release(extra_handle, extra_bytes);
    device::release(out_handle, out_bytes);
    return std::unexpected(fail("Sonic CUDA kernel failed for node '" + node->name +
                                "'"));
  }

  // A side-effect-only node (KvAppend) has an orphaned dummy output that
  // the planner drops from task.outputs, so return zero outputs to match the
  // task's declared output count. The output buffer was allocated but is never
  // consumed; release it rather than leak it.
  if (task.outputs.empty()) {
    device::release(out_handle, out_bytes);
    return std::vector<TensorValue>{};
  }

  std::vector<TensorValue> res;
  res.push_back(TensorValue::from_device(out_handle, out_bytes));
  if (extra_handle)
    res.push_back(TensorValue::from_device(extra_handle, extra_bytes));
  return res;
}

} // namespace sonicboom::planner
