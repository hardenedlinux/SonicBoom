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
#include <sonicboom/numerics.h>
#include <sonicboom/planner/plan_record.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include "../nn/cuda_graph.h"
#include "../nn/cuda_resident.h"
#include "../nn/cuda_stream.h"
#include "../quant/cuda_resident.h"
#include "async_prefetch.h"
#include "decision_registry.h"
#include "device_allocator.h"
#include "device_pool.h"
#include "glue_jit.h"

#include <nvtx3/nvToolsExt.h>  // header-only NVTX (no libnvtx link)

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

// Verify matmul launch shape (V1 ablation). Default: one row-invariant batched
// mmvq launch per projection over the step's k+1 query rows. With
// SONICBOOM_SPEC_VERIFY_BATCH=0 the step instead runs one single-column gemv
// launch per row — the pre-batching behavior. The two are bit-identical
// (cuda_resident.h: the batched kernel's per-position arithmetic is the
// single-column kernel's); only the launch shape, and how many times each
// weight block is read, differ. The switch exists to measure what the batched
// launch is worth; it is not a numerics mode.
bool spec_verify_batch_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("SONICBOOM_SPEC_VERIFY_BATCH");
    return !(e && std::string_view(e) == "0");
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
  // Persistent current-token slot; sized for the verify step's widest row set
  // (k+1 <= kMaxVerifyRows) so one Argmax node's row indices land contiguously
  // for a single D2H read. Decode reads element 0 only.
  token_buf_ = g_alloc.allocate(kMaxVerifyRows * sizeof(float));
  // Persistent position slot. Sized for the verify step's widest row set: the
  // attention dispatch hands row r a pointer to element r (the decode kernels
  // read the position from the device slot), so a verify step stages one
  // absolute position per row. Decode stages and reads element 0 only.
  pos_buf_ = g_alloc.allocate(kMaxVerifyRows * sizeof(uint64_t));
  // Prefill chunk token-id slot: sized for the largest chunk (the SWA window),
  // so any segment (<= sliding_window tokens) stages into it in place.
  const uint64_t chunk = spec_.chunk;
  if (chunk > 0)
    tokens_buf_ = g_alloc.allocate(chunk * sizeof(uint64_t));
  if (pos_buf_)
    g_alloc.fill(pos_buf_, 0, kMaxVerifyRows * sizeof(uint64_t));  // defensive: pos starts at 0

}

bool SonicCudaBackend::enable_decode_chunking(uint32_t k) {
  // Chunking is opt-in because it re-points the position/token slots the
  // kernels read: a prefill backend must keep the single-stageable slots.
  if (k <= 1) {
    // Off (also the capture-failed fallback): release the rings so the slot
    // accessors return the single pos/token slots the executor loop stages.
    if (pos_ring_) g_alloc.deallocate(pos_ring_);
    if (token_ring_) g_alloc.deallocate(token_ring_);
    if (carry_buf_) g_alloc.deallocate(carry_buf_);
    pos_ring_ = {};
    token_ring_ = {};
    carry_buf_ = {};
    chunk_ = 1;
    step_index_ = 0;
    return true;
  }
  if (chunk_ > 1) return chunk_ == k;
  hal::Buffer pos_ring = g_alloc.allocate(k * sizeof(uint64_t));
  hal::Buffer token_ring = g_alloc.allocate(k * sizeof(float));
  hal::Buffer carry = g_alloc.allocate(sizeof(float));
  if (!pos_ring || !token_ring || !carry) {
    if (pos_ring) g_alloc.deallocate(pos_ring);
    if (token_ring) g_alloc.deallocate(token_ring);
    if (carry) g_alloc.deallocate(carry);
    return false;
  }
  // Zero the rings so a warm-up step never reads an uninitialized position or
  // token id.
  g_alloc.fill(pos_ring, 0, k * sizeof(uint64_t));
  g_alloc.fill(token_ring, 0, k * sizeof(float));
  g_alloc.fill(carry, 0, sizeof(float));
  pos_ring_ = pos_ring;
  token_ring_ = token_ring;
  carry_buf_ = carry;
  chunk_ = k;
  return true;
}

SonicCudaBackend::~SonicCudaBackend() {
  // The caches_ shared_ptr holds a deleter that frees each layer's device
  // buffers; the last owner (decode or prefill backend) frees them exactly once.
  caches_.reset();
  if (token_buf_) g_alloc.deallocate(token_buf_);
  if (pos_buf_) g_alloc.deallocate(pos_buf_);
  if (tokens_buf_) g_alloc.deallocate(tokens_buf_);
  if (pos_ring_) g_alloc.deallocate(pos_ring_);
  if (token_ring_) g_alloc.deallocate(token_ring_);
  if (carry_buf_) g_alloc.deallocate(carry_buf_);
  if (argmax_mask_) g_alloc.deallocate(argmax_mask_);
  if (sample_params_) g_alloc.deallocate(sample_params_);
  if (sample_window_) g_alloc.deallocate(sample_window_);
  if (sample_cands_) g_alloc.deallocate(sample_cands_);
  if (sample_host_) cudaFreeHost(sample_host_);
}

uint64_t SonicCudaBackend::pos_slot() const {
  if (chunk_ > 1 && pos_ring_)
    return pos_ring_.id + uint64_t(step_index_) * sizeof(uint64_t);
  return pos_buf_.id;
}

uint64_t SonicCudaBackend::token_in_slot() const {
  if (chunk_ > 1 && token_ring_) {
    // Step 0 reads the host-staged carry token; step j reads step j-1's argmax.
    return step_index_ == 0
               ? carry_buf_.id
               : token_ring_.id + uint64_t(step_index_ - 1) * sizeof(float);
  }
  return token_buf_.id;
}

uint64_t SonicCudaBackend::token_out_slot() const {
  if (chunk_ > 1 && token_ring_)
    return token_ring_.id + uint64_t(step_index_) * sizeof(float);
  return token_buf_.id;
}

bool SonicCudaBackend::set_step_index(uint32_t index) {
  if (index >= chunk_) return false;
  step_index_ = index;
  return true;
}

bool SonicCudaBackend::stage_chunk(hal::StepProgram& prog, float carry,
                                   std::span<const uint64_t> positions) {
  if (chunk_ <= 1 || !carry_buf_ || !pos_ring_) return false;
  if (positions.size() != chunk_) return false;
  if (!prog.stage_upload(carry_buf_, &carry, sizeof(float))) return false;
  return prog.stage_upload(pos_ring_, positions.data(),
                           positions.size() * sizeof(uint64_t));
}

bool SonicCudaBackend::read_chunk_tokens(float* host, uint32_t count) const {
  if (chunk_ <= 1 || !token_ring_ || !host) return false;
  if (count > chunk_) return false;
  return g_alloc.copy_from_device(host, token_ring_,
                                  uint64_t(chunk_) * sizeof(float));
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

bool SonicCudaBackend::set_verify_pos(uint64_t start_pos, int rows) {
  if (!pos_buf_ || rows <= 0 || rows > kMaxVerifyRows) return false;
  uint64_t pos[kMaxVerifyRows];
  for (int r = 0; r < rows; ++r) pos[r] = start_pos + uint64_t(r);
  return g_alloc.copy_to_device(pos_buf_, pos, size_t(rows) * sizeof(uint64_t));
}

bool SonicCudaBackend::stage_verify_pos(hal::StepProgram& prog,
                                        uint64_t start_pos, int rows) {
  if (!pos_buf_ || rows <= 0 || rows > kMaxVerifyRows) return false;
  uint64_t pos[kMaxVerifyRows];
  for (int r = 0; r < rows; ++r) pos[r] = start_pos + uint64_t(r);
  return prog.stage_upload(pos_buf_, pos, size_t(rows) * sizeof(uint64_t));
}

bool SonicCudaBackend::read_token(float* host) const {
  if (!token_buf_ || !host) return false;
  return g_alloc.copy_from_device(host, token_buf_, sizeof(float));
}

bool SonicCudaBackend::read_verify_tokens(float* host, int rows) const {
  if (!token_buf_ || !host || rows <= 0 || rows > kMaxVerifyRows) return false;
  return g_alloc.copy_from_device(host, token_buf_, size_t(rows) * sizeof(float));
}

bool SonicCudaBackend::synchronize() const {
  return g_alloc.synchronize();
}

bool SonicCudaBackend::argmax_mask_init(size_t bytes) {
  if (argmax_mask_) return bytes <= argmax_mask_bytes_;
  if (bytes == 0) return false;
  argmax_mask_ = g_alloc.allocate(bytes);
  if (!argmax_mask_) return false;
  argmax_mask_bytes_ = bytes;
  // Start all-allowed: an unconstrained step is then bit-identical to the
  // unmasked kernel. A fresh allocation is not zeroed, and clear_argmax_mask
  // only rewrites a dirty buffer, so mark it dirty first: without this the
  // very first unconstrained decode in a process reads uninitialized mask
  // bytes and the masked argmax can return -1 (nothing allowed).
  argmax_mask_dirty_ = true;
  return clear_argmax_mask();
}

bool SonicCudaBackend::set_argmax_mask(std::span<const uint8_t> mask) {
  if (!argmax_mask_ || mask.size() != argmax_mask_bytes_) return false;
  if (!g_alloc.copy_to_device(argmax_mask_, mask.data(), mask.size())) return false;
  argmax_mask_dirty_ = true;
  return true;
}

bool SonicCudaBackend::clear_argmax_mask() {
  if (!argmax_mask_) return false;
  if (argmax_mask_dirty_ &&
      !g_alloc.fill(argmax_mask_, 0xFF, argmax_mask_bytes_))
    return false;
  argmax_mask_dirty_ = false;
  return true;
}

namespace {

// The device penalty window's contents: the last min(repeat_last_n, |history|)
// token ids as int32. Returns the length, or -1 when repeat_last_n exceeds the
// device limit (kSampleSelectMaxWindow) or a id does not fit.
int fill_sample_window(const nn::SamplingParams& p,
                       std::span<const int64_t> history,
                       std::span<int32_t> window) {
  if (p.repeat_last_n > uint32_t(nn::cuda::kSampleSelectMaxWindow)) return -1;
  const size_t n = std::min<size_t>(p.repeat_last_n, history.size());
  if (n > window.size()) return -1;
  const size_t start = history.size() - n;
  for (size_t i = 0; i < n; ++i) {
    const int64_t t = history[start + i];
    if (t < 0 || t > INT32_MAX) return -1;
    window[i] = int32_t(t);
  }
  return int(n);
}

// The device parameters of one sampling step (mirrors the scalars
// sample_select_dev would otherwise take as launch arguments).
nn::cuda::SampleSelectParams sample_params_of(const nn::SamplingParams& p,
                                              int window_len) {
  return nn::cuda::SampleSelectParams{int32_t(window_len), p.temperature,
                                      p.repeat_penalty, p.presence_penalty,
                                      p.frequency_penalty};
}

}  // namespace

bool SonicCudaBackend::sample_select_init(int vocab) {
  if (vocab <= 0) return false;
  if (sample_cands_) return sample_vocab_ == vocab;
  const uint64_t cand_bytes = uint64_t(nn::kSampleCandidates) *
                                  sizeof(nn::Candidate) +
                              sizeof(int32_t);
  hal::Buffer params = g_alloc.allocate(sizeof(nn::cuda::SampleSelectParams));
  hal::Buffer window =
      g_alloc.allocate(uint64_t(nn::cuda::kSampleSelectMaxWindow) * sizeof(int32_t));
  hal::Buffer cands = g_alloc.allocate(cand_bytes);
  if (!params || !window || !cands ||
      cudaHostAlloc(reinterpret_cast<void**>(&sample_host_), cand_bytes,
                    cudaHostAllocDefault) != cudaSuccess) {
    sample_host_ = nullptr;
    if (params) g_alloc.deallocate(params);
    if (window) g_alloc.deallocate(window);
    if (cands) g_alloc.deallocate(cands);
    return false;
  }
  sample_params_ = params;
  sample_window_ = window;
  sample_cands_ = cands;
  sample_vocab_ = vocab;
  sample_params_stage_.assign(sizeof(nn::cuda::SampleSelectParams), 0);
  sample_window_stage_.assign(nn::cuda::kSampleSelectMaxWindow, 0);
  // A benign initial state (temperature 1, no window, neutral penalties, empty
  // candidate list): the capture's warm-up step and any read before the first
  // stage_sample see a valid configuration instead of a division by a zeroed
  // temperature.
  const auto init_params = sample_params_of(nn::SamplingParams{}, 0);
  if (!g_alloc.copy_to_device(sample_params_, &init_params, sizeof(init_params)) ||
      !g_alloc.copy_to_device(sample_window_, sample_window_stage_.data(),
                              sample_window_stage_.size() * sizeof(int32_t)) ||
      !g_alloc.fill(cands, 0, cand_bytes)) {
    g_alloc.deallocate(params);
    g_alloc.deallocate(window);
    g_alloc.deallocate(cands);
    cudaFreeHost(sample_host_);
    sample_host_ = nullptr;
    sample_params_ = {};
    sample_window_ = {};
    sample_cands_ = {};
    sample_vocab_ = 0;
    return false;
  }
  // The grammar mask is the same bitmap the greedy Argmax node reads, so a
  // constrained sampled step filters exactly like a constrained greedy one.
  if (!argmax_mask_init(size_t((vocab + 7) / 8))) {
    g_alloc.deallocate(params);
    g_alloc.deallocate(window);
    g_alloc.deallocate(cands);
    cudaFreeHost(sample_host_);
    sample_host_ = nullptr;
    sample_params_ = {};
    sample_window_ = {};
    sample_cands_ = {};
    sample_vocab_ = 0;
    return false;
  }
  return true;
}

bool SonicCudaBackend::stage_sample(hal::StepProgram& prog,
                                    const nn::SamplingParams& p,
                                    std::span<const int64_t> history) {
  if (!sample_cands_ || sample_vocab_ <= 0) return false;
  if (!(p.temperature > 0.0f)) return false;  // the greedy graph never selects
  const int len =
      fill_sample_window(p, history, sample_window_stage_);
  if (len < 0) return false;
  const auto params = sample_params_of(p, len);
  std::memcpy(sample_params_stage_.data(), &params, sizeof(params));
  if (!prog.stage_upload(sample_params_, sample_params_stage_.data(),
                         sizeof(params)))
    return false;
  if (len > 0 &&
      !prog.stage_upload(sample_window_, sample_window_stage_.data(),
                         size_t(len) * sizeof(int32_t)))
    return false;
  return true;
}

bool SonicCudaBackend::upload_sample(const nn::SamplingParams& p,
                                     std::span<const int64_t> history) {
  if (!sample_cands_ || sample_vocab_ <= 0) return false;
  if (!(p.temperature > 0.0f)) return false;
  const int len = fill_sample_window(p, history, sample_window_stage_);
  if (len < 0) return false;
  const auto params = sample_params_of(p, len);
  std::memcpy(sample_params_stage_.data(), &params, sizeof(params));
  if (!g_alloc.copy_to_device(sample_params_, sample_params_stage_.data(),
                              sizeof(params)))
    return false;
  return len == 0 ||
         g_alloc.copy_to_device(sample_window_, sample_window_stage_.data(),
                                size_t(len) * sizeof(int32_t));
}

bool SonicCudaBackend::read_sample_candidates(nn::Candidate* out,
                                              int* out_count) const {
  if (!sample_cands_ || !sample_host_ || !out || !out_count) return false;
  constexpr uint64_t cand_bytes =
      uint64_t(nn::kSampleCandidates) * sizeof(nn::Candidate);
  if (!g_alloc.copy_from_device(sample_host_, sample_cands_,
                                cand_bytes + sizeof(int32_t)))
    return false;
  int n = 0;
  std::memcpy(&n, reinterpret_cast<const uint8_t*>(sample_host_) + cand_bytes,
              sizeof(n));
  if (n < 0 || n > nn::kSampleCandidates) return false;
  std::memcpy(out, sample_host_, size_t(n) * sizeof(nn::Candidate));
  *out_count = n;
  return true;
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

std::shared_ptr<std::vector<KvCacheDev>>
SonicCudaBackend::allocate_cache_set(const KvCacheSpec& spec, uint64_t n_ctx) {
  const uint32_t n_kv = spec.n_layers;
  auto* raw = new std::vector<KvCacheDev>();
  raw->resize(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l) {
    const KvCacheSpec::Layer& sl = spec.layers[l];
    KvCacheDev& c = (*raw)[l];
    c.n_heads_kv = sl.kv_heads;
    c.head_dim = sl.head_dim;
    // SWA layers are a fixed-size ring; global layers the full decode length.
    // The ring carries kMaxVerifyRows slots of headroom past the sliding window.
    // A verify step appends its k+1 <= kMaxVerifyRows rows in one pass, and the
    // shared-KV consumer layers read the donor's ring only after every row has
    // appended. With exactly `window` slots, once the ring is full row i+1's
    // append lands on the slot that held position pos+1-window, which is still
    // inside row i's window, and the verify step would compute something k+1
    // plain decode steps never do. With the headroom no row's append reaches a
    // position an earlier row still reads, so the ring is transparent: the
    // kernels address it as pos % n_slots and every window read spans at most
    // `window` resident slots.
    c.n_slots = (sl.window > 0) ? sl.window + kMaxVerifyRows : n_ctx;
    // The standard numerics profile stores f16 or q8 instead of f32: f16
    // halves the cache and is exact for the fp16-rounded values; q8 halves it
    // again (int8 per head_dim value plus an f32 row scale) at the cost of a
    // per-row quantization error. The attention kernels and the append know
    // the row layout from `dtype`.
    const KvStorage storage = sonicboom::numerics_kv_storage();
    c.dtype = storage == KvStorage::F32   ? KvDtype::F32
              : storage == KvStorage::Q8 ? KvDtype::Q8
                                         : KvDtype::F16;
    c.bytes = c.n_slots * c.n_heads_kv * kv_row_bytes(c.dtype, c.head_dim);
    c.k_buf = g_alloc.allocate(c.bytes);
    c.v_buf = g_alloc.allocate(c.bytes);
  }
  {
    // Record the storage decision and its VRAM cost. The dtype is the
    // numerics profile's choice (f16 under standard), and the bytes are what
    // that choice actually allocated for this context class. First record
    // wins, so the cost shown is the process's first (usually decode) cache.
    int64_t total = 0;
    for (const KvCacheDev& c : *raw) total += 2 * int64_t(c.bytes);
    planner::decisions::Entry e;
    e.name = "kv.cache";
    e.scope = "n_ctx=" + std::to_string(n_ctx);
    const KvCacheDev& first = raw->front();
    e.value = raw->empty() ? "empty"
                           : first.q8()    ? "q8"
                           : first.f16()   ? "f16"
                                           : "f32";
    e.source = planner::decisions::Source::Default;
    e.inputs = "n_ctx=" + std::to_string(n_ctx) +
               " layers=" + std::to_string(n_kv);
    e.candidates = "f32|f16|q8";
    e.reason = "the resolved KV storage (numerics profile; SONICBOOM_KV "
               "may override)";
    e.numerics =
        first.q8()
            ? "q8 re-quantizes the fp16-rounded K/V (per-row scale = amax/127)"
            : first.f16() ? "f16 storage is exact for the fp16-rounded K/V "
                            "values"
                          : "f32 storage (the pre-rounding values)";
    e.vram_bytes = total;
    planner::decisions::record(e);
  }
  // The deleter frees each layer's device buffers, so whichever backend is the
  // last owner (after share_caches) deallocates exactly once.
  return std::shared_ptr<std::vector<KvCacheDev>>(
      raw, [](std::vector<KvCacheDev>* v) {
        if (!v) return;
        for (auto& c : *v) {
          if (c.k_buf) g_alloc.deallocate(c.k_buf);
          if (c.v_buf) g_alloc.deallocate(c.v_buf);
        }
        delete v;
      });
}

void SonicCudaBackend::ensure_caches(uint64_t n_ctx) {
  caches_ = allocate_cache_set(spec_, n_ctx);
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

bool SonicCudaBackend::read_cache_row(const KvCacheDev& cache, bool is_k,
                                      uint64_t pos, std::span<float> out) {
  const uint64_t row = cache.n_heads_kv * cache.head_dim;
  const hal::Buffer& buf = is_k ? cache.k_buf : cache.v_buf;
  if (!buf || cache.n_slots == 0 || row == 0 || out.size() < row) return false;
  if (cache.q8()) {
    // q8: each (slot, KV head) row is head_dim int8 values then its f32
    // scale. Download the slot's packed rows and dequantize.
    const uint64_t row_bytes = kv_q8_row_bytes(cache.head_dim);
    std::vector<uint8_t> packed(static_cast<size_t>(row_bytes * cache.n_heads_kv));
    const uint64_t byte_off = (pos % cache.n_slots) * packed.size();
    if (!device::download(buf.id + byte_off, packed.data(), packed.size()))
      return false;
    for (uint64_t h = 0; h < cache.n_heads_kv; ++h) {
      const uint8_t* base = packed.data() + h * row_bytes;
      float scale = 0.0f;
      std::memcpy(&scale, base + cache.head_dim, sizeof(scale));
      const int8_t* q = reinterpret_cast<const int8_t*>(base);
      for (uint64_t i = 0; i < cache.head_dim; ++i)
        out[size_t(h * cache.head_dim + i)] = float(q[i]) * scale;
    }
    return true;
  }
  if (!cache.f16()) {
    const uint64_t byte_off = (pos % cache.n_slots) * row * sizeof(float);
    return device::download(buf.id + byte_off, out.data(), row * sizeof(float));
  }
  // f16 storage: download the packed row and widen exactly.
  const uint64_t byte_off = (pos % cache.n_slots) * row * sizeof(uint16_t);
  std::vector<uint16_t> packed(static_cast<size_t>(row));
  if (!device::download(buf.id + byte_off, packed.data(),
                        row * sizeof(uint16_t)))
    return false;
  for (uint64_t i = 0; i < row; ++i) {
    _Float16 h;
    std::memcpy(&h, &packed[size_t(i)], sizeof(h));
    out[size_t(i)] = float(h);
  }
  return true;
}

std::expected<std::vector<TensorValue>, RuntimeError> SonicCudaBackend::execute(
    const TaskDesc& task, const std::vector<TensorValue>& in) {
  const auto* compute = task.as_compute();
  if (!compute || compute->graph_nodes.empty())
    return std::unexpected(fail("Sonic CUDA task has no compute node"));
  const auto* node = graph_.find_node(compute->graph_nodes.front());
  if (!node)
    return std::unexpected(fail("Sonic CUDA task references an unknown graph node"));

  // VS1 prefetch markers: scheduling bookkeeping, not a kernel. Checked before
  // the output check — a marker has no outputs by construction. The runtime is
  // a no-op while SONICBOOM_ASYNC_PREFETCH is not "on".
  if (node->op == OpKind::AsyncWeightPrefetch || node->op == OpKind::SyncPrefetch) {
    int64_t layer = -1;
    attr_i(*node, "layer", layer);
    int64_t bytes = 0;
    attr_i(*node, "prefetch_bytes", bytes);
    AsyncPrefetchRuntime& prefetch = AsyncPrefetchRuntime::instance();
    if (node->op == OpKind::AsyncWeightPrefetch)
      prefetch.begin(int(layer), bytes > 0 ? uint64_t(bytes) : 0);
    else
      prefetch.wait(int(layer));
    return std::vector<TensorValue>{};
  }

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

  // B3: the verify step's row count (k+1 query positions through one graph).
  // A verify node is the decode op emitted over `rows` positions; its dispatch
  // is the decode kernel once per row (the exact n == 1 arithmetic at pos+i),
  // except the quantized matmul, which is the row-invariant batched mmvq. The
  // attribute is distinct from `batch` so the prefill branches (mmq, batched
  // dense, flash attention) never match a verify node.
  auto verify_attr = [&](int64_t& rows) { return attr_i(*node, "verify", rows); };

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
  //
  // In-place: BufferReusePass marks a same-type elementwise node whose operand
  // 0 dies at this node; the result can be written into that operand's buffer.
  // The op list is restated here so a stale or hand-made attribute cannot make
  // the backend alias a buffer into a kernel that does not support it.
  int64_t inplace = 0;
  attr_i(*node, "inplace", inplace);
  const bool inplace_op = node->op == OpKind::Scale || node->op == OpKind::Add ||
                          node->op == OpKind::Mul || node->op == OpKind::Silu;
  bool borrowed = false;
  uint64_t out_handle = 0;
  if (inplace != 0 && inplace_op && !in.empty() && in[0].on_device &&
      in[0].device.handle != 0 && in[0].device.size_bytes == out_bytes) {
    out_handle = in[0].device.handle;
    borrowed = true;
  }
  if (!out_handle) {
    out_handle = device::acquire(out_bytes);
    if (!out_handle)
      return std::unexpected(fail("device buffer allocation failed"));
  }
  float* out = dev_mut(out_handle);
  const int out_n = int(out_bytes / sizeof(float));

  bool ok = true;
  // A fused region may keep a producer's output as a second result (the prefill
  // add+norm region writes the residual sum as well as the activation). The
  // first output is `out` above; this is the one the region allocates for the
  // second, and it is returned after `out`.
  uint64_t extra_handle = 0;
  uint64_t extra_bytes = 0;
  // A merged gemv node (fuse_sibling_gemv_pass) writes one output per member
  // projection, in node->outputs order; the first is `out` above and these are
  // the rest, allocated here and returned after it.
  std::vector<std::pair<uint64_t, uint64_t>> sibling_outs;
  // NVTX range named after the graph node (its op kind, e.g. "rms_norm",
  // "quantized_matmul"), so nsys attributes each kernel launch to its operator
  // for the per-op launch-count breakdown.
  if (nvtx_enabled()) nvtxRangePushA(node->name.c_str());
  switch (node->op) {
    case OpKind::Scale: {
      ok = nn::cuda::scale_dev(dev(in[0]), float(scale_d), out, out_n);
      break;
    }
    case OpKind::Softcap: {
      // Final-logit softcap on the device (sampled decode, r7): the [vocab]
      // distribution feeds the SampleSelect node without a host round-trip.
      // `scale` is the cap. The prefill graph keeps Softcap on the CPU through
      // lm_head, so this branch is only ever reached by a sampled decode graph.
      ok = nn::cuda::softcap_dev(dev(in[0]), float(scale_d), out, out_n);
      break;
    }
    case OpKind::RmsNorm: {
      int64_t nb = 0, dim_i = 0;
      int64_t vr = 0;
      // `add_input` is set by fuse_residual_epilogue_pass: the residual of the
      // Add that consumed this norm became a second activation input, and the
      // norm kernel applies it in its apply loop (same f32 addition, one
      // launch). Verify/batch forms keep the separate add; the pass does not
      // match them.
      int64_t add_idx = 0;
      const float* add_p = nullptr;
      if (attr_i(*node, "add_input", add_idx) && add_idx > 0 &&
          add_idx < int64_t(in.size()))
        add_p = dev(in[add_idx]);
      if (verify_attr(vr)) {
        // Verify: one decode norm per row (the 1024-thread kernel decode runs;
        // the batched rms_norm_heads kernel uses a different thread count).
        if (vr <= 0 || out_n % int(vr) != 0) { ok = false; break; }
        const int row = out_n / int(vr);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = nn::cuda::rms_norm_dev(fweight(slot), dev(in[0]) + r * row, eps,
                                      out + r * row, row);
      } else if (batch_attr(nb)) {
        // Row norm over [n, dim]: reuse the per-head norm with a single
        // `dim_i`-wide head per row (position), so each row is normalized
        // independently with the full-width weight.
        attr_i(*node, "dim", dim_i);
        ok = nn::cuda::rms_norm_heads_dev(fweight(slot), dev(in[0]), eps, out,
                                          int(dim_i), int(nb));
      } else {
        ok = nn::cuda::rms_norm_dev(fweight(slot), dev(in[0]), eps, out, out_n,
                                    add_p);
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
      int64_t vr = 0;
      int64_t add_idx = 0;
      const float* add_p = nullptr;
      if (attr_i(*node, "add_input", add_idx) && add_idx > 0 &&
          add_idx < int64_t(in.size()))
        add_p = dev(in[add_idx]);
      if (verify_attr(vr) && dim_i > 0) {
        // Verify: per row, the same heads kernel the decode row runs.
        if (vr <= 0 || out_n % int(vr) != 0 ||
            (out_n / int(vr)) % int(dim_i) != 0) {
          ok = false;
          break;
        }
        const int row = out_n / int(vr);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = nn::cuda::rms_norm_heads_dev(w, dev(in[0]) + r * row, eps,
                                            out + r * row, int(dim_i),
                                            row / int(dim_i));
        break;
      }
      ok = nn::cuda::rms_norm_heads_dev(w, dev(in[0]), eps, out, int(dim_i),
                                        out_n / int(dim_i), add_p);
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
      if (int64_t vr = 0; verify_attr(vr))
        nb = vr;  // verify: rows are positions; the batched rope is per-row exact
      if (nb > 0 || batch_attr(nb)) {
        ok = nn::cuda::rope_neox_heads_batched_dev_pos(
            out, int(lc.head_dim), out_n / int(nb) / int(lc.head_dim), int(nb),
            reinterpret_cast<const uint64_t*>(pos_slot()), lc.rope_base, 1.0f, ff);
      } else {
        ok = nn::cuda::rope_neox_heads_dev_pos(
            out, int(lc.head_dim), out_n / int(lc.head_dim),
            reinterpret_cast<const uint64_t*>(pos_slot()), lc.rope_base, 1.0f, ff);
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
      // (inputs {q}). The position comes from the device slot. When the prep
      // regions already stored the row (Stage 6d: `kv_appended`), the append
      // here is skipped and only the attention runs.
      int64_t ksrc = layer;
      attr_i(*node, "kv_source_layer", ksrc);
      int64_t kv_appended = 0;
      attr_i(*node, "kv_appended", kv_appended);
      // Split-KV schedule written by the plan: chunk size in keys (0 = class
      // default). The merge order is fixed (ascending chunk index) and chunk
      // boundaries are absolute, so the attribute alone fully describes the
      // reduction tree.
      int64_t decode_chunk = 0;
      attr_i(*node, "decode_chunk", decode_chunk);
      const uint32_t cache_layer = ksrc >= 0 ? uint32_t(ksrc) : uint32_t(layer);
      KvCacheDev& cache = (*caches_)[cache_layer];
      const int row_elems = int(cache.n_heads_kv * cache.head_dim);
      const uint64_t window = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx_;
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify: one decode attention per query row. Row i appends its K/V at
        // pos+i and attends over the same split/chunk constants as the n == 1
        // call at that position, in ascending row order (row i must see rows
        // 0..i in the cache). The fused-prep cache store is not used: the
        // verify pipeline leaves the append here, where it can be per row.
        if (vr <= 0 || kv_appended != 0 || out_n % int(vr) != 0 ||
            (in.size() >= 4 && (!cache.k_buf || !cache.v_buf))) {
          ok = false;
          break;
        }
        const int q_row = out_n / int(vr);
        double attn_scale = 1.0;
        attr_f(*node, "scale", attn_scale);
        nn::cuda::attention_trace_set_layer(int(layer));
        for (int64_t r = 0; r < vr && ok; ++r) {
          if (in.size() >= 4 &&
              !nn::cuda::kv_append_dev(
                  dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
                  dev(in[1]) + r * row_elems, dev(in[2]) + r * row_elems,
                  reinterpret_cast<const uint64_t*>(pos_slot()), cache.n_slots,
                  row_elems, int(r), cache.f16(), cache.q8(),
                  int(cache.head_dim))) {
            ok = false;
            break;
          }
          ok = nn::cuda::decode_attention_dev(
              dev(in[0]) + r * q_row, dev_mut(cache.k_buf.id),
              dev_mut(cache.v_buf.id),
              reinterpret_cast<const uint64_t*>(pos_slot()), window,
              cache.n_slots, int(lc.n_heads_q), int(cache.n_heads_kv),
              int(cache.head_dim), lc.sliding_window, float(attn_scale),
              out + r * q_row, int(r), cache.f16(), uint64_t(decode_chunk),
              cache.q8());
        }
        break;
      }
      if (in.size() >= 4 && kv_appended == 0) {
        if (!cache.k_buf || !cache.v_buf ||
            !nn::cuda::kv_append_dev(dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
                                     dev(in[1]), dev(in[2]),
                                     reinterpret_cast<const uint64_t*>(pos_slot()),
                                     cache.n_slots, row_elems, 0, cache.f16(),
                                     cache.q8(), int(cache.head_dim))) {
          ok = false;
          break;
        }
      }
      double attn_scale = 1.0;
      attr_f(*node, "scale", attn_scale);
      nn::cuda::attention_trace_set_layer(int(layer));  // no-op when tracing off
      ok = nn::cuda::decode_attention_dev(
          dev(in[0]), dev_mut(cache.k_buf.id), dev_mut(cache.v_buf.id),
          reinterpret_cast<const uint64_t*>(pos_slot()), window, cache.n_slots,
          int(lc.n_heads_q), int(cache.n_heads_kv), int(cache.head_dim),
          lc.sliding_window, float(attn_scale), out, 0, cache.f16(),
          uint64_t(decode_chunk), cache.q8());
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
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify without a quantize fusion (rare: the pipeline quantizes every
        // activation feeding a matmul): one decode gemv per row.
        if (vr <= 0 || out_n % int(vr) != 0) { ok = false; break; }
        const int rows = out_n / int(vr);
        const int cols = int((in[0].device.size_bytes / sizeof(float)) / vr);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = quant::cuda::matvec_f32_dev(qweight(slot),
                                           dev(in[0]) + r * cols,
                                           out + r * rows);
      } else if (batch_attr(nb)) {
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
    case OpKind::MatmulQ8Tc: {
      // MMQ1: the prefill dispatch pass's choice of the generated int8
      // tensor-core kernel over the same q8_1 activation and weight slot as the
      // batched QuantizedMatmulQ8 it was rewritten from. A node whose copy
      // cannot be built or whose shape the generator does not serve falls back
      // to the ggml mmq kernel here, so a dispatch is never a correctness
      // condition.
      note_weight_quant(qweight(slot));
      int64_t nb = 0;
      int64_t ds_i = 0;
      attr_i(*node, "ds", ds_i);
      const auto* xq =
          reinterpret_cast<const quant::cuda::BlockQ8_1Mmq*>(dev(in[0]));
      if (batch_attr(nb) && nb > 0) {
        note_activation_quant("block_q8_1_mmq(tc)");
        ok = quant::cuda::matmul_mmq_tc_dev(qweight(slot), xq, out, int(nb),
                                            int(ds_i));
        if (!ok) {
          note_activation_quant(mmq_activation(ds_i));
          ok = quant::cuda::matmul_mmq_q8_dev(qweight(slot), xq, out, int(nb));
        }
      } else {
        // Only the batched prefill form is dispatched here (the pass rewrites
        // nothing else); a node without one is a plan error, not a no-op.
        ok = false;
      }
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
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify: the row-invariant batched mmvq over k+1 positions (B2: the
        // batched kernel's per-position arithmetic is the single-column gemv's).
        // The bias/residual folds are elementwise epilogues here, which store
        // the same f32 values as the fused gemv epilogue.
        if (vr <= 0 || out_n % int(vr) != 0) { ok = false; break; }
        note_activation_quant("q8_1");
        const auto& qw = qweight(slot);
        const auto* xq = reinterpret_cast<const quant::cuda::BlockQ8_1*>(dev(in[0]));
        int64_t bias_slot = 0, add_idx = 0;
        const bool has_bias = attr_i(*node, "bias_slot", bias_slot);
        const bool has_add = attr_i(*node, "add_input", add_idx);
        const std::span<const float> b =
            has_bias ? fweight(static_cast<model::WeightSlot>(bias_slot))
                     : std::span<const float>{};
        const float* add_p =
            has_add ? static_cast<const float*>(dev(in[add_idx])) : nullptr;
        if (spec_verify_batch_enabled()) {
          ok = quant::cuda::gemv_q8_1_batched_dev(qw, xq, out, int(vr));
        } else {
          // Serial ablation: one decode gemv per query row (the activation row
          // is [cols/32] q8_1 blocks, the output row is out_n/vr floats).
          const int rows_out = int(out_n / vr);
          const int bpr = int(qw.dims()[0] / 32);
          ok = true;
          for (int64_t r = 0; r < vr && ok; ++r)
            ok = quant::cuda::gemv_q8_1_dev(qw, xq + r * bpr,
                                            out + r * rows_out);
        }
        if (ok && has_bias && !b.empty())
          ok = nn::cuda::add_bias_dev(b, out, out, out_n);
        if (ok && has_add) ok = nn::cuda::add_dev(add_p, out, out, out_n);
        break;
      }
      if (batch_attr(nb)) {
        int64_t ds_i = 0;
        attr_i(*node, "ds", ds_i);
        note_activation_quant(mmq_activation(ds_i));
        // The node's `decomp` attribute is the plan's work decomposition for
        // this weight shape (set by the emitter from mmq_class_decomp); absent
        // or unrecognized leaves Auto, which the runtime resolves by the same
        // weight-shape policy. The attribute never depends on the batch: split-K
        // changes the summation order, so a choice that moved with the segment
        // length would break prefix-cache batch invariance.
        std::string_view decomp_s;
        quant::cuda::MmqDecomp decomp = quant::cuda::MmqDecomp::Auto;
        if (attr_s(*node, "decomp", decomp_s)) {
          if (decomp_s == "stream_k") decomp = quant::cuda::MmqDecomp::StreamK;
          else if (decomp_s == "tiled") decomp = quant::cuda::MmqDecomp::Tiled;
        }
        ok = quant::cuda::matmul_mmq_q8_dev(
            qweight(slot),
            reinterpret_cast<const quant::cuda::BlockQ8_1Mmq*>(dev(in[0])), out,
            int(nb), decomp);
      } else {
        note_activation_quant("q8_1");
        const auto* xq = reinterpret_cast<const quant::cuda::BlockQ8_1*>(dev(in[0]));
        const auto& qw = qweight(slot);
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
        // One projection: the gemv with whatever epilogue its type supports,
        // then the elementwise kernels for a bias/residual it could not take
        // (same values, more launches).
        auto run_one = [&](const quant::QuantizedTensor& w,
                           std::span<const float> bias, const float* add_p,
                           float* y, int n) -> bool {
          const bool ep = quant::cuda::gemv_bias_epilogue_supported(w.type());
          const bool bias_fused = !bias.empty() && bias.size() == size_t(n) && ep;
          const bool add_fused = add_p != nullptr && ep;
          bool r = quant::cuda::gemv_q8_1_dev(
              w, xq, y, bias_fused ? bias : std::span<const float>{},
              add_fused ? add_p : nullptr);
          if (r && !bias.empty() && !bias_fused)
            r = nn::cuda::add_bias_dev(bias, y, y, n);
          if (r && add_p && !add_fused) r = nn::cuda::add_dev(add_p, y, y, n);
          return r;
        };
        const float* add_p = has_add ? static_cast<const float*>(dev(in[add_idx]))
                                     : nullptr;
        // `s1_slot`/`s2_slot` are set by fuse_sibling_gemv_pass: this node is
        // the merged form of two or three projections that read the same
        // activation (the q/k/v trio, the gate/up pair), one output each. They
        // run as one launch — bit-identical per row — when the merged kernel
        // takes them; otherwise the members run as separate launches in order,
        // which is exactly what the unmerged plan did.
        int64_t s1_slot = 0, s2_slot = 0;
        const bool has_s1 = attr_i(*node, "s1_slot", s1_slot);
        if (has_s1 && node->outputs.size() >= 2) {
          const bool has_s2 =
              attr_i(*node, "s2_slot", s2_slot) && node->outputs.size() >= 3;
          const size_t members = has_s2 ? 3 : 2;
          const int64_t slots[3] = {slot_i, s1_slot, s2_slot};
          std::vector<quant::cuda::GemvSibling> sib;
          std::vector<std::span<const float>> biases;
          sib.reserve(members);
          biases.reserve(members);
          bool ready = true;
          int rows[3] = {out_n, 0, 0};
          for (size_t k = 0; k < members && ready; ++k) {
            const auto* td = graph_.find_tensor(node->outputs[k]);
            if (!td) { ready = false; break; }
            rows[k] = int(td->size_bytes / sizeof(float));
            int64_t bias_idx = 0;
            bool hb = false;
            if (k == 0) { bias_idx = bias_slot; hb = has_bias; }
            else if (k == 1) hb = attr_i(*node, "s1_bias_slot", bias_idx);
            else hb = attr_i(*node, "s2_bias_slot", bias_idx);
            const auto& w = qweight(static_cast<model::WeightSlot>(slots[k]));
            biases.push_back(hb ? fweight(static_cast<model::WeightSlot>(bias_idx))
                                : std::span<const float>{});
            float* yp = out;
            if (k > 0) {
              const uint64_t bytes = td->size_bytes;
              const uint64_t h = bytes ? device::acquire(bytes) : 0;
              if (!h) { ready = false; break; }
              sibling_outs.push_back({h, bytes});
              yp = dev_mut(h);
            }
            sib.push_back(quant::cuda::GemvSibling{&w, biases.back(), yp});
          }
          if (ready && quant::cuda::gemv_q8_1_siblings_dev(sib, xq)) {
            ok = true;
          } else if (ready) {
            ok = true;
            for (size_t k = 0; k < members && ok; ++k) {
              float* yp = k == 0 ? out : dev_mut(sibling_outs[k - 1].first);
              ok = run_one(*sib[k].weight, biases[k],
                           k == 0 ? add_p : nullptr, yp, rows[k]);
            }
          } else {
            ok = false;
          }
          break;  // the node's outputs are all written by the branch above
        }
        ok = run_one(qw, b, add_p, out, out_n);
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
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify: one decode norm+quantize per row into that row's contiguous
        // q8_1 blocks, so the following batched mmvq reads [rows, cols/32].
        if (vr <= 0) { ok = false; break; }
        note_activation_quant("q8_1");
        const int row = int((in[0].device.size_bytes / sizeof(float)) / vr);
        if (row <= 0 || row % 32 != 0) { ok = false; break; }
        auto* yq = reinterpret_cast<quant::cuda::BlockQ8_1*>(out);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = quant::cuda::rms_norm_quantize_q8_1_dev(
              fweight(slot), dev(in[0]) + r * row, eps,
              yq + r * (row / 32), row);
        break;
      }
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
        auto run_row = [&](const float* xp, quant::cuda::BlockQ8_1* yp,
                           int n, bool use_glue) {
          if (use_glue) {
            void* xarg = const_cast<float*>(xp);
            void* warg = const_cast<void*>(wd);
            void* yarg = yp;
            int narg = n;
            float earg = eps;
            void* args[] = {&xarg, &warg, &yarg, &narg, &earg};
            if (glue::launch(region, args, 1, compiler::kGlueBlockThreads,
                             compiler::kGlueDynamicSharedBytes))
              return true;
          }
          // The generated kernel is byte-identical to the hand kernel, which is
          // also the fallback when it cannot be compiled or loaded.
          return quant::cuda::rms_norm_quantize_q8_1_dev(w, xp, eps, yp, n);
        };
        int64_t vr = 0;
        if (verify_attr(vr) && vr > 0) {
          // Verify: one row per call, into that row's contiguous q8_1 blocks.
          const int row = int((in[0].device.size_bytes / sizeof(float)) / vr);
          auto* yq = reinterpret_cast<quant::cuda::BlockQ8_1*>(out);
          if (row <= 0 || row % 32 != 0) { ok = false; break; }
          for (int64_t r = 0; r < vr && ok; ++r)
            ok = run_row(dev(in[0]) + r * row, yq + r * (row / 32), row, true);
          break;
        }
        ok = run_row(dev(in[0]), reinterpret_cast<quant::cuda::BlockQ8_1*>(out),
                     int(in[0].device.size_bytes / sizeof(float)), true);
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
      if (region == "rms_norm_add") {
        // Prefill norm + residual add (Stage 5c): one result, the residual
        // stream with the normalized row added. Operands are (x, residual);
        // `slot` and `dim` come from the norm the region replaced.
        int64_t nb = 0, dim_i = 0;
        if (!batch_attr(nb) || !attr_i(*node, "dim", dim_i) || nb <= 0 ||
            dim_i <= 0) {
          ok = false;
          break;
        }
        const std::span<const float> w = fweight(slot);
        const void* wd =
            w.empty()
                ? nullptr
                : quant::cuda::upload_bytes_cached(w.data(),
                                                   w.size() * sizeof(float));
        if (w.empty() || wd == nullptr) {
          ok = false;
          break;
        }
        // The epilogue factor of the scalar scale a single-reader chain ran
        // after the region (FoldScaleIntoGluePass); 1.0f when there was none.
        double epi_scale = 1.0;
        attr_f(*node, "epilogue_scale", epi_scale);
        const float sarg = float(epi_scale);
        // The generated kernel's ABI: (x, r, w, y, dim, eps, scale), one block
        // per row.
        void* xarg = const_cast<float*>(dev(in[0]));
        void* rarg = const_cast<float*>(dev(in[1]));
        void* warg = const_cast<void*>(wd);
        void* yarg = out;
        int dimarg = int(dim_i);
        float earg = eps;
        float farg = sarg;
        void* args[] = {&xarg, &rarg, &warg, &yarg, &dimarg, &earg, &farg};
        ok = glue::launch(region, args, int(nb),
                          compiler::kRmsNormAddBlockThreads,
                          compiler::kRmsNormAddSharedBytes);
        if (!ok) {
          // The reference pair: norm into a scratch row set, then add, then the
          // scale the region folded.
          const uint64_t tmp = device::acquire(out_bytes);
          if (!tmp) {
            ok = false;
            break;
          }
          ok = nn::cuda::rms_norm_heads_dev(w, dev(in[0]), eps, dev_mut(tmp),
                                            int(dim_i), int(nb));
          if (ok)
            ok = nn::cuda::add_dev(dev(in[1]), dev_mut(tmp), out,
                                   int(nb * dim_i));
          if (ok && sarg != 1.0f)
            ok = nn::cuda::scale_dev(out, sarg, out, out_n);
          device::release(tmp, out_bytes);
        }
        break;
      }
      if (region == "slice_scale") {
        // The PLE column extraction (Stage 5e): a strided row slice and the
        // scale that follows it in one generated kernel. The node carries the
        // slice's offset/length/stride and the scale factor. One thread per
        // output element; the fallback is the reference pair (slice into a
        // scratch row set, then scale into the output).
        int64_t offset_i = 0, length_i = 0, stride_i = 0;
        double scale_d = 1.0;
        if (!attr_i(*node, "offset", offset_i) ||
            !attr_i(*node, "length", length_i) ||
            !attr_i(*node, "stride", stride_i) || length_i <= 0 ||
            stride_i <= 0 || !attr_f(*node, "scale", scale_d)) {
          ok = false;
          break;
        }
        const int len = int(length_i);
        if (out_n <= 0 || out_n % len != 0) {
          ok = false;
          break;
        }
        const int rows = out_n / len;
        // The generated kernel's ABI: (x, y, rows, length, offset, stride,
        // scale), one thread per output element.
        void* xarg = const_cast<float*>(dev(in[0]));
        void* yarg = out;
        int rowsarg = rows;
        int lenarg = len;
        int offarg = int(offset_i);
        int stridearg = int(stride_i);
        float sarg = float(scale_d);
        void* args[] = {&xarg,    &yarg,    &rowsarg, &lenarg,
                        &offarg,  &stridearg, &sarg};
        const int grid = (out_n + compiler::kSliceScaleBlockThreads - 1) /
                         compiler::kSliceScaleBlockThreads;
        ok = glue::launch(region, args, grid,
                          compiler::kSliceScaleBlockThreads, 0);
        if (!ok) {
          const uint64_t tmp = device::acquire(out_bytes);
          if (!tmp) {
            ok = false;
            break;
          }
          ok = nn::cuda::slice_strided_dev(dev(in[0]), dev_mut(tmp), rows, len,
                                           int(offset_i), int(stride_i));
          if (ok)
            ok = nn::cuda::scale_dev(dev_mut(tmp), float(scale_d), out, out_n);
          device::release(tmp, out_bytes);
        }
        break;
      }
      if (region == "gelu_mul") {
        // The gated activation (Stage 5f): gelu_fp16(gate) * up in one generated
        // kernel. Operands are (gate, up); one element per thread. The fallback
        // is the reference pair through a scratch row set.
        if (in.size() < 2 || out_n <= 0) {
          ok = false;
          break;
        }
        // The generated kernel's ABI: (gate, up, y, n).
        void* garg = const_cast<float*>(dev(in[0]));
        void* uarg = const_cast<float*>(dev(in[1]));
        void* yarg = out;
        int narg = out_n;
        void* args[] = {&garg, &uarg, &yarg, &narg};
        const int grid = (out_n + compiler::kGeluMulBlockThreads - 1) /
                         compiler::kGeluMulBlockThreads;
        ok = glue::launch(region, args, grid,
                          compiler::kGeluMulBlockThreads, 0);
        if (!ok) {
          const uint64_t tmp = device::acquire(out_bytes);
          if (!tmp) {
            ok = false;
            break;
          }
          ok = nn::cuda::gelu_fp16_dev(dev(in[0]), dev_mut(tmp), out_n);
          if (ok)
            ok = nn::cuda::mul_dev(dev_mut(tmp), dev(in[1]), out, out_n);
          device::release(tmp, out_bytes);
        }
        break;
      }
      if (region == "silu_mul") {
        // The SwiGLU form of the gated activation: silu(gate) * up in one
        // generated kernel. Same operands and geometry as gelu_mul; the
        // fallback is silu_kernel then mul_kernel through a scratch row set.
        if (in.size() < 2 || out_n <= 0) {
          ok = false;
          break;
        }
        // The generated kernel's ABI: (gate, up, y, n).
        void* garg = const_cast<float*>(dev(in[0]));
        void* uarg = const_cast<float*>(dev(in[1]));
        void* yarg = out;
        int narg = out_n;
        void* args[] = {&garg, &uarg, &yarg, &narg};
        const int grid = (out_n + compiler::kSiluMulBlockThreads - 1) /
                         compiler::kSiluMulBlockThreads;
        ok = glue::launch(region, args, grid,
                          compiler::kSiluMulBlockThreads, 0);
        if (!ok) {
          const uint64_t tmp = device::acquire(out_bytes);
          if (!tmp) {
            ok = false;
            break;
          }
          ok = nn::cuda::silu_dev(dev(in[0]), dev_mut(tmp), out_n);
          if (ok)
            ok = nn::cuda::mul_dev(dev_mut(tmp), dev(in[1]), out, out_n);
          device::release(tmp, out_bytes);
        }
        break;
      }
      if (region == "heads_norm_rope_cast") {
        // Attention prep (Stage 5d): per-head RMS-norm, NeoX RoPE on the Q/K
        // chains, fp16 rounding on the K/V chains, in one kernel. The norm
        // carries layer/slot (and `batch` in prefill); `rope`/`cast` say which
        // chain this is. One 256-thread block per head: the head count is the
        // region node's element count / head_dim.
        int64_t rope_i = 0, cast_i = 0, kv_store = 0, nb = 1;
        attr_i(*node, "rope", rope_i);
        attr_i(*node, "cast", cast_i);
        // Stage 6d: a KV layer's K/V prep region stores its own output row into
        // that layer's cache (kv_store 1 = K, 2 = V); the attention node it
        // feeds is marked `kv_appended` and skips its kv_append launches.
        attr_i(*node, "kv_store", kv_store);
        KvCacheDev* kv = nullptr;
        int kv_slots = 0, kv_row = 0;
        float* kv_dst = nullptr;
        if (kv_store != 0) {
          if (layer < 0 || size_t(layer) >= caches_->size() ||
              !(*caches_)[layer].k_buf || !(*caches_)[layer].v_buf) {
            ok = false;
            break;
          }
          kv = &(*caches_)[layer];
          kv_slots = int(kv->n_slots);
          kv_row = int(kv->n_heads_kv * kv->head_dim);
          kv_dst = dev_mut(kv_store == 1 ? kv->k_buf.id : kv->v_buf.id);
          if (kv_slots <= 0 || kv_row <= 0 || kv_dst == nullptr) {
            ok = false;
            break;
          }
        }
        int64_t vr = 0;
        if (verify_attr(vr)) {
          // Verify: the rows are positions; heads_per_pos = heads/rows, which
          // is exactly the decode chain's head count, and the batched rope
          // computes pos+i from the row index (same per-row arithmetic).
          if (vr <= 0) { ok = false; break; }
          nb = vr;
          if (kv_store != 0) { ok = false; break; }  // per-row append is attn's
        } else if (!batch_attr(nb)) {
          nb = 1;
        }
        const int dim = int(lc.head_dim);
        const int heads = dim > 0 ? out_n / dim : 0;
        if (dim <= 0 || (dim & 1) != 0 || heads <= 0 || out_n % dim != 0 ||
            nb <= 0 || heads % int(nb) != 0) {
          // Malformed for the reference chain too (RoPE pairs and the batched
          // launch both need these): fail rather than write a wrong buffer.
          ok = false;
          break;
        }
        const int hpp = heads / int(nb);
        const std::span<const float> w = fweight(slot);
        const std::span<const float> ff =
            rope_i != 0 && lc.attention == model::AttentionKind::Global
                ? model_->weights.rope_freqs
                : std::span<const float>{};
        // The reference chain: norm, then in-place rope (batched when the
        // launch covers more than one position), then the fp16 rounding.
        auto reference = [&]() {
          ok = nn::cuda::rms_norm_heads_dev(w, dev(in[0]), eps, out, dim,
                                            heads);
          if (ok && rope_i != 0) {
            ok = nb == 1
                     ? nn::cuda::rope_neox_heads_dev_pos(
                           out, dim, heads,
                           reinterpret_cast<const uint64_t*>(pos_slot()),
                           lc.rope_base, 1.0f, ff)
                     : nn::cuda::rope_neox_heads_batched_dev_pos(
                           out, dim, hpp, int(nb),
                           reinterpret_cast<const uint64_t*>(pos_slot()),
                           lc.rope_base, 1.0f, ff);
          }
          if (ok && cast_i != 0) ok = nn::cuda::cast_fp16_dev(out, out_n);
          // A cache-writing chain appends here too, so the fallback leaves the
          // cache exactly as the generated kernel would have. kv_append writes
          // both destinations; passing the one buffer for both is how a
          // single-buffer append runs through the existing kernel.
          if (ok && kv_store != 0) {
            ok = nn::cuda::kv_append_dev(
                kv_dst, kv_dst, out, out,
                reinterpret_cast<const uint64_t*>(pos_slot()), kv->n_slots,
                kv_row, 0, kv->f16(), kv->q8(), int(kv->head_dim));
          }
        };
        // The region is an optimization: a head wider than its shared stage
        // falls back to the reference chain (Gemma's global layers are 512,
        // its local layers 256; the region covers both).
        if (dim > compiler::kHeadsNormRopeCastMaxHeadDim) {
          reference();
          break;
        }
        // A q8 cache has no generated store variant (the store is per element,
        // the scale needs the whole row): the reference chain writes y and
        // appends the q8 row. (Stage 6d emits kv_store nowhere today; this
        // keeps the arm correct if a plan starts setting it.)
        if (kv_store != 0 && kv->q8()) {
          reference();
          break;
        }
        const void* wd =
            w.empty()
                ? nullptr
                : quant::cuda::upload_bytes_cached(w.data(),
                                                   w.size() * sizeof(float));
        const void* ffd =
            ff.empty()
                ? nullptr
                : quant::cuda::upload_bytes_cached(ff.data(),
                                                   ff.size() * sizeof(float));
        if ((!w.empty() && wd == nullptr) || (!ff.empty() && ffd == nullptr)) {
          reference();
          break;
        }
        const float log_base = std::log(lc.rope_base);
        // The generated kernel's ABI: (x, w, has_w, y, pos, head_dim, eps,
        // freq_scale, log_base, ff, heads_per_pos, do_rope, do_cast, cache,
        // n_slots, row_elems, do_store). The last four are the Stage 6d cache
        // store; `cache` is null and `do_store` 0 for a chain that writes none.
        void* xarg = const_cast<float*>(dev(in[0]));
        void* warg = const_cast<void*>(wd);
        int haswarg = w.empty() ? 0 : 1;
        void* yarg = out;
        void* posarg = reinterpret_cast<void*>(pos_slot());
        int dimarg = dim;
        float earg = eps;
        float fsarg = 1.0f;
        float lbarg = log_base;
        void* ffarg = const_cast<void*>(ffd);
        int hpparg = hpp;
        int ropearg = int(rope_i);
        int castarg = int(cast_i);
        void* carg = kv_dst;
        int slotsarg = kv_slots;
        int rowarg = kv_row;
        int storearg = int(kv_store);
        void* args[] = {&xarg,   &warg,    &haswarg, &yarg,   &posarg, &dimarg,
                        &earg,   &fsarg,   &lbarg,   &ffarg,  &hpparg, &ropearg,
                        &castarg, &carg,   &slotsarg, &rowarg, &storearg};
        // The standard profile's cache holds f16: launch the store variant
        // (distinct kernel and PTX cache entry) rather than writing f32.
        const std::string launch_region =
            kv_store != 0 && kv->f16() ? std::string(region) + "-kv16"
                                       : std::string(region);
        ok = glue::launch(launch_region, args, heads,
                          compiler::kHeadsNormRopeCastBlockThreads,
                          compiler::kHeadsNormRopeCastSharedBytes);
        if (!ok) reference();
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
      } else if (int64_t vr = 0; verify_attr(vr)) {
        // Verify: the decode fusion once per row (row-contiguous q8_1 blocks).
        if (vr <= 0) { ok = false; break; }
        note_activation_quant("q8_1");
        const int row = int((in[0].device.size_bytes / sizeof(float)) / vr);
        if (row <= 0 || row % 32 != 0) { ok = false; break; }
        auto* yq = reinterpret_cast<quant::cuda::BlockQ8_1*>(out);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = quant::cuda::gelu_mul_quantize_q8_1_dev(
              dev(in[0]) + r * row, dev(in[1]) + r * row, yq + r * (row / 32),
              row);
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
      } else if (int64_t vr = 0; verify_attr(vr)) {
        // Verify: the decode fusion once per row (row-contiguous q8_1 blocks).
        if (vr <= 0) { ok = false; break; }
        note_activation_quant("q8_1");
        const int row = int((in[0].device.size_bytes / sizeof(float)) / vr);
        if (row <= 0 || row % 32 != 0) { ok = false; break; }
        auto* yq = reinterpret_cast<quant::cuda::BlockQ8_1*>(out);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = quant::cuda::silu_mul_quantize_q8_1_dev(
              dev(in[0]) + r * row, dev(in[1]) + r * row, yq + r * (row / 32),
              row);
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
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify: the dense projection is a decode instance per row (the
        // batched dense kernel is not the n == 1 kernel; B2).
        if (vr <= 0 || out_n % int(vr) != 0) { ok = false; break; }
        const int cols =
            int(in[0].device.size_bytes / sizeof(float) / uint64_t(vr));
        const int rows = out_n / int(vr);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = nn::cuda::matvec_f32_dev(fweight(slot), cols, dev(in[0]) + r * cols,
                                        out + r * rows, rows);
        break;
      }
      if (batch_attr(nb)) {
        const int n = int(nb);
        const int cols = int(in[0].device.size_bytes / sizeof(float) / uint64_t(n));
        const int rows = out_n / n;
        // Dense layout propagation: the activation is already position-major
        // (the producer's own layout), so the batched GEMM reads it in place.
        std::string_view layout;
        const bool strided = attr_s(*node, "dense_layout", layout) &&
                             layout == "strided";
        ok = nn::cuda::matvec_f32_batched_dev(fweight(slot), cols, dev(in[0]), out,
                                              rows, n, strided);
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
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify: the decode dense instance per row (bf16-round the row on a
        // scratch copy, then the single-row gemv), as the decode path does.
        if (vr <= 0 || out_n % int(vr) != 0) { ok = false; break; }
        const int cols =
            int(in[0].device.size_bytes / sizeof(float) / uint64_t(vr));
        const uint64_t xbytes = uint64_t(cols) * sizeof(float);
        uint64_t xtmp = device::acquire(xbytes);
        if (!xtmp) { ok = false; break; }
        for (int64_t r = 0; r < vr && ok; ++r) {
          ok = device::device_copy(in[0].device.handle +
                                       uint64_t(r) * xbytes,
                                   xtmp, xbytes) &&
               nn::cuda::cast_bf16_dev(dev_mut(xtmp), cols) &&
               nn::cuda::matvec_bf16_dev(W, cols,
                                         reinterpret_cast<const float*>(xtmp),
                                         out + r * int(rows), int(rows));
        }
        device::release(xtmp, xbytes);
        break;
      }
      int64_t nb = 0;
      const bool batched = batch_attr(nb);
      const int n = int(nb);
      const int cols =
          int(in[0].device.size_bytes / sizeof(float) / uint64_t(batched ? n : 1));
      // Dense layout propagation: in the strided batched path the dense kernel
      // rounds the position-major activation itself, so the staging copy and
      // the separate bf16 cast are not needed.
      std::string_view layout;
      const bool strided = batched && attr_s(*node, "dense_layout", layout) &&
                           layout == "strided";
      // ScaleIntoDensePass folds a single-reader `scale` into the batched
      // projection (the per-layer model scaling); it only marks the batched
      // form, so the decode gemv below needs no epilogue.
      double epi_scale = 1.0;
      const float alpha = attr_f(*node, "epilogue_scale", epi_scale)
                              ? float(epi_scale)
                              : 1.0f;
      if (strided) {
        ok = nn::cuda::matvec_bf16_batched_dev(W, cols, dev(in[0]), out,
                                               int(rows), n, true, alpha);
        break;
      }
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
      int64_t vr = 0;
      if (verify_attr(vr) && stride == 0) {
        // Verify: the same sub-span copy per row of a [rows, total_per] input.
        if (vr <= 0 || length <= 0) { ok = false; break; }
        const int total = int(in[0].device.size_bytes / sizeof(float) / vr);
        if (total <= 0 || offset + length > total) { ok = false; break; }
        const int len = int(length);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = device::device_copy(in[0].device.handle +
                                       (uint64_t(r) * uint64_t(total) +
                                        uint64_t(offset)) * sizeof(float),
                                   out_handle + uint64_t(r) * uint64_t(len) *
                                                    sizeof(float),
                                   uint64_t(len) * sizeof(float));
        break;
      }
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
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify: one gather per query row, ids from the staged device slot.
        // A gather copies whole rows, so the batched form is row-invariant.
        const uint64_t* tokens_dev =
            reinterpret_cast<const uint64_t*>(tokens_buf_.id);
        if (!tokens_dev || vr <= 0) { ok = false; break; }
        if (slot == model::WeightSlot::PerLayerTokenEmbd)
          ok = quant::cuda::gather_f32_mapped_dev_batched_ids(
              model_->weights.per_layer_token_embd, tokens_dev, int(vr), out);
        else
          ok = quant::cuda::gather_f32_resident_dev_batched_ids(
              model_->weights.token_embd, tokens_dev, int(vr), out);
      } else if (batch_attr(nb)) {
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
        const float* tok_dev = dev_mut(token_in_slot());
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
      int64_t vr = 0;
      if (verify_attr(vr)) {
        // Verify: one reduction per query row over its [vocab] slice, then the
        // rows copied into the persistent token slot for one D2H read. The row
        // width comes from the logits input (the output is [rows], one index
        // per row, so out_n is the row count, not the vocabulary).
        if (vr <= 0 || in[0].device.size_bytes % (uint64_t(vr) * sizeof(float)) !=
                          0) {
          ok = false;
          break;
        }
        const int vocab =
            int(in[0].device.size_bytes / sizeof(float) / uint64_t(vr));
        if (vocab <= 0) { ok = false; break; }
        const bool masked =
            argmax_mask_ && argmax_mask_bytes_ >= size_t((vocab + 7) / 8);
        const auto* mask = reinterpret_cast<const uint8_t*>(
            masked ? uintptr_t(argmax_mask_.id) : 0);
        for (int64_t r = 0; r < vr && ok; ++r)
          ok = masked ? nn::cuda::argmax_masked_dev(dev(in[0]) + r * vocab, vocab,
                                                    mask, out + r)
                      : nn::cuda::argmax_dev(dev(in[0]) + r * vocab, vocab, out + r);
        if (ok && token_buf_)
          ok = device::device_copy(out_handle, token_out_slot(),
                                   size_t(vr) * sizeof(float));
        break;
      }
      // The kernel choice is part of a captured decode graph: a backend whose
      // mask buffer was initialized before capture always runs the masked form
      // (all-ones = the unmasked result, bit for bit).
      if (argmax_mask_) {
        const int n = int(in[0].device.size_bytes / sizeof(float));
        if (argmax_mask_bytes_ < size_t((n + 7) / 8)) { ok = false; break; }
        ok = nn::cuda::argmax_masked_dev(
            dev(in[0]), n,
            reinterpret_cast<const uint8_t*>(uintptr_t(argmax_mask_.id)), out);
      } else {
        ok = nn::cuda::argmax_dev(dev(in[0]),
                                  int(in[0].device.size_bytes / sizeof(float)), out);
      }
      if (ok && token_buf_)
        ok = device::device_copy(out_handle, token_out_slot(), sizeof(float));
      break;
    }
    case OpKind::SampleSelect: {
      // Sampled decode (r7): the terminal candidate selection. The [vocab]
      // logits stay on the device; the kernel writes the kSampleCandidates
      // best (id, z) pairs into the backend's persistent candidate buffer and
      // this node's output receives only the 4-byte count, which is all the
      // host reads before drawing the token itself (nn::sample_from_candidates).
      // The penalty window and the parameter block are persistent device
      // buffers staged by stage_sample() before replay; the grammar mask is
      // the same argmax_mask_ bitmap the greedy Argmax node applies.
      const int vocab = int(in[0].device.size_bytes / sizeof(float));
      if (vocab <= 0 || !sample_cands_ || sample_vocab_ != vocab ||
          out_bytes < sizeof(int32_t)) {
        ok = false;
        break;
      }
      const bool masked =
          argmax_mask_ && argmax_mask_bytes_ >= size_t((vocab + 7) / 8);
      const uint64_t cands = sample_cands_.id;
      const uint64_t count = cands + uint64_t(nn::kSampleCandidates) *
                                          sizeof(nn::Candidate);
      ok = nn::cuda::sample_select_dev_dyn(
          dev(in[0]), vocab,
          masked ? reinterpret_cast<const uint8_t*>(uintptr_t(argmax_mask_.id))
                 : nullptr,
          reinterpret_cast<const int32_t*>(uintptr_t(sample_window_.id)),
          reinterpret_cast<const nn::cuda::SampleSelectParams*>(
              uintptr_t(sample_params_.id)),
          reinterpret_cast<nn::Candidate*>(uintptr_t(cands)),
          reinterpret_cast<int*>(uintptr_t(count)));
      if (ok) ok = device::device_copy(count, out_handle, sizeof(int32_t));
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
      nn::cuda::attention_trace_set_layer(int(layer));  // no-op when tracing off
      ok = nn::cuda::flash_attention_chunk_dev_pos(
          dev(in[0]), dev(in[1]), dev(in[2]), dev_mut(cache.k_buf.id),
          dev_mut(cache.v_buf.id), reinterpret_cast<const uint64_t*>(pos_slot()),
          cache.n_slots, int(nb), int(lc.n_heads_q), int(cache.n_heads_kv),
          int(cache.head_dim), lc.sliding_window, float(attn_scale), out,
          cache.f16(), cache.q8());
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
              dev(in[1]), reinterpret_cast<const uint64_t*>(pos_slot()),
              cache.n_slots, row_elems, int(nb), cache.f16(), cache.q8(),
              int(cache.head_dim))) {
        ok = false;
        break;
      }
      // The append is a cache side effect; the dummy scalar output is never read,
      // so it is left unwritten (a host write to the device handle would fault).
      break;
    }
    default:
      // An op this backend has no device kernel for: the placement pass routed
      // a host-only op to the device.
      ok = false;
      break;
  }
  if (nvtx_enabled()) nvtxRangePop();

  if (!ok) {
    if (extra_handle) device::release(extra_handle, extra_bytes);
    for (const auto& e : sibling_outs) device::release(e.first, e.second);
    // A borrowed buffer belongs to the input value; the executor still owns it.
    if (!borrowed) device::release(out_handle, out_bytes);
    return std::unexpected(fail("Sonic CUDA kernel failed for node '" + node->name +
                                "'"));
  }

  // A side-effect-only node (KvAppend) has an orphaned dummy output that
  // the planner drops from task.outputs, so return zero outputs to match the
  // task's declared output count. The output buffer was allocated but is never
  // consumed; release it rather than leak it.
  if (task.outputs.empty()) {
    if (!borrowed) device::release(out_handle, out_bytes);
    for (const auto& e : sibling_outs) device::release(e.first, e.second);
    return std::vector<TensorValue>{};
  }

  std::vector<TensorValue> res;
  res.push_back(TensorValue::from_device(out_handle, out_bytes));
  if (extra_handle)
    res.push_back(TensorValue::from_device(extra_handle, extra_bytes));
  for (const auto& e : sibling_outs)
    res.push_back(TensorValue::from_device(e.first, e.second));
  return res;
}

} // namespace sonicboom::planner
