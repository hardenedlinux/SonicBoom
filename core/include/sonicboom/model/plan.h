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

#pragma once

// Gemma 4 plan emitter (Phase 6). Expands the single-token decode forward pass
// into a static, acyclic planner::Graph of transformer OpKind nodes, so model
// inference runs on the shared spine (planner + SonicBackend) rather than the
// bespoke loop in exec.cpp. Weights are NOT graph tensors: the emitter bakes a
// (layer, slot) attribute pair onto each node and the SonicBackend reads the
// bound weights straight from the model. This keeps the ~500 MB of quantized
// weights out of the planner's tensor/buffer model (which is sized for host
// float32 activations).

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>

#include <sonicboom/model/loader.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/planner/plan_record.h>

namespace sonicboom::model {

// Names the specific Gemma 4 weight tensor (or None) a decode-graph node reads.
// The emitter bakes this as an Int attribute; the SonicBackend maps (slot,
// layer) back to the bound weight. Block slots (AttnNorm .. LayerOutputScale)
// index `weights.blocks[layer]`; the rest index `weights.<field>` directly.
enum class WeightSlot : uint8_t {
  None = 0,
  TokenEmbd,
  PerLayerTokenEmbd,
  PerLayerModelProj,
  PerLayerProjNorm,
  OutputNorm,
  RopeFreqs,
  AttnNorm,
  FfnNorm,
  AttnQNorm,
  AttnKNorm,
  PostAttentionNorm,
  PostFfwNorm,
  PostNorm,
  AttnQ,
  AttnK,
  AttnV,
  AttnOutput,
  FfnGate,
  FfnUp,
  FfnDown,
  InpGate,
  Proj,
  LayerOutputScale,
  // Optional linear-layer biases, f32 [width]. An architecture either carries
  // them (Qwen2's Q/K/V) or does not (Gemma 4); the loader binds what the file
  // has and the emitter emits an AddBias node only for a bound span, so an
  // absent bias costs nothing. Added after the original slots so a slot's
  // numeric value never moves.
  AttnQBias,
  AttnKBias,
  AttnVBias,
  AttnOutputBias,
  // The untied output projection (lm head); empty when the model ties its
  // embedding and output weights, in which case the emitter reuses TokenEmbd.
  Output,
};

// Expand the 42-block Gemma 4 single-token decode (mirrors forward()/lm_head in
// exec.cpp, whose arithmetic is bit-exact against the llama.cpp baseline) into a
// static, acyclic planner::Graph. Every node is pinned to BackendTag::Sonic
// (including the shared `Add` residual nodes, which route_op would otherwise
// send to Mlir). Graph inputs are `token_id` and `pos` (Int64). The graph output
// depends on `greedy`: greedy=true (CUDA spine) emits an `Argmax` node producing
// a single float32 token index, deleting the Softcap (monotonic, so it does not
// affect argmax) and the 1 MiB logits download; greedy=false (CPU oracle spine)
// emits the softcapped logits (vocab float32), leaving argmax to the caller's
// generate loop so the oracle arithmetic stays bit-identical.
planner::Graph build_gemma4_graph(const Gemma4Model& m, bool greedy);

// Build the model-agnostic KV-cache geometry spec from the Gemma 4 config
// (Section C de-Gemma decoupling). Keeps the Gemma-specific per-layer mapping
// (layer_config / kv_donor_layer) inside the model layer; the CUDA backend only
// ever sees the resulting planner::KvCacheSpec, never Gemma4Config.
planner::KvCacheSpec kv_cache_spec(const Gemma4Config& cfg);

// Expand the whole-prompt prefill (mirrors build_gemma4_graph per block) into a
// static, acyclic planner::Graph over [n, dim] position-major batched tensors,
// where n is the prompt length and dim is innermost. Every node is pinned to
// BackendTag::Sonic; the per-token OpKinds carry a `batch` attribute and are
// dispatched by the SonicBackend looping the same per-token kernels the decode
// graph uses, so the prefill is bit-identical to n single-token decode steps.
// Graph inputs are `tokens` (Int64 [n]) and `start_pos` (Int64); the graph
// output is the final residual [n, dim] (the caller extracts row n-1 and runs
// lm_head).
planner::Graph build_gemma4_prefill_graph(const Gemma4Model& m, uint64_t n);

// The same graph with the token dimension emitted as the Boom dynamic extent
// (`?`) and bound to `n` at lowering: every tensor that carries the segment
// length is marked with TensorDesc::dynamic_stride_bytes, so one plan can
// execute a smaller segment by binding the actual length. `n` is the extent
// the plan is planned and its buffers sized at; run no larger than it.
planner::Graph build_gemma4_prefill_graph(const Gemma4Model& m, uint64_t n,
                                          bool symbolic);

// Autoregressive greedy decode through the spine: build the graph once, plan it
// once (StaticPlanner + MemoryPlanner + TransferScheduler + PlanValidator), then
// run each step through the RuntimeExecutor with a SonicBackend (CpuQ8K, the
// bit-exact oracle backend). The decode is multi-token: the backend holds a
// per-KV-layer cache and each step attends over the accumulated keys (the same
// KV-cache semantics as llama.cpp's incremental decode). Same token-stream
// contract as generate(): writes each step's sampled token id into `tokens` and
// returns the number of steps actually run (stops early on model-empty,
// out-of-vocab seed, or a plan/execute failure).
//
// `n_ctx` sizes the KV cache (global layers allocate `n_ctx` slots; sliding-window
// layers a fixed `sliding_window` ring). 0 defaults to `start_pos + tokens.size()`
// (the exact decode length). The cache is cleared at session start.
uint64_t generate_spine(const Gemma4Model& m, uint64_t start_token,
                        uint64_t start_pos, std::span<int64_t> tokens,
                        uint64_t n_ctx = 0);

// CUDA spine: the same greedy decode as generate_spine, but planned against a
// CUDA-aware snapshot so Sonic nodes with *_dev kernels run on the GPU and only
// Embedding/Softcap stay on the CPU (mirrors cuda_forward_resident's placement).
// The RuntimeExecutor runs both the SonicCudaBackend (device-resident) and the
// CPU SonicBackend (host boundary) with explicit H2D/D2H transfer tasks between
// them. Returns the number of steps actually run; 0 when the build has no CUDA
// support (SONICBOOM_USE_CUDA off), so callers gate on quant::cuda_available().
uint64_t generate_spine_cuda(const Gemma4Model& m, uint64_t start_token,
                             uint64_t start_pos, std::span<int64_t> tokens,
                             uint64_t n_ctx = 0);

// Pay the one-time decode-runtime construction for (model, n_ctx) up front.
//
// The first CUDA prefill call for an n_ctx builds the decode graph, plans it,
// constructs the backends, and captures one decode step — ~1 s on Gemma 4 E4B,
// of which the capture (warm-up + record) is most. That cost belongs to
// load/serve setup, not to the first request: a caller that knows its session's
// n_ctx (the CLI, the server, sb_llm_session_new) can pay it here and keep the
// first token's time-to-first-token at the steady-state value. The runtime is
// exactly the one the first generate call would build and is cached under the
// same (model, n_ctx) key, so a later call is a map hit. Returns false when the
// build has no CUDA support or no weights; the caller may ignore the result and
// fall back to the lazy path.
bool warmup_spine_cuda(const Gemma4Model& m, uint64_t n_ctx);

// ---------------------------------------------------------------------------
// Speculative verification (Part B). A draft of k tokens is checked by running
// the model over the k+1 query positions 0..k at once and taking each row's
// argmax. The verify step is the decode graph dealt over rows (the verify phase
// of BoomPasses): every op runs its decode kernel per row, except the quantized
// matmul, which runs the row-invariant batched mmvq — so row i is bit-identical
// to a plain decode step at pos+i, and the KV cache after the step holds exactly
// what k+1 chained decode steps would have written. That bit-identity is the
// lossless contract: the accepted tokens are the same tokens greedy decoding
// would have produced, not close to them.
// ---------------------------------------------------------------------------

// Allocate `session` a device KV cache of the model's geometry at decode length
// `n_ctx`, so two sessions on the same model/runtime have independent caches
// (a session left empty adopts the decode runtime's cache on its first prefill).
// Clears the session's filled-token history. Returns false without CUDA.
bool kv_session_allocate(const Gemma4Model& m, planner::KvCacheSession& session,
                         uint64_t n_ctx);

// The verify step: feed `tokens` at positions `start_pos .. start_pos+rows-1`
// through the row-invariant verify graph on `session`'s cache and write each
// row's argmax into `out` (out.size() >= tokens.size(), rows <=
// planner::kMaxVerifyRows, start_pos + rows <= session.n_ctx). The cache is left
// holding those rows, as `rows` plain decode steps would leave it. Returns false
// on a build/step failure. The first call for a (model, n_ctx, rows, cache)
// builds, plans and captures the step; later calls replay it.
bool verify_step_spine_cuda(const Gemma4Model& m,
                            std::span<const uint64_t> tokens,
                            uint64_t start_pos,
                            planner::KvCacheSession& session,
                            std::span<int64_t> out);

// The plain-decode reference for the verify step: feed `tokens[i]` at
// `start_pos + i` through the captured single-token decode step on `session`'s
// cache, without chaining each step's argmax into the next, and write each
// step's argmax into `out`. Used to test that verify_step_spine_cuda is
// bit-identical to k+1 decode steps. Returns false without a captured decode
// step bound to `session`'s cache (a session whose cache the decode runtime did
// not publish or adopt first).
bool decode_steps_spine_cuda(const Gemma4Model& m,
                             std::span<const uint64_t> tokens,
                             uint64_t start_pos,
                             planner::KvCacheSession& session,
                             std::span<int64_t> out);

// B4: one speculative step — draft, verify, accept, and roll the session back to
// the accepted run. `last_token` is the token the previous step sampled (it sits
// at position L = session.filled_tokens.size(), not yet in the cache); `drafts`
// are the drafter's proposals for positions L+1 .. L+k. The step feeds
// [last_token, drafts...] through the verify graph at positions L .. L+k and
// accepts the longest prefix a of `drafts` whose i-th entry equals row i's
// argmax (row i's context ends at drafts[i-1], so its argmax is exactly the
// plain-decode greedy token at position L+i+1). The verifier's own next token,
// row a's argmax, follows the accepted drafts. `out` receives accepted+1 tokens
// (out.size() >= drafts.size()+1) and *accepted_out the count a.
//
// The verify step wrote KV for all k+1 rows, but only positions L .. L+a belong
// to the accepted run: rows L+a+1 .. L+k were computed with rejected draft
// tokens. Those entries are left in place (they are past the valid length) —
// no later attention extent reaches them, and the next steps overwrite them in
// order. The session's token history records exactly the accepted run
// (last_token and drafts[0..a-1]), so prefix reuse after a speculative session
// matches only KV that belongs to accepted tokens.
bool spec_step_spine_cuda(const Gemma4Model& m, uint64_t last_token,
                          std::span<const uint64_t> drafts,
                          planner::KvCacheSession& session,
                          std::span<int64_t> out, uint64_t* accepted_out);

// Test seam (B4): copy one KV layer's K (is_k) or V rows for absolute positions
// [first, first+count) into `out` (count * n_heads_kv * head_dim floats). The
// speculative rollback test compares a verify step's valid KV region bit-for-bit
// against plain decode; nothing in the runtime reads a cache back to the host.
bool kv_read_rows_cuda(const planner::KvCacheSession& session, uint32_t layer,
                       bool is_k, uint64_t first, uint64_t count,
                       std::span<float> out);

// B6: the accept/emit decision of one speculative iteration, factored out of
// the loop so its truncation rules are testable without a model or a device.
//   accepted = the longest prefix with row_argmax[i] == drafts[i]
//   out      = drafts[0..accepted-1], then row_argmax[accepted], cut after the
//              first stop token (the stop token is emitted, exactly as the plain
//              loop emits it before breaking) or once emitted_before + emitted
//              reaches `limit`.
// `row_argmax` and `out` must have room for drafts.size() + 1 entries.
struct SpecAccept {
  uint64_t accepted = 0;  // drafts the verifier confirmed
  uint64_t emitted = 0;   // run tokens to emit (drafts + the verifier token)
  bool stopped = false;   // a stop token ended the run
};
SpecAccept spec_accept_run(std::span<const uint64_t> drafts,
                           std::span<const int64_t> row_argmax,
                           std::span<const int64_t> eos_tokens,
                           uint64_t emitted_before, uint64_t limit,
                           std::span<uint64_t> out);

// B6/B8 diagnostic counters for the speculative loop. Process-wide (the decode
// path is single-threaded); reset them before a run whose α — accepted_tokens /
// draft_tokens — is being reported.
struct SpecCounters {
  uint64_t iterations = 0;       // loop iterations
  uint64_t plain_steps = 0;      // iterations the drafter proposed nothing
  uint64_t verify_steps = 0;     // iterations that ran a verify (k > 0)
  uint64_t draft_tokens = 0;     // tokens proposed and verified
  uint64_t accepted_tokens = 0;  // drafts the verifier accepted
  uint64_t emitted_tokens = 0;   // tokens emitted
  // SONICBOOM_SPEC=auto, S2 bootstrap: iterations after the cost warm-up that
  // forced k = 1 exploration (so the acceptance-rate EMA sees real drafts
  // before the cost model starts choosing), and iterations afterwards that the
  // policy chose. S3 ends the bootstrap early after
  // SONICBOOM_SPEC_BOOTSTRAP_ABORT consecutive zero-accept iterations.
  // Warm-up iterations are neither.
  uint64_t bootstrap_steps = 0;
  uint64_t policy_steps = 0;
};
SpecCounters spec_counters();
void spec_counters_reset();

// B7: the adaptive draft-length policy, factored out for testing.
//
// With acceptance rate alpha, k drafts yield E(k) = 1 + alpha + ... + alpha^k =
// (1 - alpha^(k+1)) / (1 - alpha) emitted tokens (a geometric run); c(k) is one
// iteration's measured cost in microseconds (cost_us[k] == 0 means unmeasured,
// and k is not considered). Returns the k <= max_k maximizing E(k) / c(k), or
// max_k when nothing is measured (the fixed cap, the static default).
int64_t spec_choose_drafts(double alpha, std::span<const double> cost_us,
                           int64_t max_k);

// B7: the acceptance-rate EMA update, alpha <- 0.5*alpha + 0.5*(accepted/drafted).
// A run starts at alpha = 0.5 (the policy's initial guess).
double spec_alpha_update(double alpha, uint64_t accepted, uint64_t drafted);

// Test seam (B6): run the speculative loop over `tokens` on the model's decode
// runtime with a chosen drafter strategy, bypassing SONICBOOM_SPEC's program.
// "ngram" is the runtime's prompt-lookup drafter; "repeat" proposes the last
// accepted token `max_drafts` times, so every iteration runs a verify — the
// deterministic way for the B6 test to exercise the verify/accept/emit path.
// Because the drafter only proposes and the verifier decides, the token stream
// must still be the plain greedy stream. Returns the number of tokens written.
uint64_t spec_generate_spine_cuda(const Gemma4Model& m, uint64_t start_token,
                                  uint64_t start_pos, std::span<int64_t> tokens,
                                  uint64_t n_ctx, const char* draft_strategy);

// Prefill + decode through the CPU spine. Processes the whole `prompt` in one
// batched graph pass (build_gemma4_prefill_graph → plan_execution →
// RuntimeExecutor with the batched OpKinds), samples `tokens[0]` from the final
// hidden column's lm-head logits, then decodes the remaining tokens with the
// same planner/backend/cache as generate_spine. `use_prefill` is a dev-test
// knob: false feeds the prompt one token at a time through the per-token spine
// (the pre-Phase-7 reference), so tests can assert the two paths produce the
// identical token stream.
//
// `prompt` must be non-empty. `n_ctx` sizes the KV cache (0 defaults to
// `start_pos + prompt.size() + tokens.size()`); the prompt must fit each KV
// layer without ring wrap (n <= sliding_window for SWA layers, start_pos + n <=
// n_ctx for global layers). Returns the number of tokens written.
uint64_t generate_prefill_spine(const Gemma4Model& m,
                                std::span<const uint64_t> prompt,
                                uint64_t start_pos, std::span<int64_t> tokens,
                                uint64_t n_ctx = 0, bool use_prefill = true);

// CUDA prefill + decode: the same prompt-prefill contract as
// generate_prefill_spine, but on the device — SonicCudaBackend::prefill (batched
// f32-activation matmul + flash attention) fills the device KV caches, lm_head
// samples tokens[0] on the device, and the remaining tokens decode through the
// CUDA spine (generate_spine_cuda's per-token loop over the same backend/cache).
// `use_prefill` is a dev-test knob: false feeds the prompt one token at a time
// through the CUDA spine (the per-token reference), so tests can assert the two
// paths produce the identical token stream. Returns 0 when the build has no CUDA
// support (SONICBOOM_USE_CUDA off), so callers gate on quant::cuda_available().
uint64_t generate_prefill_spine_cuda(const Gemma4Model& m,
                                     std::span<const uint64_t> prompt,
                                     uint64_t start_pos, std::span<int64_t> tokens,
                                     uint64_t n_ctx = 0, bool use_prefill = true);

// Teacher-forced scoring on the CUDA decode spine (test seam for the
// numerics-profile acceptance gate, tools/sb-numerics). Every token is fed at
// its own position through the non-greedy decode graph, whose raw [vocab]
// logits reach the host. The model's logit softcap is then applied to them on
// the host (models without one, e.g. Qwen2, score the raw logits), so the
// scored distribution is the model's own — the one HF and llama.cpp also fold
// into their perplexity — not a sharper uncapped one. `sum_nll` accumulates the
// negative log-likelihood of
// each next token under those logits and `n_scored` is the number of scored
// positions (tokens.size() - 1 on success). `n_ctx` sizes the KV cache and is
// raised to tokens.size() when smaller. Returns false without CUDA support, on
// a runtime/execute failure, or for an out-of-vocabulary token id. This is
// measurement only: it runs a throwaway runtime, not the cached greedy one, so
// it never disturbs a session.
bool score_spine_cuda(const Gemma4Model& m, std::span<const uint64_t> tokens,
                      uint64_t n_ctx, double& sum_nll, uint64_t& n_scored);

// CUDA prefill + decode with prefix-cache reuse: the same token-stream contract
// as generate_prefill_spine_cuda, but the KV cache persists in `session` across
// calls. On each call it matches `prompt` against the session's token history
// and reuses the shared prefix's KV slots, prefilling only the new suffix
// (`prompt[prefix_len:]`). Prefix reuse requires `start_pos == 0` (the cache is
// a contiguous history anchored at position 0); a non-zero start_pos disables
// reuse and prefills the whole prompt. The cache size is `session.n_ctx`: set it
// (once, before the first call) to the largest `start_pos + prompt.size() +
// tokens.size()` the session will serve; a zero `session.n_ctx` sizes it from
// the first call's own prompt. A request larger than `session.n_ctx` is rejected
// (returns 0). The capability is generic — it knows nothing about Gemma's layer
// structure, only the session's token history and the backend-owned slot
// geometry.
//
// A prompt that the session already holds in full has no new suffix to prefill,
// and prefilling nothing yields no logits. The last token is therefore always
// processed: the reuse is capped at prompt.size()-1 and that token is prefilled
// as a one-token suffix, which produces exactly the logits a full prefill would.
// `n_reused_out` (optional) reports the prefix actually reused, so the count is
// prompt.size()-1 in that case and the matched prefix length otherwise.
uint64_t generate_prefill_spine_cuda_prefix(const Gemma4Model& m,
                                            std::span<const uint64_t> prompt,
                                            uint64_t start_pos,
                                            std::span<int64_t> tokens,
                                            planner::KvCacheSession& session,
                                            bool use_prefill = true,
                                            uint64_t* n_reused_out = nullptr);

// Build the model's plan and record what executing it does, step by step.
//
// These are the producers behind tools/plan_dump: given a loaded model and a
// mode, they return a deterministic, serializable description of the executed
// plan (see planner/plan_record.h). They deliberately build a *non-captured*
// runtime: a CUDA-graph-captured decode step replays as a single graph launch
// and never walks the executor, so nothing per-step would be observable.
//
// `record_decode_plan` runs `steps` greedy decode steps at positions 0.. steps-1
// (position == step; only the shapes matter, not the tokens). `record_prefill_plan`
// runs one batched pass over `n` tokens. Both return nullopt when the build has no
// CUDA support, the model carries no weights, or the plan fails to build.
//
// `transform`, when given, is applied to the graph the emitter built — after the
// graph-level fusion passes, before the execution plan is derived from it. It is
// the seam for putting the planned graph through another representation and back
// (the Boom IR round trip, docs/boom-ir.md) and requiring the plan
// that follows to be the same plan; it is not a place to change what runs.
//
// The overloads are overloads rather than one function with a defaulted
// parameter because libsonicboom.so is consumed through its symbol table
// (tools/plan_dump): a defaulted parameter would silently drop the existing
// entry point.
using GraphTransform = std::function<void(planner::Graph&)>;

std::optional<planner::PlanRunRecord> record_decode_plan(const Gemma4Model& m,
                                                         uint64_t n_ctx,
                                                         uint64_t steps);
std::optional<planner::PlanRunRecord> record_decode_plan(
    const Gemma4Model& m, uint64_t n_ctx, uint64_t steps,
    const GraphTransform& transform);
std::optional<planner::PlanRunRecord> record_prefill_plan(const Gemma4Model& m,
                                                          uint64_t n);
std::optional<planner::PlanRunRecord> record_prefill_plan(
    const Gemma4Model& m, uint64_t n, const GraphTransform& transform);

// The device identity string the plan record's `device` line carries, e.g.
// "NVIDIA GeForce RTX 3050 sm 20 cc 8.6"; "unknown" when the build has no CUDA
// support or no device is present. Exposed because part of the device path is
// device-dependent (the decode-attention split-KV grid is sized from the SM
// count), so anything that pins a device-produced artifact — a plan record, a
// token-stream golden — has to be able to name the device it was taken on.
std::string cuda_device_identity();

} // namespace sonicboom::model
