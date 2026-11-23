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

#include "expert_pool.h"

#include "device_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace sonicboom::planner {

namespace {

// The kernels' zeroed read-ahead tail; a slot's stride is a multiple of it so
// every slot start is too.
constexpr uint64_t kSlotAlignment = 256;

uint64_t align_up(uint64_t n) {
  const uint64_t rem = n % kSlotAlignment;
  return rem == 0 ? n : n + (kSlotAlignment - rem);
}

// The expert views of one block are the three stacked tensors split per
// expert; they must agree on the expert count and each be a usable quantized
// matrix.
uint32_t block_experts(const model::BlockWeights& block, uint32_t layer) {
  const size_t n = block.expert_gate.size();
  if (n == 0 || block.expert_up.size() != n || block.expert_down.size() != n)
    return 0;
  for (size_t e = 0; e < n; ++e) {
    if (!block.expert_gate[e].valid() || !block.expert_up[e].valid() ||
        !block.expert_down[e].valid())
      return 0;
  }
  (void)layer;
  return n > UINT32_MAX ? 0 : uint32_t(n);
}

uint64_t span_bytes(std::span<const float> s) {
  return uint64_t(s.size()) * sizeof(float);
}

uint64_t span_bytes(std::span<const uint16_t> s) {
  return uint64_t(s.size()) * sizeof(uint16_t);
}

uint64_t tensor_bytes(const quant::QuantizedTensor& t) {
  return t.valid() ? t.byte_size() : 0;
}

// Bytes of every weight the model keeps outside the routed experts. The
// device copy of a non-expert weight is paid lazily on top of the pool, so it
// is reserved up front (design §9).
uint64_t non_expert_bytes(const model::Gemma4Weights& weights) {
  uint64_t n = tensor_bytes(weights.token_embd) +
               tensor_bytes(weights.per_layer_token_embd) +
               tensor_bytes(weights.output) +
               span_bytes(weights.per_layer_proj_norm) +
               span_bytes(weights.per_layer_model_proj) +
               span_bytes(weights.output_norm) +
               span_bytes(weights.rope_freqs);
  for (const model::BlockWeights& b : weights.blocks) {
    n += span_bytes(b.attn_norm) + span_bytes(b.ffn_norm) +
         span_bytes(b.attn_q_norm) + span_bytes(b.attn_k_norm) +
         span_bytes(b.post_attention_norm) + span_bytes(b.post_ffw_norm) +
         span_bytes(b.post_norm) + span_bytes(b.attn_q_bias) +
         span_bytes(b.attn_k_bias) + span_bytes(b.attn_v_bias) +
         span_bytes(b.attn_output_bias) + span_bytes(b.inp_gate) +
         span_bytes(b.proj) + span_bytes(b.layer_output_scale) +
         span_bytes(b.expert_router_f32) + span_bytes(b.shared_gate_in) +
         span_bytes(b.gdn_qkv_f32) + span_bytes(b.gdn_gate_f32) +
         span_bytes(b.gdn_beta_f32) + span_bytes(b.gdn_alpha_f32) +
         span_bytes(b.gdn_out_f32) + span_bytes(b.gdn_conv) +
         span_bytes(b.gdn_dt) + span_bytes(b.gdn_a) + span_bytes(b.gdn_norm);
    n += tensor_bytes(b.attn_q) + tensor_bytes(b.attn_k) +
         tensor_bytes(b.attn_v) + tensor_bytes(b.attn_output) +
         tensor_bytes(b.ffn_gate) + tensor_bytes(b.ffn_up) +
         tensor_bytes(b.ffn_down) + tensor_bytes(b.expert_router) +
         tensor_bytes(b.shared_gate) + tensor_bytes(b.shared_up) +
         tensor_bytes(b.shared_down) + tensor_bytes(b.gdn_qkv) +
         tensor_bytes(b.gdn_gate) + tensor_bytes(b.gdn_beta) +
         tensor_bytes(b.gdn_alpha) + tensor_bytes(b.gdn_out);
  }
  return n;
}

bool stats_enabled() {
  const char* env = std::getenv("SONICBOOM_EXPERT_STATS");
  return env && *env && std::strcmp(env, "0") != 0;
}

// SONICBOOM_MOE_PREFETCH=on gives the pool a second staging pipeline and makes
// the backend upload predicted experts ahead of their selection node. Off
// unless set; "off"/"0" is off.
bool prefetch_on() {
  const char* env = std::getenv("SONICBOOM_MOE_PREFETCH");
  if (!env || !*env) return false;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0;
}

// The routing trace (SONICBOOM_EXPERT_TRACE=path): one line per selection
// node, "block entries id...", appended in execution order. An offline
// simulator replays it; the runtime never reads the file. Off unless set.
std::FILE* trace_file() {
  static std::FILE* f = [] {
    const char* path = std::getenv("SONICBOOM_EXPERT_TRACE");
    if (!path || !*path) return static_cast<std::FILE*>(nullptr);
    return std::fopen(path, "w");
  }();
  return f;
}

// How many selection nodes a use count survives before it halves. A decode
// step is one node per block; 1280 nodes is 32 steps at 40 blocks. This is a
// deterministic constant, not a timing measurement: the offline simulator
// priced the periods (512..8192) over the recorded traces at the pool sizes
// this card reaches. 1280 keeps the deep-context trace level with LRU while
// the prompt-heavy traces gain 2-4 hit points; a longer period wins on short
// generations and loses on a long prefix, whose prefill churn then stays
// counted for too many steps.
constexpr uint64_t kAgePeriod = 1280;

// One process-wide pool, keyed by the weights object it belongs to.
struct Registry {
  std::mutex mutex;
  const model::Gemma4Weights* key = nullptr;
  std::unique_ptr<ExpertPool> pool;
};

Registry& registry() {
  static Registry r;
  return r;
}

// The CUDA backend's wall-clock accumulators; only touched under the stats
// flag, so a normal run pays nothing.
struct TimingRegistry {
  std::mutex mutex;
  ExpertTiming timing;
};

TimingRegistry& timing_registry() {
  static TimingRegistry r;
  return r;
}

void print_stats_at_exit() {
  Registry& r = registry();
  ExpertTiming t;
  {
    TimingRegistry& tr = timing_registry();
    std::lock_guard<std::mutex> lk(tr.mutex);
    t = tr.timing;
  }
  std::lock_guard<std::mutex> lk(r.mutex);
  if (!r.pool) return;
  const ExpertPool::Stats s = r.pool->stats();
  std::fprintf(stderr,
               "expert pool: nodes=%llu hits=%llu misses=%llu uploaded=%llu "
               "bytes\n",
               static_cast<unsigned long long>(s.nodes),
               static_cast<unsigned long long>(s.hits),
               static_cast<unsigned long long>(s.misses),
               static_cast<unsigned long long>(s.bytes));
  if (s.cpu_misses != 0)
    std::fprintf(stderr, "expert cpu misses: %llu entries (no bytes moved)\n",
                 static_cast<unsigned long long>(s.cpu_misses));
  if (s.prefetch_nodes != 0 || s.prefetch_uploads != 0)
    std::fprintf(stderr,
                 "expert prefetch: calls=%llu uploads=%llu bytes=%llu "
                 "hits=%llu waste=%llu bytes\n",
                 static_cast<unsigned long long>(s.prefetch_nodes),
                 static_cast<unsigned long long>(s.prefetch_uploads),
                 static_cast<unsigned long long>(s.prefetch_bytes),
                 static_cast<unsigned long long>(s.prefetch_hits),
                 static_cast<unsigned long long>(s.prefetch_waste));
  std::fprintf(stderr,
               "expert stats: select=%.6f s experts=%.6f s route=%.6f s\n",
               t.select_s, t.experts_s, t.route_s);
}

} // namespace

// The handles are device addresses the runtime spells as u64 (the same
// spelling the HAL's buffer ids use); the accessors expose one as the opaque
// pointer the planner API carries.
const void* ExpertPool::data(uint32_t role) const noexcept {
  if (role > kDown) return nullptr;
  return reinterpret_cast<const void*>(handles_[role]);  // NOLINT(performance-no-int-to-ptr)
}

const void* ExpertPool::selection() const noexcept {
  return reinterpret_cast<const void*>(sel_handle_);  // NOLINT(performance-no-int-to-ptr)
}

ExpertPool::Stats ExpertPool::stats() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return stats_;
}

ExpertPool::~ExpertPool() {
  // Drain the copy streams before the arrays go back to the pool: a staged copy
  // may still be in flight.
  if (prefetch_staging_) device::staging_destroy(prefetch_staging_);
  prefetch_staging_ = nullptr;
  if (staging_) device::staging_destroy(staging_);
  staging_ = nullptr;
  for (uint64_t& h : handles_) {
    if (h) device::release(h, 0);
    h = 0;
  }
  if (sel_handle_) device::release(sel_handle_, 0);
  sel_handle_ = 0;
}

std::expected<std::unique_ptr<ExpertPool>, std::string> ExpertPool::create(
    const model::Gemma4Weights& weights) {
  if (!device::available())
    return std::unexpected(std::string("expert pool: no CUDA device"));
  auto pool = std::unique_ptr<ExpertPool>(new ExpertPool());

  // The block/expert tables and the largest matrix of each role. blocks_ is
  // indexed by the node's layer, so a dense block keeps an empty entry and a
  // node that names it is rejected.
  pool->blocks_.resize(weights.blocks.size());
  uint32_t max_experts = 0;
  uint32_t key_base = 0;
  for (uint32_t layer = 0; layer < weights.blocks.size(); ++layer) {
    const model::BlockWeights& b = weights.blocks[layer];
    pool->blocks_[layer].key_base = key_base;
    const uint32_t n = block_experts(b, layer);
    if (n == 0) continue;  // a dense block has no routed experts
    pool->blocks_[layer].experts = n;
    for (uint32_t e = 0; e < n; ++e) {
      const quant::QuantizedTensor* g = &b.expert_gate[e];
      const quant::QuantizedTensor* u = &b.expert_up[e];
      const quant::QuantizedTensor* d = &b.expert_down[e];
      pool->gate_of_.push_back(g);
      pool->up_of_.push_back(u);
      pool->down_of_.push_back(d);
      pool->stride_[kGate] =
          std::max(pool->stride_[kGate], align_up(g->byte_size()));
      pool->stride_[kUp] =
          std::max(pool->stride_[kUp], align_up(u->byte_size()));
      pool->stride_[kDown] =
          std::max(pool->stride_[kDown], align_up(d->byte_size()));
    }
    key_base += n;
    max_experts = std::max(max_experts, n);
  }
  if (key_base == 0 ||
      pool->stride_[kGate] == 0 || pool->stride_[kUp] == 0 ||
      pool->stride_[kDown] == 0)
    return std::unexpected(
        std::string("expert pool: the model has no routed expert weights"));
  pool->total_keys_ = key_base;

  // Pool bytes: the environment override, otherwise what the card can spare
  // beyond the non-expert weights and 1024 MB.
  const uint64_t slot_bytes =
      pool->stride_[kGate] + pool->stride_[kUp] + pool->stride_[kDown];
  uint64_t pool_bytes = 0;
  const char* mb_env = std::getenv("SONICBOOM_EXPERT_POOL_MB");
  if (mb_env && *mb_env) {
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(mb_env, &end, 10);
    if (end == mb_env || *end != '\0')
      return std::unexpected(std::string("expert pool: SONICBOOM_EXPERT_POOL_MB "
                                         "is not a byte count in MiB: ") +
                             mb_env);
    pool_bytes = uint64_t(mb) * (1024ull * 1024ull);
  } else {
    uint64_t free_mem = 0;
    if (!device::free_bytes(&free_mem))
      return std::unexpected(
          std::string("expert pool: cannot read the free device memory"));
    const uint64_t reserved = 1024ull * 1024ull * 1024ull;
    const uint64_t taken = non_expert_bytes(weights);
    pool_bytes = free_mem > reserved + taken ? free_mem - reserved - taken : 0;
    // The default must cover every expert one block can select at once: when
    // the free memory beyond the margin does not reach that floor, the floor
    // comes out of the margin. What one block needs is not a preference.
    pool_bytes = std::max(pool_bytes, slot_bytes * max_experts);
  }

  pool->n_slots_ = pool_bytes / slot_bytes;
  if (pool->n_slots_ < max_experts) {
    const uint64_t need = slot_bytes * max_experts;
    const uint64_t missing = need > pool_bytes ? need - pool_bytes : 0;
    char msg[512];
    std::snprintf(
        msg, sizeof msg,
        "expert pool: %llu bytes hold %llu slots of %llu bytes, fewer than "
        "the %u experts one block may select at once; %llu more bytes are "
        "needed (SONICBOOM_EXPERT_POOL_MB)",
        static_cast<unsigned long long>(pool_bytes),
        static_cast<unsigned long long>(pool->n_slots_),
        static_cast<unsigned long long>(slot_bytes), max_experts,
        static_cast<unsigned long long>(missing));
    return std::unexpected(std::string(msg));
  }

  pool->owner_.assign(pool->n_slots_, -1);
  pool->stamp_.assign(pool->n_slots_, 0);
  pool->slot_of_.assign(pool->total_keys_, -1);
  pool->uses_.assign(pool->total_keys_, 0);
  pool->prefetched_.assign(pool->total_keys_, 0);
  pool->handles_[kGate] = device::acquire(pool->n_slots_ * pool->stride_[kGate]);
  pool->handles_[kUp] = device::acquire(pool->n_slots_ * pool->stride_[kUp]);
  pool->handles_[kDown] =
      device::acquire(pool->n_slots_ * pool->stride_[kDown]);
  if (!pool->handles_[kGate] || !pool->handles_[kUp] || !pool->handles_[kDown])
    return std::unexpected(std::string("expert pool: cannot allocate ") +
                           std::to_string(pool_bytes) + " bytes on the device");

  // The arrays start zeroed once; select() then only has to keep the tail of a
  // slot zero, where a smaller matrix follows a larger one.
  for (uint32_t role = 0; role < 3; ++role) {
    if (!device::memset_device(pool->handles_[role], 0,
                               pool->n_slots_ * pool->stride_[role]))
      return std::unexpected(
          std::string("expert pool: cannot zero the pool arrays"));
  }

  // The pinned staging pipeline. A failed creation is not fatal: select()
  // falls back to the synchronous pageable copy.
  uint64_t stage_bytes = 16ull << 20;
  if (const char* kb_env = std::getenv("SONICBOOM_EXPERT_STAGE_KB");
      kb_env && *kb_env) {
    char* end = nullptr;
    const unsigned long long kb = std::strtoull(kb_env, &end, 10);
    if (end == kb_env || *end != '\0' || kb == 0)
      return std::unexpected(
          std::string("expert pool: SONICBOOM_EXPERT_STAGE_KB is not a "
                      "positive size in KiB: ") +
          kb_env);
    stage_bytes = uint64_t(kb) * 1024ull;
  }
  pool->staging_ = device::staging_create(stage_bytes);
  if (!pool->staging_ && stats_enabled())
    std::fprintf(stderr,
                 "expert pool: pinned staging unavailable, uploads stay "
                 "synchronous\n");

  // The prefetch pipeline (SONICBOOM_MOE_PREFETCH). Sized to hold one block's
  // predicted set in a single half; its copies run ahead of the selection node
  // that consumes them.
  if (prefetch_on()) {
    uint64_t pf_bytes = 64ull << 20;
    if (const char* kb_env = std::getenv("SONICBOOM_MOE_PREFETCH_STAGE_KB");
        kb_env && *kb_env) {
      char* end = nullptr;
      const unsigned long long kb = std::strtoull(kb_env, &end, 10);
      if (end == kb_env || *end != '\0' || kb == 0)
        return std::unexpected(
            std::string("expert pool: SONICBOOM_MOE_PREFETCH_STAGE_KB is not a "
                        "positive size in KiB: ") +
            kb_env);
      pf_bytes = uint64_t(kb) * 1024ull;
    }
    pool->prefetch_staging_ = device::staging_create(pf_bytes);
    if (!pool->prefetch_staging_)
      return std::unexpected(
          std::string("expert pool: cannot allocate the prefetch staging "
                      "buffer"));
  }

  if (stats_enabled())
    std::fprintf(stderr, "expert pool: %llu slots x %llu bytes, %llu MiB\n",
                 static_cast<unsigned long long>(pool->n_slots_),
                 static_cast<unsigned long long>(slot_bytes),
                 static_cast<unsigned long long>(pool_bytes >> 20));
  return pool;
}

std::expected<void, std::string> ExpertPool::select(
    uint32_t block, std::span<const int64_t> ids, bool upload_misses) {
  std::lock_guard<std::mutex> lk(mutex_);
  const BlockInfo& info = block < blocks_.size() ? blocks_[block] : BlockInfo{};
  if (info.experts == 0)
    return std::unexpected("expert pool: block " + std::to_string(block) +
                           " has no routed experts");
  for (int64_t id : ids) {
    if (id < 0 || uint64_t(id) >= info.experts)
      return std::unexpected("expert pool: expert " + std::to_string(id) +
                             " out of range for block " +
                             std::to_string(block));
  }
  if (ids.empty()) return {};

  if (std::FILE* f = trace_file()) {
    std::fprintf(f, "%u %zu", block, ids.size());
    for (int64_t id : ids) std::fprintf(f, " %lld", static_cast<long long>(id));
    std::fputc('\n', f);
    std::fflush(f);
  }

  const uint64_t node = ++stats_.nodes;
  // Aging LFU: every kAgePeriod nodes the per-key counts halve, so one burst
  // of use fades while a set that keeps returning holds its place. A decode
  // step is one node per block, so the counts age roughly every 26 steps.
  if (node % kAgePeriod == 0)
    for (uint32_t& u : uses_) u >>= 1;

  // The distinct keys this node needs; a slot holding one of them may not be
  // evicted.
  std::vector<int64_t> needed;
  needed.reserve(ids.size());
  for (int64_t id : ids) {
    const int64_t key = int64_t(info.key_base) + id;
    if (std::find(needed.begin(), needed.end(), key) == needed.end())
      needed.push_back(key);
  }

  // The selection buffer, one int32 slot per id.
  if (uint64_t(ids.size()) > sel_entries_) {
    const uint64_t handle = device::acquire(uint64_t(ids.size()) * sizeof(int32_t));
    if (!handle)
      return std::unexpected(
          "expert pool: cannot grow the selection buffer");
    if (sel_handle_) device::release(sel_handle_, 0);
    sel_handle_ = handle;
    sel_entries_ = uint64_t(ids.size());
  }
  sel_host_.resize(ids.size());

  // Copies issued ahead (prefetch) must finish before this node's copies touch
  // the same slot, and a prefetched slot must be complete before the kernel
  // reads it. Both are stream waits, not host waits: the copies had the block's
  // other nodes to travel under.
  if (prefetch_staging_ && prefetch_ready_) {
    if ((staging_ && !device::staging_chain(staging_, prefetch_staging_)) ||
        !device::staging_wait(prefetch_staging_))
      return std::unexpected(
          "expert pool: cannot order the prefetched copies");
    prefetch_ready_ = false;
  }

  // The copy stream must not overwrite a slot an earlier node's kernel may
  // still be reading (the synchronous cudaMemcpy this replaces got that for
  // free). One event orders all the copies of this node behind that work.
  if (staging_ && !device::staging_begin(staging_))
    return std::unexpected(
        "expert pool: cannot start the staging pipeline");

  auto put = [this](const void* host, uint64_t handle, uint64_t bytes) {
    if (staging_) return device::staging_upload(staging_, host, handle, bytes);
    return device::upload(host, handle, bytes);
  };

  for (size_t j = 0; j < ids.size(); ++j) {
    const int64_t key = int64_t(info.key_base) + ids[j];
    int32_t slot = slot_of_[key];
    if (slot >= 0) {
      ++stats_.hits;
      // A slot this select found because prefetch() brought it in: count it
      // once, as used.
      if (prefetched_[size_t(key)] != 0) {
        prefetched_[size_t(key)] = 0;
        ++stats_.prefetch_hits;
      }
    } else {
      int32_t victim = -1;
      if (upload_misses) {
        victim = victim_for(needed);
        if (victim < 0)
          return std::unexpected(
              "expert pool: every slot is needed by one node");
        if (owner_[victim] >= 0) {
          slot_of_[owner_[victim]] = -1;
          note_evicted(owner_[victim]);
        }
      } else {
        // SONICBOOM_MOE_CPU_MISSES: fill an empty slot if there is one; a full
        // pool leaves the key to the host. Never evicts, so the pool's
        // contents stop changing once it is full.
        for (uint64_t s = 0; s < n_slots_; ++s) {
          if (owner_[s] < 0) {
            victim = int32_t(s);
            break;
          }
        }
        if (victim < 0) {
          sel_host_[j] = -1;
          ++stats_.misses;
          ++stats_.cpu_misses;
          continue;
        }
      }

      const quant::QuantizedTensor* role_tensors[3] = {gate_of_[key],
                                                       up_of_[key],
                                                       down_of_[key]};
      for (uint32_t role = 0; role < 3; ++role) {
        const quant::QuantizedTensor& t = *role_tensors[role];
        const uint64_t bytes = t.byte_size();
        const uint64_t base = handles_[role] + uint64_t(victim) * stride_[role];
        if (!put(t.data().data(), base, bytes)) {
          owner_[victim] = -1;
          return std::unexpected(
              "expert pool: cannot copy an expert matrix to the device");
        }
        // Keep the kernel's read-ahead tail zero.
        const uint64_t tail = stride_[role] - bytes;
        if (tail > 0) {
          static const std::byte zeros[kSlotAlignment] = {};
          if (!put(zeros, base + bytes, tail)) {
            owner_[victim] = -1;
            return std::unexpected(
                "expert pool: cannot zero a slot's read-ahead tail");
          }
        }
        stats_.bytes += bytes;
      }
      owner_[victim] = key;
      slot_of_[key] = victim;
      ++stats_.misses;
      slot = victim;
    }
    sel_host_[j] = slot;
  }

  // Stamp every slot this node uses, once, and count the use of its key. A key
  // the host computes has no slot and left the pool unchanged.
  for (int64_t key : needed) {
    const int32_t s = slot_of_[size_t(key)];
    if (s < 0) continue;
    stamp_[size_t(s)] = node;
    ++uses_[size_t(key)];
  }

  if (!put(sel_host_.data(), sel_handle_,
           uint64_t(ids.size()) * sizeof(int32_t)))
    return std::unexpected(
        "expert pool: cannot write the selection buffer");
  if (staging_ && !device::staging_commit(staging_))
    return std::unexpected(
        "expert pool: cannot commit the staged copies");
  return {};
}

int32_t ExpertPool::victim_for(const std::vector<int64_t>& needed) const {
  // Evict the least-used slot this node does not need; a tie goes to the older
  // stamp, then the lowest slot index. An empty slot has no key, so it counts
  // as unused and is taken first.
  int32_t victim = -1;
  uint64_t victim_use = 0, victim_stamp = 0;
  for (uint64_t s = 0; s < n_slots_; ++s) {
    if (owner_[s] >= 0 &&
        std::find(needed.begin(), needed.end(), owner_[s]) != needed.end())
      continue;
    const uint64_t use =
        owner_[s] >= 0 ? uses_[size_t(owner_[s])] : uint64_t{0};
    if (victim < 0 || use < victim_use ||
        (use == victim_use && stamp_[s] < victim_stamp)) {
      victim = int32_t(s);
      victim_use = use;
      victim_stamp = stamp_[s];
    }
  }
  return victim;
}

void ExpertPool::note_evicted(int64_t key) {
  if (prefetched_[size_t(key)] == 0) return;
  prefetched_[size_t(key)] = 0;
  stats_.prefetch_waste += gate_of_[size_t(key)]->byte_size() +
                           up_of_[size_t(key)]->byte_size() +
                           down_of_[size_t(key)]->byte_size();
}

std::expected<void, std::string> ExpertPool::prefetch(
    uint32_t block, std::span<const int64_t> ids) {
  std::lock_guard<std::mutex> lk(mutex_);
  if (!prefetch_staging_ || ids.empty()) return {};
  const BlockInfo& info = block < blocks_.size() ? blocks_[block] : BlockInfo{};
  if (info.experts == 0) return {};

  // The distinct predicted keys, in rank order; the needed set protects an
  // earlier admission of this call from a later one.
  std::vector<int64_t> needed;
  needed.reserve(ids.size());
  for (int64_t id : ids) {
    if (id < 0 || uint64_t(id) >= info.experts) continue;
    const int64_t key = int64_t(info.key_base) + id;
    if (std::find(needed.begin(), needed.end(), key) == needed.end())
      needed.push_back(key);
  }
  if (needed.empty()) return {};

  std::vector<int64_t> todo;
  todo.reserve(needed.size());
  for (int64_t key : needed)
    if (slot_of_[size_t(key)] < 0) todo.push_back(key);
  if (todo.empty()) return {};
  ++stats_.prefetch_nodes;

  // The copies wait for everything already launched on the compute stream, so
  // a slot this block's kernel still reads is not overwritten under it, and
  // run on the prefetch pipeline so nothing waits for them here.
  if (!device::staging_begin(prefetch_staging_)) {
    prefetch_ready_ = false;
    return std::unexpected(
        "expert pool: cannot start the prefetch pipeline");
  }

  auto fail = [this](const char* msg) -> std::expected<void, std::string> {
    // A partial copy leaves slot contents unknown. Forget every residency (the
    // arrays are fully rewritten by the next select), so a failed prefetch can
    // never turn a later hit into a stale read. The next select misses and
    // uploads everything, which is slower but identical. The copies already
    // queued are fenced first, so that select's copies and kernel are ordered
    // after them rather than racing them.
    std::fill(owner_.begin(), owner_.end(), -1);
    std::fill(slot_of_.begin(), slot_of_.end(), -1);
    std::fill(prefetched_.begin(), prefetched_.end(), 0);
    prefetch_ready_ = device::staging_fence(prefetch_staging_);
    return std::unexpected(std::string("expert pool: ") + msg);
  };

  for (int64_t key : todo) {
    const int32_t slot = victim_for(needed);
    if (slot < 0) break;  // every slot holds a predicted expert: nothing to gain
    if (owner_[slot] >= 0) {
      slot_of_[size_t(owner_[slot])] = -1;
      note_evicted(owner_[slot]);
    }
    const quant::QuantizedTensor* role_tensors[3] = {gate_of_[size_t(key)],
                                                     up_of_[size_t(key)],
                                                     down_of_[size_t(key)]};
    for (uint32_t role = 0; role < 3; ++role) {
      const quant::QuantizedTensor& t = *role_tensors[role];
      const uint64_t bytes = t.byte_size();
      const uint64_t base = handles_[role] + uint64_t(slot) * stride_[role];
      if (!device::staging_upload(prefetch_staging_, t.data().data(), base,
                                  bytes))
        return fail("cannot copy a predicted expert to the device");
      const uint64_t tail = stride_[role] - bytes;
      if (tail > 0) {
        static const std::byte zeros[kSlotAlignment] = {};
        if (!device::staging_upload(prefetch_staging_, zeros, base + bytes,
                                    tail))
          return fail("cannot zero a slot's read-ahead tail");
      }
      stats_.prefetch_bytes += bytes;
      stats_.bytes += bytes;
    }
    owner_[slot] = key;
    slot_of_[size_t(key)] = slot;
    stamp_[slot] = stats_.nodes;
    prefetched_[size_t(key)] = 1;
    ++stats_.prefetch_uploads;
  }

  // Record the batch's completion on the copy stream; select() waits for it.
  if (!device::staging_fence(prefetch_staging_)) {
    prefetch_ready_ = false;
    return fail("cannot fence the prefetched copies");
  }
  prefetch_ready_ = true;
  return {};
}

void ExpertPool::clear() {
  std::lock_guard<std::mutex> lk(mutex_);
  std::fill(owner_.begin(), owner_.end(), -1);
  std::fill(stamp_.begin(), stamp_.end(), 0);
  std::fill(slot_of_.begin(), slot_of_.end(), -1);
  std::fill(uses_.begin(), uses_.end(), 0);
  std::fill(prefetched_.begin(), prefetched_.end(), 0);
  prefetch_ready_ = false;
  stats_ = Stats{};
}

void add_expert_timing(const ExpertTiming& t) {
  TimingRegistry& tr = timing_registry();
  std::lock_guard<std::mutex> lk(tr.mutex);
  tr.timing.select_s += t.select_s;
  tr.timing.experts_s += t.experts_s;
  tr.timing.route_s += t.route_s;
}

std::expected<ExpertPool*, std::string> expert_pool_for(
    const model::Gemma4Weights& weights) {
  Registry& r = registry();
  std::lock_guard<std::mutex> lk(r.mutex);
  if (r.pool && r.key != &weights) {
    r.pool.reset();
    r.key = nullptr;
  }
  if (!r.pool) {
    auto created = ExpertPool::create(weights);
    if (!created) return std::unexpected(created.error());
    r.pool = std::move(*created);
    r.key = &weights;
    if (stats_enabled()) std::atexit(print_stats_at_exit);
  }
  return r.pool.get();
}

void clear_expert_pool() {
  Registry& r = registry();
  std::lock_guard<std::mutex> lk(r.mutex);
  r.pool.reset();
  r.key = nullptr;
}

} // namespace sonicboom::planner
