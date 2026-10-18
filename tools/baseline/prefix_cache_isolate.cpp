// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// prefix_cache_isolate — C.4 prefix-cache isolation measurement (measure only,
// no fix). Reproduces the reuse-vs-full divergence on a real-text prompt and
// localizes it two ways:
//
//   1. Token-stream divergence: full prefill (a) / prefix-reuse batched (b) /
//      prefix-reuse per-token (c), first divergent step + top-1/top-2 logits
//      (via the SONICBOOM_DUMP_LOGITS debug hook in gemma4_plan.cpp).
//   2. KV-cache diff a vs b: for the `--new` freshly-added positions, compare
//      each KV layer's cached K and V rows slot-by-slot and report the first
//      (layer, position, slot) whose max_abs > 1e-3.
//
// The device KV buffers are read directly through hal::Buffer::id (a CUDA device
// pointer) with cudaMemcpy, so no Layer 2 API change is needed for the cache
// read. Logits come from the env-gated dump the tool turns on for its runs.
//
// Usage: prefix_cache_isolate [--prefix N] [--new N] [--steps S] [--tokens-file P]
// Build: tools/baseline/build_prefix_cache_isolate.sh

#include <sonicboom/gguf/reader.h>
#include <sonicboom/hal/buffer.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;
namespace sbhal = sonicboom::hal;

static std::vector<uint64_t> load_tokens(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    fprintf(stderr, "tokens file not found: %s\n", path.c_str());
    exit(2);
  }
  std::vector<uint64_t> t;
  uint64_t tok;
  while (in >> tok) t.push_back(tok);
  if (t.empty()) {
    fprintf(stderr, "tokens file empty: %s\n", path.c_str());
    exit(2);
  }
  return t;
}

static std::string toks(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  s += "]";
  return s;
}

// First index where `x` and `y` differ (or their common length if equal).
static size_t first_diff(const std::vector<int64_t>& x,
                         const std::vector<int64_t>& y) {
  size_t n = std::min(x.size(), y.size());
  for (size_t i = 0; i < n; ++i)
    if (x[i] != y[i]) return i;
  return n;
}

// Read one device K/V buffer's full contents to host.
static bool read_buf(const sbhal::Buffer& b, std::vector<float>& out) {
  out.resize(b.size_bytes / sizeof(float));
  cudaError_t e = cudaMemcpy(out.data(), reinterpret_cast<void*>(b.id),
                             b.size_bytes, cudaMemcpyDeviceToHost);
  return e == cudaSuccess;
}

int main(int argc, char** argv) {
  uint64_t prefix_n = 4000;
  uint64_t new_n = 100;
  uint64_t steps = 32;
  std::string tokens_file = "real_prompt.tokens";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--prefix")) prefix_n = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--new")) new_n = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--steps")) steps = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--tokens-file")) tokens_file = argv[++i];
  }
  if (prefix_n == 0 || new_n == 0 || steps == 0) {
    fprintf(stderr,
            "usage: prefix_cache_isolate [--prefix N] [--new N] [--steps S] [--tokens-file P]\n");
    return 2;
  }
  const std::vector<uint64_t> all = load_tokens(tokens_file);
  if (prefix_n + new_n > all.size()) {
    fprintf(stderr, "tokens file has %zu tokens, need %llu\n", all.size(),
            (unsigned long long)(prefix_n + new_n));
    return 2;
  }

  std::string path;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF")) path = env;
  if (path.empty()) {
    const char* root = std::getenv("SONICBOOM_ROOT");
    std::string base = root ? std::string(root) : std::string(".");
    path = base + "/models/gemma-4-E4B-it-Q3_K_M.gguf";
  }
  if (!std::filesystem::exists(path)) {
    fprintf(stderr, "model not found: %s\n", path.c_str());
    return 2;
  }
  auto r = sbgguf::Reader::load(path);
  if (!r) { fprintf(stderr, "failed to load GGUF\n"); return 1; }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) { fprintf(stderr, "load_gemma4 error\n"); return 1; }

  const std::vector<uint64_t> prefix(all.begin(), all.begin() + prefix_n);
  const std::vector<uint64_t> full(all.begin(), all.begin() + prefix_n + new_n);

  // Turn on the logit dump for every run (does not change the token stream or
  // the KV cache — only the graph tail goes Argmax -> Softcap, monotonic).
  setenv("SONICBOOM_DUMP_LOGITS", "1", 1);
  const char* dump_a = "/tmp/isolate_a.log";
  const char* dump_b = "/tmp/isolate_b.log";
  const char* dump_c = "/tmp/isolate_c.log";
  std::remove(dump_a); std::remove(dump_b); std::remove(dump_c);

  // --- path a: full prefill (fresh session -> prefix_len == 0) --------------
  std::vector<int64_t> a_toks(steps, -1);
  sbplanner::KvCacheSession sa;
  sa.n_ctx = full.size() + steps;
  {
    setenv("SONICBOOM_DUMP_LOGITS_FILE", dump_a, 1);
    sbmodel::generate_prefill_spine_cuda_prefix(*m, full, 0, a_toks, sa, true);
  }

  // --- path b: prefix reuse (seed prefix, then reuse for `full`) -------------
  std::vector<int64_t> b_toks(steps, -1);
  sbplanner::KvCacheSession sb;
  sb.n_ctx = full.size() + steps;
  {
    std::vector<int64_t> setup(1, -1);
    setenv("SONICBOOM_DUMP_LOGITS_FILE", dump_b, 1);
    sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, setup, sb, true);
    sbmodel::generate_prefill_spine_cuda_prefix(*m, full, 0, b_toks, sb, true);
  }

  // --- path c: prefix reuse, per-token reference ----------------------------
  std::vector<int64_t> c_toks(steps, -1);
  sbplanner::KvCacheSession sc;
  sc.n_ctx = full.size() + steps;
  {
    std::vector<int64_t> setup(1, -1);
    setenv("SONICBOOM_DUMP_LOGITS_FILE", dump_c, 1);
    sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, setup, sc, true);
    sbmodel::generate_prefill_spine_cuda_prefix(*m, full, 0, c_toks, sc, false);
  }

  // --- token streams + divergence -------------------------------------------
  printf("tokens (a full)  : %s\n", toks(a_toks).c_str());
  printf("tokens (b reuse) : %s\n", toks(b_toks).c_str());
  printf("tokens (c per-tok): %s\n", toks(c_toks).c_str());

  auto report_div = [&](const char* label, const std::vector<int64_t>& x,
                        const std::vector<int64_t>& y) {
    size_t d = first_diff(x, y);
    if (d == x.size() && d == y.size())
      printf("%s: identical (%zu tokens)\n", label, x.size());
    else
      printf("%s: DIVERGE at step %zu (top1 %lld vs %lld)\n", label, d,
             (long long)(d < x.size() ? x[d] : -1),
             (long long)(d < y.size() ? y[d] : -1));
  };
  report_div("b vs c", b_toks, c_toks);
  report_div("a vs c", a_toks, c_toks);

  // --- logits at the first divergent step (from the dump files) -------------
  auto dump_line = [&](const char* f, uint64_t step, std::string& out) -> bool {
    std::ifstream in(f);
    std::string line;
    char want[64];
    snprintf(want, sizeof(want), "step=%llu ", (unsigned long long)step);
    while (std::getline(in, line))
      if (line.find(want) != std::string::npos) { out = line; return true; }
    return false;
  };
  {
    size_t d_bc = first_diff(b_toks, c_toks);
    size_t d_ac = first_diff(a_toks, c_toks);
    if (d_bc < b_toks.size() || d_bc < c_toks.size()) {
      std::string lb, lc;
      if (dump_line(dump_b, d_bc, lb) && dump_line(dump_c, d_bc, lc))
        printf("  [b vs c] step %zu logits:\n    b: %s\n    c: %s\n", d_bc,
               lb.c_str(), lc.c_str());
    }
    if (d_ac < a_toks.size() || d_ac < c_toks.size()) {
      std::string la, lc;
      if (dump_line(dump_a, d_ac, la) && dump_line(dump_c, d_ac, lc))
        printf("  [a vs c] step %zu logits:\n    a: %s\n    c: %s\n", d_ac,
               la.c_str(), lc.c_str());
    }
  }

  // --- KV-cache diff a vs b -------------------------------------------------
  if (!sa.caches || !sb.caches || sa.caches->size() != sb.caches->size()) {
    fprintf(stderr, "cache publish failed (%zu vs %zu)\n",
            sa.caches ? sa.caches->size() : 0,
            sb.caches ? sb.caches->size() : 0);
    return 1;
  }
  const size_t n_layer = sa.caches->size();
  printf("KV-cache diff a vs b over %llu new positions (%zu KV layers):\n",
         (unsigned long long)new_n, n_layer);

  bool any_gt = false;
  for (size_t l = 0; l < n_layer; ++l) {
    const sbplanner::KvCacheDev& A = (*sa.caches)[l];
    const sbplanner::KvCacheDev& B = (*sb.caches)[l];
    const uint64_t n_slots = A.n_slots;
    const uint64_t row = A.n_heads_kv * A.head_dim;

    std::vector<float> ak, av, bk, bv;
    if (!read_buf(A.k_buf, ak) || !read_buf(A.v_buf, av) ||
        !read_buf(B.k_buf, bk) || !read_buf(B.v_buf, bv)) {
      fprintf(stderr, "layer %zu: buffer read failed\n", l);
      continue;
    }

    float k_max = 0.0f, v_max = 0.0f;
    int64_t first_pos = -1, first_slot = -1, first_e = -1;
    const char* which = nullptr;
    for (uint64_t i = 0; i < new_n; ++i) {
      const uint64_t pos = prefix_n + i;
      const uint64_t slot = pos % n_slots;
      const uint64_t base = slot * row;
      for (uint64_t e = 0; e < row; ++e) {
        float dk = std::fabs(ak[base + e] - bk[base + e]);
        float dv = std::fabs(av[base + e] - bv[base + e]);
        if (dk > k_max) k_max = dk;
        if (dv > v_max) v_max = dv;
        if ((dk > 1e-3f || dv > 1e-3f) && !any_gt) {
          any_gt = true;
          const uint64_t head = e / A.head_dim;
          const uint64_t de = e % A.head_dim;
          printf("  first >1e-3: layer=%zu pos=%llu slot=%llu head=%llu d=%llu "
                 "| K a=%.6f b=%.6f dk=%.6f | V a=%.6f b=%.6f dv=%.6f\n",
                 l, (unsigned long long)pos, (unsigned long long)slot,
                 (unsigned long long)head, (unsigned long long)de,
                 double(ak[base + e]), double(bk[base + e]), double(dk),
                 double(av[base + e]), double(bv[base + e]), double(dv));
        }
        if (dk > dv) { if (first_e < 0) { which = "K"; first_pos = int64_t(pos); first_slot = int64_t(slot); first_e = int64_t(e); } }
        else { if (first_e < 0) { which = "V"; first_pos = int64_t(pos); first_slot = int64_t(slot); first_e = int64_t(e); } }
      }
    }
    printf("  layer %2zu n_slots=%4llu: K_max=%.3g V_max=%.3g%s\n",
           l, (unsigned long long)n_slots, double(k_max), double(v_max),
           (k_max > 1e-3f || v_max > 1e-3f) ? "  <-- exceeds 1e-3" : "");
  }
  if (!any_gt) printf("  (no K/V element differs by more than 1e-3)\n");

  // Seed boundary: in path a, position prefix_n-1 is the first token of the
  // trailing segment (new_n+1 tokens); in path b it is the final token of the
  // prefix seed. When prefix_n % 512 == 1 that trailing seed segment is a
  // single token, i.e. exactly where the two paths differ in segment extent.
  // Positions before it are laid out in the same 512-token chunks in both
  // paths, so a per-layer diff here localizes the batch (n) dependence to the
  // last prefix token.
  if (prefix_n > 0) {
    printf("KV-cache diff a vs b at the last prefix token (pos=%llu):\n",
           (unsigned long long)(prefix_n - 1));
    bool any_seed_gt = false;
    for (size_t l = 0; l < n_layer; ++l) {
      const sbplanner::KvCacheDev& A = (*sa.caches)[l];
      const sbplanner::KvCacheDev& B = (*sb.caches)[l];
      const uint64_t row = A.n_heads_kv * A.head_dim;
      std::vector<float> ak, av, bk, bv;
      if (!read_buf(A.k_buf, ak) || !read_buf(A.v_buf, av) ||
          !read_buf(B.k_buf, bk) || !read_buf(B.v_buf, bv))
        continue;
      float km[4] = {0, 0, 0, 0}, vm[4] = {0, 0, 0, 0};
      for (uint64_t back = 1; back <= 4; ++back) {  // last 4 prefix tokens
        const uint64_t pos = prefix_n - back;
        const uint64_t base = (pos % A.n_slots) * row;
        for (uint64_t e = 0; e < row; ++e) {
          km[back - 1] = std::max(km[back - 1],
                                  std::fabs(ak[base + e] - bk[base + e]));
          vm[back - 1] = std::max(vm[back - 1],
                                  std::fabs(av[base + e] - bv[base + e]));
        }
      }
      printf("  layer %2zu: last K_max=%.3g V_max=%.3g | "
             "[-2] K=%.3g V=%.3g | [-3] K=%.3g V=%.3g | "
             "[-4] K=%.3g V=%.3g\n",
             l, double(km[0]), double(vm[0]), double(km[1]), double(vm[1]),
             double(km[2]), double(vm[2]), double(km[3]), double(vm[3]));
      if (km[0] > 1e-3f || vm[0] > 1e-3f) any_seed_gt = true;
    }
    if (!any_seed_gt) printf("  (no K/V element differs by more than 1e-3)\n");
  }

  // Per-position breakdown for the first SWA layer (l=0) and first global layer
  // (n_slots >= prefix_n+new_n), to localize which positions differ.
  for (size_t l = 0; l < n_layer; ++l) {
    const sbplanner::KvCacheDev& A = (*sa.caches)[l];
    const sbplanner::KvCacheDev& B = (*sb.caches)[l];
    const bool swa = A.n_slots < prefix_n + new_n;
    if ((swa && l != 0) || (!swa && l != 5)) continue;
    const uint64_t row = A.n_heads_kv * A.head_dim;
    std::vector<float> ak, av, bk, bv;
    if (!read_buf(A.k_buf, ak) || !read_buf(A.v_buf, av) ||
        !read_buf(B.k_buf, bk) || !read_buf(B.v_buf, bv))
      continue;
    printf("  per-position (layer %zu %s):\n", l, swa ? "SWA" : "global");
    for (uint64_t i = 0; i < new_n; ++i) {
      const uint64_t pos = prefix_n + i;
      const uint64_t slot = pos % A.n_slots;
      const uint64_t base = slot * row;
      float km = 0.0f, vm = 0.0f;
      for (uint64_t e = 0; e < row; ++e) {
        km = std::max(km, std::fabs(ak[base + e] - bk[base + e]));
        vm = std::max(vm, std::fabs(av[base + e] - bv[base + e]));
      }
      if (km > 1e-6f || vm > 1e-6f)
        printf("    pos=%llu slot=%llu K=%.3g V=%.3g\n",
               (unsigned long long)pos, (unsigned long long)slot, double(km),
               double(vm));
    }
  }

  // Overwritten SWA slots: the slot range the new positions write, and the ring
  // occupants they evict (the positions 512 back).
  for (size_t l = 0; l < n_layer; ++l) {
    const sbplanner::KvCacheDev& A = (*sa.caches)[l];
    if (A.n_slots >= prefix_n + new_n) continue;  // global layer: no wrap
    const uint64_t w = A.n_slots;
    const uint64_t first_slot = prefix_n % w;
    const uint64_t wrap = (first_slot + new_n) % w;
    printf("  SWA layer %zu: writes slots [%llu..%llu] + [0..%llu], "
           "evicting positions [%llu..%llu]\n",
           l, (unsigned long long)first_slot, (unsigned long long)(w - 1),
           (unsigned long long)wrap, (unsigned long long)(prefix_n - w),
           (unsigned long long)(prefix_n - w + new_n - 1));
  }
  return 0;
}
