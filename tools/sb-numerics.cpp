// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// sb-numerics — reduction-mode acceptance tool (boom-r3 T1).
//
// The reduction accumulator is chosen once per device (planner/reduce_mode.h):
// F64 where doubles are cheap, DF32 (double-float over f32) where they are not.
// This tool decides whether the choice is *acceptable*: it runs the same
// probes and greedy streams under each mode (each mode in its own process,
// because the mode is a process-wide static) and compares the results.
//
//   sb-numerics            compare F64 against DF32 and print the verdict
//   sb-numerics dump FILE  run this process's mode once and write its dump
//
// boom-r4 M1 adds the numerics-PROFILE acceptance gate (strict vs standard):
//
//   sb-numerics profile            run both profiles and print the gate table
//   sb-numerics profile-dump FILE MODEL   run this process's profile once
//
// The profile gate measures perplexity (teacher-forced NLL on a fixed >= 20k
// token corpus per model, relative delta <= 0.5%) on two paths: the corpus
// through the chunked prefill kernels and the first 8192 tokens through the
// per-token decode graph at n_ctx 8192. Both paths are scored with their own
// per-position distribution trace, so the KL and same-top thresholds apply to
// each, and the gate passes only when prefill and decode both pass. It also
// checks top-1 agreement over 50 prompts (>= 99%), and, within each profile,
// 3-run greedy determinism and prefix reuse vs full prefill — both
// bit-identical. The cross-profile first divergent greedy step is
// informational.
//
// SonicBoom guarantees exactly three properties:
//   (1) determinism — same input, same output, every run;
//   (2) prefix-reuse identity — output does not depend on whether, or where,
//       a prefix was reused;
//   (3) KV-capacity independence — output does not depend on n_ctx.
// Prefill-vs-decode identity is NOT guaranteed: a batched prefill and the same
// tokens fed one by one may use different arithmetic. Speculative decoding
// on/off is nevertheless bit-identical — the verify step runs decode arithmetic
// for every row (tests/core/test_spec_invariance.cpp).
//
// What is measured:
//   * per T0a kernel: max ulp to the F64-mode output (the output dtype's ulp),
//     the fraction of elements that differ, and the max / relative error
//     against an fp64 reference recomputed from the same inputs;
//   * greedy streams: 3 prompts x 128 tokens per model, CPU and CUDA spine;
//   * top-1: the first greedy token of 50 prompts per model, CPU and CUDA.
//
// DF32 is accepted when, in every kernel, the output differs from F64 by at
// most 1 ulp and its fp64-reference error is within 2x F64's; and when the
// CUDA greedy top-1 agrees with F64 on at least 99.9% of the compared steps
// (3 x 128 decode steps + 50 prefill first-tokens) for every model.
//
// Skips cleanly (exit 0) when a model is absent. Exit 1 means "DF32 not
// accepted"; that is a measurement, not a tool failure.

#include <sonicboom/compiler/GlueCodegen.h>
#include <sonicboom/dtype.h>
#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/numerics.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/vocab/vocab.h>

#include "../core/src/nn/cuda_resident.h"
#include "../core/src/planner/glue_jit.h"
#include "../core/src/planner/reduce_mode.h"
#include "../core/src/quant/cuda_resident.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbvocab = sonicboom::vocab;
namespace sbnn = sonicboom::nn;
namespace sbquant = sonicboom::quant;

namespace {

// --- deterministic probe inputs -------------------------------------------
//
// The same generator in every process and in the comparison pass, so a probe's
// input is fixed by (index, salt) alone.

uint32_t mix(uint32_t h) {
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return h;
}

float probe_x(uint64_t i, uint32_t salt) {
  const uint32_t h = mix(uint32_t(i * 2654435761u) ^ salt);
  return -2.0f + 4.0f * float(h & 0x00ffffffu) / 16777216.0f;
}

float probe_w(uint64_t i, uint32_t salt) {
  const uint32_t h = mix(uint32_t(i * 2246822519u) ^ (salt + 0x9e3779b9u));
  return 0.5f + float(h & 0x00ffffffu) / 16777216.0f;
}

// --- output encoding -------------------------------------------------------

std::string hex32(uint32_t v) {
  char b[9];
  std::snprintf(b, sizeof b, "%08x", v);
  return b;
}

uint32_t bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

float from_bits(uint32_t u) {
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// Ordered-integer mapping, so ulp distance is |a-b| on the mapped value.
int64_t ordered(uint32_t u) {
  return (u & 0x80000000u) ? -int64_t(u & 0x7fffffffu)
                           : int64_t(u);
}

int64_t ulp_distance(float a, float b) {
  return std::abs(ordered(bits(a)) - ordered(bits(b)));
}

struct Dump {
  std::string mode;
  // probe name -> values as 32-bit words (f32 bits, or raw i8 bytes widened;
  // the comparison knows each probe's kind).
  std::map<std::string, std::vector<uint32_t>> probes;
  // "cpu:<model>:<p>" / "cuda:<model>:<p>" -> token ids.
  std::map<std::string, std::vector<int64_t>> streams;
  // "cpu:<model>:<i>" -> first greedy token.
  std::map<std::string, int64_t> top1;
};

std::string probe_hex(const std::vector<uint32_t>& v) {
  std::string s;
  s.reserve(v.size() * 9);
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ',';
    s += hex32(v[i]);
  }
  return s;
}

bool write_dump(const Dump& d, const std::string& path) {
  std::ofstream f(path);
  if (!f) return false;
  f << "mode " << d.mode << "\n";
  for (const auto& [k, v] : d.probes) f << "probe " << k << " " << probe_hex(v) << "\n";
  for (const auto& [k, v] : d.streams) {
    f << "stream " << k;
    for (int64_t t : v) f << " " << t;
    f << "\n";
  }
  for (const auto& [k, v] : d.top1) f << "top1 " << k << " " << v << "\n";
  return f.good();
}

bool read_dump(const std::string& path, Dump& d) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream is(line);
    std::string tag;
    is >> tag;
    if (tag == "mode") {
      is >> d.mode;
    } else if (tag == "probe") {
      std::string name, hex;
      is >> name >> hex;
      std::vector<uint32_t> v;
      std::istringstream hs(hex);
      std::string item;
      while (std::getline(hs, item, ','))
        v.push_back(uint32_t(std::strtoul(item.c_str(), nullptr, 16)));
      d.probes[name] = std::move(v);
    } else if (tag == "stream") {
      std::string key;
      is >> key;
      std::vector<int64_t> v;
      int64_t t = 0;
      while (is >> t) v.push_back(t);
      d.streams[key] = std::move(v);
    } else if (tag == "top1") {
      std::string key;
      int64_t t = 0;
      is >> key >> t;
      d.top1[key] = t;
    }
  }
  return !d.mode.empty();
}

// --- fp64 references -------------------------------------------------------
//
// Each reference recomputes the kernel's reduction in double from the same
// inputs. It is a reference, not a bit model: the quantity compared is the
// error each mode makes against it, and DF32's error must stay within twice
// F64's.

double ref_inv(const std::vector<float>& x, float eps) {
  double s = 0.0;
  for (float v : x) s += double(v) * double(v);
  return 1.0 / std::sqrt(s / double(x.size()) + double(eps));
}

// q8_1 scale of one 32-element block: nv = x*inv*w, d = max|nv|/127.
float ref_q8_1_d(const std::vector<float>& x, const std::vector<float>& w,
                 size_t base, float eps, bool use_w) {
  std::vector<float> row(32);
  for (size_t i = 0; i < 32; ++i) row[i] = x[base + i];
  const double inv = ref_inv(x, eps);
  double amax = 0.0;
  for (size_t i = 0; i < 32; ++i) {
    const double nv = double(x[base + i]) * inv * (use_w ? double(w[base + i]) : 1.0);
    amax = std::max(amax, std::abs(nv));
  }
  return float(amax / 127.0);
}

// --- CPU probes ------------------------------------------------------------

void cpu_probes(Dump& d) {
  constexpr size_t kNorm = 2560;
  std::vector<float> x(kNorm), w(kNorm), y(kNorm);
  for (size_t i = 0; i < kNorm; ++i) {
    x[i] = probe_x(i, 0x11);
    w[i] = probe_w(i, 0x12);
  }
  if (sbnn::rms_norm(x, w, 1e-6f, y)) {
    std::vector<uint32_t> v;
    for (float f : y) v.push_back(bits(f));
    d.probes["cpu_rms_norm"] = std::move(v);
  }

  constexpr size_t kHeads = 8, kHeadDim = 256;
  std::vector<float> xh(kHeads * kHeadDim), wh(kHeadDim);
  for (size_t i = 0; i < xh.size(); ++i) xh[i] = probe_x(i, 0x13);
  for (size_t i = 0; i < kHeadDim; ++i) wh[i] = probe_w(i, 0x14);
  if (sbnn::rms_norm_heads(xh, wh, kHeadDim, 1e-6f)) {
    std::vector<uint32_t> v;
    for (float f : xh) v.push_back(bits(f));
    d.probes["cpu_rms_norm_heads"] = std::move(v);
  }

  constexpr uint64_t kCols = 1024, kRows = 8;
  std::vector<float> xv(kCols), yv(kRows), W(kCols * kRows);
  for (uint64_t i = 0; i < kCols; ++i) xv[i] = probe_x(i, 0x15);
  for (uint64_t i = 0; i < kCols * kRows; ++i) W[i] = probe_w(i, 0x16);
  if (sbnn::matvec_f32(W, kCols, xv, yv)) {
    std::vector<uint32_t> v;
    for (float f : yv) v.push_back(bits(f));
    d.probes["cpu_matvec_f32"] = std::move(v);
  }

  std::vector<uint16_t> Wb(kCols * kRows);
  for (uint64_t i = 0; i < Wb.size(); ++i) {
    // An actual bf16 value (the top half of a finite f32), not a raw random
    // bit pattern: arbitrary patterns include NaN/Inf and pollute both the
    // ulp and the reference comparisons.
    Wb[i] = uint16_t(bits(probe_w(i, 0x17)) >> 16);
  }
  if (sbnn::matvec_bf16(Wb, kCols, xv, yv)) {
    std::vector<uint32_t> v;
    for (float f : yv) v.push_back(bits(f));
    d.probes["cpu_matvec_bf16"] = std::move(v);
  }
}

// --- CUDA probes -----------------------------------------------------------

bool have_cuda() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void put_floats(Dump& d, const char* name, const float* p, size_t n) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = bits(p[i]);
  d.probes[name] = std::move(v);
}

void put_bytes(Dump& d, const char* name, const uint8_t* p, size_t n) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = p[i];
  d.probes[name] = std::move(v);
}

void cuda_probe_q8_1(Dump& d) {
  constexpr int n = 3072;
  std::vector<float> x(n), w(n);
  for (int i = 0; i < n; ++i) {
    x[i] = probe_x(uint64_t(i), 0x21);
    w[i] = probe_w(uint64_t(i), 0x22);
  }
  const int nblocks = n / 32;
  const size_t bytes = size_t(nblocks) * sizeof(sbquant::cuda::BlockQ8_1);
  float *xd = nullptr, *wd = nullptr;
  void* yd = nullptr;
  if (cudaMalloc(&xd, n * sizeof(float)) != cudaSuccess) return;
  if (cudaMalloc(&wd, n * sizeof(float)) != cudaSuccess) return;
  if (cudaMalloc(&yd, bytes) != cudaSuccess) return;
  cudaMemcpy(xd, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(yd, 0, bytes);

  const bool ok = sbquant::cuda::rms_norm_quantize_q8_1_dev(
      std::span<const float>(w), xd, 1e-6f,
      reinterpret_cast<sbquant::cuda::BlockQ8_1*>(yd), n);
  cudaDeviceSynchronize();
  std::vector<uint8_t> host(bytes);
  if (ok) cudaMemcpy(host.data(), yd, bytes, cudaMemcpyDeviceToHost);

  std::vector<float> ds;
  std::vector<uint8_t> qs;
  for (int b = 0; b < nblocks; ++b) {
    const auto* blk =
        reinterpret_cast<const sbquant::cuda::BlockQ8_1*>(
            host.data() + size_t(b) * sizeof(sbquant::cuda::BlockQ8_1));
    ds.push_back(blk->d);
    for (int i = 0; i < 32; ++i) qs.push_back(uint8_t(blk->qs[i]));
  }
  put_floats(d, "cuda_rms_norm_quantize_q8_1_d", ds.data(), ds.size());
  put_bytes(d, "cuda_rms_norm_quantize_q8_1_qs", qs.data(), qs.size());

  cudaFree(xd);
  cudaFree(wd);
  cudaFree(yd);
}

void cuda_probe_rms_norm_add(Dump& d) {
  constexpr int tokens = 8, dim = 3072;
  const size_t elems = size_t(tokens) * size_t(dim);
  std::vector<float> x(elems), r(elems), w(dim);
  for (size_t i = 0; i < elems; ++i) {
    x[i] = probe_x(i, 0x31);
    r[i] = probe_x(i, 0x32);
  }
  for (int i = 0; i < dim; ++i) w[i] = probe_w(uint64_t(i), 0x33);
  float *xd = nullptr, *rd = nullptr, *wd = nullptr, *yd = nullptr;
  if (cudaMalloc(&xd, elems * 4) != cudaSuccess) return;
  if (cudaMalloc(&rd, elems * 4) != cudaSuccess) return;
  if (cudaMalloc(&wd, dim * 4) != cudaSuccess) return;
  if (cudaMalloc(&yd, elems * 4) != cudaSuccess) return;
  cudaMemcpy(xd, x.data(), elems * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(rd, r.data(), elems * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), dim * 4, cudaMemcpyHostToDevice);
  cudaMemset(yd, 0, elems * 4);

  void* xarg = xd;
  void* rarg = rd;
  void* warg = wd;
  void* yarg = yd;
  int dimarg = dim;
  float earg = 1e-6f;
  float sarg = 1.0f;
  void* args[] = {&xarg, &rarg, &warg, &yarg, &dimarg, &earg, &sarg};
  const bool ok = sonicboom::planner::glue::launch(
      "rms_norm_add", args, tokens,
      sonicboom::compiler::kRmsNormAddBlockThreads,
      sonicboom::compiler::kRmsNormAddSharedBytes);
  cudaDeviceSynchronize();
  std::vector<float> y(elems);
  if (ok) cudaMemcpy(y.data(), yd, elems * 4, cudaMemcpyDeviceToHost);
  put_floats(d, "cuda_rms_norm_add", y.data(), y.size());

  cudaFree(xd);
  cudaFree(rd);
  cudaFree(wd);
  cudaFree(yd);
}

void cuda_probe_heads_norm(Dump& d) {
  constexpr int heads = 8, dim = 256, positions = 8;
  const size_t elems = size_t(heads) * size_t(dim);
  std::vector<float> x(elems), w(dim);
  for (size_t i = 0; i < elems; ++i) x[i] = probe_x(i, 0x41);
  for (int i = 0; i < dim; ++i) w[i] = probe_w(uint64_t(i), 0x42);
  float *xd = nullptr, *wd = nullptr, *yd = nullptr;
  uint64_t* dpos = nullptr;
  if (cudaMalloc(&xd, elems * 4) != cudaSuccess) return;
  if (cudaMalloc(&wd, dim * 4) != cudaSuccess) return;
  if (cudaMalloc(&yd, elems * 4) != cudaSuccess) return;
  if (cudaMalloc(&dpos, sizeof(uint64_t)) != cudaSuccess) return;
  cudaMemcpy(xd, x.data(), elems * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), dim * 4, cudaMemcpyHostToDevice);
  cudaMemset(yd, 0, elems * 4);
  const uint64_t pos = 37;
  cudaMemcpy(dpos, &pos, sizeof pos, cudaMemcpyHostToDevice);

  // rope=0, cast=0, w=1: the reduction and the apply only. (The rotation calls
  // libdevice trig, whose last-ulp choice belongs to the compiler, not to the
  // reduction mode.)
  void* xarg = xd;
  void* warg = wd;
  int haswarg = 1;
  void* yarg = yd;
  void* posarg = dpos;
  int dimarg = dim;
  float earg = 1e-6f;
  float fsarg = 1.0f;
  float lbarg = 9.21034037f;  // log(10000)
  void* ffarg = nullptr;
  int hpparg = heads / positions;
  int ropearg = 0;
  int castarg = 0;
  void* carg = nullptr;
  int slotsarg = 0;
  int rowarg = 0;
  int storearg = 0;
  void* args[] = {&xarg,   &warg,    &haswarg, &yarg,    &posarg,
                  &dimarg, &earg,    &fsarg,   &lbarg,   &ffarg,
                  &hpparg, &ropearg, &castarg, &carg,    &slotsarg,
                  &rowarg, &storearg};
  const bool ok = sonicboom::planner::glue::launch(
      "heads_norm_rope_cast", args, heads,
      sonicboom::compiler::kHeadsNormRopeCastBlockThreads,
      sonicboom::compiler::kHeadsNormRopeCastSharedBytes);
  cudaDeviceSynchronize();
  std::vector<float> y(elems);
  if (ok) cudaMemcpy(y.data(), yd, elems * 4, cudaMemcpyDeviceToHost);
  put_floats(d, "cuda_heads_norm_rope_cast", y.data(), y.size());

  cudaFree(xd);
  cudaFree(wd);
  cudaFree(yd);
  cudaFree(dpos);
}

void cuda_probes(Dump& d) {
  if (!have_cuda()) return;
  cuda_probe_q8_1(d);
  cuda_probe_rms_norm_add(d);
  cuda_probe_heads_norm(d);
}

// --- greedy streams and top-1 ---------------------------------------------

std::string find_model(const char* env, const char* rel) {
  std::vector<std::string> candidates;
  if (const char* e = std::getenv(env)) candidates.emplace_back(e);
  candidates.emplace_back(std::string("models/") + rel);
  candidates.emplace_back(std::string("../models/") + rel);
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

struct Loaded {
  // The model's weight spans borrow the reader's buffer, so the reader outlives
  // them here (the reader owns the file image; moving it keeps the heap buffer).
  std::optional<sbgguf::Reader> reader;
  sbmodel::Gemma4Model model;
  sbvocab::Vocab vocab;
  std::string name;
};

bool load(const std::string& path, const char* tag, Loaded& out) {
  auto r = sbgguf::Reader::load(path);
  if (!r) return false;
  auto m = sbmodel::load_gemma4(*r);
  if (!m) return false;
  auto v = sbvocab::load_vocab(*r);
  if (!v) return false;
  out.reader.emplace(std::move(*r));
  out.model = std::move(*m);
  out.vocab = std::move(*v);
  out.name = tag;
  return true;
}

std::vector<uint64_t> encode(const Loaded& l, const std::string& text) {
  const auto ids = sbvocab::tokenize(l.vocab, text);
  std::vector<uint64_t> out;
  out.reserve(ids.size());
  for (int32_t id : ids) out.push_back(uint64_t(id));
  return out;
}

const char* const kPrompts[3] = {
    "The capital of France is",
    "List the first five prime numbers.",
    "Explain in one sentence why the sky is blue.",
};

void run_streams(Dump& d, const std::string& only) {
  struct Spec {
    const char* env;
    const char* rel;
    const char* tag;
  } specs[] = {
      {"SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf", "gemma"},
      {"SONICBOOM_QWEN_GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen"},
  };
  for (const auto& s : specs) {
    // One model per process: the device weight cache is keyed by host pointer,
    // and holding two multi-GB models' uploads at once is not what the runtime
    // (or an 8 GB card) is built for.
    if (!only.empty() && only != s.tag) continue;
    const std::string path = find_model(s.env, s.rel);
    if (path.empty()) {
      std::cout << "  (skipped: " << s.tag << " model not found)\n";
      continue;
    }
    Loaded l;
    if (!load(path, s.tag, l)) {
      std::cout << "  (skipped: " << s.tag << " model failed to load)\n";
      continue;
    }
    std::cout << "  " << s.tag << ": " << path << "\n";

    for (int p = 0; p < 3; ++p) {
      const auto prompt = encode(l, kPrompts[p]);
      if (prompt.empty()) continue;
      std::vector<int64_t> toks(128, -1);
      const uint64_t n = sbmodel::generate_prefill_spine(
          l.model, prompt, 0, toks, 0, /*use_prefill=*/true);
      toks.resize(n);
      d.streams["cpu:" + l.name + ":" + std::to_string(p)] = toks;
      std::cout << "    cpu stream p" << p << ": " << n << " tokens\n";
    }
    if (have_cuda()) {
      for (int p = 0; p < 3; ++p) {
        const auto prompt = encode(l, kPrompts[p]);
        if (prompt.empty()) continue;
        std::vector<int64_t> toks(128, -1);
        const uint64_t n = sbmodel::generate_prefill_spine_cuda(
            l.model, prompt, 0, toks, 0, /*use_prefill=*/true);
        toks.resize(n);
        d.streams["cuda:" + l.name + ":" + std::to_string(p)] = toks;
        std::cout << "    cuda stream p" << p << ": " << n << " tokens\n";
      }
    }

    // Top-1 over 50 prompts: the first greedy token of each, through the same
    // prompt+one-token path.
    for (int i = 0; i < 50; ++i) {
      const std::string text = std::string(kPrompts[i % 3]) + " (" +
                               std::to_string(i) + ")";
      const auto prompt = encode(l, text);
      if (prompt.empty()) continue;
      std::vector<int64_t> one(1, -1);
      uint64_t n = sbmodel::generate_prefill_spine(l.model, prompt, 0, one, 0,
                                                   /*use_prefill=*/true);
      if (n == 0) continue;
      d.top1["cpu:" + l.name + ":" + std::to_string(i)] = one[0];
      if (have_cuda()) {
        n = sbmodel::generate_prefill_spine_cuda(l.model, prompt, 0, one, 0,
                                                 /*use_prefill=*/true);
        if (n != 0) d.top1["cuda:" + l.name + ":" + std::to_string(i)] = one[0];
      }
    }
  }
}

// --- comparison ------------------------------------------------------------

// The fp64 reference for a probe, recomputed from the same deterministic
// inputs. Probes whose output is a quantized integer (`_qs`) have no reference.
std::vector<float> probe_ref(const std::string& name) {
  constexpr size_t kNorm = 2560;
  constexpr size_t kHeads = 8, kHeadDim = 256;
  constexpr uint64_t kCols = 1024, kRows = 8;

  auto ref_norm_rows = [](std::vector<float>& x, std::vector<float>& w,
                          size_t row, float eps) {
    double s = 0.0;
    for (size_t i = 0; i < row; ++i) s += double(x[i]) * double(x[i]);
    const double inv = 1.0 / std::sqrt(s / double(row) + double(eps));
    for (size_t i = 0; i < row; ++i)
      x[i] = float(double(x[i]) * inv * double(w[i]));
  };

  if (name == "cpu_rms_norm") {
    std::vector<float> x(kNorm), w(kNorm);
    for (size_t i = 0; i < kNorm; ++i) {
      x[i] = probe_x(i, 0x11);
      w[i] = probe_w(i, 0x12);
    }
    ref_norm_rows(x, w, kNorm, 1e-6f);
    return x;
  }
  if (name == "cpu_rms_norm_heads") {
    std::vector<float> x(kHeads * kHeadDim), w(kHeadDim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = probe_x(i, 0x13);
    for (size_t i = 0; i < kHeadDim; ++i) w[i] = probe_w(i, 0x14);
    for (size_t h = 0; h < kHeads; ++h) {
      std::vector<float> row(x.begin() + h * kHeadDim,
                             x.begin() + (h + 1) * kHeadDim);
      ref_norm_rows(row, w, kHeadDim, 1e-6f);
      for (size_t i = 0; i < kHeadDim; ++i) x[h * kHeadDim + i] = row[i];
    }
    return x;
  }
  if (name == "cpu_matvec_f32" || name == "cpu_matvec_bf16") {
    std::vector<float> xv(kCols), y(kRows);
    for (uint64_t i = 0; i < kCols; ++i) xv[i] = probe_x(i, 0x15);
    const bool bf16 = name == "cpu_matvec_bf16";
    std::vector<uint16_t> Wb(kCols * kRows);
    std::vector<float> W(kCols * kRows);
    for (uint64_t i = 0; i < kCols * kRows; ++i) {
      if (bf16) {
        Wb[i] = uint16_t(bits(probe_w(i, 0x17)) >> 16);
        W[i] = sonicboom::bf16_to_f32(Wb[i]);
      } else {
        W[i] = probe_w(i, 0x16);
      }
    }
    for (uint64_t j = 0; j < kRows; ++j) {
      double acc = 0.0;
      for (uint64_t i = 0; i < kCols; ++i)
        acc += double(W[j * kCols + i]) * double(xv[i]);
      y[j] = float(acc);
    }
    return y;
  }
  if (name == "cuda_rms_norm_quantize_q8_1_d") {
    constexpr int n = 3072;
    std::vector<float> x(n), w(n);
    for (int i = 0; i < n; ++i) {
      x[i] = probe_x(uint64_t(i), 0x21);
      w[i] = probe_w(uint64_t(i), 0x22);
    }
    std::vector<float> out(n / 32);
    for (int blk = 0; blk < n / 32; ++blk)
      out[size_t(blk)] = ref_q8_1_d(x, w, size_t(blk) * 32, 1e-6f, true);
    return out;
  }
  if (name == "cuda_rms_norm_add") {
    constexpr int tokens = 8, dim = 3072;
    std::vector<float> x(size_t(tokens) * dim);
    std::vector<float> w(dim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = probe_x(i, 0x31);
    for (int i = 0; i < dim; ++i) w[size_t(i)] = probe_w(uint64_t(i), 0x33);
    for (int t = 0; t < tokens; ++t) {
      std::vector<float> row(x.begin() + size_t(t) * dim,
                             x.begin() + size_t(t + 1) * dim);
      ref_norm_rows(row, w, dim, 1e-6f);
      for (int i = 0; i < dim; ++i) {
        const size_t idx = size_t(t) * dim + size_t(i);
        // The region's epilogue: residual + normalized value.
        x[idx] = float(double(probe_x(idx, 0x32)) + double(row[size_t(i)]));
      }
    }
    return x;
  }
  if (name == "cuda_heads_norm_rope_cast") {
    constexpr int heads = 8, dim = 256;
    std::vector<float> x(size_t(heads) * dim), w(dim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = probe_x(i, 0x41);
    for (int i = 0; i < dim; ++i) w[size_t(i)] = probe_w(uint64_t(i), 0x42);
    for (int h = 0; h < heads; ++h) {
      std::vector<float> row(x.begin() + size_t(h) * dim,
                             x.begin() + size_t(h + 1) * dim);
      ref_norm_rows(row, w, dim, 1e-6f);
      for (int i = 0; i < dim; ++i) x[size_t(h) * dim + size_t(i)] = row[size_t(i)];
    }
    return x;
  }
  return {};
}

bool compare_and_report(const Dump& f64, const Dump& df32) {
  std::printf("\n  mode under test: %s vs f64\n", df32.mode.c_str());
  std::printf("  %-38s %8s %10s %7s %12s %12s %6s\n", "probe", "n",
              "differ%", "maxulp", "err(f64)", "err(df32)", "verdict");

  struct Row {
    std::string name;
    size_t n;
    double differ_pct;
    int64_t max_ulp;
    double e64, edf;
    bool integer_dtype;
    bool row_ok;
  };
  std::vector<Row> rows;

  bool ok = true;
  for (const auto& [name, av] : f64.probes) {
    const auto it = df32.probes.find(name);
    if (it == df32.probes.end()) continue;
    const auto& bv = it->second;
    const bool integer_dtype = name.find("_qs") != std::string::npos;
    size_t differ = 0;
    int64_t max_ulp = 0;
    double e64 = 0.0, edf = 0.0;

    // fp64 reference, recomputed from the same inputs.
    const std::vector<float> ref = integer_dtype ? std::vector<float>()
                                                 : probe_ref(name);

    for (size_t i = 0; i < std::min(av.size(), bv.size()); ++i) {
      if (integer_dtype) {
        const int64_t d = std::abs(int64_t(av[i]) - int64_t(bv[i]));
        if (d) ++differ;
        max_ulp = std::max(max_ulp, d);
      } else {
        const float fa = from_bits(av[i]);
        const float fb = from_bits(bv[i]);
        const int64_t u = ulp_distance(fa, fb);
        if (u) ++differ;
        max_ulp = std::max(max_ulp, u);
        if (!ref.empty() && i < ref.size()) {
          e64 = std::max(e64, std::abs(double(fa) - double(ref[i])));
          edf = std::max(edf, std::abs(double(fb) - double(ref[i])));
        }
      }
    }
    const size_t n = std::min(av.size(), bv.size());
    const double pct = n ? 100.0 * double(differ) / double(n) : 0.0;

    // Acceptance: <= 1 ulp in the output dtype, and the fp64-reference error
    // within 2x f64's (only where a reference exists).
    bool row_ok = max_ulp <= 1;
    if (!ref.empty()) row_ok = row_ok && edf <= 2.0 * e64 + 1e-30;
    if (!row_ok) ok = false;
    rows.push_back({name, n, pct, max_ulp, e64, edf, integer_dtype, row_ok});
  }

  for (const auto& r : rows) {
    std::printf("  %-38s %8zu %9.3f%% %7lld %12.3e %12.3e %6s\n",
                r.name.c_str(), r.n, r.differ_pct, (long long)r.max_ulp, r.e64,
                r.edf, r.row_ok ? "ok" : "FAIL");
  }

  // Greedy streams: agreement is exact token identity per step.
  std::printf("\n  %-28s %8s %8s %10s %10s\n", "stream", "steps", "differ",
              "agree%", "first");
  for (const auto& [key, av] : f64.streams) {
    const auto it = df32.streams.find(key);
    if (it == df32.streams.end()) continue;
    const auto& bv = it->second;
    const size_t n = std::min(av.size(), bv.size());
    size_t differ = 0;
    size_t first = SIZE_MAX;
    for (size_t i = 0; i < n; ++i) {
      if (av[i] != bv[i]) {
        ++differ;
        if (first == SIZE_MAX) first = i;
      }
    }
    const double agree = n ? 100.0 * double(n - differ) / double(n) : 100.0;
    const bool is_cuda = key.rfind("cuda:", 0) == 0;
    std::printf("  %-28s %8zu %8zu %9.3f%% %10s\n", key.c_str(), n, differ,
                agree, first == SIZE_MAX ? "-" : std::to_string(first).c_str());
    if (is_cuda && agree < 99.9) ok = false;
  }

  // Top-1 over the 50 prompts.
  std::printf("\n  %-28s %8s %8s %10s\n", "top1", "prompts", "differ",
              "agree%");
  std::map<std::string, std::pair<size_t, size_t>> agg;  // key-prefix -> (n, diff)
  for (const auto& [key, av] : f64.top1) {
    const auto it = df32.top1.find(key);
    if (it == df32.top1.end()) continue;
    const size_t colon = key.find(':', 4);
    const std::string group = key.substr(0, colon);
    auto& [n, diff] = agg[group];
    ++n;
    if (av != it->second) ++diff;
  }
  for (const auto& [group, nd] : agg) {
    const auto [n, diff] = nd;
    const double agree = n ? 100.0 * double(n - diff) / double(n) : 100.0;
    std::printf("  %-28s %8zu %8zu %9.3f%%\n", group.c_str(), n, diff, agree);
    if (group.rfind("cuda:", 0) == 0 && agree < 99.9) ok = false;
  }

  std::printf("\n  DF32 %s\n", ok ? "ACCEPTED" : "NOT ACCEPTED");
  return ok;
}

// Fold one model's dump into the aggregate. The probes are mode-independent
// inputs produced by every child; the first copy is the one reported.
void merge_dump(Dump& dst, Dump&& src) {
  if (dst.probes.empty()) dst.probes = std::move(src.probes);
  if (!src.mode.empty()) dst.mode = std::move(src.mode);
  for (auto& [k, v] : src.streams) dst.streams[k] = std::move(v);
  for (auto& [k, v] : src.top1) dst.top1[k] = v;
}

int dump_mode(const std::string& out_path, const std::string& only) {
  Dump d;
  if (auto m = sonicboom::planner::reduce_mode_env()) {
    d.mode = sonicboom::compiler::reduce_mode_name(*m);
  } else {
    d.mode = "auto";
  }
  std::printf("  mode: %s\n", d.mode.c_str());
  cpu_probes(d);
  cuda_probes(d);
  run_streams(d, only);
  if (!write_dump(d, out_path)) {
    std::fprintf(stderr, "sb-numerics: cannot write %s\n", out_path.c_str());
    return 2;
  }
  std::printf("  wrote %s\n", out_path.c_str());
  return 0;
}

bool run_child_mode(const std::string& self, const char* mode,
                    const std::string& out, const char* model) {
  std::string cmd = "LC_ALL=C SONICBOOM_REDUCE=";
  cmd += mode;
  cmd += " '";
  cmd += self;
  cmd += "' dump '";
  cmd += out;
  cmd += "' '";
  cmd += model;
  cmd += "'";
  std::printf("  running %s mode...\n", mode);
  std::fflush(stdout);
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::fprintf(stderr, "sb-numerics: %s run failed (%d)\n", mode, rc);
    return false;
  }
  return true;
}

// Print the comparison, record the verdict for this device and this host, and
// return the process exit code (0 accepted, 1 rejected, 2 could not measure).
int finish(Dump& f64, Dump& df32, bool any_model) {
  if (f64.probes.empty() && !any_model) {
    std::cout << "  (skipped: no models found)\n";
    std::cout << "sb-numerics OK\n";
    return 0;
  }
  const bool ok = compare_and_report(f64, df32);
  if (any_model) {
    std::string err;
    if (!sonicboom::planner::record_df32_acceptance(ok, &err))
      std::fprintf(stderr, "sb-numerics: %s\n", err.c_str());
    else
      std::printf("  recorded the DF32 verdict for %s (auto now selects %s)\n",
                  sonicboom::planner::device_identity().c_str(),
                  sonicboom::compiler::reduce_mode_name(
                      sonicboom::planner::device_reduce_mode()));
  }
  return ok ? 0 : 1;
}

// --- numerics-profile acceptance gate (boom-r4 M1) -------------------------

struct ProfileModel {
  const char* env;
  const char* rel;
  const char* tag;
};

constexpr ProfileModel kProfileModels[] = {
    {"SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf", "gemma"},
    {"SONICBOOM_QWEN_GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen"},
    {"SONICBOOM_DEEPSEEK_GGUF", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
     "deepseek"},
};

// The gate's thresholds and its fixed corpus are internal engineering data, so
// they live outside this repository. SONICBOOM_PROFILE_CONFIG names a small
// key=value file (blank lines and '#' comments ignored):
//
//   corpus=/path/to/corpus.txt       scored text; no default path, and the ppl
//                                    row is skipped when it is not set
//   token_budget=24000               scored token span (gate wants >= 20k)
//   ppl_max_delta_pct=0.5            strict-vs-standard relative ppl gate
//   kl_mean_max=5e-4                 mean KL(standard || strict) over every
//                                    scored corpus position
//   same_top_min=0.99                same-top-token rate over every position
//   trace_top_k=512                  per-position head exported for the KL
//   trace_top_k=512                  per-position head exported for KL
//   ppl_path=prefill                corpus path; "prefill" or "decode". The
//                                   decode pass (first 8192 tokens, n_ctx 8192)
//                                   always runs as a second pass; the gate
//                                   requires both paths.
//   gate=profile                    "profile" (strict vs standard) or "arm"
//                                    (standard baseline vs standard + pin,
//                                    envs read from strict_env/standard_env)
//   models=deepseek                 optional comma list; empty = all models
//   top1_min_prompts=50              prompts compared for top-1 (informational
//                                    since gate v2)
//   top1_min=0.99                    required top-1 agreement (informational)
//
// The defaults below are the published gate. SONICBOOM_PPL_CORPUS overrides the
// corpus for one-off runs and SONICBOOM_PPL_TOKENS the span; the recorded gate
// numbers always use the defaults.
struct ProfileGateConfig {
  std::string corpus;
  uint64_t token_budget = 24000;
  double ppl_max_delta_pct = 0.5;
  double kl_mean_max = 5e-4;
  double same_top_min = 0.99;
  size_t trace_top_k = 512;
  // Corpus scoring path: "prefill" (default; every corpus column through the
  // chunked prefill segments, so prefill-only kernels are visible) or "decode"
  // (the per-token decode graph, the pre-boom-r5 gate's path).
  std::string ppl_path = "prefill";
  size_t top1_min_prompts = 50;
  double top1_min = 0.99;
  // "profile" (default): strict vs standard, the profile acceptance gate.
  // "arm": standard baseline vs standard + an arm pin, both under the standard
  // profile. The reference child runs with `strict_env` (baseline pins) and the
  // tested child with `standard_env` (baseline pins + the arm), so a parked arm
  // (TF32 dense, f32 tree reduce, df32, d128 prefill attention) is judged by
  // the same gate-v2 metrics against the standard default it would replace.
  std::string gate = "profile";
  // Optional comma-separated model tags for an arm run (e.g. "deepseek" when
  // the arm's switch only exists on one architecture). Empty = every model.
  std::string models;
  // Extra environment for each profile's child, e.g. a pinned arm under test:
  //   standard_env=SONICBOOM_ATTN_D128=codegen SONICBOOM_TUNE_DIR=/tmp/arm-a
  // Set strict_env too when the parent environment could leak an arm into the
  // strict side; the gate then pins both explicitly.
  std::string strict_env;
  std::string standard_env;
};

ProfileGateConfig load_profile_config() {
  ProfileGateConfig cfg;
  const char* path = std::getenv("SONICBOOM_PROFILE_CONFIG");
  if (path == nullptr || *path == '\0') return cfg;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "sb-numerics: cannot read profile config %s\n", path);
    return cfg;
  }
  std::string line;
  while (std::getline(in, line)) {
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq);
    const std::string val = line.substr(eq + 1);
    if (key == "corpus") {
      cfg.corpus = val;
    } else if (key == "token_budget") {
      const long long v = std::atoll(val.c_str());
      if (v > 0) cfg.token_budget = uint64_t(v);
    } else if (key == "ppl_max_delta_pct") {
      const double v = std::atof(val.c_str());
      if (v > 0.0) cfg.ppl_max_delta_pct = v;
    } else if (key == "kl_mean_max") {
      const double v = std::atof(val.c_str());
      if (v > 0.0) cfg.kl_mean_max = v;
    } else if (key == "same_top_min") {
      const double v = std::atof(val.c_str());
      if (v > 0.0) cfg.same_top_min = v;
    } else if (key == "models") {
      cfg.models = val;
    } else if (key == "gate") {
      if (val == "profile" || val == "arm") cfg.gate = val;
    } else if (key == "ppl_path") {
      if (val == "prefill" || val == "decode") cfg.ppl_path = val;
    } else if (key == "trace_top_k") {
      const long v = std::atol(val.c_str());
      if (v > 0) cfg.trace_top_k = size_t(v);
    } else if (key == "top1_min_prompts") {
      const long v = std::atol(val.c_str());
      if (v > 0) cfg.top1_min_prompts = size_t(v);
    } else if (key == "top1_min") {
      const double v = std::atof(val.c_str());
      if (v > 0.0) cfg.top1_min = v;
    } else if (key == "strict_env") {
      cfg.strict_env = val;
    } else if (key == "standard_env") {
      cfg.standard_env = val;
    } else {
      std::fprintf(stderr, "sb-numerics: profile config: unknown key '%s'\n",
                   key.c_str());
    }
  }
  return cfg;
}

const ProfileGateConfig& profile_config() {
  static const ProfileGateConfig cfg = load_profile_config();
  return cfg;
}

uint64_t ppl_token_budget() {
  if (const char* e = std::getenv("SONICBOOM_PPL_TOKENS")) {
    const long v = std::atol(e);
    if (v > 0) return uint64_t(v);
  }
  return profile_config().token_budget;
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

std::string profile_corpus_text() {
  std::vector<std::string> candidates;
  if (const char* e = std::getenv("SONICBOOM_PPL_CORPUS"))
    candidates.push_back(e);
  const std::string& configured = profile_config().corpus;
  if (!configured.empty()) candidates.push_back(configured);
  for (const auto& path : candidates) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(path, ec))
      return read_text_file(path);
  }
  return {};
}

struct ProfileResult {
  std::string model;
  std::string profile;
  double sum_nll = 0;
  uint64_t n_scored = 0;
  // Second pass: the first 8192 corpus tokens through the per-token decode
  // graph at n_ctx 8192 (gate v3; the gate requires both passes).
  double sum_nll_decode = 0;
  uint64_t n_scored_decode = 0;
  std::map<int, uint64_t> top1;
  std::array<std::vector<int64_t>, 3> runs;
  std::vector<int64_t> reuse_full;
  std::vector<int64_t> reuse_session;
};

// Binary per-position distribution trace (`sb-numerics profile-dump` writes one
// per profile). Layout: 8-byte magic, u64 vocab, u64 rows, u64 k, then rows
// u32 targets, rows u32 argmaxes, rows f32 logsumexps, rows*k u32 ids and
// rows*k f32 logits. The head is descending; every unlisted logit is <= the
// k-th entry.
constexpr char kTraceMagic[8] = {'S', 'B', 'T', 'R', 'A', '1', '0', '0'};

bool write_score_trace(const sbmodel::ScoreTrace& t, const std::string& path) {
  const uint64_t rows = t.target.size();
  const uint64_t k = t.top_k;
  if (k == 0 || t.top.size() != rows || t.log_z.size() != rows ||
      t.ids.size() != rows * k || t.logits.size() != rows * k)
    return false;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(kTraceMagic, sizeof kTraceMagic);
  out.write(reinterpret_cast<const char*>(&t.vocab), sizeof t.vocab);
  out.write(reinterpret_cast<const char*>(&rows), sizeof rows);
  out.write(reinterpret_cast<const char*>(&k), sizeof k);
  out.write(reinterpret_cast<const char*>(t.target.data()),
            std::streamsize(rows * sizeof(uint32_t)));
  out.write(reinterpret_cast<const char*>(t.top.data()),
            std::streamsize(rows * sizeof(uint32_t)));
  out.write(reinterpret_cast<const char*>(t.log_z.data()),
            std::streamsize(rows * sizeof(float)));
  out.write(reinterpret_cast<const char*>(t.ids.data()),
            std::streamsize(rows * k * sizeof(uint32_t)));
  out.write(reinterpret_cast<const char*>(t.logits.data()),
            std::streamsize(rows * k * sizeof(float)));
  return bool(out);
}

bool read_score_trace(const std::string& path, sbmodel::ScoreTrace& t) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  char magic[8];
  uint64_t rows = 0, k = 0;
  in.read(magic, sizeof magic);
  if (!in || std::memcmp(magic, kTraceMagic, sizeof magic) != 0) return false;
  in.read(reinterpret_cast<char*>(&t.vocab), sizeof t.vocab);
  in.read(reinterpret_cast<char*>(&rows), sizeof rows);
  in.read(reinterpret_cast<char*>(&k), sizeof k);
  // Bound the allocation: a trace cannot exceed the model's vocabulary positions
  // and the gate's own head size by a wide margin.
  if (!in || rows == 0 || k == 0 || k > 1u << 20 || rows > (1u << 26))
    return false;
  t.top_k = size_t(k);
  t.target.resize(rows);
  t.top.resize(rows);
  t.log_z.resize(rows);
  t.ids.resize(rows * k);
  t.logits.resize(rows * k);
  auto read = [&](void* p, size_t bytes) {
    in.read(reinterpret_cast<char*>(p), std::streamsize(bytes));
    return bool(in);
  };
  if (!read(t.target.data(), rows * sizeof(uint32_t)) ||
      !read(t.top.data(), rows * sizeof(uint32_t)) ||
      !read(t.log_z.data(), rows * sizeof(float)) ||
      !read(t.ids.data(), rows * k * sizeof(uint32_t)) ||
      !read(t.logits.data(), rows * k * sizeof(float)))
    return false;
  return true;
}

bool write_profile_dump(const ProfileResult& r, const std::string& path) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) return false;
  auto vec = [&](const char* key, const std::vector<int64_t>& v) {
    out << key;
    for (int64_t t : v) out << ' ' << t;
    out << '\n';
  };
  out << "profile " << r.profile << '\n';
  out << "model " << r.model << '\n';
  out << "ppl " << std::setprecision(17) << r.sum_nll << ' ' << r.n_scored
      << '\n';
  out << "ppl_decode " << std::setprecision(17) << r.sum_nll_decode << ' '
      << r.n_scored_decode << '\n';
  for (const auto& [i, t] : r.top1) out << "top1 " << i << ' ' << t << '\n';
  for (int i = 0; i < 3; ++i) vec(("run" + std::to_string(i)).c_str(),
                                  r.runs[i]);
  vec("reuse_full", r.reuse_full);
  vec("reuse_session", r.reuse_session);
  return bool(out);
}

bool read_profile_dump(const std::string& path, ProfileResult& r) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key.empty()) continue;
    if (key == "profile") {
      ss >> r.profile;
    } else if (key == "model") {
      ss >> r.model;
    } else if (key == "ppl") {
      ss >> r.sum_nll >> r.n_scored;
    } else if (key == "ppl_decode") {
      ss >> r.sum_nll_decode >> r.n_scored_decode;
    } else if (key == "top1") {
      int i = 0;
      uint64_t t = 0;
      if (ss >> i >> t) r.top1[i] = t;
    } else if (key.rfind("run", 0) == 0) {
      const int i = key.size() > 3 ? std::atoi(key.c_str() + 3) : -1;
      if (i < 0 || i > 2) return false;
      int64_t t = 0;
      while (ss >> t) r.runs[size_t(i)].push_back(t);
    } else {
      std::vector<int64_t>* v = key == "reuse_full"    ? &r.reuse_full
                                : key == "reuse_session" ? &r.reuse_session
                                                         : nullptr;
      if (v == nullptr) return false;
      int64_t t = 0;
      while (ss >> t) v->push_back(t);
    }
  }
  return !r.profile.empty() && !r.model.empty();
}

// Run one model under this process's profile and fill `r`. When `trace_path`
// is non-empty the corpus scoring also writes the per-position distribution
// head there (gate v2's KL input); `decode_trace_path` is the same for the
// gate's decode pass.
bool run_profile_model(const std::string& path, const char* tag, bool cuda,
                       ProfileResult& r, const std::string& trace_path,
                       const std::string& decode_trace_path) {
  Loaded l;
  if (!load(path, tag, l)) return false;
  r.model = tag;
  r.profile = sonicboom::numerics_profile_name(
      sonicboom::numerics_profile());

  // Perplexity: the fixed corpus, teacher-forced through the CUDA spine. The
  // CPU oracle is not used: the profile's kernel choices are on the device.
  if (cuda) {
    const std::string text = profile_corpus_text();
    if (!text.empty()) {
      auto ids = encode(l, text);
      const uint64_t budget = ppl_token_budget();
      if (ids.size() > budget) ids.resize(budget);
      std::printf("  ppl corpus: %zu tokens\n", ids.size());
      std::fflush(stdout);
      bool ok = false;
      if (ids.size() >= 1000 && !trace_path.empty()) {
        const char* path_env = std::getenv("SONICBOOM_PPL_PATH");
        const bool prefill =
            !(path_env && std::string_view(path_env) == "decode");
        std::printf("  ppl path: %s\n", prefill ? "prefill" : "decode");
        std::fflush(stdout);
        sbmodel::ScoreTrace trace;
        ok = prefill ? sbmodel::score_spine_cuda_prefill_trace(
                           l.model, ids, ids.size(),
                           profile_config().trace_top_k, trace, r.sum_nll,
                           r.n_scored)
                     : sbmodel::score_spine_cuda_trace(
                           l.model, ids, ids.size(),
                           profile_config().trace_top_k, trace, r.sum_nll,
                           r.n_scored);
        if (ok && !write_score_trace(trace, trace_path)) {
          std::fprintf(stderr, "sb-numerics: cannot write trace %s\n",
                       trace_path.c_str());
          ok = false;
        }
      } else if (ids.size() >= 1000) {
        ok = sbmodel::score_spine_cuda(l.model, ids, ids.size(), r.sum_nll,
                                       r.n_scored);
      }
      if (ok)
        std::printf("  ppl: sum_nll=%.3f over %llu tokens\n", r.sum_nll,
                    (unsigned long long)r.n_scored);
      else
        std::printf("  ppl: scoring failed\n");
      std::fflush(stdout);

      // Gate v3: the same first tokens through the per-token decode graph at
      // n_ctx 8192. The pass above exercises the prefill kernels; this one
      // exercises the decode kernels, and the gate requires both.
      constexpr uint64_t kDecodeSpan = 8192;
      constexpr uint64_t kDecodeCtx = 8192;
      const size_t dn = std::min<size_t>(ids.size(), kDecodeSpan);
      std::printf("  decode corpus: %zu tokens, n_ctx %llu\n", dn,
                  (unsigned long long)kDecodeCtx);
      std::fflush(stdout);
      bool decode_ok = false;
      if (dn >= 2 && !decode_trace_path.empty()) {
        sbmodel::ScoreTrace dtrace;
        decode_ok = sbmodel::score_spine_cuda_trace(
            l.model, std::span<const uint64_t>(ids.data(), dn), kDecodeCtx,
            profile_config().trace_top_k, dtrace, r.sum_nll_decode,
            r.n_scored_decode);
        if (decode_ok && !write_score_trace(dtrace, decode_trace_path)) {
          std::fprintf(stderr, "sb-numerics: cannot write trace %s\n",
                       decode_trace_path.c_str());
          decode_ok = false;
        }
      } else if (dn >= 2) {
        decode_ok = sbmodel::score_spine_cuda(
            l.model, std::span<const uint64_t>(ids.data(), dn), kDecodeCtx,
            r.sum_nll_decode, r.n_scored_decode);
      }
      if (decode_ok)
        std::printf("  decode ppl: sum_nll=%.3f over %llu tokens\n",
                    r.sum_nll_decode,
                    (unsigned long long)r.n_scored_decode);
      else
        std::printf("  decode ppl: scoring failed\n");
      std::fflush(stdout);
    }
  }

  // Top-1 over the gate's prompt count (first greedy token).
  const size_t n_top1_prompts = profile_config().top1_min_prompts;
  for (size_t i = 0; i < n_top1_prompts; ++i) {
    const std::string text = std::string(kPrompts[i % 3]) + " (" +
                             std::to_string(i) + ")";
    auto prompt = encode(l, text);
    if (prompt.empty()) continue;
    std::vector<int64_t> one(1, -1);
    const uint64_t n = cuda ? sbmodel::generate_prefill_spine_cuda(
                                  l.model, prompt, 0, one, 0, true)
                            : sbmodel::generate_prefill_spine(
                                  l.model, prompt, 0, one, 0, true);
    if (n != 0) r.top1[i] = uint64_t(one[0]);
  }

  // 3-run greedy determinism on one fixed prompt.
  auto prompt = encode(l, kPrompts[2]);
  if (!prompt.empty()) {
    for (int run = 0; run < 3; ++run) {
      std::vector<int64_t> toks(64, -1);
      const uint64_t n = cuda ? sbmodel::generate_prefill_spine_cuda(
                                    l.model, prompt, 0, toks, 0, true)
                              : sbmodel::generate_prefill_spine(
                                    l.model, prompt, 0, toks, 0, true);
      toks.resize(n);
      r.runs[size_t(run)] = std::move(toks);
    }
  }

  // Prefix reuse vs full prefill, on the CUDA session path. Both paths run at
  // the same cache capacity: the decode split-KV chunk is sized from the
  // graph's fixed window (`lc.sliding_window`, else `n_ctx`), so two different
  // capacities split the same key range differently and change the reduction
  // association. This probe is about the cache *contents* surviving reuse, not
  // about capacity invariance, so it holds the capacity fixed.
  if (cuda) {
    auto ps = encode(l, "Explain in one sentence why the sky is blue.");
    if (ps.size() > 8) ps.resize(8);
    auto full = ps;
    for (int i = 0; i < 6; ++i) full.push_back(ps[size_t(i) % ps.size()]);
    constexpr uint64_t kReuseCtx = 128;
    std::vector<int64_t> direct(16, -1);
    const uint64_t nd = sbmodel::generate_prefill_spine_cuda(
        l.model, full, 0, direct, kReuseCtx, true);
    direct.resize(nd);
    sonicboom::planner::KvCacheSession session;
    session.n_ctx = kReuseCtx;
    std::vector<int64_t> da(1, -1), reused(16, -1);
    sbmodel::generate_prefill_spine_cuda_prefix(l.model, ps, 0, da, session,
                                                true);
    const uint64_t nr = sbmodel::generate_prefill_spine_cuda_prefix(
        l.model, full, 0, reused, session, true);
    reused.resize(nr);
    r.reuse_full = std::move(direct);
    r.reuse_session = std::move(reused);
  }
  return true;
}

std::string vec_join(const std::vector<int64_t>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ',';
    s += std::to_string(v[i]);
  }
  return s;
}

// First index at which two token streams differ (== size when they agree).
size_t first_divergence(const std::vector<int64_t>& a,
                        const std::vector<int64_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return i;
  return n;
}

// Per-position KL(standard || strict) from the two exported heads. The strict
// head is converted to a lookup; a standard head id that strict did not export
// gets strict's k-th logit, the largest value any unexported strict logit can
// have, so every such term is a lower bound on the true KL (reported with the
// standard head mass, which bounds what was omitted).
struct KlSummary {
  bool ok = false;
  size_t n = 0;
  double mean = 0, median = 0, p99 = 0, p999 = 0, max = 0;
  double min_head_mass = 1.0;
  size_t top_mismatch = 0;
};

KlSummary compare_traces(const sbmodel::ScoreTrace& strict,
                         const sbmodel::ScoreTrace& standard) {
  KlSummary s;
  if (strict.top_k == 0 || standard.top_k == 0 ||
      strict.target.size() != standard.target.size() ||
      strict.top_k != standard.top_k)
    return s;
  const size_t n = standard.target.size();
  const size_t k = standard.top_k;
  std::vector<double> kls;
  kls.reserve(n);
  std::unordered_map<uint32_t, float> strict_head;
  strict_head.reserve(k * 2);
  for (size_t row = 0; row < n; ++row) {
    if (strict.target[row] != standard.target[row]) return s;
    strict_head.clear();
    const size_t bo = row * k;
    for (size_t i = 0; i < k; ++i)
      strict_head[strict.ids[bo + i]] = strict.logits[bo + i];
    const float strict_floor = strict.logits[bo + k - 1];
    const double log_zs = strict.log_z[row];
    const double log_zt = standard.log_z[row];
    double kl = 0;
    double mass = 0;
    for (size_t i = 0; i < k; ++i) {
      const double lt = standard.logits[bo + i];
      const double pt = std::exp(lt - log_zt);
      auto it = strict_head.find(standard.ids[bo + i]);
      const double ls = (it != strict_head.end()) ? it->second : strict_floor;
      kl += pt * ((lt - log_zt) - (ls - log_zs));
      mass += pt;
    }
    kls.push_back(std::max(0.0, kl));
    s.min_head_mass = std::min(s.min_head_mass, mass);
    if (strict.top[row] != standard.top[row]) ++s.top_mismatch;
  }
  if (kls.empty()) return s;
  std::sort(kls.begin(), kls.end());
  auto q = [&](double p) {
    const size_t i = std::min(kls.size() - 1,
                              size_t(p * double(kls.size() - 1) + 0.5));
    return kls[i];
  };
  double sum = 0;
  for (double v : kls) sum += v;
  s.n = kls.size();
  s.mean = sum / double(kls.size());
  s.median = q(0.5);
  s.p99 = q(0.99);
  s.p999 = q(0.999);
  s.max = kls.back();
  s.ok = true;
  return s;
}

// Compare a strict and a standard dump for one model; returns the gate verdict
// (and prints one row per scoring path). Gate v3 (boom-r6) judges the standard
// profile on both the corpus/prefill pass and the decode pass: relative ppl
// delta, mean KL over every scored position and the same-top-token rate, each
// with the same thresholds. The old 50-prompt top-1 is printed for information
// only. Determinism, prefix-reuse identity and KV-capacity independence remain
// hard; prefill-vs-decode identity is not guaranteed.
bool compare_profile_pair(const ProfileResult& strict,
                          const ProfileResult& standard,
                          const sbmodel::ScoreTrace& strict_trace,
                          const sbmodel::ScoreTrace& standard_trace,
                          const sbmodel::ScoreTrace& strict_decode_trace,
                          const sbmodel::ScoreTrace& standard_decode_trace,
                          bool arm = false) {
  const auto& cfg = profile_config();

  // The gate-v2 metrics for one scoring path (corpus pass or decode pass).
  struct PathGate {
    double ps = 0, pt = 0, delta = 0;
    bool ppl_ok = false;
    KlSummary kl;
    double same_top = 0;
    bool kl_ok = false, same_top_ok = false;
    bool ok() const { return ppl_ok && kl_ok && same_top_ok; }
  };
  const auto judge = [&](double s_nll, uint64_t s_n, double t_nll,
                         uint64_t t_n, const sbmodel::ScoreTrace& s_trace,
                         const sbmodel::ScoreTrace& t_trace) {
    PathGate g;
    g.ps = s_n == 0 ? 0.0 : std::exp(s_nll / double(s_n));
    g.pt = t_n == 0 ? 0.0 : std::exp(t_nll / double(t_n));
    g.ppl_ok = g.ps > 0.0 &&
               std::abs(g.pt - g.ps) / g.ps <= cfg.ppl_max_delta_pct / 100.0;
    g.delta = g.ps > 0 ? 100.0 * (g.pt - g.ps) / g.ps : 0.0;
    g.kl = compare_traces(s_trace, t_trace);
    g.same_top = g.kl.ok && g.kl.n
                     ? double(g.kl.n - g.kl.top_mismatch) / double(g.kl.n)
                     : 0.0;
    g.kl_ok = g.kl.ok && g.kl.n > 0 && g.kl.mean <= cfg.kl_mean_max;
    g.same_top_ok = g.kl.ok && g.same_top >= cfg.same_top_min;
    return g;
  };

  const PathGate first = judge(strict.sum_nll, strict.n_scored,
                               standard.sum_nll, standard.n_scored,
                               strict_trace, standard_trace);
  const PathGate decode =
      judge(strict.sum_nll_decode, strict.n_scored_decode,
            standard.sum_nll_decode, standard.n_scored_decode,
            strict_decode_trace, standard_decode_trace);

  size_t agree = 0, total = 0;
  for (const auto& [i, t] : strict.top1) {
    auto it = standard.top1.find(i);
    if (it == standard.top1.end()) continue;
    ++total;
    if (it->second == t) ++agree;
  }
  const double top1 = total ? double(agree) / double(total) : 0.0;

  const bool det_ok = strict.runs[0] == strict.runs[1] &&
                      strict.runs[0] == strict.runs[2] &&
                      standard.runs[0] == standard.runs[1] &&
                      standard.runs[0] == standard.runs[2];
  const bool reuse_ok = strict.reuse_full == strict.reuse_session &&
                        standard.reuse_full == standard.reuse_session;
  const size_t div = first_divergence(strict.runs[0], standard.runs[0]);

  const char* role_a = arm ? "base" : "strict";
  const char* role_b = arm ? "arm" : "standard";
  const auto print_path = [&](const char* path, const PathGate& g,
                              const char* trailer) {
    std::printf("  %-8s %-8s ppl %s=%.4f %s=%.4f delta=%+.3f%% %s | "
                "KL mean=%.3e med=%.3e p99=%.3e p99.9=%.3e max=%.3e %s "
                "(head mass min %.9f) | same-top %.4f%% %s%s\n",
                strict.model.c_str(), path, role_a, g.ps, role_b, g.pt,
                g.delta, g.ppl_ok ? "OK" : "FAIL", g.kl.mean, g.kl.median,
                g.kl.p99, g.kl.p999, g.kl.max, g.kl_ok ? "OK" : "FAIL",
                g.kl.min_head_mass, 100.0 * g.same_top,
                g.same_top_ok ? "OK" : "FAIL", trailer);
  };
  char first_trailer[160];
  std::snprintf(first_trailer, sizeof first_trailer,
                " | det %s reuse %s | top1(info) %zu/%zu %.1f%% | "
                "first divergence step %zu",
                det_ok ? "OK" : "FAIL", reuse_ok ? "OK" : "FAIL", agree, total,
                total ? 100.0 * top1 : 0.0, div);
  char decode_trailer[64];
  std::snprintf(decode_trailer, sizeof decode_trailer, " (%llu tokens)",
                (unsigned long long)standard.n_scored_decode);
  print_path(cfg.ppl_path.c_str(), first, first_trailer);
  print_path("decode", decode, decode_trailer);
  std::fflush(stdout);
  return first.ok() && decode.ok() && det_ok && reuse_ok;
}

int profile_dump_mode(const std::string& out_path, const std::string& model,
                      const std::string& trace_path,
                      const std::string& decode_trace_path) {
  const ProfileModel* spec = nullptr;
  for (const auto& s : kProfileModels)
    if (model == s.tag) spec = &s;
  if (spec == nullptr) {
    std::fprintf(stderr, "sb-numerics: unknown model '%s'\n", model.c_str());
    return 2;
  }
  const std::string path = find_model(spec->env, spec->rel);
  if (path.empty()) {
    std::printf("  (skipped: %s model not found)\n", spec->tag);
    return 0;
  }
  ProfileResult r;
  const bool cuda = have_cuda();
  if (!run_profile_model(path, spec->tag, cuda, r, trace_path,
                         decode_trace_path)) {
    std::fprintf(stderr, "sb-numerics: %s failed to load/run\n", spec->tag);
    return 2;
  }
  if (!write_profile_dump(r, out_path)) {
    std::fprintf(stderr, "sb-numerics: cannot write %s\n", out_path.c_str());
    return 2;
  }
  std::printf("  wrote %s\n", out_path.c_str());
  return 0;
}

bool run_profile_child(const std::string& self, const char* profile,
                       const std::string& out, const char* model,
                       const std::string& trace,
                       const std::string& decode_trace,
                       const std::string& extra_env) {
  std::string cmd = "LC_ALL=C SONICBOOM_NUMERICS=";
  cmd += profile;
  cmd += " SONICBOOM_PPL_PATH=";
  cmd += profile_config().ppl_path;
  if (!extra_env.empty()) {
    cmd += ' ';
    cmd += extra_env;
  }
  cmd += " '";
  cmd += self;
  cmd += "' profile-dump '";
  cmd += out;
  cmd += "' '";
  cmd += model;
  cmd += "'";
  if (!trace.empty()) {
    cmd += " '";
    cmd += trace;
    cmd += "'";
  }
  if (!decode_trace.empty()) {
    // The CLI needs the corpus trace argument to reach the decode one; the
    // caller always passes both, so an empty corpus trace here is a bug.
    if (trace.empty()) return false;
    cmd += " '";
    cmd += decode_trace;
    cmd += "'";
  }
  std::printf("  running %s profile for %s...\n", profile, model);
  std::fflush(stdout);
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::fprintf(stderr, "sb-numerics: %s/%s run failed (%d)\n", profile,
                 model, rc);
    return false;
  }
  return true;
}

int profile_gate(const std::string& self_in) {
  if (!have_cuda()) {
    std::cout << "  (skipped: no CUDA device; the profile gate is a device "
                 "measurement)\n";
    std::cout << "sb-numerics profile OK\n";
    return 0;
  }
  const std::string self = std::filesystem::absolute(self_in).string();
  bool ok = true;
  bool any = false;
  const bool arm = profile_config().gate == "arm";
  std::printf("profile gate (%s; prefill and decode)\n",
              arm ? "standard arm vs standard baseline" : "strict vs standard");
  for (const auto& spec : kProfileModels) {
    if (!profile_config().models.empty()) {
      const std::string& m = profile_config().models;
      const std::string tag = std::string(",") + spec.tag + ",";
      const std::string list = std::string(",") + m + ",";
      if (list.find(tag) == std::string::npos) continue;
    }
    if (find_model(spec.env, spec.rel).empty()) {
      std::printf("  (skipped: %s model not found)\n", spec.tag);
      continue;
    }
    const std::string s_path =
        std::string("/tmp/sb-numerics-profile-strict-") + spec.tag + ".dump";
    const std::string t_path =
        std::string("/tmp/sb-numerics-profile-standard-") + spec.tag + ".dump";
    const std::string ks_path =
        std::string("/tmp/sb-numerics-trace-strict-") + spec.tag + ".bin";
    const std::string kt_path =
        std::string("/tmp/sb-numerics-trace-standard-") + spec.tag + ".bin";
    const std::string ksd_path =
        std::string("/tmp/sb-numerics-trace-decode-strict-") + spec.tag +
        ".bin";
    const std::string ktd_path =
        std::string("/tmp/sb-numerics-trace-decode-standard-") + spec.tag +
        ".bin";
    if (!run_profile_child(self, arm ? "standard" : "strict", s_path, spec.tag,
                           ks_path, ksd_path, profile_config().strict_env))
      return 2;
    if (!run_profile_child(self, "standard", t_path, spec.tag, kt_path,
                           ktd_path, profile_config().standard_env))
      return 2;
    ProfileResult a, b;
    sbmodel::ScoreTrace ta, tb, tda, tdb;
    if (!read_profile_dump(s_path, a) || !read_profile_dump(t_path, b) ||
        !read_score_trace(ks_path, ta) || !read_score_trace(kt_path, tb) ||
        !read_score_trace(ksd_path, tda) || !read_score_trace(ktd_path, tdb)) {
      std::fprintf(stderr,
                   "sb-numerics: cannot read %s / %s / %s / %s / %s / %s\n",
                   s_path.c_str(), t_path.c_str(), ks_path.c_str(),
                   kt_path.c_str(), ksd_path.c_str(), ktd_path.c_str());
      return 2;
    }
    any = true;
    ok = compare_profile_pair(a, b, ta, tb, tda, tdb, arm) && ok;
  }
  if (!any) std::cout << "  (skipped: no models found)\n";
  std::printf("sb-numerics profile %s\n", ok ? "ACCEPTED" : "REJECTED");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string_view(argv[1]) == "dump") {
    if (argc < 3 || argc > 4) {
      std::fprintf(stderr, "usage: %s dump <file> [gemma|qwen]\n", argv[0]);
      return 2;
    }
    return dump_mode(argv[2], argc == 4 ? argv[3] : "");
  }
  if (argc >= 2 && std::string_view(argv[1]) == "profile") {
    if (argc != 2) {
      std::fprintf(stderr, "usage: %s profile\n", argv[0]);
      return 2;
    }
    return profile_gate(argv[0]);
  }
  if (argc >= 2 && std::string_view(argv[1]) == "profile-dump") {
    if (argc < 4 || argc > 6) {
      std::fprintf(stderr, "usage: %s profile-dump <file> "
                           "<gemma|qwen|deepseek> "
                           "[trace-file [decode-trace-file]]\n",
                   argv[0]);
      return 2;
    }
    return profile_dump_mode(argv[2], argv[3], argc >= 5 ? argv[4] : "",
                             argc >= 6 ? argv[5] : "");
  }
  if (argc >= 2 && std::string_view(argv[1]) == "compare") {
    // compare <f64.dump> <df32.dump> [<f64.dump> <df32.dump> ...]
    if (argc < 4 || (argc - 2) % 2 != 0) {
      std::fprintf(stderr, "usage: %s compare <f64.dump> <df32.dump> ...\n",
                   argv[0]);
      return 2;
    }
    Dump f64, df32;
    bool any_model = false;
    for (int i = 2; i + 1 < argc; i += 2) {
      Dump a, b;
      if (!read_dump(argv[i], a) || !read_dump(argv[i + 1], b)) {
        std::fprintf(stderr, "sb-numerics: cannot read %s / %s\n", argv[i],
                     argv[i + 1]);
        return 2;
      }
      if (!a.streams.empty() && !b.streams.empty()) any_model = true;
      merge_dump(f64, std::move(a));
      merge_dump(df32, std::move(b));
    }
    return finish(f64, df32, any_model);
  }
  if (argc != 1) {
    std::fprintf(stderr,
                 "usage: %s [dump <file> [model]] | compare <a> <b>... | "
                 "profile | "
                 "profile-dump <file> <model> [trace [decode-trace]]\n"
                 "  profile reads thresholds and corpus path from "
                 "SONICBOOM_PROFILE_CONFIG (key=value file)\n",
                 argv[0]);
    return 2;
  }

  const std::string self = std::filesystem::absolute(argv[0]).string();
  const std::string dir = "/tmp";
  Dump f64, df32;
  bool any_model = false;
  for (const char* model : {"gemma", "qwen"}) {
    const std::string f64_path =
        dir + "/sb-numerics-f64-" + model + ".dump";
    const std::string df32_path =
        dir + "/sb-numerics-df32-" + model + ".dump";
    if (!run_child_mode(self, "f64", f64_path, model)) return 2;
    if (!run_child_mode(self, "df32", df32_path, model)) return 2;
    Dump a, b;
    if (!read_dump(f64_path, a) || !read_dump(df32_path, b)) {
      std::fprintf(stderr, "sb-numerics: cannot read dumps\n");
      return 2;
    }
    if (!a.streams.empty() && !b.streams.empty()) any_model = true;
    merge_dump(f64, std::move(a));
    merge_dump(df32, std::move(b));
  }
  return finish(f64, df32, any_model);
}
