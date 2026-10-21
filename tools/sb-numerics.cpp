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
#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/vocab/vocab.h>

#include "../core/src/nn/cuda_resident.h"
#include "../core/src/planner/glue_jit.h"
#include "../core/src/planner/reduce_mode.h"
#include "../core/src/quant/cuda_resident.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
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

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string_view(argv[1]) == "dump") {
    if (argc < 3 || argc > 4) {
      std::fprintf(stderr, "usage: %s dump <file> [gemma|qwen]\n", argv[0]);
      return 2;
    }
    return dump_mode(argv[2], argc == 4 ? argv[3] : "");
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
    std::fprintf(stderr, "usage: %s [dump <file> [model]] | compare <a> <b>...\n",
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
