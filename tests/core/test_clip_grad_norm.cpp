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

// CAP1 gate: global-norm gradient clipping.
//
//  * the host reference against a hand-computed norm;
//  * no clipping leaves the buffers bit-for-bit unchanged (including the
//    norm == max_norm boundary);
//  * clipping brings the global norm to max_norm (+1e-5) under a double
//    accumulator;
//  * the C ABI sb_clip_grad_norm over real sb_tensor handles;
//  * when this build has CUDA, the fused device kernels (one accumulation
//    pass over every tensor, then one scale per tensor), with their cost.

#include <sonicboom/capi.h>
#include <sonicboom/tensor.h>
#include <sonicboom/training/clip_grad_norm.h>

#ifdef SB_TEST_CUDA
#include "../../core/src/training/device_graph_kernels.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace sbtrain = sonicboom::training;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

double norm_of(const std::vector<std::vector<float>>& ts) {
  double s = 0.0;
  for (const auto& t : ts)
    for (float v : t) s += double(v) * double(v);
  return std::sqrt(s);
}

std::vector<float*> pointers(std::vector<std::vector<float>>& ts) {
  std::vector<float*> p;
  p.reserve(ts.size());
  for (auto& t : ts) p.push_back(t.data());
  return p;
}

std::vector<std::size_t> sizes_of(const std::vector<std::vector<float>>& ts) {
  std::vector<std::size_t> s;
  s.reserve(ts.size());
  for (const auto& t : ts) s.push_back(t.size());
  return s;
}

bool same_bytes(const std::vector<std::vector<float>>& a,
                const std::vector<std::vector<float>>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) return false;
    if (!a[i].empty() &&
        std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) != 0)
      return false;
  }
  return true;
}

// Deterministic pseudo-random floats in [-1, 1].
std::vector<float> random_tensor(std::size_t n, std::uint64_t& state) {
  std::vector<float> out(n);
  for (auto& v : out) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    v = float(double(state >> 33) / double(1ull << 31) - 1.0);
  }
  return out;
}

void test_host_reference() {
  std::cout << "host reference\n";

  // [3,4] has norm 5, [12] has norm 12, the empty gradient contributes
  // nothing: the global norm is 13.
  std::vector<std::vector<float>> ts{{3.0f, 4.0f}, {12.0f}, {}};
  auto ptrs = pointers(ts);
  auto sizes = sizes_of(ts);
  const double norm = sbtrain::grad_global_norm(ptrs, sizes);
  check(std::abs(norm - 13.0) < 1e-12, "grad_global_norm = 13");

  // At the boundary the buffers must not be touched at all.
  auto before = ts;
  const double n_at = sbtrain::clip_grad_norm_inplace(ptrs, sizes, 13.0);
  check(std::abs(n_at - 13.0) < 1e-12, "clip at norm == max reports 13");
  check(same_bytes(before, ts), "norm <= max_norm leaves every byte unchanged");

  // Clipping to 5 scales by 5 / (13 + 1e-6); the clipped norm is just under 5.
  auto ptrs2 = pointers(ts);
  const double n_clip = sbtrain::clip_grad_norm_inplace(ptrs2, sizes, 5.0);
  const double after = norm_of(ts);
  check(std::abs(n_clip - 13.0) < 1e-12, "clip reports the pre-clip norm");
  check(after <= 5.0 + 1e-5, "clipped global norm <= max_norm + 1e-5");
  check(after > 5.0 - 1e-4, "clipped global norm is max_norm, not below it");

  // A larger random set: the double accumulator must agree with an
  // independent double sum, and clipping must land on the threshold.
  std::uint64_t state = 7;
  std::vector<std::vector<float>> rs;
  for (std::size_t t = 0; t < 6; ++t) rs.push_back(random_tensor(1000 + 137 * t, state));
  const double want = norm_of(rs);
  auto rp = pointers(rs);
  auto rsz = sizes_of(rs);
  const double got = sbtrain::grad_global_norm(rp, rsz);
  check(std::abs(got - want) <= 1e-9 * std::max(1.0, want),
        "random set norm matches the independent double sum");
  const double max_norm = 3.25;
  const double pre = sbtrain::clip_grad_norm_inplace(rp, rsz, max_norm);
  check(std::abs(pre - want) <= 1e-9 * std::max(1.0, want),
        "clip returns the independently computed norm");
  const double now = norm_of(rs);
  check(now <= max_norm + 1e-5 && now > max_norm - 1e-4,
        "random set clipped to max_norm");

  // max_norm = 0 is the degenerate case: no division by zero, all zeros.
  auto zp = pointers(rs);
  sbtrain::clip_grad_norm_inplace(zp, rsz, 0.0);
  check(norm_of(rs) == 0.0, "max_norm = 0 zeroes the gradients");
}

void test_c_abi() {
  std::cout << "C ABI sb_clip_grad_norm\n";
  const float a[] = {3.0f, 4.0f};
  const float b[] = {12.0f};
  sb_tensor* ta = nullptr;
  sb_tensor* tb = nullptr;
  sb_error* err = nullptr;
  const int64_t da[] = {2}, db[] = {1};
  check(sb_tensor_from_f32(da, 1, a, sizeof(a), &ta, &err) == SB_OK,
        "create gradient tensor a");
  check(sb_tensor_from_f32(db, 1, b, sizeof(b), &tb, &err) == SB_OK,
        "create gradient tensor b");
  if (!ta || !tb) {
    std::cerr << "  cannot continue without tensors\n";
    ++g_failures;
    return;
  }

  sb_tensor* grads[] = {ta, tb};
  float norm = -1.0f;
  check(sb_clip_grad_norm(grads, 2, 5.0f, &norm, &err) == SB_OK,
        "clip over two handles");
  check(std::abs(norm - 13.0f) < 1e-5f, "C ABI reports the pre-clip norm 13");

  float out[2] = {0.0f, 0.0f};
  size_t written = 0;
  check(sb_tensor_bytes(ta, out, sizeof(out), &written, &err) == SB_OK &&
            written == sizeof(out),
        "read back a");
  const float s = 5.0f / (13.0f + 1e-6f);
  check(std::abs(out[0] - 3.0f * s) < 1e-6f &&
            std::abs(out[1] - 4.0f * s) < 1e-6f,
        "a was scaled by max_norm / (norm + eps)");

  // The unchanged path through the C ABI: a fresh 3-4 tensor, norm 5, no clip
  // at max_norm 5.0 (the comparison is strict, so equality is no clip).
  sb_tensor* tc = nullptr;
  check(sb_tensor_from_f32(da, 1, a, sizeof(a), &tc, &err) == SB_OK,
        "create a fresh tensor");
  sb_tensor* one[] = {tc};
  check(sb_clip_grad_norm(one, 1, 5.0f, nullptr, &err) == SB_OK,
        "clip with a NULL norm_out");
  check(sb_tensor_bytes(tc, out, sizeof(out), &written, &err) == SB_OK,
        "read back the fresh tensor");
  check(out[0] == 3.0f && out[1] == 4.0f,
        "norm == max_norm leaves the C ABI tensor unchanged");

  sb_tensor_free(ta);
  sb_tensor_free(tb);
  sb_tensor_free(tc);
  if (err) sb_error_free(err);
}

#ifdef SB_TEST_CUDA
void test_device_kernels() {
  std::cout << "device kernels\n";
  if (!nt::cuda_available()) {
    std::cout << "  (no CUDA device: device case skipped)\n";
    return;
  }
  constexpr std::size_t kPer = 1 << 20;   // 1M floats (4 MiB) per gradient
  constexpr int kCount = 8;               // 32 MiB over the whole set
  std::uint64_t state = 11;
  std::vector<std::vector<float>> host;
  for (int i = 0; i < kCount; ++i) host.push_back(random_tensor(kPer, state));
  const double want = norm_of(host);

  using sonicboom::training::detail::DevBuf;
  std::vector<DevBuf> dbuf, obuf;
  for (int i = 0; i < kCount; ++i) {
    dbuf.push_back(sonicboom::training::detail::dev_alloc(kPer * sizeof(float)));
    obuf.push_back(sonicboom::training::detail::dev_alloc(kPer * sizeof(float)));
    check(dbuf.back() != 0 && obuf.back() != 0, "allocate device buffers");
    check(sonicboom::training::detail::dev_upload(
              dbuf.back(), host[i].data(), kPer * sizeof(float)),
          "upload a gradient");
  }
  const DevBuf acc = sonicboom::training::detail::dev_alloc(sizeof(double));
  check(acc != 0, "allocate the norm accumulator");
  check(sonicboom::training::detail::dev_zero(acc, sizeof(double)),
        "zero the accumulator");

  auto accumulate = [&]() {
    check(sonicboom::training::detail::dev_zero(acc, sizeof(double)),
          "zero the accumulator");
    bool ok = true;
    for (int i = 0; i < kCount; ++i)
      ok = ok && sonicboom::training::detail::graph_sumsq_acc(
                     acc, dbuf[static_cast<std::size_t>(i)], kPer);
    double sumsq = 0.0;
    ok = ok && sonicboom::training::detail::dev_sync() &&
         sonicboom::training::detail::dev_download(acc, &sumsq, sizeof(sumsq));
    check(ok, "accumulate the global sum of squares");
    return std::sqrt(sumsq);
  };

  const double got = accumulate();
  check(std::abs(got - want) <= 1e-5 * std::max(1.0, want),
        "device accumulation matches the host double sum");

  const double max_norm = 100.0;
  const double scale = got > max_norm ? max_norm / (got + 1e-6) : 1.0;
  bool ok = true;
  for (int i = 0; i < kCount; ++i)
    ok = ok && sonicboom::training::detail::graph_scale_copy(
                   obuf[static_cast<std::size_t>(i)],
                   dbuf[static_cast<std::size_t>(i)], kPer, scale);
  check(ok && sonicboom::training::detail::dev_sync(), "scale into results");

  // Read one result back and check it elementwise; then measure the full
  // accumulate + scale cost (what one clip_grad_norm node spends).
  std::vector<float> back(kPer);
  check(sonicboom::training::detail::dev_download(obuf[0], back.data(),
                                                  kPer * sizeof(float)),
        "download a result");
  bool exact = true;
  for (std::size_t i = 0; i < kPer && exact; ++i)
    exact = std::abs(back[i] - float(double(host[0][i]) * scale)) <= 1e-6f;
  check(exact, "scaled values match the host reference");

  constexpr int kIters = 20;
  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < kIters; ++it) {
    (void)accumulate();
    for (int i = 0; i < kCount; ++i)
      (void)sonicboom::training::detail::graph_scale_copy(
          obuf[static_cast<std::size_t>(i)],
          dbuf[static_cast<std::size_t>(i)], kPer, scale);
    (void)sonicboom::training::detail::dev_sync();
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
  std::printf("  clip kernel: %.1f us for %d x %.1f MiB (%.1f GB/s)\n", us,
              kCount, double(kPer * sizeof(float)) / (1 << 20),
              (2.0 * kCount * kPer * sizeof(float)) / (us * 1e3));

  for (int i = 0; i < kCount; ++i) {
    sonicboom::training::detail::dev_free(dbuf[static_cast<std::size_t>(i)]);
    sonicboom::training::detail::dev_free(obuf[static_cast<std::size_t>(i)]);
  }
  sonicboom::training::detail::dev_free(acc);
}
#else
void test_device_kernels() {
  std::cout << "  (built without CUDA: device case skipped)\n";
}
#endif

} // namespace

int main() {
  test_host_reference();
  test_c_abi();
  test_device_kernels();
  if (g_failures == 0) {
    std::cout << "test_clip_grad_norm OK\n";
    return 0;
  }
  std::cerr << "test_clip_grad_norm FAILED: " << g_failures << " check(s)\n";
  return 1;
}
