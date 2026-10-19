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

// test_backward_ops.cpp — R3: matmul_bwd, softmax_bwd, cross_entropy_bwd.
//
// Each op is checked twice:
//   1. numerics: the device kernel's float output against an independent fp64
//      reference implemented here, at the tiny model's shapes (seq 32,
//      hidden 64, vocab 256), max-abs <= 1e-5. The host reference is checked
//      against the same fp64 reference.
//   2. gradient check: central finite differences (h = 1e-4) of the fp64
//      forward objective, max relative error <= 1e-3, at small shapes so the
//      perturb-every-element loop stays fast.
//
// The reference implementations here are written independently of
// core/src/training/backward_ops.cpp (different loop order, direct formulas)
// so a shared mistake cannot make both sides agree.

#include <sonicboom/training/backward.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kSeq = 32;
constexpr std::size_t kHidden = 64;
constexpr std::size_t kVocab = 256;

struct Ctx {
  bool ok = true;
  const char* what = "";
};

void check(Ctx& c, bool cond, const char* what) {
  if (!cond) {
    c.ok = false;
    c.what = what;
  }
}

struct Rng {
  std::uint32_t s = 0x9e3779b9u;
  std::uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float uniform(float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(next() >> 8) /
                    static_cast<float>(1u << 24);
  }
};

std::vector<float> rnd(std::size_t n, float lo, float hi, Rng& rng) {
  std::vector<float> v(n);
  for (auto& x : v)
    x = rng.uniform(lo, hi);
  return v;
}

// --- independent fp64 references -------------------------------------------

struct MatmulRef {
  std::vector<double> dA, dB;
};

MatmulRef matmul_ref(const std::vector<float>& A, const std::vector<float>& B,
                     const std::vector<float>& dY, std::size_t M,
                     std::size_t K, std::size_t N) {
  MatmulRef r;
  r.dA.assign(M * K, 0.0);
  r.dB.assign(K * N, 0.0);
  // dA = dY * B^T: loop n outermost, a different order from the library's.
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t m = 0; m < M; ++m)
      for (std::size_t k = 0; k < K; ++k)
        r.dA[m * K + k] += static_cast<double>(dY[m * N + n]) * B[k * N + n];
  // dB = A^T * dY.
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t k = 0; k < K; ++k)
      for (std::size_t n = 0; n < N; ++n)
        r.dB[k * N + n] += static_cast<double>(A[m * K + k]) * dY[m * N + n];
  return r;
}

std::vector<double> softmax_rows(const std::vector<double>& x,
                                 std::size_t rows, std::size_t cols) {
  std::vector<double> y(x.size());
  for (std::size_t r = 0; r < rows; ++r) {
    double max = -INFINITY;
    for (std::size_t c = 0; c < cols; ++c)
      max = std::max(max, x[r * cols + c]);
    double sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c)
      sum += std::exp(x[r * cols + c] - max);
    for (std::size_t c = 0; c < cols; ++c)
      y[r * cols + c] = std::exp(x[r * cols + c] - max) / sum;
  }
  return y;
}

std::vector<double> softmax_bwd_ref(const std::vector<double>& y,
                                    const std::vector<float>& dY,
                                    std::size_t rows, std::size_t cols) {
  std::vector<double> dx(y.size());
  for (std::size_t r = 0; r < rows; ++r) {
    double s = 0.0;
    for (std::size_t c = 0; c < cols; ++c)
      s += static_cast<double>(dY[r * cols + c]) * y[r * cols + c];
    for (std::size_t c = 0; c < cols; ++c)
      dx[r * cols + c] =
          y[r * cols + c] * (static_cast<double>(dY[r * cols + c]) - s);
  }
  return dx;
}

double ce_loss_ref(const std::vector<double>& logits,
                   const std::vector<std::int32_t>& targets, std::size_t rows,
                   std::size_t cols) {
  double total = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    double max = -INFINITY;
    for (std::size_t c = 0; c < cols; ++c)
      max = std::max(max, logits[r * cols + c]);
    double sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c)
      sum += std::exp(logits[r * cols + c] - max);
    total += -(logits[r * cols + static_cast<std::size_t>(targets[r])] - max -
               std::log(sum));
  }
  return total / static_cast<double>(rows);
}

std::vector<double> ce_bwd_ref(const std::vector<double>& logits,
                               const std::vector<std::int32_t>& targets,
                               std::size_t rows, std::size_t cols) {
  std::vector<double> dl(logits.size());
  for (std::size_t r = 0; r < rows; ++r) {
    double max = -INFINITY;
    for (std::size_t c = 0; c < cols; ++c)
      max = std::max(max, logits[r * cols + c]);
    double sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c)
      sum += std::exp(logits[r * cols + c] - max);
    for (std::size_t c = 0; c < cols; ++c) {
      double p = std::exp(logits[r * cols + c] - max) / sum;
      if (c == static_cast<std::size_t>(targets[r]))
        p -= 1.0;
      dl[r * cols + c] = p / static_cast<double>(rows);
    }
  }
  return dl;
}

std::vector<double> as_double(const std::vector<float>& f) {
  return std::vector<double>(f.begin(), f.end());
}

// --- R4 references ----------------------------------------------------------

struct RmsRef {
  std::vector<double> dX, dW;
};

std::vector<double> rmsnorm_fwd_ref(const std::vector<double>& x,
                                    const std::vector<double>& w,
                                    std::size_t rows, std::size_t dim,
                                    double eps) {
  std::vector<double> y(x.size());
  for (std::size_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (std::size_t d = 0; d < dim; ++d)
      ss += x[r * dim + d] * x[r * dim + d];
    const double invariant = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
    for (std::size_t d = 0; d < dim; ++d)
      y[r * dim + d] = x[r * dim + d] * invariant * w[d];
  }
  return y;
}

RmsRef rmsnorm_bwd_ref(const std::vector<double>& x,
                       const std::vector<double>& w,
                       const std::vector<float>& dY, std::size_t rows,
                       std::size_t dim, double eps) {
  RmsRef r;
  r.dX.assign(x.size(), 0.0);
  r.dW.assign(dim, 0.0);
  for (std::size_t rr = 0; rr < rows; ++rr) {
    double ss = 0.0;
    for (std::size_t d = 0; d < dim; ++d)
      ss += x[rr * dim + d] * x[rr * dim + d];
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
    double s = 0.0;
    for (std::size_t d = 0; d < dim; ++d)
      s += w[d] * static_cast<double>(dY[rr * dim + d]) * x[rr * dim + d] * inv;
    s /= static_cast<double>(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      const double xhat = x[rr * dim + d] * inv;
      r.dX[rr * dim + d] =
          inv * w[d] * static_cast<double>(dY[rr * dim + d]) - inv * xhat * s;
      r.dW[d] += static_cast<double>(dY[rr * dim + d]) * xhat;
    }
  }
  return r;
}

std::vector<double> rope_fwd_ref(const std::vector<double>& x,
                                 std::size_t rows, std::size_t heads,
                                 std::size_t head_dim, std::uint64_t pos0,
                                 double theta_base, double freq_scale) {
  std::vector<double> y(x);
  const std::size_t half = head_dim / 2;
  for (std::size_t r = 0; r < rows; ++r) {
    const double pos = static_cast<double>(pos0 + r);
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t base = (r * heads + h) * head_dim;
      for (std::size_t i = 0; i < half; ++i) {
        const double angle =
            pos * freq_scale *
            std::pow(theta_base, -2.0 * static_cast<double>(i) / head_dim);
        const double c = std::cos(angle), s = std::sin(angle);
        const double a = x[base + i], b = x[base + i + half];
        y[base + i] = a * c - b * s;
        y[base + i + half] = a * s + b * c;
      }
    }
  }
  return y;
}

std::vector<double> rope_bwd_ref(const std::vector<float>& dY,
                                 std::size_t rows, std::size_t heads,
                                 std::size_t head_dim, std::uint64_t pos0,
                                 double theta_base, double freq_scale) {
  std::vector<double> dx(dY.size());
  const std::size_t half = head_dim / 2;
  for (std::size_t r = 0; r < rows; ++r) {
    const double pos = static_cast<double>(pos0 + r);
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t base = (r * heads + h) * head_dim;
      for (std::size_t i = 0; i < half; ++i) {
        const double angle =
            pos * freq_scale *
            std::pow(theta_base, -2.0 * static_cast<double>(i) / head_dim);
        const double c = std::cos(angle), s = std::sin(angle);
        const double ga = dY[base + i], gb = dY[base + i + half];
        dx[base + i] = ga * c + gb * s;
        dx[base + i + half] = -ga * s + gb * c;
      }
    }
  }
  return dx;
}

void silu_mul_bwd_ref(const std::vector<double>& g,
                      const std::vector<double>& u,
                      const std::vector<float>& dZ,
                      std::vector<double>& dG, std::vector<double>& dU) {
  dG.assign(g.size(), 0.0);
  dU.assign(g.size(), 0.0);
  for (std::size_t i = 0; i < g.size(); ++i) {
    const double sig = 1.0 / (1.0 + std::exp(-g[i]));
    dU[i] = dZ[i] * g[i] * sig;
    dG[i] = dZ[i] * u[i] * sig * (1.0 + g[i] * (1.0 - sig));
  }
}

std::vector<double> embedding_bwd_ref(const std::vector<std::int32_t>& tokens,
                                      const std::vector<float>& dY,
                                      std::size_t rows, std::size_t dim,
                                      std::size_t vocab) {
  std::vector<double> dw(vocab * dim, 0.0);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t d = 0; d < dim; ++d)
      dw[static_cast<std::size_t>(tokens[r]) * dim + d] += dY[r * dim + d];
  return dw;
}

float max_abs(const std::vector<double>& ref, const std::vector<float>& got) {
  float m = 0.0f;
  for (std::size_t i = 0; i < got.size(); ++i)
    m = std::max(m, std::fabs(got[i] - static_cast<float>(ref[i])));
  return m;
}

float max_rel(const std::vector<double>& analytic,
              const std::vector<double>& numeric) {
  float m = 0.0f;
  for (std::size_t i = 0; i < analytic.size(); ++i) {
    const double a = analytic[i], n = numeric[i];
    const double rel =
        std::fabs(a - n) / std::max({std::fabs(a), std::fabs(n), 1e-3});
    m = std::max(m, static_cast<float>(rel));
  }
  return m;
}

// Central-difference gradient of `loss` with respect to element `i` of `x`.
template <typename Loss>
double fd_grad(Loss loss, std::vector<double> x, std::size_t i, double h) {
  const double save = x[i];
  x[i] = save + h;
  const double up = loss(x);
  x[i] = save - h;
  const double down = loss(x);
  x[i] = save;
  return (up - down) / (2.0 * h);
}

constexpr double kH = 1e-4;

} // namespace

int main() {
  Ctx c;
  Rng rng;

  if (!sonicboom::training::cuda::available()) {
    std::printf("SKIP: no CUDA device\n");
    return 0;
  }

  // ---------------------------------------------------------------- matmul --
  {
    const std::size_t M = kSeq, K = kHidden, N = kVocab;
    Rng r2;
    std::vector<float> A = rnd(M * K, -1.0f, 1.0f, r2);
    std::vector<float> B = rnd(K * N, -0.5f, 0.5f, r2);
    std::vector<float> dY = rnd(M * N, -1.0f, 1.0f, r2);
    std::vector<float> dA(M * K), dB(K * N);
    std::vector<float> hA(M * K), hB(K * N);

    const bool ran = sonicboom::training::cuda::matmul_bwd(
        A, B, dY, static_cast<int64_t>(M), static_cast<int64_t>(K),
        static_cast<int64_t>(N), dA, dB);
    check(c, ran, "matmul_bwd device run");

    const MatmulRef ref = matmul_ref(A, B, dY, M, K, N);
    sonicboom::training::matmul_bwd_f32(A.data(), B.data(), dY.data(),
                                        static_cast<int64_t>(M),
                                        static_cast<int64_t>(K),
                                        static_cast<int64_t>(N), hA.data(),
                                        hB.data());
    const float dA_err = max_abs(ref.dA, dA);
    const float dB_err = max_abs(ref.dB, dB);
    const float hA_err = max_abs(ref.dA, hA);
    const float hB_err = max_abs(ref.dB, hB);
    std::printf("matmul_bwd: device dA %.3e dB %.3e | host dA %.3e dB %.3e\n",
                dA_err, dB_err, hA_err, hB_err);
    check(c, dA_err <= 1e-5f, "matmul_bwd device dA <= 1e-5");
    check(c, dB_err <= 1e-5f, "matmul_bwd device dB <= 1e-5");
    check(c, hA_err <= 1e-5f, "matmul_bwd host dA <= 1e-5");
    check(c, hB_err <= 1e-5f, "matmul_bwd host dB <= 1e-5");

    // Grad-check at small shapes.
    const std::size_t m = 3, k = 4, n = 2;
    std::vector<float> a = rnd(m * k, -1.0f, 1.0f, r2);
    std::vector<float> b = rnd(k * n, -1.0f, 1.0f, r2);
    std::vector<float> dy = rnd(m * n, -1.0f, 1.0f, r2);
    std::vector<float> da(m * k), db(k * n);
    check(c, sonicboom::training::cuda::matmul_bwd(
                 a, b, dy, static_cast<int64_t>(m), static_cast<int64_t>(k),
                 static_cast<int64_t>(n), da, db),
          "matmul_bwd grad-check device run");

    auto loss_a = [&](const std::vector<double>& av) {
      double s = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          double y = 0.0;
          for (std::size_t q = 0; q < k; ++q)
            y += av[i * k + q] * static_cast<double>(b[q * n + j]);
          s += static_cast<double>(dy[i * n + j]) * y;
        }
      return s;
    };
    auto loss_b = [&](const std::vector<double>& bv) {
      double s = 0.0;
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          double y = 0.0;
          for (std::size_t q = 0; q < k; ++q)
            y += static_cast<double>(a[i * k + q]) * bv[q * n + j];
          s += static_cast<double>(dy[i * n + j]) * y;
        }
      return s;
    };
    std::vector<double> num_a(da.size()), num_b(db.size());
    std::vector<double> a64 = as_double(a), b64 = as_double(b);
    for (std::size_t i = 0; i < num_a.size(); ++i)
      num_a[i] = fd_grad(loss_a, a64, i, kH);
    for (std::size_t i = 0; i < num_b.size(); ++i)
      num_b[i] = fd_grad(loss_b, b64, i, kH);
    const float rel_a = max_rel(as_double(da), num_a);
    const float rel_b = max_rel(as_double(db), num_b);
    std::printf("matmul_bwd: grad-check dA %.3e dB %.3e\n", rel_a, rel_b);
    check(c, rel_a <= 1e-3f, "matmul_bwd grad-check dA <= 1e-3");
    check(c, rel_b <= 1e-3f, "matmul_bwd grad-check dB <= 1e-3");
  }

  // --------------------------------------------------------------- softmax --
  {
    Rng r3;
    std::vector<float> X = rnd(kSeq * kVocab, -3.0f, 3.0f, r3);
    std::vector<float> dY = rnd(kSeq * kVocab, -1.0f, 1.0f, r3);
    std::vector<double> X64 = as_double(X);
    const std::vector<double> Y64 = softmax_rows(X64, kSeq, kVocab);
    std::vector<float> Y(Y64.begin(), Y64.end());
    std::vector<float> dX(kSeq * kVocab), hX(kSeq * kVocab);

    check(c, sonicboom::training::cuda::softmax_bwd(
                 Y, dY, static_cast<int64_t>(kSeq),
                 static_cast<int64_t>(kVocab), dX),
          "softmax_bwd device run");
    sonicboom::training::softmax_bwd_f32(
        Y.data(), dY.data(), static_cast<int64_t>(kSeq),
        static_cast<int64_t>(kVocab), hX.data());

    const std::vector<double> ref = softmax_bwd_ref(Y64, dY, kSeq, kVocab);
    const float dev_err = max_abs(ref, dX);
    const float host_err = max_abs(ref, hX);
    std::printf("softmax_bwd: device %.3e host %.3e\n", dev_err, host_err);
    check(c, dev_err <= 1e-5f, "softmax_bwd device <= 1e-5");
    check(c, host_err <= 1e-5f, "softmax_bwd host <= 1e-5");

    // Grad-check: perturb the logits, re-run the op on the recomputed softmax.
    const std::size_t rows = 2, cols = 5;
    std::vector<float> x = rnd(rows * cols, -1.0f, 1.0f, r3);
    std::vector<float> dy = rnd(rows * cols, -1.0f, 1.0f, r3);
    std::vector<double> x64 = as_double(x);
    auto loss = [&](const std::vector<double>& xv) {
      const std::vector<double> y = softmax_rows(xv, rows, cols);
      double s = 0.0;
      for (std::size_t i = 0; i < y.size(); ++i)
        s += static_cast<double>(dy[i]) * y[i];
      return s;
    };
    const std::vector<double> y64 = softmax_rows(x64, rows, cols);
    std::vector<float> yf(y64.begin(), y64.end());
    std::vector<float> dx(rows * cols);
    check(c, sonicboom::training::cuda::softmax_bwd(
                 yf, dy, static_cast<int64_t>(rows), static_cast<int64_t>(cols),
                 dx),
          "softmax_bwd grad-check device run");
    std::vector<double> num(dx.size());
    for (std::size_t i = 0; i < num.size(); ++i)
      num[i] = fd_grad(loss, x64, i, kH);
    const float rel = max_rel(as_double(dx), num);
    std::printf("softmax_bwd: grad-check %.3e\n", rel);
    check(c, rel <= 1e-3f, "softmax_bwd grad-check <= 1e-3");
  }

  // --------------------------------------------------------- cross-entropy --
  {
    Rng r4;
    std::vector<float> logits = rnd(kSeq * kVocab, -2.0f, 2.0f, r4);
    std::vector<std::int32_t> targets(kSeq);
    for (std::size_t r = 0; r < kSeq; ++r)
      targets[r] = static_cast<std::int32_t>(r4.next() % kVocab);
    std::vector<float> dlogits(kSeq * kVocab), hlogits(kSeq * kVocab);
    float device_loss = 0.0f;

    check(c, sonicboom::training::cuda::cross_entropy_bwd(
                 logits, targets, static_cast<int64_t>(kSeq),
                 static_cast<int64_t>(kVocab), dlogits, &device_loss),
          "cross_entropy_bwd device run");
    const float host_loss = sonicboom::training::cross_entropy_bwd_f32(
        logits.data(), targets.data(), static_cast<int64_t>(kSeq),
        static_cast<int64_t>(kVocab), hlogits.data());

    const std::vector<double> logits64 = as_double(logits);
    const double ref_loss = ce_loss_ref(logits64, targets, kSeq, kVocab);
    const std::vector<double> ref = ce_bwd_ref(logits64, targets, kSeq, kVocab);
    const float dev_err = max_abs(ref, dlogits);
    const float host_err = max_abs(ref, hlogits);
    const float loss_err = std::fabs(device_loss - static_cast<float>(ref_loss));
    const float host_loss_err = std::fabs(host_loss - static_cast<float>(ref_loss));
    std::printf("cross_entropy_bwd: device %.3e host %.3e loss %.3e (host %.3e)\n",
                dev_err, host_err, loss_err, host_loss_err);
    check(c, dev_err <= 1e-5f, "cross_entropy_bwd device <= 1e-5");
    check(c, host_err <= 1e-5f, "cross_entropy_bwd host <= 1e-5");
    check(c, loss_err <= 1e-5f, "cross_entropy_bwd loss <= 1e-5");
    check(c, host_loss_err <= 1e-5f, "cross_entropy_bwd host loss <= 1e-5");

    // Grad-check.
    const std::size_t rows = 2, cols = 7;
    std::vector<float> lg = rnd(rows * cols, -1.0f, 1.0f, r4);
    std::vector<std::int32_t> tg(rows);
    for (std::size_t r = 0; r < rows; ++r)
      tg[r] = static_cast<std::int32_t>(r4.next() % cols);
    std::vector<float> dl(rows * cols);
    float dl_loss = 0.0f;
    check(c, sonicboom::training::cuda::cross_entropy_bwd(
                 lg, tg, static_cast<int64_t>(rows), static_cast<int64_t>(cols),
                 dl, &dl_loss),
          "cross_entropy_bwd grad-check device run");
    const std::vector<double> lg64 = as_double(lg);
    auto loss = [&](const std::vector<double>& lv) {
      return ce_loss_ref(lv, tg, rows, cols);
    };
    std::vector<double> num(dl.size());
    for (std::size_t i = 0; i < num.size(); ++i)
      num[i] = fd_grad(loss, lg64, i, kH);
    const float rel = max_rel(as_double(dl), num);
    std::printf("cross_entropy_bwd: grad-check %.3e\n", rel);
    check(c, rel <= 1e-3f, "cross_entropy_bwd grad-check <= 1e-3");
  }

  // ----------------------------------------------------------- RMSNorm bwd --
  {
    Rng r5;
    const std::size_t rows = kSeq, dim = kHidden;
    const float eps = 1e-6f;
    std::vector<float> x = rnd(rows * dim, -1.0f, 1.0f, r5);
    std::vector<float> w = rnd(dim, -0.5f, 0.5f, r5);
    std::vector<float> dY = rnd(rows * dim, -1.0f, 1.0f, r5);
    std::vector<float> dX(rows * dim), dW(dim), hX(rows * dim), hW(dim);

    check(c, sonicboom::training::cuda::rmsnorm_bwd(
                 x, w, dY, static_cast<int64_t>(rows), static_cast<int64_t>(dim),
                 eps, dX, dW),
          "rmsnorm_bwd device run");
    sonicboom::training::rmsnorm_bwd_f32(
        x.data(), w.data(), dY.data(), static_cast<int64_t>(rows),
        static_cast<int64_t>(dim), eps, hX.data(), hW.data());

    const RmsRef ref = rmsnorm_bwd_ref(as_double(x), as_double(w), dY, rows,
                                       dim, static_cast<double>(eps));
    const float dx_err = max_abs(ref.dX, dX), dw_err = max_abs(ref.dW, dW);
    const float hx_err = max_abs(ref.dX, hX), hw_err = max_abs(ref.dW, hW);
    std::printf("rmsnorm_bwd: device dX %.3e dW %.3e | host dX %.3e dW %.3e\n",
                dx_err, dw_err, hx_err, hw_err);
    check(c, dx_err <= 1e-5f, "rmsnorm_bwd device dX <= 1e-5");
    check(c, dw_err <= 1e-5f, "rmsnorm_bwd device dW <= 1e-5");
    check(c, hx_err <= 1e-5f, "rmsnorm_bwd host dX <= 1e-5");
    check(c, hw_err <= 1e-5f, "rmsnorm_bwd host dW <= 1e-5");

    // Grad-check.
    const std::size_t m = 2, d = 4;
    std::vector<float> xs = rnd(m * d, -1.0f, 1.0f, r5);
    std::vector<float> ws = rnd(d, -1.0f, 1.0f, r5);
    std::vector<float> dys = rnd(m * d, -1.0f, 1.0f, r5);
    std::vector<float> dxs(m * d), dws(d);
    check(c, sonicboom::training::cuda::rmsnorm_bwd(
                 xs, ws, dys, static_cast<int64_t>(m), static_cast<int64_t>(d),
                 eps, dxs, dws),
          "rmsnorm_bwd grad-check device run");
    const std::vector<double> xs64 = as_double(xs), ws64 = as_double(ws);
    auto loss_x = [&](const std::vector<double>& xv) {
      const std::vector<double> y =
          rmsnorm_fwd_ref(xv, ws64, m, d, static_cast<double>(eps));
      double s = 0.0;
      for (std::size_t i = 0; i < y.size(); ++i)
        s += static_cast<double>(dys[i]) * y[i];
      return s;
    };
    auto loss_w = [&](const std::vector<double>& wv) {
      const std::vector<double> y =
          rmsnorm_fwd_ref(xs64, wv, m, d, static_cast<double>(eps));
      double s = 0.0;
      for (std::size_t i = 0; i < y.size(); ++i)
        s += static_cast<double>(dys[i]) * y[i];
      return s;
    };
    std::vector<double> num_x(dxs.size()), num_w(dws.size());
    for (std::size_t i = 0; i < num_x.size(); ++i)
      num_x[i] = fd_grad(loss_x, xs64, i, kH);
    for (std::size_t i = 0; i < num_w.size(); ++i)
      num_w[i] = fd_grad(loss_w, ws64, i, kH);
    const float rel_x = max_rel(as_double(dxs), num_x);
    const float rel_w = max_rel(as_double(dws), num_w);
    std::printf("rmsnorm_bwd: grad-check dX %.3e dW %.3e\n", rel_x, rel_w);
    check(c, rel_x <= 1e-3f, "rmsnorm_bwd grad-check dX <= 1e-3");
    check(c, rel_w <= 1e-3f, "rmsnorm_bwd grad-check dW <= 1e-3");
  }

  // --------------------------------------------------------------- RoPE bwd --
  {
    Rng r6;
    const std::size_t rows = kSeq, heads = 4, head_dim = 16;
    const auto n = rows * heads * head_dim;
    const float theta = 10000.0f, fs = 1.0f;
    const std::uint64_t pos0 = 0;
    std::vector<float> dY = rnd(n, -1.0f, 1.0f, r6);
    std::vector<float> dX(n), hX(n);

    check(c, sonicboom::training::cuda::rope_bwd(
                 dY, static_cast<int64_t>(rows), static_cast<int64_t>(heads),
                 static_cast<int64_t>(head_dim), pos0, theta, fs, dX),
          "rope_bwd device run");
    sonicboom::training::rope_bwd_f32(
        dY.data(), static_cast<int64_t>(rows), static_cast<int64_t>(heads),
        static_cast<int64_t>(head_dim), pos0, theta, fs, hX.data());

    const std::vector<double> ref =
        rope_bwd_ref(dY, rows, heads, head_dim, pos0, theta, fs);
    const float dev_err = max_abs(ref, dX), host_err = max_abs(ref, hX);
    std::printf("rope_bwd: device %.3e host %.3e\n", dev_err, host_err);
    check(c, dev_err <= 1e-5f, "rope_bwd device <= 1e-5");
    check(c, host_err <= 1e-5f, "rope_bwd host <= 1e-5");

    // Grad-check: perturb the forward input, compare dX to dL/dx.
    const std::size_t m = 2, h = 1, hd = 4;
    const std::uint64_t p0 = 3;
    std::vector<float> x = rnd(m * h * hd, -1.0f, 1.0f, r6);
    std::vector<float> dy = rnd(m * h * hd, -1.0f, 1.0f, r6);
    std::vector<float> dx(m * h * hd);
    check(c, sonicboom::training::cuda::rope_bwd(
                 dy, static_cast<int64_t>(m), static_cast<int64_t>(h),
                 static_cast<int64_t>(hd), p0, theta, fs, dx),
          "rope_bwd grad-check device run");
    const std::vector<double> x64 = as_double(x);
    auto loss = [&](const std::vector<double>& xv) {
      const std::vector<double> y =
          rope_fwd_ref(xv, m, h, hd, p0, theta, fs);
      double s = 0.0;
      for (std::size_t i = 0; i < y.size(); ++i)
        s += static_cast<double>(dy[i]) * y[i];
      return s;
    };
    std::vector<double> num(dx.size());
    for (std::size_t i = 0; i < num.size(); ++i)
      num[i] = fd_grad(loss, x64, i, kH);
    const float rel = max_rel(as_double(dx), num);
    std::printf("rope_bwd: grad-check %.3e\n", rel);
    check(c, rel <= 1e-3f, "rope_bwd grad-check <= 1e-3");
  }

  // ----------------------------------------------------------- SwiGLU bwd --
  {
    Rng r7;
    const auto n = kSeq * kHidden;
    std::vector<float> g = rnd(n, -2.0f, 2.0f, r7);
    std::vector<float> u = rnd(n, -1.0f, 1.0f, r7);
    std::vector<float> dZ = rnd(n, -1.0f, 1.0f, r7);
    std::vector<float> dG(n), dU(n), hG(n), hU(n);

    check(c, sonicboom::training::cuda::silu_mul_bwd(g, u, dZ, dG, dU),
          "silu_mul_bwd device run");
    sonicboom::training::silu_mul_bwd_f32(g.data(), u.data(), dZ.data(),
                                          static_cast<int64_t>(n), hG.data(),
                                          hU.data());
    std::vector<double> refG, refU;
    silu_mul_bwd_ref(as_double(g), as_double(u), dZ, refG, refU);
    const float dg_err = max_abs(refG, dG), du_err = max_abs(refU, dU);
    const float hg_err = max_abs(refG, hG), hu_err = max_abs(refU, hU);
    std::printf("silu_mul_bwd: device dG %.3e dU %.3e | host dG %.3e dU %.3e\n",
                dg_err, du_err, hg_err, hu_err);
    check(c, dg_err <= 1e-5f, "silu_mul_bwd device dG <= 1e-5");
    check(c, du_err <= 1e-5f, "silu_mul_bwd device dU <= 1e-5");
    check(c, hg_err <= 1e-5f, "silu_mul_bwd host dG <= 1e-5");
    check(c, hu_err <= 1e-5f, "silu_mul_bwd host dU <= 1e-5");

    // Grad-check over g and u.
    const std::size_t m = 5;
    std::vector<float> gs = rnd(m, -2.0f, 2.0f, r7);
    std::vector<float> us = rnd(m, -1.0f, 1.0f, r7);
    std::vector<float> dzs = rnd(m, -1.0f, 1.0f, r7);
    std::vector<float> dgs(m), dus(m);
    check(c, sonicboom::training::cuda::silu_mul_bwd(gs, us, dzs, dgs, dus),
          "silu_mul_bwd grad-check device run");
    const std::vector<double> gs64 = as_double(gs), us64 = as_double(us);
    auto loss_g = [&](const std::vector<double>& gv) {
      double s = 0.0;
      for (std::size_t i = 0; i < m; ++i) {
        const double sig = 1.0 / (1.0 + std::exp(-gv[i]));
        s += static_cast<double>(dzs[i]) * (gv[i] * sig) * us64[i];
      }
      return s;
    };
    auto loss_u = [&](const std::vector<double>& uv) {
      double s = 0.0;
      for (std::size_t i = 0; i < m; ++i) {
        const double sig = 1.0 / (1.0 + std::exp(-gs64[i]));
        s += static_cast<double>(dzs[i]) * (gs64[i] * sig) * uv[i];
      }
      return s;
    };
    std::vector<double> num_g(m), num_u(m);
    for (std::size_t i = 0; i < m; ++i) {
      num_g[i] = fd_grad(loss_g, gs64, i, kH);
      num_u[i] = fd_grad(loss_u, us64, i, kH);
    }
    const float rel_g = max_rel(as_double(dgs), num_g);
    const float rel_u = max_rel(as_double(dus), num_u);
    std::printf("silu_mul_bwd: grad-check dG %.3e dU %.3e\n", rel_g, rel_u);
    check(c, rel_g <= 1e-3f, "silu_mul_bwd grad-check dG <= 1e-3");
    check(c, rel_u <= 1e-3f, "silu_mul_bwd grad-check dU <= 1e-3");
  }

  // -------------------------------------------------------- embedding bwd --
  {
    Rng r8;
    const std::size_t rows = kSeq, dim = kHidden, vocab = kVocab;
    std::vector<std::int32_t> tokens(rows);
    for (auto& t : tokens)
      t = static_cast<std::int32_t>(r8.next() % vocab);
    std::vector<float> dY = rnd(rows * dim, -1.0f, 1.0f, r8);
    std::vector<float> dW(vocab * dim), hW(vocab * dim);

    check(c, sonicboom::training::cuda::embedding_bwd(
                 tokens, dY, static_cast<int64_t>(rows),
                 static_cast<int64_t>(dim), static_cast<int64_t>(vocab), dW),
          "embedding_bwd device run");
    sonicboom::training::embedding_bwd_f32(
        tokens.data(), dY.data(), static_cast<int64_t>(rows),
        static_cast<int64_t>(dim), static_cast<int64_t>(vocab), hW.data());
    const std::vector<double> ref =
        embedding_bwd_ref(tokens, dY, rows, dim, vocab);
    const float dev_err = max_abs(ref, dW), host_err = max_abs(ref, hW);
    std::printf("embedding_bwd: device %.3e host %.3e\n", dev_err, host_err);
    check(c, dev_err <= 1e-5f, "embedding_bwd device <= 1e-5");
    check(c, host_err <= 1e-5f, "embedding_bwd host <= 1e-5");

    // Grad-check with repeated tokens (vocab 3, 4 rows).
    const std::size_t m = 4, d = 3, v = 3;
    const std::vector<std::int32_t> tk = {0, 2, 0, 1};
    std::vector<float> dy = rnd(m * d, -1.0f, 1.0f, r8);
    std::vector<float> dw(v * d);
    check(c, sonicboom::training::cuda::embedding_bwd(
                 tk, dy, static_cast<int64_t>(m), static_cast<int64_t>(d),
                 static_cast<int64_t>(v), dw),
          "embedding_bwd grad-check device run");
    auto loss = [&](const std::vector<double>& wv) {
      double s = 0.0;
      for (std::size_t r = 0; r < m; ++r)
        for (std::size_t q = 0; q < d; ++q)
          s += static_cast<double>(dy[r * d + q]) *
               wv[static_cast<std::size_t>(tk[r]) * d + q];
      return s;
    };
    std::vector<double> num(dw.size());
    std::vector<double> w64(v * d, 0.0);
    for (std::size_t i = 0; i < num.size(); ++i)
      num[i] = fd_grad(loss, w64, i, kH);
    const float rel = max_rel(as_double(dw), num);
    std::printf("embedding_bwd: grad-check %.3e\n", rel);
    check(c, rel <= 1e-3f, "embedding_bwd grad-check <= 1e-3");
  }

  if (!c.ok) {
    std::fprintf(stderr, "FAIL: %s\n", c.what);
    return 1;
  }
  std::printf("PASS: backward ops\n");
  return 0;
}
