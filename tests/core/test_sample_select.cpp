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

// r7 D2: the CUDA sampling candidate selector (sample_select_dev) must produce
// exactly nn::select_candidates_host's list -- the same count, ids and z values
// bit for bit -- for real vocabularies (Gemma 262144, Qwen 151936), with and
// without a random grammar mask, with and without a penalty window (including
// out-of-vocabulary window slots), at temperatures 0.3 / 0.8 / 1.5.
//
// 50 cases per vocabulary. Skips (exit 77) when no CUDA device is present.

#include <sonicboom/nn/sampling.h>

#include <cuda_runtime.h>

#include "../../core/src/nn/cuda_resident.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace sbnn = sonicboom::nn;
namespace sbcuda = sonicboom::nn::cuda;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
}

bool z_bits_equal(float a, float b) {
  uint32_t ua = 0, ub = 0;
  std::memcpy(&ua, &a, sizeof ua);
  std::memcpy(&ub, &b, sizeof ub);
  return ua == ub;
}

}  // namespace

int main() {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    std::fprintf(stderr, "SKIP: no CUDA device\n");
    return 77;
  }

  const int vocabs[2] = {262144, 151936};
  const float temperatures[3] = {0.3f, 0.8f, 1.5f};
  std::mt19937 rng(0x5EED1234u);
  std::uniform_real_distribution<float> logit_dist(-8.0f, 16.0f);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  constexpr int kCases = 50;
  for (int vi = 0; vi < 2; ++vi) {
    const int vocab = vocabs[vi];
    std::vector<float> logits(static_cast<size_t>(vocab));
    std::vector<uint8_t> mask(static_cast<size_t>((vocab + 7) / 8));
    std::vector<int32_t> window(1024);

    float* logits_dev = nullptr;
    uint8_t* mask_dev = nullptr;
    int32_t* window_dev = nullptr;
    sbnn::Candidate* out_dev = nullptr;
    int* count_dev = nullptr;
    if (cudaMalloc(&logits_dev, logits.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&mask_dev, mask.size()) != cudaSuccess ||
        cudaMalloc(&window_dev, window.size() * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&out_dev, sbnn::kSampleCandidates * sizeof(sbnn::Candidate)) !=
            cudaSuccess ||
        cudaMalloc(&count_dev, sizeof(int)) != cudaSuccess) {
      std::fprintf(stderr, "SKIP: device allocation failed\n");
      return 77;
    }

    for (int c = 0; c < kCases; ++c) {
      for (float& v : logits) v = logit_dist(rng);
      const bool use_mask = (c & 1) != 0;
      const bool use_window = (c & 2) != 0;
      const bool active_penalties = ((c >> 2) & 1) != 0;
      const float temperature = temperatures[c % 3];

      for (uint8_t& b : mask) b = uint8_t(byte_dist(rng));
      // Guarantee every case has a legal token, and exercise the "past the
      // bitmap" prefix too (the mask covers the whole vocabulary here).
      mask[0] |= 0x01u;
      if (use_mask) {
        // Keep a few random disallowed runs so the mask matters.
        for (size_t i = 0; i < mask.size(); i += 97) mask[i] = 0;
        mask[0] |= 0x01u;
      }

      int window_len = 0;
      std::vector<int64_t> history;
      if (use_window) {
        window_len = 1 + int(rng() % 1024);
        std::uniform_int_distribution<int> tok_dist(0, vocab - 1);
        history.resize(size_t(window_len));
        for (int i = 0; i < window_len; ++i) {
          int t = tok_dist(rng);
          // Every 11th slot is out of vocabulary: the host drops it and the
          // kernel must ignore it too.
          if (i % 11 == 10) t = (i & 1) ? -3 - (i % 7) : vocab + 5 + (i % 13);
          window[size_t(i)] = int32_t(t);
          history[size_t(i)] = int64_t(t);
        }
        // Force repeats so occurrence counts exceed one.
        for (int i = 1; i < window_len; i += 3) {
          window[size_t(i)] = window[0];
          history[size_t(i)] = history[0];
        }
      }

      sbnn::SamplingParams p;
      p.temperature = temperature;
      p.repeat_penalty = active_penalties ? 1.37f : 1.0f;
      p.presence_penalty = active_penalties ? 0.21f : 0.0f;
      p.frequency_penalty = active_penalties ? 0.07f : 0.0f;
      p.repeat_last_n = window_len ? uint32_t(window_len) : 0;

      const std::span<const uint8_t> mask_span =
          use_mask ? std::span<const uint8_t>(mask) : std::span<const uint8_t>();
      const std::vector<sbnn::Candidate> want =
          sbnn::select_candidates_host(logits, p, history, mask_span);

      if (cudaMemcpy(logits_dev, logits.data(), logits.size() * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(mask_dev, mask.data(), mask.size(),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(window_dev, window.data(), window.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        check(false, "device upload");
        continue;
      }

      const bool launched = sbcuda::sample_select_dev(
          logits_dev, vocab, use_mask ? mask_dev : nullptr,
          use_window ? window_dev : nullptr, window_len, temperature,
          p.repeat_penalty, p.presence_penalty, p.frequency_penalty, out_dev,
          count_dev);
      if (!launched || cudaDeviceSynchronize() != cudaSuccess) {
        check(false, "sample_select_dev launch");
        continue;
      }

      int count = -1;
      std::vector<sbnn::Candidate> got(sbnn::kSampleCandidates);
      if (cudaMemcpy(&count, count_dev, sizeof count, cudaMemcpyDeviceToHost) !=
              cudaSuccess ||
          cudaMemcpy(got.data(), out_dev, got.size() * sizeof(sbnn::Candidate),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        check(false, "device readback");
        continue;
      }

      bool ok = count == int(want.size());
      for (int i = 0; ok && i < count; ++i) {
        ok = got[size_t(i)].id == want[size_t(i)].id &&
             z_bits_equal(got[size_t(i)].z, want[size_t(i)].z);
      }
      if (!ok) {
        std::fprintf(stderr,
                     "    case vocab=%d c=%d T=%.2f mask=%d window=%d "
                     "penalties=%d: count got %d want %zu\n",
                     vocab, c, double(temperature), int(use_mask),
                     window_len, int(active_penalties), count, want.size());
        for (int i = 0; i < count && i < int(want.size()) && i < 8; ++i) {
          if (got[size_t(i)].id != want[size_t(i)].id ||
              !z_bits_equal(got[size_t(i)].z, want[size_t(i)].z)) {
            std::fprintf(stderr,
                         "      [%d] got id %d z %.9g, want id %d z %.9g\n", i,
                         got[size_t(i)].id, double(got[size_t(i)].z),
                         want[size_t(i)].id, double(want[size_t(i)].z));
          }
        }
        ++g_failures;
      }
    }

    // Equal logits at scale: ids 0..255 in order, all z equal.
    {
      std::vector<float> flat(static_cast<size_t>(vocab), 0.25f);
      sbnn::SamplingParams p;
      p.temperature = 1.0f;
      const std::vector<sbnn::Candidate> want =
          sbnn::select_candidates_host(flat, p, {}, {});
      if (cudaMemcpy(logits_dev, flat.data(), flat.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess) {
        const bool launched = sbcuda::sample_select_dev(
            logits_dev, vocab, nullptr, nullptr, 0, 1.0f, 1.0f, 0.0f, 0.0f,
            out_dev, count_dev);
        int count = -1;
        std::vector<sbnn::Candidate> got(sbnn::kSampleCandidates);
        if (!launched || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(&count, count_dev, sizeof count, cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(got.data(), out_dev,
                       got.size() * sizeof(sbnn::Candidate),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          check(false, "equal-logit case launch");
        } else {
          bool ok = count == int(want.size());
          for (int i = 0; ok && i < count; ++i)
            ok = got[size_t(i)].id == want[size_t(i)].id &&
                 z_bits_equal(got[size_t(i)].z, want[size_t(i)].z);
          check(ok, "equal logits: 256 lowest ids with bitwise-equal z");
        }
      }
    }

    cudaFree(count_dev);
    cudaFree(out_dev);
    cudaFree(window_dev);
    cudaFree(mask_dev);
    cudaFree(logits_dev);
  }

  if (g_failures) {
    std::fprintf(stderr, "%d sample_select test(s) failed\n", g_failures);
    return 1;
  }
  std::printf("sample_select tests passed\n");
  return 0;
}
