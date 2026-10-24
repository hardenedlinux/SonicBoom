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

// CUDA path for the MOSS-Image diffusion transformer.
//
// The checkpoint ships bf16 weights. This path quantizes every projection to
// Q8_0 *at load* (the released bf16 bytes are never up-cast into storage) and
// runs the same arithmetic as DiffusionTransformer::forward with f32
// activations resident on the device: one host call per projection, norm,
// elementwise glue and attention per block, and a single D2H copy of the final
// latent. The projection kernel is the same MMQ/q8_1 path the LLM prefill uses,
// so a Q8_0 DiT runs on the int8 tensor cores.
//
// Weight loading is chunked: one reader thread preads the next tensor into one
// of two pinned staging buffers while the main thread quantizes the current
// one and starts its asynchronous H2D copy on a dedicated stream (so disk I/O,
// quantization and upload overlap). When pread is unavailable or
// SONICBOOM_MOSS_LOAD=mmap is set, the safetensors mmap is used instead and
// each tensor's pages are dropped with madvise(DONTNEED) after use. Weights
// stay in Q8_0 on the host (mirroring the device copy) for the lifetime of the
// object.

#include <sonicboom/model/moss_dit.h>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sonicboom::moss {

// True when a usable CUDA device is present.
bool dit_cuda_available() noexcept;

class CudaDiffusionTransformer {
 public:
  static std::expected<std::unique_ptr<CudaDiffusionTransformer>, std::string>
  load(const std::filesystem::path& dir);

  CudaDiffusionTransformer(const CudaDiffusionTransformer&) = delete;
  CudaDiffusionTransformer& operator=(const CudaDiffusionTransformer&) = delete;
  ~CudaDiffusionTransformer();

  const DitConfig& config() const noexcept;
  // Bytes of Q8_0 storage (host and device are both this size).
  int64_t weight_bytes() const noexcept;

  // Drop the device-resident weights and scratch buffers (the host Q8_0
  // mirror stays); the next forward re-uploads it. Used before the VAE decode,
  // whose peak does not fit beside the DiT on an 8 GiB device. Idempotent.
  void release_device();

  // Same contract as DiffusionTransformer::forward, without the optional debug
  // dumps (the parity harness compares the returned latent).
  std::expected<DitSample, std::string> forward(
      std::span<const float> hidden, int64_t height, int64_t width,
      float timestep, std::span<const float> text, int64_t text_len,
      const std::vector<int64_t>& skip_layers = {});

 private:
  struct Impl;
  explicit CudaDiffusionTransformer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonicboom::moss
