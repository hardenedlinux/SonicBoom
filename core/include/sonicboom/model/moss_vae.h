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

// MOSS-Image VAE decoder (the FLUX.2 AutoencoderKLFlux2 decoder). The
// reference is diffusers' `AutoencoderKLFlux2.decode`, the module that the
// MOSS-Image pipeline runs after its BatchNorm unnormalization and 2x2
// pixel-unshuffle. Weights are f32 in the checkpoint; the arithmetic is f32
// on CUDA through the native-torch CUDA kernels (Layer 1 direct-call adapter).
//
// The decoder is memory-shaped for latent grids that fit an 8 GB device: the
// convolution is the im2col + GEMM kernel, whose column buffer grows with the
// spatial area.

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sonicboom::moss {

struct VaeConfig {
  int64_t latent_channels = 32;
  int64_t out_channels = 3;
  // Decoder channel widths, from the deepest block to the output.
  std::vector<int64_t> channels{384, 384, 192, 96};
  int64_t layers_per_block = 2;  // + 1 resnet per up block
  int64_t norm_groups = 32;
  double eps = 1e-6;
  double bn_eps = 1e-4;  // BatchNorm statistics' epsilon

  int64_t spatial_scale() const noexcept { return 4; }  // 2 x 2 unshuffle
};

// The decoder's BatchNorm statistics, one entry per denoised-latent channel
// (the DiT's 128, not the VAE's 32). MOSS-Image unnormalizes the latent with
// these before the 2x2 pixel-unshuffle that feeds decode().
struct VaeBatchNorm {
  std::vector<float> running_mean;
  std::vector<float> running_var;
  double eps = 0.0;
};

// One named intermediate, dumped on request for the S4 parity gate.
struct VaeStage {
  std::string name;
  int64_t n = 0, c = 0, h = 0, w = 0;
  std::vector<float> values;  // [n][c][h][w], f32
};

class VaeDecoder {
 public:
  static std::expected<std::unique_ptr<VaeDecoder>, std::string> load(
      const std::filesystem::path& dir);

  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;
  ~VaeDecoder();

  const VaeConfig& config() const noexcept;
  const VaeBatchNorm& batch_norm() const noexcept;
  int64_t weight_bytes() const noexcept;

  // latent: [latent_channels][height][width] f32, the input of
  // `AutoencoderKLFlux2.decode` (already BN-normalized and unshuffled).
  // Returns the decoded image [out_channels][8*height][8*width] f32 (three
  // 2x upsamplers).
  std::expected<std::vector<float>, std::string> decode(
      std::span<const float> latent, int64_t height, int64_t width,
      std::vector<VaeStage>* stages = nullptr) const;

  // Returns the ATen caching allocator's unused device blocks to the driver.
  // The decode temporaries are ATen allocations, and the Q8_0 DiT's next
  // upload (raw cudaMalloc) cannot reuse the cached blocks; on an 8 GiB device
  // the DiT and the cached decode peak do not fit together. No-op without
  // CUDA.
  void release_cached_memory() const;

 private:
  struct Impl;
  explicit VaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonicboom::moss
