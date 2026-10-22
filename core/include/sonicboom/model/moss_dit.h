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

// MOSS-Image diffusion transformer (the single-stream DiT that MOSS-Image
// denoises latents with). The reference is the Apache-2.0 modeling code that
// ships in the weights repository (pipeline_moss_image.py, MOSSImageTransformer
// / _Block / _TextBlock / _RoPE); this executor runs that arithmetic in f32
// from the memory-mapped bf16 checkpoint, the same policy as the text encoder.
//
// One sample per forward, evaluated at its true text length: the reference
// concatenates video and text tokens into one sequence for joint attention and
// never pads, so there is no mask and no batch axis here either.

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sonicboom::moss {

struct DitConfig {
  int64_t dim = 3072;
  int64_t ffn_dim = 8192;
  int64_t num_heads = 24;
  int64_t num_layers = 42;
  int64_t text_dim = 4096;
  int64_t freq_dim = 256;
  int64_t latent_channels = 128;
  double rope_base = 10000.0;

  int64_t head_dim() const noexcept { return dim / num_heads; }
};

// The DiT output: one latent sample [latent_channels][height][width], f32.
struct DitSample {
  int64_t height = 0;
  int64_t width = 0;
  std::vector<float> values;
};

// Optional intermediate dumps for the S3 numerics gate.
struct DitDebug {
  std::vector<float> residual_after_first_block;  // [L][dim], L = h*w + text
  std::vector<float> prehead;                      // [h*w][dim]
};

class DiffusionTransformer {
 public:
  static std::expected<std::unique_ptr<DiffusionTransformer>, std::string>
  load(const std::filesystem::path& dir);

  DiffusionTransformer(const DiffusionTransformer&) = delete;
  DiffusionTransformer& operator=(const DiffusionTransformer&) = delete;
  ~DiffusionTransformer();

  const DitConfig& config() const noexcept;
  int64_t weight_bytes() const noexcept;

  // hidden: [latent_channels][height][width] f32 (row-major, channel-major).
  // text:   [text_len][text_dim] f32 (the encoder's last hidden state).
  // timestep: the flow-matching time (sigma * 1000, the scheduler's units).
  // skip_layers: block indices to omit entirely (the reference's skip_layers,
  //              e.g. {22} for MOSS-Image's unconditional pass).
  std::expected<DitSample, std::string> forward(
      std::span<const float> hidden, int64_t height, int64_t width,
      float timestep, std::span<const float> text, int64_t text_len,
      const std::vector<int64_t>& skip_layers = {},
      DitDebug* debug = nullptr) const;

 private:
  struct Impl;
  explicit DiffusionTransformer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonicboom::moss
