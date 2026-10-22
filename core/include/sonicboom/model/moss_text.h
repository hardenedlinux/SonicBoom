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

// MOSS-Image text encoder: the Qwen3.5 text model, run as a SonicBoom CPU
// executor over the safetensors checkpoint.
//
// The conditioning signal MOSS-Image's DiT consumes is this model's last
// hidden state, so it is a full transformer forward that SonicBoom must
// compute itself. The architecture is the Qwen3.5 hybrid: 32 layers, 8 of
// which are gated full attention (16 query heads / 4 KV heads, head_dim 256,
// partial RoPE over the first 64 dims, an output gate on the attention) and 24
// of which are gated-delta-net linear attention (depthwise causal conv,
// per-head L2-normalized q/k, a per-head 128x128 recurrent state and a gated
// RMSNorm). Layer norms use Qwen3.5's (1 + weight) convention.
//
// Executor choice: the checkpoint is ~16.8 GiB of bf16 weights, which does not
// fit the 8 GiB device even at Q8_0 (~9 GiB). Running it on the CPU directly
// from the mmap'd bf16 files needs no quantization, so it is both the
// only fitting option and the numerically exact one; the choice is recorded in
// the decision registry (name "moss.encoder.executor") and can be pinned with
// SONICBOOM_MOSS_ENCODER.
//
// This header is Layer 2: it carries plain C++ types only, and no tensor bytes
// ever become a SonicBoom Tensor here. The encoder is used once per prompt to
// produce the conditioning tensor; nothing in it is on the image path.

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sonicboom::moss {

// The fields of config.json's "text_config" the forward needs.
struct TextEncoderConfig {
  int64_t hidden_size = 0;
  int64_t num_layers = 0;
  int64_t intermediate_size = 0;
  int64_t head_dim = 0;          // full-attention head dim (query/KV)
  int64_t num_heads = 0;         // full-attention query heads
  int64_t num_kv_heads = 0;      // full-attention KV heads
  int64_t conv_kernel = 0;       // linear-attention depthwise conv width
  int64_t key_head_dim = 0;      // linear-attention head dims
  int64_t num_key_heads = 0;
  int64_t value_head_dim = 0;
  int64_t num_value_heads = 0;
  int64_t vocab_size = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;
  int64_t rotary_dim = 0;        // head_dim * partial_rotary_factor
  std::vector<uint8_t> layer_is_linear;  // one flag per layer
};

// The result of one forward: row-major [tokens, hidden_size] float32.
struct HiddenStates {
  int64_t tokens = 0;
  int64_t hidden_size = 0;
  std::vector<float> values;
};

class TextEncoder {
 public:
  // Open the sharded checkpoint and config under `dir` (a text_encoder
  // directory: config.json + model.safetensors.index.json). Weights stay
  // memory-mapped; no copy is made.
  static std::expected<std::unique_ptr<TextEncoder>, std::string> load(
      const std::filesystem::path& dir);

  ~TextEncoder();
  TextEncoder(TextEncoder&&) noexcept;
  TextEncoder& operator=(TextEncoder&&) noexcept;
  TextEncoder(const TextEncoder&) = delete;
  TextEncoder& operator=(const TextEncoder&) = delete;

  // Full (unpadded) forward over `ids`; the last hidden state is returned in
  // float32. Sequential state (the delta-net recurrence and the attention
  // window) starts empty, matching the reference's use_cache=False path.
  std::expected<HiddenStates, std::string> forward(std::span<const int32_t> ids);

  const TextEncoderConfig& config() const noexcept;

  // Bytes of checkpoint tensor data mapped (sum over text-model tensors).
  int64_t weight_bytes() const noexcept;

 private:
  struct Impl;
  explicit TextEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonicboom::moss
