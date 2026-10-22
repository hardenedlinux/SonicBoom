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

// MOSS-Image inference pipeline: the denoising loop (MOSSImageScheduler, a
// flow-matching Euler sampler on a shifted-and-snapped training grid), CFG
// with the layer-22 skipped unconditional pass, optional Skip-Layer Guidance,
// and the latent decode path (BatchNorm unnormalization, 2x2 pixel-unshuffle,
// FLUX.2 VAE decode, clamp to [-1, 1]).
//
// The reference is pipeline_moss_image.py's MOSSImageScheduler and
// MOSSImagePipeline.__call__. Arithmetic is float32, as the reference's
// sampler is; the DiT and the VAE run in this project's f32 execution policy.
//
// Layer 2: plain C++ types only.

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <sonicboom/model/moss_dit.h>
#include <sonicboom/model/moss_text.h>
#include <sonicboom/model/moss_vae.h>

namespace sonicboom::moss {

// The MOSSImageScheduler schedule in the scheduler's own units: `sigmas` has
// one entry per step plus the appended terminal 0, `timesteps` is
// sigmas * 1000 (the values the DiT consumes).
struct FlowSchedule {
  std::vector<float> sigmas;
  std::vector<float> timesteps;
};

// `steps` sigmas from the shift-`shift` grid, snapped to the nearest point of
// the `train_steps`-point training grid (training_shift), with consecutive
// duplicates removed. This is MOSSImageScheduler.set_timesteps.
FlowSchedule moss_flow_schedule(int64_t steps, float shift = 5.0f,
                                int64_t train_steps = 1000,
                                float training_shift = 1.0f);

// One image's latent, [channels][height][width] float32 row-major.
struct MossLatent {
  int64_t channels = 0, height = 0, width = 0;
  std::vector<float> values;

  int64_t count() const noexcept { return channels * height * width; }
};

// A decoded image, [channels][height][width] float32 in [-1, 1].
struct MossImage {
  int64_t channels = 0, height = 0, width = 0;
  std::vector<float> values;
};

// The sampler's knobs. Defaults are the reference pipeline's defaults.
struct MossPipelineConfig {
  int64_t steps = 50;
  float guidance_scale = 3.0f;
  // Blocks omitted from the unconditional pass while skipping is active
  // (the reference's uncond_skip_layers).
  std::vector<int64_t> uncond_skip_layers{22};
  // Extra skipped positive forward for Skip-Layer Guidance.
  std::vector<int64_t> slg_layers{};
  float slg_scale = 0.0f;
  // Skipping is active where slg_start <= step_index / steps < slg_stop.
  double slg_start = 0.0;
  double slg_stop = 1.0;
};

// The reference's transformer never pads: each sample is conditioned on its
// own unpadded encoder output.
struct MossConditioning {
  std::vector<float> text;      // [text_len][text_dim]
  int64_t text_len = 0;
  std::vector<float> negative;  // [negative_len][text_dim], used when CFG runs
  int64_t negative_len = 0;
};

struct MossSampleStats {
  int64_t forwards = 0;  // DiT evaluations (2 per step with CFG)
  double seconds = 0.0;
};

class MossPipeline {
 public:
  // Opens the components under `root` (text_encoder/, transformer/, vae/).
  // `with_text` also maps the 16.8 GiB text encoder.
  static std::expected<std::unique_ptr<MossPipeline>, std::string> load(
      const std::filesystem::path& root, bool with_text = true);

  MossPipeline(const MossPipeline&) = delete;
  MossPipeline& operator=(const MossPipeline&) = delete;
  ~MossPipeline();

  const TextEncoder* text_encoder() const noexcept;
  const DiffusionTransformer* transformer() const noexcept;
  const VaeDecoder* vae() const noexcept;

  // The encoder's last hidden state for `ids` (requires with_text).
  std::expected<HiddenStates, std::string> encode(
      std::span<const int32_t> ids) const;

  // Denoise one latent. `latent` must be [latent_channels][height][width]
  // with height and width the diffusion grid (image size / 16). The latent is
  // updated in place without touching the caller's buffer.
  std::expected<MossLatent, std::string> sample(
      const MossLatent& latent, const MossConditioning& cond,
      const MossPipelineConfig& cfg, std::vector<MossLatent>* trace = nullptr,
      MossSampleStats* stats = nullptr) const;

  // Images are denoised one at a time, exactly as the reference loops over its
  // batch, so each result is bit-identical to sampling it alone.
  std::expected<std::vector<MossLatent>, std::string> sample_batch(
      std::span<const MossLatent> latents,
      std::span<const MossConditioning> conds, const MossPipelineConfig& cfg,
      std::vector<std::vector<MossLatent>>* trace = nullptr,
      MossSampleStats* stats = nullptr) const;

  // BatchNorm unnormalization, 2x2 pixel-unshuffle, VAE decode, clamp(-1, 1).
  std::expected<MossImage, std::string> decode(const MossLatent& latent) const;

  // The pipeline's refine stage. The MOSS-Image checkpoint ships no refiner,
  // so this is an identity stub: it validates the latent and returns it.
  std::expected<MossLatent, std::string> refine(const MossLatent& latent) const;

 private:
  struct Impl;
  explicit MossPipeline(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace sonicboom::moss
