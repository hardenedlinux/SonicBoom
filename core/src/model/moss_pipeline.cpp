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

// MOSS-Image pipeline executor: the reference's MOSSImageScheduler and
// MOSSImagePipeline.__call__ transcribed to f32. The scheduler's grid and
// snapping are reproduced operation for operation in single precision, since
// the timesteps the DiT sees come from those snapped values.

#include <sonicboom/model/moss_pipeline.h>

#include <cstdlib>

#if defined(SONICBOOM_USE_CUDA)
#include <sonicboom/model/moss_dit_cuda.h>
#endif

#include "../planner/decision_registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace sonicboom::moss {

namespace {

// grid(count, shift): `count` values from 1 down to lower, warped by shift.
// The reference evaluates each row in float32 (all grid inputs are f32).
void grid_into(std::vector<float>& values, int64_t count, double shift,
               int64_t total) {
  const double lower = shift / (static_cast<double>(total) + shift - 1.0);
  values.resize(static_cast<size_t>(count));
  const float s = static_cast<float>(shift);
  for (int64_t i = 0; i < count; ++i) {
    const double v = count == 1
                         ? 1.0
                         : 1.0 + static_cast<double>(i) * (lower - 1.0) /
                                     static_cast<double>(count - 1);
    const float vf = static_cast<float>(v);
    values[static_cast<size_t>(i)] = s * vf / (1.0f + (s - 1.0f) * vf);
  }
}

}  // namespace

FlowSchedule moss_flow_schedule(int64_t steps, float shift,
                                int64_t train_steps, float training_shift) {
  FlowSchedule out;
  if (steps < 1 || train_steps < 1 || !(shift > 0.0f) ||
      !(training_shift > 0.0f))
    return out;

  std::vector<float> sigmas;
  grid_into(sigmas, steps, static_cast<double>(shift), train_steps);

  // The training grid, descending 1 -> 1/total, flipped to ascending.
  std::vector<float> training;
  grid_into(training, train_steps, static_cast<double>(training_shift),
            train_steps);
  std::reverse(training.begin(), training.end());

  // searchsorted(training, sigmas), clamped to the last index.
  auto searchsorted = [&](float v) {
    auto it = std::lower_bound(training.begin(), training.end(), v);
    int64_t idx = static_cast<int64_t>(it - training.begin());
    return std::min(idx, train_steps - 1);
  };

  std::vector<float> snapped;
  snapped.reserve(sigmas.size());
  float last = 0.0f;
  bool have_last = false;
  for (float sigma : sigmas) {
    const int64_t upper = searchsorted(sigma);
    const int64_t lower = std::max<int64_t>(upper - 1, 0);
    const float to_lower = std::fabs(sigma - training[static_cast<size_t>(lower)]);
    const float to_upper = std::fabs(training[static_cast<size_t>(upper)] - sigma);
    const float pick = to_lower <= to_upper ? training[static_cast<size_t>(lower)]
                                            : training[static_cast<size_t>(upper)];
    // unique_consecutive
    if (!have_last || pick != last) {
      snapped.push_back(pick);
      last = pick;
      have_last = true;
    }
  }

  out.timesteps.resize(snapped.size());
  for (size_t i = 0; i < snapped.size(); ++i)
    out.timesteps[i] = snapped[i] * static_cast<float>(train_steps);
  out.sigmas = snapped;
  out.sigmas.push_back(0.0f);
  return out;
}

struct MossPipeline::Impl {
  std::unique_ptr<TextEncoder> text;
  std::unique_ptr<DiffusionTransformer> dit;
#if defined(SONICBOOM_USE_CUDA)
  // The device-resident Q8_0 DiT (M2). When present it replaces the f32 CPU
  // executor for every denoising forward; the CPU transformer stays loaded as
  // the reference/fallback and for the public transformer() accessor.
  std::unique_ptr<CudaDiffusionTransformer> dit_cuda;
#endif
  std::unique_ptr<VaeDecoder> vae;

  const DitConfig& dit_config() const noexcept {
#if defined(SONICBOOM_USE_CUDA)
    if (dit_cuda) return dit_cuda->config();
#endif
    return dit->config();
  }

  std::expected<DitSample, std::string> forward(
      std::span<const float> hidden, int64_t height, int64_t width,
      float timestep, std::span<const float> text, int64_t text_len,
      const std::vector<int64_t>& skip) const {
#if defined(SONICBOOM_USE_CUDA)
    if (dit_cuda)
      return dit_cuda->forward(hidden, height, width, timestep, text, text_len,
                               skip);
#endif
    return dit->forward(hidden, height, width, timestep, text, text_len, skip);
  }
};

MossPipeline::MossPipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MossPipeline::~MossPipeline() = default;

const TextEncoder* MossPipeline::text_encoder() const noexcept {
  return impl_->text.get();
}
const DiffusionTransformer* MossPipeline::transformer() const noexcept {
  return impl_->dit.get();
}
const VaeDecoder* MossPipeline::vae() const noexcept { return impl_->vae.get(); }

std::expected<std::unique_ptr<MossPipeline>, std::string> MossPipeline::load(
    const std::filesystem::path& root, bool with_text) {
  auto impl = std::make_unique<Impl>();
  if (with_text) {
    auto text = TextEncoder::load(root / "text_encoder");
    if (!text) return std::unexpected("text encoder: " + text.error());
    impl->text = std::move(*text);
  }
  auto dit = DiffusionTransformer::load(root / "transformer");
  if (!dit) return std::unexpected("transformer: " + dit.error());
  impl->dit = std::move(*dit);

  // R2: the DiT executor is an automatic choice, made once per model/device
  // and recorded in the decision registry. The device path quantizes the
  // checkpoint to Q8_0 at load and keeps every activation resident; it is the
  // only one that meets the image-generation time budget, so it wins when a
  // device is present and it loads. SONICBOOM_MOSS_DIT=cpu pins the f32 CPU
  // executor, =cuda makes a device failure fatal instead of falling back.
  {
    const char* env = std::getenv("SONICBOOM_MOSS_DIT");
    const bool want_cpu = env != nullptr && std::string(env) == "cpu";
    const bool want_cuda = env != nullptr && std::string(env) == "cuda";
    planner::decisions::Entry entry;
    entry.name = "moss.dit.executor";
    entry.scope = "transformer";
    entry.env = "SONICBOOM_MOSS_DIT";
    entry.candidates = "cpu-f32-mmap|cuda-q8_0";
    entry.numerics = "cpu: f32 activations, bf16 weights; cuda: f32 "
                     "activations, Q8_0 weights";
#if defined(SONICBOOM_USE_CUDA)
    std::string cuda_error;
    if (!want_cpu && moss::dit_cuda_available()) {
      auto cuda = CudaDiffusionTransformer::load(root / "transformer");
      if (cuda) {
        impl->dit_cuda = std::move(*cuda);
      } else {
        cuda_error = cuda.error();
      }
    } else if (!want_cpu) {
      cuda_error = "no CUDA device";
    }
    if (impl->dit_cuda) {
      entry.value = "cuda-q8_0";
      entry.source = want_cuda ? planner::decisions::Source::Env
                               : planner::decisions::Source::Default;
      entry.inputs = "weights=" +
                     std::to_string(impl->dit_cuda->weight_bytes()) +
                     "B device=8GiB";
      entry.reason = "Q8_0 fits the device and the per-forward cost is the "
                     "image-generation budget; the CPU executor is 14 s per "
                     "forward";
      entry.vram_bytes = impl->dit_cuda->weight_bytes();
    } else {
      if (want_cuda)
        return std::unexpected("transformer: CUDA DiT requested but " +
                               cuda_error);
      entry.value = "cpu-f32-mmap";
      entry.source = planner::decisions::Source::Default;
      entry.inputs = "weights=" + std::to_string(impl->dit->weight_bytes()) +
                     "B device=8GiB";
      entry.reason = cuda_error.empty()
                         ? "pinned by SONICBOOM_MOSS_DIT"
                         : "device path unavailable (" + cuda_error + ")";
      entry.vram_bytes = 0;
    }
#else
    (void)want_cpu;
    (void)want_cuda;
    entry.value = "cpu-f32-mmap";
    entry.source = planner::decisions::Source::Default;
    entry.reason = "built without CUDA";
    entry.vram_bytes = 0;
#endif
    planner::decisions::record(entry);
  }

  auto vae = VaeDecoder::load(root / "vae");
  if (!vae) return std::unexpected("vae: " + vae.error());
  impl->vae = std::move(*vae);
  return std::unique_ptr<MossPipeline>(new MossPipeline(std::move(impl)));
}

std::expected<HiddenStates, std::string> MossPipeline::encode(
    std::span<const int32_t> ids) const {
  if (impl_->text == nullptr)
    return std::unexpected("pipeline loaded without the text encoder");
  return impl_->text->forward(ids);
}

std::expected<MossLatent, std::string> MossPipeline::sample(
    const MossLatent& latent, const MossConditioning& cond,
    const MossPipelineConfig& cfg, std::vector<MossLatent>* trace,
    MossSampleStats* stats) const {
  const DitConfig& dit_cfg = impl_->dit_config();
  if (latent.channels != dit_cfg.latent_channels || latent.height <= 0 ||
      latent.width <= 0)
    return std::unexpected("sample: latent shape does not match the DiT");
  if (static_cast<int64_t>(latent.values.size()) != latent.count())
    return std::unexpected("sample: latent buffer size mismatch");
  const int64_t text_dim = dit_cfg.text_dim;
  if (cond.text_len < 1 ||
      static_cast<int64_t>(cond.text.size()) != cond.text_len * text_dim)
    return std::unexpected("sample: conditioning text size mismatch");
  const bool do_cfg = cfg.guidance_scale != 1.0f;
  if (do_cfg && (cond.negative_len < 1 ||
                 static_cast<int64_t>(cond.negative.size()) !=
                     cond.negative_len * text_dim))
    return std::unexpected("sample: CFG needs a negative conditioning");
  if (cfg.steps < 1)
    return std::unexpected("sample: steps must be positive");

  const FlowSchedule schedule = moss_flow_schedule(cfg.steps);
  if (schedule.timesteps.empty())
    return std::unexpected("sample: empty schedule");
  const int64_t steps = static_cast<int64_t>(schedule.timesteps.size());

  MossLatent current = latent;
  std::vector<float> pred;
  std::vector<float> conditional;
  static const std::vector<int64_t> kNoSkip;
  const auto start = std::chrono::steady_clock::now();
  int64_t forwards = 0;

  for (int64_t step = 0; step < steps; ++step) {
    const float timestep = schedule.timesteps[static_cast<size_t>(step)];
    const double fraction = static_cast<double>(step) / static_cast<double>(steps);
    const bool skip_active =
        cfg.slg_start <= fraction && fraction < cfg.slg_stop;

    auto cond_out = impl_->forward(
        current.values, current.height, current.width, timestep, cond.text,
        cond.text_len, kNoSkip);
    if (!cond_out) return std::unexpected(cond_out.error());
    ++forwards;
    pred = cond_out->values;
    const bool need_conditional =
        !cfg.slg_layers.empty() && cfg.slg_scale > 0.0f && skip_active;
    if (need_conditional) conditional = cond_out->values;

    if (do_cfg) {
      const std::vector<int64_t>& skip =
          (skip_active && !cfg.uncond_skip_layers.empty())
              ? cfg.uncond_skip_layers
              : kNoSkip;
      auto uncond_out = impl_->forward(
          current.values, current.height, current.width, timestep, cond.negative,
          cond.negative_len, skip);
      if (!uncond_out) return std::unexpected(uncond_out.error());
      ++forwards;
      const float scale = cfg.guidance_scale;
      for (size_t i = 0; i < pred.size(); ++i)
        pred[i] = uncond_out->values[i] + scale * (pred[i] - uncond_out->values[i]);
    }

    if (need_conditional) {
      auto skipped_out = impl_->forward(
          current.values, current.height, current.width, timestep, cond.text,
          cond.text_len, cfg.slg_layers);
      if (!skipped_out) return std::unexpected(skipped_out.error());
      ++forwards;
      if (!do_cfg) pred = skipped_out->values;
      const float slg = cfg.slg_scale;
      for (size_t i = 0; i < pred.size(); ++i)
        pred[i] += slg * (conditional[i] - skipped_out->values[i]);
    }

    const float dt = schedule.sigmas[static_cast<size_t>(step) + 1] -
                     schedule.sigmas[static_cast<size_t>(step)];
    for (size_t i = 0; i < current.values.size(); ++i)
      current.values[i] += dt * pred[i];

    if (trace != nullptr) trace->push_back(current);
  }

  if (stats != nullptr) {
    stats->forwards = forwards;
    stats->seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  }
  return current;
}

std::expected<std::vector<MossLatent>, std::string> MossPipeline::sample_batch(
    std::span<const MossLatent> latents, std::span<const MossConditioning> conds,
    const MossPipelineConfig& cfg,
    std::vector<std::vector<MossLatent>>* trace,
    MossSampleStats* stats) const {
  if (latents.size() != conds.size())
    return std::unexpected("sample_batch: latents and conditionings differ in count");
  std::vector<MossLatent> out;
  out.reserve(latents.size());
  double seconds = 0.0;
  int64_t forwards = 0;
  for (size_t i = 0; i < latents.size(); ++i) {
    MossSampleStats one;
    auto got = sample(latents[i], conds[i], cfg,
                      trace != nullptr ? &(*trace)[i] : nullptr, &one);
    if (!got) return std::unexpected(got.error());
    out.push_back(std::move(*got));
    seconds += one.seconds;
    forwards += one.forwards;
  }
  if (stats != nullptr) {
    stats->forwards = forwards;
    stats->seconds = seconds;
  }
  return out;
}

std::expected<MossImage, std::string> MossPipeline::decode(
    const MossLatent& latent) const {
  const VaeConfig& vae_cfg = impl_->vae->config();
  const VaeBatchNorm& bn = impl_->vae->batch_norm();
  const int64_t L = latent.channels;
  if (L != vae_cfg.latent_channels * 4 || latent.height <= 0 ||
      latent.width <= 0)
    return std::unexpected("decode: latent shape does not match the VAE");
  if (static_cast<int64_t>(bn.running_mean.size()) != L ||
      static_cast<int64_t>(bn.running_var.size()) != L)
    return std::unexpected("decode: VAE BatchNorm statistics unavailable");
  if (static_cast<int64_t>(latent.values.size()) != latent.count())
    return std::unexpected("decode: latent buffer size mismatch");

  const int64_t h = latent.height, w = latent.width;
  const int64_t in_c = L / 4;
  std::vector<float> unshuffled(
      static_cast<size_t>(in_c * 2 * h * 2 * w));
  const double eps = bn.eps;
  for (int64_t c = 0; c < in_c; ++c) {
    for (int64_t a = 0; a < 2; ++a) {
      for (int64_t b = 0; b < 2; ++b) {
        const int64_t src_ch = c * 4 + a * 2 + b;
        const float scale = static_cast<float>(
            std::sqrt(static_cast<double>(bn.running_var[static_cast<size_t>(src_ch)]) + eps));
        const float shift = bn.running_mean[static_cast<size_t>(src_ch)];
        const float* src = latent.values.data() + src_ch * h * w;
        for (int64_t y = 0; y < h; ++y) {
          for (int64_t x = 0; x < w; ++x) {
            const int64_t dst = ((c * 2 * h + 2 * y + a) * 2 * w) + 2 * x + b;
            unshuffled[static_cast<size_t>(dst)] =
                src[y * w + x] * scale + shift;
          }
        }
      }
    }
  }

  // The Q8_0 DiT holds ~5.4 GiB of the 8 GiB device; the VAE's conv peak does
  // not fit beside it. Release the DiT first (the host mirror stays, so a later
  // sample re-uploads without re-quantizing).
#if defined(SONICBOOM_USE_CUDA)
  if (impl_->dit_cuda) impl_->dit_cuda->release_device();
#endif
  auto decoded = impl_->vae->decode(unshuffled, 2 * h, 2 * w);
  if (!decoded) return std::unexpected(decoded.error());

  MossImage image;
  image.channels = vae_cfg.out_channels;
  image.height = 16 * h;
  image.width = 16 * w;
  image.values = std::move(*decoded);
  for (float& v : image.values) v = std::min(1.0f, std::max(-1.0f, v));
  return image;
}

std::expected<MossLatent, std::string> MossPipeline::refine(
    const MossLatent& latent) const {
  // No refiner exists in this checkpoint; the stage is an identity so the
  // pipeline shape is complete and the latent handle round-trips.
  if (latent.channels <= 0 || latent.height <= 0 || latent.width <= 0 ||
      static_cast<int64_t>(latent.values.size()) != latent.count())
    return std::unexpected("refine: malformed latent");
  return latent;
}

}  // namespace sonicboom::moss
