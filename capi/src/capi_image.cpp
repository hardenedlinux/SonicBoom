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

// Image generation through the stable C ABI (sb_image_*): the MOSS-Image
// pipeline, a request struct, progress/cancellation, RGB HWC output and a
// minimal PNG writer. All C++ failures are folded into sb_status/sb_error;
// no C++ type crosses the boundary.

#include <sonicboom/capi.h>

#include <sonicboom/model/moss_pipeline.h>
#include <sonicboom/model/moss_text.h>
#include <sonicboom/vocab/vocab.h>

#include "capi_private.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using sonicboom::capi_private::guard;
using sonicboom::capi_private::set_error;

namespace {

namespace moss = sonicboom::moss;

}  // namespace

struct sb_image_model {
  std::filesystem::path root;
  std::unique_ptr<moss::MossPipeline> pipeline;
  sonicboom::vocab::Vocab vocab;
};

// --- PNG writer ------------------------------------------------------------

namespace {

const uint32_t *crc32_table() {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1u) ? 0xedb88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  return table.data();
}

uint32_t crc32_bytes(const uint8_t *data, size_t n) {
  const uint32_t *table = crc32_table();
  uint32_t c = 0xffffffffu;
  for (size_t i = 0; i < n; ++i) c = table[(c ^ data[i]) & 0xffu] ^ (c >> 8);
  return c ^ 0xffffffffu;
}

uint32_t adler32_bytes(const uint8_t *data, size_t n) {
  uint32_t a = 1, b = 0;
  constexpr uint32_t kMod = 65521u;
  for (size_t i = 0; i < n; ++i) {
    a = (a + data[i]) % kMod;
    b = (b + a) % kMod;
  }
  return (b << 16) | a;
}

void put_be32(std::vector<uint8_t> &out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v >> 24));
  out.push_back(static_cast<uint8_t>(v >> 16));
  out.push_back(static_cast<uint8_t>(v >> 8));
  out.push_back(static_cast<uint8_t>(v));
}

// One PNG chunk: length, type, data, CRC-32 over type+data, all big-endian.
void put_chunk(std::vector<uint8_t> &out, const char type[4],
               const uint8_t *data, size_t n) {
  put_be32(out, static_cast<uint32_t>(n));
  const size_t start = out.size();
  out.insert(out.end(), type, type + 4);
  if (n) out.insert(out.end(), data, data + n);
  put_be32(out, crc32_bytes(out.data() + start, 4 + n));
}

// The zlib stream of stored (uncompressed) deflate blocks. Each block carries
// at most 65535 bytes; the final one sets BFINAL. The zlib header 0x78 0x01 is
// deflate/32K with its check bits, and the stream ends with Adler-32 of RAW.
std::vector<uint8_t> zlib_stored(const std::vector<uint8_t> &raw) {
  std::vector<uint8_t> z;
  z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
  z.push_back(0x78);
  z.push_back(0x01);
  size_t off = 0;
  do {
    const size_t n = std::min<size_t>(65535, raw.size() - off);
    const bool final = off + n == raw.size();
    z.push_back(final ? 1 : 0);  // BFINAL, BTYPE=00 (stored)
    z.push_back(static_cast<uint8_t>(n & 0xff));
    z.push_back(static_cast<uint8_t>(n >> 8));
    const uint16_t nlen = static_cast<uint16_t>(~n);
    z.push_back(static_cast<uint8_t>(nlen & 0xff));
    z.push_back(static_cast<uint8_t>(nlen >> 8));
    if (n) z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
    off += n;
  } while (off < raw.size());
  put_be32(z, adler32_bytes(raw.data(), raw.size()));
  return z;
}

// A field of a versioned request struct: read only when struct_size covers it.
template <typename T>
T field(const sb_image_request *req, size_t offset, T fallback) {
  if (req->struct_size >= offset + sizeof(T)) {
    T value;
    std::memcpy(&value, reinterpret_cast<const uint8_t *>(req) + offset,
                sizeof(T));
    return value;
  }
  return fallback;
}

sb_status cancelled(sb_error **err_out) {
  set_error(err_out, SB_ERRKIND_CANCELLED, "image generation cancelled");
  return SB_ERR_CANCELLED;
}

}  // namespace

// --- Model -----------------------------------------------------------------

sb_status sb_image_model_load(const char *dir, sb_image_model **out,
                              sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (out == nullptr || dir == nullptr || dir[0] == '\0')
          return SB_ERR_INTERNAL;
        *out = nullptr;
        const std::filesystem::path root(dir);
        auto pipeline = moss::MossPipeline::load(root, /*with_text=*/true);
        if (!pipeline) {
          set_error(err_out, SB_ERRKIND_WEIGHT, pipeline.error().c_str());
          return SB_ERR_UNSUPPORTED;
        }
        std::filesystem::path tok = root / "tokenizer" / "tokenizer.json";
        if (!std::filesystem::exists(tok)) tok = root / "tokenizer.json";
        auto vocab = sonicboom::vocab::load_tokenizer_json(tok);
        if (!vocab) {
          set_error(err_out, SB_ERRKIND_WEIGHT, vocab.error().c_str());
          return SB_ERR_UNSUPPORTED;
        }
        auto model = std::make_unique<sb_image_model>();
        model->root = root;
        model->pipeline = std::move(*pipeline);
        model->vocab = std::move(*vocab);
        *out = model.release();
        return SB_OK;
      },
      err_out);
}

void sb_image_model_free(sb_image_model *model) { delete model; }

sb_status sb_image_model_arch(const sb_image_model *model, char *out,
                              size_t cap, size_t *len_out,
                              sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (model == nullptr || len_out == nullptr) return SB_ERR_INTERNAL;
        static constexpr char kArch[] = "moss-image";
        // Same convention as sb_llm_model_arch: the length excludes the NUL,
        // and cap must leave room for it.
        const size_t needed = sizeof kArch - 1;
        *len_out = needed;
        if (out == nullptr || cap <= needed) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "the buffer is too small for the architecture name");
          return SB_ERR_BINDING;
        }
        std::memcpy(out, kArch, needed + 1);
        return SB_OK;
      },
      err_out);
}

// --- Generation ------------------------------------------------------------

sb_status sb_image_generate(sb_image_model *model, const sb_image_request *req,
                            sb_image_progress_cb cb, void *user,
                            uint8_t *rgb_out, size_t cap, size_t *bytes_out,
                            sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (model == nullptr || req == nullptr || bytes_out == nullptr)
          return SB_ERR_INTERNAL;

        const char *prompt = field<const char *>(req, offsetof(sb_image_request, prompt), nullptr);
        const char *negative =
            field<const char *>(req, offsetof(sb_image_request, negative_prompt), nullptr);
        const auto width =
            field<uint32_t>(req, offsetof(sb_image_request, width), 0);
        const auto height =
            field<uint32_t>(req, offsetof(sb_image_request, height), 0);
        const auto steps =
            field<uint32_t>(req, offsetof(sb_image_request, steps), 0);
        const auto guidance =
            field<float>(req, offsetof(sb_image_request, guidance_scale), 3.0f);
        const auto seed =
            field<uint64_t>(req, offsetof(sb_image_request, seed), 0);
        const auto batch =
            field<uint32_t>(req, offsetof(sb_image_request, batch), 1);

        if (prompt == nullptr) {
          set_error(err_out, SB_ERRKIND_BINDING, "prompt must not be NULL");
          return SB_ERR_BINDING;
        }
        if (width < 16 || height < 16 || width % 16 || height % 16) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "width and height must be multiples of 16");
          return SB_ERR_BINDING;
        }
        if (steps < 1) {
          set_error(err_out, SB_ERRKIND_BINDING, "steps must be >= 1");
          return SB_ERR_BINDING;
        }
        if (batch < 1) {
          set_error(err_out, SB_ERRKIND_BINDING, "batch must be >= 1");
          return SB_ERR_BINDING;
        }
        if (!std::isfinite(guidance)) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "guidance_scale must be finite");
          return SB_ERR_BINDING;
        }

        // Two-call sizing: the byte count is known from the request, before
        // any device work.
        const size_t pixels =
            static_cast<size_t>(batch) * height * width;
        if (pixels / batch / height != width ||
            pixels > std::numeric_limits<size_t>::max() / 3) {
          set_error(err_out, SB_ERRKIND_INTERNAL, "image buffer size overflow");
          return SB_ERR_INTERNAL;
        }
        const size_t needed = pixels * 3;
        *bytes_out = needed;
        if (rgb_out == nullptr || cap < needed) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "rgb_out is NULL or too small; *bytes_out has the needed "
                    "byte count");
          return SB_ERR_BINDING;
        }

        // Conditioning: the pipeline's chat rendering, then one encoder pass
        // per prompt (the reference does the same, with no padding).
        auto encode = [&](const std::string &text,
                          std::vector<float> &dst, int64_t &len) -> sb_status {
          const std::string rendered = moss::moss_chat_prompt(text);
          const auto ids =
              sonicboom::vocab::tokenize(model->vocab, rendered, true, false);
          auto hidden = model->pipeline->encode(ids);
          if (!hidden) {
            set_error(err_out, SB_ERRKIND_RUNTIME, hidden.error().c_str());
            return SB_ERR_RUNTIME;
          }
          dst = std::move(hidden->values);
          len = hidden->tokens;
          return SB_OK;
        };
        moss::MossConditioning cond;
        if (const sb_status st = encode(prompt, cond.text, cond.text_len);
            st != SB_OK)
          return st;
        const bool do_cfg = guidance != 1.0f;
        if (do_cfg) {
          const sb_status st = encode(negative ? negative : "", cond.negative,
                                      cond.negative_len);
          if (st != SB_OK) return st;
        }

        moss::MossPipelineConfig cfg;
        cfg.steps = steps;
        cfg.guidance_scale = guidance;
        const auto &dit_cfg = model->pipeline->transformer()->config();
        const int64_t grid_h = height / 16, grid_w = width / 16;
        std::vector<moss::MossLatent> latents;
        latents.reserve(batch);
        for (uint32_t i = 0; i < batch; ++i) {
          moss::MossLatent l;
          l.channels = dit_cfg.latent_channels;
          l.height = grid_h;
          l.width = grid_w;
          l.values = moss::seeded_noise(l.count(),
                                        seed + static_cast<uint64_t>(i));
          latents.push_back(std::move(l));
        }
        std::vector<moss::MossConditioning> conds(latents.size(), cond);

        // The C callback returns non-zero to cancel; the pipeline's boolean
        // progress callback returning false stops with "cancelled".
        moss::MossProgress progress;
        bool was_cancelled = false;
        if (cb != nullptr) {
          progress = [&](int64_t image_index, int64_t step,
                         int64_t total) -> bool {
            if (cb(user, static_cast<uint32_t>(image_index),
                   static_cast<uint32_t>(step),
                   static_cast<uint32_t>(total)) != 0) {
              was_cancelled = true;
              return false;
            }
            return true;
          };
        }

        auto sampled =
            model->pipeline->sample_batch(latents, conds, cfg, nullptr, nullptr,
                                          progress);
        if (!sampled) {
          if (was_cancelled || sampled.error() == "cancelled")
            return cancelled(err_out);
          set_error(err_out, SB_ERRKIND_RUNTIME, sampled.error().c_str());
          return SB_ERR_RUNTIME;
        }

        std::vector<moss::MossImage> images;
        images.reserve(sampled->size());
        for (const auto &latent : *sampled) {
          auto refined = model->pipeline->refine(latent);
          if (!refined) {
            set_error(err_out, SB_ERRKIND_RUNTIME, refined.error().c_str());
            return SB_ERR_RUNTIME;
          }
          auto image = model->pipeline->decode(*refined);
          if (!image) {
            set_error(err_out, SB_ERRKIND_RUNTIME, image.error().c_str());
            return SB_ERR_RUNTIME;
          }
          if (image->channels != 3 ||
              image->height != static_cast<int64_t>(height) ||
              image->width != static_cast<int64_t>(width)) {
            set_error(err_out, SB_ERRKIND_INTERNAL,
                      "decoded image does not match the requested size");
            return SB_ERR_INTERNAL;
          }
          images.push_back(std::move(*image));
        }

        // [-1, 1] CHW floats -> uint8 RGB HWC: clamp, scale, round.
        for (size_t i = 0; i < images.size(); ++i) {
          const auto &img = images[i];
          uint8_t *dst = rgb_out + i * static_cast<size_t>(height) * width * 3;
          const int64_t plane = img.height * img.width;
          for (int64_t y = 0; y < img.height; ++y) {
            for (int64_t x = 0; x < img.width; ++x) {
              uint8_t *px = dst + (y * static_cast<int64_t>(width) + x) * 3;
              for (int64_t c = 0; c < 3; ++c) {
                const float v = img.values[static_cast<size_t>(c * plane +
                                                               y * img.width +
                                                               x)];
                const float t = std::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f);
                px[c] = static_cast<uint8_t>(std::lrintf(t * 255.0f));
              }
            }
          }
        }
        *bytes_out = needed;
        return SB_OK;
      },
      err_out);
}

sb_status sb_image_write_png(const char *path, const uint8_t *rgb,
                             uint32_t width, uint32_t height,
                             sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (path == nullptr || rgb == nullptr || width == 0 || height == 0) {
          set_error(err_out, SB_ERRKIND_INTERNAL,
                    "png: path, rgb, width and height are required");
          return SB_ERR_INTERNAL;
        }
        const size_t row_bytes = static_cast<size_t>(width) * 3;
        std::vector<uint8_t> raw;
        raw.reserve(static_cast<size_t>(height) * (row_bytes + 1));
        for (uint32_t y = 0; y < height; ++y) {
          raw.push_back(0);  // filter type 0 (None)
          const uint8_t *row = rgb + static_cast<size_t>(y) * row_bytes;
          raw.insert(raw.end(), row, row + row_bytes);
        }

        std::vector<uint8_t> file;
        static const uint8_t kSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
        file.insert(file.end(), kSignature, kSignature + 8);
        uint8_t ihdr[13];
        ihdr[0] = static_cast<uint8_t>(width >> 24);
        ihdr[1] = static_cast<uint8_t>(width >> 16);
        ihdr[2] = static_cast<uint8_t>(width >> 8);
        ihdr[3] = static_cast<uint8_t>(width);
        ihdr[4] = static_cast<uint8_t>(height >> 24);
        ihdr[5] = static_cast<uint8_t>(height >> 16);
        ihdr[6] = static_cast<uint8_t>(height >> 8);
        ihdr[7] = static_cast<uint8_t>(height);
        ihdr[8] = 8;   // bit depth
        ihdr[9] = 2;   // colour type: truecolour RGB
        ihdr[10] = 0;  // compression: deflate
        ihdr[11] = 0;  // filter method
        ihdr[12] = 0;  // interlace: none
        put_chunk(file, "IHDR", ihdr, sizeof ihdr);
        const std::vector<uint8_t> z = zlib_stored(raw);
        put_chunk(file, "IDAT", z.data(), z.size());
        put_chunk(file, "IEND", nullptr, 0);

        std::ofstream out(path, std::ios::binary);
        if (!out) {
          set_error(err_out, SB_ERRKIND_RUNTIME,
                    "png: cannot open the output file");
          return SB_ERR_RUNTIME;
        }
        out.write(reinterpret_cast<const char *>(file.data()),
                  static_cast<std::streamsize>(file.size()));
        if (!out) {
          set_error(err_out, SB_ERRKIND_RUNTIME, "png: write failed");
          return SB_ERR_RUNTIME;
        }
        return SB_OK;
      },
      err_out);
}
