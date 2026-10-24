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

// sb-moss — MOSS-Image bring-up driver.
//
//   sb-moss encode --model DIR (--text TEXT | --tokens "1,2,3") --out FILE
//                  [--raw] [--dump-tokens]
//
// Runs the Qwen3.5 text encoder over a prompt and writes the last hidden state
// as "MOSSHST1", int64 tokens, int64 hidden, then float32[tokens * hidden].
// --raw tokenizes TEXT exactly as given; the default wraps it in the encoder's
// chat template (the same rendering the MOSS-Image pipeline's encode_prompt
// uses with enable_thinking=True). --dump-tokens prints the ids only and does
// not run the model.

#include <sonicboom/model/moss_dit.h>
#if defined(SONICBOOM_TOOL_CUDA)
#include <sonicboom/model/moss_dit_cuda.h>
#endif
#include <sonicboom/model/moss_text.h>
#include <sonicboom/model/moss_pipeline.h>
#include <sonicboom/model/moss_vae.h>
#include <sonicboom/vocab/vocab.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: sb-moss encode --model DIR (--text TEXT | --tokens LIST) --out FILE\n"
         "                      [--raw] [--dump-tokens]\n"
         "       sb-moss tokens --model DIR (--text TEXT | --chat) [--raw]\n"
         "       sb-moss dit --model DIR --input FILE --out FILE [--skip LIST]\n"
         "\n"
         "  --model DIR    a text_encoder directory (config.json + index + shards)\n"
         "                 or, for `dit`, a transformer directory\n"
         "  --text TEXT    prompt text\n"
         "  --tokens LIST  comma-separated token ids (skips the tokenizer)\n"
         "  --raw          do not apply the chat template\n"
         "  --dump-tokens  print the token ids and exit\n"
         "  --out FILE     hidden-state output (`encode`) / DiT output (`dit`)\n"
         "  --input FILE   a MOSSDIT1 reference file (`dit`): its inputs are run\n"
         "                 and its outputs are replaced by ours\n"
         "  --skip LIST    block indices to omit, e.g. 22 (`dit`)\n"
         "  --cuda         run `dit` on the Q8_0 device-resident CUDA path\n"
         "  --stages FILE  dump per-stage activations (`vae`, MOSVSG01)\n"
         "\n"
         "sample: image size is height x width (multiples of 16); the DiT grid is\n"
         "height/16 x width/16. --embeds/--neg-embeds take MOSSHST1 files (the\n"
         "text encoder's output, e.g. from `encode` or a reference dump); --text\n"
         "runs this project's tokenizer and encoder. --latent-in takes a MOSLAT01\n"
         "file (int64 B,C,H,W then f32), --seed draws the initial noise instead.\n";
}

// The pipeline's chat rendering for one user turn with enable_thinking=True
// (MOSS-Image's encode_prompt): the Qwen3.5 template emits no system block for
// a single-user conversation, so the text is the user turn plus the assistant
// generation prompt and its empty thinking block.
std::string chat_prompt(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 64);
  out += "<|im_start|>user\n";
  out += text;
  out += "<|im_end|>\n<|im_start|>assistant\n<think>\n";
  return out;
}

std::vector<int32_t> parse_ids(const std::string& list) {
  std::vector<int32_t> ids;
  std::stringstream ss(list);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    ids.push_back(static_cast<int32_t>(std::stol(item)));
  }
  return ids;
}

std::vector<int64_t> parse_layers(const std::string& list) {
  std::vector<int64_t> out;
  std::stringstream ss(list);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::stoll(item));
  }
  return out;
}

// Reads one "MOSSDIT1" file: the inputs and the arrays the writer produced.
struct DitFile {
  int64_t height = 0, width = 0, text_len = 0;
  float timestep = 0.0f;
  std::vector<float> latent, text, out, block0, prehead;
};

std::optional<DitFile> read_dit(const std::string& path, int64_t channels,
                                int64_t text_dim, int64_t dim,
                                std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return std::nullopt;
  }
  auto read = [&](void* dst, size_t bytes) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(in);
  };
  char magic[8];
  DitFile f;
  if (!read(magic, 8) || std::memcmp(magic, "MOSSDIT1", 8) != 0 ||
      !read(&f.height, 8) || !read(&f.width, 8) || !read(&f.text_len, 8) ||
      !read(&f.timestep, 4)) {
    *error = path + ": not a MOSSDIT1 file";
    return std::nullopt;
  }
  const int64_t hw = f.height * f.width;
  const int64_t tokens = hw + f.text_len;
  auto read_vec = [&](std::vector<float>& v, size_t n) {
    v.resize(n);
    return read(v.data(), n * sizeof(float));
  };
  if (!read_vec(f.latent, static_cast<size_t>(channels * hw)) ||
      !read_vec(f.text, static_cast<size_t>(f.text_len * text_dim)) ||
      !read_vec(f.out, static_cast<size_t>(channels * hw)) ||
      !read_vec(f.block0, static_cast<size_t>(tokens * dim)) ||
      !read_vec(f.prehead, static_cast<size_t>(hw * dim))) {
    *error = path + ": truncated";
    return std::nullopt;
  }
  return f;
}

int run_dit(const std::string& model_dir, const std::string& input_path,
            const std::string& out_path, const std::string& skip_arg,
            bool cuda) {
  if (input_path.empty() || out_path.empty()) {
    std::cerr << "sb-moss dit: --input and --out are required\n";
    return 2;
  }
  sonicboom::moss::DitConfig cfg;
  std::expected<sonicboom::moss::DitSample, std::string> sample =
      std::unexpected(std::string("not run"));
  std::vector<float> debug_residual, debug_prehead;
  double elapsed_seconds = 0.0;

  if (cuda) {
#if defined(SONICBOOM_TOOL_CUDA)
    if (!sonicboom::moss::dit_cuda_available()) {
      std::cerr << "sb-moss dit: CUDA is not available\n";
      return 1;
    }
    auto model = sonicboom::moss::CudaDiffusionTransformer::load(model_dir);
    if (!model) {
      std::cerr << "sb-moss: " << model.error() << "\n";
      return 1;
    }
    cfg = (*model)->config();
    std::cerr << "dit(cuda): " << cfg.num_layers << " layers, dim " << cfg.dim
              << ", " << cfg.num_heads << " heads, Q8_0 weights "
              << (*model)->weight_bytes() << " B\n";
    std::string error;
    auto ref =
        read_dit(input_path, cfg.latent_channels, cfg.text_dim, cfg.dim, &error);
    if (!ref) {
      std::cerr << "sb-moss: " << error << "\n";
      return 1;
    }
    const std::vector<int64_t> skip = parse_layers(skip_arg);
    const auto start = std::chrono::steady_clock::now();
    sample = (*model)->forward(ref->latent, ref->height, ref->width,
                               ref->timestep, ref->text, ref->text_len, skip);
    elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (!sample) {
      std::cerr << "sb-moss: " << sample.error() << "\n";
      return 1;
    }
#else
    std::cerr << "sb-moss dit: this build has no CUDA support\n";
    return 2;
#endif
  } else {
    auto model = sonicboom::moss::DiffusionTransformer::load(model_dir);
    if (!model) {
      std::cerr << "sb-moss: " << model.error() << "\n";
      return 1;
    }
    cfg = (*model)->config();
    std::cerr << "dit: " << cfg.num_layers << " layers, dim " << cfg.dim << ", "
              << cfg.num_heads << " heads, weights " << (*model)->weight_bytes()
              << " B\n";

    std::string error;
    auto ref =
        read_dit(input_path, cfg.latent_channels, cfg.text_dim, cfg.dim, &error);
    if (!ref) {
      std::cerr << "sb-moss: " << error << "\n";
      return 1;
    }
    const std::vector<int64_t> skip = parse_layers(skip_arg);

    sonicboom::moss::DitDebug debug;
    const auto start = std::chrono::steady_clock::now();
    sample = (*model)->forward(ref->latent, ref->height, ref->width,
                               ref->timestep, ref->text, ref->text_len, skip,
                               &debug);
    elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (!sample) {
      std::cerr << "sb-moss: " << sample.error() << "\n";
      return 1;
    }
    debug_residual = std::move(debug.residual_after_first_block);
    debug_prehead = std::move(debug.prehead);
  }

  std::string error;
  auto ref = read_dit(input_path, cfg.latent_channels, cfg.text_dim, cfg.dim,
                      &error);
  if (!ref) {
    std::cerr << "sb-moss: " << error << "\n";
    return 1;
  }
  const auto elapsed = std::chrono::duration<double>(elapsed_seconds);

  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "sb-moss: cannot write " << out_path << "\n";
    return 1;
  }
  auto write = [&](const void* p, size_t bytes) {
    out.write(static_cast<const char*>(p), static_cast<std::streamsize>(bytes));
  };
  auto write_vec = [&](const std::vector<float>& v) {
    write(v.data(), v.size() * sizeof(float));
  };
  write("MOSSDIT1", 8);
  write(&ref->height, 8);
  write(&ref->width, 8);
  write(&ref->text_len, 8);
  write(&ref->timestep, 4);
  write_vec(ref->latent);
  write_vec(ref->text);
  write_vec(sample->values);
  write_vec(debug_residual);
  write_vec(debug_prehead);
  if (!out) {
    std::cerr << "sb-moss: write failed for " << out_path << "\n";
    return 1;
  }

  double energy = 0.0;
  for (float v : sample->values) energy += double(v) * v;
  const double rms = std::sqrt(energy / double(sample->values.size()));
  std::cerr << "dit forward" << (cuda ? " (cuda)" : "") << ": " << ref->text_len
            << " text + " << ref->height * ref->width << " video tokens in "
            << elapsed.count() << " s, out rms " << rms << "\n";
  return 0;
}

// Reads one "MOSVAE1" file: the latent and the arrays the writer produced.
struct VaeFile {
  int64_t channels = 0, height = 0, width = 0, out_channels = 0;
  std::vector<float> latent, out;
};

std::optional<VaeFile> read_vae(const std::string& path, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return std::nullopt;
  }
  auto read = [&](void* dst, size_t bytes) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(in);
  };
  // The Python reference writer emits the 7-character magic "MOSVAE1"
  // immediately followed by the header, so the header starts at offset 7.
  char magic[7];
  VaeFile f;
  if (!read(magic, 7) || std::memcmp(magic, "MOSVAE1", 7) != 0 ||
      !read(&f.channels, 8) || !read(&f.height, 8) || !read(&f.width, 8) ||
      !read(&f.out_channels, 8)) {
    *error = path + ": not a MOSVAE1 file";
    return std::nullopt;
  }
  auto read_vec = [&](std::vector<float>& v, int64_t n) {
    v.resize(static_cast<size_t>(n));
    return read(v.data(), v.size() * sizeof(float));
  };
  const int64_t s = f.height * f.width;
  if (!read_vec(f.latent, f.channels * s) ||
      !read_vec(f.out, f.out_channels * 64 * s)) {
    *error = path + ": truncated";
    return std::nullopt;
  }
  return f;
}

// Writes our stage tensors in the reference writer's MOSVSG01 layout.
bool write_vae_stages(const std::string& path,
                      const std::vector<sonicboom::moss::VaeStage>& stages) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  auto write = [&](const void* p, size_t bytes) {
    out.write(static_cast<const char*>(p), static_cast<std::streamsize>(bytes));
  };
  write("MOSVSG01", 8);
  const int64_t count = static_cast<int64_t>(stages.size());
  write(&count, 8);
  for (const auto& st : stages) {
    const int32_t name_len = static_cast<int32_t>(st.name.size());
    write(&name_len, 4);
    write(st.name.data(), st.name.size());
    const int32_t ndim = 4;
    write(&ndim, 4);
    const int64_t dims[4] = {st.n, st.c, st.h, st.w};
    write(dims, sizeof(dims));
    write(st.values.data(), st.values.size() * sizeof(float));
  }
  return static_cast<bool>(out);
}

int run_vae(const std::string& model_dir, const std::string& input_path,
            const std::string& out_path, const std::string& stages_path) {
  if (input_path.empty()) {
    std::cerr << "sb-moss vae: --input is required\n";
    return 2;
  }
  auto model = sonicboom::moss::VaeDecoder::load(model_dir);
  if (!model) {
    std::cerr << "sb-moss: " << model.error() << "\n";
    return 1;
  }
  const auto& cfg = (*model)->config();
  std::cerr << "vae: " << cfg.channels.size() << " up blocks, latent "
            << cfg.latent_channels << " ch, weights " << (*model)->weight_bytes()
            << " B\n";

  std::string error;
  auto ref = read_vae(input_path, &error);
  if (!ref) {
    std::cerr << "sb-moss: " << error << "\n";
    return 1;
  }
  if (ref->channels != cfg.latent_channels || ref->out_channels != cfg.out_channels) {
    std::cerr << "sb-moss: input channel mismatch\n";
    return 1;
  }

  std::vector<sonicboom::moss::VaeStage> stages;
  const auto start = std::chrono::steady_clock::now();
  auto image = (*model)->decode(ref->latent, ref->height, ref->width,
                                stages_path.empty() ? nullptr : &stages);
  if (!image) {
    std::cerr << "sb-moss: " << image.error() << "\n";
    return 1;
  }
  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start);

  double energy = 0.0;
  for (float v : *image) energy += double(v) * v;
  const double rms = std::sqrt(energy / double(image->size()));
  std::cerr << "vae decode: " << ref->height << "x" << ref->width << " -> "
            << ref->height * 8 << "x" << ref->width * 8 << " in "
            << elapsed.count() << " s, out rms " << rms << "\n";

  if (!out_path.empty()) {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
      std::cerr << "sb-moss: cannot write " << out_path << "\n";
      return 1;
    }
    auto write = [&](const void* p, size_t bytes) {
      out.write(static_cast<const char*>(p), static_cast<std::streamsize>(bytes));
    };
    write("MOSVAE1", 7);
    write(&ref->channels, 8);
    write(&ref->height, 8);
    write(&ref->width, 8);
    write(&ref->out_channels, 8);
    write(ref->latent.data(), ref->latent.size() * sizeof(float));
    write(image->data(), image->size() * sizeof(float));
    if (!out) {
      std::cerr << "sb-moss: write failed for " << out_path << "\n";
      return 1;
    }
  }
  if (!stages_path.empty() && !write_vae_stages(stages_path, stages)) {
    std::cerr << "sb-moss: cannot write " << stages_path << "\n";
    return 1;
  }
  return 0;
}

// Reads one "MOSSHST1" hidden-state file (the encoder's last hidden state).
std::optional<sonicboom::moss::HiddenStates> read_hidden(
    const std::string& path, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return std::nullopt;
  }
  char magic[8];
  sonicboom::moss::HiddenStates h;
  if (!in.read(magic, 8) || std::memcmp(magic, "MOSSHST1", 8) != 0 ||
      !in.read(reinterpret_cast<char*>(&h.tokens), 8) ||
      !in.read(reinterpret_cast<char*>(&h.hidden_size), 8)) {
    *error = path + ": not a MOSSHST1 file";
    return std::nullopt;
  }
  h.values.resize(static_cast<size_t>(h.tokens * h.hidden_size));
  if (!in.read(reinterpret_cast<char*>(h.values.data()),
               static_cast<std::streamsize>(h.values.size() * sizeof(float)))) {
    *error = path + ": truncated";
    return std::nullopt;
  }
  return h;
}

// A batched latent file: "MOSLAT01", int64 B, C, H, W, then f32 values.
struct LatentBatch {
  int64_t batch = 0, channels = 0, height = 0, width = 0;
  std::vector<float> values;
};

std::optional<LatentBatch> read_latents(const std::string& path,
                                        std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return std::nullopt;
  }
  char magic[8];
  LatentBatch b;
  if (!in.read(magic, 8) || std::memcmp(magic, "MOSLAT01", 8) != 0 ||
      !in.read(reinterpret_cast<char*>(&b.batch), 8) ||
      !in.read(reinterpret_cast<char*>(&b.channels), 8) ||
      !in.read(reinterpret_cast<char*>(&b.height), 8) ||
      !in.read(reinterpret_cast<char*>(&b.width), 8)) {
    *error = path + ": not a MOSLAT01 file";
    return std::nullopt;
  }
  b.values.resize(static_cast<size_t>(b.batch * b.channels * b.height * b.width));
  if (!in.read(reinterpret_cast<char*>(b.values.data()),
               static_cast<std::streamsize>(b.values.size() * sizeof(float)))){
    *error = path + ": truncated";
    return std::nullopt;
  }
  return b;
}

std::vector<float> seeded_noise(int64_t n, uint64_t seed) {
  // splitmix64 + Box-Muller: deterministic, but not torch's generator (parity
  // runs pass an explicit --latent-in).
  std::vector<float> out(static_cast<size_t>(n));
  auto next = [&]() {
    seed += 0x9e3779b97f4a7c15ull;
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  };
  auto uniform = [&]() {
    return (static_cast<double>(next() >> 11) + 0.5) *
           (1.0 / 9007199254740992.0);
  };
  for (int64_t i = 0; i + 1 < n; i += 2) {
    const double u1 = uniform(), u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    out[static_cast<size_t>(i)] =
        static_cast<float>(r * std::cos(2.0 * 3.14159265358979323846 * u2));
    out[static_cast<size_t>(i + 1)] =
        static_cast<float>(r * std::sin(2.0 * 3.14159265358979323846 * u2));
  }
  if (n % 2 == 1)
    out[static_cast<size_t>(n - 1)] =
        static_cast<float>(std::sqrt(-2.0 * std::log(uniform())) *
                           std::cos(2.0 * 3.14159265358979323846 * uniform()));
  return out;
}

int run_sample(const std::string& root, const std::string& text_arg,
               const std::string& embeds_path, const std::string& negative_arg,
               const std::string& neg_embeds_path, int64_t height,
               int64_t width, int64_t steps, double guidance,
               const std::string& latent_in, uint64_t seed, int64_t batch,
               const std::string& latent_out, const std::string& image_out,
               const std::string& trace_path, bool no_skip,
               const std::string& slg_arg, double slg_scale, double slg_start,
               double slg_stop, bool no_decode, bool decode_only, bool raw) {
  if (height <= 0 || width <= 0 || height % 16 || width % 16) {
    std::cerr << "sb-moss sample: height and width must be multiples of 16\n";
    return 2;
  }
  const bool have_embeds = !embeds_path.empty();
  const bool have_text = !text_arg.empty();
  if (have_embeds == have_text) {
    std::cerr << "sb-moss sample: give exactly one of --text and --embeds\n";
    return 2;
  }

  const auto start_all = std::chrono::steady_clock::now();
  auto pipeline = sonicboom::moss::MossPipeline::load(root, /*with_text=*/have_text);
  if (!pipeline) {
    std::cerr << "sb-moss: " << pipeline.error() << "\n";
    return 1;
  }
  const double load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start_all)
                            .count();
  // One machine-readable line per run so tools/bench-image.sh never parses
  // free text. `sample_s` is the denoising loop, `decode_s` the VAE, and
  // `total_s` the whole process from load to the last byte written.
  auto emit = [&](int64_t images, int64_t forwards, double sample_s,
                  double decode_s) {
    const double total_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start_all)
                               .count();
    std::cout << std::setprecision(6) << "RESULT sample batch=" << images
              << " steps=" << steps << " forwards=" << forwards
              << " load_s=" << load_s << " sample_s=" << sample_s
              << " decode_s=" << decode_s << " total_s=" << total_s << "\n";
  };
  const auto& dit_cfg = (*pipeline)->transformer()->config();

  // Conditioning: the reference runs the encoder once per prompt with no
  // padding, so each text keeps its own length.
  sonicboom::moss::MossConditioning cond;
  auto load_cond = [&](const std::string& path,
                       std::vector<float>& dst, int64_t& len) -> bool {
    std::string error;
    auto h = read_hidden(path, &error);
    if (!h) {
      std::cerr << "sb-moss: " << error << "\n";
      return false;
    }
    if (h->hidden_size != dit_cfg.text_dim) {
      std::cerr << "sb-moss: " << path << " has hidden size "
                << h->hidden_size << ", expected " << dit_cfg.text_dim << "\n";
      return false;
    }
    dst = std::move(h->values);
    len = h->tokens;
    return true;
  };
  auto encode_prompt_text = [&](const std::string& prompt,
                                std::vector<float>& dst, int64_t& len) -> bool {
    std::filesystem::path tok = std::filesystem::path(root) / "tokenizer" /
                                "tokenizer.json";
    auto vocab = sonicboom::vocab::load_tokenizer_json(tok);
    if (!vocab) {
      std::cerr << "sb-moss: " << vocab.error() << "\n";
      return false;
    }
    const std::string rendered = raw ? prompt : chat_prompt(prompt);
    const auto ids = sonicboom::vocab::tokenize(*vocab, rendered, true, false);
    auto hidden = (*pipeline)->encode(ids);
    if (!hidden) {
      std::cerr << "sb-moss: " << hidden.error() << "\n";
      return false;
    }
    dst = std::move(hidden->values);
    len = hidden->tokens;
    return true;
  };
  if (have_embeds) {
    if (!load_cond(embeds_path, cond.text, cond.text_len)) return 1;
  } else if (!encode_prompt_text(text_arg, cond.text, cond.text_len)) {
    return 1;
  }
  const bool do_cfg = static_cast<float>(guidance) != 1.0f;
  if (do_cfg) {
    if (!neg_embeds_path.empty()) {
      if (!load_cond(neg_embeds_path, cond.negative, cond.negative_len))
        return 1;
    } else if (!encode_prompt_text(negative_arg, cond.negative,
                                   cond.negative_len)) {
      return 1;
    }
  }
  std::cerr << "sampler: " << cond.text_len << " text tokens, "
            << cond.negative_len << " negative tokens, guidance " << guidance
            << "\n";

  const int64_t grid_h = height / 16, grid_w = width / 16;
  sonicboom::moss::MossPipelineConfig cfg;
  cfg.steps = steps;
  cfg.guidance_scale = static_cast<float>(guidance);
  cfg.slg_scale = static_cast<float>(slg_scale);
  cfg.slg_start = slg_start;
  cfg.slg_stop = slg_stop;
  if (no_skip) cfg.uncond_skip_layers.clear();
  if (!slg_arg.empty()) cfg.slg_layers = parse_layers(slg_arg);

  std::vector<sonicboom::moss::MossLatent> latents;
  int64_t in_batch = batch;
  if (!latent_in.empty()) {
    std::string error;
    auto file = read_latents(latent_in, &error);
    if (!file) {
      std::cerr << "sb-moss: " << error << "\n";
      return 1;
    }
    if (file->channels != dit_cfg.latent_channels || file->height != grid_h ||
        file->width != grid_w) {
      std::cerr << "sb-moss: latent file is [" << file->batch << ","
                << file->channels << "," << file->height << "," << file->width
                << "], expected [" << batch << "," << dit_cfg.latent_channels
                << "," << grid_h << "," << grid_w << "]\n";
      return 1;
    }
    in_batch = file->batch;
    for (int64_t i = 0; i < in_batch; ++i) {
      sonicboom::moss::MossLatent l;
      l.channels = file->channels;
      l.height = file->height;
      l.width = file->width;
      const size_t off = static_cast<size_t>(i * l.count());
      l.values.assign(file->values.begin() + off,
                      file->values.begin() + off + l.count());
      latents.push_back(std::move(l));
    }
  } else {
    for (int64_t i = 0; i < in_batch; ++i) {
      sonicboom::moss::MossLatent l;
      l.channels = dit_cfg.latent_channels;
      l.height = grid_h;
      l.width = grid_w;
      l.values = seeded_noise(l.count(), seed + static_cast<uint64_t>(i));
      latents.push_back(std::move(l));
    }
  }
  std::vector<sonicboom::moss::MossConditioning> conds(latents.size(), cond);

  if (decode_only) {
    std::vector<sonicboom::moss::MossLatent> decoded_inputs = latents;
    const auto decode_start = std::chrono::steady_clock::now();
    int64_t ic = 0, ih = 0, iw = 0;
    std::vector<float> all;
    for (const auto& l : decoded_inputs) {
      auto image = (*pipeline)->decode(l);
      if (!image) {
        std::cerr << "sb-moss: " << image.error() << "\n";
        return 1;
      }
      ic = image->channels;
      ih = image->height;
      iw = image->width;
      all.insert(all.end(), image->values.begin(), image->values.end());
    }
    if (latent_out.empty() && image_out.empty()) {
      std::cerr << "sb-moss sample: --decode-only needs --image-out\n";
      return 2;
    }
    if (!image_out.empty()) {
      std::ofstream out(image_out, std::ios::binary);
      if (!out) {
        std::cerr << "sb-moss: cannot write " << image_out << "\n";
        return 1;
      }
      const int64_t b = static_cast<int64_t>(decoded_inputs.size());
      out.write("MOSIMG01", 8);
      out.write(reinterpret_cast<const char*>(&b), 8);
      out.write(reinterpret_cast<const char*>(&ic), 8);
      out.write(reinterpret_cast<const char*>(&ih), 8);
      out.write(reinterpret_cast<const char*>(&iw), 8);
      out.write(reinterpret_cast<const char*>(all.data()),
                static_cast<std::streamsize>(all.size() * sizeof(float)));
    }
    const double decode_only_s = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() -
                                     decode_start)
                                     .count();
    std::cerr << "decode only: " << decoded_inputs.size() << " image(s) " << ic
              << "x" << ih << "x" << iw << " in " << decode_only_s << " s\n";
    emit(static_cast<int64_t>(decoded_inputs.size()), 0, 0.0, decode_only_s);
    return 0;
  }

  const sonicboom::moss::FlowSchedule schedule =
      sonicboom::moss::moss_flow_schedule(cfg.steps);
  std::cerr << std::setprecision(9) << "schedule: "
            << schedule.timesteps.size() << " steps";
  for (size_t i = 0; i < schedule.timesteps.size(); ++i)
    std::cerr << " " << schedule.timesteps[i];
  std::cerr << "\n";

  std::vector<std::vector<sonicboom::moss::MossLatent>> trace(latents.size());
  sonicboom::moss::MossSampleStats stats;
  auto sampled = (*pipeline)->sample_batch(
      latents, conds, cfg, trace_path.empty() ? nullptr : &trace, &stats);
  if (!sampled) {
    std::cerr << "sb-moss: " << sampled.error() << "\n";
    return 1;
  }
  std::cerr << "sample: " << sampled->size() << " image(s), " << stats.forwards
            << " DiT forwards in " << stats.seconds << " s ("
            << stats.seconds / double(std::max<int64_t>(stats.forwards, 1))
            << " s/forward)\n";

  // The pipeline's refine stage is an identity stub in this checkpoint.
  std::vector<sonicboom::moss::MossLatent> refined;
  for (const auto& l : *sampled) {
    auto r = (*pipeline)->refine(l);
    if (!r) {
      std::cerr << "sb-moss: " << r.error() << "\n";
      return 1;
    }
    refined.push_back(std::move(*r));
  }

  if (!latent_out.empty()) {
    std::ofstream out(latent_out, std::ios::binary);
    if (!out) {
      std::cerr << "sb-moss: cannot write " << latent_out << "\n";
      return 1;
    }
    const int64_t b = static_cast<int64_t>(refined.size());
    const int64_t c = refined.empty() ? 0 : refined[0].channels;
    const int64_t h = refined.empty() ? 0 : refined[0].height;
    const int64_t w = refined.empty() ? 0 : refined[0].width;
    out.write("MOSLAT01", 8);
    out.write(reinterpret_cast<const char*>(&b), 8);
    out.write(reinterpret_cast<const char*>(&c), 8);
    out.write(reinterpret_cast<const char*>(&h), 8);
    out.write(reinterpret_cast<const char*>(&w), 8);
    for (const auto& l : refined)
      out.write(reinterpret_cast<const char*>(l.values.data()),
                static_cast<std::streamsize>(l.values.size() * sizeof(float)));
    if (!out) {
      std::cerr << "sb-moss: write failed for " << latent_out << "\n";
      return 1;
    }
  }

  if (!trace_path.empty()) {
    std::ofstream out(trace_path, std::ios::binary);
    if (!out) {
      std::cerr << "sb-moss: cannot write " << trace_path << "\n";
      return 1;
    }
    const int64_t b = static_cast<int64_t>(trace.size());
    const int64_t s = trace.empty() ? 0 : static_cast<int64_t>(trace[0].size());
    out.write("MOSLTR01", 8);
    out.write(reinterpret_cast<const char*>(&b), 8);
    out.write(reinterpret_cast<const char*>(&s), 8);
    for (const auto& row : trace)
      for (const auto& l : row)
        out.write(reinterpret_cast<const char*>(l.values.data()),
                  static_cast<std::streamsize>(l.values.size() * sizeof(float)));
    if (!out) {
      std::cerr << "sb-moss: write failed for " << trace_path << "\n";
      return 1;
    }
  }

  if (no_decode || image_out.empty()) {
    emit(static_cast<int64_t>(refined.size()), stats.forwards, stats.seconds,
         0.0);
    return 0;
  }

  std::ofstream out(image_out, std::ios::binary);
  if (!out) {
    std::cerr << "sb-moss: cannot write " << image_out << "\n";
    return 1;
  }
  const auto decode_start = std::chrono::steady_clock::now();
  int64_t ic = 0, ih = 0, iw = 0;
  std::vector<float> all;
  for (const auto& l : refined) {
    auto image = (*pipeline)->decode(l);
    if (!image) {
      std::cerr << "sb-moss: " << image.error() << "\n";
      return 1;
    }
    ic = image->channels;
    ih = image->height;
    iw = image->width;
    all.insert(all.end(), image->values.begin(), image->values.end());
  }
  const auto decode_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - decode_start).count();
  const int64_t b = static_cast<int64_t>(refined.size());
  out.write("MOSIMG01", 8);
  out.write(reinterpret_cast<const char*>(&b), 8);
  out.write(reinterpret_cast<const char*>(&ic), 8);
  out.write(reinterpret_cast<const char*>(&ih), 8);
  out.write(reinterpret_cast<const char*>(&iw), 8);
  out.write(reinterpret_cast<const char*>(all.data()),
            static_cast<std::streamsize>(all.size() * sizeof(float)));
  if (!out) {
    std::cerr << "sb-moss: write failed for " << image_out << "\n";
    return 1;
  }
  std::cerr << "decode: " << b << " image(s) " << ic << "x" << ih << "x" << iw
            << " in " << decode_seconds << " s\n";
  emit(b, stats.forwards, stats.seconds, decode_seconds);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string_view command = argv[1];
  if (command != "encode" && command != "tokens" && command != "dit" &&
      command != "vae" && command != "sample") {
    usage();
    return 2;
  }

  std::string model_dir;
  std::string text;
  std::string ids_arg;
  std::string out_path;
  std::string input_path;
  std::string skip_arg;
  std::string stages_path;
  std::string negative_arg;
  std::string embeds_path;
  std::string neg_embeds_path;
  std::string latent_in;
  std::string latent_out;
  std::string image_out;
  std::string trace_path;
  std::string slg_arg;
  int64_t height = 64;
  int64_t width = 64;
  int64_t steps = 50;
  int64_t batch = 1;
  double guidance = 3.0;
  double slg_scale = 0.0;
  double slg_start = 0.0;
  double slg_stop = 1.0;
  uint64_t seed = 0;
  bool no_skip = false;
  bool no_decode = false;
  bool decode_only = false;
  bool cuda = false;
  bool have_text = false;
  bool raw = false;
  bool dump_tokens = false;
  for (int i = 2; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "sb-moss: " << what << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") {
      model_dir = next("--model");
    } else if (a == "--text") {
      text = next("--text");
      have_text = true;
    } else if (a == "--tokens") {
      ids_arg = next("--tokens");
    } else if (a == "--out") {
      out_path = next("--out");
    } else if (a == "--input") {
      input_path = next("--input");
    } else if (a == "--skip") {
      skip_arg = next("--skip");
    } else if (a == "--stages") {
      stages_path = next("--stages");
    } else if (a == "--negative") {
      negative_arg = next("--negative");
    } else if (a == "--embeds") {
      embeds_path = next("--embeds");
    } else if (a == "--neg-embeds") {
      neg_embeds_path = next("--neg-embeds");
    } else if (a == "--height") {
      height = std::stoll(next("--height"));
    } else if (a == "--width") {
      width = std::stoll(next("--width"));
    } else if (a == "--steps") {
      steps = std::stoll(next("--steps"));
    } else if (a == "--batch") {
      batch = std::stoll(next("--batch"));
    } else if (a == "--guidance") {
      guidance = std::stod(next("--guidance"));
    } else if (a == "--latent-in") {
      latent_in = next("--latent-in");
    } else if (a == "--latent-out") {
      latent_out = next("--latent-out");
    } else if (a == "--image-out") {
      image_out = next("--image-out");
    } else if (a == "--trace") {
      trace_path = next("--trace");
    } else if (a == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(next("--seed")));
    } else if (a == "--slg") {
      slg_arg = next("--slg");
    } else if (a == "--slg-scale") {
      slg_scale = std::stod(next("--slg-scale"));
    } else if (a == "--slg-start") {
      slg_start = std::stod(next("--slg-start"));
    } else if (a == "--slg-stop") {
      slg_stop = std::stod(next("--slg-stop"));
    } else if (a == "--no-skip") {
      no_skip = true;
    } else if (a == "--no-decode") {
      no_decode = true;
    } else if (a == "--decode-only") {
      decode_only = true;
    } else if (a == "--cuda") {
      cuda = true;
    } else if (a == "--raw") {
      raw = true;
    } else if (a == "--dump-tokens") {
      dump_tokens = true;
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::cerr << "sb-moss: unknown option " << a << "\n";
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::cerr << "sb-moss: --model is required\n";
    return 2;
  }

  if (command == "dit")
    return run_dit(model_dir, input_path, out_path, skip_arg, cuda);
  if (command == "vae") return run_vae(model_dir, input_path, out_path, stages_path);
  if (command == "sample") {
    return run_sample(model_dir, text, embeds_path, negative_arg, neg_embeds_path,
                      height, width, steps, guidance, latent_in, seed, batch,
                      latent_out, image_out, trace_path, no_skip, slg_arg,
                      slg_scale, slg_start, slg_stop, no_decode, decode_only,
                      raw);
  }

  std::vector<int32_t> ids;
  if (!ids_arg.empty()) {
    ids = parse_ids(ids_arg);
  } else {
    if (!have_text) {
      std::cerr << "sb-moss: one of --text or --tokens is required\n";
      return 2;
    }
    // The tokenizer sits beside the text encoder in the MOSS-Image layout, so
    // fall back to the sibling directory when the encoder directory has none.
    std::filesystem::path tokenizer_dir =
        std::filesystem::path(model_dir) / "tokenizer.json";
    if (!std::filesystem::exists(tokenizer_dir)) {
      tokenizer_dir = std::filesystem::path(model_dir).parent_path() /
                      "tokenizer" / "tokenizer.json";
    }
    auto vocab = sonicboom::vocab::load_tokenizer_json(tokenizer_dir);
    if (!vocab) {
      std::cerr << "sb-moss: " << vocab.error() << "\n";
      return 1;
    }
    const std::string prompt = raw ? text : chat_prompt(text);
    ids = sonicboom::vocab::tokenize(*vocab, prompt, /*parse_special=*/true,
                                     /*add_special=*/false);
  }

  if (dump_tokens || command == "tokens") {
    for (size_t i = 0; i < ids.size(); ++i) {
      std::cout << (i ? " " : "") << ids[i];
    }
    std::cout << "\n";
    if (dump_tokens || command == "tokens") return 0;
  }

  auto encoder = sonicboom::moss::TextEncoder::load(model_dir);
  if (!encoder) {
    std::cerr << "sb-moss: " << encoder.error() << "\n";
    return 1;
  }
  const auto& cfg = (*encoder)->config();
  std::cerr << "encoder: " << cfg.num_layers << " layers, hidden "
            << cfg.hidden_size << ", weights " << (*encoder)->weight_bytes()
            << " B\n";

  const auto start = std::chrono::steady_clock::now();
  auto hidden = (*encoder)->forward(ids);
  if (!hidden) {
    std::cerr << "sb-moss: " << hidden.error() << "\n";
    return 1;
  }
  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start);
  std::cerr << "forward: " << ids.size() << " tokens in " << elapsed.count()
            << " s (" << ids.size() / elapsed.count() << " tok/s)\n";

  if (out_path.empty()) {
    std::cerr << "sb-moss: --out is required for encode\n";
    return 2;
  }
  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "sb-moss: cannot write " << out_path << "\n";
    return 1;
  }
  out.write("MOSSHST1", 8);
  out.write(reinterpret_cast<const char*>(&hidden->tokens), sizeof(int64_t));
  out.write(reinterpret_cast<const char*>(&hidden->hidden_size), sizeof(int64_t));
  out.write(reinterpret_cast<const char*>(hidden->values.data()),
            static_cast<std::streamsize>(hidden->values.size() * sizeof(float)));
  if (!out) {
    std::cerr << "sb-moss: write failed for " << out_path << "\n";
    return 1;
  }
  std::cerr << "wrote " << out_path << " (" << hidden->tokens << "x"
            << hidden->hidden_size << " f32)\n";
  return 0;
}
