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

// MOSS-Image DiT forward, a direct transcription of the Apache-2.0 reference
// (MOSS-Image-SFT/transformer/pipeline_moss_image.py: MOSSImageTransformer,
// _Block, _TextBlock, _RoPE, _attention, _situ). Everything is f32 from the
// memory-mapped bf16 checkpoint.
//
// The pieces worth naming, because they differ from the sonicboom transformer
// blocks:
//   * one shared adaLN modulation for all blocks: a single Linear(dim, 6*dim)
//     applied to the time embedding, split into
//     (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f);
//   * a joint image+text sequence, no mask, SDPA scale 1/sqrt(head_dim);
//   * 4-axis half-split RoPE over 64 of the 128 head dims: coordinates
//     (r, seq, h, w), 16 bands per axis, dim i paired with i+64;
//   * SiTU-GLU (4*tanh(g/4)*sigmoid(g) * 25*tanh(h/25)) in the FFN;
//   * the final head is norm -> (1+scale, shift) -> Linear(dim, channels).

#include <sonicboom/model/moss_dit.h>

#include "../safetensors/json.h"

#include <sonicboom/dtype.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/safetensors/reader.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>

namespace sonicboom::moss {
namespace {

constexpr float kRmsEps = 1e-6f;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float silu(float x) { return x * sigmoid(x); }

std::string shape_string(std::span<const uint64_t> shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

std::expected<std::string, std::string> read_text(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in.good() && !in.eof()) return std::unexpected("cannot read " + p.string());
  return ss.str();
}

}  // namespace

struct DiffusionTransformer::Impl {
  DitConfig cfg;
  std::unique_ptr<safetensors::ShardedReader> reader;
  int64_t weight_bytes = 0;

  // Per-block norm vectors, f32 from bf16 (the weights are plain scales: the
  // reference applies them after the normalization, no 1+ convention).
  struct BlockNorms {
    std::vector<float> norm1;  // dim
    std::vector<float> norm2;  // dim
    std::vector<float> q_norm;  // head_dim
    std::vector<float> k_norm;  // head_dim
  };
  std::vector<BlockNorms> norms;
  struct TextNorms {
    std::vector<float> norm1, norm2, q_norm, k_norm;
  };
  std::vector<TextNorms> text_norms;
  std::vector<float> final_norm;  // dim

  std::vector<float> rope_bands;  // head_dim / 8

  // --- configuration -------------------------------------------------------

  std::expected<void, std::string> load_config(const std::filesystem::path& dir) {
    auto text = read_text(dir / "config.json");
    if (!text) return std::unexpected(text.error());
    std::string error;
    auto root = safetensors::detail::parse_json(*text, &error);
    if (!root) return std::unexpected("config.json: " + error);

    auto int_of = [&](const char* key) -> std::expected<int64_t, std::string> {
      const auto* v = root->find(key);
      if (!v || !v->is_integer) {
        return std::unexpected(std::string("config.json: ") + key +
                               " missing or not an integer");
      }
      return v->integer;
    };
    for (auto [key, dest] :
         {std::pair{"dim", &cfg.dim},
          std::pair{"ffn_dim", &cfg.ffn_dim},
          std::pair{"num_heads", &cfg.num_heads},
          std::pair{"num_layers", &cfg.num_layers},
          std::pair{"text_dim", &cfg.text_dim},
          std::pair{"freq_dim", &cfg.freq_dim},
          std::pair{"latent_channels", &cfg.latent_channels}}) {
      auto got = int_of(key);
      if (!got) return std::unexpected(got.error());
      *dest = *got;
    }
    const auto* base = root->find("rope_base");
    if (base && base->type == safetensors::detail::JsonValue::Type::Number) {
      cfg.rope_base = base->number;
    }
    if (cfg.dim % cfg.num_heads != 0 || (cfg.head_dim() % 8) != 0) {
      return std::unexpected("config.json: dim must divide by num_heads and "
                             "head_dim must be a multiple of 8");
    }
    if (cfg.freq_dim % 2 != 0) {
      return std::unexpected("config.json: freq_dim must be even");
    }
    return {};
  }

  // --- weight access -------------------------------------------------------

  std::expected<std::span<const uint16_t>, std::string> bf16(
      std::string_view name, std::initializer_list<int64_t> shape) {
    const safetensors::ShardedTensor* t = reader->tensor(name);
    if (!t) return std::unexpected("missing tensor " + std::string(name));
    if (t->info.dtype != safetensors::DType::BF16) {
      return std::unexpected(std::string(name) + ": expected BF16, found " +
                             std::string(safetensors::dtype_name(t->info.dtype)));
    }
    if (t->info.shape.size() != shape.size()) {
      return std::unexpected(std::string(name) + ": shape " +
                             shape_string(t->info.shape) + " != expected");
    }
    size_t i = 0;
    for (const int64_t dim : shape) {
      if (t->info.shape[i++] != static_cast<uint64_t>(dim)) {
        return std::unexpected(std::string(name) + ": shape " +
                               shape_string(t->info.shape) + " != expected");
      }
    }
    auto bytes = reader->tensor_data(name);
    if (!bytes) return std::unexpected("unreadable tensor " + std::string(name));
    return std::span<const uint16_t>(
        reinterpret_cast<const uint16_t*>(bytes->data()), t->info.n_elements);
  }

  std::vector<float> f32_of(std::span<const uint16_t> w) {
    std::vector<float> out(w.size());
    for (size_t i = 0; i < w.size(); ++i) out[i] = bf16_to_f32(w[i]);
    return out;
  }

  std::expected<std::vector<float>, std::string> f32_tensor(
      std::string_view name, std::initializer_list<int64_t> shape) {
    auto w = bf16(name, shape);
    if (!w) return std::unexpected(w.error());
    return f32_of(*w);
  }

  std::expected<void, std::string> load_weights() {
    norms.resize(static_cast<size_t>(cfg.num_layers));
    text_norms.resize(2);
    for (int64_t l = 0; l < cfg.num_layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      BlockNorms& n = norms[static_cast<size_t>(l)];
      for (auto [suffix, dest] :
           {std::pair{"norm1.weight", &n.norm1},
            std::pair{"norm2.weight", &n.norm2}}) {
        auto v = f32_tensor(p + suffix, {cfg.dim});
        if (!v) return std::unexpected(v.error());
        *dest = std::move(*v);
      }
      for (auto [suffix, dest] :
           {std::pair{"q_norm.weight", &n.q_norm},
            std::pair{"k_norm.weight", &n.k_norm}}) {
        auto v = f32_tensor(p + suffix, {cfg.head_dim()});
        if (!v) return std::unexpected(v.error());
        *dest = std::move(*v);
      }
    }
    for (int64_t b = 0; b < 2; ++b) {
      const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
      TextNorms& n = text_norms[static_cast<size_t>(b)];
      for (auto [suffix, dest] :
           {std::pair{"norm1.weight", &n.norm1},
            std::pair{"norm2.weight", &n.norm2}}) {
        auto v = f32_tensor(p + suffix, {cfg.dim});
        if (!v) return std::unexpected(v.error());
        *dest = std::move(*v);
      }
      for (auto [suffix, dest] :
           {std::pair{"q_norm.weight", &n.q_norm},
            std::pair{"k_norm.weight", &n.k_norm}}) {
        auto v = f32_tensor(p + suffix, {cfg.head_dim()});
        if (!v) return std::unexpected(v.error());
        *dest = std::move(*v);
      }
    }
    auto fn = f32_tensor("final_norm_video.weight", {cfg.dim});
    if (!fn) return std::unexpected(fn.error());
    final_norm = std::move(*fn);

    // The reference derives the bands from the config (rope.bands in the
    // checkpoint is a replicated constant; _load_from_state_dict recomputes it).
    const int64_t bands = cfg.head_dim() / 8;
    rope_bands.resize(static_cast<size_t>(bands));
    for (int64_t j = 0; j < bands; ++j) {
      rope_bands[static_cast<size_t>(j)] =
          1.0f / static_cast<float>(std::pow(cfg.rope_base, double(j) / double(bands)));
    }
    return {};
  }

  // --- pieces of the forward ----------------------------------------------

  // Y[rows][out] = X[rows][cols] @ W^T, W bf16 [out][cols].
  std::expected<void, std::string> linear(std::string_view name, int64_t out,
                                          int64_t cols,
                                          std::span<const float> X, int64_t rows,
                                          std::span<float> Y,
                                          std::string_view bias = {}) {
    auto w = bf16(name, {out, cols});
    if (!w) return std::unexpected(w.error());
    if (!nn::matmul_bf16(*w, static_cast<uint64_t>(cols), X,
                         static_cast<uint64_t>(rows), Y)) {
      return std::unexpected("matmul_bf16 failed for " + std::string(name));
    }
    if (!bias.empty()) {
      auto b = bf16(bias, {out});
      if (!b) return std::unexpected(b.error());
      for (int64_t r = 0; r < rows; ++r) {
        float* row = Y.data() + r * out;
        for (int64_t i = 0; i < out; ++i) row[i] += bf16_to_f32((*b)[static_cast<size_t>(i)]);
      }
    }
    return {};
  }

  void rms_norm_rows(const float* x, const std::vector<float>& weight,
                     int64_t rows, int64_t cols, float* y) {
    const std::span<const float> w(weight);
    for (int64_t r = 0; r < rows; ++r) {
      nn::rms_norm(std::span<const float>(x + r * cols, cols), w, kRmsEps,
                   std::span<float>(y + r * cols, cols));
    }
  }

  // dim i pairs with i+half; angle k = coords[axis]*bands[j], axis*bands+j = k.
  static void rotate(float* v, int64_t heads, int64_t head_dim, int64_t token,
                     int64_t stride, const float* cos, const float* sin) {
    const int64_t half = head_dim / 2;
    for (int64_t h = 0; h < heads; ++h) {
      float* row = v + h * head_dim;
      const float* c = cos + token * stride;
      const float* s = sin + token * stride;
      for (int64_t i = 0; i < half; ++i) {
        const float a = row[i], b = row[i + half];
        row[i] = a * c[i] - b * s[i];
        row[i + half] = b * c[i] + a * s[i];
      }
    }
  }

  // SDPA without a mask over the whole sequence.
  void attention(std::vector<float>& out, const std::vector<float>& q,
                 const std::vector<float>& k, const std::vector<float>& v,
                 int64_t tokens, int64_t heads, int64_t head_dim) {
    const int64_t dim = heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float* qp = q.data();
    const float* kp = k.data();
    const float* vp = v.data();
    float* op = out.data();
    std::vector<float> score(static_cast<size_t>(tokens));
    for (int64_t t = 0; t < tokens; ++t) {
      for (int64_t h = 0; h < heads; ++h) {
        const float* qrow = qp + t * dim + h * head_dim;
        float max = -INFINITY;
        for (int64_t s = 0; s < tokens; ++s) {
          const float* krow = kp + s * dim + h * head_dim;
          double acc = 0.0;
          for (int64_t i = 0; i < head_dim; ++i) acc += double(qrow[i]) * krow[i];
          const float e = static_cast<float>(acc) * scale;
          score[static_cast<size_t>(s)] = e;
          max = std::max(max, e);
        }
        double sum = 0.0;
        for (int64_t s = 0; s < tokens; ++s) {
          const float e = std::exp(score[static_cast<size_t>(s)] - max);
          score[static_cast<size_t>(s)] = e;
          sum += e;
        }
        const float inv = static_cast<float>(1.0 / sum);
        float* orow = op + t * dim + h * head_dim;
        for (int64_t i = 0; i < head_dim; ++i) orow[i] = 0.0f;
        for (int64_t s = 0; s < tokens; ++s) {
          const float p = score[static_cast<size_t>(s)] * inv;
          const float* vrow = vp + s * dim + h * head_dim;
          for (int64_t i = 0; i < head_dim; ++i) orow[i] += p * vrow[i];
        }
      }
    }
  }

  std::expected<void, std::string> text_blocks(
      std::vector<float>& x, int64_t tokens, const float* cos, const float* sin,
      int64_t stride) {
    const int64_t dim = cfg.dim, hd = cfg.head_dim(), heads = cfg.num_heads;
    const int64_t ffn = cfg.ffn_dim;
    std::vector<float> y(x.size()), qkv(static_cast<size_t>(tokens * 3 * dim));
    std::vector<float> attn_out(x.size()), proj(x.size());
    std::vector<float> u(static_cast<size_t>(tokens * ffn)),
        g(static_cast<size_t>(tokens * ffn)), act(static_cast<size_t>(tokens * ffn));
    std::vector<float> ffn_out(x.size());
    for (int64_t b = 0; b < 2; ++b) {
      const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
      const TextNorms& n = text_norms[static_cast<size_t>(b)];
      rms_norm_rows(x.data(), n.norm1, tokens, dim, y.data());
      if (auto r = linear(p + "qkv.weight", 3 * dim, dim, y, tokens, qkv,
                          p + "qkv.bias");
          !r) {
        return std::unexpected(r.error());
      }
      std::vector<float> q(static_cast<size_t>(tokens * dim)),
          k(static_cast<size_t>(tokens * dim)),
          v(static_cast<size_t>(tokens * dim));
      for (int64_t t = 0; t < tokens; ++t) {
        std::memcpy(q.data() + t * dim, qkv.data() + t * 3 * dim,
                    sizeof(float) * dim);
        std::memcpy(k.data() + t * dim, qkv.data() + t * 3 * dim + dim,
                    sizeof(float) * dim);
        std::memcpy(v.data() + t * dim, qkv.data() + t * 3 * dim + 2 * dim,
                    sizeof(float) * dim);
      }
      nn::rms_norm_heads(q, n.q_norm, static_cast<uint64_t>(hd), kRmsEps);
      nn::rms_norm_heads(k, n.k_norm, static_cast<uint64_t>(hd), kRmsEps);
      for (int64_t t = 0; t < tokens; ++t) {
        rotate(q.data() + t * dim, heads, hd, t, stride, cos, sin);
        rotate(k.data() + t * dim, heads, hd, t, stride, cos, sin);
      }
      attention(attn_out, q, k, v, tokens, heads, hd);
      if (auto r = linear(p + "proj.weight", dim, dim, attn_out, tokens, proj,
                          p + "proj.bias");
          !r) {
        return std::unexpected(r.error());
      }
      for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

      rms_norm_rows(x.data(), n.norm2, tokens, dim, y.data());
      if (auto r = linear(p + "ffn_up.weight", ffn, dim, y, tokens, u,
                          p + "ffn_up.bias");
          !r) {
        return std::unexpected(r.error());
      }
      if (auto r = linear(p + "ffn_gate.weight", ffn, dim, y, tokens, g,
                          p + "ffn_gate.bias");
          !r) {
        return std::unexpected(r.error());
      }
      for (int64_t i = 0; i < tokens * ffn; ++i) {
        const float hidden = u[static_cast<size_t>(i)];
        const float gate = g[static_cast<size_t>(i)];
        act[static_cast<size_t>(i)] =
            (4.0f * std::tanh(gate / 4.0f) * sigmoid(gate)) *
            (25.0f * std::tanh(hidden / 25.0f));
      }
      if (auto r = linear(p + "ffn_down.weight", dim, ffn, act, tokens, ffn_out,
                          p + "ffn_down.bias");
          !r) {
        return std::unexpected(r.error());
      }
      for (size_t i = 0; i < x.size(); ++i) x[i] += ffn_out[i];
    }
    return {};
  }

  std::expected<void, std::string> block(int64_t l, std::vector<float>& x,
                                         int64_t tokens, const float* cos,
                                         const float* sin, int64_t stride,
                                         const float* shift_a,
                                         const float* scale_a,
                                         const float* gate_a,
                                         const float* shift_f,
                                         const float* scale_f,
                                         const float* gate_f) {
    const int64_t dim = cfg.dim, hd = cfg.head_dim(), heads = cfg.num_heads;
    const int64_t ffn = cfg.ffn_dim;
    const std::string p = "layers." + std::to_string(l) + ".";
    const BlockNorms& n = norms[static_cast<size_t>(l)];

    std::vector<float> y(static_cast<size_t>(tokens * dim));
    rms_norm_rows(x.data(), n.norm1, tokens, dim, y.data());
    for (int64_t t = 0; t < tokens; ++t) {
      float* row = y.data() + t * dim;
      for (int64_t i = 0; i < dim; ++i) row[i] = row[i] * (1.0f + scale_a[i]) + shift_a[i];
    }
    std::vector<float> q(static_cast<size_t>(tokens * dim)),
        k(static_cast<size_t>(tokens * dim)),
        v(static_cast<size_t>(tokens * dim));
    if (auto r = linear(p + "proj.q.weight", dim, dim, y, tokens, q); !r)
      return std::unexpected(r.error());
    if (auto r = linear(p + "proj.k.weight", dim, dim, y, tokens, k); !r)
      return std::unexpected(r.error());
    if (auto r = linear(p + "proj.v.weight", dim, dim, y, tokens, v); !r)
      return std::unexpected(r.error());
    nn::rms_norm_heads(q, n.q_norm, static_cast<uint64_t>(hd), kRmsEps);
    nn::rms_norm_heads(k, n.k_norm, static_cast<uint64_t>(hd), kRmsEps);
    for (int64_t t = 0; t < tokens; ++t) {
      rotate(q.data() + t * dim, heads, hd, t, stride, cos, sin);
      rotate(k.data() + t * dim, heads, hd, t, stride, cos, sin);
    }
    std::vector<float> attn_out(static_cast<size_t>(tokens * dim));
    attention(attn_out, q, k, v, tokens, heads, hd);
    std::vector<float> o(static_cast<size_t>(tokens * dim));
    if (auto r = linear(p + "proj.o.weight", dim, dim, attn_out, tokens, o); !r)
      return std::unexpected(r.error());
    for (int64_t t = 0; t < tokens; ++t) {
      float* row = x.data() + t * dim;
      const float* add = o.data() + t * dim;
      for (int64_t i = 0; i < dim; ++i) row[i] += add[i] * gate_a[i];
    }

    rms_norm_rows(x.data(), n.norm2, tokens, dim, y.data());
    for (int64_t t = 0; t < tokens; ++t) {
      float* row = y.data() + t * dim;
      for (int64_t i = 0; i < dim; ++i) row[i] = row[i] * (1.0f + scale_f[i]) + shift_f[i];
    }
    std::vector<float> ug(static_cast<size_t>(tokens * 2 * ffn));
    if (auto r = linear(p + "proj.ffn_up_gate.weight", 2 * ffn, dim, y, tokens, ug); !r)
      return std::unexpected(r.error());
    std::vector<float> act(static_cast<size_t>(tokens * ffn));
    for (int64_t t = 0; t < tokens; ++t) {
      const float* hidden = ug.data() + t * 2 * ffn;
      const float* gate = hidden + ffn;
      float* dst = act.data() + t * ffn;
      for (int64_t i = 0; i < ffn; ++i) {
        dst[i] = (4.0f * std::tanh(gate[i] / 4.0f) * sigmoid(gate[i])) *
                 (25.0f * std::tanh(hidden[i] / 25.0f));
      }
    }
    std::vector<float> down(static_cast<size_t>(tokens * dim));
    if (auto r = linear(p + "proj.ffn_down.weight", dim, ffn, act, tokens, down); !r)
      return std::unexpected(r.error());
    for (int64_t t = 0; t < tokens; ++t) {
      float* row = x.data() + t * dim;
      const float* add = down.data() + t * dim;
      for (int64_t i = 0; i < dim; ++i) row[i] += add[i] * gate_f[i];
    }
    return {};
  }
};

std::expected<std::unique_ptr<DiffusionTransformer>, std::string>
DiffusionTransformer::load(const std::filesystem::path& dir) {
  auto impl = std::unique_ptr<Impl>(new Impl());
  if (auto r = impl->load_config(dir); !r) return std::unexpected(r.error());

  auto reader =
      safetensors::ShardedReader::load(dir / "diffusion_pytorch_model.safetensors.index.json");
  if (!reader) return std::unexpected("safetensors: " + reader.error().message);
  impl->reader = std::make_unique<safetensors::ShardedReader>(std::move(*reader));

  for (const auto& t : impl->reader->tensors()) {
    impl->weight_bytes += static_cast<int64_t>(
        t.info.n_elements * safetensors::dtype_size(t.info.dtype).value_or(0));
  }
  if (auto r = impl->load_weights(); !r) return std::unexpected(r.error());
  return std::unique_ptr<DiffusionTransformer>(new DiffusionTransformer(std::move(impl)));
}

DiffusionTransformer::DiffusionTransformer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DiffusionTransformer::~DiffusionTransformer() = default;

const DitConfig& DiffusionTransformer::config() const noexcept { return impl_->cfg; }

int64_t DiffusionTransformer::weight_bytes() const noexcept {
  return impl_->weight_bytes;
}

std::expected<DitSample, std::string> DiffusionTransformer::forward(
    std::span<const float> hidden, int64_t height, int64_t width,
    float timestep, std::span<const float> text, int64_t text_len,
    const std::vector<int64_t>& skip_layers, DitDebug* debug) const {
  Impl& m = *impl_;
  const DitConfig& cfg = m.cfg;
  const int64_t dim = cfg.dim, hd = cfg.head_dim(), heads = cfg.num_heads;
  const int64_t ffn = cfg.ffn_dim, channels = cfg.latent_channels;
  if (height <= 0 || width <= 0) return std::unexpected("forward: bad latent grid");
  if (text_len <= 0) return std::unexpected("forward: empty text sequence");
  const int64_t hw = height * width;
  const int64_t tokens = hw + text_len;
  if (hidden.size() < static_cast<size_t>(channels * hw)) {
    return std::unexpected("forward: hidden_states shorter than [C][H][W]");
  }
  if (text.size() < static_cast<size_t>(text_len * cfg.text_dim)) {
    return std::unexpected("forward: encoder_hidden_states shorter than [S][text_dim]");
  }
  for (int64_t l : skip_layers) {
    if (l < 0 || l >= cfg.num_layers) return std::unexpected("forward: skip_layers out of range");
  }

  // Time embedding: fp64 sinusoid (cos first, then sin), f32 MLP.
  const int64_t half = cfg.freq_dim / 2;
  std::vector<float> temb(static_cast<size_t>(cfg.freq_dim));
  for (int64_t i = 0; i < half; ++i) {
    const double freq = std::pow(10000.0, -double(i) / double(half));
    const double phase = double(timestep) * freq;
    temb[static_cast<size_t>(i)] = static_cast<float>(std::cos(phase));
    temb[static_cast<size_t>(half + i)] = static_cast<float>(std::sin(phase));
  }
  std::vector<float> t1(static_cast<size_t>(dim)), t2(static_cast<size_t>(dim));
  if (auto r = m.linear("time_embedding.0.weight", dim, cfg.freq_dim, temb, 1, t1,
                        "time_embedding.0.bias");
      !r) {
    return std::unexpected(r.error());
  }
  for (auto& v : t1) v = silu(v);
  if (auto r = m.linear("time_embedding.2.weight", dim, dim, t1, 1, t2,
                        "time_embedding.2.bias");
      !r) {
    return std::unexpected(r.error());
  }
  std::vector<float> mod(static_cast<size_t>(6 * dim)),
      head_mod(static_cast<size_t>(2 * dim));
  if (auto r = m.linear("shared_block_modulation.weight", 6 * dim, dim, t2, 1, mod,
                        "shared_block_modulation.bias");
      !r) {
    return std::unexpected(r.error());
  }
  if (auto r = m.linear("video_head_modulation.weight", 2 * dim, dim, t2, 1, head_mod,
                        "video_head_modulation.bias");
      !r) {
    return std::unexpected(r.error());
  }
  const float* shift_a = mod.data();
  const float* scale_a = shift_a + dim;
  const float* gate_a = shift_a + 2 * dim;
  const float* shift_f = shift_a + 3 * dim;
  const float* scale_f = shift_a + 4 * dim;
  const float* gate_f = shift_a + 5 * dim;
  const float* head_shift = head_mod.data();
  const float* head_scale = head_shift + dim;

  // 4-axis half-split RoPE: 16 bands per axis, (r, seq, h, w).
  const int64_t bands = hd / 8;
  const int64_t stride = 4 * bands;  // = head_dim / 2
  std::vector<float> rcos(static_cast<size_t>(tokens * stride)),
      rsin(static_cast<size_t>(tokens * stride));
  for (int64_t t = 0; t < tokens; ++t) {
    double coord[4] = {0.0, 0.0, 0.0, 0.0};
    if (t < hw) {
      const int64_t ih = t / width, iw = t % width;
      coord[1] = double(text_len);
      coord[2] = double(ih) - double(height - 1) / 2.0;
      coord[3] = double(iw) - double(width - 1) / 2.0;
    } else {
      coord[1] = double(t - hw);
    }
    for (int64_t axis = 0; axis < 4; ++axis) {
      for (int64_t j = 0; j < bands; ++j) {
        const double angle =
            coord[axis] * double(m.rope_bands[static_cast<size_t>(j)]);
        rcos[static_cast<size_t>(t * stride + axis * bands + j)] =
            static_cast<float>(std::cos(angle));
        rsin[static_cast<size_t>(t * stride + axis * bands + j)] =
            static_cast<float>(std::sin(angle));
      }
    }
  }

  // Video tokens: [C][H][W] -> [H*W][C], then Linear(C, dim).
  std::vector<float> video_src(static_cast<size_t>(hw * channels));
  for (int64_t p = 0; p < hw; ++p) {
    for (int64_t c = 0; c < channels; ++c) {
      video_src[static_cast<size_t>(p * channels + c)] =
          hidden[static_cast<size_t>(c * hw + p)];
    }
  }
  std::vector<float> x(static_cast<size_t>(tokens * dim), 0.0f);
  if (auto r = m.linear("video_embedder.weight", dim, channels, video_src, hw,
                        std::span<float>(x.data(), static_cast<size_t>(hw * dim)),
                        "video_embedder.bias");
      !r) {
    return std::unexpected(r.error());
  }

  // Text refinement on the text-only sequence, with the text half of the rope.
  std::vector<float> tx(static_cast<size_t>(text_len * dim));
  if (auto r = m.linear("text_embedder.proj.weight", dim, cfg.text_dim, text,
                        text_len, tx, "text_embedder.proj.bias");
      !r) {
    return std::unexpected(r.error());
  }
  if (auto r = m.text_blocks(tx, text_len, rcos.data() + hw * stride,
                             rsin.data() + hw * stride, stride);
      !r) {
    return std::unexpected(r.error());
  }
  std::memcpy(x.data() + hw * dim, tx.data(), sizeof(float) * text_len * dim);

  for (int64_t l = 0; l < cfg.num_layers; ++l) {
    const bool skip =
        std::find(skip_layers.begin(), skip_layers.end(), l) != skip_layers.end();
    if (!skip) {
      if (auto r = m.block(l, x, tokens, rcos.data(), rsin.data(), stride, shift_a,
                           scale_a, gate_a, shift_f, scale_f, gate_f);
          !r) {
        return std::unexpected(r.error());
      }
    }
    if (l == 0 && debug) debug->residual_after_first_block = x;
  }

  // Head on the video half: the residual stream's video rows, then
  // final_norm_video -> (1+scale, shift) -> Linear(dim, channels).
  if (debug) {
    debug->prehead.assign(x.begin(), x.begin() + hw * dim);
  }
  std::vector<float> y(static_cast<size_t>(hw * dim));
  m.rms_norm_rows(x.data(), m.final_norm, hw, dim, y.data());
  for (int64_t p = 0; p < hw; ++p) {
    float* row = y.data() + p * dim;
    for (int64_t i = 0; i < dim; ++i) {
      row[i] = row[i] * (1.0f + head_scale[i]) + head_shift[i];
    }
  }

  std::vector<float> out(static_cast<size_t>(hw * channels));
  if (auto r = m.linear("final_linear_video.weight", channels, dim, y, hw, out); !r)
    return std::unexpected(r.error());

  DitSample sample;
  sample.height = height;
  sample.width = width;
  sample.values.resize(static_cast<size_t>(channels * hw));
  for (int64_t p = 0; p < hw; ++p) {
    for (int64_t c = 0; c < channels; ++c) {
      sample.values[static_cast<size_t>(c * hw + p)] =
          out[static_cast<size_t>(p * channels + c)];
    }
  }
  return sample;
}

}  // namespace sonicboom::moss
