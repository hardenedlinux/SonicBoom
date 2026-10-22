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

// Qwen3.5 text encoder forward (MOSS-Image conditioning).
//
// Everything is computed in float32 from the memory-mapped bf16 weights. The
// reference model runs the activations in bf16; the float32 path is the same
// arithmetic without the per-op rounding, so the two agree up to accumulation
// order (measured in the S2 gate).
//
// The two layer kinds follow the reference implementations in
// transformers/models/qwen3_5/module_qwen3_5.py exactly:
//
//   linear_attention (Qwen3_5GatedDeltaNet)
//     in_proj_qkv / in_proj_z / in_proj_b / in_proj_a, causal depthwise
//     conv1d(k=4) + SiLU, beta = sigmoid(b), g = -exp(A_log) * softplus(a +
//     dt_bias), per-head L2 normalization of q/k, the recurrent gated delta
//     rule (equivalent to torch_recurrent_gated_delta_rule) with a 128x128
//     state per value head, then RMSNormGated(z) and out_proj.
//     Qwen3.5 repeats the 16 key heads onto the 32 value heads, so value head
//     h reads key/query head h/2.
//
//   full_attention (Qwen3_5Attention)
//     q_proj emits [query | gate] per head, q_norm/k_norm are head-wise RMS
//     norms with the (1 + weight) convention, RoPE rotates the first 64 dims
//     with HF's rotate_half layout, 16 query heads over 4 KV heads, and the
//     attention output is scaled by sigmoid(gate) before o_proj.

#include <sonicboom/model/moss_text.h>

#include "../planner/decision_registry.h"
#include "../safetensors/json.h"

#include <sonicboom/dtype.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/safetensors/reader.h>
#include <sonicboom/thread_pool.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace sonicboom::moss {
namespace {

constexpr std::string_view kPrefix = "language_model.";
constexpr float kL2NormEps = 1e-6f;  // FLA's use_qk_l2norm_in_kernel epsilon

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float silu(float x) { return x * sigmoid(x); }

// torch's softplus with beta=1, threshold=20.
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

std::expected<std::string, std::string> read_text(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in.good() && !in.eof()) return std::unexpected("cannot read " + p.string());
  return ss.str();
}

std::string shape_string(std::span<const uint64_t> shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

}  // namespace

struct TextEncoder::Impl {
  TextEncoderConfig cfg;
  std::unique_ptr<safetensors::ShardedReader> reader;
  int64_t weight_bytes = 0;

  // Per-layer f32 (1 + bf16 weight) norm vectors, precomputed at load.
  struct LayerNorms {
    std::vector<float> input;        // hidden
    std::vector<float> post_attn;    // hidden
    std::vector<float> q_norm;       // head_dim (full attention only)
    std::vector<float> k_norm;       // head_dim
  };
  std::vector<LayerNorms> norms;
  std::vector<float> final_norm;     // hidden

  // Linear-attention constants (f32 from bf16).
  struct LinearConsts {
    std::vector<float> a_log;      // num_value_heads
    std::vector<float> dt_bias;    // num_value_heads
    std::vector<float> gate_norm;  // value_head_dim (weight, not 1 + weight)
  };
  std::vector<LinearConsts> linear_consts;

  std::vector<float> rope_inv_freq;  // rotary_dim / 2

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

  std::vector<float> one_plus(std::span<const uint16_t> w) {
    std::vector<float> out(w.size());
    for (size_t i = 0; i < w.size(); ++i) out[i] = 1.0f + bf16_to_f32(w[i]);
    return out;
  }

  // --- configuration -------------------------------------------------------

  std::expected<void, std::string> load_config(const std::filesystem::path& dir) {
    auto text = read_text(dir / "config.json");
    if (!text) return std::unexpected(text.error());
    std::string error;
    auto root = safetensors::detail::parse_json(*text, &error);
    if (!root) return std::unexpected("config.json: " + error);
    const auto* c = root->find("text_config");
    if (!c) return std::unexpected("config.json: no text_config");

    auto int_of = [&](const char* key) -> std::expected<int64_t, std::string> {
      const auto* v = c->find(key);
      if (!v || !v->is_integer) {
        return std::unexpected(std::string("config.json: text_config.") + key +
                               " missing or not an integer");
      }
      return v->integer;
    };
    auto num_of = [&](const char* key, float fallback) -> float {
      const auto* v = c->find(key);
      return v && v->type == safetensors::detail::JsonValue::Type::Number
                 ? static_cast<float>(v->number)
                 : fallback;
    };

    for (auto [key, dest] :
         {std::pair{"hidden_size", &cfg.hidden_size},
          std::pair{"num_hidden_layers", &cfg.num_layers},
          std::pair{"intermediate_size", &cfg.intermediate_size},
          std::pair{"head_dim", &cfg.head_dim},
          std::pair{"num_attention_heads", &cfg.num_heads},
          std::pair{"num_key_value_heads", &cfg.num_kv_heads},
          std::pair{"linear_conv_kernel_dim", &cfg.conv_kernel},
          std::pair{"linear_key_head_dim", &cfg.key_head_dim},
          std::pair{"linear_num_key_heads", &cfg.num_key_heads},
          std::pair{"linear_value_head_dim", &cfg.value_head_dim},
          std::pair{"linear_num_value_heads", &cfg.num_value_heads},
          std::pair{"vocab_size", &cfg.vocab_size}}) {
      auto got = int_of(key);
      if (!got) return std::unexpected(got.error());
      *dest = *got;
    }
    cfg.rms_norm_eps = num_of("rms_norm_eps", 1e-6f);

    const auto* rope = c->find("rope_parameters");
    if (!rope) return std::unexpected("config.json: text_config.rope_parameters missing");
    const auto* theta = rope->find("rope_theta");
    const auto* partial = rope->find("partial_rotary_factor");
    if (!theta || !partial) {
      return std::unexpected("config.json: rope_theta / partial_rotary_factor missing");
    }
    cfg.rope_theta = static_cast<float>(theta->number);
    cfg.rotary_dim = static_cast<int64_t>(cfg.head_dim * partial->number);
    if (cfg.rotary_dim <= 0 || cfg.rotary_dim > 128 || cfg.rotary_dim % 2 != 0) {
      return std::unexpected("config.json: unsupported rotary dim " +
                             std::to_string(cfg.rotary_dim));
    }

    const auto* types = c->find("layer_types");
    if (!types || types->array.size() != static_cast<size_t>(cfg.num_layers)) {
      return std::unexpected("config.json: text_config.layer_types missing or wrong length");
    }
    cfg.layer_is_linear.resize(types->array.size());
    for (size_t i = 0; i < types->array.size(); ++i) {
      const std::string& kind = types->array[i].string;
      if (kind == "linear_attention") {
        cfg.layer_is_linear[i] = 1;
      } else if (kind == "full_attention") {
        cfg.layer_is_linear[i] = 0;
      } else {
        return std::unexpected("config.json: unknown layer type '" + kind + "'");
      }
    }
    return {};
  }

  std::expected<void, std::string> load_weights() {
    norms.resize(cfg.num_layers);
    linear_consts.resize(cfg.num_layers);
    for (int64_t l = 0; l < cfg.num_layers; ++l) {
      const std::string base = std::string(kPrefix) + "layers." + std::to_string(l) + ".";
      auto in = bf16(base + "input_layernorm.weight", {cfg.hidden_size});
      if (!in) return std::unexpected(in.error());
      norms[l].input = one_plus(*in);
      auto post = bf16(base + "post_attention_layernorm.weight", {cfg.hidden_size});
      if (!post) return std::unexpected(post.error());
      norms[l].post_attn = one_plus(*post);

      if (cfg.layer_is_linear[l]) {
        const std::string p = base + "linear_attn.";
        auto gate_norm = bf16(p + "norm.weight", {cfg.value_head_dim});
        if (!gate_norm) return std::unexpected(gate_norm.error());
        linear_consts[l].gate_norm = f32_of(*gate_norm);
        auto a_log = bf16(p + "A_log", {cfg.num_value_heads});
        if (!a_log) return std::unexpected(a_log.error());
        linear_consts[l].a_log = f32_of(*a_log);
        auto dt = bf16(p + "dt_bias", {cfg.num_value_heads});
        if (!dt) return std::unexpected(dt.error());
        linear_consts[l].dt_bias = f32_of(*dt);
      } else {
        const std::string p = base + "self_attn.";
        auto qn = bf16(p + "q_norm.weight", {cfg.head_dim});
        if (!qn) return std::unexpected(qn.error());
        norms[l].q_norm = one_plus(*qn);
        auto kn = bf16(p + "k_norm.weight", {cfg.head_dim});
        if (!kn) return std::unexpected(kn.error());
        norms[l].k_norm = one_plus(*kn);
      }
    }
    auto fn = bf16(std::string(kPrefix) + "norm.weight", {cfg.hidden_size});
    if (!fn) return std::unexpected(fn.error());
    final_norm = one_plus(*fn);

    rope_inv_freq.resize(cfg.rotary_dim / 2);
    for (int64_t j = 0; j < cfg.rotary_dim / 2; ++j) {
      rope_inv_freq[j] =
          static_cast<float>(std::pow(cfg.rope_theta, -2.0 * j / cfg.rotary_dim));
    }
    return {};
  }

  // --- pieces of the forward ----------------------------------------------

  // out rows = X @ W^T for a bf16 weight; aborts with the failing tensor name.
  std::expected<void, std::string> linear(std::string_view name,
                                          std::initializer_list<int64_t> shape,
                                          std::span<const float> X, int64_t n,
                                          std::span<float> Y) {
    auto w = bf16(name, shape);
    if (!w) return std::unexpected(w.error());
    if (!nn::matmul_bf16(*w, static_cast<uint64_t>(shape.begin()[1]), X,
                         static_cast<uint64_t>(n), Y)) {
      return std::unexpected("matmul_bf16 failed for " + std::string(name));
    }
    return {};
  }

  // y = x * rsqrt(mean(x^2) + eps) * weight, per row of `cols`.
  void rms_norm_rows(const float* x, const float* weight, int64_t rows,
                     int64_t cols, float eps, float* y) {
    for (int64_t r = 0; r < rows; ++r) {
      nn::rms_norm(std::span<const float>(x + r * cols, cols),
                   std::span<const float>(weight, cols), eps,
                   std::span<float>(y + r * cols, cols));
    }
  }

  void rope(float* q_or_k, int64_t heads, int64_t head_dim, int64_t pos) {
    const int64_t half = cfg.rotary_dim / 2;
    for (int64_t h = 0; h < heads; ++h) {
      float* v = q_or_k + h * head_dim;
      float rotated[128];
      for (int64_t i = 0; i < half; ++i) {
        const float angle = static_cast<float>(pos) * rope_inv_freq[i];
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        rotated[i] = v[i] * c - v[i + half] * s;
        rotated[i + half] = v[i + half] * c + v[i] * s;
      }
      for (int64_t i = 0; i < cfg.rotary_dim; ++i) v[i] = rotated[i];
    }
  }

  std::expected<void, std::string> linear_attention(int64_t l,
                                                    std::span<const float> h,
                                                    int64_t n, float* out) {
    const int64_t hidden = cfg.hidden_size;
    const int64_t key_dim = cfg.num_key_heads * cfg.key_head_dim;
    const int64_t value_dim = cfg.num_value_heads * cfg.value_head_dim;
    const int64_t qkv_dim = 2 * key_dim + value_dim;
    const std::string p = std::string(kPrefix) + "layers." + std::to_string(l) +
                          ".linear_attn.";

    std::vector<float> qkv(static_cast<size_t>(n * qkv_dim));
    if (auto r = linear(p + "in_proj_qkv.weight", {qkv_dim, hidden}, h, n, qkv); !r)
      return r;
    std::vector<float> z(static_cast<size_t>(n * value_dim));
    if (auto r = linear(p + "in_proj_z.weight", {value_dim, hidden}, h, n, z); !r)
      return r;
    std::vector<float> beta_raw(static_cast<size_t>(n * cfg.num_value_heads));
    if (auto r = linear(p + "in_proj_b.weight", {cfg.num_value_heads, hidden}, h, n,
                        beta_raw); !r)
      return r;
    std::vector<float> a_raw(static_cast<size_t>(n * cfg.num_value_heads));
    if (auto r = linear(p + "in_proj_a.weight", {cfg.num_value_heads, hidden}, h, n,
                        a_raw); !r)
      return r;

    // Causal depthwise conv over the mixed qkv, left pad conv_kernel - 1.
    auto conv_w = bf16(p + "conv1d.weight", {qkv_dim, 1, cfg.conv_kernel});
    if (!conv_w) return std::unexpected(conv_w.error());
    std::vector<float> mixed(static_cast<size_t>(n * qkv_dim));
    const int64_t k = cfg.conv_kernel;
    for (int64_t t = 0; t < n; ++t) {
      for (int64_t c = 0; c < qkv_dim; ++c) {
        float acc = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
          const int64_t src = t + j - (k - 1);
          if (src < 0) continue;
          acc += bf16_to_f32((*conv_w)[static_cast<size_t>(c * k + j)]) *
                 qkv[static_cast<size_t>(src * qkv_dim + c)];
        }
        mixed[static_cast<size_t>(t * qkv_dim + c)] = silu(acc);
      }
    }

    // Per-head gated delta rule. State layout: [head][k][v] in f32.
    const int64_t num_v = cfg.num_value_heads;
    const int64_t dk = cfg.key_head_dim;
    const int64_t dv = cfg.value_head_dim;
    const int64_t repeat = num_v / cfg.num_key_heads;
    std::vector<float> state(static_cast<size_t>(num_v * dk * dv), 0.0f);
    std::vector<float> attn_out(static_cast<size_t>(n * value_dim));

    ThreadPool::instance().parallel_for(0, static_cast<uint64_t>(num_v),
                                       [&](uint64_t hb, uint64_t he) {
      const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
      std::vector<float> qn(static_cast<size_t>(dk)), kn(static_cast<size_t>(dk));
      std::vector<float> kv_mem(static_cast<size_t>(dv));
      std::vector<float> delta(static_cast<size_t>(dv));
      for (uint64_t head = hb; head < he; ++head) {
        float* s = state.data() + head * dk * dv;
        const int64_t key_head = static_cast<int64_t>(head) / repeat;
        for (int64_t t = 0; t < n; ++t) {
          const float* row = mixed.data() + t * qkv_dim;
          const float* q_src = row + key_head * dk;
          const float* k_src = row + key_dim + key_head * dk;
          const float* v_src = row + 2 * key_dim + static_cast<int64_t>(head) * dv;

          double qsq = 0.0, ksq = 0.0;
          for (int64_t i = 0; i < dk; ++i) {
            qsq += double(q_src[i]) * q_src[i];
            ksq += double(k_src[i]) * k_src[i];
          }
          const float q_inv = static_cast<float>(1.0 / std::sqrt(qsq + kL2NormEps));
          const float k_inv = static_cast<float>(1.0 / std::sqrt(ksq + kL2NormEps));
          for (int64_t i = 0; i < dk; ++i) {
            qn[i] = q_src[i] * q_inv * scale;
            kn[i] = k_src[i] * k_inv;
          }

          const float beta = sigmoid(beta_raw[static_cast<size_t>(t * num_v + head)]);
          const float g = -std::exp(linear_consts[l].a_log[head]) *
                          softplus(a_raw[static_cast<size_t>(t * num_v + head)] +
                                   linear_consts[l].dt_bias[head]);
          const float decay = std::exp(g);

          // state *= decay; then kv_mem = state^T k (the reference's
          // (last_state * k).sum(-2) after the decay multiply).
          for (int64_t i = 0; i < dk; ++i) {
            float* row_i = s + i * dv;
            for (int64_t j = 0; j < dv; ++j) row_i[j] *= decay;
          }
          for (int64_t j = 0; j < dv; ++j) kv_mem[j] = 0.0f;
          for (int64_t i = 0; i < dk; ++i) {
            const float* row_i = s + i * dv;
            const float ki = kn[i];
            for (int64_t j = 0; j < dv; ++j) kv_mem[j] += row_i[j] * ki;
          }
          for (int64_t j = 0; j < dv; ++j) {
            delta[j] = (v_src[j] - kv_mem[j]) * beta;
          }
          float* o = attn_out.data() + (t * num_v + static_cast<int64_t>(head)) * dv;
          for (int64_t j = 0; j < dv; ++j) o[j] = 0.0f;
          for (int64_t i = 0; i < dk; ++i) {
            float* row_i = s + i * dv;
            const float ki = kn[i];
            const float qi = qn[i];
            for (int64_t j = 0; j < dv; ++j) {
              row_i[j] += ki * delta[j];
              o[j] += row_i[j] * qi;
            }
          }
        }
      }
    });

    // Gated head norm (weight, not 1 + weight) then out_proj.
    const float* gn = linear_consts[l].gate_norm.data();
    for (int64_t t = 0; t < n; ++t) {
      for (int64_t head = 0; head < num_v; ++head) {
        const int64_t base = t * value_dim + head * dv;
        double sumsq = 0.0;
        for (int64_t i = 0; i < dv; ++i) {
          sumsq += double(attn_out[static_cast<size_t>(base + i)]) *
                   attn_out[static_cast<size_t>(base + i)];
        }
        const float inv =
            static_cast<float>(1.0 / std::sqrt(sumsq / double(dv) + cfg.rms_norm_eps));
        for (int64_t i = 0; i < dv; ++i) {
          const size_t index = static_cast<size_t>(base + i);
          attn_out[index] = attn_out[index] * inv * gn[i] *
                            silu(z[index]);
        }
      }
    }
    return linear(p + "out_proj.weight", {hidden, value_dim}, attn_out, n,
                  std::span<float>(out, static_cast<size_t>(n * hidden)));
  }

  std::expected<void, std::string> full_attention(int64_t l,
                                                  std::span<const float> h,
                                                  int64_t n, float* out) {
    const int64_t hidden = cfg.hidden_size;
    const int64_t hd = cfg.head_dim;
    const int64_t heads = cfg.num_heads;
    const int64_t kv_heads = cfg.num_kv_heads;
    const int64_t group = heads / kv_heads;
    const std::string p = std::string(kPrefix) + "layers." + std::to_string(l) +
                          ".self_attn.";

    std::vector<float> qg(static_cast<size_t>(n * heads * hd * 2));
    if (auto r = linear(p + "q_proj.weight", {heads * hd * 2, hidden}, h, n, qg); !r)
      return r;
    std::vector<float> k(static_cast<size_t>(n * kv_heads * hd));
    if (auto r = linear(p + "k_proj.weight", {kv_heads * hd, hidden}, h, n, k); !r)
      return r;
    std::vector<float> v(static_cast<size_t>(n * kv_heads * hd));
    if (auto r = linear(p + "v_proj.weight", {kv_heads * hd, hidden}, h, n, v); !r)
      return r;

    // Split [query | gate] per head and normalize the queries in place.
    std::vector<float> q(static_cast<size_t>(n * heads * hd));
    std::vector<float> gate(static_cast<size_t>(n * heads * hd));
    for (int64_t t = 0; t < n; ++t) {
      for (int64_t hh = 0; hh < heads; ++hh) {
        const int64_t src = (t * heads + hh) * hd * 2;
        const int64_t dst = (t * heads + hh) * hd;
        std::memcpy(q.data() + dst, qg.data() + src, sizeof(float) * hd);
        std::memcpy(gate.data() + dst, qg.data() + src + hd, sizeof(float) * hd);
      }
    }

    for (int64_t t = 0; t < n; ++t) {
      nn::rms_norm_heads(std::span<float>(q.data() + t * heads * hd,
                                         static_cast<size_t>(heads * hd)),
                         norms[l].q_norm, static_cast<uint64_t>(hd),
                         cfg.rms_norm_eps);
      nn::rms_norm_heads(std::span<float>(k.data() + t * kv_heads * hd,
                                         static_cast<size_t>(kv_heads * hd)),
                         norms[l].k_norm, static_cast<uint64_t>(hd),
                         cfg.rms_norm_eps);
      rope(q.data() + t * heads * hd, heads, hd, t);
      rope(k.data() + t * kv_heads * hd, kv_heads, hd, t);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> attn(static_cast<size_t>(n * heads * hd));
    ThreadPool::instance().parallel_for(0, static_cast<uint64_t>(n * heads),
                                       [&](uint64_t b, uint64_t e) {
      std::vector<float> scores(static_cast<size_t>(n));
      for (uint64_t item = b; item < e; ++item) {
        const int64_t t = static_cast<int64_t>(item) / heads;
        const int64_t hh = static_cast<int64_t>(item) % heads;
        const int64_t kvh = hh / group;
        const float* qv = q.data() + (t * heads + hh) * hd;
        float max_score = -std::numeric_limits<float>::infinity();
        for (int64_t p = 0; p <= t; ++p) {
          const float* kv = k.data() + (p * kv_heads + kvh) * hd;
          double dot = 0.0;
          for (int64_t i = 0; i < hd; ++i) dot += double(qv[i]) * kv[i];
          scores[static_cast<size_t>(p)] = static_cast<float>(dot) * scale;
          max_score = std::max(max_score, scores[static_cast<size_t>(p)]);
        }
        double denom = 0.0;
        for (int64_t p = 0; p <= t; ++p) {
          const float e = std::exp(scores[static_cast<size_t>(p)] - max_score);
          scores[static_cast<size_t>(p)] = e;
          denom += e;
        }
        const float inv = static_cast<float>(1.0 / denom);
        float* o = attn.data() + (t * heads + hh) * hd;
        for (int64_t i = 0; i < hd; ++i) o[i] = 0.0f;
        for (int64_t p = 0; p <= t; ++p) {
          const float wgt = scores[static_cast<size_t>(p)] * inv;
          const float* vv = v.data() + (p * kv_heads + kvh) * hd;
          for (int64_t i = 0; i < hd; ++i) o[i] += wgt * vv[i];
        }
      }
    });

    for (size_t i = 0; i < attn.size(); ++i) attn[i] *= sigmoid(gate[i]);
    return linear(p + "o_proj.weight", {hidden, heads * hd}, attn, n,
                  std::span<float>(out, static_cast<size_t>(n * hidden)));
  }

  std::expected<void, std::string> mlp(int64_t l, std::span<const float> h,
                                       int64_t n, float* out) {
    const int64_t hidden = cfg.hidden_size;
    const int64_t inter = cfg.intermediate_size;
    const std::string p = std::string(kPrefix) + "layers." + std::to_string(l) + ".mlp.";
    std::vector<float> gate(static_cast<size_t>(n * inter));
    std::vector<float> up(static_cast<size_t>(n * inter));
    if (auto r = linear(p + "gate_proj.weight", {inter, hidden}, h, n, gate); !r)
      return r;
    if (auto r = linear(p + "up_proj.weight", {inter, hidden}, h, n, up); !r)
      return r;
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = silu(gate[i]) * up[i];
    return linear(p + "down_proj.weight", {hidden, inter}, gate, n,
                  std::span<float>(out, static_cast<size_t>(n * hidden)));
  }
};

std::expected<std::unique_ptr<TextEncoder>, std::string> TextEncoder::load(
    const std::filesystem::path& dir) {
  auto impl = std::unique_ptr<Impl>(new Impl());
  if (auto r = impl->load_config(dir); !r) return std::unexpected(r.error());

  auto reader = safetensors::ShardedReader::load(dir / "model.safetensors.index.json");
  if (!reader) {
    return std::unexpected("safetensors: " + reader.error().message);
  }
  impl->reader = std::make_unique<safetensors::ShardedReader>(std::move(*reader));

  auto embed = impl->bf16(std::string(kPrefix) + "embed_tokens.weight",
                          {impl->cfg.vocab_size, impl->cfg.hidden_size});
  if (!embed) return std::unexpected(embed.error());

  for (const auto& t : impl->reader->tensors()) {
    if (t.info.name.rfind(kPrefix, 0) != 0) continue;
    impl->weight_bytes +=
        static_cast<int64_t>(t.info.n_elements * safetensors::dtype_size(t.info.dtype).value_or(0));
  }
  if (auto r = impl->load_weights(); !r) return std::unexpected(r.error());

  // R2: the executor is an automatic choice made once per model/device. The
  // bf16 checkpoint does not fit the 8 GiB device in any quantized form that
  // keeps the conditioning gate's accuracy (Q8_0 is already ~9 GiB), so the
  // choice is the zero-copy CPU executor. Recorded here so sb-explain and the
  // persisted registry see it; SONICBOOM_MOSS_ENCODER pins it.
  const char* env = std::getenv("SONICBOOM_MOSS_ENCODER");
  planner::decisions::Entry entry;
  entry.name = "moss.encoder.executor";
  entry.scope = "text_encoder";
  entry.value = env && *env ? env : "cpu-bf16-mmap";
  entry.source = env && *env ? planner::decisions::Source::Env
                             : planner::decisions::Source::Default;
  entry.env = "SONICBOOM_MOSS_ENCODER";
  entry.inputs = "weights=" + std::to_string(impl->weight_bytes) + "B device=8GiB";
  entry.candidates = "cpu-bf16-mmap|gpu-q8_0|gpu-q4_K";
  entry.reason =
      "bf16 does not fit 8 GiB and Q8_0 (~9 GiB) does not either; the mmap'd "
      "CPU path needs no load-time quantization, so it is exact as well as "
      "the only fitting option";
  entry.numerics = "f32 activation, bf16 weights, no quantization";
  entry.vram_bytes = 0;
  planner::decisions::record(entry);

  return std::unique_ptr<TextEncoder>(new TextEncoder(std::move(impl)));
}

TextEncoder::TextEncoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TextEncoder::~TextEncoder() = default;
TextEncoder::TextEncoder(TextEncoder&&) noexcept = default;
TextEncoder& TextEncoder::operator=(TextEncoder&&) noexcept = default;

const TextEncoderConfig& TextEncoder::config() const noexcept { return impl_->cfg; }

int64_t TextEncoder::weight_bytes() const noexcept { return impl_->weight_bytes; }

std::expected<HiddenStates, std::string> TextEncoder::forward(
    std::span<const int32_t> ids) {
  Impl& m = *impl_;
  const int64_t n = static_cast<int64_t>(ids.size());
  if (n <= 0) return std::unexpected("forward: empty token sequence");
  const int64_t hidden = m.cfg.hidden_size;

  auto embed = m.bf16(std::string(kPrefix) + "embed_tokens.weight",
                      {m.cfg.vocab_size, hidden});
  if (!embed) return std::unexpected(embed.error());

  std::vector<float> x(static_cast<size_t>(n * hidden));
  for (int64_t t = 0; t < n; ++t) {
    if (ids[static_cast<size_t>(t)] < 0 ||
        ids[static_cast<size_t>(t)] >= m.cfg.vocab_size) {
      return std::unexpected("forward: token id out of range");
    }
    const uint16_t* row =
        embed->data() + static_cast<int64_t>(ids[static_cast<size_t>(t)]) * hidden;
    for (int64_t i = 0; i < hidden; ++i) {
      x[static_cast<size_t>(t * hidden + i)] = bf16_to_f32(row[i]);
    }
  }

  std::vector<float> h(static_cast<size_t>(n * hidden));
  std::vector<float> mix(static_cast<size_t>(n * hidden));
  for (int64_t l = 0; l < m.cfg.num_layers; ++l) {
    m.rms_norm_rows(x.data(), m.norms[l].input.data(), n, hidden,
                    m.cfg.rms_norm_eps, h.data());
    auto mixer = m.cfg.layer_is_linear[l] ? m.linear_attention(l, h, n, mix.data())
                                          : m.full_attention(l, h, n, mix.data());
    if (!mixer) return std::unexpected(mixer.error());
    for (size_t i = 0; i < x.size(); ++i) x[i] += mix[i];

    m.rms_norm_rows(x.data(), m.norms[l].post_attn.data(), n, hidden,
                    m.cfg.rms_norm_eps, h.data());
    if (auto r = m.mlp(l, h, n, mix.data()); !r) return std::unexpected(r.error());
    for (size_t i = 0; i < x.size(); ++i) x[i] += mix[i];
  }

  HiddenStates out;
  out.tokens = n;
  out.hidden_size = hidden;
  out.values.resize(static_cast<size_t>(n * hidden));
  m.rms_norm_rows(x.data(), m.final_norm.data(), n, hidden, m.cfg.rms_norm_eps,
                  out.values.data());
  return out;
}

}  // namespace sonicboom::moss
