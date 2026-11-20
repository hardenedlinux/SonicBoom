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

#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>

#include "../quant/foreign_dequant.h"
#include "../quant/quantize_q5_k.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numbers>
#include <string>
#include <string_view>

namespace sonicboom::model {

namespace {

// ggml type ids the target model uses (re-declared from the format; see the
// GGUF reader's type table and docs/gguf-reader.md).
constexpr uint32_t kF32 = 0;
constexpr uint32_t kBF16 = 30;
constexpr uint32_t kQ3K = 11;
constexpr uint32_t kQ4K = 12;
constexpr uint32_t kQ5K = 13;
constexpr uint32_t kQ4_0 = 2, kF16 = 1;

std::string dims_string(const std::vector<uint64_t>& dims) {
  std::string out = "[";
  for (size_t i=0;i<dims.size();++i) { if(i) out += ", "; out += std::to_string(dims[i]); }
  return out + "]";
}

// Types with no native kernel, converted to Q5_K at load time. Q5_0/Q8_0 used
// to be here; they are native now (QuantType::Q5_0 / Q8_0), so they bind
// through the ordinary quantized path and are never rewritten.
const char* foreign_type_name(uint32_t id) {
  switch(id) { case kF16:return "f16"; case kQ4_0:return "q4_0"; default: break; }
  return "unknown";
}

// Metadata readers: every value is widened losslessly by the reader's Value.
std::optional<uint64_t> read_u64(const gguf::Reader& r, std::string_view key) {
  const gguf::Value* v = r.find(key);
  if (!v) return std::nullopt;
  switch (v->type) {
    case gguf::ValueType::UInt8:
    case gguf::ValueType::UInt16:
    case gguf::ValueType::UInt32:
    case gguf::ValueType::UInt64:
      return v->u;
    case gguf::ValueType::Int8:
    case gguf::ValueType::Int16:
    case gguf::ValueType::Int32:
    case gguf::ValueType::Int64:
      return v->i < 0 ? std::optional<uint64_t>{} : uint64_t(v->i);
    default:
      return std::nullopt;
  }
}

std::optional<double> read_f64(const gguf::Reader& r, std::string_view key) {
  const gguf::Value* v = r.find(key);
  if (!v) return std::nullopt;
  if (v->type == gguf::ValueType::Float32 || v->type == gguf::ValueType::Float64)
    return v->f;
  return std::nullopt;
}

// sliding_window_pattern: an array of Bool (true = sliding-window layer).
std::optional<std::vector<bool>> read_bool_array(const gguf::Reader& r,
                                                 std::string_view key) {
  const gguf::Value* v = r.find(key);
  if (!v || v->type != gguf::ValueType::Array) return std::nullopt;
  std::vector<bool> out;
  out.reserve(v->elems.size());
  for (const gguf::Value& e : v->elems) {
    if (e.type != gguf::ValueType::Bool) return std::nullopt;
    out.push_back(e.b);
  }
  return out;
}

std::optional<std::string_view> read_string(const gguf::Reader& r,
                                            std::string_view key) {
  const gguf::Value* v = r.find(key);
  if (!v || v->type != gguf::ValueType::String) return std::nullopt;
  return std::string_view(v->s);
}

// An array of integers (e.g. rope dimension sections, per-layer expert widths).
std::optional<std::vector<uint64_t>> read_u64_array(const gguf::Reader& r,
                                                    std::string_view key) {
  const gguf::Value* v = r.find(key);
  if (!v || v->type != gguf::ValueType::Array) return std::nullopt;
  std::vector<uint64_t> out;
  out.reserve(v->elems.size());
  for (const gguf::Value& e : v->elems) {
    switch (e.type) {
      case gguf::ValueType::UInt8:
      case gguf::ValueType::UInt16:
      case gguf::ValueType::UInt32:
      case gguf::ValueType::UInt64:
        out.push_back(e.u);
        break;
      case gguf::ValueType::Int8:
      case gguf::ValueType::Int16:
      case gguf::ValueType::Int32:
      case gguf::ValueType::Int64:
        if (e.i < 0) return std::nullopt;
        out.push_back(uint64_t(e.i));
        break;
      default:
        return std::nullopt;
    }
  }
  return out;
}

// A per-layer flag array, either bools or integers (nonzero = set). The
// reference stores recurrent_layers as an integer array.
std::optional<std::vector<bool>> read_flag_array(const gguf::Reader& r,
                                                 std::string_view key) {
  const gguf::Value* v = r.find(key);
  if (!v || v->type != gguf::ValueType::Array) return std::nullopt;
  std::vector<bool> out;
  out.reserve(v->elems.size());
  for (const gguf::Value& e : v->elems) {
    switch (e.type) {
      case gguf::ValueType::Bool:
        out.push_back(e.b);
        break;
      case gguf::ValueType::UInt8:
      case gguf::ValueType::UInt16:
      case gguf::ValueType::UInt32:
      case gguf::ValueType::UInt64:
        out.push_back(e.u != 0);
        break;
      case gguf::ValueType::Int8:
      case gguf::ValueType::Int16:
      case gguf::ValueType::Int32:
      case gguf::ValueType::Int64:
        if (e.i < 0) return std::nullopt;
        out.push_back(e.i != 0);
        break;
      default:
        return std::nullopt;
    }
  }
  return out;
}

// A key the file may write as one integer or as a per-layer array. The config
// carries one value for the whole model, so a non-uniform array is refused
// rather than approximated. Absent reads as an empty optional.
std::expected<std::optional<uint64_t>, std::string>
read_u64_or_arr(const gguf::Reader& r, std::string_view key) {
  const gguf::Value* v = r.find(key);
  if (!v) return std::optional<uint64_t>();
  if (v->type != gguf::ValueType::Array) {
    const auto x = read_u64(r, key);
    if (!x) return std::unexpected(std::string(key) + " is not an integer");
    return x;
  }
  const auto arr = read_u64_array(r, key);
  if (!arr || arr->empty())
    return std::unexpected(std::string(key) + " is not an integer array");
  for (uint64_t x : *arr)
    if (x != arr->front())
      return std::unexpected(std::string(key) +
                             " has per-layer values that differ");
  return std::optional<uint64_t>(arr->front());
}

// --- generic RoPE frequency scaling -----------------------------------------
//
// `{arch}.rope.scaling.*` is the GGUF form of the long-context RoPE schemes
// (linear and YaRN; llama.cpp's parser lives in llama-model.cpp, the per-pair
// arithmetic in ggml's rope_yarn). The kernels here express a per-pair
// frequency multiplier as a DIVISOR in the same `freq_factors` slot Gemma's
// precomputed `rope_freqs.weight` uses:
//
//   theta(i) = pos * base^(-2i/d) / ff[i]
//
// llama.cpp computes theta = theta_extrap * alpha(i), where
//   alpha(i) = freq_scale*(1 - ramp(i)*ext) + ramp(i)*ext
// with freq_scale = 1/factor, ext = 0 for linear and 1 for YaRN (a YaRN file
// without a factor is the identity, because freq_scale = 1). So ff[i] =
// 1/alpha(i) reproduces every scheme in the existing kernels, on CPU and GPU,
// with no model-specific code and no new kernel entry point.
//
// The ramp and its correction dims are ggml's verbatim formulas:
//   corr(beta) = n_rot*ln(orig_ctx/(beta*2*pi)) / (2*ln(base))
//   lo = max(0, floor(corr(beta_fast)))   hi = min(n_rot-1, ceil(corr(beta_slow)))
//   ramp(i) = 1 - clamp((i - lo)/max(0.001, hi - lo), 0, 1)
// YaRN also has a magnitude term: llama.cpp multiplies cos and sin by
// m = (1 + 0.1*ln factor) (times `attn_factor`), i.e. q and k each by m and the
// scores by m^2. It is not a frequency, so it does not belong in ff; it is
// returned in `rope_attn_scale`, which the emitter folds into the attention
// temperature (see the weights field). That is exact only with full rope, which
// is the only YaRN configuration accepted here.
//
// A precomputed `rope_freqs.weight` table, when present, is multiplied in: the
// kernels divide by ff, and two chained divisors compose by multiplication.
//
// Returns an error string on unsupported metadata; sets nothing when the file
// requests no scaling. `head_dim` is the largest per-layer head dim (the table
// covers head_dim/2 pairs; pairs past `rope.dimension_count` keep ff = 1).
std::optional<std::string> apply_rope_scaling(const gguf::Reader& r,
                                              std::string_view arch,
                                              uint32_t head_dim, float base,
                                              uint64_t ctx_train,
                                              Gemma4Weights& w) {
  const std::string p(arch);
  auto key = [&](std::string_view suffix) { return p + std::string(suffix); };

  const auto type = read_string(r, key(".rope.scaling.type"));
  if (!type || *type == "none") return std::nullopt;
  if (*type != "linear" && *type != "yarn")
    return "unsupported rope scaling type: " + std::string(*type);

  const auto factor = read_f64(r, key(".rope.scaling.factor"));
  if (!factor || *factor <= 0.0) return std::nullopt;  // unset == identity
  // `attn_factor` is YaRN's magnitude scale (llama.cpp multiplies it into the
  // net mscale, see below); anything else in the file is frequency scaling.
  const double attn_factor =
      read_f64(r, key(".rope.scaling.attn_factor")).value_or(1.0);
  if (const auto lm = read_f64(r, key(".rope.scaling.yarn_log_multiplier"));
      lm && std::abs(*lm - 1.0) > 1e-6)
    return "unsupported yarn_log_multiplier (magnitude scaling)";

  const bool yarn = (*type == "yarn");
  const double fs = 1.0 / *factor;
  uint32_t n_rot = head_dim;
  if (const auto v = read_u64(r, key(".rope.dimension_count"));
      v && *v > 0 && *v <= head_dim)
    n_rot = uint32_t(*v);
  const uint64_t orig =
      read_u64(r, key(".rope.scaling.original_context_length")).value_or(ctx_train);
  const double beta_fast =
      read_f64(r, key(".rope.scaling.yarn_beta_fast")).value_or(32.0);
  const double beta_slow =
      read_f64(r, key(".rope.scaling.yarn_beta_slow")).value_or(1.0);

  double lo = 0.0, hi = 0.0;
  if (yarn) {
    const auto corr = [&](double beta) {
      return double(n_rot) *
             std::log(double(orig) / (beta * 2.0 * std::numbers::pi)) /
             (2.0 * std::log(double(base)));
    };
    lo = std::max(0.0, std::floor(corr(beta_fast)));
    hi = std::min(double(n_rot) - 1.0, std::ceil(corr(beta_slow)));
  }

  std::vector<float> table(head_dim / 2, 1.0f);
  for (size_t i = 0; i < n_rot / 2; ++i) {
    double alpha = fs;
    if (yarn) {
      const double y = (double(i) - lo) / std::max(0.001, hi - lo);
      const double ramp = 1.0 - std::min(1.0, std::max(0.0, y));
      alpha = fs * (1.0 - ramp) + ramp;  // ext = 1 for YaRN
    }
    table[i] = alpha > 0.0 ? float(1.0 / alpha) : 1.0f;
  }

  if (!w.rope_freqs.empty()) {
    if (w.rope_freqs.size() != table.size())
      return "rope scaling covers " + std::to_string(table.size()) +
             " pairs but rope_freqs.weight has " +
             std::to_string(w.rope_freqs.size());
    for (size_t i = 0; i < table.size(); ++i) table[i] *= w.rope_freqs[i];
  }
  w.rope_freqs_table = std::move(table);
  w.rope_freqs = w.rope_freqs_table;

  // YaRN's magnitude scale. llama.cpp computes m = (factor>1 ? 1 + 0.1*ln
  // factor : 1) and applies it to cos and sin (the `attn_factor` key scales it
  // once more), so q and k are each multiplied by m and the attention scores by
  // m^2. Folding m^2 into the emitted attention temperature is exact only when
  // EVERY rope pair is rotated: with partial rope the unrotated tail is also
  // m-scaled by that formulation, which a score scale cannot reproduce. Refuse
  // that combination instead of computing a different model.
  if (yarn) {
    if (n_rot != head_dim)
      return "YaRN scaling with partial rope (rope.dimension_count " +
             std::to_string(n_rot) + " of " + std::to_string(head_dim) +
             ") is not supported";
    const double m =
        (*factor <= 1.0 ? 1.0 : 1.0 + 0.1 * std::log(*factor)) * attn_factor;
    w.rope_attn_scale = float(m * m);
  }
  return std::nullopt;
}

// Tensor binding with exact type + shape validation. All errors accumulate into
// `errs` (a newline-joined string) so a single load reports every mismatch.
bool dims_match(const std::vector<uint64_t>& got,
                std::initializer_list<uint64_t> want) {
  if (got.size() != want.size()) return false;
  size_t i = 0;
  for (uint64_t w : want) {
    if (got[i++] != w) return false;
  }
  return true;
}

std::optional<quant::QuantizedTensor> bind_quant(
    const gguf::Reader& r, std::string_view name,
    std::initializer_list<uint32_t> expect_types,
    std::initializer_list<uint64_t> dims, std::string& errs,
    std::vector<std::vector<uint8_t>>& owned,
    std::map<uint32_t,uint64_t>& converted) {
  const gguf::TensorInfo* t = r.tensor(name);
  if (!t) {
    errs += "missing tensor: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  bool type_ok = false;
  for (uint32_t et : expect_types)
    if (t->type == et) type_ok = true;
  const bool shape_ok = dims_match(t->dims, dims);
  const bool native_quant_ok = quant::quant_type_from_ggml(t->type).has_value();
  // The `% 256` test is the Q5_K super-block geometry the conversion targets,
  // so only types that are still foreign are considered here at all.
  const bool foreign_ok = !quant::quant_type_from_ggml(t->type) &&
      (t->type == kF16 || t->type == kQ4_0) &&
      !t->dims.empty() && t->dims[0] % 256 == 0;
  if ((!type_ok && !native_quant_ok && !foreign_ok) || !shape_ok) {
    errs += "tensor type/shape mismatch: ";
    errs += name; errs += " (ggml type "; errs += std::to_string(t->type);
    errs += ", dims "; errs += dims_string(t->dims); errs += ")";
    errs += "\n";
    return std::nullopt;
  }
  auto data = r.tensor_data(*t);
  if (!data) {
    errs += "tensor data unresolvable: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  auto qt = quant::quantized_tensor_from_gguf(t->type, t->dims, *data);
  if (!qt && foreign_ok) {
    auto f32 = quant::dequant_foreign_to_f32(t->type,
        reinterpret_cast<const uint8_t*>(data->data()), t->n_elements);
    if (!f32) {
      errs += "cannot convert tensor "; errs += name; errs += " (ggml type ";
      errs += std::to_string(t->type); errs += ", dims "; errs += dims_string(t->dims);
      errs += "): "; errs += f32.error(); errs += "\n";
      return std::nullopt;
    }
    auto bytes = quant::quantize_f32_to_q5_k(f32->data(), f32->size());
    owned.push_back(std::move(bytes));
    auto& buffer = owned.back();
    qt.emplace(quant::QuantType::Q5_K, t->dims,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(buffer.data()), buffer.size()));
    ++converted[t->type];
  }
  if (!qt || !qt->valid()) {
    errs += "quantized tensor invalid: ";
    errs += name; errs += " (ggml type "; errs += std::to_string(t->type);
    errs += ", dims "; errs += dims_string(t->dims); errs += ")";
    errs += "\n";
    return std::nullopt;
  }
  return qt;
}

std::optional<std::span<const float>> bind_f32(const gguf::Reader& r,
                                               std::string_view name,
                                               uint64_t n, std::string& errs) {
  const gguf::TensorInfo* t = r.tensor(name);
  if (!t || t->type != kF32 || t->dims.size() != 1 || t->dims[0] != n) {
    errs += "f32 tensor mismatch: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  auto data = r.tensor_data(*t);
  if (!data || data->size() != n * sizeof(float)) {
    errs += "f32 tensor data unresolvable: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  // GGUF aligns tensor data to 32 bytes, so the float* cast is aligned.
  return std::span<const float>(reinterpret_cast<const float*>(data->data()), n);
}

// 2-D f32 tensor (e.g. the per-layer gate matrices inp_gate / proj). Shape is
// validated exactly; the element count is taken from the tensor info.
std::optional<std::span<const float>> bind_f32_2d(const gguf::Reader& r,
                                                  std::string_view name,
                                                  std::initializer_list<uint64_t> dims,
                                                  std::string& errs) {
  const gguf::TensorInfo* t = r.tensor(name);
  if (!t || t->type != kF32 || !dims_match(t->dims, dims)) {
    errs += "f32 2d tensor mismatch: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  auto data = r.tensor_data(*t);
  if (!data || data->size() != t->n_elements * sizeof(float)) {
    errs += "f32 tensor data unresolvable: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  return std::span<const float>(reinterpret_cast<const float*>(data->data()),
                                t->n_elements);
}

// A projection stored either way ("as in the file", the way the mixture's
// router is bound): an f32 tensor is held as f32, anything else goes through the
// quantized path (a native quantized type as it is, F16/Q4_0 converted).
void bind_quant_or_f32(const gguf::Reader& r, std::string_view name,
                       std::initializer_list<uint64_t> dims,
                       quant::QuantizedTensor& quant_out,
                       std::span<const float>& f32_out, std::string& errs,
                       std::vector<std::vector<uint8_t>>& owned,
                       std::map<uint32_t,uint64_t>& converted) {
  const gguf::TensorInfo* t = r.tensor(name);
  if (t && t->type == kF32) {
    if (auto s = bind_f32_2d(r, name, dims, errs)) f32_out = *s;
    return;
  }
  if (auto q = bind_quant(r, name, {}, dims, errs, owned, converted))
    quant_out = *q;
}

std::optional<std::span<const uint16_t>> bind_bf16(const gguf::Reader& r,
                                                   std::string_view name,
                                                   std::initializer_list<uint64_t> dims,
                                                   uint64_t n, std::string& errs) {
  const gguf::TensorInfo* t = r.tensor(name);
  if (!t || t->type != kBF16 || !dims_match(t->dims, dims)) {
    errs += "bf16 tensor mismatch: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  auto data = r.tensor_data(*t);
  if (!data || data->size() != n * sizeof(uint16_t)) {
    errs += "bf16 tensor data unresolvable: ";
    errs += name;
    errs += "\n";
    return std::nullopt;
  }
  return std::span<const uint16_t>(reinterpret_cast<const uint16_t*>(data->data()),
                                   n);
}

// One 2-D view per expert over a stacked quantized tensor
// `[per_expert_dims..., n_expert]` (GGUF order, expert last). The slicing and
// its validation live in weights.cpp (expert_views); this resolves the tensor
// and reports its name with any error.
std::vector<quant::QuantizedTensor> bind_experts(
    const gguf::Reader& r, std::string_view name,
    std::initializer_list<uint64_t> per_expert_dims, uint64_t n_expert,
    std::string& errs) {
  const gguf::TensorInfo* t = r.tensor(name);
  if (!t) {
    errs += "missing tensor: ";
    errs += name;
    errs += "\n";
    return {};
  }
  auto data = r.tensor_data(*t);
  if (!data) {
    errs += "tensor data unresolvable: ";
    errs += name;
    errs += "\n";
    return {};
  }
  auto views = expert_views(t->type, t->dims, *data, n_expert,
                            std::vector<uint64_t>(per_expert_dims));
  if (!views) {
    errs += "cannot bind expert tensor ";
    errs += name;
    errs += ": ";
    errs += views.error();
    errs += "\n";
    return {};
  }
  return std::move(*views);
}

// The merged form of the routed gate and up projections:
// `[n_embd, 2*ff, n_expert]`, split per expert into gate/up halves.
bool bind_expert_gate_up(const gguf::Reader& r, std::string_view name,
                         uint64_t n_embd, uint64_t ff, uint64_t n_expert,
                         std::vector<quant::QuantizedTensor>& gate,
                         std::vector<quant::QuantizedTensor>& up,
                         std::string& errs) {
  const gguf::TensorInfo* t = r.tensor(name);
  if (!t) {
    errs += "missing tensor: ";
    errs += name;
    errs += "\n";
    return false;
  }
  auto data = r.tensor_data(*t);
  if (!data) {
    errs += "tensor data unresolvable: ";
    errs += name;
    errs += "\n";
    return false;
  }
  auto views = expert_gate_up_views(t->type, t->dims, *data, n_expert, n_embd,
                                    ff);
  if (!views) {
    errs += "cannot bind expert tensor ";
    errs += name;
    errs += ": ";
    errs += views.error();
    errs += "\n";
    return false;
  }
  gate = std::move(views->gate);
  up = std::move(views->up);
  return true;
}

} // namespace

std::expected<Gemma4Config, std::string> load_gemma4_config(const gguf::Reader& r) {
  Gemma4Config c;
  c.arch = ModelArch::Gemma4;

  const auto block_count = read_u64(r, "gemma4.block_count");
  const auto embd_len = read_u64(r, "gemma4.embedding_length");
  const auto ff_len = read_u64(r, "gemma4.feed_forward_length");
  const auto head_count = read_u64(r, "gemma4.attention.head_count");
  const auto head_count_kv = read_u64(r, "gemma4.attention.head_count_kv");
  const auto key_len = read_u64(r, "gemma4.attention.key_length");
  const auto key_len_swa = read_u64(r, "gemma4.attention.key_length_swa");
  const auto per_layer_input =
      read_u64(r, "gemma4.embedding_length_per_layer_input");
  const auto context_len = read_u64(r, "gemma4.context_length");
  const auto sliding_window = read_u64(r, "gemma4.attention.sliding_window");
  const auto pattern = read_bool_array(r, "gemma4.attention.sliding_window_pattern");
  const auto rms_eps = read_f64(r, "gemma4.attention.layer_norm_rms_epsilon");
  const auto softcap = read_f64(r, "gemma4.final_logit_softcapping");
  const auto freq_base = read_f64(r, "gemma4.rope.freq_base");
  const auto freq_base_swa = read_f64(r, "gemma4.rope.freq_base_swa");
  const auto shared_kv = read_u64(r, "gemma4.attention.shared_kv_layers");

  auto invalid = [](const char* what) {
    return std::unexpected(std::string("missing/invalid metadata: ") + what);
  };
  if (!block_count || *block_count == 0) return invalid("gemma4.block_count");
  if (!embd_len || *embd_len == 0) return invalid("gemma4.embedding_length");
  if (!ff_len || *ff_len == 0) return invalid("gemma4.feed_forward_length");
  if (!head_count || *head_count == 0)
    return invalid("gemma4.attention.head_count");
  if (!head_count_kv || *head_count_kv == 0)
    return invalid("gemma4.attention.head_count_kv");
  if (!key_len || *key_len == 0) return invalid("gemma4.attention.key_length");
  if (!key_len_swa || *key_len_swa == 0)
    return invalid("gemma4.attention.key_length_swa");
  if (!per_layer_input || *per_layer_input == 0)
    return invalid("gemma4.embedding_length_per_layer_input");
  if (!context_len || *context_len == 0) return invalid("gemma4.context_length");
  if (!sliding_window || *sliding_window == 0)
    return invalid("gemma4.attention.sliding_window");
  if (!pattern) return invalid("gemma4.attention.sliding_window_pattern");
  if (!rms_eps || *rms_eps < 0.0)
    return invalid("gemma4.attention.layer_norm_rms_epsilon");
  if (!softcap || *softcap <= 0.0)
    return invalid("gemma4.final_logit_softcapping");
  if (!freq_base || *freq_base <= 0.0) return invalid("gemma4.rope.freq_base");
  if (!freq_base_swa || *freq_base_swa <= 0.0)
    return invalid("gemma4.rope.freq_base_swa");
  if (!shared_kv) return invalid("gemma4.attention.shared_kv_layers");

  c.block_count = uint32_t(*block_count);
  c.embedding_length = uint32_t(*embd_len);
  c.feed_forward_length = uint32_t(*ff_len);
  c.head_count = uint32_t(*head_count);
  c.head_count_kv = uint32_t(*head_count_kv);
  c.head_dim_global = uint32_t(*key_len);
  c.head_dim_swa = uint32_t(*key_len_swa);
  c.per_layer_input = uint32_t(*per_layer_input);
  c.context_length = *context_len;
  c.sliding_window = *sliding_window;
  c.rms_epsilon = float(*rms_eps);
  c.final_logit_softcapping = float(*softcap);
  c.rope_base = float(*freq_base);
  c.rope_base_swa = float(*freq_base_swa);
  c.shared_kv_layers = uint32_t(*shared_kv);
  c.sliding_window_pattern = *pattern;

  // token_embd.shape[1] defines vocab_size (shared with the weight binder).
  {
    const gguf::TensorInfo* t = r.tensor("token_embd.weight");
    if (!t || t->dims.size() != 2 || t->dims[0] != c.embedding_length ||
        t->dims[1] == 0) {
      return std::unexpected(std::string("token_embd.weight absent or malformed"));
    }
    c.vocab_size = t->dims[1];
  }

  if (!c.valid()) return std::unexpected(std::string("inconsistent config"));
  return c;
}

std::expected<Gemma4Config, std::string> load_qwen2_config(const gguf::Reader& r) {
  Gemma4Config c;
  c.arch = ModelArch::Qwen2;

  const auto block_count = read_u64(r, "qwen2.block_count");
  const auto context_len = read_u64(r, "qwen2.context_length");
  const auto embd_len = read_u64(r, "qwen2.embedding_length");
  const auto ff_len = read_u64(r, "qwen2.feed_forward_length");
  const auto head_count = read_u64(r, "qwen2.attention.head_count");
  const auto head_count_kv = read_u64(r, "qwen2.attention.head_count_kv");
  const auto rms_eps = read_f64(r, "qwen2.attention.layer_norm_rms_epsilon");
  const auto freq_base = read_f64(r, "qwen2.rope.freq_base");

  auto invalid = [](const char* what) {
    return std::unexpected(std::string("missing/invalid metadata: ") + what);
  };
  if (!block_count || *block_count == 0) return invalid("qwen2.block_count");
  if (!context_len || *context_len == 0) return invalid("qwen2.context_length");
  if (!embd_len || *embd_len == 0) return invalid("qwen2.embedding_length");
  if (!ff_len || *ff_len == 0) return invalid("qwen2.feed_forward_length");
  if (!head_count || *head_count == 0)
    return invalid("qwen2.attention.head_count");
  if (!head_count_kv || *head_count_kv == 0)
    return invalid("qwen2.attention.head_count_kv");
  if (!rms_eps || *rms_eps < 0.0)
    return invalid("qwen2.attention.layer_norm_rms_epsilon");
  if (!freq_base || *freq_base <= 0.0) return invalid("qwen2.rope.freq_base");

  c.block_count = uint32_t(*block_count);
  c.context_length = *context_len;
  c.embedding_length = uint32_t(*embd_len);
  c.feed_forward_length = uint32_t(*ff_len);
  c.head_count = uint32_t(*head_count);
  c.head_count_kv = uint32_t(*head_count_kv);
  c.head_dim_global = c.embedding_length / c.head_count;  // 896 / 14 == 64
  c.head_dim_swa = 0;              // no sliding-window attention
  c.per_layer_input = 0;           // no per-layer embedding input
  c.sliding_window = 0;
  c.rms_epsilon = float(*rms_eps);
  c.final_logit_softcapping = 0.0f;  // no logit softcap
  c.rope_base = float(*freq_base);
  c.rope_base_swa = 0.0f;
  c.shared_kv_layers = 0;          // no shared KV
  c.sliding_window_pattern.clear();  // every layer is global

  // token_embd.shape[1] defines vocab_size (shared with the weight binder).
  {
    const gguf::TensorInfo* t = r.tensor("token_embd.weight");
    if (!t || t->dims.size() != 2 || t->dims[0] != c.embedding_length ||
        t->dims[1] == 0) {
      return std::unexpected(std::string("token_embd.weight absent or malformed"));
    }
    c.vocab_size = t->dims[1];
  }

  if (!c.valid()) return std::unexpected(std::string("inconsistent config"));
  return c;
}

// Qwen3.6-MoE: 40 gated-DeltaNet blocks with a full-attention block every
// `full_attention_interval`-th, each block a 256-expert MoE (8 selected) plus
// one shared expert, partial sectioned RoPE, and gated attention (the Q
// projection is twice as wide). The hparam readers follow the reference's
// qwen35moe load_arch_hparams: expert feed-forward length may be a per-layer
// array, the recurrent-layer flags are an explicit array when the file carries
// one and are derived from the interval otherwise, and the DeltaNet
// hyperparameters are required.
std::expected<Gemma4Config, std::string> load_qwen35moe_config(
    const gguf::Reader& r) {
  Gemma4Config c;
  c.arch = ModelArch::Qwen35Moe;

  const auto block_count = read_u64(r, "qwen35moe.block_count");
  const auto context_len = read_u64(r, "qwen35moe.context_length");
  const auto embd_len = read_u64(r, "qwen35moe.embedding_length");
  const auto ff_res = read_u64_or_arr(r, "qwen35moe.feed_forward_length");
  const auto expert_ff_res =
      read_u64_or_arr(r, "qwen35moe.expert_feed_forward_length");
  const auto shared_ff =
      read_u64(r, "qwen35moe.expert_shared_feed_forward_length");
  const auto expert_count = read_u64(r, "qwen35moe.expert_count");
  const auto expert_used = read_u64(r, "qwen35moe.expert_used_count");
  const auto weights_scale = read_f64(r, "qwen35moe.expert_weights_scale");
  const auto head_count = read_u64(r, "qwen35moe.attention.head_count");
  const auto head_count_kv = read_u64(r, "qwen35moe.attention.head_count_kv");
  const auto key_len = read_u64(r, "qwen35moe.attention.key_length");
  const auto value_len = read_u64(r, "qwen35moe.attention.value_length");
  const auto rms_eps = read_f64(r, "qwen35moe.attention.layer_norm_rms_epsilon");
  const auto freq_base = read_f64(r, "qwen35moe.rope.freq_base");
  const auto rope_dims = read_u64(r, "qwen35moe.rope.dimension_count");
  const auto rope_sections = read_u64_array(r, "qwen35moe.rope.dimension_sections");
  const auto interval = read_u64(r, "qwen35moe.full_attention_interval");
  const auto recurrent =
      read_flag_array(r, "qwen35moe.attention.recurrent_layers");
  const auto ssm_conv = read_u64(r, "qwen35moe.ssm.conv_kernel");
  const auto ssm_inner = read_u64(r, "qwen35moe.ssm.inner_size");
  const auto ssm_state = read_u64(r, "qwen35moe.ssm.state_size");
  const auto ssm_dt = read_u64(r, "qwen35moe.ssm.time_step_rank");
  const auto ssm_group = read_u64(r, "qwen35moe.ssm.group_count");
  const auto nextn = read_u64(r, "qwen35moe.nextn_predict_layers");

  auto invalid = [](const char* what) {
    return std::unexpected(std::string("missing/invalid metadata: ") + what);
  };
  if (!ff_res)
    return std::unexpected(std::string("invalid metadata: ") + ff_res.error());
  if (!expert_ff_res)
    return std::unexpected(std::string("invalid metadata: ") +
                           expert_ff_res.error());
  const std::optional<uint64_t> ff_len = *ff_res;
  const std::optional<uint64_t> expert_ff = *expert_ff_res;
  if (!block_count || *block_count == 0) return invalid("qwen35moe.block_count");
  if (!context_len || *context_len == 0) return invalid("qwen35moe.context_length");
  if (!embd_len || *embd_len == 0) return invalid("qwen35moe.embedding_length");
  if (ff_len && *ff_len == 0) return invalid("qwen35moe.feed_forward_length");
  if (!head_count || *head_count == 0)
    return invalid("qwen35moe.attention.head_count");
  if (!head_count_kv || *head_count_kv == 0)
    return invalid("qwen35moe.attention.head_count_kv");
  if (!key_len || *key_len == 0) return invalid("qwen35moe.attention.key_length");
  if (!value_len || *value_len != *key_len)
    return invalid("qwen35moe.attention.value_length (must equal key_length)");
  if (!rms_eps || *rms_eps < 0.0)
    return invalid("qwen35moe.attention.layer_norm_rms_epsilon");
  if (!freq_base || *freq_base <= 0.0) return invalid("qwen35moe.rope.freq_base");
  if (!expert_count || *expert_count == 0) return invalid("qwen35moe.expert_count");
  if (!expert_used || *expert_used == 0 || *expert_used > *expert_count)
    return invalid("qwen35moe.expert_used_count");
  if (!rope_sections || rope_sections->size() != 4)
    return invalid("qwen35moe.rope.dimension_sections (4 sections)");
  if (!ssm_conv || *ssm_conv == 0) return invalid("qwen35moe.ssm.conv_kernel");
  if (!ssm_inner || *ssm_inner == 0) return invalid("qwen35moe.ssm.inner_size");
  if (!ssm_state || *ssm_state == 0) return invalid("qwen35moe.ssm.state_size");
  if (!ssm_dt || *ssm_dt == 0) return invalid("qwen35moe.ssm.time_step_rank");
  if (!ssm_group || *ssm_group == 0) return invalid("qwen35moe.ssm.group_count");

  c.block_count = uint32_t(*block_count);
  c.context_length = *context_len;
  c.embedding_length = uint32_t(*embd_len);
  c.head_count = uint32_t(*head_count);
  c.head_count_kv = uint32_t(*head_count_kv);
  c.head_dim_global = uint32_t(*key_len);
  c.head_dim_swa = 0;         // no sliding-window attention
  c.per_layer_input = 0;
  c.sliding_window = 0;
  c.rms_epsilon = float(*rms_eps);
  c.final_logit_softcapping = 0.0f;
  c.rope_base = float(*freq_base);
  c.rope_base_swa = 0.0f;
  c.shared_kv_layers = 0;
  c.sliding_window_pattern.clear();
  c.n_expert = uint32_t(*expert_count);
  c.n_expert_used = uint32_t(*expert_used);
  // The reference falls back to feed_forward_length / expert_used_count when
  // the expert width is absent and to feed_forward_length for the shared
  // expert. Both keys may be written as per-layer arrays; a non-uniform array
  // was refused above.
  if (expert_ff) {
    c.expert_ff = uint32_t(*expert_ff);
  } else if (ff_len && *ff_len % c.n_expert_used == 0) {
    c.expert_ff = uint32_t(*ff_len / c.n_expert_used);
  }
  if (c.expert_ff == 0) return invalid("qwen35moe.expert_feed_forward_length");
  if (shared_ff) {
    c.shared_expert_ff = uint32_t(*shared_ff);
  } else if (ff_len) {
    c.shared_expert_ff = uint32_t(*ff_len);
  }
  if (c.shared_expert_ff == 0)
    return invalid("qwen35moe.expert_shared_feed_forward_length");
  // The file may omit feed_forward_length (there is no dense FFN). The
  // config's field then carries the expert width, so the generic readers see
  // the block's feed-forward size and valid() holds.
  c.feed_forward_length = ff_len ? uint32_t(*ff_len) : c.expert_ff;
  c.expert_weights_scale = weights_scale ? float(*weights_scale) : 1.0f;
  c.full_attention_interval = interval ? uint32_t(*interval) : 4;
  if (c.full_attention_interval == 0)
    return invalid("qwen35moe.full_attention_interval");
  c.ssm_conv_kernel = uint32_t(*ssm_conv);
  c.ssm_inner_size = uint32_t(*ssm_inner);
  c.ssm_state_size = uint32_t(*ssm_state);
  c.ssm_time_step_rank = uint32_t(*ssm_dt);
  c.ssm_group_count = uint32_t(*ssm_group);
  c.n_layer_nextn = nextn ? uint32_t(*nextn) : 0;
  // A file that carries multi-token-prediction blocks declares them in
  // nextn_predict_layers: the last k blocks belong to the speculative head,
  // not to the trunk. The main model is the trunk; those blocks' tensors
  // (nextn.* and the head's own attention/experts) are left unbound.
  if (c.n_layer_nextn >= c.block_count)
    return invalid("qwen35moe.nextn_predict_layers (covers the whole trunk)");
  c.block_count -= c.n_layer_nextn;
  c.rope_dimension_count = rope_dims ? uint32_t(*rope_dims) : 0;
  if (c.rope_dimension_count > c.head_dim_global)
    return invalid("qwen35moe.rope.dimension_count (exceeds head_dim)");
  c.rope_dimension_sections.reserve(rope_sections->size());
  for (uint64_t v : *rope_sections) c.rope_dimension_sections.push_back(uint32_t(v));

  // Layer kinds: the explicit per-layer array wins; the file may cover the
  // MTP blocks too, so only the trunk's entries are kept. Without the array,
  // every layer whose 1-based index is not a multiple of the interval is a
  // recurrent (gated DeltaNet) layer.
  if (recurrent) {
    if (recurrent->size() < c.block_count)
      return invalid("qwen35moe.attention.recurrent_layers (too short)");
    c.recurrent_layers.assign(recurrent->begin(),
                              recurrent->begin() + c.block_count);
  } else {
    c.recurrent_layers.resize(c.block_count);
    for (uint32_t i = 0; i < c.block_count; ++i)
      c.recurrent_layers[i] = (i + 1) % c.full_attention_interval != 0;
  }

  // token_embd.shape[1] defines vocab_size (shared with the weight binder).
  {
    const gguf::TensorInfo* t = r.tensor("token_embd.weight");
    if (!t || t->dims.size() != 2 || t->dims[0] != c.embedding_length ||
        t->dims[1] == 0) {
      return std::unexpected(std::string("token_embd.weight absent or malformed"));
    }
    c.vocab_size = t->dims[1];
  }

  if (!c.valid()) return std::unexpected(std::string("inconsistent config"));
  return c;
}

std::expected<ModelArch, std::string> detect_arch(const gguf::Reader& r) {
  const auto a = read_string(r, "general.architecture");
  if (!a) return std::unexpected(std::string("missing general.architecture"));
  if (*a == "gemma4") return ModelArch::Gemma4;
  if (*a == "qwen2") return ModelArch::Qwen2;
  if (*a == "qwen35moe") return ModelArch::Qwen35Moe;
  return std::unexpected("unknown architecture: " + std::string(*a));
}

std::expected<Gemma4Config, std::string> load_config(const gguf::Reader& r) {
  const auto arch = detect_arch(r);
  if (!arch) return std::unexpected(arch.error());
  switch (*arch) {
    case ModelArch::Gemma4: return load_gemma4_config(r);
    case ModelArch::Qwen2:  return load_qwen2_config(r);
    case ModelArch::Qwen35Moe: return load_qwen35moe_config(r);
  }
  return std::unexpected(std::string("unknown architecture"));
}

// Qwen2 weight binding (Qwen2.5-7B-Instruct, Q4_K_M). A standard decoder
// block: no per-layer embedding gate, no Q/K per-head norm, no post-attention /
// post-ffw norm, no whole-block output scale — and two things Gemma 4 does not
// have: a bias on each of the Q/K/V projections, and an untied output
// projection ("output.weight", which must be a real tensor here; when a Qwen2
// file omits it llama.cpp falls back to the token embedding, and so does the
// emitter). Q5_0 and Q8_0 are native quant types (they bind through the
// ordinary quantized path), so nothing is converted at load time.
std::expected<Gemma4Model, std::string> load_qwen2(const gguf::Reader& r) {
  Gemma4Model m;

  auto cfg = load_qwen2_config(r);
  if (!cfg) return std::unexpected(std::move(cfg.error()));
  m.config = std::move(*cfg);

  Gemma4Config& c = m.config;
  Gemma4Weights& w = m.weights;
  const uint32_t d = c.embedding_length;
  const uint32_t ff = c.feed_forward_length;
  const uint64_t vocab = c.vocab_size;

  std::string errs;
  std::map<uint32_t,uint64_t> converted;
  if (auto q = bind_quant(r, "token_embd.weight", {}, {d, vocab}, errs, m.owned_quant_buffers, converted))
    w.token_embd = *q;
  if (auto q = bind_quant(r, "output.weight", {}, {d, vocab}, errs, m.owned_quant_buffers, converted))
    w.output = *q;
  if (auto s = bind_f32(r, "output_norm.weight", d, errs)) w.output_norm = *s;

  w.blocks.clear();
  w.blocks.reserve(c.block_count);
  for (uint32_t l = 0; l < c.block_count; ++l) {
    const LayerConfig lc = c.layer_config(l);
    const uint64_t q_width = uint64_t(lc.n_heads_q) * lc.head_dim;
    const uint64_t kv_width = uint64_t(lc.n_heads_kv) * lc.head_dim;

    const std::string p = "blk." + std::to_string(l) + ".";
    BlockWeights bw;
    if (auto s = bind_f32(r, p + "attn_norm.weight", d, errs)) bw.attn_norm = *s;
    if (auto s = bind_f32(r, p + "ffn_norm.weight", d, errs)) bw.ffn_norm = *s;
    if (auto q = bind_quant(r, p + "attn_q.weight", {}, {d, q_width}, errs, m.owned_quant_buffers, converted))
      bw.attn_q = *q;
    if (auto q = bind_quant(r, p + "attn_k.weight", {}, {d, kv_width}, errs, m.owned_quant_buffers, converted))
      bw.attn_k = *q;
    if (auto q = bind_quant(r, p + "attn_v.weight", {}, {d, kv_width}, errs, m.owned_quant_buffers, converted))
      bw.attn_v = *q;
    if (auto q = bind_quant(r, p + "attn_output.weight", {}, {q_width, d}, errs, m.owned_quant_buffers, converted))
      bw.attn_output = *q;
    if (auto s = bind_f32(r, p + "attn_q.bias", q_width, errs)) bw.attn_q_bias = *s;
    if (auto s = bind_f32(r, p + "attn_k.bias", kv_width, errs)) bw.attn_k_bias = *s;
    if (auto s = bind_f32(r, p + "attn_v.bias", kv_width, errs)) bw.attn_v_bias = *s;
    // The output projection's bias is optional in the format (llama.cpp reads
    // it as TENSOR_NOT_REQUIRED) and absent from the Qwen2.5 files; bind it only
    // when present so an absent tensor is not reported as a binding error.
    if (r.tensor(p + "attn_output.bias") != nullptr)
      if (auto s = bind_f32(r, p + "attn_output.bias", d, errs)) bw.attn_output_bias = *s;
    if (auto q = bind_quant(r, p + "ffn_gate.weight", {}, {d, ff}, errs, m.owned_quant_buffers, converted))
      bw.ffn_gate = *q;
    if (auto q = bind_quant(r, p + "ffn_up.weight", {}, {d, ff}, errs, m.owned_quant_buffers, converted))
      bw.ffn_up = *q;
    if (auto q = bind_quant(r, p + "ffn_down.weight", {}, {ff, d}, errs, m.owned_quant_buffers, converted))
      bw.ffn_down = *q;
    w.blocks.push_back(bw);
  }

  if (!errs.empty()) return std::unexpected(std::move(errs));

  // Generic long-context RoPE scaling (qwen2.rope.scaling.*). Every Qwen2 layer
  // applies the same table; the kind==Global gate at the rope call sites is the
  // Gemma partial-RoPE rule, which is satisfied here because all layers are
  // global.
  if (auto err = apply_rope_scaling(r, "qwen2", c.head_dim_global, c.rope_base,
                                    c.context_length, w))
    return std::unexpected(std::move(*err));

  m.plan.clear();
  m.plan.reserve(c.block_count);
  for (uint32_t l = 0; l < c.block_count; ++l) m.plan.push_back(c.layer_config(l));

  return m;
}

// Qwen3.6-MoE weight binding. The trunk (embedding, output, output norm, a pre-
// attention norm and a pre-FFN norm per block) and the full-attention layers
// (Q/K/V/output projections, Q/K per-head norms) bind into the existing
// container. The MoE part binds one router per block (quantized or f32,
// whichever the file stores), per-expert views over the stacked gate/up/down
// tensors, and the shared expert's three projections plus its gate vector.
// Recurrent (gated DeltaNet) layers bind their QKV+gate, beta, alpha and out
// projections, the convolution kernel, dt and A, and the per-key-head norm; the
// recurrence itself is kernel/state work downstream. The vision tower and the
// trailing multi-token-prediction blocks are not read.
std::expected<Gemma4Model, std::string> load_qwen35moe(const gguf::Reader& r) {
  Gemma4Model m;

  auto cfg = load_qwen35moe_config(r);
  if (!cfg) return std::unexpected(std::move(cfg.error()));
  m.config = std::move(*cfg);

  Gemma4Config& c = m.config;
  Gemma4Weights& w = m.weights;
  const uint32_t d = c.embedding_length;
  const uint64_t vocab = c.vocab_size;

  std::string errs;
  std::map<uint32_t,uint64_t> converted;
  if (auto q = bind_quant(r, "token_embd.weight", {}, {d, vocab}, errs,
                          m.owned_quant_buffers, converted))
    w.token_embd = *q;
  // The output projection is optional; without it the model ties the output to
  // the token embedding (the emitter reuses TokenEmbd).
  if (r.tensor("output.weight") != nullptr)
    if (auto q = bind_quant(r, "output.weight", {}, {d, vocab}, errs,
                            m.owned_quant_buffers, converted))
      w.output = *q;
  if (auto s = bind_f32(r, "output_norm.weight", d, errs))
    w.output_norm = *s;

  // Gated DeltaNet geometry (validated for the recurrent layers).
  const uint64_t key_dim = uint64_t(c.ssm_state_size) * c.ssm_group_count;
  const uint64_t value_dim = uint64_t(c.ssm_state_size) * c.ssm_time_step_rank;
  const uint64_t conv_dim = key_dim * 2 + value_dim;
  // The three conditions the recurrence and its CUDA layout require; a model
  // that violates one fails to load, naming the condition.
  if (c.ssm_group_count == 0 ||
      c.ssm_time_step_rank % c.ssm_group_count != 0)
    errs += "gated delta net: ssm_time_step_rank (" +
            std::to_string(c.ssm_time_step_rank) +
            ") is not a multiple of ssm_group_count (" +
            std::to_string(c.ssm_group_count) + ")\n";
  if (value_dim != uint64_t(c.ssm_inner_size))
    errs += "gated delta net: ssm_state_size * ssm_time_step_rank (" +
            std::to_string(value_dim) + ") does not equal ssm_inner_size (" +
            std::to_string(c.ssm_inner_size) + ")\n";
  if (c.ssm_state_size % 32 != 0)
    errs += "gated delta net: ssm_state_size (" +
            std::to_string(c.ssm_state_size) + ") is not a multiple of 32\n";

  w.blocks.clear();
  w.blocks.reserve(c.block_count);
  for (uint32_t l = 0; l < c.block_count; ++l) {
    const LayerConfig lc = c.layer_config(l);
    const uint64_t head_width = uint64_t(lc.n_heads_q) * lc.head_dim;
    const uint64_t q_width = 2 * head_width;
    const uint64_t kv_width = uint64_t(lc.n_heads_kv) * lc.head_dim;

    const std::string p = "blk." + std::to_string(l) + ".";
    BlockWeights bw;
    if (auto s = bind_f32(r, p + "attn_norm.weight", d, errs)) bw.attn_norm = *s;
    // The block's pre-FFN norm is post_attention_norm in this architecture.
    if (auto s = bind_f32(r, p + "post_attention_norm.weight", d, errs))
      bw.ffn_norm = *s;

    const bool recurrent = l < c.recurrent_layers.size() &&
                           c.recurrent_layers[l];
    if (recurrent) {
      // Gated DeltaNet: five projections stored as in the file, the
      // convolution kernel, the dt bias and A, and the per-key-head norm.
      bind_quant_or_f32(r, p + "attn_qkv.weight",
                        {d, key_dim * 2 + value_dim}, bw.gdn_qkv,
                        bw.gdn_qkv_f32, errs, m.owned_quant_buffers, converted);
      bind_quant_or_f32(r, p + "attn_gate.weight", {d, value_dim},
                        bw.gdn_gate, bw.gdn_gate_f32, errs,
                        m.owned_quant_buffers, converted);
      bind_quant_or_f32(r, p + "ssm_beta.weight", {d, c.ssm_time_step_rank},
                        bw.gdn_beta, bw.gdn_beta_f32, errs,
                        m.owned_quant_buffers, converted);
      bind_quant_or_f32(r, p + "ssm_alpha.weight", {d, c.ssm_time_step_rank},
                        bw.gdn_alpha, bw.gdn_alpha_f32, errs,
                        m.owned_quant_buffers, converted);
      bind_quant_or_f32(r, p + "ssm_out.weight", {value_dim, d}, bw.gdn_out,
                        bw.gdn_out_f32, errs, m.owned_quant_buffers, converted);
      if (auto s = bind_f32_2d(r, p + "ssm_conv1d.weight",
                               {c.ssm_conv_kernel, conv_dim}, errs))
        bw.gdn_conv = *s;
      if (auto s = bind_f32(r, p + "ssm_dt.bias", c.ssm_time_step_rank, errs))
        bw.gdn_dt = *s;
      if (auto s = bind_f32(r, p + "ssm_a", c.ssm_time_step_rank, errs))
        bw.gdn_a = *s;
      if (auto s = bind_f32(r, p + "ssm_norm.weight", c.ssm_state_size, errs))
        bw.gdn_norm = *s;
    } else {
      // Full attention: the Q projection is twice as wide as the attention
      // head output (its second half per head gates the attention result).
      if (auto q = bind_quant(r, p + "attn_q.weight", {}, {d, q_width}, errs,
                              m.owned_quant_buffers, converted))
        bw.attn_q = *q;
      if (auto q = bind_quant(r, p + "attn_k.weight", {}, {d, kv_width}, errs,
                              m.owned_quant_buffers, converted))
        bw.attn_k = *q;
      if (auto q = bind_quant(r, p + "attn_v.weight", {}, {d, kv_width}, errs,
                              m.owned_quant_buffers, converted))
        bw.attn_v = *q;
      if (auto q = bind_quant(r, p + "attn_output.weight", {},
                              {head_width, d}, errs, m.owned_quant_buffers,
                              converted))
        bw.attn_output = *q;
      if (auto s = bind_f32(r, p + "attn_q_norm.weight", lc.head_dim, errs))
        bw.attn_q_norm = *s;
      if (auto s = bind_f32(r, p + "attn_k_norm.weight", lc.head_dim, errs))
        bw.attn_k_norm = *s;
    }

    // Router: whichever form the file stores.
    const std::string router_name = p + "ffn_gate_inp.weight";
    const gguf::TensorInfo* rt = r.tensor(router_name);
    if (!rt) {
      errs += "missing tensor: ";
      errs += router_name;
      errs += "\n";
    } else if (rt->type == kF32) {
      if (auto s = bind_f32_2d(r, router_name, {d, c.n_expert}, errs))
        bw.expert_router_f32 = *s;
    } else if (auto q = bind_quant(r, router_name, {}, {d, c.n_expert}, errs,
                                   m.owned_quant_buffers, converted)) {
      bw.expert_router = *q;
    }

    // Routed experts: one merged gate/up tensor when the file has it,
    // otherwise two.
    const std::string gate_up_name = p + "ffn_gate_up_exps.weight";
    if (r.tensor(gate_up_name) != nullptr) {
      bind_expert_gate_up(r, gate_up_name, d, c.expert_ff, c.n_expert,
                          bw.expert_gate, bw.expert_up, errs);
    } else {
      bw.expert_gate = bind_experts(r, p + "ffn_gate_exps.weight",
                                    {d, c.expert_ff}, c.n_expert, errs);
      bw.expert_up = bind_experts(r, p + "ffn_up_exps.weight",
                                  {d, c.expert_ff}, c.n_expert, errs);
    }
    bw.expert_down = bind_experts(r, p + "ffn_down_exps.weight",
                                  {c.expert_ff, d}, c.n_expert, errs);

    // Shared expert.
    if (auto q = bind_quant(r, p + "ffn_gate_shexp.weight", {},
                            {d, c.shared_expert_ff}, errs,
                            m.owned_quant_buffers, converted))
      bw.shared_gate = *q;
    if (auto q = bind_quant(r, p + "ffn_up_shexp.weight", {},
                            {d, c.shared_expert_ff}, errs,
                            m.owned_quant_buffers, converted))
      bw.shared_up = *q;
    if (auto q = bind_quant(r, p + "ffn_down_shexp.weight", {},
                            {c.shared_expert_ff, d}, errs,
                            m.owned_quant_buffers, converted))
      bw.shared_down = *q;
    if (auto s = bind_f32(r, p + "ffn_gate_inp_shexp.weight", d, errs))
      bw.shared_gate_in = *s;

    w.blocks.push_back(bw);
  }

  if (!errs.empty()) return std::unexpected(std::move(errs));

  m.plan.clear();
  m.plan.reserve(c.block_count);
  for (uint32_t l = 0; l < c.block_count; ++l) m.plan.push_back(c.layer_config(l));

  return m;
}

std::expected<Gemma4Model, std::string> load_gemma4(const gguf::Reader& r) {
  Gemma4Model m;

  // The architecture decides which weight directory to bind; the entry point
  // stays single so a caller can load either model family without knowing the
  // architecture in advance (detect_arch reads general.architecture).
  const auto arch = detect_arch(r);
  if (!arch) return std::unexpected(arch.error());
  if (*arch == ModelArch::Qwen2) {
    // Host-side only, like the Gemma path: the device and the per-shape
    // numerics resolution are paid by ensure_device_ready() on first device
    // use.
    return load_qwen2(r);
  }
  if (*arch == ModelArch::Qwen35Moe) return load_qwen35moe(r);

  auto cfg = load_gemma4_config(r);
  if (!cfg) return std::unexpected(std::move(cfg.error()));
  m.config = std::move(*cfg);

  // --- config (already loaded above) ---------------------------------------
  Gemma4Config& c = m.config;

  // --- weights -------------------------------------------------------------
  Gemma4Weights& w = m.weights;
  const uint32_t d = c.embedding_length;
  const uint32_t ff = c.feed_forward_length;
  const uint64_t per_layer_dim = uint64_t(c.block_count) * c.per_layer_input;

  // vocab_size was set by load_gemma4_config (token_embd dims[1]); the weight
  // binders below re-validate that same tensor's type/shape against the config.
  std::string errs;
  std::map<uint32_t,uint64_t> converted;
  if (auto q = bind_quant(r, "token_embd.weight", {kQ3K}, {d, c.vocab_size}, errs, m.owned_quant_buffers, converted))
    w.token_embd = *q;
  if (auto q = bind_quant(r, "per_layer_token_embd.weight", {kQ4K},
                          {per_layer_dim, c.vocab_size}, errs, m.owned_quant_buffers, converted))
    w.per_layer_token_embd = *q;
  if (auto s = bind_f32(r, "per_layer_proj_norm.weight", c.per_layer_input, errs))
    w.per_layer_proj_norm = *s;
  if (auto s = bind_bf16(r, "per_layer_model_proj.weight", {d, per_layer_dim},
                         d * per_layer_dim, errs))
    w.per_layer_model_proj = *s;
  if (auto s = bind_f32(r, "output_norm.weight", d, errs)) w.output_norm = *s;
  // rope_freqs is a 256-float precomputed table (head_dim_swa entries); not
  // required by the reference execution, which recomputes freqs from base.
  if (auto s = bind_f32(r, "rope_freqs.weight", c.head_dim_swa, errs))
    w.rope_freqs = *s;

  // Per-block tensors, one BlockWeights per layer.
  w.blocks.clear();
  w.blocks.reserve(c.block_count);
  for (uint32_t l = 0; l < c.block_count; ++l) {
    const LayerConfig lc = c.layer_config(l);
    const uint64_t q_width = uint64_t(lc.n_heads_q) * lc.head_dim;
    const uint64_t kv_width = uint64_t(lc.n_heads_kv) * lc.head_dim;

    const std::string p = "blk." + std::to_string(l) + ".";
    BlockWeights bw;
    if (auto s = bind_f32(r, p + "attn_norm.weight", d, errs)) bw.attn_norm = *s;
    if (auto s = bind_f32(r, p + "ffn_norm.weight", d, errs)) bw.ffn_norm = *s;
    // q/k per-head norm: dim == head_dim (256 SWA / 512 global). Bound for all
    // 42 blocks; the arithmetic (RMSNorm vs. scalar scale) is unresolved and
    // not consumed by run_block yet.
    if (auto s = bind_f32(r, p + "attn_q_norm.weight", lc.head_dim, errs))
      bw.attn_q_norm = *s;
    if (auto s = bind_f32(r, p + "attn_k_norm.weight", lc.head_dim, errs))
      bw.attn_k_norm = *s;
    if (auto s = bind_f32(r, p + "post_attention_norm.weight", d, errs))
      bw.post_attention_norm = *s;
    if (auto s = bind_f32(r, p + "post_ffw_norm.weight", d, errs))
      bw.post_ffw_norm = *s;
    if (auto s = bind_f32(r, p + "post_norm.weight", d, errs))
      bw.post_norm = *s;
    if (auto q = bind_quant(r, p + "attn_q.weight", {kQ3K}, {d, q_width}, errs, m.owned_quant_buffers, converted))
      bw.attn_q = *q;
    if (auto q = bind_quant(r, p + "attn_k.weight", {kQ3K}, {d, kv_width}, errs, m.owned_quant_buffers, converted))
      bw.attn_k = *q;
    // attn_v / ffn_down are mixed-precision in the Unsloth files (q6_K in 21
    // of the 42 blocks, q5_K in the rest); both are native quant types, and the
    // binder accepts any native type, so the file's mix is kept as written.
    if (auto q = bind_quant(r, p + "attn_v.weight", {kQ5K, kQ4K}, {d, kv_width}, errs, m.owned_quant_buffers, converted))
      bw.attn_v = *q;
    if (auto q = bind_quant(r, p + "attn_output.weight", {kQ4K}, {q_width, d}, errs, m.owned_quant_buffers, converted))
      bw.attn_output = *q;
    if (auto q = bind_quant(r, p + "ffn_gate.weight", {kQ3K}, {d, ff}, errs, m.owned_quant_buffers, converted))
      bw.ffn_gate = *q;
    if (auto q = bind_quant(r, p + "ffn_up.weight", {kQ3K}, {d, ff}, errs, m.owned_quant_buffers, converted))
      bw.ffn_up = *q;
    if (auto q = bind_quant(r, p + "ffn_down.weight", {kQ5K, kQ4K}, {ff, d}, errs, m.owned_quant_buffers, converted))
      bw.ffn_down = *q;
    // Per-layer embedding gating (f32): inp_gate [d, per], proj [per, d]. Bound
    // and shape-validated; the gating arithmetic is unresolved (not consumed).
    const uint64_t per = c.per_layer_input;
    if (auto s = bind_f32_2d(r, p + "inp_gate.weight", {d, per}, errs))
      bw.inp_gate = *s;
    if (auto s = bind_f32_2d(r, p + "proj.weight", {per, d}, errs))
      bw.proj = *s;
    if (auto s = bind_f32(r, p + "layer_output_scale.weight", 1, errs))
      bw.layer_output_scale = *s;
    w.blocks.push_back(bw);
  }

  if (!errs.empty()) return std::unexpected(std::move(errs));

  // Generic RoPE scaling, composed with the precomputed partial-RoPE divisors
  // bound above. The table covers the global layers' head dim (the widest one);
  // sliding-window layers take the empty divisor span at the call sites.
  if (auto err = apply_rope_scaling(r, "gemma4",
                                    std::max(c.head_dim_global, c.head_dim_swa),
                                    c.rope_base, c.context_length, w))
    return std::unexpected(std::move(*err));

  if (!converted.empty()) {
    uint64_t total=0; for (const auto& [id,count] : converted) total+=count;
    std::fprintf(stderr,"sonicboom: converted %llu tensors to Q5_K at load time (types: ",
                 static_cast<unsigned long long>(total));
    bool first=true;
    for (const auto& [id,count] : converted) {
      std::fprintf(stderr,"%s%s x%llu",first?"":", ",foreign_type_name(id),
                   static_cast<unsigned long long>(count)); first=false;
    }
    std::fprintf(stderr,"); native kernels pending\n");
  }

  // --- execution plan ------------------------------------------------------
  m.plan.clear();
  m.plan.reserve(c.block_count);
  for (uint32_t l = 0; l < c.block_count; ++l) m.plan.push_back(c.layer_config(l));

  // Host-side only. The device context and the model's numerics-affecting
  // choices are resolved by ensure_device_ready() on first device use, so a
  // caller can load and fork without leaving a CUDA context in the parent.
  return m;
}

} // namespace sonicboom::model
