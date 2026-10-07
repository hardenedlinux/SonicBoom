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

#include <cstdint>
#include <string>
#include <string_view>

namespace sonicboom::model {

namespace {

// ggml type ids the target model uses (re-declared from the format; see the
// GGUF reader's type table and design/gguf-reader.md).
constexpr uint32_t kF32 = 0;
constexpr uint32_t kBF16 = 30;
constexpr uint32_t kQ3K = 11;
constexpr uint32_t kQ4K = 12;
constexpr uint32_t kQ5K = 13;

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
    std::initializer_list<uint64_t> dims, std::string& errs) {
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
  if (!type_ok || !dims_match(t->dims, dims)) {
    errs += "tensor type/shape mismatch: ";
    errs += name;
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
  if (!qt || !qt->valid()) {
    errs += "quantized tensor invalid: ";
    errs += name;
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

} // namespace

std::expected<Gemma4Model, std::string> load_gemma4(const gguf::Reader& r) {
  Gemma4Model m;

  // --- config (metadata) ---------------------------------------------------
  Gemma4Config& c = m.config;
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

  std::string errs;
  auto need = [&](bool ok, const char* what) {
    if (!ok) {
      errs += "missing/invalid metadata: ";
      errs += what;
      errs += "\n";
    }
  };
  need(block_count && *block_count > 0, "gemma4.block_count");
  need(embd_len && *embd_len > 0, "gemma4.embedding_length");
  need(ff_len && *ff_len > 0, "gemma4.feed_forward_length");
  need(head_count && *head_count > 0, "gemma4.attention.head_count");
  need(head_count_kv && *head_count_kv > 0, "gemma4.attention.head_count_kv");
  need(key_len && *key_len > 0, "gemma4.attention.key_length");
  need(key_len_swa && *key_len_swa > 0, "gemma4.attention.key_length_swa");
  need(per_layer_input && *per_layer_input > 0,
       "gemma4.embedding_length_per_layer_input");
  need(context_len && *context_len > 0, "gemma4.context_length");
  need(sliding_window && *sliding_window > 0, "gemma4.attention.sliding_window");
  need(pattern.has_value(), "gemma4.attention.sliding_window_pattern");
  need(rms_eps && *rms_eps >= 0.0, "gemma4.attention.layer_norm_rms_epsilon");
  need(softcap && *softcap > 0.0, "gemma4.final_logit_softcapping");
  need(freq_base && *freq_base > 0.0, "gemma4.rope.freq_base");
  need(freq_base_swa && *freq_base_swa > 0.0, "gemma4.rope.freq_base_swa");
  need(shared_kv.has_value(), "gemma4.attention.shared_kv_layers");

  if (!errs.empty()) return std::unexpected(std::move(errs));

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
  c.sliding_window_pattern = std::move(*pattern);

  if (!c.valid()) return std::unexpected(std::string("inconsistent config"));

  // --- weights -------------------------------------------------------------
  Gemma4Weights& w = m.weights;
  const uint32_t d = c.embedding_length;
  const uint32_t ff = c.feed_forward_length;
  const uint64_t per_layer_dim = uint64_t(c.block_count) * c.per_layer_input;

  // token_embd.shape[1] also defines vocab_size.
  {
    const gguf::TensorInfo* t = r.tensor("token_embd.weight");
    if (!t || t->dims.size() != 2 || t->dims[0] != d || t->dims[1] == 0) {
      return std::unexpected(std::string("token_embd.weight absent or malformed"));
    }
    c.vocab_size = t->dims[1];
  }
  if (auto q = bind_quant(r, "token_embd.weight", {kQ3K}, {d, c.vocab_size}, errs))
    w.token_embd = *q;
  if (auto q = bind_quant(r, "per_layer_token_embd.weight", {kQ4K},
                          {per_layer_dim, c.vocab_size}, errs))
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
    if (auto q = bind_quant(r, p + "attn_q.weight", {kQ3K}, {d, q_width}, errs))
      bw.attn_q = *q;
    if (auto q = bind_quant(r, p + "attn_k.weight", {kQ3K}, {d, kv_width}, errs))
      bw.attn_k = *q;
    // attn_v / ffn_down are mixed-precision (Unsloth Q3_K_M): q5_K in the first
    // two blocks, q4_K elsewhere — both are supported, so accept either.
    if (auto q = bind_quant(r, p + "attn_v.weight", {kQ5K, kQ4K}, {d, kv_width}, errs))
      bw.attn_v = *q;
    if (auto q = bind_quant(r, p + "attn_output.weight", {kQ4K}, {q_width, d}, errs))
      bw.attn_output = *q;
    if (auto q = bind_quant(r, p + "ffn_gate.weight", {kQ3K}, {d, ff}, errs))
      bw.ffn_gate = *q;
    if (auto q = bind_quant(r, p + "ffn_up.weight", {kQ3K}, {d, ff}, errs))
      bw.ffn_up = *q;
    if (auto q = bind_quant(r, p + "ffn_down.weight", {kQ5K, kQ4K}, {ff, d}, errs))
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

  // --- execution plan ------------------------------------------------------
  m.plan.clear();
  m.plan.reserve(c.block_count);
  for (uint32_t l = 0; l < c.block_count; ++l) m.plan.push_back(c.layer_config(l));

  return m;
}

} // namespace sonicboom::model
