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

#include <sonicboom/vocab/vocab.h>

#include <sonicboom/gguf/reader.h>

#include "ggml_derived/bpe_tokenizer.h"

namespace sonicboom::vocab {

namespace {

using gguf::Value;
using gguf::ValueType;

// Widen an integer metadata scalar (signed or unsigned) to int64_t.
int64_t scalar_int(const Value& v) {
  switch (v.type) {
    case ValueType::UInt8:
    case ValueType::UInt16:
    case ValueType::UInt32:
    case ValueType::UInt64:
      return static_cast<int64_t>(v.u);
    case ValueType::Int8:
    case ValueType::Int16:
    case ValueType::Int32:
    case ValueType::Int64:
      return v.i;
    default:
      return 0;
  }
}

int64_t opt_int(const gguf::Reader& reader, std::string_view key, int64_t dflt) {
  const Value* v = reader.find(key);
  return v ? scalar_int(*v) : dflt;
}

bool opt_bool(const gguf::Reader& reader, std::string_view key, bool dflt) {
  const Value* v = reader.find(key);
  return v ? v->b : dflt;
}

}  // namespace

std::expected<Vocab, std::string> load_vocab(const gguf::Reader& reader) {
  const Value* model = reader.find("tokenizer.ggml.model");
  if (!model || model->s != "gemma4") {
    return std::unexpected("unsupported tokenizer model (expected 'gemma4')");
  }

  const Value* tokens = reader.find("tokenizer.ggml.tokens");
  const Value* merges = reader.find("tokenizer.ggml.merges");
  if (!tokens || tokens->type != ValueType::Array ||
      tokens->elem_type != ValueType::String) {
    return std::unexpected("tokenizer.ggml.tokens missing or not a string array");
  }
  if (!merges || merges->type != ValueType::Array ||
      merges->elem_type != ValueType::String) {
    return std::unexpected("tokenizer.ggml.merges missing or not a string array");
  }

  Vocab v;
  const size_t n = tokens->elems.size();
  v.tokens.resize(n);
  v.type.assign(n, TokenType::Normal);
  for (size_t i = 0; i < n; ++i) {
    v.tokens[i] = tokens->elems[i].s;
    v.token_to_id.emplace(v.tokens[i], static_cast<int32_t>(i));
  }

  // token types (optional; default normal). Mirrors GGUF tokenizer.ggml.token_type.
  if (const Value* tt = reader.find("tokenizer.ggml.token_type");
      tt && tt->type == ValueType::Array) {
    for (size_t i = 0; i < n && i < tt->elems.size(); ++i) {
      const int64_t ty = scalar_int(tt->elems[i]);
      if (ty >= 1 && ty <= 6) v.type[i] = static_cast<TokenType>(ty);
    }
  }

  // BPE merges -> (left, right) -> rank. Each merge string is "<left> <right>"
  // (split at the first space at index >= 1, like llama.cpp).
  v.bpe_ranks.reserve(merges->elems.size());
  for (size_t i = 0; i < merges->elems.size(); ++i) {
    const std::string& word = merges->elems[i].s;
    const size_t pos = word.find(' ', 1);
    if (pos == std::string::npos) continue;
    v.bpe_ranks.emplace(
        std::make_pair(word.substr(0, pos), word.substr(pos + 1)),
        static_cast<int32_t>(i));
  }

  v.bos = static_cast<int32_t>(opt_int(reader, "tokenizer.ggml.bos_token_id", -1));
  v.eos = static_cast<int32_t>(opt_int(reader, "tokenizer.ggml.eos_token_id", -1));
  v.unk = static_cast<int32_t>(opt_int(reader, "tokenizer.ggml.unknown_token_id", -1));

  v.add_bos = opt_bool(reader, "tokenizer.ggml.add_bos_token", false);
  v.add_eos = opt_bool(reader, "tokenizer.ggml.add_eos_token", false);
  v.add_space_prefix = opt_bool(reader, "tokenizer.ggml.add_space_prefix", false);

  // Gemma-4 fixed configuration (llama.cpp): spaces are escaped to ▁, spaces
  // are cleaned up on detokenize, and add_bos is force-enabled.
  v.escape_whitespaces = true;
  v.clean_spaces = true;
  if (!v.add_bos) v.add_bos = true;

  ggml_bpe::apply_token_type_overrides(v);
  return v;
}

std::vector<int32_t> tokenize(const Vocab& v, std::string_view text,
                              bool parse_special, bool add_special) {
  return ggml_bpe::tokenize(v, text, parse_special, add_special);
}

std::string detokenize(const Vocab& v, const int32_t* tokens, size_t n) {
  return ggml_bpe::detokenize(v, tokens, n);
}

}  // namespace sonicboom::vocab
