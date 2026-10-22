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

// Hugging Face tokenizer.json -> Vocab (byte-level BPE, qwen2 pre-tokenizer).
//
// The file is the serialized form of the "tokenizers" library pipeline:
// normalizer, pre_tokenizer, model (BPE vocab + merges) and decoder. Only the
// pieces the SonicBoom tokenizer actually reproduces are read, and each one is
// checked by name: the vocab and merge tables are taken verbatim, the model
// must be BPE, and the pre-tokenizer must be the qwen2 regex. Anything else is
// an error rather than a silent approximation.

#include <sonicboom/vocab/vocab.h>

#include "../safetensors/json.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace sonicboom::vocab {
namespace {

using safetensors::detail::JsonValue;

// The pre-tokenizer regex the Gpt2Qwen2 splitter implements (llama.cpp's
// unicode_regex_split_custom_qwen2). The regex text is compared literally.
constexpr std::string_view kQwen2PretokenizeRegex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

std::expected<std::string, std::string> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in.good() && !in.eof()) return std::unexpected("cannot read " + path.string());
  return ss.str();
}

std::optional<JsonValue> parse_file(const std::filesystem::path& path,
                                    std::string* error) {
  auto text = read_file(path);
  if (!text) {
    *error = text.error();
    return std::nullopt;
  }
  return safetensors::detail::parse_json(*text, error);
}

// The regex string of the first Split pre-tokenizer, or "" when absent.
std::string split_regex(const JsonValue& pre) {
  if (const JsonValue* parts = pre.find("pretokenizers")) {
    for (const JsonValue& part : parts->array) {
      if (part.find("type") && part.find("type")->string == "Split") {
        if (const JsonValue* pat = part.find("pattern")) {
          if (const JsonValue* re = pat->find("Regex")) return re->string;
        }
      }
    }
  }
  return {};
}

int32_t id_of(const Vocab& v, const std::string& piece) {
  auto it = v.token_to_id.find(piece);
  return it == v.token_to_id.end() ? -1 : it->second;
}

}  // namespace

std::expected<Vocab, std::string> load_tokenizer_json(
    const std::filesystem::path& path) {
  std::string error;
  auto root = parse_file(path, &error);
  if (!root) return std::unexpected("tokenizer.json: " + error);

  const JsonValue* model = root->find("model");
  if (!model) return std::unexpected("tokenizer.json: no model");
  const JsonValue* type = model->find("type");
  if (!type || type->string != "BPE") {
    return std::unexpected("unsupported tokenizer model '" +
                           (type ? type->string : std::string("(missing)")) +
                           "' (expected 'BPE')");
  }

  const JsonValue* pre = root->find("pre_tokenizer");
  if (!pre) return std::unexpected("tokenizer.json: no pre_tokenizer");
  const std::string regex = split_regex(*pre);
  if (regex.empty()) {
    return std::unexpected("tokenizer.json: pre_tokenizer has no Split regex");
  }
  if (regex != kQwen2PretokenizeRegex) {
    return std::unexpected("unsupported pre-tokenizer regex '" + regex +
                           "' (expected the qwen2 pattern)");
  }

  Vocab v;
  v.kind = TokenizerKind::Gpt2Qwen2;

  const JsonValue* vocab = model->find("vocab");
  if (!vocab || vocab->type != JsonValue::Type::Object) {
    return std::unexpected("tokenizer.json: model.vocab missing");
  }
  int64_t max_id = -1;
  for (const auto& [piece, value] : vocab->object) {
    if (!value.is_integer || value.integer < 0) {
      return std::unexpected("tokenizer.json: non-integer id for token '" + piece + "'");
    }
    max_id = std::max(max_id, value.integer);
  }
  if (max_id < 0) return std::unexpected("tokenizer.json: empty vocabulary");
  v.tokens.assign(static_cast<size_t>(max_id) + 1, std::string());
  v.type.assign(static_cast<size_t>(max_id) + 1, TokenType::Normal);
  for (const auto& [piece, value] : vocab->object) {
    const auto id = static_cast<size_t>(value.integer);
    v.tokens[id] = piece;
    v.token_to_id.emplace(piece, static_cast<int32_t>(id));
  }

  const JsonValue* merges = model->find("merges");
  if (!merges || merges->type != JsonValue::Type::Array) {
    return std::unexpected("tokenizer.json: model.merges missing");
  }
  v.bpe_ranks.reserve(merges->array.size());
  for (size_t i = 0; i < merges->array.size(); ++i) {
    const JsonValue& m = merges->array[i];
    std::string left, right;
    if (m.type == JsonValue::Type::Array && m.array.size() == 2) {
      left = m.array[0].string;
      right = m.array[1].string;
    } else if (m.type == JsonValue::Type::String) {
      const size_t pos = m.string.find(' ', 1);
      if (pos == std::string::npos) continue;
      left = m.string.substr(0, pos);
      right = m.string.substr(pos + 1);
    } else {
      continue;
    }
    v.bpe_ranks.emplace(std::make_pair(left, right), static_cast<int32_t>(i));
  }

  // Added tokens: ids at or beyond the base vocab, plus any base-vocab special.
  // Every added token is matched as one piece by parse_special (the HF
  // AddedVocabulary splits on all of them); `special` only decides whether it is
  // a Control token that detokenize strips, so `<think>` (special=false, but
  // still one id) is UserDefined and `<|im_end|>` is Control.
  if (const JsonValue* added = root->find("added_tokens")) {
    for (const JsonValue& tok : added->array) {
      const JsonValue* id = tok.find("id");
      const JsonValue* content = tok.find("content");
      if (!id || !content || !id->is_integer) continue;
      const auto index = static_cast<size_t>(id->integer);
      if (index >= v.tokens.size()) {
        v.tokens.resize(index + 1);
        v.type.resize(index + 1, TokenType::Normal);
      }
      if (v.tokens[index].empty()) v.tokens[index] = content->string;
      const JsonValue* special = tok.find("special");
      v.type[index] = special && special->boolean ? TokenType::Control
                                                  : TokenType::UserDefined;
      v.token_to_id.emplace(content->string, static_cast<int32_t>(index));
    }
  }

  // Framework token ids live in the sibling tokenizer_config.json; the
  // tokenizer pipeline itself has no notion of bos/eos.
  const std::filesystem::path config_path =
      path.parent_path() / "tokenizer_config.json";
  if (std::filesystem::exists(config_path)) {
    auto config = parse_file(config_path, &error);
    if (!config) return std::unexpected("tokenizer_config.json: " + error);
    auto lookup = [&](const char* key) -> int32_t {
      const JsonValue* entry = config->find(key);
      if (!entry || entry->type != JsonValue::Type::String) return -1;
      return id_of(v, entry->string);
    };
    v.bos = lookup("bos_token");
    v.eos = lookup("eos_token");
    v.unk = lookup("unk_token");
    const JsonValue* add_bos = config->find("add_bos_token");
    if (add_bos && add_bos->type == JsonValue::Type::Bool) {
      v.add_bos = add_bos->boolean;
    }
    const JsonValue* add_eos = config->find("add_eos_token");
    if (add_eos && add_eos->type == JsonValue::Type::Bool) {
      v.add_eos = add_eos->boolean;
    }
  }

  // Byte-level BPE (llama.cpp's gpt2/qwen2 configuration): the piece text is
  // already byte-encoded, and the qwen2 pre-tokenizer turns off the
  // detokenize-time space clean-up.
  v.escape_whitespaces = false;
  v.clean_spaces = false;
  v.add_space_prefix = false;
  return v;
}

}  // namespace sonicboom::vocab
