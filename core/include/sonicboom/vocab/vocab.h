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

// GGUF vocabulary (tokenizer) for the LLM inference surface.
//
// `Vocab` is the Layer 2 tokenizer representation: a token -> piece table, the
// BPE merge ranks, and the tokenizer flags needed to reproduce llama.cpp's
// tokenize/detokenize. It owns no weights, no device state, and exposes no
// native-torch / ATen / c10 types.
//
// Two vocabulary kinds share it, selected from the GGUF's
// tokenizer.ggml.model / tokenizer.ggml.pre:
//
//   Gemma4     — SPM-style BPE over ▁-escaped text (tokenizer.ggml.model=gemma4)
//   Gpt2Qwen2  — byte-level BPE with the qwen2 pre-tokenizer regex
//                (tokenizer.ggml.model=gpt2, tokenizer.ggml.pre=qwen2)
//
// Everything else about the two — merge ranks, token types, special-token
// handling, the bos/eos contract — is the same data, which is why the kind is
// one enum field and not two types.
//
// The BPE/detokenize *algorithms* live in core/src/vocab/ggml_derived/ (MIT,
// derived from ggml); this header and core/src/vocab/vocab.cpp only carry the
// SonicBoom-owned data layout and the GGUF loading glue.

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sonicboom::gguf {
class Reader;
}

namespace sonicboom::vocab {

// Token type ids, matching GGUF tokenizer.ggml.token_type and llama.cpp's
// LLAMA_TOKEN_TYPE values. `type` in a loaded Vocab is the *effective* type:
// llama.cpp post-processing promotes a handful of control-looking tokens (e.g.
// "<eos>", "<|tool_response>") to control so they are stripped on detokenize.
enum class TokenType : uint8_t {
  Normal = 1,
  Unknown = 2,
  Control = 3,
  UserDefined = 4,
  Unused = 5,
  Byte = 6,
};

// Which pre-tokenizer the vocabulary uses. The two kinds differ in how a piece
// of text is cut into words and how a word's bytes are spelled in the merge
// tables; the merge tables themselves and the detokenize rules are common.
enum class TokenizerKind : uint8_t {
  Gemma4 = 0,    // SPM-style BPE: ▁-escaped input, <0xNN> byte fallback
  Gpt2Qwen2 = 1, // byte-level BPE with the qwen2 pre-tokenizer regex
};

struct Vocab {
  std::vector<std::string> tokens;  // id -> piece text (▁-escaped)
  std::vector<TokenType> type;      // id -> effective type

  TokenizerKind kind = TokenizerKind::Gemma4;

  std::unordered_map<std::string, int32_t> token_to_id;

  // BPE merge ranks: (left, right) -> rank (lower merges first). The pair hash
  // mirrors llama.cpp's pair_hash.
  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
      return std::hash<std::string>{}(p.first) ^
             (std::hash<std::string>{}(p.second) << 1);
    }
  };
  std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash>
      bpe_ranks;

  int32_t bos = -1;
  int32_t eos = -1;
  int32_t unk = -1;

  bool add_bos = false;
  bool add_eos = false;
  bool add_space_prefix = false;
  bool clean_spaces = true;
  bool escape_whitespaces = true;

  bool empty() const noexcept { return tokens.empty(); }
};

// Load the tokenizer from a GGUF reader. Returns an error string when the file
// has no tokenizer, or one whose model / pre-tokenizer is not supported (the
// error names the value that was found). Gemma-4 force-enables add_bos like
// llama.cpp (its prompt always starts with <bos>); other architectures take
// add_bos from the file.
std::expected<Vocab, std::string> load_vocab(const gguf::Reader& reader);

// Load a Hugging Face `tokenizer.json` (the "tokenizers" library format) as a
// Gpt2Qwen2 Vocab. The file must use the byte-level BPE model with the qwen2
// pre-tokenizer regex (the Qwen2/Qwen3 family); any other model type or regex
// is rejected by name rather than approximated. `added_tokens` carry their ids
// and are typed Control when flagged special, so tokenize(..., parse_special)
// emits `<|im_start|>`-style markers as single ids, exactly like a GGUF
// vocabulary with the same pieces.
//
// The serialized normalizer (NFC for the Qwen family) is accepted but not
// applied: the tokenizer is a byte-level BPE over UTF-8, which for text that is
// already NFC is identical, and no ICU normalizer is linked. eos/bos/unk ids are
// read from a sibling `tokenizer_config.json` when present.
std::expected<Vocab, std::string> load_tokenizer_json(
    const std::filesystem::path& path);

// tokenize: text -> token ids. When add_bos is set the <bos> token is prepended
// (matching llama-tokenize's default), so a prompt round-trips through
// detokenize().
//
// parse_special is opt-in and defaults off. With it on, occurrences of the
// vocabulary's CONTROL / USER_DEFINED pieces are emitted as their own token id
// instead of being BPE'd as text — so a chat template's turn markers become
// single tokens. It is a text-to-control-token channel: only parse markers the
// caller generated itself, never the user's text. With the flag off the output
// is byte-identical to the text-only tokenizer.
//
// add_special (on by default, llama.cpp's flag of the same name) wraps the text
// in the vocabulary's framework tokens: <bos> in front when add_bos is set,
// <eos> after when add_eos is. A caller that tokenizes a prompt piece by piece
// passes it for the first piece only, so <bos> is added exactly once.
std::vector<int32_t> tokenize(const Vocab& v, std::string_view text,
                              bool parse_special = false, bool add_special = true);

// detokenize: token ids -> text. Strips control/unknown tokens (including
// <bos>), unescapes ▁ to space, expands <0xNN> byte tokens, and applies the
// clean-up-tokenization-spaces pass. detokenize(tokenize(x)) == x for normal
// text.
std::string detokenize(const Vocab& v, const int32_t* tokens, size_t n);

}  // namespace sonicboom::vocab
