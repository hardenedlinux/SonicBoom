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
//
// ---------------------------------------------------------------------------
// Derived from ggml (llama.cpp), MIT-licensed. The SPM-style BPE tokenizer and
// detokenizer below (symbol-chain BPE merge, <0xNN> byte fallback, ▁-space
// escape/unescape, unicode_len_utf8, token_to_byte, clean_spaces, and the
// token-type override) are ported from the following upstream source:
//
//   llama.cpp/src/llama-vocab.cpp  — llm_tokenizer_bpe_session, detokenize
//   llama.cpp/src/unicode.cpp      — unicode_len_utf8, regex split (newlines)
//
// Upstream commit: llama.cpp 43fe9c64281ef735046adc025e9e7559a1f659a5
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <sonicboom/vocab/vocab.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <queue>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonicboom::vocab::ggml_bpe {

// --- unicode / whitespace helpers -------------------------------------------

// Number of bytes of the UTF-8 code point whose leading byte is `src`.
inline size_t unicode_len_utf8(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  return lookup[static_cast<uint8_t>(src) >> 4];
}

inline void replace_all(std::string& s, std::string_view from, std::string_view to) {
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

inline void escape_whitespace(std::string& text) {
  replace_all(text, " ", "\xe2\x96\x81");  // space -> U+2581
}

inline void unescape_whitespace(std::string& text) {
  replace_all(text, "\xe2\x96\x81", " ");  // U+2581 -> space
}

// Byte token text ("<0xNN>") -> the raw byte.
inline uint8_t token_to_byte(const std::string& text) {
  return static_cast<uint8_t>(std::strtol(text.substr(3, 2).c_str(), nullptr, 16));
}

// --- effective token type override (llama.cpp post-processing) --------------

// llama.cpp promotes control-looking tokens (the hard-coded end-of-generation
// list) to control, and resets gemma-4's "</s>" back to normal. These decide
// whether a token is stripped on detokenize.
inline void apply_token_type_overrides(Vocab& v) {
  auto is_eog = [](const std::string& t) {
    static const std::initializer_list<const char*> eog = {
        "<|eot_id|>",   "<|im_end|>",   "<|end|>",     "<|return|>",
        "<|call|>",     "<|flush|>",    "<|calls|>",   "<end_of_turn>",
        "<|endoftext|>", "</s>",        "<|eom_id|>",  "<EOT>",
        "_<EOT>",       "[EOT]",        "[EOS]",       "<|end_of_text|>",
        "<end_of_utterance>", "<eos>",  "<turn|>",     "<|tool_response>",
        "<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c" ">",
        "[e~["};
    for (const char* e : eog) {
      if (t == e) return true;
    }
    return false;
  };

  bool has_tool_response = false;
  int32_t slash_s = -1;
  for (size_t i = 0; i < v.tokens.size(); ++i) {
    const std::string& t = v.tokens[i];
    if (t == "<|tool_response>") has_tool_response = true;
    if (t == "</s>") slash_s = static_cast<int32_t>(i);
    if (is_eog(t)) {
      if (v.type[i] == TokenType::Normal || v.type[i] == TokenType::UserDefined) {
        v.type[i] = TokenType::Control;
      }
    }
  }
  // gemma4 (and plamo) keep a normal "</s>": it is removed from the EOG list.
  if (has_tool_response && slash_s >= 0) {
    v.type[slash_s] = TokenType::Normal;
  }
}

// --- BPE tokenize -----------------------------------------------------------

namespace {

struct Symbol {
  int32_t prev = -1;
  int32_t next = -1;
  const char* text = nullptr;
  size_t n = 0;
};

struct Bigram {
  int32_t left = 0;
  int32_t right = 0;
  std::string text;
  int32_t rank = 0;
};

struct BigramCmp {
  bool operator()(const Bigram& l, const Bigram& r) const {
    return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
  }
};

using BigramQueue = std::priority_queue<Bigram, std::vector<Bigram>, BigramCmp>;

void add_new_bigram(const Vocab& v, const std::vector<Symbol>& symbols,
                    BigramQueue& queue, int32_t left, int32_t right) {
  if (left == -1 || right == -1) return;
  std::string left_token(symbols[left].text, symbols[left].n);
  std::string right_token(symbols[right].text, symbols[right].n);
  auto it = v.bpe_ranks.find(std::make_pair(left_token, right_token));
  if (it == v.bpe_ranks.end()) return;
  Bigram bigram;
  bigram.left = left;
  bigram.right = right;
  bigram.text = left_token + right_token;
  bigram.rank = it->second;
  queue.push(std::move(bigram));
}

void tokenize_word(const Vocab& v, const std::string& word,
                   std::vector<int32_t>& out) {
  std::vector<Symbol> symbols;
  symbols.reserve(word.size());
  BigramQueue queue;

  size_t offset = 0;
  // gemma4 fix (llama.cpp PR #21343): a run of newlines that is itself a token
  // is emitted as one token instead of byte-falling-back per byte.
  if (word.find_first_not_of('\n') == std::string::npos) {
    auto it = v.token_to_id.find(word);
    if (it != v.token_to_id.end()) {
      out.push_back(it->second);
      return;
    }
  }

  int32_t index = 0;
  while (offset < word.size()) {
    Symbol sym;
    size_t char_len = std::min(word.size() - offset, unicode_len_utf8(word[offset]));
    sym.text = word.data() + offset;
    sym.n = char_len;
    offset += char_len;
    sym.prev = index - 1;
    sym.next = offset == word.size() ? -1 : index + 1;
    ++index;
    symbols.push_back(sym);
  }
  for (int32_t i = 1; i < static_cast<int32_t>(symbols.size()); ++i) {
    add_new_bigram(v, symbols, queue, i - 1, i);
  }

  while (!queue.empty()) {
    Bigram bigram = queue.top();
    queue.pop();

    Symbol& left = symbols[bigram.left];
    Symbol& right = symbols[bigram.right];
    if (left.n == 0 || right.n == 0) continue;
    std::string left_token(left.text, left.n);
    std::string right_token(right.text, right.n);
    if (left_token + right_token != bigram.text) continue;  // stale bigram

    left.n += right.n;
    right.n = 0;
    left.next = right.next;
    if (right.next >= 0) symbols[right.next].prev = bigram.left;

    add_new_bigram(v, symbols, queue, left.prev, bigram.left);
    add_new_bigram(v, symbols, queue, bigram.left, left.next);
  }

  static const char* hex = "0123456789ABCDEF";
  for (const auto& sym : symbols) {
    if (sym.n == 0) continue;
    const std::string str(sym.text, sym.n);
    auto it = v.token_to_id.find(str);
    if (it != v.token_to_id.end()) {
      out.push_back(it->second);
    } else {
      // non-byte-encoded BPE (gemma4): byte fallback uses <0xNN> uppercase hex
      for (size_t j = 0; j < str.size(); ++j) {
        const uint8_t ch = static_cast<uint8_t>(str[j]);
        const char buf[7] = {'<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0};
        auto bit = v.token_to_id.find(std::string(buf, 6));
        if (bit != v.token_to_id.end()) out.push_back(bit->second);
      }
    }
  }
}

}  // namespace

// One raw (unescaped) span -> tokens: ▁-escape, then the gemma4 pre-tokenizer
// "[^\n]+|[\n]+" — split into newline / non-newline runs. \n (0x0A) is a single
// byte and never a UTF-8 continuation byte, so a byte-level split is equivalent
// to the code-point split in llama.cpp — and then BPE each run.
inline void tokenize_span(const Vocab& v, std::string_view span,
                          std::vector<int32_t>& out) {
  std::string escaped(span);
  if (v.escape_whitespaces) escape_whitespace(escaped);

  size_t i = 0;
  while (i < escaped.size()) {
    const bool nl = escaped[i] == '\n';
    size_t j = i;
    while (j < escaped.size() && (escaped[j] == '\n') == nl) ++j;
    tokenize_word(v, std::string(escaped.data() + i, j - i), out);
    i = j;
  }
}

// The vocabulary's special pieces: CONTROL and USER_DEFINED tokens, longest
// piece first — llama.cpp's cache_special_tokens order (it sorts the same two
// types by text length), so that matching at a position is longest-match.
inline std::vector<int32_t> special_token_ids(const Vocab& v) {
  std::vector<int32_t> ids;
  for (size_t i = 0; i < v.tokens.size(); ++i) {
    const TokenType t = v.type[i];
    if (t == TokenType::Control || t == TokenType::UserDefined)
      ids.push_back(static_cast<int32_t>(i));
  }
  std::stable_sort(ids.begin(), ids.end(), [&v](int32_t a, int32_t b) {
    return v.tokens[a].size() > v.tokens[b].size();
  });
  return ids;
}

// text -> token ids (prepends <bos> when add_bos, matching llama-tokenize).
//
// `parse_special` is opt-in. When set, the text is first split at exact
// occurrences of the vocabulary's CONTROL / USER_DEFINED pieces (longest match
// at every position): each match contributes its own id directly and the spans
// between matches are tokenized as usual. It is off by default because it is a
// text-to-control-token channel — with it on, a message that merely *contains*
// "<turn|>" becomes a real turn boundary. Callers parse the markers they
// generated themselves (the chat template) and never the user's text.
//
// llama.cpp keeps matching USER_DEFINED pieces when parse_special is false;
// SonicBoom does not, so with the flag off the output is exactly what the
// text-only tokenizer produced before this parameter existed.
//
// `add_special` is llama.cpp's flag of the same name: it wraps the text in the
// vocabulary's framework tokens (<bos> / <eos>, when the vocab asks for them).
// A caller that tokenizes a prompt piece by piece passes it for the first piece
// only, so <bos> is added exactly once.
inline std::vector<int32_t> tokenize(const Vocab& v, std::string_view text,
                                     bool parse_special = false,
                                     bool add_special = true) {
  std::vector<int32_t> out;
  if (add_special && v.add_bos && v.bos >= 0) out.push_back(v.bos);

  if (!parse_special) {
    tokenize_span(v, text, out);
  } else {
    const std::vector<int32_t> specials = special_token_ids(v);
    size_t start = 0, i = 0;
    while (i < text.size()) {
      int32_t hit = -1;
      for (const int32_t id : specials) {
        const std::string& piece = v.tokens[id];
        if (piece.size() <= text.size() - i &&
            text.compare(i, piece.size(), piece) == 0) {
          hit = id;
          break;  // longest piece first
        }
      }
      if (hit < 0) {
        ++i;
        continue;
      }
      if (i > start) tokenize_span(v, text.substr(start, i - start), out);
      out.push_back(hit);
      i += v.tokens[hit].size();
      start = i;
    }
    if (start < text.size()) tokenize_span(v, text.substr(start), out);
  }

  if (add_special && v.add_eos && v.eos >= 0) out.push_back(v.eos);
  return out;
}

// --- detokenize -------------------------------------------------------------

// clean_up_tokenization_spaces: three in-place passes, transcribed from
// llama.cpp's llama_vocab::impl::detokenize.
inline void clean_spaces(std::string& text) {
  // first pass: characters ?!.,
  const int32_t total1 = static_cast<int32_t>(text.size());
  int32_t total = total1 ? 1 : 0;
  for (int32_t i = 1; i < total1; ++i) {
    const char x = text[i];
    if (text[i - 1] == ' ') {
      if (x == '?' || x == '!' || x == '.' || x == ',') total--;  // remove space
    }
    text[total++] = x;
  }

  // second pass: strip single apostrophe between spaces
  const int32_t total2 = total;
  total = total2 ? 1 : 0;
  for (int32_t i = 1; i < total2; ++i) {
    const char x = text[i];
    if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ') {
      total--;            // remove prev space
      text[++i] = '\0';   // remove next space
    }
    text[total++] = x;
  }

  // third pass: apostrophe contractions
  const int32_t total3 = total;
  total = total3 ? 1 : 0;
  for (int32_t i = 1; i < total3; ++i) {
    const char x = text[i];
    if (text[i - 1] == ' ') {
      if (x == '\'' && i + 1 < total3) {
        const char x1 = text[i + 1];
        if (x1 == 't' || x1 == 'd') {
          // keep space
        } else if (x1 == 's' || x1 == 'm') {
          total--;  // remove space
        } else if (i + 2 < total3) {
          const char x2 = text[i + 2];
          if (x1 == 'l' && x2 == 'l') {
            // keep space
          } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {
            total--;  // remove space
          }
        }
      }
    }
    text[total++] = x;
  }
  text.resize(total);
}

// token ids -> text. Strips control/unknown tokens (including <bos>), unescapes
// ▁ to space, expands <0xNN> byte tokens, applies clean_spaces.
inline std::string detokenize(const Vocab& v, const int32_t* tokens, size_t n) {
  std::string result;
  bool remove_space = v.add_space_prefix;  // false for gemma4

  for (size_t i = 0; i < n; ++i) {
    const int32_t t = tokens[i];
    if (t < 0 || t >= static_cast<int32_t>(v.tokens.size())) continue;
    const TokenType ty = v.type[t];

    if (ty == TokenType::Control || ty == TokenType::Unknown ||
        ty == TokenType::Unused) {
      continue;  // stripped on detokenize (remove_special semantics)
    }

    std::string piece;
    if (ty == TokenType::Byte) {
      piece = std::string(1, static_cast<char>(token_to_byte(v.tokens[t])));
    } else if (ty == TokenType::UserDefined) {
      piece = v.tokens[t];
    } else {  // Normal
      piece = v.tokens[t];
      if (v.escape_whitespaces) unescape_whitespace(piece);
    }

    if (remove_space && !piece.empty() && piece[0] == ' ') piece.erase(0, 1);
    remove_space = false;
    result += piece;
  }

  if (v.clean_spaces) clean_spaces(result);
  return result;
}

}  // namespace sonicboom::vocab::ggml_bpe
