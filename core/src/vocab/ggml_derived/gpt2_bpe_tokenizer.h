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
// Derived from ggml (llama.cpp), MIT-licensed. The byte-level BPE pre-tokenizer
// below is the qwen2 pattern, transcribed from:
//
//   llama.cpp/src/unicode.cpp — unicode_regex_split_custom_qwen2
//                               (matcher for
//                                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
//                                "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
//                                "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
//                                "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
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
//
// SonicBoom's changes: upstream's matcher is written over a span list and
// returns offsets into a shared codepoint vector; this is the same automaton
// over one span, returning the words themselves. The BPE merge loop itself is
// shared with the gemma4 tokenizer (bpe_tokenizer.h) instead of duplicated.
// ---------------------------------------------------------------------------

#pragma once

#include <sonicboom/vocab/vocab.h>

#include "bpe_tokenizer.h"
#include "unicode_byte_map.h"
#include "unicode_cpt_flags.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::vocab::ggml_gpt2 {

// --- qwen2 pre-tokenizer ----------------------------------------------------

// Split one span into the words the BPE merge tables are keyed on, i.e. the
// qwen2 regex above. The alternation is tried in the order the regex states,
// at each position, exactly as the hand-written matcher does; `_get_flags`
// off the end of the span yields the empty flag set, so every arm's lookahead
// terminates there.
inline std::vector<std::string> qwen2_split(std::string_view span) {
  using ggml_unicode::CptFlags;
  constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

  const std::vector<uint32_t> cpts = ggml_unicode::cpts_from_utf8(span);
  const size_t n = cpts.size();

  auto cpt_at = [&](size_t pos) -> uint32_t {
    return pos < n ? cpts[pos] : kOutOfRange;
  };
  auto flags_at = [&](size_t pos) -> CptFlags {
    return pos < n ? ggml_unicode::cpt_flags(cpts[pos]) : CptFlags{};
  };

  std::vector<std::string> words;
  size_t prev_end = 0;
  auto add_word = [&](size_t end) {
    if (end > prev_end) {
      std::string word;
      for (size_t p = prev_end; p < end; ++p) word += ggml_unicode::cpt_to_utf8(cpts[p]);
      words.push_back(std::move(word));
    }
    prev_end = end;
  };

  size_t pos = 0;
  while (pos < n) {
    const uint32_t cpt = cpt_at(pos);
    const CptFlags flags = flags_at(pos);

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (cpt == '\'' && pos + 1 < n) {
      const uint32_t c1 = ggml_unicode::ascii_tolower(cpt_at(pos + 1));
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
        add_word(pos + 2);
        pos += 2;
        continue;
      }
      if (pos + 2 < n) {
        const uint32_t c2 = ggml_unicode::ascii_tolower(cpt_at(pos + 2));
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
          add_word(pos + 3);
          pos += 3;
          continue;
        }
      }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
      if (flags.is_letter || flags_at(pos + 1).is_letter) {
        ++pos;
        while (flags_at(pos).is_letter) ++pos;
        add_word(pos);
        continue;
      }
    }

    // \p{N}
    if (flags.is_number) {
      add_word(++pos);
      continue;
    }

    // <space>?[^\s\p{L}\p{N}]+[\r\n]*
    CptFlags flags2 = (cpt == ' ') ? flags_at(pos + 1) : flags;
    if (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags.any) {
      if (cpt == ' ') ++pos;
      while (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.any) {
        flags2 = flags_at(++pos);
      }
      uint32_t cpt2 = cpt_at(pos);
      while (cpt2 == '\r' || cpt2 == '\n') {
        cpt2 = cpt_at(++pos);
      }
      add_word(pos);
      continue;
    }

    // \s*[\r\n]+ | \s+(?!\S) | \s+
    size_t num_whitespaces = 0;
    size_t last_end_r_or_n = 0;
    while (flags_at(pos + num_whitespaces).is_whitespace) {
      const uint32_t cpt2 = cpt_at(pos + num_whitespaces);
      if (cpt2 == '\r' || cpt2 == '\n') {
        last_end_r_or_n = pos + num_whitespaces + 1;
      }
      ++num_whitespaces;
    }
    if (last_end_r_or_n > 0) {
      pos = last_end_r_or_n;
      add_word(pos);
      continue;
    }
    // \s+(?!\S): a trailing run, but the last whitespace belongs to the next
    // word — upstream's negative lookahead, written as `pos + num_whitespaces
    // < n` plus the "at least two" guard.
    if (num_whitespaces > 1 && pos + num_whitespaces < n) {
      pos += num_whitespaces - 1;
      add_word(pos);
      continue;
    }
    if (num_whitespaces > 0) {
      pos += num_whitespaces;
      add_word(pos);
      continue;
    }

    add_word(++pos);  // one codepoint that matched nothing above
  }

  return words;
}

// One span -> tokens: the qwen2 split, then byte-level encode each word, then
// run it through the shared BPE merge.
inline void tokenize_span(const Vocab& v, std::string_view span,
                          std::vector<int32_t>& out) {
  for (const std::string& word : qwen2_split(span)) {
    ggml_bpe::tokenize_word(v, ggml_unicode::byte_encode(word), out,
                            /*byte_encoded=*/true);
  }
}

// text -> token ids. Same contract as the gemma4 tokenizer: `parse_special`
// is opt-in and `add_special` adds the vocabulary's <bos>/<eos> when the vocab
// asks for them.
inline std::vector<int32_t> tokenize(const Vocab& v, std::string_view text,
                                     bool parse_special = false,
                                     bool add_special = true) {
  std::vector<int32_t> out;
  if (add_special && v.add_bos && v.bos >= 0) out.push_back(v.bos);

  if (!parse_special) {
    tokenize_span(v, text, out);
  } else {
    const std::vector<int32_t> specials = ggml_bpe::special_token_ids(v);
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

// token ids -> text: the same detokenizer as the gemma4 vocabulary, whose
// Normal-token branch decodes through the byte table when the vocabulary is not
// whitespace-escaped — which is what a byte-level vocabulary needs.
inline std::string detokenize(const Vocab& v, const int32_t* tokens, size_t n) {
  return ggml_bpe::detokenize(v, tokens, n);
}

}  // namespace sonicboom::vocab::ggml_gpt2
