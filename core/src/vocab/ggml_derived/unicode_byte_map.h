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
// Derived from ggml (llama.cpp), MIT-licensed. UTF-8 <-> codepoint conversion
// and the GPT-2 byte-level encoding, transcribed from:
//
//   llama.cpp/src/unicode.cpp — unicode_cpt_to_utf8, unicode_cpt_from_utf8,
//                               unicode_cpts_from_utf8, unicode_byte_to_utf8_map,
//                               unicode_utf8_to_byte_map, unicode_byte_encoding_process
//   llama.cpp/src/llama-vocab.cpp — llama_decode_text
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
// SonicBoom's changes: the maps are built as plain arrays instead of
// std::unordered_map (the index spaces are 256 bytes and 256 strings), and
// invalid UTF-8 is handled by the same U+FFFD substitution upstream does rather
// than by throwing.
// ---------------------------------------------------------------------------

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sonicboom::vocab::ggml_unicode {

// ---------------------------------------------------------------------------
// UTF-8 <-> codepoint
// ---------------------------------------------------------------------------

// Encode one codepoint as UTF-8. Codepoints above U+10FFFF cannot arise here
// (they are never produced by cpts_from_utf8 and never stored in a GGUF vocab),
// so upstream's throw becomes an empty string.
inline std::string cpt_to_utf8(uint32_t cpt) {
  std::string result;
  if (cpt <= 0x7F) {
    result.push_back(char(cpt));
    return result;
  }
  if (cpt <= 0x7FF) {
    result.push_back(char(0xC0 | ((cpt >> 6) & 0x1F)));
    result.push_back(char(0x80 | (cpt & 0x3F)));
    return result;
  }
  if (cpt <= 0xFFFF) {
    result.push_back(char(0xE0 | ((cpt >> 12) & 0x0F)));
    result.push_back(char(0x80 | ((cpt >> 6) & 0x3F)));
    result.push_back(char(0x80 | (cpt & 0x3F)));
    return result;
  }
  if (cpt <= 0x10FFFF) {
    result.push_back(char(0xF0 | ((cpt >> 18) & 0x07)));
    result.push_back(char(0x80 | ((cpt >> 12) & 0x3F)));
    result.push_back(char(0x80 | ((cpt >> 6) & 0x3F)));
    result.push_back(char(0x80 | (cpt & 0x3F)));
    return result;
  }
  return result;
}

// Decode the codepoint starting at utf8[offset]. Returns false (leaving offset
// untouched) where upstream throws std::invalid_argument; the caller is
// responsible for the U+FFFD substitution and for advancing the offset.
inline bool cpt_from_utf8(std::string_view utf8, size_t& offset, uint32_t& out) {
  const auto byte = [&](size_t i) { return uint8_t(utf8[i]); };
  if (offset >= utf8.size()) {
    return false;
  }
  if (!(byte(offset) & 0x80)) {
    out = byte(offset);
    offset += 1;
    return true;
  }
  if (!(byte(offset) & 0x40)) {
    return false;
  }
  if (!(byte(offset) & 0x20)) {
    if (offset + 1 >= utf8.size() || !((byte(offset + 1) & 0xC0) == 0x80)) {
      return false;
    }
    out = ((byte(offset) & 0x1F) << 6) | (byte(offset + 1) & 0x3F);
    offset += 2;
    return true;
  }
  if (!(byte(offset) & 0x10)) {
    if (offset + 2 >= utf8.size() || !((byte(offset + 1) & 0xC0) == 0x80) ||
        !((byte(offset + 2) & 0xC0) == 0x80)) {
      return false;
    }
    out = ((byte(offset) & 0x0F) << 12) | ((byte(offset + 1) & 0x3F) << 6) |
          (byte(offset + 2) & 0x3F);
    offset += 3;
    return true;
  }
  if (!(byte(offset) & 0x08)) {
    if (offset + 3 >= utf8.size() || !((byte(offset + 1) & 0xC0) == 0x80) ||
        !((byte(offset + 2) & 0xC0) == 0x80) || !((byte(offset + 3) & 0xC0) == 0x80)) {
      return false;
    }
    out = ((byte(offset) & 0x07) << 18) | ((byte(offset + 1) & 0x3F) << 12) |
          ((byte(offset + 2) & 0x3F) << 6) | (byte(offset + 3) & 0x3F);
    offset += 4;
    return true;
  }
  return false;
}

// An invalid byte becomes one U+FFFD, and the offset advances by one — the
// same recovery upstream performs.
inline std::vector<uint32_t> cpts_from_utf8(std::string_view utf8) {
  std::vector<uint32_t> result;
  result.reserve(utf8.size());
  size_t offset = 0;
  while (offset < utf8.size()) {
    uint32_t cpt = 0;
    if (cpt_from_utf8(utf8, offset, cpt)) {
      result.push_back(cpt);
    } else {
      ++offset;
      result.push_back(0xFFFD);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level encoding
// ---------------------------------------------------------------------------

// Every byte the vocabulary can contain is mapped to a printable codepoint so
// that no byte, including whitespace and the control bytes, has to appear
// literally in the vocabulary. Bytes 0x21-0x7E, 0xA1-0xAC and 0xAE-0xFF map to
// themselves; the remaining 68 bytes map to U+0100.. in increasing byte order,
// which is why a space is stored as U+0120 and shows up as "Ġ" in the
// vocabularies' token strings.
inline const std::array<std::string, 256>& byte_to_utf8_map() {
  static const std::array<std::string, 256> map = [] {
    std::array<std::string, 256> m;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {
      m[ch] = cpt_to_utf8(uint32_t(ch));
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {
      m[ch] = cpt_to_utf8(uint32_t(ch));
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {
      m[ch] = cpt_to_utf8(uint32_t(ch));
    }
    int n = 0;
    for (int ch = 0; ch < 256; ++ch) {
      if (m[ch].empty()) {
        m[ch] = cpt_to_utf8(uint32_t(256 + n));
        ++n;
      }
    }
    return m;
  }();
  return map;
}

inline const std::unordered_map<std::string, uint8_t>& utf8_to_byte_map() {
  static const std::unordered_map<std::string, uint8_t> map = [] {
    std::unordered_map<std::string, uint8_t> m;
    const auto& forward = byte_to_utf8_map();
    for (int ch = 0; ch < 256; ++ch) {
      m.emplace(forward[ch], uint8_t(ch));
    }
    return m;
  }();
  return map;
}

// Byte-level encode a word for the BPE merge tables. Upstream re-serializes the
// word from its codepoints first (unicode_byte_encoding_process), so an invalid
// byte is encoded as the UTF-8 of U+FFFD rather than as itself; this does the
// same.
inline std::string byte_encode(std::string_view word) {
  const auto& map = byte_to_utf8_map();
  const std::vector<uint32_t> cpts = cpts_from_utf8(word);
  std::string encoded;
  encoded.reserve(cpts.size() * 2);
  for (const uint32_t cpt : cpts) {
    for (const char c : cpt_to_utf8(cpt)) {
      encoded += map[uint8_t(c)];
    }
  }
  return encoded;
}

// Byte-level decode, the inverse used when detokenizing a gpt2-model
// vocabulary. Returns false if some codepoint has no byte (which cannot happen
// for a well-formed vocabulary) so the caller can fall back without throwing.
inline bool byte_decode(std::string_view utf8, std::string& out) {
  const auto& map = utf8_to_byte_map();
  out.clear();
  for (const uint32_t cpt : cpts_from_utf8(utf8)) {
    const auto it = map.find(cpt_to_utf8(cpt));
    if (it == map.end()) {
      return false;
    }
    out.push_back(char(it->second));
  }
  return true;
}

}  // namespace sonicboom::vocab::ggml_unicode
