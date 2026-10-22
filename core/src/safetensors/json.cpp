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

#include "json.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_set>

namespace sonicboom::safetensors::detail {

const JsonValue* JsonValue::find(std::string_view key) const {
  if (type != Type::Object) return nullptr;
  for (const auto& [k, v] : object)
    if (k == key) return &v;
  return nullptr;
}

namespace {

constexpr int kMaxDepth = 128;

class Parser {
public:
  Parser(std::string_view text, std::string* error) : text_(text), error_(error) {}

  std::optional<JsonValue> parse() {
    skip_ws();
    auto v = parse_value(0);
    if (!v) return std::nullopt;
    skip_ws();
    if (pos_ != text_.size()) return fail("trailing characters after the document");
    return v;
  }

private:
  std::nullopt_t fail(std::string_view what) {
    *error_ = std::string(what) + " at byte " + std::to_string(pos_);
    return std::nullopt;
  }

  void skip_ws() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
      else break;
    }
  }

  bool consume(std::string_view lit) {
    if (text_.substr(pos_, lit.size()) != lit) return false;
    pos_ += lit.size();
    return true;
  }

  std::optional<JsonValue> parse_value(int depth) {
    if (depth > kMaxDepth) return fail("nesting too deep");
    if (pos_ >= text_.size()) return fail("unexpected end of input");
    const char c = text_[pos_];
    switch (c) {
      case '{': return parse_object(depth);
      case '[': return parse_array(depth);
      case '"': {
        JsonValue v;
        v.type = JsonValue::Type::String;
        if (!parse_string(v.string)) return std::nullopt;
        return v;
      }
      case 't': {
        if (!consume("true")) return fail("invalid literal");
        JsonValue v;
        v.type = JsonValue::Type::Bool;
        v.boolean = true;
        return v;
      }
      case 'f': {
        if (!consume("false")) return fail("invalid literal");
        JsonValue v;
        v.type = JsonValue::Type::Bool;
        return v;
      }
      case 'n': {
        if (!consume("null")) return fail("invalid literal");
        JsonValue v;
        return v;
      }
      default: return parse_number();
    }
  }

  std::optional<JsonValue> parse_object(int depth) {
    JsonValue v;
    v.type = JsonValue::Type::Object;
    ++pos_;  // '{'
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return v;
    }
    // Duplicate keys are rejected with a hash set rather than a linear scan:
    // a tokenizer.json vocabulary object has ~250k members, for which the
    // scan was quadratic.
    std::unordered_set<std::string> seen;
    while (true) {
      skip_ws();
      if (pos_ >= text_.size() || text_[pos_] != '"')
        return fail("expected an object key");
      std::string key;
      if (!parse_string(key)) return std::nullopt;
      skip_ws();
      if (pos_ >= text_.size() || text_[pos_] != ':')
        return fail("expected ':' after an object key");
      ++pos_;
      skip_ws();
      auto member = parse_value(depth + 1);
      if (!member) return std::nullopt;
      if (!seen.insert(key).second) return fail("duplicate object key");
      v.object.emplace_back(std::move(key), std::move(*member));
      skip_ws();
      if (pos_ >= text_.size()) return fail("unterminated object");
      if (text_[pos_] == ',') { ++pos_; continue; }
      if (text_[pos_] == '}') { ++pos_; return v; }
      return fail("expected ',' or '}' in an object");
    }
  }

  std::optional<JsonValue> parse_array(int depth) {
    JsonValue v;
    v.type = JsonValue::Type::Array;
    ++pos_;  // '['
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return v;
    }
    while (true) {
      skip_ws();
      auto elem = parse_value(depth + 1);
      if (!elem) return std::nullopt;
      v.array.push_back(std::move(*elem));
      skip_ws();
      if (pos_ >= text_.size()) return fail("unterminated array");
      if (text_[pos_] == ',') { ++pos_; continue; }
      if (text_[pos_] == ']') { ++pos_; return v; }
      return fail("expected ',' or ']' in an array");
    }
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  bool parse_hex4(uint32_t& out) {
    if (pos_ + 4 > text_.size()) { fail("truncated \\u escape"); return false; }
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else { fail("invalid hex digit in \\u escape"); return false; }
    }
    out = v;
    return true;
  }

  bool parse_string(std::string& out) {
    ++pos_;  // opening quote
    while (true) {
      if (pos_ >= text_.size()) { fail("unterminated string"); return false; }
      const unsigned char c = static_cast<unsigned char>(text_[pos_]);
      if (c == '"') { ++pos_; return true; }
      if (c < 0x20) { fail("control character in a string"); return false; }
      if (c != '\\') { out.push_back(char(c)); ++pos_; continue; }
      ++pos_;
      if (pos_ >= text_.size()) { fail("unterminated escape"); return false; }
      const char esc = text_[pos_++];
      switch (esc) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t cp = 0;
          if (!parse_hex4(cp)) return false;
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (!consume("\\u")) { fail("unpaired high surrogate"); return false; }
            uint32_t low = 0;
            if (!parse_hex4(low)) return false;
            if (low < 0xDC00 || low > 0xDFFF) { fail("invalid low surrogate"); return false; }
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("unpaired low surrogate");
            return false;
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("invalid escape"); return false;
      }
    }
  }

  std::optional<JsonValue> parse_number() {
    const size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    const size_t digits_begin = pos_;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    if (pos_ == digits_begin) return fail("invalid number");
    bool integral = true;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      integral = false;
      ++pos_;
      const size_t frac_begin = pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
      if (pos_ == frac_begin) return fail("invalid fraction");
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      integral = false;
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      const size_t exp_begin = pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
      if (pos_ == exp_begin) return fail("invalid exponent");
    }

    const std::string_view literal = text_.substr(start, pos_ - start);
    JsonValue v;
    v.type = JsonValue::Type::Number;
    v.number = std::strtod(std::string(literal).c_str(), nullptr);
    if (integral) {
      int64_t parsed = 0;
      const auto res = std::from_chars(literal.data(), literal.data() + literal.size(), parsed);
      if (res.ec == std::errc() && res.ptr == literal.data() + literal.size()) {
        v.integer = parsed;
        v.is_integer = true;
      }
    }
    return v;
  }

  std::string_view text_;
  std::string* error_;
  size_t pos_ = 0;
};

} // namespace

std::optional<JsonValue> parse_json(std::string_view text, std::string* error) {
  std::string scratch;
  if (!error) error = &scratch;
  Parser parser(text, error);
  return parser.parse();
}

} // namespace sonicboom::safetensors::detail
