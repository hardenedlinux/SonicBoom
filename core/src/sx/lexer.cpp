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

#include "lexer.h"

#include <cctype>
#include <charconv>
#include <system_error>
#include <utility>

namespace sonicboom::sx::detail {

namespace {

bool is_delimiter(char c) {
  return c == '\0' || c == '(' || c == ')' || c == '"' || c == ';' || c == '#' ||
         c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

} // namespace

char Lexer::peek(size_t off) const {
  size_t p = pos_ + off;
  return p < text_.size() ? text_[p] : '\0';
}

void Lexer::advance() {
  if (pos_ < text_.size()) {
    if (text_[pos_] == '\n') {
      ++line_;
      col_ = 1;
    } else {
      ++col_;
    }
    ++pos_;
  }
}

void Lexer::skip_ws_and_comments() {
  for (;;) {
    char c = peek();
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      advance();
      continue;
    }
    if (c == ';') { // line comment to end of line
      while (peek() != '\0' && peek() != '\n') {
        advance();
      }
      continue;
    }
    break;
  }
}

std::expected<Token, Error> Lexer::next_token() {
  skip_ws_and_comments();
  SourceLocation loc = here();
  char c = peek();
  if (c == '\0') {
    Token t;
    t.kind = TokenKind::End;
    t.loc = loc;
    return t;
  }
  if (c == '(') {
    advance();
    Token t;
    t.kind = TokenKind::LParen;
    t.loc = loc;
    return t;
  }
  if (c == ')') {
    advance();
    Token t;
    t.kind = TokenKind::RParen;
    t.loc = loc;
    return t;
  }
  if (c == '"') {
    return read_string();
  }
  if (c == '#') {
    return read_bool();
  }
  if (std::isdigit(static_cast<unsigned char>(c)) ||
      (c == '-' && std::isdigit(static_cast<unsigned char>(peek(1))))) {
    return read_number();
  }
  return read_symbol();
}

std::expected<Token, Error> Lexer::read_string() {
  SourceLocation loc = here();
  advance(); // opening quote
  std::string out;
  for (;;) {
    char c = peek();
    if (c == '\0') {
      return std::unexpected(
          sx_error(ErrorCategory::Lexical, loc, "unterminated string literal"));
    }
    if (c == '"') {
      advance();
      break;
    }
    if (c == '\\') {
      advance();
      char e = peek();
      if (e == '\0') {
        return std::unexpected(sx_error(ErrorCategory::Lexical, loc,
                                        "unterminated string literal"));
      }
      switch (e) {
        case 'n': out.push_back('\n'); advance(); break;
        case 't': out.push_back('\t'); advance(); break;
        case 'r': out.push_back('\r'); advance(); break;
        case '\\': out.push_back('\\'); advance(); break;
        case '"': out.push_back('"'); advance(); break;
        default: // lenient: unknown escape keeps the backslash literally
          out.push_back('\\');
          out.push_back(e);
          advance();
          break;
      }
    } else {
      out.push_back(c);
      advance();
    }
  }
  Token t;
  t.kind = TokenKind::String;
  t.loc = loc;
  t.text = std::move(out);
  return t;
}

std::expected<Token, Error> Lexer::read_bool() {
  SourceLocation loc = here();
  advance(); // '#'
  char c = peek();
  Token t;
  t.loc = loc;
  if (c == 't') {
    advance();
    t.kind = TokenKind::Bool;
    t.b = true;
    return t;
  }
  if (c == 'f') {
    advance();
    t.kind = TokenKind::Bool;
    t.b = false;
    return t;
  }
  return std::unexpected(
      sx_error(ErrorCategory::Lexical, loc, "invalid literal: expected #t or #f"));
}

std::expected<Token, Error> Lexer::read_number() {
  SourceLocation loc = here();
  std::string atom;
  while (!is_delimiter(peek())) {
    atom.push_back(peek());
    advance();
  }

  // Try integer first (accepts a leading '-').
  {
    int64_t v = 0;
    const char* b = atom.data();
    const char* e = atom.data() + atom.size();
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec == std::errc{} && p == e) {
      Token t;
      t.kind = TokenKind::Int;
      t.loc = loc;
      t.i = v;
      return t;
    }
  }
  // Then float.
  {
    double v = 0.0;
    const char* b = atom.data();
    const char* e = atom.data() + atom.size();
    auto [p, ec] =
        std::from_chars(b, e, v, std::chars_format::general);
    if (ec == std::errc{} && p == e) {
      Token t;
      t.kind = TokenKind::Float;
      t.loc = loc;
      t.f = v;
      return t;
    }
  }
  return std::unexpected(sx_error(ErrorCategory::Lexical, loc,
                                  "invalid number literal '" + atom + "'"));
}

std::expected<Token, Error> Lexer::read_symbol() {
  SourceLocation loc = here();
  std::string sym;
  while (!is_delimiter(peek())) {
    sym.push_back(peek());
    advance();
  }
  Token t;
  t.kind = TokenKind::Symbol;
  t.loc = loc;
  t.text = std::move(sym);
  return t;
}

std::expected<std::vector<Token>, Error> Lexer::tokenize() {
  std::vector<Token> out;
  for (;;) {
    auto t = next_token();
    if (!t) {
      return std::unexpected(t.error());
    }
    out.push_back(std::move(*t));
    if (out.back().kind == TokenKind::End) {
      return out;
    }
  }
}

} // namespace sonicboom::sx::detail
