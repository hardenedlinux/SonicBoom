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

// Internal generic s-expression tree + reader (layer 2 of the parser). It turns
// a flat token stream into a nested tree of atoms and lists; the builder layer
// (parser.cpp) then walks this tree into the typed IR. Not part of the public
// API.

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "lexer.h"
#include <sonicboom/sx/parser.h>

namespace sonicboom::sx::detail {

// A generic s-expression node: an atom or a list of nodes.
struct SExpr {
  enum class Kind : uint8_t { Symbol, String, Int, Float, Bool, List };

  Kind kind = Kind::List;
  SourceLocation loc;
  std::string text;          // Symbol name, or String contents
  int64_t i = 0;
  double f = 0.0;
  bool b = false;
  std::vector<SExpr> items;  // List

  bool is_list() const { return kind == Kind::List; }
  bool is_symbol(std::string_view s) const {
    return kind == Kind::Symbol && text == s;
  }
};

// Human-readable rendering of a node, for error messages.
std::string describe(const SExpr& e);

// Reads the token stream into a single s-expression tree. Errors are
// Syntax-category (unbalanced parens, unexpected end of input, stray ')').
class Reader {
 public:
  explicit Reader(const std::vector<Token>& tokens) : tokens_(tokens) {}

  // Reads exactly one s-expression (advancing past it).
  std::expected<SExpr, Error> read_one();

  // True when the next token is End (no more forms remain).
  bool at_end() const { return peek().kind == TokenKind::End; }

  // Location of the next token (for "trailing form" errors).
  SourceLocation peek_location() const { return peek().loc; }

 private:
  const Token& peek() const { return tokens_[pos_]; }
  void advance() {
    if (pos_ < tokens_.size()) {
      ++pos_;
    }
  }

  const std::vector<Token>& tokens_;
  size_t pos_ = 0;
};

} // namespace sonicboom::sx::detail
