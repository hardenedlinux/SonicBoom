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

// Internal S-Expr tokenizer (layer 1 of the parser). Not part of the public
// SonicBoom API.

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include <sonicboom/sx/parser.h>

namespace sonicboom::sx::detail {

// Construct a structured Error (shared by the lexer, reader, and builder).
inline Error sx_error(ErrorCategory category, SourceLocation location,
                      std::string message) {
  return Error{category, location, std::move(message)};
}

enum class TokenKind : uint8_t {
  LParen,
  RParen,
  Symbol, // fixed-vocabulary keyword / op-name / dtype / data tag
  String, // "..." (contents unescaped in .text)
  Int,    // integer literal (optional leading '-')
  Float,  // float literal (contains '.', optional exponent)
  Bool,   // #t / #f
  End,    // end of input (always the final token)
};

struct Token {
  TokenKind kind = TokenKind::End;
  SourceLocation loc;
  std::string text; // Symbol name, or String contents (unescaped)
  int64_t i = 0;
  double f = 0.0;
  bool b = false;
};

// Splits source text into tokens, handling whitespace, `;` line comments,
// quoted strings (with basic escapes), `#t`/`#f`, integers, and floats.
class Lexer {
 public:
  explicit Lexer(std::string_view text) : text_(text) {}

  // Returns the token vector (ending in an End token) on success, or a Lexical
  // error.
  std::expected<std::vector<Token>, Error> tokenize();

 private:
  char peek(size_t off = 0) const;
  void advance();
  SourceLocation here() const { return SourceLocation{line_, col_}; }

  void skip_ws_and_comments();
  std::expected<Token, Error> next_token();
  std::expected<Token, Error> read_string();
  std::expected<Token, Error> read_bool();
  std::expected<Token, Error> read_number();
  std::expected<Token, Error> read_symbol();

  std::string_view text_;
  size_t pos_ = 0;
  int line_ = 1;
  int col_ = 1;
};

} // namespace sonicboom::sx::detail
