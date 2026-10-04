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

#include "sexpr.h"

#include <utility>

namespace sonicboom::sx::detail {

std::string describe(const SExpr& e) {
  switch (e.kind) {
    case SExpr::Kind::Symbol: return "'" + e.text + "'";
    case SExpr::Kind::String: return "\"" + e.text + "\"";
    case SExpr::Kind::Int: return std::to_string(e.i);
    case SExpr::Kind::Float: return std::to_string(e.f);
    case SExpr::Kind::Bool: return e.b ? "#t" : "#f";
    case SExpr::Kind::List: return "(...)";
  }
  return "?";
}

std::expected<SExpr, Error> Reader::read_one() {
  const Token& t = peek();
  switch (t.kind) {
    case TokenKind::End:
      return std::unexpected(
          sx_error(ErrorCategory::Syntax, t.loc, "unexpected end of input"));
    case TokenKind::RParen:
      return std::unexpected(
          sx_error(ErrorCategory::Syntax, t.loc, "unexpected ')'"));
    case TokenKind::LParen: {
      SourceLocation loc = t.loc;
      advance();
      SExpr list;
      list.kind = SExpr::Kind::List;
      list.loc = loc;
      for (;;) {
        const Token& cur = peek();
        if (cur.kind == TokenKind::End) {
          return std::unexpected(sx_error(ErrorCategory::Syntax, loc,
                                          "unbalanced '(': expected ')'"));
        }
        if (cur.kind == TokenKind::RParen) {
          advance();
          break;
        }
        auto item = read_one();
        if (!item) {
          return std::unexpected(item.error());
        }
        list.items.push_back(std::move(*item));
      }
      return list;
    }
    case TokenKind::Symbol: {
      SExpr a;
      a.kind = SExpr::Kind::Symbol;
      a.loc = t.loc;
      a.text = t.text;
      advance();
      return a;
    }
    case TokenKind::String: {
      SExpr a;
      a.kind = SExpr::Kind::String;
      a.loc = t.loc;
      a.text = t.text;
      advance();
      return a;
    }
    case TokenKind::Int: {
      SExpr a;
      a.kind = SExpr::Kind::Int;
      a.loc = t.loc;
      a.i = t.i;
      advance();
      return a;
    }
    case TokenKind::Float: {
      SExpr a;
      a.kind = SExpr::Kind::Float;
      a.loc = t.loc;
      a.f = t.f;
      advance();
      return a;
    }
    case TokenKind::Bool: {
      SExpr a;
      a.kind = SExpr::Kind::Bool;
      a.loc = t.loc;
      a.b = t.b;
      advance();
      return a;
    }
  }
  return std::unexpected(
      sx_error(ErrorCategory::Syntax, t.loc, "unexpected token"));
}

} // namespace sonicboom::sx::detail
