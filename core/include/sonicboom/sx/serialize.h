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

// S-Expr v0.1 printer: the inverse of `parse_document`. Serializes a Document
// back to the frozen text format (see design/s-expr-v0-spec.md), losslessly
// round-tripping whatever the parser accepts. Pure SonicBoom C++ — no MLIR,
// Guile, or native-torch.

#include <sonicboom/sx/ir.h>
#include <sonicboom/sx/parser.h>  // Error

#include <expected>
#include <string>

namespace sonicboom::sx {

// Serialize `doc` to S-Expr v0.1 text. Returns an Error only when the IR holds
// a value the frozen format cannot represent (an out-of-range dtype); a
// well-formed Document always round-trips: parse(serialize(doc)) == doc.
std::expected<std::string, Error> serialize_document(const Document& doc);

} // namespace sonicboom::sx
