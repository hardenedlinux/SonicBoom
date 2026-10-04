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

// Public entry point for parsing the frozen SonicBoom S-Expr v0.1 format into
// the native C++ IR (see ir.h). No MLIR, Guile, Python, or ONNX involvement;
// this is the S-Expr text → SonicBoom IR stage only.

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

#include <sonicboom/sx/ir.h>

namespace sonicboom::sx {

// A 1-based position in the source text. `line == 0` means "unavailable"
// (semantic errors raised over the whole document carry no single location).
struct SourceLocation {
  int line = 0;
  int column = 0;
};

// Broad classes of failure. The category is advisory; every error also carries
// a human-readable message and, when available, a source location.
enum class ErrorCategory : uint8_t {
  Lexical,     // the text could not be tokenized
  Syntax,      // malformed s-expression / a form has the wrong shape
  Semantic,    // well-formed but violates a v0 rule
  Unsupported, // well-formed but a future/unknown feature (fail-safe)
};

struct Error {
  ErrorCategory category;
  SourceLocation location;
  std::string message;
};

// Parse a SonicBoom S-Expr v0.1 document. Returns the Document on success, or a
// structured Error on failure. No exception crosses this boundary; the
// std::expected result is an explicit value-or-error status.
std::expected<Document, Error> parse_document(std::string_view text);

} // namespace sonicboom::sx
