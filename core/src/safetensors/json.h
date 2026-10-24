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

// Small RFC 8259 JSON reader, internal to the safetensors loader.
//
// Scope: exactly what a model file header needs — objects, arrays, strings
// (with escapes and surrogate pairs), numbers, booleans and null. It is not a
// serializer and does not preserve number spelling. Duplicate object keys are
// rejected (the safetensors header spec forbids them). Numbers are exposed both
// as double and, when the literal has no fraction or exponent, as an exact
// int64 so byte offsets and shapes are never rounded.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonicboom::safetensors::detail {

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  int64_t integer = 0;
  bool is_integer = false;  // literal had no '.' or exponent, and fits int64
  std::string string;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;

  // Object member lookup; nullptr when absent or when this is not an object.
  const JsonValue* find(std::string_view key) const;
};

// Parses one complete JSON document. On failure returns nullopt and fills
// *error with a message that includes the byte offset.
std::optional<JsonValue> parse_json(std::string_view text, std::string* error);

} // namespace sonicboom::safetensors::detail
