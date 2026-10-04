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

// SonicBoom S-Expr v0.1 IR — the native C++ data model for a parsed document.
//
// This is the *serialized model / graph IR* (the upper layer that feeds MLIR
// lowering), not the Layer 2 runtime types (nt::). It is pure C++ with no
// dependency on native-torch, Guile, MLIR, or ONNX. The frozen format this
// models is documented in design/s-expr-v0-spec.md and must not be redesigned
// here.

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace sonicboom::sx {

// --- Type system -----------------------------------------------------------

// The frozen ten-name dtype set (spec §5).
enum class DType : uint8_t {
  Float32,
  Float16,
  BFloat16,
  Float64,
  Int8,
  UInt8,
  Int16,
  Int32,
  Int64,
  Bool,
};

// Textual name of a dtype as it appears in the format (e.g. "float32").
const char* dtype_name(DType d);

// Static, non-negative dimensions (spec §5). rank == dims.size(); a scalar
// tensor has an empty dims vector.
struct Shape {
  std::vector<int64_t> dims;
};

// The only value type in v0: a tensor of a given dtype and static shape.
struct TensorType {
  DType dtype;
  Shape shape;
};

// --- Attributes ------------------------------------------------------------

enum class AttrValueKind : uint8_t {
  Int,
  Float,
  String,
  Bool,
  Ints,
  Floats,
  Strings,
  Bools,
};

// Inline scalar/vector metadata attached to a node (spec §9). Not an SSA value.
struct Attribute {
  std::string name;
  std::variant<int64_t, double, std::string, bool,
               std::vector<int64_t>, std::vector<double>,
               std::vector<std::string>, std::vector<bool>>
      value;

  AttrValueKind kind() const {
    return static_cast<AttrValueKind>(value.index());
  }
};

// --- Parameters ------------------------------------------------------------

// Flat external binary sidecar reference (spec §7.3).
struct ExternalData {
  std::string file;
  int64_t offset;
  int64_t length;
};

// Inline literal list; element type is implied by the parameter's dtype.
struct ValuesData {
  std::vector<std::variant<int64_t, double>> values;
};

using ParameterData = std::variant<ExternalData, ValuesData>;

// A parameter unifies weights and constants (spec §7). There is no separate
// `constant` form and no `role` tag.
struct Parameter {
  std::string name;
  TensorType type;
  ParameterData data;
};

// --- Nodes -----------------------------------------------------------------

struct Node {
  struct Output {
    std::string name;
    TensorType type;
  };

  std::string op_name;             // canonical lowercase symbol (spec §8)
  std::vector<std::string> inputs; // SSA references (value names)
  std::vector<Output> outputs;     // SSA definitions (name + type)
  std::vector<Attribute> attributes;
  std::string domain = "default";  // default per spec §8
  std::optional<int64_t> version;  // nullopt = "graph opset version" (spec §8)
};

// --- Graph -----------------------------------------------------------------

struct Graph {
  struct Opset {
    std::string domain;
    int64_t version;
  };
  struct Input {
    std::string name;
    TensorType type;
  };

  std::string name;
  std::vector<Opset> opsets;
  std::vector<Input> inputs;
  std::vector<std::string> outputs; // name-only references (spec §4.4)
  std::vector<Parameter> parameters;
  std::vector<Node> nodes;
};

// --- Document --------------------------------------------------------------

struct Document {
  int64_t version_major;
  int64_t version_minor;
  Graph graph;
};

} // namespace sonicboom::sx
