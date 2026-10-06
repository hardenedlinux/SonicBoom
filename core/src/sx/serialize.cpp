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

#include <sonicboom/sx/serialize.h>

#include <charconv>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace sonicboom::sx {

namespace {

std::string fmt_int(int64_t v) { return std::to_string(v); }

// A float literal the lexer re-reads as a Float, not an Int. std::to_chars
// (general) gives the shortest round-trip representation; a value with no
// '.', 'e', or 'E' (e.g. "2" for 2.0) would otherwise re-parse as an integer.
std::string fmt_double(double v) {
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general);
  std::string s(buf, res.ptr);
  if (s.find_first_of(".eE") == std::string::npos && s != "inf" && s != "nan" &&
      s != "-inf" && s != "-nan") {
    s += ".0";
  }
  return s;
}

// Quote + escape a string literal (inverse of the lexer's read_string escapes).
std::string fmt_string(const std::string& s) {
  std::string out;
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c); break;
    }
  }
  out.push_back('"');
  return out;
}

std::string fmt_shape(const Shape& s) {
  std::string out = "(shape";
  for (int64_t d : s.dims) {
    out += ' ';
    out += fmt_int(d);
  }
  out += ')';
  return out;
}

std::string fmt_type(const TensorType& t) {
  return "(tensor " + std::string(dtype_name(t.dtype)) + " " + fmt_shape(t.shape) +
         ")";
}

std::string fmt_values(const ValuesData& d) {
  std::string out;
  for (const auto& v : d.values) {
    out += ' ';
    if (const auto* i = std::get_if<int64_t>(&v)) {
      out += fmt_int(*i);
    } else {
      out += fmt_double(std::get<double>(v));
    }
  }
  return out;
}

std::string fmt_data(const ParameterData& d) {
  if (const auto* ext = std::get_if<ExternalData>(&d)) {
    return "(data :external " + fmt_string(ext->file) + " " + fmt_int(ext->offset) +
           " " + fmt_int(ext->length) + ")";
  }
  return "(data :values" + fmt_values(std::get<ValuesData>(d)) + ")";
}

std::string fmt_attr(const Attribute& a) {
  std::string out = "(" + a.name;
  switch (a.kind()) {
    case AttrValueKind::Int:
      out += " (int " + fmt_int(std::get<int64_t>(a.value)) + ")";
      break;
    case AttrValueKind::Float:
      out += " (float " + fmt_double(std::get<double>(a.value)) + ")";
      break;
    case AttrValueKind::String:
      out += " (string " + fmt_string(std::get<std::string>(a.value)) + ")";
      break;
    case AttrValueKind::Bool:
      out += std::get<bool>(a.value) ? " (bool #t)" : " (bool #f)";
      break;
    case AttrValueKind::Ints: {
      out += " (ints";
      for (int64_t v : std::get<std::vector<int64_t>>(a.value)) {
        out += ' ';
        out += fmt_int(v);
      }
      out += ')';
      break;
    }
    case AttrValueKind::Floats: {
      out += " (floats";
      for (double v : std::get<std::vector<double>>(a.value)) {
        out += ' ';
        out += fmt_double(v);
      }
      out += ')';
      break;
    }
    case AttrValueKind::Strings: {
      out += " (strings";
      for (const std::string& v : std::get<std::vector<std::string>>(a.value)) {
        out += ' ';
        out += fmt_string(v);
      }
      out += ')';
      break;
    }
    case AttrValueKind::Bools: {
      out += " (bools";
      for (bool v : std::get<std::vector<bool>>(a.value)) {
        out += v ? " #t" : " #f";
      }
      out += ')';
      break;
    }
  }
  out += ')';
  return out;
}

std::string fmt_input(const Graph::Input& in) {
  return "(input " + fmt_string(in.name) + " " + fmt_type(in.type) + ")";
}

std::string fmt_opset(const Graph::Opset& o) {
  return "(opset " + fmt_string(o.domain) + " " + fmt_int(o.version) + ")";
}

std::string fmt_parameter(const Parameter& p) {
  return "(parameter " + fmt_string(p.name) + " " + fmt_type(p.type) + " " +
         fmt_data(p.data) + ")";
}

std::string fmt_value_def(const Node::Output& o) {
  return "(" + fmt_string(o.name) + " " + fmt_type(o.type) + ")";
}

std::string fmt_node(const Node& n) {
  std::string out = "(node " + n.op_name + " (inputs";
  for (const std::string& in : n.inputs) {
    out += ' ';
    out += fmt_string(in);
  }
  out += ") (outputs";
  for (const Node::Output& o : n.outputs) {
    out += ' ';
    out += fmt_value_def(o);
  }
  out += ')';
  if (!n.attributes.empty()) {
    out += " (attrs";
    for (const Attribute& a : n.attributes) {
      out += ' ';
      out += fmt_attr(a);
    }
    out += ')';
  }
  if (n.domain != "default") {
    out += " (domain " + fmt_string(n.domain) + ")";
  }
  if (n.version.has_value()) {
    out += " (version " + fmt_int(*n.version) + ")";
  }
  out += ')';
  return out;
}

bool representable(const Document& doc) {
  for (const Graph::Input& in : doc.graph.inputs)
    if (std::string_view(dtype_name(in.type.dtype)) == "?")
      return false;
  for (const Parameter& p : doc.graph.parameters)
    if (std::string_view(dtype_name(p.type.dtype)) == "?")
      return false;
  for (const Node& n : doc.graph.nodes)
    for (const Node::Output& o : n.outputs)
      if (std::string_view(dtype_name(o.type.dtype)) == "?")
        return false;
  return true;
}

} // namespace

std::expected<std::string, Error> serialize_document(const Document& doc) {
  if (!representable(doc)) {
    return std::unexpected(
        Error{ErrorCategory::Semantic, SourceLocation{},
              "document holds a dtype outside the frozen ten-name set"});
  }

  const Graph& g = doc.graph;
  std::string out;
  out += "(sonicboom-s-expr\n";
  out += "  (version " + fmt_int(doc.version_major) + " " +
         fmt_int(doc.version_minor) + ")\n";
  out += "  (graph\n";
  out += "    (name " + fmt_string(g.name) + ")\n";
  for (const Graph::Opset& o : g.opsets)
    out += "    " + fmt_opset(o) + "\n";

  out += "    (inputs";
  if (g.inputs.empty()) {
    out += ")\n";
  } else {
    out += "\n";
    for (const Graph::Input& in : g.inputs)
      out += "      " + fmt_input(in) + "\n";
    out += "    )\n";
  }

  out += "    (outputs\n";
  for (const std::string& name : g.outputs)
    out += "      (output " + fmt_string(name) + ")\n";
  out += "    )\n";

  out += "    (parameters";
  if (g.parameters.empty()) {
    out += ")\n";
  } else {
    out += "\n";
    for (const Parameter& p : g.parameters)
      out += "      " + fmt_parameter(p) + "\n";
    out += "    )\n";
  }

  out += "    (nodes";
  if (g.nodes.empty()) {
    out += ")\n";
  } else {
    out += "\n";
    for (const Node& n : g.nodes)
      out += "      " + fmt_node(n) + "\n";
    out += "    )\n";
  }

  out += "  )\n";
  out += ")\n";
  return out;
}

} // namespace sonicboom::sx
