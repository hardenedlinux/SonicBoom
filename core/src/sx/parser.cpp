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

#include <sonicboom/sx/parser.h>

// Builder + validator layers (3 and 4 of the parser). The builder walks the
// generic s-expression tree into the typed Document IR, enforcing all
// form-level v0 rules (arity, atom kinds, dtype/shape validity, data layouts,
// attribute tags). The validator then enforces the cross-cutting semantic rules
// that need the whole graph (single SSA definition, name resolution,
// topological order, opset domain declaration).

#include <string>
#include <unordered_map>
#include <utility>

#include "lexer.h"
#include "sexpr.h"

namespace sonicboom::sx {

namespace {

using detail::SExpr;
using detail::describe;
using detail::sx_error;

using ExpectedDType = std::expected<DType, Error>;
using ExpectedShape = std::expected<Shape, Error>;
using ExpectedType = std::expected<TensorType, Error>;
using ExpectedData = std::expected<ParameterData, Error>;
using ExpectedAttr = std::expected<Attribute, Error>;
using ExpectedValueDef = std::expected<Node::Output, Error>;
using ExpectedNode = std::expected<Node, Error>;
using ExpectedGraph = std::expected<Graph, Error>;
using ExpectedDocument = std::expected<Document, Error>;

bool is_list(const SExpr& e) { return e.kind == SExpr::Kind::List; }

// True if `e` is a list whose first element is the symbol `head`.
bool form(const SExpr& e, std::string_view head) {
  return is_list(e) && !e.items.empty() && e.items[0].is_symbol(head);
}

// A one-word description of a form head, for "unsupported" messages.
std::string head_name(const SExpr& e) {
  if (e.kind == SExpr::Kind::Symbol) {
    return e.text;
  }
  return describe(e);
}

bool dtype_from_name(std::string_view name, DType& out) {
  static const struct {
    std::string_view name;
    DType d;
  } kTable[] = {
      {"float32", DType::Float32}, {"float16", DType::Float16},
      {"bfloat16", DType::BFloat16}, {"float64", DType::Float64},
      {"int8", DType::Int8}, {"uint8", DType::UInt8},
      {"int16", DType::Int16}, {"int32", DType::Int32},
      {"int64", DType::Int64}, {"bool", DType::Bool},
  };
  for (const auto& e : kTable) {
    if (e.name == name) {
      out = e.d;
      return true;
    }
  }
  return false;
}

// --- type system -----------------------------------------------------------

ExpectedDType parse_dtype(const SExpr& e) {
  if (e.kind != SExpr::Kind::Symbol) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                    "expected a dtype symbol, got " + describe(e)));
  }
  DType d;
  if (!dtype_from_name(e.text, d)) {
    return std::unexpected(
        sx_error(ErrorCategory::Semantic, e.loc, "invalid dtype '" + e.text + "'"));
  }
  return d;
}

ExpectedShape parse_shape(const SExpr& e) {
  if (!form(e, "shape")) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                    "expected (shape <dim>*)"));
  }
  Shape s;
  for (size_t i = 1; i < e.items.size(); ++i) {
    const SExpr& d = e.items[i];
    if (d.kind != SExpr::Kind::Int) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, d.loc,
                                      "shape dimension must be an integer, got " +
                                          describe(d)));
    }
    if (d.i < 0) {
      return std::unexpected(sx_error(ErrorCategory::Semantic, d.loc,
                                      "shape dimension must be non-negative, got " +
                                          std::to_string(d.i)));
    }
    s.dims.push_back(d.i);
  }
  return s;
}

ExpectedType parse_type(const SExpr& e) {
  if (!is_list(e) || e.items.size() != 3 || !e.items[0].is_symbol("tensor")) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                    "expected (tensor <dtype> <shape>)"));
  }
  auto d = parse_dtype(e.items[1]);
  if (!d) {
    return std::unexpected(d.error());
  }
  auto s = parse_shape(e.items[2]);
  if (!s) {
    return std::unexpected(s.error());
  }
  return TensorType{*d, *s};
}

// --- data ------------------------------------------------------------------

ExpectedData parse_data(const SExpr& e) {
  if (!is_list(e) || e.items.size() < 2 || !e.items[0].is_symbol("data")) {
    return std::unexpected(
        sx_error(ErrorCategory::Syntax, e.loc,
                 "expected (data :external <file> <offset> <length>) or "
                 "(data :values <atom>*)"));
  }
  const SExpr& tag = e.items[1];
  if (tag.kind != SExpr::Kind::Symbol) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, tag.loc,
                                    "expected :external or :values data tag, got " +
                                        describe(tag)));
  }

  if (tag.is_symbol(":external")) {
    if (e.items.size() != 5) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                      ":external data requires <file> <offset> <length>"));
    }
    if (e.items[2].kind != SExpr::Kind::String) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.items[2].loc,
                                      ":external file must be a string"));
    }
    if (e.items[3].kind != SExpr::Kind::Int || e.items[4].kind != SExpr::Kind::Int) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                      ":external offset and length must be integers"));
    }
    if (e.items[3].i < 0 || e.items[4].i < 0) {
      return std::unexpected(sx_error(ErrorCategory::Semantic, e.loc,
                                      ":external offset and length must be non-negative"));
    }
    ExternalData d;
    d.file = e.items[2].text;
    d.offset = e.items[3].i;
    d.length = e.items[4].i;
    return d;
  }

  if (tag.is_symbol(":values")) {
    ValuesData d;
    for (size_t i = 2; i < e.items.size(); ++i) {
      const SExpr& a = e.items[i];
      if (a.kind == SExpr::Kind::Int) {
        d.values.push_back(static_cast<int64_t>(a.i));
      } else if (a.kind == SExpr::Kind::Float) {
        d.values.push_back(a.f);
      } else {
        return std::unexpected(sx_error(ErrorCategory::Semantic, a.loc,
                                        ":values elements must be numeric literals, got " +
                                            describe(a)));
      }
    }
    return d;
  }

  return std::unexpected(sx_error(ErrorCategory::Unsupported, tag.loc,
                                  "unsupported data layout '" + tag.text + "'"));
}

// --- attributes ------------------------------------------------------------

ExpectedAttr parse_attr(const SExpr& e) {
  if (!is_list(e) || e.items.size() != 2) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                    "expected attribute (name <typed-value>)"));
  }
  if (e.items[0].kind != SExpr::Kind::Symbol) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.items[0].loc,
                                    "attribute name must be a symbol, got " +
                                        describe(e.items[0])));
  }
  Attribute a;
  a.name = e.items[0].text;

  const SExpr& tv = e.items[1];
  if (!is_list(tv) || tv.items.empty() || tv.items[0].kind != SExpr::Kind::Symbol) {
    return std::unexpected(
        sx_error(ErrorCategory::Syntax, tv.loc,
                 "expected typed attribute value "
                 "(int|float|string|bool|ints|floats|strings|bools)"));
  }
  const std::string& tag = tv.items[0].text;

  if (tag == "int") {
    if (tv.items.size() != 2 || tv.items[1].kind != SExpr::Kind::Int) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, tv.loc,
                                      "(int <int>) requires one integer"));
    }
    a.value = static_cast<int64_t>(tv.items[1].i);
  } else if (tag == "float") {
    if (tv.items.size() != 2 || tv.items[1].kind != SExpr::Kind::Float) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, tv.loc,
                                      "(float <float>) requires one float"));
    }
    a.value = tv.items[1].f;
  } else if (tag == "string") {
    if (tv.items.size() != 2 || tv.items[1].kind != SExpr::Kind::String) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, tv.loc,
                                      "(string <string>) requires one string"));
    }
    a.value = tv.items[1].text;
  } else if (tag == "bool") {
    if (tv.items.size() != 2 || tv.items[1].kind != SExpr::Kind::Bool) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, tv.loc,
                                      "(bool #t|#f) requires one boolean"));
    }
    a.value = tv.items[1].b;
  } else if (tag == "ints") {
    std::vector<int64_t> v;
    for (size_t i = 1; i < tv.items.size(); ++i) {
      if (tv.items[i].kind != SExpr::Kind::Int) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, tv.items[i].loc,
                                        "(ints ...) elements must be integers"));
      }
      v.push_back(tv.items[i].i);
    }
    a.value = std::move(v);
  } else if (tag == "floats") {
    std::vector<double> v;
    for (size_t i = 1; i < tv.items.size(); ++i) {
      if (tv.items[i].kind != SExpr::Kind::Float) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, tv.items[i].loc,
                                        "(floats ...) elements must be floats"));
      }
      v.push_back(tv.items[i].f);
    }
    a.value = std::move(v);
  } else if (tag == "strings") {
    std::vector<std::string> v;
    for (size_t i = 1; i < tv.items.size(); ++i) {
      if (tv.items[i].kind != SExpr::Kind::String) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, tv.items[i].loc,
                                        "(strings ...) elements must be strings"));
      }
      v.push_back(tv.items[i].text);
    }
    a.value = std::move(v);
  } else if (tag == "bools") {
    std::vector<bool> v;
    for (size_t i = 1; i < tv.items.size(); ++i) {
      if (tv.items[i].kind != SExpr::Kind::Bool) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, tv.items[i].loc,
                                        "(bools ...) elements must be booleans"));
      }
      v.push_back(tv.items[i].b);
    }
    a.value = std::move(v);
  } else {
    return std::unexpected(sx_error(ErrorCategory::Syntax, tv.items[0].loc,
                                    "unknown attribute value tag '" + tag + "'"));
  }
  return a;
}

// --- nodes -----------------------------------------------------------------

ExpectedValueDef parse_value_def(const SExpr& e) {
  if (!is_list(e) || e.items.size() != 2) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                    "expected value definition (name <type>)"));
  }
  if (e.items[0].kind != SExpr::Kind::String) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.items[0].loc,
                                    "value name must be a string, got " +
                                        describe(e.items[0])));
  }
  auto t = parse_type(e.items[1]);
  if (!t) {
    return std::unexpected(t.error());
  }
  Node::Output o;
  o.name = e.items[0].text;
  o.type = *t;
  return o;
}

ExpectedNode parse_node(const SExpr& e) {
  // (node <op> (inputs <string>*) (outputs <value-def>*)
  //       [(attrs <attr>*)] [(domain <string>)] [(version <int>)])
  if (e.items.size() < 2 || e.items[1].kind != SExpr::Kind::Symbol) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                    "node requires a symbol op-name"));
  }
  Node n;
  n.op_name = e.items[1].text;

  bool saw_inputs = false;
  bool saw_outputs = false;
  bool saw_attrs = false;
  bool saw_domain = false;
  bool saw_version = false;

  for (size_t i = 2; i < e.items.size(); ++i) {
    const SExpr& sub = e.items[i];
    if (!is_list(sub) || sub.items.empty() || sub.items[0].kind != SExpr::Kind::Symbol) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                      "unexpected form in node: " + describe(sub)));
    }
    const std::string& head = sub.items[0].text;

    if (head == "inputs") {
      if (saw_inputs) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "duplicate (inputs ...) in node"));
      }
      if (saw_outputs || saw_attrs || saw_domain || saw_version) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(inputs ...) must precede (outputs ...)"));
      }
      for (size_t j = 1; j < sub.items.size(); ++j) {
        if (sub.items[j].kind != SExpr::Kind::String) {
          return std::unexpected(sx_error(ErrorCategory::Syntax, sub.items[j].loc,
                                          "node input must be a string, got " +
                                              describe(sub.items[j])));
        }
        n.inputs.push_back(sub.items[j].text);
      }
      saw_inputs = true;
    } else if (head == "outputs") {
      if (saw_outputs) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "duplicate (outputs ...) in node"));
      }
      if (!saw_inputs) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(outputs ...) must follow (inputs ...)"));
      }
      if (saw_attrs || saw_domain || saw_version) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(outputs ...) must precede (attrs ...)/(domain ...)/(version ...)"));
      }
      for (size_t j = 1; j < sub.items.size(); ++j) {
        auto vd = parse_value_def(sub.items[j]);
        if (!vd) {
          return std::unexpected(vd.error());
        }
        n.outputs.push_back(std::move(*vd));
      }
      saw_outputs = true;
    } else if (head == "attrs") {
      if (saw_attrs) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "duplicate (attrs ...) in node"));
      }
      if (!saw_outputs) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(attrs ...) must follow (outputs ...)"));
      }
      if (saw_domain || saw_version) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(attrs ...) must precede (domain ...)/(version ...)"));
      }
      for (size_t j = 1; j < sub.items.size(); ++j) {
        auto at = parse_attr(sub.items[j]);
        if (!at) {
          return std::unexpected(at.error());
        }
        n.attributes.push_back(std::move(*at));
      }
      saw_attrs = true;
    } else if (head == "domain") {
      if (saw_domain) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "duplicate (domain ...) in node"));
      }
      if (!saw_outputs) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(domain ...) must follow (outputs ...)"));
      }
      if (saw_version) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(domain ...) must precede (version ...)"));
      }
      if (sub.items.size() != 2 || sub.items[1].kind != SExpr::Kind::String) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(domain <string>) requires one string"));
      }
      n.domain = sub.items[1].text;
      saw_domain = true;
    } else if (head == "version") {
      if (saw_version) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "duplicate (version ...) in node"));
      }
      if (!saw_outputs) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(version ...) must follow (outputs ...)"));
      }
      if (sub.items.size() != 2 || sub.items[1].kind != SExpr::Kind::Int) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, sub.loc,
                                        "(version <int>) requires one integer"));
      }
      if (sub.items[1].i < 0) {
        return std::unexpected(sx_error(ErrorCategory::Semantic, sub.loc,
                                        "node version must be non-negative"));
      }
      n.version = sub.items[1].i;
      saw_version = true;
    } else {
      return std::unexpected(sx_error(ErrorCategory::Unsupported, sub.loc,
                                      "unsupported node form '" + head + "'"));
    }
  }

  if (!saw_inputs || !saw_outputs) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                    "node requires (inputs ...) and (outputs ...)"));
  }
  return n;
}

// --- graph -----------------------------------------------------------------

ExpectedGraph parse_graph(const SExpr& e) {
  // (graph (name <string>) (opset <string> <int>)*
  //        (inputs <input>*) (outputs <output>*)
  //        (parameters <parameter>*) (nodes <node>*))
  Graph g;
  size_t i = 1;

  auto next = [&]() -> const SExpr* {
    return (i < e.items.size()) ? &e.items[i] : nullptr;
  };

  // name (required, first)
  {
    const SExpr* s = next();
    if (!s || !form(*s, "name")) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                      "graph requires (name <string>) as its first section"));
    }
    if (s->items.size() != 2 || s->items[1].kind != SExpr::Kind::String) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, s->loc,
                                      "(name <string>) requires one string"));
    }
    g.name = s->items[1].text;
    ++i;
  }

  // opset*
  for (;;) {
    const SExpr* s = next();
    if (!s || !form(*s, "opset")) {
      break;
    }
    if (s->items.size() != 3 || s->items[1].kind != SExpr::Kind::String ||
        s->items[2].kind != SExpr::Kind::Int) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, s->loc,
                                      "(opset <string> <int>) requires a string and an integer"));
    }
    if (s->items[2].i < 0) {
      return std::unexpected(sx_error(ErrorCategory::Semantic, s->loc,
                                      "opset version must be non-negative"));
    }
    Graph::Opset o;
    o.domain = s->items[1].text;
    o.version = s->items[2].i;
    g.opsets.push_back(std::move(o));
    ++i;
  }

  // inputs (required)
  {
    const SExpr* s = next();
    if (!s || !form(*s, "inputs")) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                      "graph requires (inputs ...)"));
    }
    for (size_t j = 1; j < s->items.size(); ++j) {
      const SExpr& in = s->items[j];
      if (!is_list(in) || in.items.size() != 3 || !in.items[0].is_symbol("input")) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, in.loc,
                                        "expected (input <string> <type>)"));
      }
      if (in.items[1].kind != SExpr::Kind::String) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, in.items[1].loc,
                                        "input name must be a string"));
      }
      auto t = parse_type(in.items[2]);
      if (!t) {
        return std::unexpected(t.error());
      }
      Graph::Input gi;
      gi.name = in.items[1].text;
      gi.type = *t;
      g.inputs.push_back(std::move(gi));
    }
    ++i;
  }

  // outputs (required)
  {
    const SExpr* s = next();
    if (!s || !form(*s, "outputs")) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                      "graph requires (outputs ...)"));
    }
    for (size_t j = 1; j < s->items.size(); ++j) {
      const SExpr& out = s->items[j];
      if (!is_list(out) || out.items.size() != 2 || !out.items[0].is_symbol("output")) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, out.loc,
                                        "expected (output <string>)"));
      }
      if (out.items[1].kind != SExpr::Kind::String) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, out.items[1].loc,
                                        "output name must be a string"));
      }
      g.outputs.push_back(out.items[1].text);
    }
    ++i;
  }

  // parameters (required)
  {
    const SExpr* s = next();
    if (!s || !form(*s, "parameters")) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                      "graph requires (parameters ...)"));
    }
    for (size_t j = 1; j < s->items.size(); ++j) {
      const SExpr& p = s->items[j];
      if (!is_list(p) || p.items.size() != 4 || !p.items[0].is_symbol("parameter")) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, p.loc,
                                        "expected (parameter <string> <type> <data>)"));
      }
      if (p.items[1].kind != SExpr::Kind::String) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, p.items[1].loc,
                                        "parameter name must be a string"));
      }
      auto t = parse_type(p.items[2]);
      if (!t) {
        return std::unexpected(t.error());
      }
      auto d = parse_data(p.items[3]);
      if (!d) {
        return std::unexpected(d.error());
      }
      Parameter param;
      param.name = p.items[1].text;
      param.type = *t;
      param.data = std::move(*d);
      g.parameters.push_back(std::move(param));
    }
    ++i;
  }

  // nodes (required)
  {
    const SExpr* s = next();
    if (!s || !form(*s, "nodes")) {
      return std::unexpected(sx_error(ErrorCategory::Syntax, e.loc,
                                      "graph requires (nodes ...)"));
    }
    for (size_t j = 1; j < s->items.size(); ++j) {
      const SExpr& nd = s->items[j];
      if (!is_list(nd) || nd.items.empty() || !nd.items[0].is_symbol("node")) {
        return std::unexpected(sx_error(ErrorCategory::Syntax, nd.loc,
                                        "expected (node <op> ...)"));
      }
      auto node = parse_node(nd);
      if (!node) {
        return std::unexpected(node.error());
      }
      g.nodes.push_back(std::move(*node));
    }
    ++i;
  }

  // no extra sections
  if (i != e.items.size()) {
    const SExpr& extra = e.items[i];
    return std::unexpected(sx_error(ErrorCategory::Unsupported, extra.loc,
                                    "unsupported graph section '" +
                                        head_name(extra) + "'"));
  }

  return g;
}

// --- document --------------------------------------------------------------

ExpectedDocument build(const SExpr& root) {
  if (root.kind != SExpr::Kind::List) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, root.loc,
                                    "document must be a list"));
  }
  if (root.items.empty()) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, root.loc,
                                    "empty document"));
  }
  const SExpr& head = root.items[0];
  if (!head.is_symbol("sonicboom-s-expr")) {
    return std::unexpected(sx_error(ErrorCategory::Unsupported, head.loc,
                                    "unknown top-level form '" + head_name(head) +
                                        "' (expected sonicboom-s-expr)"));
  }
  if (root.items.size() < 3) {
    return std::unexpected(sx_error(
        ErrorCategory::Syntax, root.loc,
        "expected (sonicboom-s-expr (version <major> <minor>) (graph ...))"));
  }
  if (root.items.size() > 3) {
    const SExpr& extra = root.items[3];
    return std::unexpected(sx_error(ErrorCategory::Unsupported, extra.loc,
                                    "unsupported top-level form '" +
                                        head_name(extra) + "'"));
  }

  const SExpr& ver = root.items[1];
  if (!form(ver, "version") || ver.items.size() != 3) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, ver.loc,
                                    "expected (version <major> <minor>)"));
  }
  if (ver.items[1].kind != SExpr::Kind::Int || ver.items[2].kind != SExpr::Kind::Int) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, ver.loc,
                                    "version numbers must be integers"));
  }
  Document doc;
  doc.version_major = ver.items[1].i;
  doc.version_minor = ver.items[2].i;
  if (doc.version_major != 0 || doc.version_minor > 1) {
    return std::unexpected(sx_error(
        ErrorCategory::Unsupported, ver.loc,
        "unsupported S-Expr version " + std::to_string(doc.version_major) + "." +
            std::to_string(doc.version_minor) + " (supported: 0.x with x <= 1)"));
  }

  const SExpr& g = root.items[2];
  if (!is_list(g) || g.items.empty() || !g.items[0].is_symbol("graph")) {
    return std::unexpected(sx_error(ErrorCategory::Syntax, g.loc,
                                    "expected (graph ...)"));
  }
  auto graph = parse_graph(g);
  if (!graph) {
    return std::unexpected(graph.error());
  }
  doc.graph = std::move(*graph);
  return doc;
}

// --- validation ------------------------------------------------------------

// Cross-cutting semantic rules (spec §14): single SSA definition, name
// resolution, topological order, and opset-domain declaration. These need the
// whole graph and are checked after the Document is built, so their errors
// carry no source location.
std::optional<Error> validate(const Document& doc) {
  const Graph& g = doc.graph;
  // name -> producer node index; -1 for a graph input or parameter.
  std::unordered_map<std::string, int64_t> defs;
  defs.reserve(g.inputs.size() + g.parameters.size() + g.nodes.size() * 2 + 4);

  auto define = [&](const std::string& name, int64_t idx) -> std::optional<Error> {
    auto [it, inserted] = defs.emplace(name, idx);
    if (!inserted) {
      return Error{ErrorCategory::Semantic, SourceLocation{},
                   "duplicate definition of value '" + name + "'"};
    }
    return std::nullopt;
  };

  for (const auto& in : g.inputs) {
    if (auto e = define(in.name, -1)) {
      return e;
    }
  }
  for (const auto& p : g.parameters) {
    if (auto e = define(p.name, -1)) {
      return e;
    }
  }

  for (size_t i = 0; i < g.nodes.size(); ++i) {
    const Node& n = g.nodes[i];
    for (const auto& in_name : n.inputs) {
      auto it = defs.find(in_name);
      if (it == defs.end()) {
        return Error{ErrorCategory::Semantic, SourceLocation{},
                     "node '" + n.op_name + "' references unknown value '" +
                         in_name + "'"};
      }
      if (it->second >= 0 && it->second >= static_cast<int64_t>(i)) {
        return Error{ErrorCategory::Semantic, SourceLocation{},
                     "node '" + n.op_name + "' consumes value '" + in_name +
                         "' before its producer (violates topological order)"};
      }
    }
    for (const auto& out : n.outputs) {
      if (auto e = define(out.name, static_cast<int64_t>(i))) {
        return e;
      }
    }

    if (n.domain != "default") {
      bool declared = false;
      for (const auto& o : g.opsets) {
        if (o.domain == n.domain) {
          declared = true;
          break;
        }
      }
      if (!declared) {
        return Error{ErrorCategory::Semantic, SourceLocation{},
                     "node '" + n.op_name + "' uses domain '" + n.domain +
                         "' that is not declared by (opset ...)"};
      }
    }
  }

  for (const auto& out_name : g.outputs) {
    if (defs.find(out_name) == defs.end()) {
      return Error{ErrorCategory::Semantic, SourceLocation{},
                   "graph output references unknown value '" + out_name + "'"};
    }
  }
  return std::nullopt;
}

} // namespace

std::expected<Document, Error> parse_document(std::string_view text) {
  detail::Lexer lexer(text);
  auto tokens = lexer.tokenize();
  if (!tokens) {
    return std::unexpected(tokens.error());
  }

  detail::Reader reader(*tokens);
  auto root = reader.read_one();
  if (!root) {
    return std::unexpected(root.error());
  }
  if (!reader.at_end()) {
    return std::unexpected(detail::sx_error(ErrorCategory::Syntax,
                                            reader.peek_location(),
                                            "unexpected trailing form after document"));
  }

  auto doc = build(*root);
  if (!doc) {
    return std::unexpected(doc.error());
  }

  if (auto err = validate(*doc)) {
    return std::unexpected(*err);
  }

  return std::move(*doc);
}

} // namespace sonicboom::sx
