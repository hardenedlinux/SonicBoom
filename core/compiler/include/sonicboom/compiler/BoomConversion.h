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

// planner::Graph <-> Boom IR (design/mlir-migration.md §7(a)). The conversion is
// a translation, not a transformation: what comes back must be the graph that
// went in — the same tensors, the same node ids, the same attributes in the same
// order — because the model fingerprint hashes all of it in that order
// (planner/graph_adapter.cpp) and every execution plan is derived from it. A
// conversion that "improves" anything is a conversion that changes plans.
//
// The conversion is also the place where the IR's rules are *met* rather than
// assumed: the graph op is verified before it is handed back, so a conversion
// that produced an IR the verifier rejects fails here, at the conversion, instead
// of surfacing later as a mystery in a pass.

#pragma once

#include "sonicboom/compiler/BoomOps.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <sonicboom/planner/graph.h>

#include <optional>
#include <string>

namespace sonicboom::compiler {

// Graph -> Boom IR: the graph becomes one boom.graph op inside a module (the op
// is IsolatedFromAbove, so it needs a top level to sit at) whose body block
// arguments are the graph's free variables in ascending tensor-id order.
//
// Returns an empty module, with a diagnostic on `context`, when some part of the
// graph has no Boom IR form (a tensor that is neither a Boom value type nor
// representable as a q8_1 layout, an attribute value that is not one of
// sx::Attribute's kinds) or when the resulting IR fails verification.
mlir::OwningOpRef<mlir::ModuleOp> graphToBoom(const planner::Graph &graph,
                                              mlir::MLIRContext *context);

// Boom IR -> Graph. `module` must hold exactly one boom.graph op, and that op
// must be the one `graphToBoom` builds: this direction reads the tensor table,
// `result_ids`, `node_name`, `node_id`, `node_attrs` and the optional `backend`
// back out, and refuses (returning nullopt after a diagnostic) rather than
// inventing a field the IR does not carry.
//
// A dynamic dimension (`?`) is the graph's *extent*, the prefill segment size.
// `dynamic_extent` binds it: every `?` becomes that value, and the tensor's
// byte size is computed from the bound shape. The resulting TensorDesc records
// the extent through dynamic_stride_bytes, so a later execution can bind a
// smaller extent to the same plan. A module with a dynamic dimension and a zero
// extent is refused (there is no concrete graph to plan).
std::optional<planner::Graph> boomToGraph(mlir::ModuleOp module,
                                          uint64_t dynamic_extent = 0);

// The IR as text, for inspection and for diffing two conversions. The ops carry
// no custom assembly format, so this is MLIR's generic form: verbose, but
// unambiguous and parseable back.
std::string dumpBoom(mlir::Operation *op);

} // namespace sonicboom::compiler
