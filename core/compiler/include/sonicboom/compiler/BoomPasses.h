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

// The Boom IR pass pipeline's entry point (design/mlir-migration.md §7c): one
// phase's passes, run over a module the emitter built.
//
// A plan site holds the IR — it builds the module, runs this, and lowers the
// result — so there is no graph-level entry: the passes read Boom IR, and a
// caller that has only a planner::Graph has nothing they accept.
//
// Ablation: SONICBOOM_IR_PASS_DISABLE=<comma-separated pass names> skips those
// passes for the run. The names and their meaning are documented on
// planner::IrPassPhase (planner/ir_passes.h); the switch itself is read in
// BoomPasses.cpp, where the passes and their names live.

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <sonicboom/planner/ir_passes.h>

namespace sonicboom::compiler {

// Runs `phase`'s passes over the module's boom.graph op. Returns false — with a
// diagnostic on the context and a message on stderr — when a pass fails; the IR
// it leaves behind is then the failing pass's output, not the phase's, so the
// caller must not lower it as if the pipeline had run.
//
// The module is verified after every pass by the pass manager, so a pass that
// leaves the IR in a state the dialect rejects fails here, naming that pass,
// rather than three passes later.
//
// `skip_kv_append_fusion` leaves the K/V append on the attention node instead
// of folding it into the prep regions. The verify step (B3) needs that: its
// attention appends one row per query position, which a region that stores a
// single row at the base position cannot express.
bool runBoomPasses(mlir::ModuleOp module, planner::IrPassPhase phase,
                   bool skip_kv_append_fusion = false);

} // namespace sonicboom::compiler
