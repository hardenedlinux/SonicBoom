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
// This is the entry point for a caller that holds the IR. A caller that holds a
// planner::Graph instead runs the same passes through
// planner::ir_pass_pipeline (planner/ir_passes.h), which is a graph in and a
// graph out; both end in the one pass list in BoomPasses.cpp, so the two cannot
// disagree about what a phase does.
//
// Ablation: SONICBOOM_IR_PASS_DISABLE=<comma-separated pass names> skips those
// passes for the run. The names and their meaning are documented on
// planner::ir_pass_pipeline, where the switch's readers look for them.

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
bool runBoomPasses(mlir::ModuleOp module, planner::IrPassPhase phase);

} // namespace sonicboom::compiler
