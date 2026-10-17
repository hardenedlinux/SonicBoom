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

// The Boom IR rewrite pipeline: one phase's Graph-level fusion passes, run as
// MLIR passes over Boom IR (design/mlir-migration.md §7c).
//
// This is the only rewrite path. The passes were ported from the planner's
// Graph rewrites, one pass at a time, until the port was complete — the plan
// goldens in tests/data/plans pinned each one — and the Graph rewrites and the
// switch that chose between the two were then removed. A pass is therefore
// either a registered Boom IR pass or it does not exist; there is no second
// implementation for a behavior to diverge from.
//
// The header carries no MLIR: nothing on the runtime side has to name a
// compiler type to run the pipeline. The passes themselves live in the compiler
// library, behind Layer 2.

#pragma once

#include <sonicboom/planner/graph.h>

namespace sonicboom::planner {

// Which phase's pipeline to run. The two the emitter builds differ in their
// pass list and nothing else (design/mlir-migration.md §2 lists both).
enum class IrPassPhase { Decode, Prefill };

// Run `phase`'s rewrite pipeline over `graph` through Boom IR: the graph is
// converted, the phase's passes run as MLIR passes, and the result is converted
// back.
//
// Ablation: SONICBOOM_IR_PASS_DISABLE=<comma-separated pass names> skips those
// passes for the run, one fusion at a time, so the plan diff can say what each
// contributes. The names are the pass class names in BoomPasses.cpp —
// QuantizeOncePass, RmsNormQuantizePass, GeluMulQuantizePass,
// SiluMulQuantizePass, BiasIntoGemvPass, ResidualAddIntoGemvPass (decode) and
// RmsNormMmqPass, GeluMulMmqPass, SiluMulMmqPass (prefill); a phase that does
// not contain a named pass simply ignores it. A disabled pass can also remove a
// later pass's fusion — the passes run in sequence over one graph, and the
// Quantize nodes QuantizeOncePass inserts are what RmsNormQuantizePass and
// GeluMulQuantizePass fold away — so a plan delta is read as what that pass and
// the folds it feeds contribute, not as that pass alone. An unrecognized name
// is reported on stderr with the valid names and disables nothing — a typo must
// not pass for an ablation. The switch is read in the pipeline itself, where the
// passes and their names live (BoomPasses.cpp).
//
// A conversion or pass failure is reported on stderr and leaves `graph` as it
// was handed over — unfused, which is correct but not what the phase's pipeline
// would have produced. There is deliberately no fallback: a silent one would
// make a broken IR pass indistinguishable from a working one, and the plan
// comparison that exists to catch that would pass.
void ir_pass_pipeline(Graph &graph, IrPassPhase phase);

} // namespace sonicboom::planner
