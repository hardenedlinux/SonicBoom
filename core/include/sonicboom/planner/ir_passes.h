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

// Which Boom IR pipeline a plan is built through (design/mlir-migration.md §7c).
//
// This is the only rewrite path. The passes were ported from the planner's
// Graph rewrites, one pass at a time, until the port was complete — the plan
// goldens in tests/data/plans pinned each one — and the Graph rewrites and the
// switch that chose between the two were then removed. A pass is therefore
// either a registered Boom IR pass or it does not exist; there is no second
// implementation for a behavior to diverge from.
//
// The passes themselves, and the entry that runs them, live in the compiler
// library (compiler/BoomPasses.h runBoomPasses): a plan site emits Boom IR, runs
// the phase's passes over it, and lowers the result — there is no graph-level
// entry, because a graph is not what the passes read any more. What is left here
// is the one thing a plan site has to name without naming MLIR: the phase.
//
// Ablation: SONICBOOM_IR_PASS_DISABLE=<comma-separated pass names> skips those
// passes for the run, one fusion at a time, so the plan diff can say what each
// contributes. The names are the pass class names in BoomPasses.cpp —
// QuantizeOncePass, RmsNormQuantizePass, GeluMulQuantizePass,
// SiluMulQuantizePass, BiasIntoGemvPass, ResidualAddIntoGemvPass,
// FuseResidualEpiloguePass (decode) and
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
// A pass failure fails the plan: the IR is not lowered as if the phase had run,
// because an unfused graph planned as though it were fused is a slower plan
// reported as the pipeline's, not an error.

#pragma once

namespace sonicboom::planner {

// Which phase's pipeline to run. The two the emitter builds differ in their
// pass list and nothing else (design/mlir-migration.md §2 lists both).
//
// Verify is the speculative-decoding verify step (B3): the decode pipeline
// except that FuseKvAppendIntoPrep leaves the KV append on the attention node.
// The verify graph deals one decode over k+1 query positions, and its attention
// appends one row per position, which a prep region that stores a single row at
// the base position cannot express.
//
// Training is the pipeline for a traced forward+backward graph
// (core/src/training/device_graph.cpp): the training vocabulary has none of the
// inference folds' inputs, so the phase carries only the passes that are
// semantic-neutral for it — today BufferReusePass, whose in-place marks the
// device executor may honour but need not.
enum class IrPassPhase { Decode, Prefill, Verify, Training };

} // namespace sonicboom::planner
