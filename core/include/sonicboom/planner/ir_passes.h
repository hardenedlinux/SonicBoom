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
// MLIR passes over Boom IR instead (design/mlir-migration.md §7c).
//
// A phase's pipeline has two implementations that are required to leave the
// *same* graph — the same nodes, in the same order, over the same tensor table
// (BoomConversion.h states why that is the whole contract). The plan goldens in
// tests/data/plans pin it, and it is what makes this a port rather than a second
// set of rewrites. During the migration the two implementations are mixed: the
// passes whose port has landed run as Boom IR passes, and the rest still run as
// the Graph rewrites they were, at one boundary in one place (BoomPasses.cpp).
//
// The header carries no MLIR: the choice of pipeline is a switch a runtime
// caller makes, and nothing on the runtime side has to name a compiler type to
// make it. The pass themselves live in the compiler library, behind Layer 2.

#pragma once

#include <sonicboom/planner/graph.h>

namespace sonicboom::planner {

// Which phase's pipeline to run. The two the emitter builds differ in their
// pass list and nothing else (design/mlir-migration.md §2 lists both).
enum class IrPassPhase { Decode, Prefill };

// Whether this build carries the Boom IR passes at all: they live in the
// compiler library, which is built and linked only when the build has MLIR
// (SONICBOOM_ENABLE_MLIR; core/compiler/CMakeLists.txt).
bool ir_passes_available();

// Whether the runtime should derive the plan through Boom IR:
// SONICBOOM_IR_PASSES set to anything but "0" (and the build has them). Read on
// every call rather than cached — the flag is a live toggle, so a differential
// test can turn it on and off inside one process and compare the two pipelines
// over the same model, the same reason quant::cuda::q8_1_enabled reads its own
// env flag live (cuda_quant_matmul.cu:1311). Off by default: the Graph rewrites
// are what libsonicboom.so runs unless the environment asks otherwise.
bool ir_passes_enabled();

// Run `phase`'s rewrite pipeline over `graph` through Boom IR: the graph is
// converted, the phase's landed passes run as MLIR passes, and the result is
// converted back and handed to the passes whose port has not landed. The
// pipeline therefore always implements the whole phase, whichever half a pass
// currently lives in.
//
// A conversion or pass failure is reported on stderr and leaves `graph` as it
// was handed over — unfused, which is correct but not what the phase's pipeline
// would have produced. There is deliberately no fallback to the Graph rewrites:
// a silent one would make a broken IR pass indistinguishable from a working one,
// and the plan comparison that exists to catch that would pass.
void ir_pass_pipeline(Graph &graph, IrPassPhase phase);

} // namespace sonicboom::planner
