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

// The Boom IR pass pipeline's entry point (docs/boom-ir.md): one
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
//
// `fuse_silu_mul` enables FuseSiluMulPass: the silu->mul chain that feeds a
// dense projection becomes one generated region. It is the caller's numerics
// decision, not a compiler one — the generated region's exp is covered by the
// standard profile's acceptance gate, while strict keeps the two reference
// kernels its plan goldens record.
//
// `rows_mmq` enables the mmq producer fusions in the decode/verify pipeline:
// the matmuls of those graphs carry the prefill lowering's `batch`/`dim`/`ds`
// attributes (the emitter writes them when the decode step runs on the mmq
// matmul), and the two fusions then fold the norm and the gated activation into
// the mmq activation producers at this step's row count, resizing the
// activation buffers to the mmq layout (decode buffers are smaller than the
// mmq form, so the fusion is allowed to grow them).
bool runBoomPasses(mlir::ModuleOp module, planner::IrPassPhase phase,
                   bool skip_kv_append_fusion = false,
                   bool fuse_silu_mul = false, bool rows_mmq = false);

// Resolve the consumer a graph declares (BoomOps.td, GraphOp::consumer) into
// the graph that consumer needs. Run on the emitter's IR before the phase's
// passes — and before lowering a graph the passes do not run on. It:
//
//   * runs the demand analysis (BoomDemand.h) from the declared consumer;
//   * removes what the consumer does not need — a monotone map whose result is
//     only ranked, a node nothing reads (ablation switch: PruneByDemandPass);
//   * narrows what is read only on a runtime index list to the list's
//     capacity, marking the weight matmul that produces it `gather`;
//   * appends the argmax of an "argmax" consumer.
//
// A "sample_select" consumer stays declared for finishConsumer; every other
// declaration is gone when this returns. A module with no declaration is left
// alone, so every caller may run it. False, with a diagnostic, on failure.
bool resolveConsumer(mlir::ModuleOp module);

// The second half, after the phase's passes: append the candidate selection of
// a "sample_select" consumer to the finished graph and drop the declaration.
// The selection reads the logits buffer the passes laid out; it is added last
// so that it takes no part in their buffer assignment. A module that declares
// nothing is left alone. False when a declaration resolveConsumer should have
// consumed is still present.
bool finishConsumer(mlir::ModuleOp module);

// Run BufferReusePass alone over every boom.graph in `module`, so a caller can
// see the `inplace`/`buf` marks a full pipeline would produce on that node
// list. Analysis tooling only (boom-r4 F2): the runtime never calls it — the
// phase pipeline does — and a caller that runs it should do so on a clone, the
// way the liveness comparison does.
bool markBufferReuse(mlir::ModuleOp module);

// The MMQ1 prefill dispatch switch (prefill_dispatch_pass), read from
// SONICBOOM_MMQ_TC:
//   off (default) — never rewrite: the ggml mmq path. This is the default
//                   because the generated kernel measured slower than the ggml
//                   mmq kernel on the reference GPU (Qwen2.5-0.5B pp512:
//                   17.0k tok/s off vs 14.1k on at the best tile), so opting in
//                   must be deliberate until the tile autotuner has a winner.
//   auto          — rewrite a batched matmul to the generated int8 tensor-core
//                   kernel when the whole graph's per-weight q8_1 copy fits the
//                   budget (the Gemma 4 E4B decision on a large-VRAM machine).
//   on            — rewrite regardless of the budget (for verifying the path on
//                   a machine with the VRAM headroom; the runtime still falls
//                   back per node if a copy cannot be built).
// Anything else is reported on stderr and read as off.
enum class MmqTcMode { Auto, On, Off };
MmqTcMode mmq_tc_mode();

// The q8_1 weight-copy budget in MiB (SONICBOOM_MMQ_TC_MAX_MB, default 2048).
// The copies are additional to the K-quant weights already resident, so this is
// the VRAM a dispatch decision may spend, not the model size.
int mmq_tc_max_mb();

// The VS1 async-prefetch switch (AsyncPrefetchPass), read from
// SONICBOOM_ASYNC_PREFETCH:
//   on  — insert boom.async_weight_prefetch / boom.sync_prefetch markers when
//         the graph's model-bound weights exceed the VRAM threshold;
//   off — the default; no marker is inserted.
// Unset reads as off. Anything else is reported on stderr and read as off.
enum class AsyncPrefetchMode { On, Off };
AsyncPrefetchMode async_prefetch_mode();

// The model-weight footprint above which the markers are inserted, in MiB
// (SONICBOOM_VRAM_THRESHOLD_MB, default 6000). The default is the model size
// that no longer leaves the decode working set comfortable on an 8 GiB device.
int vram_threshold_mb();

} // namespace sonicboom::compiler
