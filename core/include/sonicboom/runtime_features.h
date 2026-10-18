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

// Runtime features that change performance and can be turned off one at a time
// for ablation:
//
//   SONICBOOM_FEATURE_DISABLE=cuda_graph_decode,split_kv,...
//
// The list names features (see kFeatureNames in runtime_features.cpp); an
// unknown name is a hard error rather than a silent no-op, so a typo cannot be
// read as "no effect". The pre-existing single-purpose variables keep working
// as aliases of these features (SONICBOOM_CUDA_GRAPHS=0 disables both CUDA-graph
// captures, SONICBOOM_GEMV_F32 disables the q8_1 dispatch, and so on).
//
// The switch is process-global and read once: set it before the library builds
// its first runtime.

namespace sonicboom::runtime {

enum class Feature {
  // Capture/replay of one greedy decode step (CUDA graph).
  CudaGraphDecode,
  // Capture/replay of the full-chunk prefill segment (CUDA graph).
  CudaGraphPrefill,
  // Reuse a cached prefill capture instead of recapturing it.
  CaptureReuse,
  // Split-KV decode attention (n_chunks partials + merge). Off = one block per
  // KV head over the whole window.
  SplitKv,
  // Cache prefill segment plans by segment length.
  PlanCache,
  // Lower prefill segments from symbolic IR and bind the segment length at
  // execute time (off = one concrete plan per exact segment length).
  SymbolicPrefill,
  // q8_1 activation quantization + dp4a/MMQ dispatch (off = fp32 GEMV).
  Q8_1Dispatch,
  // Return an activation's device buffer to the pool as soon as its last
  // consumer has read it (off = hold every buffer until the step ends).
  BufferReuse,
  // Run the one-time decode-runtime warm-up before the timed region.
  Warmup,
  // Fuse decode glue regions into compiler-generated kernels: the Boom IR pass
  // emits `fused_region` nodes and the CUDA backend runs the MLIR->NVVM->PTX
  // kernel (off = the region keeps its hand-written kernel).
  GlueCodegen,
  // Measure the mmq candidates (J tile x decomposition) on the real call and
  // cache the fastest per weight-shape class, (quant type, M, blocks per row)
  // (off = select_J's min-tile heuristic + the tiled class policy).
  Autotune,
};

// True unless the feature is named in SONICBOOM_FEATURE_DISABLE or turned off
// by its alias variable. Aborts with a diagnostic naming the valid features if
// the disable list names an unknown feature.
bool feature_enabled(Feature feature) noexcept;

// Convenience inverse of feature_enabled().
bool feature_disabled(Feature feature) noexcept;

// The canonical feature names, for diagnostics and tests. Returns nullptr when
// `index` is past the end. The returned pointer has static lifetime.
const char* feature_name(Feature feature) noexcept;

} // namespace sonicboom::runtime
