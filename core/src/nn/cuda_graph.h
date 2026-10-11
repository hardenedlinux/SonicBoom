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

#include <cuda_runtime.h>

#include <cstdio>

// Minimal CUDA Graph capture/replay wrapper for one decode step. The caller
// begins capture, runs the step body on the returned stream (via
// sonicboom::cuda::StreamScope), then ends capture and replays the instantiated
// graph once per step.
//
// NOT part of the public Layer 2 surface: lives under core/src/ and is included
// only from translation units compiled with <cuda_runtime.h> available (the
// CUDA spine in core/src/model/gemma4_plan.cpp).
namespace sonicboom::cuda {

class GraphStep {
 public:
  GraphStep() = default;
  ~GraphStep() { destroy(); }
  GraphStep(const GraphStep&) = delete;
  GraphStep& operator=(const GraphStep&) = delete;

  // Create the internal stream and begin capture (thread-local mode: only this
  // thread's work on the stream is captured). Returns false on failure.
  bool begin() {
    if (cudaStreamCreate(&stream_) != cudaSuccess) return false;
    if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess) {
      destroy();
      return false;
    }
    return true;
  }

  // End capture and instantiate the graph. On any failure the stream/graph are
  // torn down and the wrapper becomes invalid (callers fall back to the
  // per-step dispatch path). Returns false on failure.
  bool end() {
    cudaError_t e = cudaStreamEndCapture(stream_, &graph_);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "[cuda-graph] end capture failed: %s\n",
                   cudaGetErrorString(e));
      destroy();
      return false;
    }
    e = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "[cuda-graph] instantiate failed: %s\n",
                   cudaGetErrorString(e));
      destroy();
      return false;
    }
    return true;
  }

  // Launch the captured graph on the internal stream (async).
  bool replay() { return cudaGraphLaunch(graph_exec_, stream_) == cudaSuccess; }

  // Block until the stream's work (the last replay) completes.
  bool sync() { return cudaStreamSynchronize(stream_) == cudaSuccess; }

  cudaStream_t stream() const { return stream_; }
  bool valid() const { return graph_exec_ != nullptr; }

 private:
  void destroy() {
    if (graph_exec_) {
      cudaGraphExecDestroy(graph_exec_);
      graph_exec_ = nullptr;
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
};

}  // namespace sonicboom::cuda
