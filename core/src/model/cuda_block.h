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

#include <cstdint>
#include <span>

#include <sonicboom/model/loader.h>

// Internal device-resident single-token forward (Phase 6a Option A). Runs all
// blocks with activations kept on the device and synchronizes only at the end,
// mirroring run_block_core's arithmetic (see core/src/model/exec.cpp). The
// signature is CUDA-free so the host execution layer (exec.cpp) can call it;
// the CUDA runtime and kernels live in core/src/model/cuda_block.cu.
//
// NOT part of the public Layer 2 surface (lives under core/src/).

namespace sonicboom::model {

// Full single-token forward on the device: equivalent to
// forward(m, token_id, pos, out, quant::MatmulBackend::Cuda) with resident
// activations (no per-op host<->device copies). `out` (length embedding_length)
// receives the last block's l_out. Returns false when CUDA is unavailable, the
// model is empty, out is too short, or any device step fails.
bool cuda_forward_resident(const Gemma4Model& m, uint64_t token_id, uint64_t pos,
                           std::span<float> out);

// Free the device activation pool (call alongside nn::cuda::clear_cache() and
// quant::cuda::clear_cache() when a model is reloaded at a new address).
void cuda_resident_clear();

} // namespace sonicboom::model
