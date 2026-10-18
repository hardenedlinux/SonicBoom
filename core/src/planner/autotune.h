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

// One autotune service for the device kernels that have a per-shape-class
// configuration choice (attention tiles, MMQ J, ...): a disk cache keyed by
// GPU + op + shape class, shared by every tuner, plus the timing loop they
// all use.
//
// Cache location:
//   $SONICBOOM_TUNE_DIR, else $XDG_CACHE_HOME/sonicboom/tune, else
//   $HOME/.cache/sonicboom/tune
//   /<gpu-name>-sm<arch>-<driver>/<op>.json
// Each file is a JSON object of shape-class -> object-of-int. A missing or
// unreadable file is an empty cache; the first tuner run rewrites the whole
// file. Choices are per (op, shape class, GPU), never per batch size or
// position: the same shape class must resolve the same way for a reused
// prefix and a full prefill.
//
// Switches (ablation groups):
//   SONICBOOM_AUTOTUNE=0      use the static per-class defaults, never tune
//   SONICBOOM_AUTOTUNE=force  ignore cached choices and retune every class
//   unset                     use cached choices, tune on a miss
// The tuner only runs outside CUDA-graph capture (host timing is illegal
// there); a capture uses the cached/default choice.
//
// Layer placement: internal runtime machinery (core/src/planner), not part
// of the Layer 2 public headers.

#ifdef SONICBOOM_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <functional>
#include <map>
#include <string>
#include <string_view>

namespace sonicboom::tune {

// SONICBOOM_AUTOTUNE=0 disables tuning (the static defaults stay in force).
bool enabled();

// SONICBOOM_AUTOTUNE=force: ignore the disk cache, retune every class.
bool force();

// The device directory name (<gpu-name>-sm<arch>-<driver>). Computed once.
// "cpu" when the build has no CUDA support.
const std::string &device_dir();

// Full path of an op's cache file (for diagnostics/reporting).
std::string cache_path(std::string_view op);

// Cached choice for (op, key); empty when absent or when SONICBOOM_AUTOTUNE=0.
// `force()` makes this always return empty so the caller retunes.
std::map<std::string, int> lookup(std::string_view op, std::string_view key);

// Store a choice, updating the in-memory map and rewriting `<op>.json`.
// A no-op when tuning is disabled.
void store(std::string_view op, std::string_view key,
           const std::map<std::string, int> &values);

// Drop the in-memory maps so the next lookup re-reads from disk.
void clear_memory_cache();

#ifdef SONICBOOM_USE_CUDA
// Median of `reps` individually timed launches after `warmups` untimed ones.
// The events are recorded on `stream`, which must be the stream the launch
// lambda uses (the caller's current stream). Returns a negative value when no
// valid measurement could be taken (event failure, launch failure); the caller
// keeps its default then.
double median_launch_ms(const std::function<bool()> &launch, cudaStream_t stream,
                        int warmups = 2, int reps = 5);
#else
double median_launch_ms(const std::function<bool()> &launch, int warmups = 2,
                        int reps = 5);
#endif

} // namespace sonicboom::tune
