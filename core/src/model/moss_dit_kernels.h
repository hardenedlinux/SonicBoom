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

// CUDA entry points for the MOSS-Image DiT, in the C++20 dialect nvcc accepts
// (CUDA 12.0 has no -std=c++23). The host orchestration in moss_dit_cuda.cpp
// (C++23: safetensors Reader, std::expected) calls these; the kernels and the
// CUDA API stay in moss_dit_kernels.cu. All activation pointers are device
// pointers; `total` counts floats, not rows.
//
// NOT part of the public Layer 2 surface: lives under core/src/.

#include <cstddef>
#include <cstdint>

namespace sonicboom::moss::cuda_dit {

bool device_available();

// Device memory and copies. `host` for upload_pinned is a cudaHostAlloc
// pointer; upload is asynchronous on `stream` (a stream_handle).
void* device_alloc(size_t bytes);
void device_free(void* p);
bool upload(void* dev, const void* host, size_t bytes);
bool download(void* host, const void* dev, size_t bytes);
bool copy_d2d(void* dst, const void* src, size_t bytes);
bool upload_async(void* dev, const void* pinned_host, size_t bytes,
                  void* stream);

void* pinned_alloc(size_t bytes);
void pinned_free(void* p);
void* stream_create();
void stream_destroy(void* stream);
bool stream_sync(void* stream);
// Returns the last launch error and clears it (empty string when clean).
const char* last_error();

bool rms_norm_rows(const float* x, const float* w, float eps, float* y,
                   int dim, long rows);
bool rms_norm_heads(float* v, const float* w, float eps, int head_dim,
                    long rows_heads);
bool modulate(float* y, const float* shift, const float* scale, int dim,
              long total);
bool gated_add(float* x, const float* add, const float* gate, int dim,
               long total);
bool add(float* x, const float* add, long total);
bool bias_add(float* y, const float* b, int width, long total);
bool situ_glu_fused(const float* ug, float* act, int ffn, long total);
bool situ_glu_split(const float* up, const float* gate, float* act, long total);
bool rope_table(float* v, const float* rcos, const float* rsin, int heads,
                int head_dim, long total);
bool sdpa_full(const float* q, const float* k, const float* v, int tokens,
               int heads, int head_dim, float scale, float* out);
// Drop the score-plane scratch the batched-GEMM attention path allocates (the
// VAE decode needs the device memory; the next forward reallocates it).
void sdpa_release_scratch();
bool video_to_tokens(const float* src, float* dst, int channels, int hw);
bool tokens_to_video(const float* src, float* dst, int channels, int hw);
bool qkv_split(const float* qkv, float* q, float* k, float* v, int dim,
               long total);

}  // namespace sonicboom::moss::cuda_dit
