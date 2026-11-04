// SonicBoom: Q8_1 activation quantization for the upstream MMQ path.
//
// The quantize_mmq_q8_1 kernel body is transcribed verbatim from llama.cpp
// commit 43fe9c64281ef735046adc025e9e7559a1f659a5
// (ggml/src/ggml-cuda/quantize.cu).  Only the launch plumbing is new: the
// SonicBoom wrapper passes plain pointers/dimensions instead of ggml tensors,
// uses no expert/scatter ids, and allocates no ggml pool memory.
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#pragma once

#include "mmq.cuh"

// SonicBoom: copied from ggml/src/ggml-cuda/quantize.cuh.
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <mmq_q8_1_ds_layout ds_layout, bool scatter>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3*s03 + i2*s02 + i01*s01;
    }

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // write the block once (normal) or to each of the token's compact rows (scatter)
    const int nwrite = scatter ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block*ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
            ib = ib0 + k_block*ne1 + blockIdx.x;
        }

        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs/64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs/32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs/32]  = d;
            }
        }
    }
    GGML_UNUSED(n_expert_used);
}

// SonicBoom: launch helper derived from quantize_mmq_q8_1_cuda() in
// ggml/src/ggml-cuda/quantize.cu with the ggml tensor/ids plumbing removed.
// x is [ne1][ne00] with row stride s01 (in floats); the output layout is the
// upstream block_q8_1_mmq layout [ne0/QK8_1_MMQ][ne1], matching what
// mul_mat_q consumes.  ne2 == ne3 == 1.
static void quantize_mmq_q8_1_upstream(
        const float * x, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false>
                <<<num_blocks, block_size, 0, stream>>>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1, /*n_expert_used=*/0);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
