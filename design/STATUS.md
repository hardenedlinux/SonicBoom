# Project Status

- Branch: `optimize` at `dd47858`.
- Q6_K implementation: `2e361d1`; benchmark update: `dd47858`; not pushed.
- API server: `3dd60dd`; present on `origin/master` (pushed).
- Load-time fallback commits are in the local history; the two commits above are the current optimize work.

## Completed

- GGUF load-time quant fallback to Q5_K for supported foreign types.
- Native Q6_K CPU and CUDA: dequant, gather, f32 GEMV, q8_1 MMVQ single and batched.
- Q6_K uses D4_Q8_1 activation layout.
- OpenAI-compatible API server uses the real SonicBoom backend; checks a–f passed.

## Disabled / Known Issues

- Q6_K MMQ weight path is disabled (`TODO(q6k-mmq)`).
- Q6_K prefill uses dequantization plus cuBLAS.

## Decode and Prefill Benchmarks

Tok/s; ratios are SonicBoom / llama.cpp. llama.cpp checkout: `43fe9c64281`.

| Model / metric | SonicBoom | llama.cpp | Ratio |
|---|---:|---:|---:|
| Q3_K_M tg64 depth 0 | 51.65 | 45.22 | 1.14x |
| Q3_K_M tg64 depth 512 | 49.96 | 44.76 | 1.12x |
| Q3_K_M tg64 depth 2048 | 47.13 | 44.08 | 1.07x |
| Q3_K_M pp512 | 1148 | 1949 | 0.59x |
| Q3_K_M pp2048 | 1058 | 1993 | 0.53x |
| Q3_K_M pp4100 | 1051 | 1944 | 0.54x |
| Q4_K_M tg64 depth 0 | 49.34 | 54.99 | 0.90x |
| Q4_K_M tg64 depth 512 | 48.29 | 54.30 | 0.89x |
| Q4_K_M tg64 depth 2048 | 46.30 | 53.32 | 0.87x |
| Q4_K_M pp512 | 1082.6 | 2048.02 | 0.53x |
| Q4_K_M pp2048 | 989.2 | 2084.09 | 0.47x |
| Q4_K_M pp4100 | 986.6 | 2035.26 | 0.48x |

## Decode Gap Findings

- Q4_K_M SonicBoom minus llama.cpp, GPU kernel time per step: norm +931 us;
  elementwise +853 us; Q6_K MMVQ +597 us; attention +392 us; Q4_K MMVQ -252 us.
- Kernel launches per step: SonicBoom 1279, llama.cpp 1084.
- Q3_K_M lead mainly comes from llama.cpp's fused Q3_K up+gate kernel:
  45% of 192 GB/s reference peak vs SonicBoom's separate kernels at 76%.
- Local llama.cpp source checkout: `/home/alexon/Project/llama.cpp`
  (`43fe9c64281`).

## Next Steps

1. Q6_K MMVQ launch configuration.
2. Norm/elementwise fusion (design pending).
3. Attention split/merge.
4. Q6_K MMQ.
5. Reduce shared-library size.
