# Models tested

This page is the record of every model SonicBoom has been run on, including
models that have since been retired. A model that was measured once stays
listed here even when it leaves `models/`, so a result quoted somewhere in the
project can always be traced back to the file it was measured on.

Model files are not part of the repository; `models/` is a local directory.
The reference files, their sizes and checksums are in
[Building SonicBoom](build.html) and in [Benchmarks](benchmark.html).

## Status

| status | meaning |
|---|---|
| **reference** | used by the committed tests, goldens and benchmark tables |
| **measured** | run through `tools/bench.sh` or an equivalent probe for a comparison or a decision; never used in tests, goldens or fixtures |
| **unsupported** | the SonicBoom loader refuses the architecture; any number quoted for it was produced by another runtime and is labelled with that runtime |
| **retired** | tested at some point, no longer part of the set |

Sizes are the local file sizes in bytes.

## Language models SonicBoom runs

| model | family | quant | size | status |
|---|---|---|---:|---|
| `gemma-4-E4B-it-Q5_K_M.gguf` | Gemma 4 E4B-it | Q5_K_M | 5,481,798,784 | reference — tests, goldens, benchmarks, serving |
| `qwen2.5-7b-instruct-1m-q4_k_m.gguf` | Qwen2.5-7B-Instruct-1M | Q4_K_M | 4,683,073,888 | reference — tests, goldens, benchmarks, serving |
| `Qwen3.6-35B-A3B-Q4_K_M.gguf` | Qwen3.6-35B-A3B (MoE) | Q4_K_M | 20,542,732,576 | reference — benchmarks; expert-pool and long-context work |
| `Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q5_K.gguf` | Qwen3.6-35B-A3B (MoE), abliterated | Q5_K | 25,346,479,328 | measured — abliterated Q5_K against the official Q4_K_M |
| `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf` | Gemma 4 E4B (uncensored) | Q6_K_P | 6,249,794,528 | measured — recommended uncensored Gemma file |
| `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf` | Gemma 4 E4B (uncensored) | Q8_K_P | 8,133,226,464 | measured 2026-11, file not kept |
| `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | DeepSeek-R1-Distill-Qwen-1.5B (dense) | Q4_K_M | 1,117,321,312 | measured — decode/prefill attention studies |

`Q6_K_P` is the recommended uncensored Gemma file. It decodes 22% faster than
`Q8_K_P` (tg64 41.291 vs 33.901 tok/s, prefill 2768.4 vs 2727.4 tok/s) and is
1.9 GB smaller. On five prompts of 256 greedy tokens each (three Chinese
writing, two English), scoring the same token sequences under both files over
the full vocabulary gives a per-token KL(Q8‖Q6) of 1.11e-2 mean and 3.38e-1
max, with top-1 agreement on 1302 of 1372 positions (94.9%): the quality
difference is negligible for writing, so only `Q6_K_P` is kept. The `Q8_K_P`
row stays for the record; measured 2026-11, the file was deleted. Their `K_P`
quantization places the per-layer projections in the quantized format, which
older SonicBoom loaders rejected; the current loader accepts it.

The Huihui file carries an extra block (40) for its MTP head; it is
measurement-only and never part of a test or golden.

## Image models

| model | family | size | status |
|---|---|---:|---|
| `MOSS-Image-SFT` (`OpenMOSS-Team/MOSS-Image-SFT`, safetensors) | MOSS-Image (text-to-image) | 27,229,060,886 | measured — end-to-end image generation through the C ABI, Guile and the server |

MOSS-Image's text encoder is a Qwen3.5-family 9B model in the checkpoint's own
safetensors; the DiT and VAE run through SonicBoom's image path. Only the
language models above appear in the README's benchmark tables.

## Models SonicBoom refuses

These were downloaded and tested for the record; SonicBoom has no loader for
the architecture, and every tool refuses at load. The measurements quoted for
them come from a llama.cpp build and say so.

| model | family | quant | size | status |
|---|---|---|---:|---|
| `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` | Inkling 2 Mini (MoE, looped core layers) | Q4_K | 16,525,559,776 | unsupported — measured in llama.cpp |
| `Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf` | Inkling 2 Mini (MoE, looped core layers) | IQ3_S | 13,340,510,176 | unsupported — measured in llama.cpp |
| `mmproj-Inkling-2-Mini-Uncensored-F16.gguf` | Inkling 2 Mini vision projector | F16 | 639,348,448 | unsupported — used by the llama.cpp vision CLI |

The measurements used llama.cpp b12651, the build carrying Inkling 2 support.
On the 8 GB card Q4_K runs only with host op-offload disabled (the op-offload
compute buffer alone exceeds the card); IQ3_S runs with op-offload on at
`-ngl 8`.

SonicBoom's refusal is clean: the loader fails with `load_gemma4 error`,
`tools/bench.sh` reports each metric empty, and no device memory is left
allocated.

## Retired

| model | quant | status |
|---|---|---|
| `gemma-4-E4B-it-Q3_K_M.gguf` | Q3_K_M | retired — the performance reference before Q5_K_M; replaced by Q5_K_M |
| `gemma-4-E4B-it-Q4_K_M.gguf` | Q4_K_M | retired — load-time quantization-fallback validation (2026-10-14); never the reference |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Q4_K_M | retired — not part of the test set; no longer tested |

## Evaluation policy

- Gemma 4 E4B is evaluated on **Q5_K_M**. Published Gemma numbers recorded
  before the switch were measured on Q3_K_M and say so where they are quoted.
- Qwen2.5-7B-Instruct-1M is evaluated on **Q4_K_M**; Qwen3.6-35B-A3B on
  **Q4_K_M**.
- Uncensored, abliterated and vision-projector files are **measurement-only**
  by construction. They are never used in tests, goldens or fixtures, and no
  committed expectation depends on them.
- A model is listed here once it has been loaded by SonicBoom or, for an
  unsupported architecture, once SonicBoom has been asked to load it and
  refused it.
