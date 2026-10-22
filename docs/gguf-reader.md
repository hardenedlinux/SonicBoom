# GGUF v3 Reader — implementation, boundaries, and Gemma 4 gap

Status: reader implemented + unit-tested; real-file sanity-checked. No inference
backend built in this task.

## Scope

Add an independent, self-contained GGUF v3 reader to SonicBoom. The reader
parses the GGUF on-disk format and exposes each tensor's raw (possibly
quantized) bytes by offset + computed size. It does **not** build a compute
graph, schedule work, touch a device backend, or dequantize — those are
explicitly out of scope for this task.

## Files

- `core/include/sonicboom/gguf/reader.h` — public Layer 2 header.
- `core/src/gguf/reader.cpp` — implementation.
- `tests/core/test_gguf.cpp` — unit tests (registered in CTest).

The reader is a new `core/src/gguf/` subsystem globbed into `libsonicboom.so`
(`SONICBOOM_GGUF_SOURCES`), mirroring the existing `sx/` and `planner/`
subsystem layout.

## What is parsed (GGUF v3)

- **Header** — magic `0x46554747` ("GGUF" little-endian), `version` (must be 3),
  `n_tensors`, `n_kv` (all little-endian, bounds-checked).
- **KV metadata** — every value type (`uint8/16/32/64`, `int8/16/32/64`,
  `float32/64`, `bool`, `string`, `array`), including arrays of strings
  (tokenizer vocabularies). Scalars are widened losslessly.
- **Tensor directory** — `name`, `n_dimensions`, `dimensions`, ggml `type`,
  `offset` (relative to the data section). Element count and byte size are
  computed from the type table.
- **Alignment** — `general.alignment` (default 32); the data section starts at
  the aligned position after the tensor directory.
- **Raw data access** — `tensor_data()` returns a `std::span<const std::byte>`
  into the backing buffer, bounds-checked against the data section.

## ggml type table

Ids `0..30` are mapped to `(name, block_size, bytes_per_block)` — the format
reference needed to compute tensor byte sizes (f32, f16, bf16, i8/16/32/64,
f64, q4_0/q4_1/q5_0/q5_1/q8_0/q8_1, q2_K…q8_K, and the iq_* types). Unknown
type ids yield a `nullopt` byte size and the tensor is still listed — parsing
never hard-fails on an unknown type. No dequantization constants or kernels are
implemented; the table is format metadata only.

## Dependency boundary

The reader is **pure SonicBoom C++23**. It has no dependency on ggml,
llama.cpp, MLIR, native-torch, or any device backend; the format constants are
re-declared by reference to the public GGUF v3 specification
(`ggml/docs/gguf.md`). This is verified by the fact that `reader.cpp` compiles
with only `-I core/include` and nothing else. It mirrors only the *format*,
never ggml's execution code.

## Error model

`ErrorCategory::{Io, Format, Truncated}` plus a message and (when known) the
byte offset. Corrupted inputs (bad magic, unsupported version, truncated
header/string/tensor-info, zero-dim tensors) are rejected cleanly, never via
UB.

## Test results

`tests/core/test_gguf.cpp` builds synthetic GGUF files in memory and covers:
header + metadata read, all value types (incl. string arrays), the tensor
directory, alignment + offsets, raw byte access, boundary checks (out-of-range
offset/size → `nullopt`), unknown types (graceful), corrupted/truncated files,
and file-backed load.

Result: **`test_gguf OK`** (all checks pass) — compiled and run standalone
against `reader.cpp`; registered in CTest as `test_gguf`.

## Real-model validation

Validated against a real Gemma 4 GGUF file now present in the workspace
(`models/gemma-4-E4B-it-Q3_K_M.gguf`, 4.06 GB, GGUF v3):

| file | size | tensors | result |
|---|---|---|---|
| `~/Project/velum/models/hift.gguf` | 83 MB | 246 | parsed OK (vocoder, f32) |
| `~/Project/velum/models/llm.gguf` | 2.5 GB | 293 | parsed OK (transformer LLM, f32) |
| `models/gemma-4-E4B-it-Q3_K_M.gguf` | 4.06 GB | 720 | parsed OK (Gemma 4 E4B-It, mixed quant) |

The Gemma 4 file parses cleanly end to end: `version=3`, `n_kv=56`,
`alignment=32`, `general.architecture = "gemma4"`, `general.name =
"Gemma-4-E4B-It"`, context 131072, `embedding_length=2560`, 42 blocks, GQA
heads 8/2, sliding window 512, `final_logit_softcapping=30`. **All 720 tensors**
resolve to a correct data span (`resolvable=720 bad=0 unknown_type=0`), proving
the header → KV → tensor-directory → offset/alignment → raw-bytes path on a real
quantized model.

Tensor dtype mix (the "_M" in Q3_K_M means mixed):

| ggml type | count |
|---|---|
| `f32` | 423 (norms, biases, `layer_output_scale`) |
| `q3_K` | 169 (`token_embd`, `attn_q/k`, `ffn_gate/up`) |
| `q4_K` | 123 (`attn_output`, `per_layer_token_embd`) |
| `q5_K` | 4 (`attn_v`, `ffn_down`) |
| `bf16` | 1 (`per_layer_model_proj`) |

Two Gemma-4-specific structure notes relevant to execution: (1) there is **no
`output.weight`** — the LM head is a per-layer token embedding
(`per_layer_model_proj.weight` [2560×10752, bf16] × `per_layer_token_embd.weight`
[10752×262144, q4_K]); (2) every block carries a learned `layer_output_scale`
(f32 scalar) applied to its residual output. Neither is a plain tied embedding.
The reader exposes all of this by name/shape/offset/bytes; none of the dequant
or computation is in scope here.

## Gemma 4 compute gap (existing capability vs. required)

SonicBoom's current compute surface, assessed without building an inference
backend:

- **Operators** (S-Expr v0.1 / planner `OpKind`): `conv`, `relu`, `add`,
  `max_pool`, `reduce_mean`, `reshape`, `gemm`, `softmax` — 8 ops.
- **Dtypes** (`sx::DType`): `float32/16`, `bfloat16`, `float64`, `int8/16/32/64`,
  `uint8`, `bool` — no quantized block types.
- **Engines**: S-Expr → MLIR (CPU JIT) and native-torch boxed dispatch.

Gemma 4 is a decoder-only transformer. Mapping requirement → current support:

| Gemma 4 requirement | present | gap |
|---|---|---|
| Embedding lookup | — | no `gather`/embedding op |
| RMSNorm / LayerNorm | — | no norm op (only `reduce_mean`, no `pow`/`sqrt`/`mul` to compose it) |
| RoPE | — | no rotary-position op |
| QKV projection | `gemm` ✓ | — |
| Head split / reshape / transpose | `reshape` ✓ | no `transpose`/`split` |
| Scaled dot-product attention | `gemm` + `softmax` ✓ | no `mul` for the scale, no fused attention |
| QK-norm | — | depends on missing RMSNorm |
| MLP (up/gate/down) | `gemm` ✓ | — |
| Activation (GELU/SiLU/SwiGLU) | — | no activation op (only `relu`) |
| Residual add | `add` ✓ | — |
| LM head | `gemm` ✓ | — |
| Quantized weights (GGUF Q4_K etc.) | — | no dequant, no quantized GEMM, no quantized dtype in the IR |
| KV cache / decoding (top-k, top-p, sampling) | — | not a graph op; absent |

**Concrete missing operators for a Gemma 4 path**: `rms_norm` (or `layer_norm`),
`rope`, `gelu`/`silu`/`swiglu`, `embedding`/`gather`, `transpose`, and a
`mul` (attention scaling). On top of the operator set, quantized execution needs
dequantization kernels and a quantized dtype — both explicitly deferred here.

None of these are implemented in this task; the deliverable is the reader plus
this gap list.
