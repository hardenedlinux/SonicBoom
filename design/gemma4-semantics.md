# Gemma 4 E4B-It — block semantics (Phase 5C-1 → baseline-resolved)

Phase 5C-1 closed the *weight binding* gap and recorded the semantic questions
that gate numerical correctness. This revision **resolves** those questions from
the authoritative llama.cpp source (`src/models/gemma4.cpp`,
`src/llama-graph.cpp`, `src/llama-model.cpp`, `ggml/src/ggml-cpu/vec.h`), now that
llama.cpp is available (local checkout, HEAD 43fe9c642).
Status is `confirmed` only where the source states it directly; the differential
*run* (dump llama.cpp tensors, compare against SonicBoom) is the remaining
execution step and is not claimed here.

## 1. Tensor account: bound / validated / consumed

All 17 per-block tensor groups are bound and shape/type-validated by
`loader.cpp`. "Consumed" means `run_block` actually uses the tensor. After this
revision the 7 formerly-unconsumed groups have **resolved** arithmetic (§3);
consuming them in `run_block` is the implementation step (§5).

| # | tensor | bound | validated | arithmetic |
|---|---|---|---|---|
| 1 | `attn_norm` | yes | [2560] | input RMSNorm |
| 2 | `ffn_norm` | yes | [2560] | pre-FFN RMSNorm |
| 3 | `attn_q_norm` | yes | [head_dim] | per-head RMSNorm (scalar-broadcast weight) |
| 4 | `attn_k_norm` | yes | [head_dim] | per-head RMSNorm (scalar-broadcast weight) |
| 5 | `post_attention_norm` | yes | [2560] | post-attention RMSNorm |
| 6 | `post_ffw_norm` | yes | [2560] | post-FFN RMSNorm |
| 7 | `post_norm` | yes | [2560] | per-layer-gate post RMSNorm |
| 8 | `attn_q` | yes | [2560, q_width] | Q projection |
| 9 | `attn_k` | yes | [2560, kv_width] | K projection |
| 10 | `attn_v` | yes | [2560, kv_width], q5/q4 | V projection |
| 11 | `attn_output` | yes | [q_width, 2560] | O projection |
| 12 | `ffn_gate` | yes | [2560, ff] | gate proj |
| 13 | `ffn_up` | yes | [2560, ff] | up proj |
| 14 | `ffn_down` | yes | [ff, 2560], q5/q4 | down proj |
| 15 | `inp_gate` | yes | [2560, 256] | per-layer gate down-proj (2560→256) |
| 16 | `proj` | yes | [256, 2560] | per-layer gate up-proj (256→2560) |
| 17 | `layer_output_scale` | yes | [1] | whole-block output scale |

Non-block tensors: all bound. `rope_freqs` is now understood to be a RoPE
`freq_factors` mask (§3.9), consumed only by the global (full-attention) layers.

## 2. Raw-weight evidence (read directly from the GGUF, independent of any reference)

Unchanged from the 5C-1 revision — every statement was measured from the tensor
bytes. Highlights that the source now explains:

- **`attn_q_norm` / `attn_k_norm` are scalar broadcasts** (all entries equal).
  → explained by §3.2: the per-head RMSNorm weight is a scalar in this GGUF.
- **Norm weights are large/negative** (mean ~10, range −24…+290), not `≈1`.
  → explained by §3.1: the GGUF stores the *effective* scale (HF `1+weight`
  folded at conversion time), so llama.cpp multiplies it directly.
- **`layer_output_scale` is a learned per-block scalar** (0.061…0.840).
  → explained by §3.8: it scales the whole block output.
- **`rope_freqs` [256] = [64×1.0, 192×1e30]** → explained by §3.9.

## 3. Resolved semantics (llama.cpp authoritative source)

### 3.1 RMSNorm weight convention — **confirmed A**

`build_norm(..., LLM_NORM_RMS, ...)` (llama-graph.cpp:1642,1655) is
`ggml_rms_norm(x, eps)` then `ggml_mul(w)` — **no** `1+weight`, **no** bias here:

```
y = x * rsqrt(mean(x²) + eps) * w
```

The GGUF stores the effective scale (the `1+weight` of the HF source is folded
into the stored value at conversion), which is why §2 sees large/negative
values. SonicBoom's `nn::rms_norm` already implements exactly this. ✓ no change.

### 3.2 Q/K norm (per-head) semantics — **confirmed**

gemma4.cpp:223,254 apply `build_norm` (RMSNorm) to the per-head Q and K, and
gemma4.cpp:255 applies `ggml_rms_norm(V, eps)` **without** a weight:

```
q = rms_norm(q, eps) * attn_q_norm   // weight is a scalar broadcast (§2)
k = rms_norm(k, eps) * attn_k_norm
v = rms_norm(v, eps)                 // no weight
```

The per-head dim for these RMSNorms is `head_dim` (256 SWA / 512 global); the
stored weight is a scalar broadcast, so the effective op is "normalize the head
to unit norm, then scale by a constant." This is what makes the attention scale
(§3.4) valid. Q and K become near-unit vectors; the dot product is a cosine
similarity.

### 3.3 FFN activation — **confirmed GELU (tanh), not SiLU**

gemma4.cpp:350-355 calls `build_ffn(..., LLM_FFN_GELU, LLM_FFN_PAR, ...)`.
`LLM_FFN_GELU` + parallel gate → `ggml_geglu_split(gate_out, up_out)` =
`gelu(gate_out) * up_out`, then the down projection:

```
f = ffn_down @ ( gelu(ffn_gate @ f) * (ffn_up @ f) )
```

`ggml_gelu` (vec.h:968-969) is the tanh approximation:

```
gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * x * (1 + 0.044715 * x²) ))
```

This is HF `gelu_pytorch_tanh`. The current `exec.cpp` SiLU is **wrong**.

### 3.4 Attention scaling — **confirmed scale = 1.0 (no scaling)**

gemma4.cpp:12 sets `hparams.f_attention_scale = 1.0f` ("no pre-attn scaling"),
passed as the softmax `kq_scale` (llama-graph.cpp:2797):

```
attn = softmax(q·k * 1.0, causal_mask) @ v
```

There is **no** `1/sqrt(head_dim)` and **no** `1/16`; the q/k RMSNorm (§3.2)
supplies the normalization. The current `scaled_dot_product_attention` (which
scales by `1/sqrt(head_dim)`) is **wrong** for every layer.

### 3.5 Attention-logit softcapping — **confirmed disabled**

No `attn_soft_cap` for Gemma 4; the softmax has no tanh pre-capping. Distinct
from §3.6. ✓ (current `run_block` passes softcap 0 — correct).

### 3.6 Final-logit softcapping — **confirmed 30.0, `cap·tanh(x/cap)`**

gemma4.cpp:425-429: `scale(1/30) → tanh → scale(30)`. Value 30.0 from the GGUF.
Not part of `run_block`; belongs to the lm-head stage.

### 3.7 Per-layer embedding gating — **confirmed**

gemma4.cpp:367-388 (per block) plus 469-528 (the per-layer input):

```
# once, at the top (project_per_layer_inputs):
ple   = per_layer_token_embd[token] * sqrt(n_embd_per_layer)      # sqrt(256)
proj  = per_layer_model_proj @ inpL_scaled * (1/sqrt(n_embd))     # 1/sqrt(2560)
proj  = rms_norm(proj, eps) * per_layer_proj_norm                 # weight [256]
inp_per_layer = (proj + ple) * (1/sqrt(2))                        # [256, n_layer, n_tokens]

# per block l, after the FFN residual x2:
g = gelu(inp_gate @ x2)            # inp_gate [2560,256] → [256]
g = g * inp_per_layer[l]           # [256]
g = proj @ g                       # proj [256,2560] → [2560]
g = rms_norm(g, eps) * post_norm   # post_norm [2560]
x3 = x2 + g                        # residual
```

The 5C-1 orientation note is confirmed: `inp_gate` is the down-to-256 gate, `proj`
is the up-to-2560 projection (opposite to the HF gemma3n *names*, matching the
llama.cpp tensor definitions at gemma4.cpp:137-139).

### 3.8 `layer_output_scale` application point — **confirmed whole-block**

gemma4.cpp:391-393 applies it once, after the per-layer gate residual, to the
entire block output:

```
out = x3 * layer_output_scale
```

It is **not** a per-residual scale. The current `exec.cpp` (which scales each
`attn_out`/`ffn_out` axpy) is **wrong**.

### 3.9 RoPE — **confirmed NEOX; `rope_freqs` is a global-layer `freq_factors` mask**

`LLM_ARCH_GEMMA4` → `LLAMA_ROPE_TYPE_NEOX` (llama-model.cpp:3152). gemma4.cpp
passes `rope_freqs` as `freq_factors` to `ggml_rope_ext` **only** on non-SWA
(global) layers (gemma4.cpp:201-203,226), and `nullptr` on SWA layers. The
`1e30` sentinel (§2) is llama.cpp's `freq_factors` "disable this rotation
frequency" convention (partial RoPE across head dims). `freq_base` per layer:
SWA → `rope_freq_base_train_swa` = 10000, global → `rope_freq_base_train` = 1e6
(matches the loader's per-layer `rope_base`). The exact `ggml_rope_ext`
proportional-RoPE arithmetic is the one detail still to transcribe into the
baseline's reference graph.

## 4. Authoritative reference graph (single token, block l)

```
inpL   = tok_embd[token] * sqrt(n_embd)                       # n_embd=2560
# attention
h      = rms_norm(inpL, eps) * attn_norm
q,k,v  = attn_q@h, attn_k@h, attn_v@h
q,k    = rms_norm(q,eps)*q_norm, rms_norm(k,eps)*k_norm
v      = rms_norm(v, eps)
q,k    = rope_NEOX(q,k, pos, freq_base, freq_factors?)       # factors on global only
a      = softmax(q·k * 1.0, causal) @ v
o      = attn_output @ a
o      = rms_norm(o, eps) * post_attention_norm
x1     = o + inpL
# FFN
f      = rms_norm(x1, eps) * ffn_norm
f      = ffn_down @ ( gelu(ffn_gate@f) * (ffn_up@f) )
f      = rms_norm(f, eps) * post_ffw_norm
x2     = f + x1
# per-layer gate
g      = gelu(inp_gate @ x2) * inp_per_layer[l]
g      = proj @ g
g      = rms_norm(g, eps) * post_norm
x3     = x2 + g
out    = x3 * layer_output_scale
```

## 5. What is intentionally not done (this revision)

- **`run_block` math is not yet changed.** This revision documents the resolved
  arithmetic; the implementation edits (q/k/v norm, GELU, attention scale 1.0,
  post-norms, per-layer gate, whole-block `layer_output_scale`, token-embedding
  `sqrt(2560)` scale, per-layer input pipeline) are the next step and are
  itemized in §3/§4.
- No claim of block numerical equivalence, full 42-layer inference, or
  incremental decoding.
- No llama.cpp/GGML/Python runtime dependency added to `libsonicboom.so`; the
  differential run is a test-local reference only.

### Deltas vs. the current `exec.cpp`

| # | current `exec.cpp` | authoritative | change |
|---|---|---|---|
| token embd | unscaled dequant | `* sqrt(2560)` | fix |
| per-layer input | `proj @ rms_norm(ple)` (wrong direction) | `(rms_norm(model_proj@inp) + ple*sqrt(256)) * 1/sqrt(2)` | rewrite |
| q/k/v norm | none | q/k `rms_norm*scalar`, v `rms_norm` | add |
| attention scale | `1/sqrt(head_dim)` | `1.0` | fix |
| attention residual | `hidden += out_scale*attn` | `o = post_attn_norm(wo(attn)); x1 = o + x` | fix |
| FFN activation | SiLU | GELU-tanh | fix |
| FFN residual | `hidden += out_scale*ffn` (ffn_norm on `hidden`) | `f = post_ffw_norm(down(gelu(gate)*up)); x2 = f + x1` | fix |
| per-layer gate | not consumed | §3.7 | add |
| layer_output_scale | per-residual axpy | whole-block multiply | fix |

---

## 6. Numerical-equivalence findings (Phase 5C-2 differential run)

The §3–§4 arithmetic was implemented in `run_block`/`embed_*` and validated
against a llama.cpp baseline (dump tool `tools/baseline/gemma4_dump.cpp`, compare
tool `tools/baseline/compare_dumps.cpp`). The differential run surfaced three
*arithmetic-level* divergences that source-reading alone does not reveal — each
is a place where llama.cpp's CPU kernels do not compute the naive f32 formula.
All three are now reproduced in SonicBoom so the two agree to within fp noise.

### 6.1 q8_K activation quantization (the quantized matmul)

llama.cpp's quantized matmul does **not** dot the packed weight against an f32
activation. `ggml_vec_dot_q*_K_q8_K` quantizes the *activation* to q8_K first:
per 256-element block, `m` is the signed max-magnitude element, scale
`d = -m/127`, and each activation `x[j]` becomes `qs[j] = min(127, nearest_int(-127*x[j]/m))`,
then the dot is `Σ qs[j]·d · w[j]`. A full-f32 activation produces a
systematically different result (~0.1–2% on transformer activations).

Fix: `quant::matvec_f32_q8_K` / `matmul_f32_q8_K` (`core/src/quant/quantized_matmul.cpp`)
gather each batch column, quantize to q8_K (`quantize_q8_K_to_f32`, replicating
`quantize_row_q8_K` with ggml's `nearest_int` magic-constant round-half-to-even),
then run the streaming dot. The 7 projection matvecs in `run_block` use it.

Residual: `Qcur/Kcur/Vcur` RMS ≈ 2e-5 — fp32 accumulation-order noise between
SonicBoom's f32 dot and ggml's int32-accumulating integer dot. This is the
accepted noise floor and is **not** chased to bit-exactness.

### 6.2 fp16 value cast in flash attention

llama.cpp's `build_attn_mha` casts the f32 K and V to fp16 before
`ggml_flash_attn_ext`. For single-token decode the softmax over one logit is 1,
so `kqv_out == fp16(V)` (the K cast is irrelevant until multi-token KV
attention). A full-f32 attention would be *more* accurate but would not match
the baseline.

Fix: `cast_v_to_fp16` (`core/src/model/exec.cpp`) round-trips V through
`std::float16_t` after `Vcur_normed`.

### 6.3 fp16-table GELU (GGML_GELU_FP16)

`ggml/src/ggml-cpu/vec.h` builds with `GGML_GELU_FP16` defined, so both
`ggml_vec_gelu_f32` (the standalone `ggml_gelu`, used by the per-layer gate) and
`ggml_vec_geglu_f32` (the FFN geglu) take the **fp16 lookup-table** branch, not
the precise `ggml_gelu_f32`:

```
x <= -10 → 0;  x >= 10 → x;
else     → fp16( gelu_tanh( fp16(x) ) )     # input and result both fp16-rounded
```

The fp16 quantization is ~2^-11 relative — much larger than the fp32 tanh error —
so this is the dominant divergence source in the FFN and per-layer gate.

Fix: `nn::gelu_fp16` (`core/src/nn/activation.cpp`) reproduces the table lookup
bit-for-bit (both round-trips via `std::float16_t`, the exact `SQRT_2_OVER_PI` /
`GELU_COEF_A` constants and operation order), and both `run_block` call sites
(FFN gate, per-layer gate) use it. `nn::gelu` remains the accurate tanh-GELU
reference (still used by `test_nn`).

### 6.4 layer_output_scale is real and applied whole-block

`blk.%d.layer_output_scale.weight` is a genuine learned per-block scalar
(measured 0.061 / 0.160 / 0.445 for layers 0/1/41), applied once to the whole
block output. llama.cpp's `cb("out_scaled")` and `cb("l_out")` name the *same*
tensor (gemma4's `build_cvec` is identity), so only `l_out` survives to the
baseline dump. `run_block` applies the scale and emits only `l_out` to match.

## 7. Differential result (layer 0, token 0, pos 0)

`sonicboom_dump` vs the baseline, `compare_dumps` (tolerance 1e-4). The tool flags
a tensor when **either** `max_abs` or `max_rel` exceeds 1e-4, so the "mismatch"
count is dominated by relative error on near-zero elements — not by large
absolute error. `only in candidate: 0` (every SonicBoom stage has an baseline
counterpart); the 1628 "only in reference" tensors are the baseline graph's
weights/intermediates that SonicBoom does not emit.

| stage | max_abs | rms |
|---|---|---|
| inp_scaled / attn_norm | 0 | 0 |
| Qcur / Kcur / Vcur | 1.7e-4 / 1.8e-4 / 6.9e-5 | 2.0e-5 / 2.5e-5 / 1.7e-5 |
| Qcur/Kcur/Vcur _normed/_pos | 5.0e-6 / 6.0e-7 / 3.1e-6 | 6.2e-7 … 7.3e-7 |
| kqv_out | 1.9e-6 | 9.4e-8 |
| attn_post_norm / attn_out | 1.8e-4 | 5.8e-6 |
| ffn_norm / ffn_up / ffn_gate | 5.5e-6 / 1.1e-5 / 1.2e-5 | 5.7e-7 … 1.6e-6 |
| ffn_geglu | 1.8e-3 | 1.9e-5 |
| ffn_out / ffn_post_norm | 1.0e-4 / 9.2e-5 | 6.5e-6 / 1.0e-5 |
| pe_in | 1.7e-4 | 1.1e-5 |
| per_layer_embd_out | 3.9e-3 | 5.9e-4 |
| l_out | 2.4e-4 | 3.6e-5 |

Summary: 22 common tensors, worst rms 5.9e-4 (per_layer_embd_out), first
mismatch `Qcur-0`. All error is downstream of the two quantizing kernels (§6.1
q8_K dot, §6.3 fp16 GELU); no stage shows a structural (formula/orientation)
divergence. The `ffn_geglu`/`per_layer_embd_out` absolute errors are the fp16
GELU table snapping small q8_K input differences to different fp16 table entries
(the `max_rel` spikes like 0.27 on `per_layer_embd_out` are near-zero-element
division, not absolute divergence).

## 8. Not claimed

- **No** full-model equivalence, 42-layer inference, or incremental decoding.
  Only a single layer-0 block at token 0 / pos 0 is validated; chaining,
  shared-KV (layers 24–41), and the lm-head remain.
- The q8_K dot is f32-accumulated, not int32-exact (see §6.1); no bit-exactness
  claim for the attention/FFN fp ops.

## 9. Phase 5C-2 acceptance report

**Source revisions.**
- SonicBoom: `f64c418d53a857d205be15fd0ee3f8f3de8b5861` (branch `master`).
- llama.cpp baseline (reference-only): `43fe9c64281ef735046adc025e9e7559a1f659a5`
  (local checkout).

**Worktree status.** Nothing committed; the whole native transformer path is
uncommitted on `master`. Modified: `core/CMakeLists.txt`,
`tests/core/CMakeLists.txt`. Untracked (new this phase): the `core/include/sonicboom/{gguf,model,nn,quant}/`
and `core/src/{gguf,model,nn,quant}/` trees, `tests/core/{ggml_baseline.h,test_dequant.cpp,test_gemma4.cpp,test_gguf.cpp,test_nn.cpp,test_quantized_matmul.cpp}`,
`tools/baseline/` (dump/compare tools + build scripts), and the
`design/{gguf-reader,gemma4-model-map,gemma4-execution-gap,gemma4-baseline,gemma4-semantics}.md`
docs. No branch switch, no commit, no push, no destructive op.

**Changed files (this phase — §5 rewrite + §6 fixes).**
- `core/src/model/exec.cpp` — full `run_block` rewrite (q/k/v RMSNorm, NEOX RoPE
  w/ freq-factors, attention scale 1.0, GELU FFN, post-attn/post-ffn norms,
  per-layer gate, whole-block `layer_output_scale`), `embed_token`/`embed_per_layer`,
  `cast_v_to_fp16`, `gelu_fp16` call sites, `out_scaled`→`l_out` collapse.
- `core/src/nn/activation.cpp` / `core/include/sonicboom/nn/activation.h` —
  `gelu_tanh_f32` helper + `gelu_fp16` (GGML_GELU_FP16 table semantics).
- `core/src/quant/quantized_matmul.cpp` / `.h` — q8_K activation quantization
  (`matvec_f32_q8_K`, `matmul_f32_q8_K`, `quantize_q8_K_to_f32`, `nearest_int`).
- `tests/core/test_nn.cpp` — `test_gelu_fp16` regression.
- `tools/baseline/{gemma4_dump.cpp,sonicboom_dump.cpp,compare_dumps.cpp}` +
  `build{,_compare,_sonicboom}.sh` — baseline/candidate dump + comparator.

**Commands.**
```
cmake --build build --target sonicboom -j            # SHARED libsonicboom.so
cd build && ctest --output-on-failure                # 45/45 pass
tools/baseline/build.sh                                # baseline dump (links llama.cpp, dev-only)
tools/baseline/build_sonicboom.sh                      # candidate dump (links libsonicboom.so)
tools/baseline/gemma4_dump  <model.gguf> --layer 0 --token 0 --pos 0 > tools/baseline/baseline.dump
tools/baseline/sonicboom_dump <model.gguf> --layer 0 --token 0 --pos 0 > tools/baseline/sonicboom.dump
tools/baseline/compare_dumps tools/baseline/baseline.dump tools/baseline/sonicboom.dump
```

**Test results.** `ctest` 45/45 passed, 0 failed (includes `test_nn` with
`test_gelu_fp16`, `test_gemma4`, `test_quantized_matmul`, `test_dequant`,
`test_gguf`, plus all regression tests).

**Unresolved discrepancies (localized, tolerated).**
1. q8_K dot accumulation — SonicBoom accumulates in f32; ggml in int32. Residual
   `Qcur/Kcur/Vcur` rms ~2e-5. Not chased to bit-exactness (§6.1).
2. fp16-table GELU snapping — small q8_K input differences map to different fp16
   table entries, yielding `ffn_geglu`/`per_layer_embd_out` max_abs ~2e-3/4e-3
   and near-zero-element `max_rel` spikes up to 0.27 (§6.3, §7). Absolute error
   is the noise floor, not a formula divergence.

**Not claimed.** No full-model equivalence, no 42-layer inference, no incremental
decoding (§8); only layer 0 / token 0 / pos 0 is validated.

## 10. Phase 5D acceptance report — 42-layer chaining + shared-KV, bit-exact

**Result.** All 42 blocks chain through the residual stream with Gemma 4's
shared-KV (layers 24–41 reuse the V cache of donors 22 SWA / 23 global), and
every SonicBoom-emitted intermediate matches the llama.cpp baseline **bit-for-bit**
(max_abs = max_rel = rms = 0 on all 794 common tensors, 0 mismatches).

**Key insight — bit-exactness, not just closeness.** The q8_K activation
quantization (`quantize_row_q8_K_ref`) is discontinuous: a ~1e-7 residual
difference crosses a quantization boundary and amplifies ~1000× per layer.
41 layers of that made the f32-vs-double accumulation-order difference diverge
from layer ~12 onward. The only stable fix was to reproduce ggml's scalar
reduction semantics exactly, so both sides round identically at every q8_K
boundary.

**Baseline rebuilt scalar.** The AVX2 llama.cpp build reduces in a different
order (4×8-lane tree in `ggml_vec_dot_f32/bf16` and the AVX2 q4/q3/q5_K dots)
that a scalar implementation cannot cheaply match. The baseline was therefore
rebuilt with SIMD disabled (`GGML_AVX/AVX2/F16C/FMA/SSE42/BMI2=OFF`), which
selects ggml's `ggml_float = double` scalar reductions — deterministic and
directly bit-matchable. The AVX2 baseline dump was preserved at
`/tmp/baseline_avx2.dump` for reference; the committed comparison is against the
scalar `tools/baseline/baseline.dump`.

**Code changes (this phase — three accumulation-order fixes).**
- `core/src/nn/norm.cpp` — `rms_norm` sum now `double(v * v)` (f32 product
  first, then double accumulate), matching ggml's `sum += (ggml_float)(x[i]*x[i])`.
- `core/src/model/exec.cpp` — `matvec_f32` (inp_gate/proj) accumulates in double
  with each f32 product rounded before adding, then casts to f32.
- `core/src/model/exec.cpp` — `embed_per_layer` bf16 matvec accumulates in
  double; the result is cast to f32 *before* the `1/sqrt(d)` projection scale
  (reproducing `ggml_mul_mat` → f32 → `ggml_scale`).

**Remaining (out of this phase's scope — the lm-head).** `forward_traced`
returns the final hidden state (`l_out-41`) but does not yet apply the final
`output_norm` (`result_norm`) or the vocab projection (`result_output`, the
q4_K output matrix × 262144 vocab). The baseline's `node_*`, weight (`norm*`,
`per_layer_proj`), `h_nextn`, `inp_per_layer_selected`, and `(reshaped)/(view)`
tensors are ggml graph internals SonicBoom intentionally does not emit.

## 11. Phase 5E acceptance report — lm-head, bit-exact logits

**Result.** The full single-token forward now extends through the lm-head and is
bit-exact against the scalar baseline end-to-end: 797 common tensors, 0
mismatches, worst rms 0 (`compare_dumps`). This includes the two head tensors
`h_nextn` / `result_norm` (both 0) and `result_output` (262144 softcapped
logits, 0).

**lm-head semantics (weight-tied).** The GGUF has no `output.weight`; llama.cpp
ties `output` to `token_embd.weight` (`TENSOR_DUPLICATED`). The head is:
`result_norm = rms_norm(final_hidden, output_norm)`, `logits = token_embd @
result_norm` (q8_K-fused q3_K matvec), then final-logit softcapping.

**Softcapping must multiply by the reciprocal, not divide.** llama.cpp emits
`ggml_scale(cur, 1/sc) → ggml_tanh → ggml_scale(cur, sc)`, i.e.
`sc * tanh(x * (1.0f/sc))`. A naive `sc * tanh(x / sc)` uses correctly-rounded
division, which differs from `x * (1.0f/sc)` by up to 1 ULP; tanh×30 amplifies
that to a max_abs of ~4.8e-6 (rms 7.3e-7) on `result_output`. Matching ggml's
multiply-by-reciprocal order closes it to bit-exactness.

**Code changes (this phase).**
- `core/src/model/exec.cpp` / `core/include/sonicboom/model/exec.h` — added
  `lm_head(model, hidden, logits)` (output norm + tied q3_K projection +
  softcap), and extended `forward_traced` to emit `h_nextn`/`result_norm`/
  `result_output` after the block loop (trace-only; `forward` still returns the
  2560-dim final hidden state, `lm_head` returns the 262144-dim logits).

**Tests.** `ctest` 45/45 passed, 0 failed.
