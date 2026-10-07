# Gemma 4 E4B-It — differential oracle (Phase 5B)

Status: **unblocked.** llama.cpp is available at `/path/to/llama.cpp`
(HEAD 43fe9c642). The blocked semantic questions are now **resolved from the
authoritative source** — see `design/gemma4-semantics.md` §3, which supersedes
§5–§6 here. The oracle remains a *development/test* aid only — it adds no
llama.cpp / ggml runtime dependency to SonicBoom and is never linked into
`libsonicboom.so`.

## 1. Blocker — resolved

The "llama.cpp not installed / no network" blocker is gone. The reference is:

- `/path/to/llama.cpp/src/models/gemma4.cpp` — authoritative Gemma 4
  E4B architecture (the `general.architecture = "gemma4"` layout; E4B = 42
  layers via the `case 42` in `load_arch_hparams`).
- `src/models/gemma3.cpp`, `src/models/gemma3n.cpp` — sibling architectures for
  cross-reference.
- `src/llama-graph.cpp` — `build_norm` / `build_ffn` / `build_attn_mha` /
  `build_inp_embd` semantics.
- `src/llama-model.cpp` — `llama_model_rope_type` (`GEMMA4 → NEOX`).
- `ggml/src/ggml-cpu/vec.h` — the exact `ggml_gelu_f32` (tanh) formula.
- `conversion/gemma.py` — GGUF converter (norm weights folded to effective scale).

Local ground-truth assets (unchanged):

- `models/gemma-4-E4B-it-Q3_K_M.gguf` — the weight file.
- `tests/core/ggml_oracle.h` — verbatim ggml dequant transcription, already
  bit-exact-checking dequant via `test_dequant` (the one oracle tier that was
  always runnable and is passing).

## 2. Oracle architecture (design, ready when a source is available)

Follow velum's pattern, but keep it out of the runtime:

```text
                        ┌─────────────────────────────┐
  gemma-4-E4B-it.gguf ─▶│  reference runner (ggml)    │──dump──▶ *.bin  (golden)
                        │  loads GGUF, builds graph,  │
                        │  computes per-layer, dumps  │
                        └─────────────────────────────┘
  gemma-4-E4B-it.gguf ─▶│  SonicBoom native path      │──write─▶ *.bin  (actual)
                        │  (model::exec)              │
                        └─────────────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────┐
                        │  test_gemma4_oracle: load   │
                        │  both, compare per-tensor   │
                        │  with tolerances / bit-exact │
                        └─────────────────────────────┘
```

- **Where it lives:** a `tests/` executable only. It links a *test-local* copy of
  the ggml graph builder (or, once obtainable, a pinned llama.cpp subset as a
  static test dependency). It is **not** added to `sonicboom`'s target or
  `libsonicboom.so`.
- **Two comparison tiers** (already the project's convention):
  - *Bit-exact dequant* — token embedding, and every quantized matvec input, are
    exactly reproducible; zero tolerance. (Covered today by `ggml_oracle.h`.)
  - *Floating-point op* — RMSNorm, RoPE, attention, gating: compare in fp32 with
    an absolute/relative tolerance (start `rtol 1e-4 / atol 1e-5`, loosen only
    with evidence of accumulation order differences).
- **Fixtures:** a fixed prompt string, or a small set of token ids
  (`SONICBOOM_GEMMA_GGUF`-selected model, a checked-in `tokens.bin`); no random
  inputs.

## 3. Comparable intermediate tensors

The reference runner must dump (and SonicBoom must produce) these, in order:

1. `token_embd` — bit-exact dequant of `token_embd.weight[row]`.
2. `per_layer_embd` — `per_layer_token_embd.weight[row]` reshaped
   `[42, 256]`, after the per-layer norm/projection (`per_layer_proj_norm`,
   `per_layer_model_proj`).
3. per block `l`, single token, position `p`:
   - `input` (residual into the block), `input_norm`
   - `q`, `q_norm`, `k`, `k_norm`, `v` (after per-head norm, before RoPE)
   - `q_rope`, `k_rope` (after RoPE)
   - `attn_out` (before `post_attention_norm`), `post_attn`
   - `pre_ffn`, `ffn_out`, `post_ffn`
   - the per-layer gate path: `inp_gate_in`, `inp_gate_out`, `proj_out`, `post_norm`
   - `block_out` (after `layer_output_scale`)
4. `output_norm`, and multi-token `logits` (after `final_logit_softcapping`).

## 4. Verified facts (read directly from the GGUF)

These are not assumptions — every item was observed in the file's tensor
directory / metadata.

**Non-block (6 tensors):**

| tensor | dtype | dims |
|---|---|---|
| `token_embd.weight` | q3_K | [2560, 262144] |
| `per_layer_token_embd.weight` | q4_K | [10752, 262144] |
| `per_layer_proj_norm.weight` | f32 | [256] |
| `per_layer_model_proj.weight` | bf16 | [2560, 10752] |
| `output_norm.weight` | f32 | [2560] |
| `rope_freqs.weight` | f32 | [256] |

**Per-block (17 tensors, identical names for all 42 blocks):** see
`gemma4-model-map.md`. The phase-5A-relevant corrections:

- `attn_q_norm.weight` / `attn_k_norm.weight` are per-head RMSNorm weights whose
  dim = the layer's head_dim (**256** on sliding-window layers, **512** on global
  layers — verified on `blk.0` and `blk.5`).
- `inp_gate.weight` [2560, 256], `proj.weight` [256, 2560], `post_norm.weight`
  [2560] are constant across layers (per-layer-input dim 256, independent of
  head_dim). This is the per-layer embedding gating path.
- `layer_output_scale.weight` [1] is a per-block scalar.

**Metadata:**

| key | value |
|---|---|
| `gemma4.final_logit_softcapping` | **30.0** (present) |
| `gemma4.attention.attn_logit_softcapping` | **absent** → attention logit softcapping is *disabled* (not 50.0) |
| `gemma4.attention.query_pre_attn_scalar` | **absent** → attention scale must come from another rule |
| `gemma4.hidden_act` | **absent** → activation must come from another rule |
| `gemma4.attention.scale` | **absent** |

## 5. Source-code evidence (HF transformers, non-executable)

From `modeling_gemma3.py` / `modeling_gemma3n.py` (Gemma 3 and Gemma 3n text):

- **4 norms per layer:** `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm` — matching the
  GGUF's `attn_norm` / `post_attention_norm` / `ffn_norm` / `post_ffw_norm`.
- **Residual flow (gemma3):** `residual=x; x=input_norm(x); x=attn(x);
  x=post_attn_norm(x); x=residual+x; residual=x; x=pre_ffn_norm(x); x=mlp(x);
  x=post_ffn_norm(x); x=residual+x`.
- **RMSNorm:** `x * rsqrt(mean(x²)+eps) * (1 + weight)` — the weight is applied
  as `(1+w)`, *not* as `w` (gemma3 line 133-139).
- **q/k per-head norm:** applied before RoPE (`q_norm`/`k_norm`, gemma3 312-313).
- **Attention scale:** `query_pre_attn_scalar**-0.5` (gemma3 273); the GGUF does
  not carry `query_pre_attn_scalar`, so the effective scale is unresolved.
- **Token-embedding scale:** `embed_scale = hidden_size**0.5` (gemma3 466).
- **Per-layer embedding (gemma3n 1721-1753):**
  `per_layer_inputs = embed_tokens_per_layer(ids)` reshaped `[n, 42, 256]`
  (embed scale `256**0.5`); `proj = rms_norm(model_proj(main_emb) * hidden**-0.5)`
  reshaped `[n, 42, 256]`; combined `(proj + per_layer_inputs) * (1/√2)`.
- **Per-block gating (gemma3n 1462-1470):**
  `h = gate(main_pred)` (Linear 2560→256); `h = act(h)`; `h = h * per_layer_input`;
  `h = proj(h)` (Linear 256→2560); `h = post_norm(h)`; then added back.
  → the GGUF's `proj.weight` [256,2560] ↔ `inp_gate.weight` [2560,256] pair is
  this mechanism, but the *exact* llama.cpp formula (orientation, activation,
  add vs. scale point) is **unresolved**.
- **`final_logit_softcapping`:** `logits = cap * tanh(logits/cap)` (gemma3 675-678).

Note: gemma3n additionally has AltUp (`altup_*`) and LAuREL (`linear_left/right`,
`post_laurel_norm`) modules that have **no corresponding tensors in this GGUF**,
so "gemma4" is the plain per-layer-embedding architecture (llama.cpp's Gemma 3
layout), *not* the full nano variant.

## 6. Semantics — resolved

The six questions are resolved from the llama.cpp source; see
`design/gemma4-semantics.md` §3 for the authoritative answers and §4 for the
exact single-block reference graph. Summary:

| # | question | resolution |
|---|---|---|
| 1 | `inp_gate`/`proj`/`post_norm` gating | `g = gelu(inp_gate@x2) * inp_per_layer[l]; g = proj@g; g = rms_norm(g)*post_norm; x3 = x2 + g` |
| 2 | `layer_output_scale` | whole-block multiply after the gate residual |
| 3 | attention scale | `f_attention_scale = 1.0` — no `1/sqrt(head_dim)` (q/k RMSNorm supplies normalization) |
| 4 | FFN activation | `gelu_pytorch_tanh` (0.5x(1+tanh(√(2/π)x(1+0.044715x²)))) |
| 5 | RoPE / `rope_freqs` | NEOX; `rope_freqs` is a `freq_factors` mask on global layers only (1e30 disables a frequency) |
| 6 | per-layer embedding | gated (mul) into the residual, §3.7 |

The implementation edit is now permitted (Phase 5B "don't change math until
evidence lands" is satisfied by the source); it is tracked separately from this
doc.

## 7. Oracle plan (next)

Two comparison tiers, as before:

- **Bit-exact dequant** — already passing (`test_dequant` via `ggml_oracle.h`).
- **Per-tensor fp32 diff** — dump llama.cpp's intermediate tensors and compare
  against SonicBoom's `run_block`/`embed_*` once the math in §6 is implemented.

Concrete options for producing the golden tensors (to be chosen when the math is
landed):

1. **Modified-llama.cpp dump** — enable the `cb(...)` graph callbacks in
   `gemma4.cpp` (a `LLAMA_DEBUG`-style build) to dump `attn_norm`, `Qcur`,
   `Kcur`, `Vcur`, `Qcur_normed`, `kqv_out`, `attn_post_norm`, `ffn_out`,
   `per_layer_embd_out`, `l_out`, `result_norm`, `result_output` for a fixed
   token at a fixed position. Most precise; requires a local llama.cpp build.
2. **Black-box logits** — run `llama-cli` on a fixed prompt and compare
   `llama_get_logits` (or `--logits`-style dump) against SonicBoom's full
   forward pass. End-to-end; needs the 42-layer loop wired first.

## 8. Reproducible build/run (what works today)

- `test_dequant` — bit-exact dequant oracle (ggml) vs SonicBoom, already passing.
- `test_gemma4` — real-model structural checks; the blk.1 (Q5_K) → blk.2 (Q4_K)
  mixed-precision boundary and the 17-tensor binding completeness are asserted.

```sh
cmake --build build -j && ctest --test-dir build --output-on-failure
```
