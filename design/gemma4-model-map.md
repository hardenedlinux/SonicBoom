# Gemma 4 E4B-It — model architecture map

Facts read from `models/gemma-4-E4B-it-Q3_K_M.gguf` (GGUF v3, 720 tensors) with
the SonicBoom GGUF reader — not from upstream docs or llama.cpp. This is the
reference for Phase 5 (model execution); every number here was observed in the
file's metadata/tensor directory.

## Identity

- `general.architecture = "gemma4"`, `general.name = "Gemma-4-E4B-It"`,
  `general.size_label = "7.5B"`, quantized by Unsloth (Q3_K_M).
- 42 blocks (`gemma4.block_count = 42`), context 131072.

## Global hyperparameters

| Key | Value |
|---|---|
| `embedding_length` | 2560 (model dim) |
| `feed_forward_length` | 10240 |
| `head_count` / `head_count_kv` | 8 / 2 |
| head_dim | 256 (sliding-window) / 512 (global) — uniform per layer, q/k/v alike |
| `layer_norm_rms_epsilon` | 1e-6 |
| `final_logit_softcapping` | 30.0 |
| `rope.freq_base` (global) | 1e6 |
| `rope.freq_base_swa` | 10000 |
| `rope.dimension_count` / `_swa` | 512 / 256 (= head_dim) |
| `attention.key_length` / `value_length` | 512 / 512 (= head_dim, global) |
| `attention.key_length_swa` / `value_length_swa` | 256 / 256 (= head_dim, swa) |
| `attention.sliding_window` | 512 |
| `attention.shared_kv_layers` | 18 |
| `embedding_length_per_layer_input` | 256 |
| vocab_size | 262144 (from `tokenizer.ggml.tokens`) |
| bos / eos / unknown / pad / mask | 2 / 106 / 3 / 0 / 4 |

## Hybrid attention (per-layer)

`gemma4.attention.sliding_window_pattern` (42 bits, layer 0..41):
`111110111110111110111110111110111110111110` — `1` = sliding-window layer,
`0` = global (full-context) layer. Global layers are 5, 11, 17, 23, 29, 35, 41
(every 6th; 7 layers). The other 35 are sliding-window (window 512).

Per-layer attention projections (contiguous dim = kv/query width):

| kind | layers | `attn_q` | `attn_k` / `attn_v` | `attn_output` |
|---|---|---|---|---|
| sliding-window | 35 | [2560, 2048] | [2560, 512] | [2048, 2560] |
| global | 7 | [2560, 4096] | [2560, 1024] | [4096, 2560] |

With `head_count = 8` / `head_count_kv = 2` everywhere, the per-head dim is
uniform within each layer (query, key, and value share it):

- sliding-window layers: head_dim = 256 (8 q-heads × 256, 2 kv-heads × 256)
- global layers:        head_dim = 512 (8 q-heads × 512, 2 kv-heads × 512)

So the global layers double the head width (and use `rope.freq_base` 1e6) rather
than adding query heads. There is no q/k/v head-dim asymmetry — a single
`head_dim` per layer is enough for the attention kernel.

## Tensor inventory (dtype histogram)

f32 × 423, q3_K × 169, q4_K × 123, q5_K × 4, bf16 × 1. All quantized weights are
within SonicBoom's supported set (q3_K/q4_K/q5_K); the single bf16 tensor is
`per_layer_model_proj.weight`.

## Non-block tensors

| tensor | dtype | dims |
|---|---|---|
| `token_embd.weight` | q3_K | [2560, 262144] |
| `output_norm.weight` | f32 | [2560] |
| `per_layer_token_embd.weight` | q4_K | [10752, 262144] |
| `per_layer_proj_norm.weight` | f32 | [256] |
| `per_layer_model_proj.weight` | bf16 | [2560, 10752] |
| `rope_freqs.weight` | f32 | [256] |

Notes:
- 10752 = 42 × 256: `per_layer_token_embd` packs one 256-dim embedding per
  block; `per_layer_model_proj` (bf16) projects the per-layer embedding into
  the 2560-dim stream.
- `rope_freqs.weight` [256] is not a plain frequency table: it is
  `[64 × 1.0, 192 × 1e30]` — a `freq_factors`-style mask (the `1e30` sentinel
  disables a rotation frequency). Its exact role in RoPE is unresolved
  (design/gemma4-semantics.md §3.9).
- No `output.weight`: logits are produced through the per-layer projection
  path + `output_norm`.

## Per-block tensors (every block, layers 0..41)

Each block has **17** tensors (verified by direct dump of the tensor directory,
not 9/10 as earlier Phase 5A drafts assumed). Grouped semantically:

| group | tensors | dtype | dims |
|---|---|---|---|
| norms ×4 | `attn_norm.weight` (input) | f32 | [2560] |
| | `post_attention_norm.weight` | f32 | [2560] |
| | `ffn_norm.weight` (pre-ffn) | f32 | [2560] |
| | `post_ffw_norm.weight` | f32 | [2560] |
| attention | `attn_q.weight` | q3_K | [2560, 2048 or 4096] |
| | `attn_k.weight` | q3_K | [2560, 512 or 1024] |
| | `attn_v.weight` | q4_K / q5_K | [2560, 512 or 1024] |
| | `attn_output.weight` | q4_K | [2048 or 4096, 2560] |
| per-head norm | `attn_q_norm.weight` | f32 | [256] |
| | `attn_k_norm.weight` | f32 | [256] |
| ffn | `ffn_gate.weight` | q3_K | [2560, 10240] |
| | `ffn_up.weight` | q3_K | [2560, 10240] |
| | `ffn_down.weight` | q4_K / q5_K | [10240, 2560] |
| per-layer gate | `inp_gate.weight` | f32 | [2560, 256] |
| | `proj.weight` | f32 | [256, 2560] |
| | `post_norm.weight` | f32 | [2560] |
| output scale | `layer_output_scale.weight` | f32 | [1] |

This tensor set is **llama.cpp's Gemma 3 per-layer-embedding architecture**
(`general.architecture = "gemma4"` is the quantizer's label for it; the layout
is the Gemma 3 per-layer-input design, not HF `Gemma3TextModel` and not the
Gemma 3n "nano" AltUp/LAuREL variant). The Phase 5A loader binds only the
9 quantized + 2 norm + `layer_output_scale` tensors; the remaining 7 (`attn_q_norm`,
`attn_k_norm`, `post_attention_norm`, `post_ffw_norm`, `post_norm`, `inp_gate`,
`proj`) are **not yet bound** — see the semantics gap below.

### Mixed precision (`attn_v` / `ffn_down`)

`attn_v.weight` and `ffn_down.weight` are **not** a uniform dtype. Unsloth's
Q3_K_M keeps the first two blocks (`blk.0`, `blk.1`) at q5_K and quantizes the
remaining 40 (`blk.2`..`blk.41`) to q4_K:

- q5_K: `blk.{0,1}.attn_v.weight`, `blk.{0,1}.ffn_down.weight` (4 tensors)
- q4_K: `blk.{2..41}.attn_v.weight`, `blk.{2..41}.ffn_down.weight` (80 tensors)

Both dtypes are in SonicBoom's supported set, so the loader accepts either and
validates the shape; execution is dtype-agnostic (the quantized matmul reads the
per-tensor type). The blk.1 → blk.2 boundary is asserted in `test_gemma4`.
