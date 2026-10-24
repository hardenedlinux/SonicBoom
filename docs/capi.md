# Route 2 C ABI draft: the inference backend behind `(sonicboom serve backend)`

Status: the serving path is implemented — `(sonicboom serve real-backend)`
drives `libsonicboom.so` through these functions. §2 keeps the design's fuller
declaration list; what the path does not use yet is listed in §6.

This document maps the backend protocol of the Guile inference service
(`bindings/guile/scheme/sonicboom/serve/backend.scm`) to the C ABI that
`libsonicboom.so` exports, so the mock backend can be replaced by a
libsonicboom one without changing the service.

"Route 2" is Path 2 of the execution-path plan: GGUF → model loader →
planner spine → native-torch / hand-written CUDA / MLIR kernels (Gemma 4 E4B
today). The functions below are a facade over that path. They follow
`capi/include/sonicboom/capi.h` and the C API rules of `CLAUDE.md`:

- pure C: opaque handles, C scalars, caller-provided buffers; no C++ classes,
  templates, STL, exceptions, RTTI; no native-torch / ATen / c10 / CUDA / MLIR
  types;
- every fallible call returns `sb_status` and takes `sb_error **err_out` with
  the ownership discipline already defined in `capi.h`;
- the implementation lives in `capi/src/` and calls the Layer 2 / model API of
  the core; the Guile binding calls these functions only, never `core/` or
  `third_party/native-torch/`.

The prefix `sb_llm_` keeps these functions apart from the S-Expr graph API
(`sb_parse` / `sb_compile` / …) already in `capi.h`.

## 1. Protocol → ABI map

| `(sonicboom serve backend)` | C ABI | notes |
|---|---|---|
| `load-model` | `sb_llm_model_load`, `sb_llm_model_free` | GGUF path + config struct |
| `model-info` | `sb_llm_model_get_info` | id, family, context length, vocabulary size, EOS ids |
| `tokenize` | `sb_llm_tokenize` | special tokens parsed as whole tokens |
| `detokenize` | `sb_llm_detokenize` | UTF-8 bytes; may end inside a character |
| `render-chat` | **none** | stays in Scheme (`serve/chat-template.scm`); the ABI only reports the family |
| `create-session`, `free-session` | `sb_llm_session_new`, `sb_llm_session_free` | a KV cache + its token history |
| `fork-session` (optional) | `sb_llm_session_fork` | copies the first n positions; may return `SB_ERR_UNSUPPORTED` |
| `prefill` | `sb_llm_prefill` | returns the number of reused tokens |
| sampler (`make-sampler`) | `sb_llm_sampler_new`, `sb_llm_sampler_free` | parameters + random state |
| `step` | `sb_llm_step` | sample, append, return the token |
| constraint hook | `sb_llm_session_set_grammar`, `sb_llm_session_set_json_schema`, `sb_llm_session_clear_grammar` | installed on the session, restarted by each prefill |
| `backend-stats` | `sb_llm_model_get_stats` | prefill / reuse / step counters |

Chat templates stay out of the ABI on purpose: GGUF carries the template as
Jinja text (`tokenizer.chat_template`), which SonicBoom does not interpret; the
service renders by family with hand-written Scheme templates. The ABI reports
the family (from `general.architecture` and the template, e.g. `gemma`, `qwen`)
and the service picks the template.

## 2. Declarations

```c
/* Opaque handles. */
typedef struct sb_llm_model sb_llm_model;      /* weights, plan, tokenizer      */
typedef struct sb_llm_session sb_llm_session;  /* a KV cache and its tokens     */
typedef struct sb_llm_sampler sb_llm_sampler;  /* sampling parameters + RNG     */

typedef int32_t sb_token;                      /* a vocabulary index            */

/* Status codes appended to sb_status (appending keeps the ABI). */
/*   SB_ERR_CONTEXT     = 7   the session would exceed its context length      */
/*   SB_ERR_CONSTRAINT  = 8   the constraint callback failed or allowed nothing */
/*   SB_ERR_UNSUPPORTED = 9   an optional function this backend doesn't have   */

typedef enum sb_llm_device {
  SB_LLM_DEVICE_CPU = 0,
  SB_LLM_DEVICE_CUDA = 1,
} sb_llm_device;

/* Versioned by size: the caller sets struct_size = sizeof(...); fields are
 * only ever appended, and the implementation reads the ones that fit. */
typedef struct sb_llm_model_config {
  size_t struct_size;
  sb_llm_device device;
  uint64_t max_context;     /* 0: the model's own context length            */
  uint32_t n_threads;       /* CPU threads, 0: default                       */
} sb_llm_model_config;

typedef struct sb_llm_model_info {
  size_t struct_size;
  const char *id;           /* e.g. the GGUF general.name; owned by the model */
  const char *family;       /* "gemma", "qwen", ...; owned by the model       */
  uint64_t context_length;
  uint32_t vocab_size;
  const sb_token *eos;      /* tokens that end a reply; owned by the model    */
  size_t n_eos;
} sb_llm_model_info;

typedef struct sb_llm_sampler_params {
  size_t struct_size;
  float temperature;        /* 0: greedy                                     */
  float top_p;              /* 1: off                                        */
  uint32_t top_k;           /* 0: off                                        */
  uint64_t seed;
  int has_seed;             /* 0: seeded from a clock                        */
} sb_llm_sampler_params;

/* Constraint: fills mask[0 .. vocab_size) with 1 for allowed tokens, 0 for
 * the others, given the tokens generated so far by this request. Returns 0,
 * or non-zero to fail the step (SB_ERR_CONSTRAINT). */
typedef struct sb_llm_constraint {
  size_t struct_size;
  void *user;
  int (*allow)(void *user, const sb_token *generated, size_t n_generated,
               uint8_t *mask, size_t vocab_size);
} sb_llm_constraint;

typedef struct sb_llm_stats {
  size_t struct_size;
  uint64_t prefill_calls, prefill_tokens, reused_tokens, computed_tokens;
  uint64_t steps, sessions_live, forks;
} sb_llm_stats;

/* Model */
sb_status sb_llm_model_load(const char *gguf_path, const sb_llm_model_config *config,
                            sb_llm_model **model_out, sb_error **err_out);
void sb_llm_model_free(sb_llm_model *model);
sb_status sb_llm_model_get_info(const sb_llm_model *model, sb_llm_model_info *info_out,
                                sb_error **err_out);
sb_status sb_llm_model_get_stats(const sb_llm_model *model, sb_llm_stats *stats_out,
                                 sb_error **err_out);

/* Tokenizer. Two-call sizing like sb_retrieve_output: with cap too small the
 * call returns SB_ERR_BINDING and writes the needed count to *n_out. */
sb_status sb_llm_tokenize(const sb_llm_model *model, const char *text, size_t len,
                          sb_token *out, size_t cap, size_t *n_out, sb_error **err_out);
sb_status sb_llm_detokenize(const sb_llm_model *model, const sb_token *tokens, size_t n,
                            char *out, size_t cap, size_t *len_out, sb_error **err_out);

/* Sessions */
sb_status sb_llm_session_new(sb_llm_model *model, uint64_t n_ctx,
                             sb_llm_session **session_out, sb_error **err_out);
void sb_llm_session_free(sb_llm_session *session);
sb_status sb_llm_session_fork(const sb_llm_session *session, uint64_t n_tokens,
                              sb_llm_session **session_out, sb_error **err_out);

/* Make the session hold exactly tokens[0 .. n); keep the longest common prefix
 * with what it holds, compute the rest. *n_reused_out: tokens kept. */
sb_status sb_llm_prefill(sb_llm_session *session, const sb_token *tokens, size_t n,
                         uint64_t *n_reused_out, sb_error **err_out);

/* Sampler */
sb_status sb_llm_sampler_new(const sb_llm_sampler_params *params,
                             sb_llm_sampler **sampler_out, sb_error **err_out);
void sb_llm_sampler_free(sb_llm_sampler *sampler);

/* Sample the next token from the logits after the session's last token
 * (constraint may be NULL), append it (run its forward step, so its KV and
 * the next logits exist), return it. */
sb_status sb_llm_step(sb_llm_session *session, sb_llm_sampler *sampler,
                      const sb_llm_constraint *constraint, sb_token *token_out,
                      sb_error **err_out);
```

## 3. Semantics

- **Session = one sequence.** A session holds a token history at positions
  `0 .. n-1` and their KV entries, exactly like `planner::KvCacheSession`
  (`filled_tokens` + device caches) used by
  `generate_prefill_spine_cuda_prefix` today. `sb_llm_prefill` is that
  function's prefix matching (`common_prefix_len`), split out of the generate
  loop: truncate to the common prefix, prefill the suffix, leave the logits of
  the last position ready.
- **`sb_llm_step` appends.** Sampling reads the current logits; the sampled
  token is then run through one decode step. The token after an end-of-turn is
  never needed, but that last step is kept: it puts the end-of-turn token into
  the cache, which is exactly what the next turn of the conversation starts
  with (the service's multi-turn prefix reuse relies on it).
- **Fork** copies positions `0 .. n_tokens-1` (KV slots and history) into a new
  session, on the device. With a paged KV cache it becomes block sharing with
  copy-on-write; the ABI does not change. Backends without it return
  `SB_ERR_UNSUPPORTED`; the session pool then takes a session over instead.
- **Sampler state.** Temperature 0 is greedy (`argmax`, today's only mode in
  the spine). The random state lives in the sampler, one per request, so a
  seed reproduces a request on the same build and device. The GPU sampler may
  use a counter-based generator (Philox); its stream need not match the CPU
  one, in line with the v1 rule that GPU numerics need not be bit-identical.
- **Constraint (constrained decoding).** A session may carry a grammar
  (`sb_llm_session_set_grammar`, `sb_llm_session_set_json_schema`). The
  grammar consumes the text a token spells — the same bytes
  `sb_llm_detokenize` emits — and the allowed-token bitmap is applied to the
  logits on the device by the argmax kernel. The model's EOS ids are the
  grammar's stop tokens: they are allowed only where the grammar can end, and
  a stop that finishes it ends the output and is not emitted. The grammar
  restarts at every `sb_llm_prefill`, so installing it before the prefill also
  masks the turn's first token. `sb_llm_json_schema_to_gbnf` is the schema
  compiler alone, with optional literal PREFIX/SUFFIX (how the server composes
  the `<tool_call>` envelope); unsupported schema keywords are rejected by
  name, never silently ignored. A bare schema compiles to JSON text (RFC 8259),
  so up to 16 bytes of whitespace may precede the value and none may follow it
  (a finished value leaves only the stop tokens); an envelope is matched
  exactly. `sb_llm_session_clear_grammar` returns to plain
  greedy decoding. Installing the grammar a session already holds resets it in
  place instead of recompiling, so a server that applies one constraint turn
  after turn keeps its vocabulary trie and the masks of the states it has
  visited warm (the first visit to a permissive state walks the whole trie).
  An unconstrained decode uses the same captured (masked) argmax kernel with an
  all-ones buffer, so it is bit-identical to the unmasked path. Without a CUDA device the CPU fallback recomputes the mask per token
  and re-prefills the history (O(n^2)) — the CPU reference path's price, not
  the performance target.
- **Context.** `prefill` or `step` past `n_ctx` returns `SB_ERR_CONTEXT`
  and leaves the session unchanged.

## 4. Ownership and threading

- Every `*_new` / `*_load` / `*_fork` returns a handle the caller owns and
  releases with the matching `*_free`; `*_free(NULL)` is a no-op.
- **Sessions and samplers keep their model alive.** The model is reference
  counted inside the library: `sb_llm_model_free` drops the caller's reference,
  and the weights go when the last session is freed too. Garbage-collected
  bindings (Guile guardians / finalizers) do not free in a guaranteed order, so
  the ABI must not require "sessions before the model".
- Strings and arrays in `sb_llm_model_info` belong to the model and stay valid
  until it is destroyed. Input buffers are borrowed for the duration of the
  call; outputs go to caller buffers.
- A session or sampler is used by one thread at a time. Sessions of one model
  may be called from different threads; device work is serialized inside the
  library (single device). The service serializes requests anyway (its queue).
- Errors: the `capi.h` discipline unchanged (caller frees `sb_error`, reused
  slots are freed before being overwritten, no exception crosses the ABI).

## 5. Guile side

`(sonicboom serve real-backend)` builds a `make-backend` record from these
functions through `(system foreign)` (`bindings/guile/scheme/sonicboom/ffi.scm`
plus the `(sonicboom inference)` wrappers):

- handles become pointers wrapped with finalizers (safe thanks to §4's
  reference counting);
- `tokenize` / `detokenize` use the two-call sizing;
- `detokenize` returns bytes; the binding decodes UTF-8 with replacement, and
  the engine already holds back a trailing U+FFFD until the next token;
- constrained decoding is `set-grammar!` / `set-json-schema!` /
  `json-schema->gbnf` on the session; the serve layer passes its grammar
  record to the prefill of each turn, and a `#f` from `step` (the grammar's
  stop token ended the reply) becomes `finish_reason: stop`.
- a procedure constraint `(candidates) -> candidates` is applied by the mock
  backend's sampler on the host.

## 6. What the core provides

| the serving path uses | today |
|---|---|
| persistent sessions | `sb_llm_session_new` over `planner::KvCacheSession`; `sb_llm_prefill` keeps the longest common prefix |
| sampling (temperature, top-k, top-p, seed) | `sb_llm_decode` |
| logit mask before sampling | the session's compiled grammar; the argmax kernel applies its bitmap |
| tokenizer and EOS ids from GGUF metadata | `sb_llm_tokenize(_ex)`, `sb_llm_detokenize`, `sb_llm_model_eos` |

Not in the ABI yet: session fork (`sb_llm_session_fork`), a per-step sampler
(`sb_llm_sampler_new` / `sb_llm_step` — `sb_llm_decode` runs a request's steps
through a token callback instead), and stats counters. The service above the
backend supplies its own request statistics.
