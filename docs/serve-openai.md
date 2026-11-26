# OpenAI-compatible local inference service (Guile + GNU Artanis)

Status: **prototype**, with two implementations of the backend protocol. The
default `--backend real` drives a GGUF through libsonicboom's C ABI (the Guile
binding), and `--backend mock` is the deterministic in-process backend the
tests use. The application layer of the production inference route (Route 2 /
Path 2): an HTTP service speaking the OpenAI
chat-completions API, written in Guile on GNU Artanis. The backend interface is
the one a libsonicboom backend implements; its C ABI is drafted in
[`capi.md`](capi.md).

## 1. Layout

```text
bindings/guile/scheme/sonicboom/serve/
├── backend.scm        the backend protocol (record of procedures), the sampler
├── real-backend.scm   the libsonicboom backend, through the Guile binding
├── mock-backend.scm   a deterministic mock implementing the protocol
├── chat-template.scm  hand-written chat templates: gemma, qwen; tool-call text
├── sessions.scm       session pool: prefix matching, fork, LRU, token budget
├── engine.scm         FIFO request queue (one device) + the generation loop
├── residency.scm      the model catalog and the one-model-on-the-device switch
├── openai.scm         request validation, response / chunk / error objects
├── sse.scm            a minimal SSE writer on Artanis (no Artanis change)
└── app.scm            the single-file Artanis app: the HTTP layer only

bindings/guile/tests/serve/
├── test-serve-units.scm    below HTTP (templates, mock, sampler, pool, engine)
├── test-serve-http.scm     end to end with Guile's HTTP client, raw SSE, WebSocket
├── test-serve-images.scm   POST /v1/images/generations against the real model
└── test-openai-sdk.py      the official openai Python SDK against the app
```

Dependencies point down: `app.scm` → `openai`, `engine`, `residency`, `sse` →
`sessions` → `backend` ← `mock-backend`, `real-backend`. Only `app.scm` and `sse.scm` import
Artanis; the other modules need Guile and guile-json (4.7.3, built from
upstream source) only. `real-backend.scm` reaches libsonicboom through
`(sonicboom inference)`, which sits on the C ABI; it never touches the core or
native-torch directly.

## 2. Backend protocol

`(sonicboom serve backend)`: `make-backend` takes the procedures

| procedure | contract |
|---|---|
| `load-model config` | → model handle |
| `model-info model` | alist: `id`, `family`, `context-length`, `eos-tokens` |
| `tokenize model text` / `detokenize model tokens` | special tokens are single tokens |
| `render-chat model messages tools tool-choice` | → prompt text (both backends use `chat-template.scm`) |
| `create-session model` / `free-session session` | a session is a KV cache holding one token sequence |
| `fork-session session n` (optional) | a new session holding the first n tokens, without recomputing them |
| `prefill session tokens [constraint]` | the session holds exactly `tokens` afterwards; the longest common prefix with what it held is kept; a grammar constraint starts at this turn's first token; → number of reused tokens |
| `step session sampler constraint` | sample the next token, append it, return it; `#f` when the constraint has closed the reply |
| `generate session sampler constraint max-tokens on-token` (optional) | decode up to `max-tokens` in one call, calling `(on-token token)` per produced token; `#f` from `on-token` stops the decode; → number of tokens produced |
| `stats model` | counters (optional) |

A **sampler** (`make-sampler #:temperature #:top-p #:top-k #:seed
#:presence-penalty #:frequency-penalty`) carries one request's parameters and
random state; temperature 0 is greedy. The reference logits sampler
(`sample-from-logits`) applies temperature, top-k and top-p. The libsonicboom
backend passes the whole sampler through on every step: each `step` is one
`boom_llm_decode` call (`max_tokens` 1) on the persistent session, and the core's
sampled path keeps its draw state across calls, so the per-step stream matches
a one-shot decode with the same parameters. `presence_penalty` and
`frequency_penalty` reach the C ABI's sampled path; the mock sampler has no
penalty model.

The real backend also implements `generate`: one `boom_llm_decode` call covers
the whole reply, and its token callback drives the engine's per-token work
(the session history, the text filter, cancellation). A reply is therefore one
decode call rather than one per token, which is what speculative decoding
needs. `SONICBOOM_SERVE_STEP=1` selects the per-token `step` loop instead; the
two paths produce the same content, finish reasons and usage.

A **constraint** shapes the reply. There are two forms:

- a record (`make-json-schema-constraint`, `make-tool-call-constraint`,
  `make-gbnf-constraint`) naming a grammar. The real backend compiles and
  installs it as the session's grammar before the prefill, so the turn's first
  token is masked too; every token must keep the reply's text inside the
  grammar's language, and a stop token ends the reply where the grammar can
  end. `step` then returns `#f` and the engine reports `finish_reason: stop`.
  The mock ignores grammar records — it has no decoder — and answers
  unconstrained.
- a procedure `(candidates) → candidates`, a token filter applied by the
  mock's host sampler before sampling; it is the form the mock's tests use.
  The real backend samples on the device (there is no host candidate list), so
  it rejects a procedure constraint instead of silently ignoring it.

`prefill` resets the grammar, so a constraint is per turn: the engine passes it
to both `prefill` and every `step` of that turn.

`sample-from-logits` is a reference sampler for backends that produce logits
on the host (the mock uses it; the real backend's C ABI returns the sampled
token).

### The real backend

`real-backend.scm` loads one GGUF through `(sonicboom inference)` (the default
is `models/qwen2.5-7b-instruct-1m-q4_k_m.gguf` served as `qwen2.5-7b`, with a
context length of 16384 and one pooled session; `--model` / `--model-id` /
`--context-length` / `--max-sessions` override them).
The chat-template family is selected from the GGUF architecture (`gemma4` →
`gemma`, `qwen2` and `qwen35moe` → `qwen`), stop tokens are the union of the model's EOS ids
and the template's end-of-turn markers, and sessions are KV-cache handles from
`make-session` / `prefill!` / `generate-tokens`. `--backend real` is the
default; `tools/test-api-server.sh` is the committed smoke test for this path,
because the unit and HTTP suites run the mock and need no model file.

### The mock backend

Deterministic, so the tests can predict every byte:

- tokenizer: one token per character (`256 + code point`), the family's
  special tokens and `<tool_call>`, `</tool_call>` as single tokens;
- the reply is planned from the prompt text at the first step after a prefill:
  - the last user turn holds tool results → `Tool result: <result>`;
  - tools are offered and the last user line names one (its name, or the last
    part of a snake_case name: "weather" for `get_weather`), or the prompt
    demands a call → one `<tool_call>` with arguments filled from the line
    (strings: the last word; numbers: the first number);
  - otherwise → `You said: <last line of the last user turn>`;
  - then the end-of-turn token;
- logits: the planned token 8, the successors of the previous character in a
  small character bigram table `log(1 + count)` (≤ ~3), so greedy gives the
  plan exactly and seeded sampling is reproducible but may stray;
- `step-delay` / `prefill-delay` (seconds per token) simulate device time;
- counters: prefill calls / tokens, reused and computed tokens, steps,
  sessions, forks.

## 3. Chat templates

Hand-written Scheme per family, no Jinja (`chat-template.scm`):

- **gemma**: `<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`;
  no system role, the system text (and the tools block) open the first user
  turn;
- **qwen**: ChatML `<|im_start|>role\n…<|im_end|>\n`. A `qwen35moe`
  (Qwen3.5/3.6) model gets the empty `<think>\n\n</think>\n\n` block after
  `<|im_start|>assistant\n`, as its own template renders with thinking off:
  the server has no reasoning-content channel, and this is the byte-for-byte
  prompt llama.cpp's template produces with `enable_thinking: false`. The
  block is text, not control tokens.

The engine can record what it tokenized: with `SONICBOOM_SERVE_PROMPT_TRACE`
set to a directory, every accepted request writes the rendered prompt bytes to
`<n>.txt` and the token ids prefill receives to `<n>.ids` there. It is off
unless set; it is meant for cross-engine prompt-identity checks.

Tools follow the Qwen2.5 convention for both families: a `# Tools` section
listing one JSON signature per line inside `<tools></tools>`, and calls as
`<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>`. Tool results go
back in a user turn as `<tool_response>\n…\n</tool_response>`. Gemma has no
tool tokens of its own; using that wording for it is SonicBoom's convention.
`tool_choice` `none` hides the tools; `required` or a named function adds a
line demanding the call (with a real model, a constraint should enforce it).

An assistant message is rendered exactly as the model produced it, so the
next turn's prompt starts with the tokens the session already holds.

## 4. Sessions and the prefix cache

`(sonicboom serve sessions)`, one pool per model, `--max-sessions` sessions
and a `--token-budget` of held tokens. There is one pool per model, and only
the model resident on the device (section 6) has one: switching models drops
the pool it left, so prefix caches do not survive a switch. The pool mirrors each session's tokens.
For a prompt, the idle session with the longest common prefix (least recently
used on ties) is the best match:

- the prompt extends everything it holds → that session (multi-turn
  continuation: the whole previous turn is reused);
- a partial match and room for one more session → **fork** the prefix into a
  new session, so the matched one keeps its own continuation (two
  conversations with the same system prompt both keep their caches);
- no match and room → a new session;
- no room → the best match is taken over (overwritten after the prefix).

After a request the session holds the prompt and every sampled token (the
end-of-turn included). Then LRU idle sessions are freed until the held tokens
fit the budget. `usage.prompt_tokens_details.cached_tokens` is the backend's
`prefill` result.

## 5. Queue and generation

`(sonicboom serve engine)`:

- **Queue.** One device: one unit of work runs at a time, in arrival order —
  a chat completion, an image generation, or an explicit model load/unload.
  `engine-admit!` (in the HTTP handler, before any response is written) takes
  a place or raises `queue-full` → HTTP 429 `rate_limit_error` / `queue_full`
  when `--queue` requests already wait. `engine-run` waits for the turn and
  always leaves the queue; `engine-drop!` releases a place never run.
- **Prepare.** When a request's turn comes, its model is made resident
  (`engine`'s prepare hook calls `residency-ensure!`) before it runs, so a
  switch never happens under a running generation. Chat requests name the
  model they need; image requests and the load/unload endpoints run as jobs
  on the same FIFO (`engine-run-job`).
- **Loop.** Render, tokenize (a prompt at the context length → 400
  `context_length_exceeded`), take a session, prefill, then step until an
  end-of-turn token, `max_tokens`, the context length, a stop string, or
  cancellation.
- **Text filter.** Stop strings end the reply and are not emitted;
  `<tool_call>…</tool_call>` spans become tool calls. Text that may be the
  beginning of a stop string or of `<tool_call>` is held back until decided,
  so no partial marker ever reaches the client. A trailing U+FFFD (an
  incomplete multi-byte character from a byte-level tokenizer) also waits.
- **Events.** `(on-event 'content text)` and `(on-event 'tool-call index id
  name arguments)`; a `#f` result cancels: the client is gone.
- **finish_reason**: `stop`, `length`, `tool_calls` (a call was made and the
  model ended its turn).

## 6. HTTP layer

`app.scm` is a single-file Artanis app in the style of the Artanis examples:

| Artanis example | used for |
|---|---|
| `examples/test.scm` | single-file app shape: `(init-server)`, `get` / `post` routes, `:id` route parameters, `#:mime 'json` + `(:mime rc obj)` for JSON, `response-emit` for other statuses, `(run …)` |
| `examples/ENTRY.websocket` | `#:websocket 'raw` routes returning an `on-message` dispatcher, `ws-send`, `ws-message-text`, `#:timeout 0`, `call-with-runner` for blocking work |
| `examples/protocols/chat.scm` | JSON in text frames (the service encodes chunk JSON itself on a `raw` route instead of installing an app protocol, which needs an `art` project directory) |

Endpoints:

| endpoint | |
|---|---|
| `GET /v1/models`, `GET /v1/models/:id` | the served chat models: one real model (default `qwen2.5-7b`), or under `--backend mock` the two mock models (`sonicboom-mock-gemma`, `sonicboom-mock-qwen`). Each object carries `loaded`; `family` is `null` until the model has been loaded once, and stays known after it is unloaded |
| `GET /sonicboom/models` | the whole catalog, chat and image: per entry `id`, `kind` (`chat`/`image`), `arch`, `loaded`, `vram_mib`, `last_load_ms`, `last_switch_ms` (not OpenAI) |
| `POST /sonicboom/models/:id/load`, `POST /sonicboom/models/:id/unload` | make a catalog model resident, or free it; queued like a generation, 404 `model_not_found` for an unknown id (not OpenAI) |
| `POST /v1/chat/completions` | `stream: false` → `chat.completion`; `stream: true` → SSE of `chat.completion.chunk`, then `data: [DONE]` |
| `POST /v1/images/generations` | MOSS-Image: `{"created", "data": [{"b64_json": …}]}`, each entry a PNG; the header `sonicboom-job-id` names the generation, `stream: true` answers with SSE progress instead |
| `DELETE /sonicboom/jobs/:id` | cancel a generation by job id: `{"id", "status": "cancelled"}`; 404 `job_not_found` for an unknown or finished job (not OpenAI) |
| `GET /v1/ws/chat` | WebSocket: one request (the same JSON body) per text message; the reply is the chunk objects as text messages, then `[DONE]`; a bad request gets an `error` object |
| `GET /sonicboom/styles` | the style assets: per entry `id`, `version`, `model`, `layer` (null for a card-only style), `vector` (whether it has one), `system` (whether it has one), `exemplars` (count) and `created`; the vector itself is not returned (not OpenAI; see "Style assets") |
| `POST /sonicboom/styles` | create (or replace) a style; queued like a generation because building a vector runs the model; 400 `invalid_style` for a bad body (not OpenAI) |
| `DELETE /sonicboom/styles/:id` | forget a style: `{"id", "deleted": true}`; 404 `style_not_found` for an unknown id (not OpenAI) |
| `GET /sonicboom/stats` | engine, pool and backend counters (not OpenAI) |
| `POST /sonicboom/replay/:replay_id` | regenerate a stored reply from its request body and seed; 404 `replay_not_found` when the id is unknown or no replay store is configured (not OpenAI; see `docs/reproducible-api.md`) |
| `GET /sonicboom/history` | a minimal HTML page of recent stored replies with seed, model, time and replay/copy-seed links; empty without a replay store (not OpenAI) |
| `GET /health` | `ok` |

Request fields: `model`, `messages` (string or text-part content; roles
`system`, `developer`, `user`, `assistant` with `tool_calls`, `tool`), `stream`,
`stream_options.include_usage`, `max_tokens` / `max_completion_tokens`,
`stop` (string or up to 4), `temperature`, `top_p`, `top_k` (extension),
`min_p` (extension), `seed`, `repeat_penalty` (extension), `presence_penalty`,
`frequency_penalty`, `n` (1 only), `tools`, `tool_choice`,
`response_format`, `sonicboom` (the extension object: `{"style": {"id",
"version", "alpha"}}`, see "Style assets"). Unknown fields are ignored; a
sampling value out of range is a 400 naming the parameter.

Omitted sampling defaults differ from the OpenAI service: **an omitted
`temperature` is 0 (greedy, not 1.0)**, so a request that says nothing samples
nothing and repeats exactly. `top_p` defaults to 1, `top_k` and `min_p` to 0
(off) and `repeat_penalty` to 1 (off); `presence_penalty` and
`frequency_penalty` default to 0 and are validated to [-2, 2]. Accepted
ranges: `temperature` [0, 2], `top_p` and `min_p` [0, 1], `top_k` a
non-negative integer, `repeat_penalty` ≥ 0 (0 is read as its 1 default).

A seed is only meaningful when `temperature` > 0. A sampling request that
omits `seed` gets a fresh one; a greedy request ignores an explicit seed. The
reply carries the seed as the top-level `seed` field (`null` for greedy), on
the completion object and on the first and final chunks of a streamed reply,
and a sampled reply also has the `X-SonicBoom-Seed` response header. Every
reply also carries a `sonicboom` object — `fingerprint`, `replay_id` and
`guarantee` — naming the scope inside which the same request and seed are
bit-identical. Replaying the request with that `seed` reproduces the reply,
including across a restart of the server on the same machine and build.
`docs/reproducible-api.md` defines the fingerprint and the replay store;
briefly, `--replay-store DIR` (off by default) keeps the request body locally,
`POST /sonicboom/replay/:replay_id` regenerates a stored reply, and
`GET /sonicboom/history` lists recent replies.

With the real backend, `temperature`, `top_p`, `top_k`, `min_p`, `seed`,
`repeat_penalty` and the two penalties are passed to `boom_llm_decode` on
every step (in the C ABI's order: penalties, then top-k / min-p / top-p, then
the draw). A request with `temperature` > 0 therefore samples; the penalties
and `repeat_penalty` are part of the C ABI's sampled path, so they take effect
only when `temperature` > 0 — the greedy default is a plain argmax and ignores
them. The mock backend honors `temperature`, `top_p`, `top_k` and `min_p`; it
has no penalty or repetition model, so `repeat_penalty` and the two penalties
are accepted and ignored there.

### Request log

`--request-log PATH` appends one JSON line per request:

```json
{"id": "req-6773a6c1-12", "time": 1795418553, "end": 1795418554,
 "method": "POST", "path": "/v1/chat/completions", "status": "done",
 "ms": 1214, "model": "qwen2.5-7b", "kind": "chat", "queue_ms": 3,
 "switch_ms": 812, "prompt_tokens": 42, "completion_tokens": 128,
 "cached_tokens": 0, "size": null, "steps": null, "n": null,
 "free_mib_start": 5210, "free_mib_end": 5188,
 "seed": 2871560934, "fingerprint": "03d4602d080e8a6e",
 "replay_id": "rp-4bd0f1a2c3d4e5f6"}
```

- `id` names the request; `time` and `end` are its start and finish in Unix
  seconds; `ms` is the whole request.
- `status` is `done`, `cancelled` (the client went away, or the job was
  cancelled), `failed` (any other error response) or `rejected` (the queue was
  full, a 429).
- `model` and `kind` (`chat` or `image`) say which model the request was for.
- `prompt_tokens`, `completion_tokens` and `cached_tokens` are the chat
  result's usage. `size` (`[width, height]`), `steps` and `n` are an image
  request's shape. Fields that do not apply are `null`.
- `queue_ms` is how long the request waited for the device (null if it never
  got its turn); `switch_ms` is the model switch made for it, when one
  happened (null otherwise).
- `free_mib_start` and `free_mib_end` are the device memory free at the
  request's start and end, read from `nvidia-smi`; null when it cannot be
  read.
- `seed` is the sampling seed the reply used (`null` for greedy),
  `fingerprint` is the reply's fingerprint id, and `replay_id` names the
  reply. None of them contains request content.

Request and response bodies are never written — no prompt, no conversation,
no image description — so prompts stay out of the file. A streaming request's
line is written when its stream ends. An unusable path fails at startup, not
at the first request. The WebSocket route is not logged: its handler is a
long-lived callback factory, not a response.

### Constrained replies

Two request fields compile to a grammar constraint on the real backend:

- `response_format: {"type": "json_schema", "json_schema": {"schema": …}}`
  constrains the reply to one JSON value of that schema. `{"type":
  "json_object"}` constrains it to any JSON object; `{"type": "text"}` and an
  absent field are unconstrained. The supported schema subset is the one in
  `core/include/sonicboom/vocab/grammar.h`; an unsupported keyword (pattern,
  format, not, …) is a 400 naming the keyword, not a silently ignored
  constraint.
- A forced tool choice (`tool_choice: "required"`, or the named form
  `{"type": "function", "function": {"name": …}}`) constrains the reply to the
  tool-call envelope the chat template already understands:
  `<tool_call>` newline `{"name": …, "arguments": …}` newline `</tool_call>`.
  The arguments object is the tool's `parameters` schema; `required` allows
  any of the offered tools (`anyOf`). The tool-call parser then goes through
  the same path as an unconstrained call, so the result is an ordinary
  `tool_calls` reply.

`response_format` wins when both are given; `tool_choice: "auto"` and
`"none"` do not constrain. A bare JSON reply is JSON text (RFC 8259): up to 16
bytes of whitespace may precede the value, so a model that opens its answer
with a newline keeps that choice; nothing may follow the value, so the reply
ends as soon as it is complete instead of padding whitespace up to the bound.
A tool-call envelope is exact on both sides. Greedy decoding is unchanged
wherever the unconstrained token is allowed, so a model that answers with
valid JSON produces the same tokens it would have produced anyway.

### Style assets

A **style** is a named, versioned writing style a request can ask for with
`"sonicboom": {"style": {"id": …, "version": …, "alpha": …}}`. It has two
independent halves:

- **card** — a system text plus up to 8 user/assistant exemplar turns, put in
  front of the request's own messages. They are a fixed prefix, so the session
  pool reuses them across requests the same way it reuses a conversation.
- **vector** — a prefill-time steering vector: the mean residual stream at one
  block over the exemplar replies, minus the same mean over a fixed set of
  neutral reference sentences (plain factual prose, in
  `serve/styles.scm`). The difference is added to the residual stream right
  after that block during prefill and every decode step; `alpha` scales it per
  request. The registry stores the unscaled vector.

A style created without `layer` has only a card. A vector needs a real backend
(residual capture and the embedding width; the mock backend cannot build one),
and building it runs one prefill per exemplar and reference text on a throwaway
cache — so creation goes through the engine's queue, with the model resident.
The registry is in memory, like the replay store; it does not survive the
process. Creating the same id again replaces it; a request that names a
different `version` than the registry holds is a 409
`style_version_mismatch`, an unknown id a 404 `style_not_found`, and a style
measured under another model a 400 `style_model_mismatch` (a steering vector
is a property of one model's residual space).

Create a style:

```http
POST /sonicboom/styles
{"id": "plain", "version": 1, "model": "qwen2.5-7b",
 "system": "Write in short, plain sentences. State facts directly.",
 "exemplars": [
   {"prompt": "Describe the moon.",
    "reply": "The moon is a rock in the sky. It has no light of its own."},
   {"prompt": "Explain rain.",
    "reply": "Rain is water that falls from clouds. It waters the plants."}],
 "layer": 20}
```

Omit `layer` for a card-only style (the same request without it), and `system`
or `exemplars` as wanted — a style needs at least one of them. Use it:

```http
POST /v1/chat/completions
{"model": "qwen2.5-7b", "messages": [{"role": "user", "content": "Explain what a river is."}],
 "sonicboom": {"style": {"id": "plain", "alpha": 0.1}}}
```

`alpha` defaults to 1.0 (the vector as measured) and is validated to
[-10, 10]; a negative value steers away from the style. The first request
under a style computes the card prefix; later ones keep it cached
(`cached_tokens` shows it). Changing a session's style (including clearing it)
retires its KV prefix, so the next request re-prefills from position 0.

Generation is deterministic for a given prompt, seed and style, and **identical
with speculative decoding on or off** — a steered session takes strict decode
steps, because the speculative verify graphs carry no injection. In exchange,
a steered decode step is not CUDA-graph-captured today (the vector arrives as
a per-step host-to-device copy), which is where the speed cost below comes
from.

**Measured** (RTX 3050, greedy, 5 writing prompts, 64 new tokens max; drift =
1 − cosine between the final-layer mean residual of the new reply and the
exemplars' mean under the same model, so 0 is "the style's own register"):

| arm | Qwen2.5-7B, block 20 (drift, per prompt) | decode tok/s |
|---|---|---|
| no style | 0.19–0.25 | 40.8 |
| card only | 0.09–0.21 | 40.0 |
| card + vector, α 0.05 | 0.094–0.188 | 38.8 |
| card + vector, α 0.1 | 0.088–0.186 | 37.4 |

The exemplars' own mean sits 0.122 from the neutral references on this model.
Coherence holds to about **α 0.1 on Qwen2.5-7B block 20** (0.2 and above
degenerates into repetition); on Qwen3.6-35B-A3B block 20 (the Huihui demo)
α ≤ 0.3 stays coherent and 1.0 breaks; on Gemma-4-E4B block 28 the drift
metric saturates (the model's outputs sit close to the exemplars' register
either way: no style 0.015–0.026, card/vector 0.012–0.034), while the samples
still show the intended change from markdown lists to short plain sentences.
Treat the metric as a sanity check, not a tuner: pick α per model by reading
the output.

Samples, Qwen2.5-7B, prompt "Explain what a river is.":

- no style — "A river is a natural watercourse, typically flowing over land and
  carrying a significant amount of water. Rivers are formed by various
  processes…"
- card only — "A river is a stream of water that flows in a channel. It carries
  water from high ground to the sea."
- card + vector α 0.1 — "A river is a stream of water. It flows from one place
  to another."

The cost of the card prefix is one prefill per prompt at first use, then none
(reused). The vector costs ~3–7% of decode throughput on Qwen2.5-7B (40.8 →
37.4 tok/s at α 0.1; the whole loss is the uncaptured step — a plain run with
CUDA graphs off measures 2.5% slower than the captured one, and a steered run
measures the same either way). Reproducibility holds inside the usual scope:
with `temperature` 0.8 and the same seed, two styled requests are
byte-identical, and a new seed differs.

### One model on the device

The server keeps a catalog of the models it can serve — the chat model(s) and
the image model — and only one of them is on the device at a time (the KV
cache and the DiT's buffers do not fit together). The first chat model in the
catalog is loaded at startup; anything else loads on demand.

A request names its chat model; when that request reaches the front of the
queue the server unloads the resident model (its session pool, then the model
itself, then a device trim) and loads the requested one, measuring the switch.
A model that is not resident can still be requested: `/v1/models` lists it with
`loaded: false` and keeps reporting its `family` once it has been loaded once.
`POST /sonicboom/models/:id/load` and `.../unload` do the same switch
explicitly; both return the entry's info. `/sonicboom/models` shows the
catalog, what is loaded, how much VRAM the loaded model holds (`memory.free`
delta around the load, 0 when it cannot be measured), and the times of the
last load and last switch.

The queue is shared by chat, image and load/unload jobs, so a switch is never
made while another unit of work runs, and `--queue` bounds the whole device.

### Image generation

`POST /v1/images/generations` runs the MOSS-Image pipeline through
`(sonicboom image)` (C ABI `boom_image_*`). The model directory is
`--image-model` (default `models/MOSS-Image-SFT`); the image model is the
catalog entry `moss-image`, listed from startup, and loaded by the switch when
an image request reaches the front of the queue — so a chat-only run never
claims the DiT's device memory. The generation runs under the same queue as a
chat completion, one at a time (the pipeline's device buffers are reused in
place); it does not use a chat session pool.

| field | |
|---|---|
| `prompt` | required, non-empty |
| `n` | 1..10 (default 1); image *i* starts from seed + *i* |
| `size` | `"WIDTHxHEIGHT"`, both multiples of 16 (default `512x512`) |
| `response_format` | `"b64_json"` only (the default); anything else is a 400 |
| `seed`, `steps`, `guidance_scale`, `negative_prompt` | extensions (defaults 0, 50, 3.0, none) |
| `stream` | extension (default false): SSE progress, see below |

The response is `{"created": <unix time>, "data": [{"b64_json": <PNG>}, …]}`.
Each PNG is written by the C ABI's own writer (8-bit RGB, stored deflate
blocks) and base64-encoded in Scheme — Guile 3.0.9 has no `(ice-9 base64)`.
The same request twice returns the same pixels, because both the noise and
the sampler are seeded. A model that cannot be loaded, and a generation
failure, are 500s (`server_error`).

Every image reply, streamed or not, carries the header `sonicboom-job-id:
job-<16 hex digits>`, the id under which the generation can be cancelled (the
id is registered when the request is parsed, so it also names a request still
waiting in the queue). `stream: true` answers with Server-Sent Events:

```
data: {"step": 1, "total": 2, "image": 0}

data: {"created": 1795416250, "data": [{"b64_json": "…"}]}

data: [DONE]
```

One event per denoising step (the 1-based step, the total, and the image's
index in the batch), then the same images object the non-streamed reply holds
(a byte-identical `data` array for the same request), then `[DONE]`.

Cancellation stops a generation after the current denoising step; the C ABI's
progress callback returns the stop, so the pipeline's device buffers are
released before the request ends. Two ways to cancel:

- `DELETE /sonicboom/jobs/<id>` from any connection: 200
  `{"id": "<id>", "status": "cancelled"}`, or 404 `job_not_found` once the
  job has ended. A cancelled non-streamed request answers 409
  `invalid_request_error` / code `cancelled`; a cancelled stream ends with
  `data: {"id": "<id>", "status": "cancelled"}` and then `[DONE]`.
- The client of a streamed request goes away: the next event write fails and
  the generation is cancelled the same way (there is no one left to answer).
  A non-streamed request is not watched while it generates; cancel it by id.

Streaming: the first chunk carries `delta.role`, then content deltas (one per
token here), the tool-call deltas (a header with `index`, `id`, `type`, `name`
and empty `arguments`, then the arguments in pieces), a chunk with
`finish_reason`, the usage chunk (empty `choices`) when
`stream_options.include_usage` is set, then `[DONE]`. A tool call is emitted
once its closing tag is generated (its arguments are not parsed while still
open).

Errors use the OpenAI shape `{"error": {"message", "type", "param", "code"}}`:
400 `invalid_request_error`, 404 `model_not_found` / `job_not_found`, 409
`cancelled`, 429 `queue_full`, 500 `server_error`. In a stream that already
started, a chat error is sent as an `error` event followed by `[DONE]` (an
image stream ends with the cancelled object instead).

Cancellation: a streamed chat reply (SSE or WebSocket) stops at the next token
after the client goes away; a chat request is not cancellable by id. An image
generation is cancellable by id, and a streamed one also stops when its client
goes away (see "Image generation"). A non-streamed request is not watched
while it generates.

Streaming over Artanis: Artanis has no streaming body; `sse.scm` streams by
switching the connection to its own Ragnarok protocol for one response, and
the WebSocket replies are generated in a thread so the connection task can
write each chunk. Details, and the Artanis changes that would replace both
workarounds: [`artanis-streaming-notes.md`](artanis-streaming-notes.md).

Options: `--backend real|mock`, `--model` (default
`models/qwen2.5-7b-instruct-1m-q4_k_m.gguf`), `--model-id` (default
`qwen2.5-7b`), `--image-model`, `--host`, `--port`, `--queue`,
`--max-sessions` (default 1 for the real backend, 4 for the mock), `--token-budget`,
`--context-length` (default 16384 for the real backend, 8192 for the mock),
`--step-delay`, `--prefill-delay`, `--runtime-dir`, `--request-log`,
`--replay-store`. The app sets `server.websocket = true` and
`server.runners = queue + 2` (each running or waiting stream holds a runner).
`--replay-store DIR` is off by default and stores request bodies locally; see
[`reproducible-api.md`](reproducible-api.md) for what it keeps and the privacy
consequences.

## 7. Running

Environment used (Ubuntu 24.04): Guile 3.0.9 from apt and guile-json 4.7.3
built from its upstream source (`https://github.com/aconchillo/guile-json`,
tag `4.7.3`, `autoreconf -i && ./configure --prefix=… && make install`, with
that prefix on `GUILE_LOAD_PATH`/`GUILE_LOAD_COMPILED_PATH`); guile-curl and
guile-redis from git; libargon2; GNU Artanis from
`https://gitlab.com/hardenedlinux/artanis` (commit `9100e1d`,
`./autogen.sh && ./configure && make && make install`).

```sh
# the service (default --backend real; needs libsonicboom.so and a GGUF)
guile -L bindings/guile/scheme bindings/guile/scheme/sonicboom/serve/app.scm \
      --port 8000 --model models/qwen2.5-7b-instruct-1m-q4_k_m.gguf \
      --model-id qwen2.5-7b --context-length 16384 --max-sessions 1

# the same service with no model file or shared library
guile -L bindings/guile/scheme bindings/guile/scheme/sonicboom/serve/app.scm --backend mock --port 8000

# tests, from the repository root
guile -L bindings/guile/scheme bindings/guile/tests/serve/test-serve-units.scm
guile -L bindings/guile/scheme bindings/guile/tests/serve/test-serve-http.scm
python3 bindings/guile/tests/serve/test-openai-sdk.py      # pip install openai
```

Results: units 85/85, HTTP 169/169, images 36/36; openai SDK (3.29.0) 24/24
when last run.

## 8. Limits of the prototype

- `n` must be 1; no `logprobs`.
- `response_format` supports `json_schema` and `json_object` (and `text`); the
  schema subset is the compiler's — `pattern`, `format`, `not` and the other
  keywords listed in [`capi.md`](capi.md) are 400s, not silently ignored.
- One sequence per session; fork copies (no paged KV sharing).
- The queue does not notice a client that leaves while waiting; it is
  cancelled at its first token.
- No authentication; the `Authorization` header is ignored.
- `/v1/images/generations` is queued behind other device work and cancellable,
  but it holds the image model for the whole generation; a non-streamed
  request keeps the device until it finishes unless it is cancelled by id.
  `response_format` `url` is not implemented (no storage).
- A model switch drops the session pool it leaves; the prefix cache of a
  model is rebuilt after it comes back.
