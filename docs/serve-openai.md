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
`gemma`, `qwen2` → `qwen`), stop tokens are the union of the model's EOS ids
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
- **qwen**: ChatML `<|im_start|>role\n…<|im_end|>\n`.

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
| `POST /v1/images/generations` | MOSS-Image: `{"created", "data": [{"b64_json": …}]}`, each entry a PNG |
| `GET /v1/ws/chat` | WebSocket: one request (the same JSON body) per text message; the reply is the chunk objects as text messages, then `[DONE]`; a bad request gets an `error` object |
| `GET /sonicboom/stats` | engine, pool and backend counters (not OpenAI) |
| `GET /health` | `ok` |

Request fields: `model`, `messages` (string or text-part content; roles
`system`, `developer`, `user`, `assistant` with `tool_calls`, `tool`), `stream`,
`stream_options.include_usage`, `max_tokens` / `max_completion_tokens`,
`stop` (string or up to 4), `temperature`, `top_p`, `top_k` (extension),
`min_p` (extension), `seed`, `repeat_penalty` (extension), `presence_penalty`,
`frequency_penalty`, `n` (1 only), `tools`, `tool_choice`,
`response_format`. Unknown fields are ignored; a sampling value out of range
is a 400 naming the parameter.

Omitted sampling defaults differ from the OpenAI service: **an omitted
`temperature` is 0 (greedy, not 1.0)**, so a request that says nothing samples
nothing and repeats exactly. `top_p` defaults to 1, `top_k` and `min_p` to 0
(off) and `repeat_penalty` to 1 (off); `presence_penalty` and
`frequency_penalty` default to 0 and are validated to [-2, 2]. Accepted
ranges: `temperature` [0, 2], `top_p` and `min_p` [0, 1], `top_k` a
non-negative integer, `repeat_penalty` ≥ 0 (0 is read as its 1 default).

A seed is only meaningful when `temperature` > 0. A sampling request that
omits `seed` gets a fresh one, and the reply carries it as `sonicboom_seed`
(a non-OpenAI field on the completion object, and on the first chunk of a
streamed reply); replaying the request with that `seed` reproduces the reply,
including across a restart of the server. A greedy request reports
`sonicboom_seed` 0 and ignores it.

With the real backend, `temperature`, `top_p`, `top_k`, `min_p`, `seed`,
`repeat_penalty` and the two penalties are passed to `boom_llm_decode` on
every step (in the C ABI's order: penalties, then top-k / min-p / top-p, then
the draw). A request with `temperature` > 0 therefore samples; the penalties
and `repeat_penalty` are part of the C ABI's sampled path, so they take effect
only when `temperature` > 0 — the greedy default is a plain argmax and ignores
them. The mock backend honors `temperature`, `top_p`, `top_k` and `min_p`; it
has no penalty or repetition model, so `repeat_penalty` and the two penalties
are accepted and ignored there.

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

The response is `{"created": <unix time>, "data": [{"b64_json": <PNG>}, …]}`.
Each PNG is written by the C ABI's own writer (8-bit RGB, stored deflate
blocks) and base64-encoded in Scheme — Guile 3.0.9 has no `(ice-9 base64)`.
The same request twice returns the same pixels, because both the noise and
the sampler are seeded. A model that cannot be loaded, and a generation
failure, are 500s (`server_error`); a cancelled generation would be
`BOOM_ERR_CANCELLED`, but the server never cancels one.

Streaming: the first chunk carries `delta.role`, then content deltas (one per
token here), the tool-call deltas (a header with `index`, `id`, `type`, `name`
and empty `arguments`, then the arguments in pieces), a chunk with
`finish_reason`, the usage chunk (empty `choices`) when
`stream_options.include_usage` is set, then `[DONE]`. A tool call is emitted
once its closing tag is generated (its arguments are not parsed while still
open).

Errors use the OpenAI shape `{"error": {"message", "type", "param", "code"}}`:
400 `invalid_request_error`, 404 `model_not_found`, 429 `queue_full`, 500
`server_error`. In a stream that already started, an error is sent as an
`error` event followed by `[DONE]`.

Cancellation: a streamed reply (SSE or WebSocket) stops at the next token
after the client goes away. A non-streamed request is not watched while it
generates.

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
`--step-delay`, `--prefill-delay`, `--runtime-dir`. The
app sets `server.websocket = true` and `server.runners = queue + 2` (each
running or waiting stream holds a runner).

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

Results: units 80/80, HTTP 133/133, images 16/16; openai SDK (3.29.0) 24/24
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
- `/v1/images/generations` is queued but not cancellable: it holds the image
  model for the whole generation, and a client that goes away waits for it to
  finish. `response_format` `url` is not implemented (no storage).
- A model switch drops the session pool it leaves; the prefix cache of a
  model is rebuilt after it comes back.
