# OpenAI-compatible local inference service (Guile + GNU Artanis)

Status: **prototype**, with a deterministic mock backend. The application layer
of the production inference route (Route 2 / Path 2 of
`design/execution-paths.md`): an HTTP service speaking the OpenAI
chat-completions API, written in Guile on GNU Artanis. The backend interface is
the one a libsonicboom backend will implement; its C ABI is drafted in
`design/route2-capi-draft.md`.

## 1. Layout

```text
bindings/guile/scheme/sonicboom/serve/
├── backend.scm        the backend protocol (record of procedures), the sampler
├── mock-backend.scm   a deterministic mock implementing the protocol
├── chat-template.scm  hand-written chat templates: gemma, qwen; tool-call text
├── sessions.scm       session pool: prefix matching, fork, LRU, token budget
├── engine.scm         FIFO request queue (one device) + the generation loop
├── openai.scm         request validation, response / chunk / error objects
├── sse.scm            a minimal SSE writer on Artanis (no Artanis change)
└── app.scm            the single-file Artanis app: the HTTP layer only

bindings/guile/tests/serve/
├── test-serve-units.scm   below HTTP (templates, mock, sampler, pool, engine)
├── test-serve-http.scm    end to end with Guile's HTTP client, raw SSE, WebSocket
└── test-openai-sdk.py     the official openai Python SDK against the app
```

Dependencies point down: `app.scm` → `openai`, `engine`, `sse` → `sessions` →
`backend` ← `mock-backend`. Only `app.scm` and `sse.scm` import Artanis; the
other modules need Guile and guile-json only, and are what a libsonicboom
backend reuses as is.

## 2. Backend protocol

`(sonicboom serve backend)`: `make-backend` takes the procedures

| procedure | contract |
|---|---|
| `load-model config` | → model handle |
| `model-info model` | alist: `id`, `family`, `context-length`, `eos-tokens` |
| `tokenize model text` / `detokenize model tokens` | special tokens are single tokens |
| `render-chat model messages tools tool-choice` | → prompt text (the mock delegates to `chat-template.scm`) |
| `create-session model` / `free-session session` | a session is a KV cache holding one token sequence |
| `fork-session session n` (optional) | a new session holding the first n tokens, without recomputing them |
| `prefill session tokens` | the session holds exactly `tokens` afterwards; the longest common prefix with what it held is kept; → number of reused tokens |
| `step session sampler constraint` | sample the next token, append it, return it |
| `stats model` | counters (optional) |

A **sampler** (`make-sampler #:temperature #:top-p #:top-k #:seed`) carries one
request's parameters and random state; temperature 0 is greedy. A
**constraint** is a token filter `(candidates) → candidates` applied before
sampling, the hook for constrained decoding (tool-call or JSON grammars). The
mock applies it; nothing in the service installs one yet.
`sample-from-logits` is a reference sampler for backends that produce logits
on the host (the mock uses it; a device backend samples on the device).

## 3. The mock backend

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

## 4. Chat templates

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

## 5. Sessions and the prefix cache

`(sonicboom serve sessions)`, one pool per model, `--max-sessions` sessions
and a `--token-budget` of held tokens. The pool mirrors each session's tokens.
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

## 6. Queue and generation

`(sonicboom serve engine)`:

- **Queue.** One device: one request generates at a time, in arrival order.
  `engine-admit!` (in the HTTP handler, before any response is written) takes
  a place or raises `queue-full` → HTTP 429 `rate_limit_error` / `queue_full`
  when `--queue` requests already wait. `engine-run` waits for the turn and
  always leaves the queue; `engine-drop!` releases a place never run.
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

## 7. HTTP layer

`app.scm` is a single-file Artanis app in the style of the Artanis examples:

| Artanis example | used for |
|---|---|
| `examples/test.scm` | single-file app shape: `(init-server)`, `get` / `post` routes, `:id` route parameters, `#:mime 'json` + `(:mime rc obj)` for JSON, `response-emit` for other statuses, `(run …)` |
| `examples/ENTRY.websocket` | `#:websocket 'raw` routes returning an `on-message` dispatcher, `ws-send`, `ws-message-text`, `#:timeout 0`, `call-with-runner` for blocking work |
| `examples/protocols/chat.scm` | JSON in text frames (the service encodes chunk JSON itself on a `raw` route instead of installing an app protocol, which needs an `art` project directory) |

Endpoints:

| endpoint | |
|---|---|
| `GET /v1/models`, `GET /v1/models/:id` | the two mock models (`sonicboom-mock-gemma`, `sonicboom-mock-qwen`) |
| `POST /v1/chat/completions` | `stream: false` → `chat.completion`; `stream: true` → SSE of `chat.completion.chunk`, then `data: [DONE]` |
| `GET /v1/ws/chat` | WebSocket: one request (the same JSON body) per text message; the reply is the chunk objects as text messages, then `[DONE]`; a bad request gets an `error` object |
| `GET /sonicboom/stats` | engine, pool and backend counters (not OpenAI) |
| `GET /health` | `ok` |

Request fields: `model`, `messages` (string or text-part content; roles
`system`, `developer`, `user`, `assistant` with `tool_calls`, `tool`), `stream`,
`stream_options.include_usage`, `max_tokens` / `max_completion_tokens`,
`stop` (string or up to 4), `temperature`, `top_p`, `top_k` (extension),
`seed`, `n` (1 only), `tools`, `tool_choice`. Unknown fields are ignored.

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
workarounds: `design/artanis-streaming-notes.md`.

Options: `--host`, `--port`, `--queue`, `--max-sessions`, `--token-budget`,
`--context-length`, `--step-delay`, `--prefill-delay`, `--runtime-dir`. The
app sets `server.websocket = true` and `server.runners = queue + 2` (each
running or waiting stream holds a runner).

## 8. Running

Environment used (Ubuntu 24.04): Guile 3.0.9 and guile-json from apt;
guile-curl and guile-redis from git; libargon2; GNU Artanis from
`https://gitlab.com/hardenedlinux/artanis` (commit `9100e1d`,
`./autogen.sh && ./configure && make && make install`).

```sh
# the service
guile -L bindings/guile/scheme bindings/guile/scheme/sonicboom/serve/app.scm --port 8000

# tests, from the repository root
guile -L bindings/guile/scheme bindings/guile/tests/serve/test-serve-units.scm
guile -L bindings/guile/scheme bindings/guile/tests/serve/test-serve-http.scm
python3 bindings/guile/tests/serve/test-openai-sdk.py      # pip install openai
```

Results: units 47/47, HTTP 86/86, openai SDK (3.29.0) 24/24.

## 9. Limits of the prototype

- `n` must be 1; no `logprobs`; `response_format` is ignored (a JSON-mode
  constraint is the intended implementation).
- One sequence per session; fork copies (no paged KV sharing).
- The queue does not notice a client that leaves while waiting; it is
  cancelled at its first token.
- No authentication; the `Authorization` header is ignored.
