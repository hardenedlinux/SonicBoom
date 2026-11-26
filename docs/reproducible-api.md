# Reproducible generation: seed, fingerprint, replay

SonicBoom's serving layer can make a sampled reply repeatable: the same request
body and the same seed, on the same machine and build, produce the same bytes.
This document defines the fields that say so, the scope the guarantee covers,
and the optional local replay store.

The implementation is the Guile serving layer
(`bindings/guile/scheme/sonicboom/serve/`); the guarantee is about the engine
below it, so it holds for `--backend real` and, for protocol tests,
`--backend mock`.

## 1. The seed

Sampling is keyed by `(seed, absolute position, history)`. A request with
`temperature` > 0 samples; `temperature` 0 is greedy (argmax).

- A sampling request that names `seed` uses it. One that omits it gets a fresh
  one, and the reply reports it, so a client that keeps the reply can ask for
  the same sample again.
- A greedy request ignores `seed`. Its reply carries `"seed": null` (the field
  is present, with JSON null), and no `X-SonicBoom-Seed` header.
- A sampled reply carries the seed as the top-level `seed` field (a JSON
  number, 0 .. 2^32 − 1) and in the `X-SonicBoom-Seed` response header. Header
  names are case-insensitive; the HTTP server canonicalises this one as
  `X-Sonicboom-Seed` on the wire.

The same seed under a different request, or a different seed under the same
request, is not promised to differ: with a peaky distribution the draw can be
the same. Reproducibility is the guarantee; variety is a property of the
distribution.

## 2. The `sonicboom` object

Every chat completion — streamed or not, greedy or sampled — carries a
top-level `sonicboom` object:

```json
"sonicboom": {
  "fingerprint": {
    "engine": "v0.0.6-187-gd44a66e",
    "commit": "d44a66e81216",
    "numerics": "standard",
    "gpu": "NVIDIA GeForce RTX 3050",
    "driver": "595.84",
    "tune": "e234db4c87d28a43",
    "model": "sha256:8f3c1d0a9b7e2f41",
    "id": "03d4602d080e8a6e"
  },
  "replay_id": "rp-4bd0f1a2c3d4e5f6",
  "guarantee": "bit-identical"
}
```

- `replay_id` names this reply. It is random per reply and has no meaning
  unless a replay store is configured (§4).
- `guarantee` is `"bit-identical"` when the reply was produced inside the
  fingerprint's scope. A replay (§4) on a server whose fingerprint no longer
  matches the stored one reports `"fingerprint-mismatch"` instead; the reply is
  still returned, but it is not promised to match.

### 2.1 Fingerprint: the scope of the guarantee

A fingerprint names exactly the inputs outside the request that can change the
bytes. The record is:

| field | meaning |
|---|---|
| `engine` | libsonicboom's `git_tag` (or `unknown`) |
| `commit` | the library build's `git_commit`, first 12 hex digits |
| `numerics` | the resolved numerics profile, `strict` or `standard` |
| `gpu` | `nvidia-smi`'s GPU name (`none` when unreadable) |
| `driver` | `nvidia-smi`'s driver version (`none` when unreadable) |
| `tune` | FNV-1a 64 over the names and contents of every file under the tune root (see below), 16 hex; `none` when the root does not exist |
| `model` | `sha256:` + the first 16 hex of the GGUF's SHA-256; for a model with no file (the mock backend) `none:<model-id>` |
| `id` | FNV-1a 64 over the canonical text of the other fields, 16 hex |

Two servers give the same `id` exactly when they agree on every field. The
guarantee is: **the same request body and seed, on a server with the same
fingerprint `id`, produce the same reply bytes** — across requests, across a
restart, and across two server processes. It does not cross a change in engine
build, numerics profile, GPU or driver, tune tables, or model file.

The tune root is `$SONICBOOM_TUNE_DIR`, else `$XDG_CACHE_HOME/sonicboom/tune`,
else `$HOME/.cache/sonicboom/tune`. The model hash is `sha256sum` of the whole
file, cached in the process and in `$TMPDIR` (keys: basename, size, mtime) so a
second server does not re-read a multi-gigabyte GGUF. The server part of the
record is computed once per process.

The fingerprint is a serving-layer report: it is not part of the C ABI, and no
Layer 2 header exposes it.

What the request itself contributes is ordinary: the messages, the rendered
chat template, the sampling parameters, `max_tokens`, stop strings and
constraints. The same fingerprint with a different request or seed is a
different sample.

## 3. Streaming (SSE)

With `"stream": true`:

- the first `chat.completion.chunk` carries `seed` and the `sonicboom` object;
- interior content chunks carry neither;
- the final chunk (the one with `finish_reason`) repeats both;
- when `stream_options.include_usage` is set, the usage chunk carries them too;
- then `data: [DONE]`.

So a client can take the seed from the first chunk and ignore the rest, and a
relay can attach the object to the finished reply from the last.

## 4. The replay store

`--replay-store DIR` (off by default) makes replies regenerable. While it is
on, the server writes one JSON file per completed chat reply,
`DIR/<replay_id>.json`:

```json
{"replay_id": "rp-…", "seed": 2871560934, "model": "qwen2.5-7b",
 "time": 1795418553, "fingerprint": "03d4602d080e8a6e",
 "request": { … the request body, with the deciding seed added … }}
```

The stored body is the client's request as decoded, with the `seed` field set
to the seed that actually drew the reply (`seed` is left absent for greedy).
The server keeps the last `replay_id` it wrote per reply; nothing else is
added, no completion text is stored, and the file lives only under `DIR`.

### 4.1 Regenerate

```sh
curl -s -X POST http://127.0.0.1:8000/sonicboom/replay/rp-4bd0f1a2c3d4e5f6
```

The server loads the record, re-admits the stored body through the normal
queue, runs the engine, and returns a normal (non-streamed) completion with
the stored `replay_id`, the stored seed and the current `sonicboom` object.
`guarantee` is `"bit-identical"` when the current fingerprint `id` equals the
stored one, `"fingerprint-mismatch"` otherwise. An unknown id, a bad id, or a
server without a replay store answers 404 `replay_not_found`. The stored model
must still be served under the same model id.

### 4.2 History

`GET /sonicboom/history` returns a minimal HTML page: the most recent 100
stored replies, newest first, with time, model, seed (or "greedy") and two
actions per row — **replay** (POSTs the replay endpoint and prints the JSON
below) and **copy seed**. It reads only the files in the replay directory, so
it is empty without `--replay-store`.

## 5. Privacy

The replay store exists to keep prompts and replies locally; it is opt-in
because it does. Specifically:

- The **request log** (`--request-log`) never contains content: it records
  `seed`, `fingerprint` (the id) and `replay_id` in addition to its counters,
  but not the messages or the completion.
- The **replay store** does contain the request body — including the prompt and
  the conversation — in plain JSON under `DIR`. It never contains the
  completion text (a replay re-derives it). Treat `DIR` as sensitive: it is
  not encrypted, not rotated, and has no retention limit beyond what you
  delete. Without `--replay-store`, nothing is written and `replay_id` is
  still reported but names nothing.
- The **fingerprint** contains no prompt: engine, commit, numerics, GPU,
  driver, tune digest and a model-file hash are all properties of the
  deployment.

## 6. What is tested

- `bindings/guile/tests/serve/test-serve-http.scm`: the seed appears in the
  reply, in the header, in the first and final SSE chunks; greedy replies carry
  `seed: null` and no header; the same request and seed are byte-identical
  twice and in a second server process; replay returns the same content, seed
  and id; the history page lists the reply; a bad id is 404.
- `tests/core/test_guarantees.cpp` holds the engine-level invariants the
  serving layer relies on, with real models: determinism across fresh
  processes, prefix reuse identity, KV-capacity independence and
  `SONICBOOM_SPEC=off`/`on` identity.
- `tools/repro-conformance` (see `third-party-engines/`) checks the same
  properties through any OpenAI-compatible HTTP server, including
  llama-server, and records what other engines do.
