# SonicBoom — Guile binding

The official Guile binding for SonicBoom, the native tensor runtime. It is the
first-class language binding and sits at Layer 3:

```
Guile → SonicBoom C API (capi/) → SonicBoom Core → native-torch → CPU / CUDA
```

Every numeric operation is dispatched across the C API into
`libsonicboom.so`; the binding never reaches into `core/layer1/` or
`third_party/native-torch/`, and no ATen/c10 type crosses the C ABI. What
Scheme does is build graphs, hold handles, and drive loops. Per-element
arithmetic is not written in Scheme.

Guile 3.0.9 or later.

## Running

```sh
# from the repository root, after cmake --build build
LD_LIBRARY_PATH=build/core guile --no-auto-compile \
    -L bindings/guile/scheme bindings/guile/tests/test-nn.scm
```

`build/core` holds `libsonicboom.so`. `GUILE_AUTO_COMPILE=0` keeps the test
output free of `.go` cache notes. The CTest target is `ctest -R guile`.

## Modules

| module | what it is |
|---|---|
| `(sonicboom tensor)` | Guile-side tensors: shape/dtype/data, constructors, element access, parameters |
| `(sonicboom nn)` | layers and models, and the S-Expr text they emit |
| `(sonicboom model)` | a trainable model: layer structure + one native tensor per parameter |
| `(sonicboom train)` | the native autograd tape: forward ops, backward, SGD |
| `(sonicboom executable)` | compile an S-Expr graph once, run it many times |
| `(sonicboom ffi)` | the C ABI entry points themselves; the other modules build on it |
| `(sonicboom inference)` | a GGUF LLM: load, session, prefill, generate, stream |
| `(sonicboom trace symbolic)` | the tracer's operator vocabulary |
| `(sonicboom trace tracer)` | `trace-function`: trace a procedure, replay it |
| `(sonicboom trace macro)` | `define-traced`: the macro that records a procedure into a graph |
| `(sonicboom trace graph)` | the traced graph as data, and its S-Expr form |
| `(sonicboom trace lower)` | lower a traced graph onto the runtime's op set and run it |
| `(sonicboom trace interp)` | the pure-Guile reference interpreter for S-Expr documents |
| `(sonicboom trace layers)` | traced layer builders (linear, rms-norm, softmax, ...) |
| `(sonicboom trace autodiff)` | `grad`, `value-and-grad` for the tracer |
| `(sonicboom trace optim)` | `sgd-update`, `adam-init`, `adam-update` for the tracer |
| `(sonicboom serve ...)` | an OpenAI-compatible HTTP server on GNU Artanis (see below) |

## (sonicboom tensor)

Guile-side tensors. These are the layer parameters and the data the tracer
works on; the C ABI has its own native tensor handles (see `(sonicboom train)`
and `(sonicboom model)`).

- `make-tensor data shape #:dtype #:requires-grad` — a tensor from a flat list
- `%make-tensor`, `bytevector->tensor`, `tensor->bytevector` — the raw constructors and the byte view
- `tensor-dtype`, `tensor-shape`, `tensor-data`, `tensor-numel` — accessors
- `tensor-ref t i`, `tensor-set! t i v` — element access by flat index
- `tensor->list`, `list->tensor data shape`, `tensor-copy` — conversion and copy
- `zeros shape`, `ones shape`, `randn shape`, `rand` — constructors
- `rand` takes no arguments and is not a tensor constructor; `randn` is the normal initializer
- `parameter data #:shape #:dtype`, `parameter?` — a tensor that carries a gradient
- `tensor-requires-grad?`, `tensor-grad`, `set-tensor-grad!`, `parameter-grad`, `parameter-grad-set!` — gradient access
- `dtype-size`, `valid-dtype?`, `valid-shape?` — introspection

```scheme
(use-modules (sonicboom tensor))

(define x (list->tensor '(1 2 3 4 5 6) '(2 3)))
(tensor-shape x)      ; => (2 3)
(tensor-ref x 4)      ; => 5.0
(tensor-set! x 0 9.0)
(tensor->list x)      ; => (9.0 2.0 3.0 4.0 5.0 6.0)
```

## (sonicboom nn)

Layers compose into a model; a model emits the S-Expr text the runtime
executes.

- `linear in out #:bias #:init #:name` — an affine layer; the weight is laid out `[out, in]` and `#:init` is `'randn` or `'zeros`
- `relu #:name` — a rectifier
- `sequential layer ...` — a model from layers, applied in order
- `layer?`, `layer-kind`, `layer-name`, `layer-parameters`, `layer-parameter-ref` — layer introspection
- `model?`, `model-layers`, `model-parameters`, `model-parameter-ref` — model introspection; a parameter name is `"<layer>.<key>"`, e.g. `"l1.weight"`
- `model-graph-text model input-dims` — the model's graph as S-Expr v0.1 text, ready for `compile-s-expr`
- `model-forward model x` — run the emitted text through the runtime; returns a Guile tensor
- `model-trace model input-dims` — trace the same layers to an S-Expr document datum
- `model-trace-forward model x` — run that traced graph through the runtime

```scheme
(use-modules (sonicboom nn))

(define m (sequential (linear 4 8 #:name 'l1)
                      (relu #:name 'act)
                      (linear 8 3 #:init 'zeros #:name 'l2)))
(define x (list->tensor '(0.5 -1.25 2.0 0.75) '(1 4)))

(display (model-graph-text m '(1 4)))   ; the S-Expr text
(tensor->list (model-forward m x))
(tensor->list (model-trace-forward m x))  ; the same numbers, traced and lowered
```

`model-forward` and `model-trace-forward` are two routes over the same
parameters: the first runs the text `apply` emits, the second traces the
layers and lowers the result. Both are checked to agree in
`tests/test-nn-trace.scm`.

## (sonicboom model)

A trainable model: the `(sonicboom nn)` structure plus one native tensor per
parameter, which is the authoritative store. Training mutates those tensors
in place through the native autograd; export writes them out.

- `make-model input-dims structure #:init-values` — native parameters from the structure; `#:init-values` is an alist of `("layer.key" . flat-list)` for a deterministic start
- `nn-model?`, `nn-model-structure`, `nn-model-params`, `nn-model-input-dims` — accessors
- `model-param model name` — the native tensor for `"layer.key"`
- `model-forward-native model x tape` — a forward pass on `tape`; `x` is left untouched
- `model-train! model x tgt #:epochs #:lr` — plain SGD over the whole dataset; returns initial and final loss
- `model-export! model dir` — write `model.sx` + `weights.bin`, loadable in C++ with no Guile

```scheme
(use-modules (sonicboom nn) (sonicboom model) (sonicboom train))

(define x   (tensor-from-list '(4 2) '(0 0 0 1 1 0 1 1)))
(define tgt (tensor-from-list '(4 1) '(0 1 1 0)))
(define m (make-model '(4 2)
                      (sequential (linear 2 4 #:name 'fc1)
                                  (relu #:name 'act1)
                                  (linear 4 1 #:name 'fc2))))

(call-with-values (lambda () (model-train! m x tgt #:epochs 3000 #:lr 0.5))
  (lambda (first last) (format #t "loss ~a -> ~a~%" first last)))
(model-export! m "/tmp/xor-model")
```

## (sonicboom train)

The native autograd surface. Tensors cross the C ABI as little-endian float32
bytevectors or as opaque native handles.

- `tensor-from-list shape list`, `tensor->list t` — native tensor handle from/to a flat list
- `list->f32-bv`, `f32-bv->list` — the same conversion at the bytevector level
- `tensor-free!` — release a native tensor handle
- `make-tape`, `tape-free!`, `tape-zero-grad!` — the autograd tape
- `autograd-linear x w b tape` — `x @ w^T + b`, recording onto the tape
- `autograd-relu x tape`, `autograd-mse pred target tape` — the other recorded ops
- `autograd-backward loss tape` — backpropagate from the loss
- `autograd-grad tape param` — the gradient accumulated for a parameter
- `autograd-sgd-step! tape param lr` — one SGD update of a parameter
- `train-xor #:epochs #:lr` — the self-contained XOR demo; returns the before/after loss

Note `tensor->list` here is the native-handle reader; `(sonicboom tensor)`
exports the same name for Guile tensors.

```scheme
(use-modules (sonicboom train))

(call-with-values (lambda () (train-xor #:epochs 3000 #:lr 0.5))
  (lambda (first last) (format #t "~a -> ~a~%" first last)))
```

## (sonicboom executable)

An S-Expr graph parsed and compiled once, then run many times.

- `compile-s-expr text` — `sb_parse` + `sb_compile` once; returns an executable
- `executable-run exe input-bv` — one run; returns `(values output-bv #f)`, or `(values #f message)` on failure
- `executable-graph-text`, `executable-input-nbytes`, `executable-output-nbytes` — what it was compiled from
- `executable?`, `executable-freed?`, `free-executable!` — lifetime; freeing is idempotent
- `run-s-expr/cached text input` — compile-once wrapper over `(sonicboom ffi)`'s `run-s-expr`
- `cached-executable text` — the cache entry for a graph text
- `executable-cache-enabled` — a parameter (default `#t`) turning the cache off
- `clear-executable-cache!` — free and drop every cached entry
- `reap-executables!` — release the handles the guardian has collected

```scheme
(use-modules (sonicboom executable) (sonicboom nn))

(define text (model-graph-text m '(1 4)))
(define exe  (compile-s-expr text))            ; parsed and compiled once
(call-with-values (lambda () (executable-run exe input-bytes))
  (lambda (out msg) (if out (use out) (display msg))))
(free-executable! exe)
```

Lifetime: an executable owns a native handle. Guile 3.0.9 has no
`set-finalizer!`, so executables are registered with a guardian and released by
`reap-executables!` after the GC collects them.

## (sonicboom ffi)

The C ABI entry points and their plumbing. Modules above use this; a caller
rarely needs it directly.

- `proc name` — the C function for a symbol in `libsonicboom.so`
- `run-s-expr text input-bv` — parse, execute and retrieve in one call
- `libsonicboom-available?` — whether the library loaded at all
- `sb-status-ok?`, `sb-status->message`, `SB_ERR_BINDING` — status handling
- `make-slot`, `slot-ptr`, `slot-ref`, `capture-error` — out-parameter helpers

## (sonicboom trace ...) — the tracer

The tracer records a *procedure* into a graph. Use `(sonicboom trace macro)`
for `define-traced`, which rewrites the procedure's control flow so branches
become graph `where` nodes; `(sonicboom trace symbolic)` is the operator
vocabulary the body is written in (or the modules provide through
`trace-opaque`).

Documented endpoints:

- `trace-function proc` — wrap a procedure so calling it traces once and replays (segments, branches, guards) afterwards
- `traced-function?`, `trace-stats`, `trace-segments`, `trace-tree`, `trace-warnings`, `trace-strict-warnings` — what the wrapper recorded
- `trace->sexpr proc arg ...` — trace once and return the S-Expr v0.1 text
- `define-traced (name arg ...) body ...` — define a traced procedure; see `tests/trace/test-trace-forms.scm`
- `interpret-sexpr doc inputs`, `interpret-sexpr-text text inputs` — the pure-Guile reference interpreter; `inputs` is an alist `(name . ctensor)`
- `sgd-update`, `adam-init`, `adam-update`, `grad`, `value-and-grad` — optimizer and autodiff steps usable inside a traced body
- `(sonicboom trace layers)`: `linear`, `softmax`, `log-softmax`, `layer-norm`, `rms-norm`, `cross-entropy`, `embedding`, `causal-mask`

## (sonicboom trace lower) — traced graphs on the runtime

Lowers a traced forward graph onto the op set the runtime's MLIR path accepts
(`relu add reshape conv max_pool reduce_mean gemm`, `core/src/sx/lowering.cpp`)
and executes it through `libsonicboom.so`.

- `trace-graph proc arg ...` — trace `proc` and return the document datum
- `traced->s-expr graph` — the lowered document *text*; raises an error naming the op for any node with no lowering
- `run-traced graph inputs` — lower and run; returns the output ctensor

```scheme
(use-modules (sonicboom trace tensor) (sonicboom trace symbolic)
             (sonicboom trace tracer) (sonicboom trace lower))

(define x (make-ctensor 'float32 '(2 2) '(1.0 -2.0 3.0 -4.0)))
(define graph (trace-graph (lambda (i) (relu i)) x))

;; the input's name comes from the document's inputs section
(define (input-name doc) (cadr (car (cdr (assq 'inputs (cdr (caddr doc)))))))

(display (traced->s-expr graph))               ; the S-Expr text, '(relu) nodes
(ctensor->list (run-traced graph (list (cons (input-name graph) x))))
```

A `linear` (matmul + rank-1 bias add) is fused into one `gemm` node when the
matmul's output has exactly one consumer; nothing else is fused.

## (sonicboom inference)

A GGUF language model through the minimal LLM C ABI: the built-in tokenizer,
a session, prefill with prefix reuse, and greedy decode. Handles are foreign
pointers and are freed explicitly.

- `load-model path`, `free-model! model` — load a GGUF
- `make-session model n-ctx`, `free-session! session` — a session holds a token history and a context window
- `prefill! session tokens` — set the session's prompt; returns how many leading tokens were reused from the previous prompt (at most `length - 1`: the last token is always re-processed, its activation being what produces the next-token logits)
- `tokenize model text #:special?` — UTF-8 text to token ids (a `<bos>` is prepended); `#:special? #t` also matches the vocabulary's control pieces (`<|turn>`, `<turn|>`) verbatim, one token each
- `tokenize-segments model segments` — a rendered chat prompt as `(kind . text)` segments: `'special` segments are tokenized with `#:special? #t`, `'text` segments never are; `<bos>` is added once, by the first segment
- `detokenize model tokens` — token ids back to text
- `generate model session prompt max-tokens` — tokenize, prefill, decode greedily, detokenize
- `generate-tokens session prompt max-tokens #:prefill?` — the same, returning token ids; `#:prefill? #f` continues from what the session holds
- `generate-stream model session prompt max-tokens proc #:prefill?` — call `(proc token chunk)` once per produced token; chunks never split a UTF-8 character and concatenate to exactly what `generate` returns; returns the token ids

```scheme
(use-modules (sonicboom inference))

(define model   (load-model "models/gemma-4-E4B-it-Q3_K_M.gguf"))
(define session (make-session model 512))

(display (generate model session "Hello" 16))

(generate-stream model session (tokenize model "Hello") 16
                 (lambda (token chunk) (display chunk) (force-output)))
(free-session! session)
(free-model! model)
```

## The HTTP server (`(sonicboom serve ...)`)

A single-file GNU Artanis app over the same inference binding, with the
OpenAI chat-completions shape:

```sh
tools/run-api-server.scm            # or: PORT=8000 tools/run-api-server.scm
tools/test-api-server.sh            # curl smoke test against the real model
```

`/health`, `/v1/models`, `/v1/models/:id`, `/v1/chat/completions` (JSON and
SSE), `/v1/ws/chat` (WebSocket) and `/sonicboom/stats`.

## Limitations

These are current facts, not policy; several are recorded as NEEDS-CORE work.

- **The C ABI binds exactly one input and returns exactly one output**
  (`capi.h:136`). A graph with several inputs — a training step, for
  instance — cannot be expressed as one executable.
- **The traced route lowers forward graphs only**, and only a subset of them:
  `matmul` (+ rank-1 `add`), `add`, `relu`, `reshape` and `reduce_mean` have a
  lowering. `sub`, `mul`, `div`, `exp`, `log`, `tanh`, `gelu`, `permute`,
  `bmm`, comparisons (`greater`/`less`/`equal`), `where`, gather/scatter,
  `concat`, `slice`, `expand` and `reduce_sum` raise. In particular a training
  step cannot be lowered: `where`/`greater` are the relu's backward mask.
- **A rank-0 result cannot be expressed** — a reduction over every axis
  lowers to `reduce_mean` + `reshape`, and the reshape of a 0-element target
  shape is rejected.
- **A graph output is always a copy the caller owns.** The C ABI frees the
  pointer the compiled entry returns (`core/src/sx/exec.cpp:389`), so an output
  that would bufferize as a *view* — a `reshape` of the input, of a parameter,
  or of a computed tensor — is copied into a fresh tensor before it is
  returned. Non-view outputs are returned as they are, with no extra copy.
- **The tracer names closed-over constants itself** (`c0`, `c2`, ...), so a
  traced graph does not carry the model's `"l1.weight"` names. Parameters are
  matched by shape and value, not by name.
- **Inference is greedy only** — temperature 0, top-p 1, top-k 0. A sampling
  request is rejected (HTTP 400) rather than silently ignored.
- **Detokenization is not additive**: the tokenizer's space cleanup runs over
  the whole text and looks across token boundaries, so streaming must
  re-detokenize the prefix (which `generate-stream` does; that is O(tokens²)
  per call).
- **A prompt already resident in a session's KV cache is reused down to its last
  token.** `sb_llm_prefill` reuses at most `n - 1` tokens and re-prefills the
  final one, so the prefill suffix is never empty: an identical repeat of a
  prompt succeeds and reports `n_reused = n - 1`.
- **Control-token parsing is opt-in, and the server turns it on only for the
  template's own markers.** Gemma 4's turn markers are `<|turn>` and `<turn|>`
  (not the `<start_of_turn>`/`<end_of_turn>` spelling, which is not in this
  vocabulary); by default `tokenize` treats them as ordinary text, so a message
  that spells one out stays text and cannot forge a turn boundary. The chat
  template renders a *segment* list in which its own markers are `'special` and
  every message is `'text`; `tokenize-segments` parses the first and never the
  second, which is what makes the injection impossible while the markers become
  the vocabulary's single control tokens (`<|turn>` = 105, `<turn|>` = 106 —
  the same id as the model's eos). `sb_llm_tokenize` keeps its old text-only
  behaviour, byte for byte; the new flag lives in `sb_llm_tokenize_ex`.
- Python is not used anywhere in the binding.

## Tests

| test | what it covers |
|---|---|
| `tests/test-tensor.scm` | tensors, parameters, dtypes, shapes (no native library) |
| `tests/test-nn.scm` | layers, model composition, forward passes through the runtime |
| `tests/test-nn-trace.scm` | the emitted-text and traced routes agree on the same model |
| `tests/test-train.scm` | the native autograd tape and XOR training |
| `tests/test-executable.scm` | compile-once, reuse, lifetime, the cache |
| `tests/test-trace-lower.scm` | lowering coverage, and the runtime matching the reference interpreter |
| `tests/trace/test-trace-forms.scm` | the tracer's forms and graph breaks |
| `tests/test-inference.scm` | the LLM C ABI: golden decode, round-trips, prefix reuse, streaming, opt-in control-token parsing |
| `tests/test-e2e-export.scm` + `tests/run-e2e.sh` | train → export → an independent C++ loader |
| `tests/serve/*` | the HTTP server, over HTTP/SSE/WebSocket and as units |
