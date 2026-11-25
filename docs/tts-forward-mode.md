# Forward-mode device graphs (TTS)

Status: implemented and tested. This page describes the inference mode of the
device graph, the weight store it binds from, and the Guile surface
`(sonicboom compile)`. The GPU proof (a small MLP and a Qwen2-shaped block run
against PyTorch reference vectors) lives in the private test suite; the public
suite exercises the same API on the CPU interpreter.

## What it is

A **forward-mode device graph** is one traced Scheme function compiled into an
acyclic inference graph and executed node by node on the GPU. It is the same
machinery as the training device graph (`DeviceGraph`), in the other mode:

- the graph is lowered Blade → Boom IR and run through the **Forward** pass
  phase (dead-node elimination, then buffer reuse) instead of the Training
  phase;
- its weights are **not** inlined in the graph text: each is a
  `(data :bound "<name>")` constant parameter, recorded by name at compile and
  resolved from a **weight store** before the first run;
- CUDA-graph capture of the node launches is **on by default** per compiled
  shape;
- the executable owns its device memory, and the weight store owns the weight
  buffers that every executable bound to it shares.

Nothing in this path changes the training mode: a training graph compiles and
runs exactly as before (its tests and goldens are the guard).

## The weight store

A weight store maps one file once and supplies its named tensors to compiled
graphs:

- **safetensors** (the PyTorch/TTS world) and **GGUF** (the LLM world). `auto`
  detects GGUF from its magic and reads anything else as safetensors; an
  explicit format that the file is not is an error.
- A graph reads **float32** weights. An f32 entry is supplied as stored; f16,
  bf16 and f64 entries are converted on the host when the store is opened with
  `convert-to-f32`. A quantized GGUF entry (`Q4_K`, `Q6_K`, …) cannot be
  supplied to a float32 graph and is reported by name and type when a graph
  asks for it.
- Each weight is uploaded to the device **on first use** and cached in the
  store, so several graphs bound to one store — a flow step, a vocoder chunk,
  a language-model head — upload the weights once and share them.
- The store may be freed before the executables: an executable keeps the store
  (and its device buffers) alive itself.

## The Guile surface

```scheme
(use-modules (sonicboom compile)
             (sonicboom trace symbolic))

(define model (load-model "flow.safetensors"))

(define (dit-step x w)
  (let* ((h (matmul x (w "blocks.0.attn.to_q.weight" 'float32 '(64 64))))
         (h (matmul h (w "blocks.0.ff.down.weight" 'float32 '(128 64))))
         (b (w "blocks.0.ff.down.bias" 'float32 '(1 64))))
    (relu (add h b))))

(define fwd (compile-forward model
                             '((x float32 (2 64)))
                             dit-step
                             #:backend 'auto))
(fwd (make-ctensor 'float32 '(2 64) (make-list 128 0.5)))
```

`compile-forward` calls `dit-step` once, on symbolic tensors:

- one symbolic tensor per `inputs` entry, in order, then a **weight accessor**
  `w` last: `(w NAME DTYPE SHAPE)` returns the symbolic parameter for the
  weight `NAME` in the store, with the given dtype and shape. Ask for the same
  name again with the same shape and you get the same parameter; asking with
  another shape or dtype is an error, because one name denotes one stored
  tensor. Only float32 weights are supplied.
- The function must return a symbolic tensor or a list of them. It must not
  branch on a tensor value (`bool` on one): a data-dependent branch cannot be
  one forward graph, and is reported.

The result is an ordinary procedure taking one ctensor per input and returning
the output tensor, or the list of outputs in return order. Runs of one compiled
forward are serialized; different compiled forwards may run concurrently.

`#:backend` selects `'device` (requires a CUDA build), `'cpu` (the interpreter,
always available) or `'auto` (device when the loaded library has CUDA). Both
backends bind the **same** weights from the same file: the device uploads them,
the CPU interpreter copies them from the store's host bytes.

`compiled-forward-memory-stats` reports the device memory a compiled forward
holds — activation slots (the buffer-reuse assignment), bound weights (owned by
the store), scratch, and their sum — as an alist. A CPU forward reports zero.

The compiled forward owns its native executable: a GC guardian releases it when
the procedure is collected, `free-compiled-forward!` releases it explicitly
(idempotent; later runs raise), and `reap-compiled-forwards!` releases
collected ones. `load-model` has the matching `free-weights!` and
`reap-weights!`.

### Shapes and buckets

Shapes are static per executable. A variable-length input is handled by
**buckets** — the length rounded up to a compiled size — with the caller
passing the valid length explicitly, usually as a mask input the graph applies
(its meaning is the model's). Compiling one graph per bucket against one store
is the supported pattern; the mask is what makes the padding harmless, and the
test of a bucket is that the same input at two buckets gives bit-identical
valid regions, with capture on and off. If a model cannot express that mask
(its attention would read the padded keys), the bucket sizes are a correctness
choice, not a packing trick: compile only the sizes the model supports.

## Errors

A compile reports, with the names involved:

- a weight the store lacks (`weight '…' is not in '<file>'`);
- a stored shape that differs from the graph's, naming both shapes, and a
  stored type the float32 graph cannot read, naming the type;
- an f16/bf16/f64 store opened without `convert-to-f32`;
- an input name reused by a generated value, a duplicate input name, an
  undeclared value the graph reads, and a data-dependent branch.

## C ABI (append-only)

The layer-2 API stays free of CUDA and native-torch types; the additions are
new entries in `capi/include/sonicboom/capi.h`:

- `boom_compile_device_ex(doc, inference, …)` — compile a device graph in either
  mode; `inference = 0` is exactly `boom_compile_device`.
- `boom_weights_open`, `boom_weights_free` — open/close a weight store.
- `boom_weights_data_f32`, `boom_weights_tensor_info` — a borrowed host f32
  view of a weight (how the CPU fallback binds) and the stored type/shape/bytes
  of an entry (how a caller checks a graph's weight before binding).
- `boom_executable_bind_weights` — bind every external weight of an inference
  executable by name.
- `boom_executable_set_capture` — turn CUDA-graph capture on or off (inference
  graphs default on).
- `boom_executable_memory_stats` — the per-executable memory breakdown above.

Weights are bound through the same `boom_run_multi` entry as any other
executable; an inference graph is bound by name through `boom_input_name_at` /
`boom_output_name_at`, exactly as a training graph is.
