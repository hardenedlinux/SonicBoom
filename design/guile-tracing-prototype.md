# Guile tracing prototype — design and limitations

Status: **research prototype**. It is pure Guile 3.0 and does not depend on
`libsonicboom.so`. It is not built or tested by CMake/ctest; whether to add it
to ctest is a separate decision.

## 1. What it is

A tracer that turns a Scheme function written against a small tensor op set
into S-Expr v0.1 graphs (`design/s-expr-v0-spec.md`). It handles data-dependent
control flow with delimited continuations:

- **graph break**: when user code needs the concrete value of a symbolic
  tensor, tracing suspends at a prompt. The handler runs the partial graph on a
  pluggable executor, then resumes the continuation with the concrete value.
- **trace-tree cache**: a later call follows cached segments and only traces
  again when it reaches a branch value that has not been traced yet.
- **multi-resume**: for short side-effect-free continuations, the continuation
  is resumed once with `#t` and once with `#f`, and both results are joined by a
  `where` node. This removes the break.
- **DCE**: each emitted segment is pruned once, working back from its outputs.

The prototype exists to test the ideas, not to provide a runtime. Graphs are run
by a reference interpreter written in Scheme.

## 2. Layout

```text
bindings/guile/scheme/sonicboom/trace/
  tensor.scm    (sonicboom trace tensor)    concrete reference tensors ("ctensor")
  ops.scm       (sonicboom trace ops)       op table: type rule + reference kernel per op
  graph.scm     (sonicboom trace graph)     node/parameter records, S-Expr v0.1 emission
  dce.scm       (sonicboom trace dce)       dead-node elimination
  interp.scm    (sonicboom trace interp)    reference S-Expr interpreter (default executor)
  symbolic.scm  (sonicboom trace symbolic)  symbolic tensors, tracing mode, user ops, `bool`
  tracer.scm    (sonicboom trace tracer)    prompts, graph breaks, trace tree, multi-resume
bindings/guile/tests/trace/
  test-trace.scm
```

None of these modules imports `(sonicboom ffi)`, `(sonicboom tensor)` or any
other module that loads `libsonicboom.so`. No existing file or CMakeLists.txt
was changed.

## 3. Running the tests

From the repository root, with Guile 3.0.9+ (`apt install guile-3.0`):

```sh
guile --no-auto-compile -L bindings/guile/scheme \
    bindings/guile/tests/trace/test-trace.scm
```

The script exits with status 0 when every test passes and 1 otherwise. SRFI-64
also writes `sonicboom-trace.log` to the current directory. Running from a
scratch directory with an absolute `-L` path keeps that log out of the tree.

## 4. Design

### 4.1 Tensors and modes

- A **symbolic tensor** (`<sym>`) holds only a value name, a dtype and a static
  shape.
- A **ctensor** (concrete tensor) holds a dtype, a shape and a flat row-major
  vector of Scheme numbers or booleans. It is used only by eager mode and the
  reference interpreter.
- The user ops are `add sub mul matmul relu neg sum gt lt where`, plus `bool`.
  They are defined once in `symbolic.scm`, and the mode is the `current-tracer`
  parameter:
  - `#f`: eager mode. The op is computed with the reference kernel.
  - a recorder: tracing mode. If any operand is symbolic, the op appends a node
    and returns a new symbolic tensor. If every operand is concrete, the op is
    still computed, which gives constant folding for free. Concrete operands of
    a recorded node become inline `(parameter … (data :values …))` entries.
- The tracer switches mode with `parameterize`. `(tracing?)` reports the
  current mode.
- Shapes and dtypes come from the type rule in `ops.scm`. The interpreter uses
  the kernel from the same table entry, so traced types and computed types
  cannot drift apart. The interpreter also checks that every node's declared
  output type matches what it computed.

S-Expr op names: `add sub mul matmul relu neg greater less where reduce_sum`.
`sum` is emitted as `reduce_sum` with `(axes (ints …))` and `(keepdims (int 0|1))`
attributes. Empty `axes` means all axes. Each traced function's segments are
named `trace`, `trace.t`, `trace.t.f`, …, after their path in the tree.

### 4.2 Graph break with `call-with-prompt` / `abort-to-prompt`

`bool` is the only place user code can observe a tensor value. If the argument
is a symbolic tensor and the current path has not already fixed its value, it
calls `(abort-to-prompt %break-tag x)`.

The tracer runs the user function as
`(parameterize ((current-tracer rec)) (call-with-prompt %break-tag body handler))`.
The handler receives the continuation `k` and the condition. The recorder `rec`
holds the partial graph recorded so far. The handler then does one of two
things:

1. **Multi-resume** (§4.4) if enabled and possible.
2. **Graph break** otherwise. It emits the current segment as an S-Expr v0.1
   document whose outputs are the condition plus every value the segment
   defines (at this point it cannot know which of them the continuation will
   read). It runs that document on the executor and stores the results in the
   call's concrete environment. It records a branch node holding the segment,
   the condition name and `k`. Then it resumes `(k v)` under a new prompt with
   a fresh recorder, so the rest of the function becomes the next segment.

Why the `parameterize` sits outside the prompt: a Guile partial continuation
captures only the dynamic bindings between the prompt and the abort. Bindings
outside the prompt come from wherever the continuation is resumed. So when `k`
is resumed, the user's ops record into the new recorder, while any
`parameterize` the user code set up inside is restored. The prototype relies on
this, and it was checked on Guile 3.0.9.

A recorder carries **assumptions**: a map from condition name to the boolean
taken on the current path. A second `bool` on the same condition along the
same path returns the known value without breaking again.

`(executor text inputs) → outputs` is the pluggable executor interface. `text`
is the S-Expr text, `inputs` and `outputs` are alists of `(name . ctensor)`. The
default is `interpret-sexpr-text`, which re-reads the text with `read` on every
run, so the emitted text is round-tripped every time. You can replace it with
the `current-executor` parameter or `(trace-function f #:executor …)`.

### 4.3 Trace tree and cache

`trace-function` returns a wrapper procedure. Tensor arguments become symbolic
inputs named `arg<i>`. Other arguments are static: they are passed through as
is and are part of the cache key. The cache key (signature) is the list of
`(tensor dtype shape)` or `(static value)` entries. Each signature owns one
trace tree:

- A **leaf** holds a segment that ends with the function's return value.
- A **branch** holds a segment that ends with a condition, the suspended
  continuation `k`, and its children keyed by `#t` / `#f`.

On a call:

1. **No tree for this signature.** Trace from the start, then run the tree.
2. **Tree exists.** Run segments from the root. At each branch, read the
   condition and follow the matching child.
3. **No child for this value (cache miss).** Rebuild every concrete value on
   the path (a later break in the new suffix may read any of them), then
   resume the stored continuation of that branch with the new value. Only the
   new suffix is traced. The prefix segments are not traced again.

Value names come from one counter per traced function. This keeps names unique
across all resumptions of all continuations, so segments traced at different
times can refer to each other by name.

After the tree changes, `analyze!` recomputes each segment bottom-up:

```text
outputs(N) = { key(N) } ∪ (defs(N) ∩ needs(children))
needs(N)   = inputs(DCE(N, outputs(N))) ∪ (needs(children) − defs(N))
```

`key(N)` is the condition for a branch and the return value for a leaf. Each
segment is pruned with DCE and emitted once per tree change. Calls that hit
the cache reuse the emitted text.

`trace-stats` returns these counters:

| Counter | Meaning |
|---|---|
| `calls` | total calls |
| `cache-hits` | calls that traced nothing |
| `traces` | fresh traces plus resumptions after a cache miss |
| `segments-traced` | segments traced |
| `graph-breaks` | graph breaks taken |
| `merges` | successful multi-resume joins |
| `executor-calls` | executor invocations |

`trace-tree` shows the shape of the tree, and `trace-segments` lists each
segment's S-Expr text.

### 4.4 Multi-resume

With `#:multi-resume? #t`, the handler first tries `try-merge`:

- It resumes `(k #t)` and `(k #f)` under **speculative** recorders. These are
  O(1) copies of the current recorder, because node lists are shared-tail
  lists, and each copy has the matching assumption added.
- Each side runs until the function returns. A side may itself hit a `bool`.
  In that case it tries a nested merge, up to `#:max-depth` levels deep. A side
  never takes a graph break: if it needs one, the speculation fails.
- A side that records more than `#:max-side-nodes` nodes throws, which ends
  that speculation.
- If both sides finish and their results have the same dtype and shape, the
  recorder becomes prefix + `#t` side + `#f` side + `where(c, out_t, out_f)`.
  If both sides return the same value, no `where` node is added. This is a leaf
  of the tree, with no break.
- Otherwise the handler falls back to a graph break. The same `k` is still
  valid, because a delimited continuation can be resumed more than once.

The condition must have dtype `bool`. A `bool` on a numeric tensor always
breaks.

`#:multi-resume? #t` is the user's statement that the code after each `bool`
point has no Scheme-level side effects. The tracer cannot check this.

### 4.5 DCE

`eliminate-dead-nodes` makes one backward pass from the segment's outputs. A
node is live if any of its outputs is live. A parameter is live if a live node
or a graph output references it. Every op in the prototype is pure, so this is
exact.

DCE runs on every emitted segment:

- partial graphs at a break, whose outputs are all defined values, so it only
  drops unused parameters;
- cached segments, whose outputs are the minimal set from §4.3;
- merged multi-resume segments.

### 4.6 What the tests check

`bindings/guile/tests/trace/test-trace.scm` has 49 checks. Every traced result
is compared with the same computation written independently in plain Scheme
over nested lists, not with the prototype's kernels.

- **No branch** (`relu(matmul(x, w) + b)`, with bias broadcast):
  - node order and types in the emitted text;
  - the interpreter run on the text;
  - the traced call;
  - the second call hits the cache.
- **DCE:** dead nodes and unused constant parameters are removed.
- **Single data-dependent `if`, without multi-resume:**
  - the first call breaks once;
  - the same branch value hits the cache;
  - a new branch value resumes the stored continuation, tracing exactly one new
    segment with no new break;
  - both children are then cached;
  - each segment is pruned (`reduce_sum greater` / `mul` / `sub`).
- **Single `if` with multi-resume:**
  - one segment `reduce_sum greater mul sub where`, with no breaks;
  - both branch values are served by the same graph;
  - a budget of 0 falls back to a break.
- **`if` inside a loop** (3 iterations, static loop count), checked over 6
  inputs that take different paths:
  - with breaks: paths are 3 branches deep, repeated inputs all hit the cache,
    and a different static argument gets its own tree;
  - full multi-resume: one graph, no breaks;
  - tight budget: a mix of breaks and merges.
- **Misc:**
  - a repeated `bool` on the same condition causes only one break;
  - the executor can be plugged in by keyword and by parameter;
  - a traced function called while another is being traced is inlined;
  - `matmul` followed by `if`;
  - eager mode;
  - `bool` on a tensor with more than one element is an error.

These are the measured numbers for the loop test with the 6 inputs, from
`trace-stats` and the segment node counts:

| Mode | traces | breaks | merges | segments (nodes each) |
|---|---|---|---|---|
| breaks only | 4 | 6 | 0 | 10 (2–4 nodes) |
| multi-resume, budget 64 | 1 | 0 | 7 | 1 (49 nodes) |
| multi-resume, budget 8 | 3 | 3 | 3 | 6 (2–9 nodes) |

## 5. Limitations

- **Multi-resume duplicates the whole rest of the function.** The delimited
  continuation runs to the end of the function, not just to the end of the
  `if`, so the code after the branch is traced once per side. Nested branches
  therefore grow exponentially: 3 loop iterations already give 49 nodes. The
  budget and depth limits keep this in check. Real reconvergence would need
  either a syntactic `if` form that the tracer can see, or CSE/hash-consing
  across sides. Neither is implemented.
- **Purity is asserted, not checked.** If a resumed continuation performs
  `set!`, I/O, or mutation of captured state, it will do so once per resume, or
  on a later call after a cache miss. Graph-break resumption has the same
  caveat: a stored continuation from an earlier call is re-entered on a cache
  miss, which assumes the Scheme code between `bool` points depends only on the
  signature and the branch values. This is the same "guard" assumption a
  Python tracer makes, but there are no guards on closed-over Scheme state.
- **Unrewindable continuations.** If `bool` is reached through a C frame (for
  example a Scheme callback passed to a C-implemented primitive), Guile cannot
  resume the continuation. Plain Scheme loops, `map`, named `let` and `do` are
  fine.
- **Cache misses are slow.** A miss replays the whole path with full outputs to
  rebuild the concrete environment. Partial graphs at a break export every
  defined value. Both choices are simple, not efficient.
- **One output per function.** The traced function must return one tensor, or
  a Scheme number or boolean that gets lifted to a constant. Lists or tuples of
  tensors are not supported.
- **Static shapes only**, as in S-Expr v0. Each new input shape or static
  argument gets a new tree.
- **Small op set:**
  - `matmul` is 2-D only;
  - broadcasting follows NumPy rules;
  - Scheme reals stand in for float32 (no rounding to binary32), so numerics
    are not float32-exact;
  - `NaN` and `inf` are not representable in the emitted text;
  - the op table is the prototype's own, not SonicBoom's operator registry.
- **`bool` is the only concretization point.** There is no
  `item`/value-specialization and no shape-dependent break, because shapes are
  static and known at trace time.
- **Branch condition dtype.** Only `bool`-dtype conditions can be merged; a
  `bool` on a numeric tensor always breaks.
- **Not thread-safe.** Each traced function keeps per-call mutable state.
- **No native execution.** Wiring to `libsonicboom.so` would mean an executor
  that hands each segment's text to the native S-Expr runtime. The executor
  interface is designed for that, but it is not done here.
