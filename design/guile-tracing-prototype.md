# Guile tracing prototype — design and limitations

Status: **research prototype**. It is pure Guile 3.0 and does not depend on
`libsonicboom.so`. It is not built or tested by CMake/ctest; whether to add it
to ctest is a separate decision.

## 1. What it is

A tracer that turns a Scheme function written against a small tensor op set
into S-Expr v0.1 graphs (`design/s-expr-v0-spec.md`). Data-dependent control
flow is handled at two levels:

- **Syntactic branches (preferred).** `define-traced` / `traced-lambda` rewrite
  the body's `if` / `cond` / `when` / `unless` into `%branch`. On a symbolic
  condition, `%branch` traces both arms and joins them with a `where` node at
  the `if`. No continuation is involved, and the graph grows linearly in the
  number of `if`s.
- **Continuations (fallback).** For `if`s the macro cannot see, the user calls
  `bool`, which suspends tracing with `abort-to-prompt`. Three mechanisms work
  on that suspension:
  - **graph break**: run the partial graph on a pluggable executor and resume
    the continuation with the concrete value;
  - **trace-tree cache**: later calls follow cached segments and only trace
    again for a branch value not seen before;
  - **multi-resume**: resume the continuation with both `#t` and `#f` and join
    the two results with `where`.

On top of either mechanism:

- **guards**: the traced function's free variables are part of the cache key;
- **side-effect checks**: output written while tracing is captured and
  reported, and `trace-print` provides run-time printing;
- **conservative mode** (`#:retrace-on-miss?`): on a cache miss, trace again
  from the start instead of resuming a stored continuation;
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
  tracer.scm    (sonicboom trace tracer)    prompts, graph breaks, trace tree, multi-resume,
                                            guards, retrace-on-miss, output capture
  macro.scm     (sonicboom trace macro)     define-traced / traced-lambda, %branch
bindings/guile/tests/trace/
  test-trace.scm         continuation mechanism, S-Expr, DCE, cache (49 checks)
  test-trace-macro.scm   macro branches, guards, effects, retrace-on-miss (40 checks)
```

None of these modules imports `(sonicboom ffi)`, `(sonicboom tensor)` or any
other module that loads `libsonicboom.so`. No file outside the trace
directories and this document was changed.

## 3. Running the tests

From the repository root, with Guile 3.0.9+ (`apt install guile-3.0`):

```sh
guile --no-auto-compile -L bindings/guile/scheme \
    bindings/guile/tests/trace/test-trace.scm
guile --no-auto-compile -L bindings/guile/scheme \
    bindings/guile/tests/trace/test-trace-macro.scm
```

- Each script exits with status 0 when every check passes and 1 otherwise.
- Both scripts were also run without `--no-auto-compile`, where Guile compiles
  every file before running it, and pass that way too. This matters for
  `macro.scm`, which does work at expansion time.
- `test-trace-macro.scm` prints the node-count table from §4.7 to stdout.
- SRFI-64 writes `sonicboom-trace.log` / `sonicboom-trace-macro.log` to the
  current directory. Running from a scratch directory with an absolute `-L`
  path keeps those logs out of the tree.

## 4. Design

### 4.1 Tensors and modes

- A **symbolic tensor** (`<sym>`) holds only a value name, a dtype and a static
  shape.
- A **ctensor** (concrete tensor) holds a dtype, a shape and a flat row-major
  vector of Scheme numbers or booleans. It is used only by eager mode and the
  reference interpreter.
- The user ops are `add sub mul matmul relu neg sum gt lt where trace-print`,
  plus `bool`. They are defined once in `symbolic.scm`, and the mode is the
  `current-tracer` parameter:
  - `#f`: eager mode. The op is computed with the reference kernel.
  - a recorder: tracing mode. If any operand is symbolic, the op appends a node
    and returns a new symbolic tensor. If every operand is concrete, a pure op
    is still computed, which gives constant folding for free. Effectful ops
    (`print`) are always recorded. Concrete operands of a recorded node become
    inline `(parameter … (data :values …))` entries.
- The tracer switches mode with `parameterize`. `(tracing?)` reports the
  current mode.
- Shapes and dtypes come from the type rule in `ops.scm`. The interpreter uses
  the kernel from the same table entry, so traced types and computed types
  cannot drift apart. The interpreter also checks that every node's declared
  output type matches what it computed.

S-Expr op names: `add sub mul matmul relu neg greater less where reduce_sum
print`. `sum` is emitted as `reduce_sum` with `(axes (ints …))` and
`(keepdims (int 0|1))` attributes. Empty `axes` means all axes. `print` carries
a `(label (string …))` attribute. Segments are named `trace`, `trace.t`,
`trace.t.f`, …, after their path in the tree.

### 4.2 Graph break with `call-with-prompt` / `abort-to-prompt`

`bool` forces a concrete decision. If the argument is a symbolic tensor and
neither the current path nor an enclosing `%branch` arm has already fixed its
value, it calls `(abort-to-prompt %break-tag x)`.

The tracer runs the user function as:

```scheme
(parameterize ((current-tracer rec)
               (current-output-port recording-port))
  (call-with-prompt %break-tag body handler))
```

The handler receives the continuation `k` and the condition. The recorder `rec`
holds the partial graph recorded so far. The handler then does one of two
things:

1. **Multi-resume** (§4.4) if enabled and possible.
2. **Graph break** otherwise.
   - It emits the current segment as an S-Expr v0.1 document. The outputs are
     the condition plus every value the segment defines, because at this point
     it cannot know which of them the continuation will read.
   - It runs that document on the executor, with output silenced so `print`
     nodes do not fire twice, and stores the results in the call's concrete
     environment.
   - It records a branch node holding the segment, the condition name and `k`.
   - It resumes `(k v)` under a new prompt with a fresh recorder, so the rest
     of the function becomes the next segment.

Why the `parameterize` sits outside the prompt: a Guile partial continuation
captures only the dynamic bindings between the prompt and the abort. Bindings
outside the prompt come from wherever the continuation is resumed. So when `k`
is resumed, the user's ops record into the new recorder and print into the new
recording port, while any `parameterize` the user code set up inside is
restored. This was checked on Guile 3.0.9.

The captured stack segment is only the user function's frames between that
prompt and `bool`; the tracer's and the caller's frames are outside it. It does
not grow across repeated resumptions, because each new abort captures from the
new prompt. Cache-hit calls run no user code and touch no continuation.

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
is and are part of the cache key. The cache key is:

- the **signature**: one `(tensor dtype shape)` or `(static value)` entry per
  argument;
- the **guard values** (§4.8).

Each key owns one trace tree:

- A **leaf** holds a segment that ends with the function's return value.
- A **branch** holds a segment that ends with a condition, the suspended
  continuation `k` (not kept in retrace-on-miss mode), and its children keyed by
  `#t` / `#f`.

On a call:

1. **No tree for this key.** Trace from the start, then run the tree.
2. **Tree exists.** Run segments from the root. At each branch, read the
   condition and follow the matching child.
3. **No child for this value (cache miss).** Silently rebuild every concrete
   value on the path, because a later break in the new suffix may read any of
   them. Then one of:
   - resume the stored continuation of that branch with the new value, so only
     the new suffix is traced;
   - in retrace-on-miss mode, trace again from the start (§4.10).

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
| `traces` | fresh traces, cache-miss resumptions and retraces |
| `guard-misses` | fresh traces caused by a changed guard value |
| `retraces` | retraces done in retrace-on-miss mode |
| `retrace-divergences` | retraces whose prefix did not match the cache |
| `segments-traced` | segments traced |
| `graph-breaks` | graph breaks taken |
| `merges` | continuation multi-resume joins |
| `branch-merges` | macro `if` joins (`where` nodes from `%branch`) |
| `output-warnings` | tracing runs that wrote to the output port |
| `executor-calls` | executor invocations |

Three functions let you inspect a traced function:

- `trace-tree` returns each tree with its signature and data guard values;
- `trace-segments` returns each segment's S-Expr text;
- `trace-warnings` returns the output captured while tracing.

### 4.4 Multi-resume (continuation)

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

Because a delimited continuation runs to the end of the function, each side
re-traces everything after the `if`. §4.7 shows the cost.

`#:multi-resume? #t` is the user's statement that the code after each `bool`
point has no Scheme-level side effects. The tracer cannot check this.

### 4.5 DCE

`eliminate-dead-nodes` makes one backward pass from the segment's outputs:

- A node is live if it is effectful (`print`) or any of its outputs is live.
- A parameter is live if a live node or a graph output references it.

DCE runs on every emitted segment:

- partial graphs at a break, whose outputs are all defined values, so it only
  drops unused parameters;
- cached segments, whose outputs are the minimal set from §4.3;
- merged segments, from multi-resume and from `%branch`.

### 4.6 Syntactic branches: `define-traced` / `traced-lambda`

```scheme
(define-traced (f x n) [#:option value …] body …)
(traced-lambda formals [#:option value …] body …)   ; options as trace-function
```

Both expand to `(trace-function (lambda formals body′) #:guards … options …)`.

**The rewrite.** `body′` is `body` inside a `let-syntax` that rebinds `if`,
`cond`, `when` and `unless` in the user's own syntactic context. The
`let-syntax` uses `datum->syntax` on the macro keyword, so the keywords written
in the body are captured. Each use becomes:

| Written in the body | Becomes |
|---|---|
| `(if test then else)` | `(%branch 'loc test (lambda () then) (lambda () else))` |
| one-armed `if`, `when`, `unless` | the same, with an empty arm, marked as a statement |
| `cond` (including `else` and `=>`) | a chain of `%branch` calls |

`loc` is the form's `syntax-source` (file, line, column). Macros defined
elsewhere, and `and`/`or`/`case`/`do`, still expand to the core `if` and are
not rewritten.

**`%branch` at run time:**

- **The test is a ctensor:** branch on its truth value (one element).
- **The test is any other non-symbolic value:** ordinary Scheme `if`
  semantics, so static loop counters and `(= i n)` work unchanged.
- **The test is a symbolic tensor while tracing:** it must be a one-element
  `bool` tensor, otherwise an error with the location is raised.
  - If the condition is already known (an enclosing arm or the current path
    fixed it), only that arm runs.
  - Otherwise both thunks run. Each runs with the condition fixed in
    `%branch-assumptions`, which is a parameter, so a continuation captured
    inside an arm restores it. The two results are then joined:

    | Arm results | Join |
    |---|---|
    | the same object | that object |
    | two tensors | identical dtype and shape required, then `where(c, a, b)` |
    | a tensor and a Scheme number or boolean | the scalar becomes a rank-0 constant of the tensor's dtype, then `where` (it broadcasts) |
    | two Scheme numbers or booleans | rank-0 constants (`float32` / `bool`), then `where` |
    | pairs/lists and vectors | joined element by element |
    | equal non-tensor values | kept |
    | anything else | error |

    Errors read `traced if at <file>:<line>:<col>: …`, and a dtype/shape
    mismatch reports both arms.
  - Statement forms (one-armed `if`, `when`, `unless`) trace both arms and
    return unspecified.

The arms are thunks, so the code after the `if` runs once, on the joined
value. No prompt, abort or continuation is involved, and the graph grows
linearly (§4.7).

**Fallback.** An `if` the rewrite cannot see is still a core `if`. Put `bool`
on its test to get the continuation mechanism of §4.2–§4.4:

```scheme
(define-syntax my-if (syntax-rules () ((_ c a b) (if c a b))))
(define-traced (f x) (my-if (bool (gt (sum x) 0.0)) …))
```

Without `bool`, a core `if` on a symbolic tensor treats the record as true.
That is the prototype's main hazard (§5).

### 4.7 Node counts: syntactic join vs continuation multi-resume

The test is `if` inside a loop with a static iteration count n, traced into one
graph with no graph break. Node counts are the total number of nodes in the
emitted graph, measured by `test-trace-macro.scm`:

| iterations | macro `where` | continuation multi-resume |
|---|---|---|
| 1 | 7 | 7 |
| 2 | 14 | 21 |
| 3 | 21 | 49 |
| 4 | 28 | 105 |

- **Macro:** each iteration adds `reduce_sum greater mul sub neg add where`,
  so the count is 7n.
- **Continuation multi-resume:** each side re-traces the remaining iterations,
  so T(n) = 2·T(n−1) + 7, that is 7·(2ⁿ − 1): exponential. The 49 at n = 3 is
  the figure from the first version of this prototype.

### 4.8 Guards on free variables

A cached graph bakes in every value it closed over, for example
`(mul x k)` → parameter `k = 2.0`. `traced-lambda` therefore makes the body's
free variables guards. The guard values at call time are part of the cache key,
and a change causes a fresh trace (counted in `guard-misses`). Going back to an
earlier value hits the earlier trace again.

Free variables are found by walking every identifier in the body:

- **Outer lexicals.** At expansion time, `syntax-local-binding` reports
  `lexical` for them. Each becomes a guard thunk `(lambda () v)`.
- **Module-level names** (`global`). Each is looked up in the defining module
  when the function is called. Doing it at call time matters: when a file is
  compiled, its later top-level definitions are not bound yet while it is being
  expanded, so an expansion-time check would miss them.
- **Unbound names** read as a fixed marker and never cause a miss. Most of them
  are names bound inside the body itself.
- **Macros, core syntax and the formals** are skipped.

A name bound inside the body that shadows an outer binding adds an extra guard
on the outer one, which can cause an unnecessary retrace but never a missing
check.

How guard values are compared:

| Value | Snapshot | Comparison |
|---|---|---|
| numbers | as is | `eqv?` |
| strings, pairs, vectors | copied when the trace is stored | structurally, so in-place mutation is detected |
| procedures, records (ctensors), everything else | as is | identity |

`trace-tree` lists the data guards, leaving out procedures and unbound names.

`trace-function` also takes `#:guards` (an alist of `(name . thunk)`) for
callers that build traced functions without the macro.

### 4.9 Side effects

- **Output while tracing.** While user code is traced, including speculative
  multi-resume sides and both `%branch` arms, `current-output-port` is a fresh
  string port. If anything was written, the tracer:
  - prints a warning to `current-error-port`;
  - records the text (`trace-warnings`);
  - counts it (`output-warnings`).

  The text is not passed through, and it will not reappear on cached calls.
- **`trace-print`.** `(trace-print x [label])` returns `x` and prints
  `label: values`.
  - In eager mode it prints immediately.
  - While tracing it records an effectful `print` node, so it prints at run
    time with run-time values, once per call. DCE keeps it even when its result
    is unused.
  - Executions that only rebuild values (partial graphs at a break, path
    rebuilds after a miss) run with output silenced, so a call prints each
    `print` once.
- **Conservative mode.** `#:retrace-on-miss? #t` (§4.10).

### 4.10 Retrace on miss

With `#:retrace-on-miss? #t`, branch nodes keep no continuation. A cache miss
does not resume a stored `k`:

1. Trace the function again from the start on the current inputs. The new
   trace follows the same path up to the new branch value, using only
   within-call continuations at its graph breaks.
2. Match the new trace's prefix segments node by node against the cached path:
   same ops, attributes, parameter values and input structure. This yields a
   renaming from the new value names to the cached ones.
3. Rename the new suffix and graft it under the cached branch.

So the cache keeps the same tree as the continuation mode. The test checks this
for `if` inside a loop. Each miss re-runs the user's Scheme code from the start
(`segments-traced` is higher), and no stack segment stays alive in the cache.

If the prefix does not match, which means some state the guards do not cover
changed, the tracer:

- warns;
- counts it in `retrace-divergences`;
- replaces that tree with the fresh trace;
- restarts the call from the root.

### 4.11 What the tests check

Every traced result is compared with the same computation written
independently in plain Scheme over nested lists, not with the prototype's
kernels.

**`test-trace.scm` (49 checks):**

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

**`test-trace-macro.scm` (40 checks):**

- **Macro `if` inside a loop**, over the same 6 inputs:
  - one trace, no break, 3 `where` joins;
  - later calls hit the cache;
  - the graph is the loop unrolled;
  - the node-count table: linear 7n for the macro, 49 at n = 3 for the
    continuation version, superlinear growth for the continuation version.
- **`cond`, `when`, `unless`, nesting and returned values:**
  - three-arm `cond`;
  - `when`/`unless`/`if` with concrete tests keep Scheme semantics;
  - Scheme numbers in arms become constants joined by `where`;
  - lists of tensors are joined element by element;
  - nested `if`s.
- **Errors:**
  - arms with different shapes give an error naming `test-trace-macro.scm:<line>`;
  - a non-bool symbolic condition is an error.
- **Fallback:** an `if` inside a macro defined outside the body, written with
  `bool`, takes a graph break and resumes the continuation on the miss.
- **Guards:**
  - changing a closure's free variable gives a fresh trace and correct
    results, and going back to the old value hits the first trace;
  - a module variable changed with `set!` is detected;
  - only data variables are listed in the key.
- **Side effects:**
  - `display` while tracing is captured, warned once, and not written;
  - `trace-print` prints once per call with run-time values and survives DCE;
  - `trace-print` across a break and a miss still prints once per call;
  - `trace-print` in eager mode prints immediately.
- **Retrace on miss:**
  - single `if` and `if` inside a loop give correct results;
  - the grafted tree has the same shape as the continuation version;
  - later calls hit the cache;
  - more segments are traced (prefixes re-traced), with no divergence.

These are the measured numbers for the `bool` loop test with the 6 inputs, from
`trace-stats` and the segment node counts:

| Mode | traces | breaks | merges | segments (nodes each) |
|---|---|---|---|---|
| breaks only | 4 | 6 | 0 | 10 (2–4 nodes) |
| multi-resume, budget 64 | 1 | 0 | 7 | 1 (49 nodes) |
| multi-resume, budget 8 | 3 | 3 | 3 | 6 (2–9 nodes) |
| macro `if` (no `bool`) | 1 | 0 | 0 (3 `branch-merges`) | 1 (21 nodes) |

## 5. Limitations

### Syntactic branches

- **A core `if` on a symbolic tensor is silently wrong.** Only `if` / `cond` /
  `when` / `unless` written literally in a `define-traced` / `traced-lambda`
  body are rewritten. Code that reaches a core `if` sees the symbolic tensor as
  a true value without any error. This covers:
  - `and`, `or`, `case`, `do`;
  - macros defined elsewhere;
  - procedures called from the body;
  - plain `trace-function` procedures.

  Such code must use `bool`, which falls back to continuations.
- **Both arms always run while tracing**, including their side effects. Arms
  are traced, not executed, but Scheme-level effects in them happen once per
  arm.
- **Statement forms on a symbolic test** (`when`, `unless`, one-armed `if`)
  record both arms' nodes. A `trace-print` inside one prints whichever way the
  condition goes.
- **Joins are value-level.** Two arms that return the same tensor need no
  join. Otherwise every tensor-valued leaf of the result is selected with
  `where`, even if most of the arm work is shared. There is no CSE across arms.
- **Scheme numbers become tensors after a join.** Two different Scheme numbers
  from the arms become a rank-0 tensor, so code that then uses the result as a
  Scheme number fails.

### Guards

- **Guards cover the body's direct free variables only.**
  - State read inside called procedures, through accessors, or through
    mutable objects held in a guarded variable is not guarded, unless the
    variable itself is a pair, vector or string, which are snapshotted.
  - Records, including ctensors, compare by identity, so mutating a ctensor's
    data vector in place is not detected.
  - Procedures compare by identity, so redefining a helper does invalidate the
    cache, but a helper that reads changed state does not.
- **Over-guarding.** A body-internal name that shadows an outer binding adds
  an unnecessary guard.
- **Calls inside a trace.** A traced function called while another is being
  traced is inlined, and its own guards are not added to the caller's key.

### Effects

- **Only output to `current-output-port` is detected.** Ports opened by the
  user, `current-error-port`, `set!` and any other mutation are not.
- **`trace-print` on a merged path** (multi-resume or `%branch`) prints on both
  sides at run time, because `where` only selects values.
- **Retrace divergence.** When it happens the tree is replaced and the call
  restarts from the root, so `print` nodes in the already-executed prefix can
  fire twice in that call.

### Continuations

- **Multi-resume duplicates the whole rest of the function** and grows
  exponentially (§4.7). The budget and depth limits contain it. Prefer the
  macro.
- **Stored continuations assume purity.** Re-entering a stored continuation on
  a miss assumes the Scheme code between `bool` points depends only on the
  cache key: signature, guards and branch values. `#:retrace-on-miss? #t`
  avoids re-entering old continuations, at the cost of re-running user code
  from the start.
- **Unrewindable continuations.** If `bool` is reached through a C frame (for
  example a Scheme callback passed to a C-implemented primitive), Guile cannot
  resume the continuation. Plain Scheme loops, `map`, named `let` and `do` are
  fine.

### General

- **Cache misses are slow.** A miss replays the whole path with full outputs to
  rebuild the concrete environment. Partial graphs at a break export every
  defined value. Both choices are simple, not efficient.
- **One output per function.** The traced function must return one tensor, or
  a Scheme number or boolean that gets lifted to a constant. A `%branch` join
  can return lists, but the function's final result must still be one value.
- **Static shapes only**, as in S-Expr v0. Each new input shape or static
  argument gets a new tree.
- **Small op set:**
  - `matmul` is 2-D only;
  - broadcasting follows NumPy rules;
  - Scheme reals stand in for float32 (no rounding to binary32), so numerics
    are not float32-exact;
  - `NaN` and `inf` are not representable in the emitted text;
  - the op table is the prototype's own, not SonicBoom's operator registry.
- **Branch condition dtype.** Conditions must be `bool` tensors, both for
  `%branch` and for multi-resume. A `bool` on a numeric tensor always breaks.
- **Not thread-safe.** Each traced function keeps per-call mutable state.
- **No native execution.** Wiring to `libsonicboom.so` would mean an executor
  that hands each segment's text to the native S-Expr runtime. The executor
  interface is designed for that, but it is not done here.
