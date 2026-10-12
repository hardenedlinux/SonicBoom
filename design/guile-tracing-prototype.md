# Guile tracing prototype — design and limitations

Status: **research prototype**. It is pure Guile 3.0 and does not depend on
`libsonicboom.so`. It is not built or tested by CMake/ctest; whether to add it
to ctest is a separate decision. It is written so that other Scheme
implementations can borrow and port it (§6).

## 1. What it is

A tracer that turns a Scheme function written against a small tensor op set
into S-Expr v0.1 graphs (`design/s-expr-v0-spec.md`). Data-dependent control
flow is handled at two levels:

- **Surface rewriting (preferred).** `define-traced` / `traced-lambda` run the
  body through a code walker written only in `syntax-rules`. It rewrites these
  R7RS control forms so that a symbolic condition is resolved inside the graph:
  `if`, `when`, `unless`, `cond` (including `=>`), `and`, `or`, `case`, `do`,
  named `let`.
  - Conditionals trace both arms and join them with `where`.
  - `and` / `or` become logical tensor ops.
  - Nothing is suspended, and the graph grows linearly in the number of
    conditionals.
- **Continuations (fallback).** Where the rewrite cannot reach (macros defined
  elsewhere, ordinary procedures called from the body), the user decides with
  `bool`. `bool` suspends tracing with a delimited continuation. Three
  mechanisms work on that suspension:
  - **graph break**: run the partial graph on a pluggable executor and resume
    with the concrete value;
  - **trace-tree cache**: later calls follow cached segments and trace again
    only for unseen branch values;
  - **multi-resume**: resume with both values and join with `where`.

On top of either mechanism:

- **guards**: the traced function's free variables are part of the cache key;
- **side-effect checks**: output written while tracing is captured and
  reported, and `trace-print` provides run-time printing;
- **conservative mode** (`#:retrace-on-miss?`);
- **strict mode**: reports symbolic tensors reaching procedures the rewrite
  does not cover (`trace-pure` exempts a procedure);
- **loops**: `do` and named `let` with a static trip count are unrolled up to
  `#:max-unroll` iterations (default 64); a longer loop becomes a host loop
  that runs one cached body graph per iteration (§4.14);
- **reverse-mode autodiff** (§4.16): `grad` appends the backward pass to the
  traced graph, so a whole training step (forward, loss, gradients, SGD
  update) is one graph;
- **ATen names**: every emitted node whose op has a PyTorch Core ATen
  counterpart carries it as an `aten` attribute (§4.17);
- **DCE**: each emitted segment is pruned once, working back from its outputs.

A traced function may return a tensor or a structure (lists, vectors) of
tensors; the graph then has one output per tensor.

Every implementation-specific mechanism sits in one module, `compat.scm`.

The prototype exists to test the ideas, not to provide a runtime. Graphs are run
by a reference interpreter written in Scheme.

## 2. Layout

```text
bindings/guile/scheme/sonicboom/trace/
  compat.scm    (sonicboom trace compat)    ALL implementation-specific mechanisms (§6)
  tensor.scm    (sonicboom trace tensor)    concrete reference tensors ("ctensor")
  ops.scm       (sonicboom trace ops)       op table: type rule + reference kernel per op
  graph.scm     (sonicboom trace graph)     node/parameter records, S-Expr v0.1 emission
  dce.scm       (sonicboom trace dce)       dead-node elimination
  interp.scm    (sonicboom trace interp)    reference S-Expr interpreter (default executor)
  symbolic.scm  (sonicboom trace symbolic)  symbolic tensors, tracing mode, user ops, `bool`
  tracer.scm    (sonicboom trace tracer)    prompts, graph breaks, trace tree, multi-resume,
                                            guards, retrace-on-miss, output capture
  loop.scm      (sonicboom trace loop)      %loop: unrolling with a limit, host loops
  autodiff.scm  (sonicboom trace autodiff)  reverse-mode grad on the traced graph
  macro.scm     (sonicboom trace macro)     syntax-rules walker, %branch/%and/%or/%case,
                                            strict mode (%call)
bindings/guile/tests/trace/
  test-trace.scm         continuation mechanism, S-Expr, DCE, cache        (49 checks)
  test-trace-macro.scm   if/cond/when/unless, guards, effects, retrace     (40 checks)
  test-trace-forms.scm   and/or/case/do/named let/cond =>, nesting, host
                         loops, strict mode, compat boundary               (46 checks)
  test-trace-grad.scm    gradients vs finite differences, double where,
                         ATen attributes, XOR training                     (33 checks)
```

None of these modules imports `(sonicboom ffi)`, `(sonicboom tensor)` or any
other module that loads `libsonicboom.so`. No file outside the trace
directories and this document was changed.

`compat` imports no other prototype module. The modules that need a mechanism
from it (`dce`, `interp`, `symbolic`, `tracer`, `loop`, `autodiff`, `macro`)
import it. `tensor`, `ops` and `graph` need none.

## 3. Running the tests

From the repository root, with Guile 3.0.9+ (`apt install guile-3.0`):

```sh
for t in test-trace test-trace-macro test-trace-forms test-trace-grad; do
  guile --no-auto-compile -L bindings/guile/scheme \
      bindings/guile/tests/trace/$t.scm || echo "$t FAILED"
done
```

- Each script exits with status 0 when every check passes and 1 otherwise.
- All four were also run without `--no-auto-compile`, where Guile compiles
  every file first, and pass that way too.
- `test-trace-macro.scm` prints the node-count table from §4.9.
- `test-trace-grad.scm` prints the largest gradient error and the XOR loss
  curve.
- SRFI-64 writes one `.log` per script to the current directory. Running from
  a scratch directory with an absolute `-L` path keeps the logs out of the tree.

## 4. Design

### 4.1 Tensors and modes

- A **symbolic tensor** (`<sym>`) holds only a value name, a dtype and a static
  shape.
- A **ctensor** (concrete tensor) holds a dtype, a shape and a flat row-major
  vector of Scheme numbers or booleans. It is used only by eager mode and the
  reference interpreter.
- The user ops are:
  `add sub mul div matmul transpose reshape expand relu neg sum gt lt where
  equal-to logical-and logical-or logical-not trace-print`, plus `bool` and
  `grad`. They are defined once in
  `symbolic.scm`, and the mode is the `current-tracer` parameter:
  - `#f`: eager mode. The op is computed with the reference kernel.
  - a recorder: tracing mode. If any operand is symbolic, the op appends a node
    and returns a new symbolic tensor. If every operand is concrete, a pure op
    is still computed (constant folding). Effectful ops (`print`) are always
    recorded. Concrete operands of a recorded node become inline
    `(parameter … (data :values …))` entries.
- The tracer switches mode with `parameterize`. `(tracing?)` reports the
  current mode.
- Shapes and dtypes come from the type rule in `ops.scm`. The interpreter uses
  the kernel from the same table entry, so traced types and computed types
  cannot drift apart. The interpreter also checks every node's declared output
  type against what it computed.

S-Expr op names: `add sub mul div matmul transpose reshape expand relu neg
greater less equal and or not where reduce_sum print loop`.

- `sum` is emitted as `reduce_sum` with `(axes (ints …))` and
  `(keepdims (int 0|1))` attributes. Empty `axes` means all axes.
- `print` carries a `(label (string …))` attribute.
- `reshape` / `expand` carry `(shape (ints …))`.
- `loop` (host loops, §4.14) carries `body` (the body graph's S-Expr text),
  `trip_count` and `state_count`.
- Every node whose op has a Core ATen counterpart also carries
  `(aten (string "aten::…"))` (§4.17). Kernels ignore it.
- Segments are named `trace`, `trace.t`, `trace.t.f`, …, after their path in
  the tree.

### 4.2 Graph break with delimited continuations

`bool` forces a concrete decision. If the argument is a symbolic tensor and
neither the current path nor an enclosing rewritten form has already fixed its
value, it calls `(%abort-to-prompt %break-tag x)`.

The tracer runs the user function as:

```scheme
(call-with-output-captured
  (lambda ()
    (parameterize ((current-tracer rec))
      (%call-with-prompt %break-tag body handler))))
```

The `%`-prefixed procedures come from compat. The handler receives the
continuation `k` and the condition. The recorder `rec` holds the partial graph
recorded so far. The handler then does one of two things:

1. **Multi-resume** (§4.4) if enabled and possible.
2. **Graph break** otherwise.
   - It emits the current segment as an S-Expr v0.1 document. The outputs are
     the condition plus every value the segment defines, because it cannot yet
     know which of them the continuation will read.
   - It runs that document on the executor with output suppressed, so `print`
     nodes do not fire twice, and stores the results in the call's concrete
     environment.
   - It records a branch node holding the segment, the condition name and `k`.
   - It resumes `(k v)` under a new prompt with a fresh recorder.

The tracer and output bindings sit outside the prompt. A partial continuation
captures only the dynamic bindings between the prompt and the abort, so a
resumed `k` records into the new recorder and port, while any `parameterize`
inside the user code is restored.

The captured segment is only the user function's frames between the prompt
and `bool`. It does not grow across resumptions. Cache-hit calls run no user
code and touch no continuation.

A recorder carries **assumptions** (condition name → boolean on the current
path). A second `bool` on the same condition returns the known value.

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
- the **guard values** (§4.10).

Each key owns one trace tree:

- A **leaf** holds a segment that ends with the function's return value.
- A **branch** holds a segment that ends with a condition, the suspended
  continuation `k` (not kept in retrace-on-miss mode), and its children keyed by
  `#t` / `#f`.

On a call:

1. **No tree for this key.** Trace from the start, then run the tree.
2. **Tree exists.** Run segments from the root, following the child that
   matches each condition.
3. **No child for this value (cache miss).** Silently rebuild every concrete
   value on the path, then one of:
   - resume the stored continuation with the new value;
   - in retrace-on-miss mode, trace again from the start (§4.13).

Value names come from one counter per traced function, so segments traced at
different times can refer to each other by name.

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
| `branch-merges` | `where` joins from rewritten conditionals |
| `logic-merges` | `and` / `or` nodes from rewritten `and` / `or` |
| `loop-breaks` | rewritten branches resolved by a break because they re-entered themselves (§4.8) |
| `host-loops` | `loop` nodes emitted (§4.14) |
| `loop-body-traces` | host-loop iterations traced |
| `loop-body-hits` | host-loop iterations whose body graph equalled the previous one |
| `output-warnings` | tracing runs that wrote to the output port |
| `strict-warnings` | strict-mode reports |
| `executor-calls` | executor invocations |

Four functions let you inspect a traced function:

- `trace-tree` returns each tree with its signature and data guard values;
- `trace-segments` returns each segment's S-Expr text;
- `trace-warnings` returns the output captured while tracing;
- `trace-strict-warnings` returns the strict-mode reports.

### 4.4 Multi-resume (continuation)

With `#:multi-resume? #t`, the handler first tries `try-merge`:

- It resumes `(k #t)` and `(k #f)` under speculative recorders. These are O(1)
  copies, because node lists are shared-tail lists.
- Each side runs until the function returns. Nested `bool`s try nested merges,
  up to `#:max-depth` levels deep.
- A side that needs a break, or that records more than `#:max-side-nodes`
  nodes, fails the speculation.
- If both sides finish and their results have the same dtype and shape, the
  recorder becomes prefix + `#t` side + `#f` side + `where(c, out_t, out_f)`.
- Otherwise the handler falls back to a graph break. The same `k` is still
  valid.

A delimited continuation runs to the end of the function, so each side
re-traces everything after the `if` (§4.9).

`#:multi-resume? #t` is the user's statement that the code after each `bool`
point has no Scheme-level side effects.

### 4.5 DCE

`eliminate-dead-nodes` makes one backward pass from the segment's outputs:

- A node is live if it is effectful (`print`) or any of its outputs is live.
- A parameter is live if a live node or a graph output references it.

DCE runs on every emitted segment:

- partial graphs at a break;
- cached segments;
- merged segments.

### 4.6 Why surface macros

Data-dependent control flow is rewritten **at the surface**: on the forms the
user wrote, by a `syntax-rules` walker, before the implementation's expander
and compiler see them. This was chosen for these reasons:

- **Portability.** `syntax-rules` is the one macro system every R7RS Scheme
  shares. The walker, the output forms and the run-time procedures (`%branch`,
  `%and`, `%or`, `%case`, `%call`) are plain R7RS plus SRFI 1/9. Another
  implementation can take `macro.scm` almost as is. Only three small expansion
  hooks in `compat.scm` (§6.2) touch the host expander, and each has a
  portable fallback.
- **Independence from the host compiler.** The rewrite does not rely on any
  implementation's expanded or intermediate representation, its node types,
  or its optimisation passes. Those differ between implementations and
  between releases of one implementation.
- **Predictable, reviewable coverage.** What is rewritten is exactly the set of
  standard forms written literally in the body (§4.7). A reader can tell from
  the source which `if`s are traced into the graph.
- **Recursion and loops stay visible.** Named `let` and `do` are recognised as
  written, so a loop with a static trip count unrolls, and a loop whose trip
  count depends on the data is detected at run time (§4.8).

The cost is coverage. A surface rewrite cannot see:

- **Macros defined elsewhere.** Their expansions use the core `if`. The walker
  leaves such macro uses exactly as written.
- **Procedures called from the body** that were not themselves defined with
  `define-traced`. Their `if`s are core `if`s.

In both places a symbolic tensor reaching a core `if` is taken as a true value,
without any error. The fallback chain for these cases:

1. **`bool`.** Writing `(bool c)` at such a point forces a decision through the
   continuation mechanism (§4.2–§4.4): a graph break, or a multi-resume join.
   It is always correct.
2. **Strict mode** (§4.12) reports at trace time every procedure that receives
   a symbolic tensor without being covered, so the uncovered `if`s can be found
   and fixed, either by defining the procedure with `define-traced` or by
   adding `bool`.
3. **`trace-opaque`** marks a form the walker must leave alone, which is the
   explicit way to keep a core `if`.

### 4.7 The walker

```scheme
(define-traced (f x n) body …)
(define-traced* (option …) (f x n) body …)     ; options as trace-function
(traced-lambda formals body …)
(traced-lambda* (option …) formals body …)
```

All four expand to:

```scheme
(trace-function (lambda formals body′)
                #:guards (%free-variable-guards formals body …)
                option …)
```

**How it walks.** `body′` comes from a continuation-passing `syntax-rules`
walker.
- `(%walk form k)` rewrites `form` and passes the result to the continuation
  macro `k`.
- Standard forms are matched as `syntax-rules` literals, so a form is
  recognised only if its keyword is the standard binding.
- Binding forms are walked according to their shape, so binding lists and
  formals are never mistaken for calls.

**What each form becomes:**

| Form | Rewritten to | On a symbolic condition |
|---|---|---|
| `(if c a b)` | `(%branch site c (λ () a) (λ () b))` | both arms traced, results joined with `where` |
| `(if c a)`, `when`, `unless` | `%branch` marked as a statement | both arms traced, result unspecified |
| `cond` (`else`, `=>`, test-only clauses) | a `%branch` chain, one site per clause | one `where` per clause; `=>` receives the symbolic condition |
| `and`, `or` | `%and` / `%or` chains | no short-circuit: the rest is evaluated too, combined with the `and` / `or` op (§4.15); operands must be bool |
| `case` (`=>`, `else`, `else =>`) | `(%case site key clauses else)` | each clause tests `equal(key, d)` per datum, OR-ed, then a `%branch` chain |
| `do` | `(%loop site (λ (loop v …) (%branch site test …)) init …)` | static trip count: unrolled, or a host loop beyond `#:max-unroll` (§4.14); a symbolic test is a data-dependent loop (§4.8) |
| named `let` | `(%loop site (λ (name v …) body′) init …)` | same as `do`; conditionals inside follow their own rows |
| `lambda`, `case-lambda`, `(define (f …) …)` | body walked, procedure marked covered | — |
| `let`, `let*`, `letrec`, `letrec*`, `let-values`, `let*-values`, `define`, `set!`, `begin`, `parameterize`, `delay`, `delay-force` | inner expressions walked | — |
| `quote`, `quasiquote`, `(trace-opaque form)` | left as is | — |
| a call `(f arg …)` | `(%call f arg′ …)` | strict-mode check (§4.12) |
| any other macro use | left exactly as written, arguments included | — |

`%branch`, `%and`, `%or` and `%case` look at the value of the test or key:

- **A concrete ctensor:** decided by its single element.
- **Any other non-symbolic value:** R7RS semantics exactly. `(and 1 n)`,
  `(case n ((3 4) => …))`, `(do ((i 0 (+ i 1))) ((= i n) s))` behave as usual.
  This is tested.
- **Symbolic, while tracing:** the condition must be a one-element `bool`
  tensor, otherwise an error is raised with the form's location.

**Joining arm results** (a symbolic `%branch`):

| Arm results | Join |
|---|---|
| the same object | that object |
| two tensors | identical dtype and shape required, then `where(c, a, b)` |
| a tensor and a Scheme scalar | the scalar becomes a rank-0 constant of the tensor's dtype, then `where` |
| two different Scheme scalars | rank-0 constants, then `where` |
| pairs, lists or vectors | joined element by element |
| equal non-tensor values | kept |
| anything else | error with the location |

Each arm runs with the condition fixed in `%branch-assumptions`. That is a
parameter, so a continuation captured inside an arm restores it.

**Distinguishing calls from macro uses.** For a form whose head is not a
standard keyword, the walker emits `(%maybe-call %call original walked)`.

- If the head names a variable, the compat hook expands this to the walked
  call.
- If the head names a macro or core syntax, the hook expands it to the
  original form. Unknown macros are therefore never broken by the rewrite, and
  their arguments are not rewritten.

**Sites.** `(%site form)` is the form's source location, a file/line/column
alist (compat). It is used in error messages and as the branch identity in
§4.8. Clauses of one `cond` / `case` / `and` / `or` get distinct sites,
`#(site marker)`.

### 4.8 Data-dependent loops

A loop whose exit test is symbolic, written with `do`, named `let` or plain
recursion, cannot be unrolled: tracing both arms of the exit test would recurse
forever. `%branch` (and `%and` / `%or`) keep a parameter of the sites whose
join is in progress.

1. When a site is re-entered from inside its own join, the inner call signals.
2. The outer call catches the signal and drops what its join recorded,
   restoring the recorder's node lists.
3. It counts a `loop-break` and resolves the condition with `bool`, which is a
   graph break.

The loop therefore produces one segment per iteration in the trace tree, and
different trip counts become different paths, each cached. A loop with a
static trip count (`(= i n)` on a Scheme counter) never reaches this, because
its test is concrete.

Detection is per site, so only genuine re-entry triggers it. An `if` evaluated
as a loop argument completes before the recursive call.

### 4.9 Node counts: syntactic join vs continuation multi-resume

The test is `if` inside a loop with a static iteration count n, traced into one
graph with no graph break. Node counts are the total number of nodes in the
emitted graph, measured by `test-trace-macro.scm`:

| iterations | macro `where` | continuation multi-resume |
|---|---|---|
| 1 | 7 | 7 |
| 2 | 14 | 21 |
| 3 | 21 | 49 |
| 4 | 28 | 105 |

- **Macro:** linear, 7n.
- **Continuation multi-resume:** T(n) = 2·T(n−1) + 7 = 7·(2ⁿ − 1), because each
  side re-traces the rest of the function.

### 4.10 Guards on free variables

A cached graph bakes in every value it closed over. The guard values at call
time are part of the cache key, and a change causes a fresh trace (counted in
`guard-misses`). Going back to an earlier value hits the earlier trace again.

`(%free-variable-guards formals body …)` is a compat hook. In Guile it:

- walks the body's identifiers;
- asks `syntax-local-binding` which ones are outer lexicals, and guards those
  with `(lambda () v)`;
- guards module-level names by looking them up in the defining module at call
  time. A compiled file is expanded before its later top-level definitions are
  bound, so an expansion-time check would miss them.

Unbound names read as a fixed marker and never cause a miss.

How guard values are compared:

| Value | Snapshot | Comparison |
|---|---|---|
| numbers | as is | `eqv?` |
| strings, pairs, vectors | copied when the trace is stored | structurally |
| procedures, records | as is | identity |

`trace-tree` lists the data guards, leaving out procedures and unbound names.
`trace-function` also takes `#:guards` (an alist of `(name . thunk)`).

### 4.11 Side effects

- **Output while tracing.** While user code is traced, including speculative
  sides and both arms of rewritten conditionals, `current-output-port` is
  captured. If anything was written, the tracer:
  - prints a warning to `current-error-port`;
  - records the text (`trace-warnings`);
  - counts it (`output-warnings`).

  The text is not passed through.
- **`trace-print`.** `(trace-print x [label])` returns `x` and prints
  `label: values`.
  - In eager mode it prints immediately.
  - While tracing it records an effectful `print` node, which prints at run
    time, once per call, and is kept by DCE.
  - Executions that only rebuild values run with output suppressed.
- **Conservative mode.** `#:retrace-on-miss? #t` (§4.13).

### 4.12 Strict mode

`(traced-lambda* (#:strict? #t) …)`, or `(parameterize ((strict-tracing #t)) …)`
as the default for traced functions created without `#:strict?`.

Every call in a traced body goes through `%call`. While tracing in strict mode,
if an argument carries a symbolic tensor (directly, or inside a list or vector
up to depth 2) and the callee is not covered, it reports the callee once per
traced function:

- on `current-error-port`;
- in `trace-strict-warnings`;
- in the `strict-warnings` counter.

These callees are covered and never reported:

- the tensor ops and `bool`;
- traced functions;
- procedures created inside a traced body: `lambda`, `case-lambda`, internal
  `define`, named `let`;
- the structural procedures (`list`, `cons`, `car`, `vector-ref`, `apply`,
  `map`, `for-each`, …), which only move values;
- ordinary procedures declared with `(trace-pure proc)`: the user's statement
  that `proc` has no `if` on tensor values. `trace-pure` returns `proc`, so
  `(define f (trace-pure (lambda …)))` also works.

For the higher-order ones (`map`, `for-each`, `apply`), the procedure argument
is checked instead.

Example: `plain-abs` is defined with an ordinary `define` and an ordinary `if`:

```scheme
(define (plain-abs x) (if (gt (sum x) 0.0) x (neg x)))          ; plain define
(define-traced (traced-abs x) (if (gt (sum x) 0.0) x (neg x)))
(define f (traced-lambda* (#:strict? #t) (x)
            (add (plain-abs x) (traced-abs x))))
(f #<ctensor ((-1.0 2.0) (-3.0 -4.0))>)
;; => ((0.0 0.0) (0.0 0.0))     ; expected 2·|x|: plain-abs returned x
;; stderr:
;; sonicboom-trace: strict: procedure plain-abs received a symbolic tensor but
;; is not defined with define-traced/traced-lambda: a core `if` inside it takes
;; the tensor as true (define it with define-traced, or decide with bool)
```

Strict mode only reports. It cannot see whether the callee actually contains
an `if`, so it can report procedures that are harmless, such as a helper that
only does arithmetic. It cannot see symbolic tensors smuggled through mutable
state.

### 4.13 Retrace on miss

With `#:retrace-on-miss? #t`, branch nodes keep no continuation. On a cache
miss:

1. Trace again from the start on the current inputs.
2. Match the new prefix against the cached path, node by node, which gives a
   renaming from the new value names to the cached ones.
3. Graft the renamed new suffix under the cached branch.

If the prefix does not match, the tracer warns, counts a `retrace-divergence`,
replaces the tree with the fresh trace, and restarts the call.

### 4.14 Loop unroll limit and host loops

`do` and named `let` expand to `(%loop site (lambda (self v …) body) init …)`
(`loop.scm`). One call of the lambda is one iteration; it continues by calling
`self`.

- **Eager mode:** ordinary named-let recursion.
- **Tracing, at most `#:max-unroll` iterations** (a `trace-function` /
  `traced-lambda*` option, default 64; `#f` never stops unrolling): the loop
  is unrolled into the graph.
- **Tracing, more iterations, with tensors among the loop variables:** the
  loop instance signals its entry. The unrolled iterations are dropped, by
  restoring the recorder, and the loop becomes a host loop.

A host loop works in three steps:

1. **Probe the iterations.** Each iteration's body is traced once more into a
   scratch recorder, on fresh symbolic loop variables. Static loop variables
   (the counter) keep their concrete values, so the Scheme control runs as
   usual, and `self` returns a "continue" token instead of recursing. The
   result is one small graph per iteration:
   - inputs `s0…` are the tensor loop variables;
   - inputs `x0…` are values from outside the loop, such as weights;
   - outputs are the next values of the tensor loop variables.

   Names are canonicalised, so two iterations that do the same tensor work
   produce identical text.
2. **Emit `loop` nodes.** Consecutive identical bodies are run-length encoded.
   Each run becomes one `loop` node in the main graph: inputs are the state
   plus the outside values, outputs are the new state, and the attributes are
   `body` (the text), `trip_count` and `state_count`.
3. **Trace the exit iteration** into the main graph, on the last `loop` node's
   outputs.

At run time the executor runs a `loop` node by executing its body graph
`trip_count` times. The reference interpreter parses each distinct body text
once and caches it (`interp-body-cache-stats`).

What the test of a 100-step `do` (limit 64) measures:

- the graph is `loop reduce_sum`, one `loop` node with `trip_count 100`;
- `loop-body-hits` is 99: the bodies of iterations 2 to 100 matched the cached
  body;
- the body graph is parsed once and reused for the other 99 iterations;
- a second call is a trace-cache hit that runs one graph and reuses the parsed
  body 100 times.

If the counter enters the tensor computation (`(add acc i)`), the bodies
differ, so every iteration is its own run (`loop-body-hits` 0). The result is
still correct.

Requirements:

- the loop continues only by a tail call to `self`;
- tensor loop variables keep their dtype and shape;
- the body takes no graph break.

### 4.15 `and` / `or` on symbolic conditions: no short-circuit

When the first operand of `and` / `or` is a symbolic tensor, its truth is not
known at trace time, so short-circuiting is impossible:

- **Both sides are evaluated.** The rest of the `and` (or `or`) is evaluated
  too, under the assumption that it is reached, and the result is the
  elementwise `and` / `or` tensor op of the two bool values.
- **`where` selects at run time.** When the result is an `if` test, the `if`
  becomes `where(c, then, else)`, and both arms are always computed.

This is the usual vectorised-select semantics. It is safe in the forward pass:
an arm that would fail or produce `inf` / `NaN` where it is not selected is
computed anyway, but `where` discards those positions.

**The backward pass is not automatically safe.** The gradient of
`where(c, a, b)` towards `a` is `where(c, g, 0)`, which is zero where `a` was
not selected. If `a = f(x)` and `f'(x)` is infinite there (`1/x` at `x = 0`),
the chain rule computes `0 · inf = NaN`. The NaN reaches `x`'s gradient even
though that position never influenced the output.

The usual remedy is the **double where**: wrap the derivative itself in a
second `where`, so the non-finite local derivative never meets the zero. §4.16
implements this inside the gradient transform, so user code does not need the
manual pattern of sanitising the input with an extra `where` (as is common in
JAX code). The tests check it for `where(x > 0, 1/x, 0)` at `x = 0`, and for
`1/x` inside a rewritten `if` whose condition is false.

### 4.16 Reverse-mode autodiff

`(grad loss wrt)` is called inside a traced function, on a symbolic one-element
`loss` and a symbolic tensor or list of them. It returns their gradients as
symbolic tensors appended to the graph being traced, so the traced function
can return loss, gradients and updated parameters together as one graph.
`(value-and-grad f)` wraps `f` as a traced function returning
`(loss gradient …)` for every tensor argument.

**Gradient rules** live in `ops.scm`, one pure function per op:

```scheme
(rule emit shape ins out g attrs)  ; -> list of input gradients (#f: none)
```

- `emit` creates a backward node.
- `shape` gives a value's shape.
- `ins` / `out` are the forward values, `g` the output gradient.

Rules exist for `add sub mul div neg matmul transpose reshape expand relu
reduce_sum where print`. The comparison and logical ops have no gradient.
`loop` has no rule, so differentiating through a host loop is an error; loops
under `#:max-unroll` are unrolled and differentiable.

**The transform** works as follows:

1. It finds the nodes of the current segment that are forward-reachable from
   `wrt`.
2. It seeds `loss` with 1 and walks those nodes in reverse topological order,
   applying each rule.
3. Elementwise ops return gradients at the output shape, which are reduced to
   the input shape with `sum-to-shape` (broadcasting).
4. Gradients reaching one value from several uses are added.
5. A `wrt` that does not influence the loss gets a zero gradient.

Forward values the backward nodes do not read, and forward nodes nothing
needs, are removed by the segment's DCE. For example, `grad(sum(x·x), x)`
alone leaves no `reduce_sum` in the graph.

**Double where.** A gradient carries the `where` conditions it passed through
as masks `(c, side)`:

- At each elementwise op, the local gradients are wrapped in
  `where(c, local, 0)` (or `where(c, 0, local)`) before reduction and
  accumulation.
- Elementwise masks follow values through elementwise ops of unchanged shape.
- Scalar masks (the rank-0 condition of a rewritten `if`) hold for every
  position and follow values through every op.

**Graph breaks.** The walk covers the current segment. If the loss depends on
a value computed in an earlier segment (before a graph break), `grad` refuses
with an error instead of silently treating that value as a constant.
Differentiating across breaks would mean saving each segment's inputs and
chaining the per-segment gradients. It is left out of the prototype, and
functions must not break before `grad`. Conditionals rewritten by the macro
never break.

**Checked** against central finite differences in double precision
(`h = 1e-5`, relative error `|a − n| / max(|a|, |n|, 1e-3)`):

- every op, including broadcasting and `sum` with axes and keepdims;
- a traced `if` (both sides);
- `case` inside a loop (three branches);
- an MLP loss.

The largest error measured is 1.4e-9.

**Training.** A traced XOR step contains the forward pass of a 2-8-1 ReLU MLP,
the MSE loss, `grad` and the SGD update (lr 0.1). It is traced once, into a
35-node graph. Each of the 1001 steps is then a cache hit that runs exactly
one graph. The loss goes from 0.866 to below 1e-6 by step 400.

### 4.17 Op names and PyTorch Core ATen

| Prototype op (S-Expr) | User function | Core ATen | Notes |
|---|---|---|---|
| `add` | `add` | `aten::add.Tensor` | `alpha` = 1 |
| `sub` | `sub` | `aten::sub.Tensor` | `alpha` = 1 |
| `mul` | `mul` | `aten::mul.Tensor` | |
| `div` | `div` | `aten::div.Tensor` | true division |
| `neg` | `neg` | `aten::neg` | |
| `matmul` | `matmul` | `aten::mm` | 2-D only; general `aten::matmul` is not Core ATen (it decomposes) |
| `transpose` | `transpose` | `aten::permute` | dims `(1 0)`; `aten::t` / `transpose.int` are not Core ATen |
| `reshape` | `reshape` | `aten::view` | contiguous data assumed; `aten::reshape` decomposes to `view` |
| `expand` | `expand` | `aten::expand` | |
| `relu` | `relu` | `aten::relu` | |
| `reduce_sum` | `sum` | `aten::sum.dim_IntList` | empty `axes` = all dims |
| `greater` | `gt` | `aten::gt.Tensor` | scalar operands are 0-d tensors here |
| `less` | `lt` | `aten::lt.Tensor` | |
| `equal` | `equal-to` | `aten::eq.Tensor` | |
| `and` | `logical-and` | `aten::logical_and` | |
| `or` | `logical-or` | `aten::logical_or` | |
| `not` | `logical-not` | `aten::logical_not` | |
| `where` | `where` | `aten::where.self` | |
| `print` | `trace-print` | **none** | an effect; PyTorch has no ATen print op |
| `loop` | (host loops) | **none** | control flow; the nearest PyTorch constructs are higher-order ops such as `while_loop` / `scan`, not ATen operators |
| parameters (constants) | — | **none as an op** | graph constants (`get_attr` / lifted constants in an exported program) |

The emitted graphs carry the ATen name on every node that has one.

The names follow the Core ATen IR operator set as published for PyTorch 2.x.
They were not cross-checked against the operator subset migrated into
`third_party/native-torch/`, which `design/native-torch-migration.md` governs.
Checking them is a separate task.

### 4.18 What the tests check

Every traced result is compared with the same computation written
independently in plain Scheme over nested lists.

**`test-trace.scm` (49):**

- S-Expr structure and round trip;
- DCE;
- single `if` with graph breaks:
  - cache hit on the same branch value;
  - continuation re-entry on a new value, tracing exactly one new segment;
  - per-segment pruning;
- multi-resume;
- `bool`-based `if` in a loop under the three modes;
- assumptions, the pluggable executor, inlining of nested traced functions,
  eager mode, errors.

**`test-trace-macro.scm` (40):**

- macro `if` in a loop: one graph, 7n nodes, and the node-count table;
- `cond`, `when`, `unless`, nested `if`, joins of numbers and lists;
- errors with `file:line`;
- the fallback through a foreign macro with `bool`;
- closure and module-variable guards;
- output capture and `trace-print`;
- retrace-on-miss: same tree shape, cache hits, no divergence.

**`test-trace-forms.scm` (46):**

- `and`, `or`, and with nested or, or with nested and, `logical-not` of `and`;
- a non-bool operand of `and` is an error with the location;
- `case` with numeric datums, `=>`, `else`, `else =>`;
- `case` with no `else` is an error;
- `case` inside a loop (static count, 8 joins, no break);
- `do` with a static count and a symbolic step (unrolled);
- `do` with a symbolic exit test: correct trip counts, `loop-breaks`, cache hits;
- named `let` with a symbolic exit test;
- named `let` around `do` around `cond =>` (one graph);
- `cond =>` receiving the symbolic condition;
- `cond =>` around `case` around `if` of `and`;
- R7RS semantics of every rewritten form on concrete values;
- `trace-opaque`;
- strict mode:
  - a plain helper is reported, and its wrong result shown;
  - the same helper with `define-traced` is silent and correct;
  - ops, internal define, named `let` and `map` over a covered lambda are
    silent;
  - `map` over a plain procedure reports that procedure, once;
  - off by default;
  - can be enabled with the `strict-tracing` parameter;
- host loops:
  - a 100-step `do` (limit 64) matches plain Scheme, emits one `loop` node,
    and iterations 2 to 100 match the cached body;
  - the body graph is parsed once per text, and the second call is one
    cached graph;
  - a 10-step `do` stays unrolled;
  - a named `let` with `#:max-unroll 8`;
  - bodies that change per iteration give one run each;
  - `#:max-unroll #f` always unrolls;
- `trace-pure` silences strict mode for that procedure only;
- portability: no module except `compat.scm` uses prompts, void ports,
  procedure properties, hash tables, `syntax-case`, `syntax-source`,
  `syntax-local-binding`, `datum->syntax`, `eval-when`, `catch` or `throw`.

**`test-trace-grad.scm` (33):**

- finite-difference gradient checks:
  - `add` (plus a broadcast bias), `sub`, `mul`, `div`, `neg`, `matmul`,
    `relu`;
  - `sum`: all axes, axis 0, axis 1 with keepdims;
  - `where`, `transpose`, `reshape`, `expand`;
  - comparison and logical ops carry no gradient;
- composites: traced `if` (both sides), `case` in a loop (3 branches), MLP
  loss;
- double where:
  - `where(x > 0, 1/x, 0)` at `x = 0` gives a finite, exact gradient;
  - `1/x` in a false `if` arm gives 0, not NaN;
- DCE keeps only what the backward pass needs;
- ATen attributes on every node;
- `grad` across a graph break is refused;
- XOR:
  - loss below 0.05;
  - traced once, then 1000 cache hits;
  - one graph per step.

Measured numbers for the `bool` loop test (6 inputs), and the macro version:

| Mode | traces | breaks | merges | segments (nodes each) |
|---|---|---|---|---|
| breaks only | 4 | 6 | 0 | 10 (2–4 nodes) |
| multi-resume, budget 64 | 1 | 0 | 7 | 1 (49 nodes) |
| multi-resume, budget 8 | 3 | 3 | 3 | 6 (2–9 nodes) |
| macro `if` (no `bool`) | 1 | 0 | 0 (3 `branch-merges`) | 1 (21 nodes) |

## 5. Limitations

### Surface rewriting

- **Uncovered core `if`s.** Only standard forms written literally in a traced
  body are rewritten. Macros defined elsewhere (left untouched) and procedures
  not defined with `define-traced` see a symbolic tensor as true in a core
  `if`. Use `bool` there. Strict mode reports the procedures; it cannot report
  foreign macro uses, which are left as written.
- **`quasiquote` is not walked**, so conditionals inside `unquote` are core
  `if`s.
- **Both arms run while tracing**, including Scheme-level effects in them.
  - Statement forms (`when`, `unless`, one-armed `if`) on a symbolic test keep
    both arms' nodes, so a `trace-print` inside one prints whichever way the
    condition goes.
  - The same holds for `do` result expressions.
- **`and` / `or` on a symbolic condition** require bool operands and become
  logical ops: `(and c x)` with a non-bool `x` is an error, not R7RS's "x or
  #f".
- **`case` on a symbolic key:**
  - only number datums (or booleans for a bool key) can match, compared with
    `=`;
  - other datums never match;
  - a `case` without `else` cannot produce a value.
- **Joins are value-level.** Every tensor-valued leaf of an arm result is
  selected with `where`. There is no CSE across arms. Two different Scheme
  numbers from the arms become a rank-0 tensor.
- **Data-dependent loops cost one extra iteration per break.** Detecting
  re-entry traces part of an iteration, which is dropped. If a graph break
  happened inside the dropped part, its nodes stay in that segment (dead, and
  removed by DCE, but `print` nodes among them would remain).
- **Bodies are walked before expansion.** Keywords are matched by binding
  (`syntax-rules` literals), so a keyword shadowed by a local binding is
  correctly not rewritten. But a form produced *later* by the expansion of a
  local `define-syntax` inside the body is not walked.

### Guards

- **Guards cover the body's direct free variables only.**
  - State read inside called procedures or through mutable objects is not
    guarded, except pairs, vectors and strings held directly in a guarded
    variable, which are snapshotted.
  - Records, including ctensors, and procedures compare by identity.
- **Calls inside a trace.** A traced function called while another is traced
  is inlined, and its own guards are not added to the caller's key.
- **Portability.** Without the compat guard hook (§6.2) there are no automatic
  guards. Pass `#:guards` explicitly.

### Effects

- **Only output to `current-output-port` is detected.** Other ports, `set!`
  and mutation are not.
- **`trace-print` on a merged path** prints on both sides at run time.
- **Retrace divergence.** A divergence restarts the call, so `print` nodes in
  the already-run prefix can fire twice in that call.

### Continuations

- **Multi-resume grows exponentially** (§4.9). Prefer the macro.
- **Stored continuations assume purity.** Re-entering one assumes the code
  between `bool` points depends only on the cache key. `#:retrace-on-miss? #t`
  avoids re-entering old continuations.
- **Unrewindable continuations.** On Guile, a `bool` reached through a C frame
  cannot be resumed.

### Loops

- **Tail calls only.** A host loop must continue only by a tail call to its
  `self`, keep the dtype and shape of its tensor variables, and take no graph
  break in the body. Each violation is an error with the loop's location.
- **Every iteration is traced at trace time.** Probing runs the Scheme body of
  every iteration once, to follow the static control and compare bodies.
  Cached calls afterwards run no Scheme code.
- **Effects in the body.** A `print` inside a host-loop body prints every
  iteration at run time, as it should. But DCE treats a `loop` node as pure, so
  a loop whose outputs are unused is dropped together with its prints.
- **Unbounded body cache.** The interpreter's body cache keeps every distinct
  body text for the life of the process.
- **Executors must implement `loop`.** Any executor other than the reference
  interpreter has to implement the `loop` node.

### Autodiff

- **No graph breaks before `grad`.** Gradients are taken within one segment
  (§4.16). `grad` must be called while tracing; `value-and-grad` provides that.
- **Loss and ops.** The loss must have one element. There are no gradients
  through `loop` nodes (host loops) or the comparison and logical ops.
- **Only `where` gets the double-where treatment**, and only for gradients
  flowing through it. A user op with an infinite derivative that is not under
  a `where` still produces `inf` / `NaN`, as in any framework.
- **Higher-order derivatives** (`grad` of a gradient) are untested.

### General

- **Cache misses are slow.** A miss rebuilds full paths, and partial graphs
  export every defined value.
- **Results.** A traced function returns tensors, or lists or vectors of them.
  - Scheme numbers and booleans in the result become constant tensors.
  - Other data (symbols, strings) is returned as traced.
  - Multi-resume (continuation) joins only single-tensor results.
- **Static shapes only**, as in S-Expr v0.
- **Small op set:**
  - `matmul` is 2-D only;
  - broadcasting follows NumPy rules;
  - Scheme reals stand in for float32;
  - `NaN` and `inf` are not representable in the emitted text;
  - the op table is the prototype's own.
- **Conditions must be one-element `bool` tensors.** A `bool` on a numeric
  tensor always breaks.
- **Not thread-safe.**
- **No native execution.** Wiring to `libsonicboom.so` would mean an executor
  that hands each segment's text to the native S-Expr runtime.

## 6. Porting guide

The prototype is meant to be borrowed. Everything a port must re-implement is
the interface of `compat.scm`. `loop.scm`, `autodiff.scm` and the rest use only
compat, R7RS and SRFI 1/9 (plus the mechanical items of §6.3); the
compat-boundary test in `test-trace-forms.scm` covers them. The rest is either portable R7RS or mechanical
translation (§6.3).

### 6.1 Run-time interface

| compat | Contract | Guile | Port to |
|---|---|---|---|
| `(%make-prompt-tag name)` | a fresh tag | `make-prompt-tag` | SRFI 226 / Racket: `make-continuation-prompt-tag` |
| `(%call-with-prompt tag thunk handler)` | run `thunk`; on abort, call `(handler k v …)` with the composable continuation up to this prompt; `k` must be re-enterable more than once | `call-with-prompt` | SRFI 226 / Racket: `call-with-continuation-prompt` with a handler; shift/reset: see below |
| `(%abort-to-prompt tag v …)` | capture up to the prompt and abort to its handler; returns what `k` is later called with | `abort-to-prompt` | SRFI 226 / Racket: `(call-with-composable-continuation (λ (k) (abort-current-continuation tag k v …)) tag)` |
| `(%raise-signal tag d …)` / `(%catch-signal tag thunk handler)` | non-local exit by tag; the handler runs after unwinding | `throw` / `catch` | R7RS `raise` of a tagged record + `guard` |
| `(call-with-output-captured thunk)` | `(values result text)` of what `thunk` wrote to `current-output-port` | string port + `parameterize` | the same in R7RS (`open-output-string`, `parameterize`) |
| `(call-with-output-suppressed thunk)` | run `thunk`, discarding its output | void port | a string port that is dropped |
| `(trace-warn msg)` | one line on `current-error-port` | `display` | R7RS |
| `(procedure-tag! p key v)`, `(procedure-tag p key)` | attach data to a procedure object | procedure properties | a weak `eq?` table (SRFI 124/125, R6RS `make-weak-eq-hashtable`); a plain `eq?` table if leaking is acceptable |
| `(procedure-label p)` | a name for messages | `procedure-name` | the implementation's procedure-name facility, or `p` itself |
| `make-table`, `table-ref`, `table-set!`, `table-count`, `table->alist` | `equal?`-keyed mutable table | Guile hash tables | SRFI 69 / SRFI 125 / R6RS hashtables |
| `%structural-procedures`, `%higher-order-procedures` | procedure objects that only move values (strict mode) | core plus SRFI-1 `map` / `for-each` | the implementation's own procedure objects |
| `%unbound-guard`, `(%global-guard module name)` | guard support | module lookup | whatever §6.2's guard hook needs |

**Delimited control requirements.**

- `k` is a *composable* (delimited) continuation that is called several times:
  - on a graph-break resume;
  - on a later cache miss;
  - twice per multi-resume.
- The tracer calls `k` inside a new `%call-with-prompt`, and an abort during
  `(k v)` must reach that new prompt.
- Dynamic bindings (`parameterize`) made outside the prompt must come from the
  resumption site; those made between prompt and abort must be restored. Both
  SRFI 226 and Racket give exactly this.

**On top of shift/reset only.** For example `reset`/`shift` from an
implementation's partial-continuation library, or a library over native
one-shot continuations.

- Emulate a single prompt tag (the tracer uses only `%break-tag`) by tagging
  the results of `reset`:
  - normal return → `(value . v)`;
  - abort → `(abort k v …)`.
- `%call-with-prompt` dispatches on the tag.
- A resumed `k` that aborts again returns an `abort` record from its own
  implicit `reset`. The wrapper handed to the handler must then re-abort to the
  current `reset`, composing the two continuations. This is the standard
  encoding of prompts with abort in terms of `shift`/`reset`.

**Do not build this on full `call/cc`.** Every capture would copy the whole
stack up to the top level, rather than only the user function's frames between
the prompt and `bool`, and re-entry would need `dynamic-wind`-level care.

### 6.2 Expansion-time hooks

The walker in `macro.scm` is pure `syntax-rules`. It calls exactly three
macros from compat; in Guile they use `syntax-case`.

| Hook | Guile | Portable fallback | What is lost |
|---|---|---|---|
| `(%site form)` | `syntax-source` → `'((filename …) (line …) (column …))` | `(syntax-rules () ((_ f) 'f))`: the quoted form | `file:line:col` in messages; two textually identical conditionals in one function then share a loop-detection site, so a re-entry could be reported falsely and fall back to a break (still correct) |
| `(%maybe-call call original walked)` | `syntax-local-binding` on the head: variable → `(call . walked)`, otherwise `original` | `(syntax-rules () ((_ c o (h . a)) (c h . a)))`: always a call | foreign macro uses inside a traced body break; wrap them in `(trace-opaque …)`. Implementations that can ask whether an identifier is bound as syntax (for example Racket's `syntax-local-value`) can implement the Guile behaviour |
| `(%free-variable-guards formals body …)` | `syntax-local-binding` + call-time module lookup | `(syntax-rules () ((_ . _) '()))` | automatic guards; pass `#:guards` explicitly |

### 6.3 Mechanical translations

These are not mechanisms, only library spelling:

- **Modules.** `define-module` / `#:use-module` → `define-library` / `import`.
- **Keyword arguments.** `define*` with `#:key` / `#:optional`, and option
  keywords like `#:strict?`, become an option alist or plain optional
  arguments. In `traced-lambda*` the options are passed through to
  `trace-function` unchanged, so only that procedure's signature changes.
- **Pattern matching.** `(ice-9 match)` → Wright/Shinn `match`, which is
  available for most implementations.
- **SRFIs and small helpers.** SRFI 1 (`fold`, `iota`, `any`, `every`,
  `filter-map`, …), SRFI 9 records (R7RS `define-record-type`), SRFI 28
  `format` (`~a` / `~s`), SRFI 13/130 string procedures (`string-join`,
  `string-contains`), and `assq-ref` / `assoc-ref` / `acons` (two-line
  helpers).
- **Tests.** SRFI 64 tests; `current-source-location` in one test is Guile's.
