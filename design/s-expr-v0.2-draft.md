# SonicBoom S-Expr v0.2 — Draft

> Status: **DRAFT**, not frozen. Derived from the graphs the Guile tracing
> prototype (`bindings/guile/scheme/sonicboom/trace/`) actually emits.
> Base: [`s-expr-v0-spec.md`](s-expr-v0-spec.md) (v0.1, FROZEN). Everything in
> v0.1 stays valid; this draft lists only the increments.
> Checked by: `(sonicboom trace validate)` (§11).

## 0. Summary of increments

| # | Increment | Kind | Core / optional |
|---|---|---|---|
| 1 | Multi-output graphs: ordered outputs, pass-through outputs | semantics on v0.1 syntax | **core** |
| 2 | Tensor-list inputs: flat inputs named `<base>.<k>`, bound by name | naming convention + semantics | **core** (convention), optional (structure metadata) |
| 3 | `aten` node attribute: Core ATen dispatch key | attribute convention + validity rule | **core** to accept and check; dispatch by it is optional |
| 4 | Constant parameters: rank-0, integer and bool constants, value-count rule | tightened v0.1 rule | **core** |
| 5 | `loop` node: host loop over a nested body graph | new op with a nested document | **optional** |
| 6 | Side-effect nodes (`print`) and their ordering semantics | new op class | **optional** |
| 7 | dtype and shape rules: a typing rule per op, checked against declarations | new validity rule | **core** |
| 8 | Attribute type representation: per-attribute tags, booleans, empty lists, document strings | conventions on v0.1 attribute syntax | **core** |
| 9 | Op vocabulary with arities | new normative table | **core**, except `loop` and `print` |
| 10 | Version `(version 0 2)` | header | **core** |

The rules for **optional** features:

- A reader that does not implement an optional feature MUST reject a document
  that uses it, with an "unsupported feature" error. This is the v0.1
  fail-safe rule, §12. It MUST NOT skip the node.
- A document uses `loop` or `print` iff it contains a node with that op. No
  separate declaration is needed, because the op names are reserved.

The prototype still writes `(version 0 1)` on every document, including ones
that use v0.2 increments; its output format is not changed by this draft. The
validator accepts both versions with the v0.2 rules. Once the draft is
adopted, a writer MUST declare `(version 0 2)` whenever it uses any increment
marked "new" above (items 3 and 5–7 and the vocabulary of 9).

---

## 1. Multi-output graphs (core)

**Syntax.** Unchanged from v0.1:

```text
(outputs (output <string>)*)
```

**Semantics (new).**

- **Positional.** The graph's results are the listed values *in order*. An
  executor returns them positionally. Result `i` is the value of the `i`-th
  `output`.
- **Defined and unique.** Each output names a value defined in the document:
  a node output, a graph input or a parameter. A name appears at most once in
  `outputs`.
- **Pass-through.** An output may name a graph **input**, in which case the
  value passes straight through. The prototype emits this when a traced
  function returns one of its arguments.
- **Host-side structure.** How results are grouped into lists or vectors on
  the host is not part of the graph. The tracer keeps a result template and
  fills it from the positional results.

**Minimal example** (two results, the second passed through):

```lisp
(sonicboom-s-expr
  (version 0 2)
  (graph
    (name "two-outputs")
    (inputs
      (input "arg0.0" (tensor float32 (shape 2)))
      (input "arg1" (tensor float32 (shape 2)))
      (input "arg0.1" (tensor float32 (shape 2))))
    (outputs
      (output "v0")
      (output "arg0.1"))
    (parameters)
    (nodes
      (node add (inputs "arg0.0" "arg1") (outputs ("v0" (tensor float32 (shape 2))))
            (attrs (aten (string "aten::add.Tensor")))))))
```

## 2. Tensor-list inputs (core convention)

**Syntax.** Unchanged: flat `(input <string> <type>)` entries.

**Naming convention (new).**

- A function argument that is a single tensor is the input `"arg<i>"`.
- An argument that is a list or vector of tensors becomes inputs
  `"arg<i>.<k>"`, with `k` = 0, 1, … in depth-first order of the tensors
  inside it. Non-tensor data inside the argument is static (part of the
  tracer's cache key) and does not appear in the graph.

**Semantics (new).**

- **Bind by name.** Inputs are bound **by name**. Their order in `(inputs …)`
  is not significant: the prototype lists them in first-use order, for
  example `arg0.0 arg1 arg0.1` above.
- **Unused inputs are omitted.** An input that no surviving node or output
  uses does not appear at all.
- **List reconstruction is host-side.** A host that passes a list rebuilds the
  names from the convention. Example: a training step `(params ms vs t x y)`
  with 18 parameter tensors uses inputs `arg0.0 … arg0.17`, `arg1.0 …`,
  `arg2.0 …`, `arg3`, `arg4`, `arg5`.

**Optional, not emitted yet.** An explicit `(input-group <string> <string>*)`
section naming the members of each list, for readers that want the structure
without parsing names. It is reserved and not used by the prototype.

**Minimal example:** the graph in §1. `arg0` was a two-element list.

## 3. The `aten` attribute (core to accept and check; dispatch optional)

**Syntax.** An ordinary v0.1 attribute:

```text
(aten (string "<namespace>::<op>[.<overload>]"))
```

**Semantics (new).**

- **The name.** It names the PyTorch Core ATen operator the node corresponds
  to, for example `"aten::add.Tensor"`, `"aten::mm"`, `"aten::where.self"`.
  It is the key a native executor MAY use to dispatch the node directly to a
  native-torch kernel. The registered schema is found by this name; the
  inputs are passed in order, and the remaining operands come from the node's
  other attributes.
- **op-name stays canonical.** The `op-name` remains the canonical identity;
  `aten` adds no semantics.
- **Validity rule.** A node whose op has an entry in the mapping table (§9)
  MUST carry `aten` equal to that entry. A node whose op has no entry (`loop`,
  `print`) MUST NOT carry one.
- **Readers.** A v0.1 reader that ignores the attribute still computes the
  same result.

Whether native-torch actually has a kernel for each name is recorded in
`design/guile-tracing-prototype.md`, §4.17.

**Minimal example:**

```lisp
(node mul (inputs "arg0" "c0") (outputs ("v1" (tensor float32 (shape 2))))
      (attrs (aten (string "aten::mul.Tensor"))))
```

## 4. Constant parameters (core)

**Syntax.** Unchanged v0.1 `parameter` with `(data :values <atom>*)`.

**Tightened rules (new).**

- **Value count.** The number of values is exactly the element count of the
  type, the product of the dims; it is 1 for rank 0 `(shape)`. Order is
  row-major.
- **Value kinds.** Values must fit the dtype:
  - float dtypes: reals, written with a decimal point or exponent (`2.0`, not
    `2`);
  - integer dtypes: exact integers;
  - `bool`: `#t` / `#f`.
- **Constants from the source.** The prototype uses constants for literals in
  the traced code, for example a rank-0 `0.0` for `(where c x 0.0)`. Two
  other kinds also appear: integer index tensors (`int64`, for
  `index_select` / `gather`) and bool masks (a causal mask).
- **Constants from the backward pass.** Gradient rules introduce zero blocks
  for the `index_add` and `scatter_add` bases and for `slice` padding. They
  are ordinary constants.
- **Graph-local.** A constant belongs to the graph that defines it. The same
  literal appearing twice gives two parameters.

**Minimal example:**

```lisp
(parameters
  (parameter "c0" (tensor int64 (shape 2)) (data :values 1 0))
  (parameter "c2" (tensor bool (shape 2 2)) (data :values #t #f #t #t))
  (parameter "c3" (tensor float32 (shape)) (data :values 0.0)))
```

## 5. `loop` node: host loop (optional)

**Syntax:**

```text
(node loop
  (inputs <state>… <invariant>…)
  (outputs (<name> <type>)…)              ; one per state input
  (attrs (body (string "<document>"))     ; a complete S-Expr document
         (trip_count (int <n>))           ; n >= 0
         (state_count (int <k>))))        ; 0 <= k <= number of inputs
```

**Semantics.**

- **Inputs.** The first `k` inputs are the loop state; the rest are loop
  invariants (values from outside the loop, such as weights).
- **The body.** `body` is a complete S-Expr document. Its inputs are named
  `"s0"…"s<k-1>"` (the state) and `"x0"…` (the invariants), typed exactly as
  the node's inputs. Its outputs are the next state, in order and with the
  same types.
- **Execution.** The executor runs the body `n` times, feeding each
  iteration's outputs back as the next iteration's `s` inputs. The node's
  outputs are the final state. With `n = 0`, the outputs are the inputs.
- **The body is fixed.** The body graph is the same for every iteration, so
  an executor parses or compiles it once.
- **Effects.** `print` nodes in the body run every iteration.
- **Static trip count only.** `trip_count` is fixed in the document. A loop
  whose trip count depends on data is not expressible; the prototype takes a
  graph break per iteration instead.
- **Run-length encoding.** Consecutive iterations whose bodies differ become
  separate `loop` nodes.

**Minimal example** (halve a vector three times; this is the prototype's
output, with the body text written by `write`):

```lisp
(sonicboom-s-expr
  (version 0 2)
  (graph
    (name "halve3")
    (inputs
      (input "arg0" (tensor float32 (shape 2))))
    (outputs
      (output "v14"))
    (parameters)
    (nodes
      (node loop (inputs "arg0") (outputs ("v14" (tensor float32 (shape 2))))
            (attrs (body (string "(sonicboom-s-expr (version 0 2) (graph (name \"loop-body\") (inputs (input \"s0\" (tensor float32 (shape 2)))) (outputs (output \"t1\")) (parameters (parameter \"p0\" (tensor float32 (shape)) (data :values 0.5))) (nodes (node mul (inputs \"s0\" \"p0\") (outputs (\"t1\" (tensor float32 (shape 2)))) (attrs (aten (string \"aten::mul.Tensor\")))))))"))
                   (trip_count (int 3))
                   (state_count (int 1)))))))
```

## 6. Side-effect nodes: `print` (optional)

**Syntax:**

```text
(node print (inputs <x>) (outputs (<y> <type-of-x>)) (attrs (label (string "<label>"))))
```

**Semantics.**

- **Value.** `y` is `x`, an identity. The effect is writing
  `<label>: <values>` to the executor's output channel.
- **Ordering.** Effectful nodes run in **document order** relative to each
  other. Relative to pure nodes, they run after their inputs are computed.
- **DCE.** An effectful node is live even if its output is unused, so dead
  code elimination MUST keep it.
- **Re-execution.** An executor re-executing a graph only to rebuild values
  (the prototype does this after a graph break or a cache miss) MUST suppress
  the effects, so each real run prints exactly once.
- **Both branch sides.** A `print` under a `where`-joined branch prints
  whichever way the condition goes, because both sides are computed.
- **No `aten`.** `print` has no ATen counterpart and carries no `aten`
  attribute.

**Minimal example:**

```lisp
(nodes
  (node mul (inputs "arg0" "c0") (outputs ("v1" (tensor float32 (shape 2))))
        (attrs (aten (string "aten::mul.Tensor"))))
  (node print (inputs "v1") (outputs ("v2" (tensor float32 (shape 2))))
        (attrs (label (string "dbl"))))
  (node add (inputs "arg0" "c3") (outputs ("v4" (tensor float32 (shape 2))))
        (attrs (aten (string "aten::add.Tensor")))))
```

## 7. dtype and shape rules (core)

v0.1 requires every value to be typed but leaves the meaning of ops to the
registry. v0.2 makes the typing rules normative for the vocabulary of §9. A
document is valid only if every node's declared output type equals the type
its rule derives from the input types and attributes.

| Ops | dtype rule | Shape rule |
|---|---|---|
| `add sub mul div` | inputs equal, numeric; output the same | NumPy broadcasting of the inputs |
| `greater less` | inputs equal, numeric; output `bool` | broadcasting |
| `equal` | inputs equal; output `bool` | broadcasting |
| `and or` / `not` | inputs `bool`; output `bool` | broadcasting / unchanged |
| `neg relu` | numeric; unchanged | unchanged |
| `exp log tanh sqrt gelu` | float; unchanged | unchanged |
| `where` | condition `bool`; branches equal; output = branch dtype | broadcast of all three |
| `matmul` | equal | `[m k] × [k n] → [m n]` |
| `bmm` | equal | `[b m k] × [b k n] → [b m n]` |
| `permute` | unchanged | `out[i] = in[dims[i]]`; `dims` a permutation |
| `reshape` | unchanged | `shape` attr, same element count |
| `expand` | unchanged | `shape` attr, which the input must broadcast to |
| `reduce_sum reduce_mean reduce_max` | unchanged | drop the `axes` (all axes if empty); keep them as 1 if `keepdims` = 1 |
| `index_select` | `x` any, `index` 1-D integer | `x` with `axis` replaced by `len(index)` |
| `index_add` | `base`, `source`, index 1-D integer | output = `base`; `source` has the `index_select` shape |
| `gather` | `index` integer, same rank as `x` | `index`'s shape |
| `scatter_add` | `index` and `source` same shape | output = `base` |
| `concat` | all equal | equal off `axis`; sizes summed on `axis` |
| `slice` | unchanged | `axis` extent `end − start`, `0 ≤ start ≤ end ≤ extent` |
| `print` | unchanged | unchanged |
| `loop` | §5 | §5 |

Constants obey the same rules as any value. A Scheme literal combined with an
`int64` tensor becomes an `int64` constant.

## 8. Attribute type representation (core)

v0.1 syntax: `(<name> (<tag> <value>…))`, with tags `int float string bool
ints floats strings bools`. v0.2 fixes the tag of each attribute the
vocabulary uses:

| Attribute | Tag | Used by | Notes |
|---|---|---|---|
| `axes` | `ints` | `reduce_*` | may be empty: `(axes (ints))` means all axes; negative values count from the end |
| `keepdims` | `int` | `reduce_*` | 0 or 1. Booleans of the vocabulary are written as `int` 0/1, as in ONNX |
| `shape` | `ints` | `reshape`, `expand` | |
| `dims` | `ints` | `permute` | a permutation of `0 … rank−1` |
| `axis` | `int` | `index_select index_add gather scatter_add concat slice` | |
| `start`, `end` | `int` | `slice` | step is 1 |
| `label` | `string` | `print` | |
| `aten` | `string` | every mapped op (§3) | |
| `body` | `string` | `loop` | a whole document, escaped by the writer's string syntax |
| `trip_count`, `state_count` | `int` | `loop` | |

Other rules:

- An empty list attribute is written with the tag and no values: `(ints)`.
- The attribute list of a node with no attributes is `(attrs)`.
- An attribute not listed for an op is an error, except `aten`.

**Minimal example:**

```lisp
(node reduce_sum (inputs "v1") (outputs ("v2" (tensor float32 (shape 2 1))))
      (attrs (axes (ints 1)) (keepdims (int 1)) (aten (string "aten::sum.dim_IntList"))))
```

## 9. Op vocabulary (core, except `loop` and `print`)

| op | arity | ATen (§3) |
|---|---|---|
| `add` `sub` `mul` `div` | 2 | `aten::add.Tensor` `aten::sub.Tensor` `aten::mul.Tensor` `aten::div.Tensor` |
| `neg` `relu` `gelu` | 1 | `aten::neg` `aten::relu` `aten::gelu` (tanh approximation) |
| `exp` `log` `tanh` `sqrt` | 1 | `aten::exp` `aten::log` `aten::tanh` `aten::sqrt` |
| `matmul` `bmm` | 2 | `aten::mm` `aten::bmm` |
| `permute` `reshape` `expand` | 1 | `aten::permute` `aten::view` `aten::expand` |
| `reduce_sum` `reduce_mean` `reduce_max` | 1 | `aten::sum.dim_IntList` `aten::mean.dim` `aten::amax` |
| `greater` `less` `equal` | 2 | `aten::gt.Tensor` `aten::lt.Tensor` `aten::eq.Tensor` |
| `and` `or` / `not` | 2 / 1 | `aten::logical_and` `aten::logical_or` / `aten::logical_not` |
| `where` | 3 | `aten::where.self` |
| `index_select` `gather` | 2 | `aten::index_select` `aten::gather` |
| `index_add` `scatter_add` | 3 | `aten::index_add` `aten::scatter_add` |
| `concat` | ≥ 1 | `aten::cat` |
| `slice` | 1 | `aten::slice.Tensor` |
| `print` *(optional)* | 1 | — |
| `loop` *(optional)* | ≥ state_count | — |

Each op produces exactly one output, except `loop`, which produces
`state_count` outputs.

The vocabulary is what the prototype emits. v0.1's statement that op meaning
belongs to a registry (§8 of v0.1) still holds. This table is the registry for
the v0.2 core, and other ops remain possible in other domains.

## 10. Version (core)

```text
(sonicboom-s-expr (version 0 2) (graph …))
```

- The minor increment is additive (v0.1 §11): a v0.2 reader accepts v0.1
  documents.
- A v0.1 reader must reject `(version 0 2)`, because the minor version is
  greater than its own.
- The draft's own transition rule is in §0: the prototype still writes
  `(0 1)`, and the validator accepts it.

## 11. Validation

`(sonicboom trace validate)` checks a document text against this draft:

- **Structure:** version, sections, form shapes.
- **Types:** dtype enum, static dims, every definition typed.
- **Constants:** the `:values` count and kinds.
- **Names:** single definition, topological order, outputs defined and unique.
- **Nodes:** known op, arity, attribute tags.
- **Typing:** the output type re-derived by the §7 rule equals the
  declaration.
- **`aten`:** present and equal to the §9 table entry, absent for `loop` and
  `print`.
- **`loop`:** the body is validated recursively, and its `s` / `x` inputs and
  outputs must match the node.

Errors carry `line L, column C` from the reader and a document path such as
`nodes[12] (node add)`, or `… body > nodes[0] …` inside a loop body.

`bindings/guile/tests/trace/test-trace-validate.scm` validates:

- every graph the other prototype test files execute;
- the examples of this draft;
- a set of deliberately broken documents.

## 12. Out of scope (unchanged from v0.1, or deferred)

- **Shapes.** Dynamic or symbolic shapes.
- **Control flow.** Data-dependent control flow inside a graph: the prototype
  resolves it with `where` joins or graph breaks between documents.
- **Calls.** Subgraph calls other than `loop`'s body.
- **Types.** Quantized types and non-tensor values.
- **Trace-tree structure.** How a trace tree's segments are chained (branch
  conditions, segment inputs exported by earlier segments) is host-side
  executor state, not part of a document. Each segment is a self-contained
  v0.2 document.
