# SonicBoom S-Expr v0 — Specification

> Status: FROZEN (v0.1)
> Inputs: `models/resnet18.onnx` design-input report; the 7 resolved Open
> Questions (recorded in §15).

---

## 1. Scope

S-Expr is SonicBoom's **upper, language-agnostic, serializable model / Graph IR**.

```
ONNX ─────┐
          ├──→ S-Expr ──→ MLIR ──→ Runtime
Scheme ───┘
```

S-Expr describes **what a model is and what its data dependencies are**. It
deliberately contains no execution semantics: no dispatch, no lowering, no
memory placement, no kernel selection, no scheduling, no JIT, no runtime state.

v0 expresses ResNet18-class ONNX DAGs (straight-line dataflow) and is
constructible directly from Scheme. It is a **text** format: human-readable,
git-diff-friendly, parseable by a C++ parser, generatable by Scheme, and
independent of both Guile and MLIR.

Two v0 invariants govern the format:

- **Every value carries an explicit type.** No shape/type inference is required
  to read a value's type.
- **Nodes are topologically ordered.** A value's producer precedes its
  consumers, enabling single-pass parsing and straight lowering.

---

## 2. Design Principles

1. **Minimal.** Only concepts grounded in the real model (ResNet18). No
   speculative abstractions.
2. **Language-agnostic.** S-expression *syntax* only. No Scheme semantics, no
   Guile, no MLIR concepts.
3. **Self-describing and unambiguous.** Fixed keyword symbols for structure;
   string literals for all user-supplied names; explicit type tags for every
   scalar/vector value.
4. **SSA, value-named dataflow.** Every tensor is a named value; nodes
   reference values by name.
5. **Explicit typing.** A value's type lives at its single definition site
   (input, parameter, or node output); nothing is inferred.
6. **Topological node order.** Producer-before-consumer, enforced (§14).
7. **Additive extensibility.** Future features add new forms; they never change
   or remove v0 forms (see §12).
8. **Data separable from structure.** Weight/constant payload may live in an
   external flat binary blob; the S-Expr file carries names, types, and
   references.

---

## 3. Syntax

### 3.1 Atoms

| Atom | Meaning |
|---|---|
| `symbol` | fixed-vocabulary keyword: structure heads, op names, dtype names, attr type tags |
| `"string"` | user-supplied name, path, or string value |
| `<int>` | integer literal (dimensions, offsets, integer attribute values) |
| `<float>` | float literal (float attribute values, inline float data) |
| `#t` / `#f` | boolean literal |

Rule: **all user-supplied names are strings** (`"input"`, `"conv1.weight"`).
Symbols are reserved for the fixed format vocabulary.

### 3.2 Grammar (BNF)

```text
document    ::= (sonicboom-s-expr (version <int> <int>) <graph>)
graph       ::= (graph (name <string>)
                       (opset <string> <int>)*
                       (inputs <input>*)
                       (outputs <output>*)
                       (parameters <parameter>*)
                       (nodes <node>*))

input       ::= (input <string> <type>)
output      ::= (output <string>)
parameter   ::= (parameter <string> <type> <data>)
node        ::= (node <op-name>
                       (inputs <string>*)
                       (outputs <value-def>*)
                       (attrs <attr>*)
                       (domain <string>)?        ; default "default"
                       (version <int>)?)         ; default: graph opset version
value-def   ::= (<string> <type>)

attr        ::= (<name> <typed-value>)
typed-value ::= (int <int>)
              | (float <float>)
              | (string <string>)
              | (bool #t|#f)
              | (ints <int>*)
              | (floats <float>*)
              | (strings <string>*)
              | (bools #t|#f*)

type        ::= (tensor <dtype> <shape>)
dtype       ::= float32 | float16 | bfloat16 | float64
              | int8 | uint8 | int16 | int32 | int64 | bool
shape       ::= (shape <dim>*)
dim         ::= <int>                              ; v0: static, non-negative

data        ::= (data :external <string> <int> <int>)   ; file, byte-offset, byte-length
              | (data :values <atom>*)                  ; inline literal list (dtype implied)
```

### 3.3 Notes

- `op-name` is a lowercase canonical symbol (`conv`, `relu`, `add`, `gemm`, …).
  The op-name vocabulary is **not** part of this format spec; it belongs to a
  separate operator registry (§8). The importer maps source op names (e.g. ONNX
  `Conv`) to canonical symbols.
- `value-def` is the only way a node introduces a value, and it **always carries
  a type**. There is no separate type-annotation section.
- `(opset "default" 20)` states the operator semantic version (§11). A node may
  override with `(domain …)` / `(version …)` for custom operators.
- `data` supports exactly two layouts in v0: a flat external binary sidecar
  (`:external`) and inline typed literals (`:values`). No container format
  (safetensors, etc.) is introduced.

---

## 4. Semantic Model

The model is a **named SSA dataflow graph** over explicitly-typed tensor values.

- A **Value** is a named slot (`"input"`, `"getitem"`, `"add_3"`) carrying a
  **type** (`(tensor float32 (shape 1 64 112 112))`).
- Values are **defined exactly once** (single static assignment). They are
  *introduced* by three constructs, and **every introduction site carries the
  value's type**:
  1. a graph **input** — `(input "input" (tensor …))`,
  2. a **parameter** — `(parameter "conv1.weight" (tensor …) (data …))`,
  3. a node **output** — `("getitem" (tensor …))` inside a node's `(outputs …)`.
- Values are **referenced** by name in node `inputs` and in graph `outputs`.
  Fan-out is legal: a value may be referenced by many consumers.

### 4.1 Value vs Tensor

- **Value** is the *dataflow slot* (an SSA name + its type).
- **Tensor** is the *type* of a value. In v0 every value is a tensor value;
  `(tensor …)` is the only value type.
- Future non-tensor values (tuples, lists, optional, subgraph handles) would add
  new `type` forms (§12), not a new "Value" concept.

### 4.2 Parameter / Constant vs Value

- A **parameter** is a *named, typed tensor with data*. It is one of the three
  ways a value is introduced.
- v0 **unifies** weights and constants into a single `parameter` concept (§7).
  There is no separate `constant` form and no `role` tag.

### 4.3 How a Node references and produces Values

- `inputs` is an ordered list of value names (SSA references).
- `outputs` is an ordered list of `(<name> <type>)` pairs (SSA definitions). The
  node introduces each output name **and its type** into scope.
- Because every value's type is at its definition, a reader never needs
  inference: it registers output types as it walks nodes in order.

### 4.4 Input / Output

- `(input <string> <type>)` — declares a graph input and its type (required; it
  is not otherwise recoverable).
- `(output <string>)` — selects a single existing value as the graph result.
  It carries no type: the referenced value's type is its definition-site type.

### 4.5 Attributes vs constant-tensor inputs (important)

These are **different and both present** in ResNet18:

- **Attribute** — compile-time scalar/vector metadata that is part of the
  node's identity, inline in the node: `(attrs (group (int 1)) (strides (ints 2 2)))`.
  It is not a value and does not participate in dataflow.
- **Constant-tensor input** — a `parameter` value that flows into the node as a
  normal input, e.g. ReduceMean's `axes` (`val_187`) and Reshape's `shape`
  (`val_191`) in ONNX opset 20. It *is* a value in the SSA graph.

S-Expr preserves this distinction exactly: attributes are inline `(attrs …)`;
constant tensors are `parameter` values referenced in `inputs`.

---

## 5. Type System

```text
type   ::= (tensor <dtype> <shape>)
dtype  ::= float32 | float16 | bfloat16 | float64
         | int8 | uint8 | int16 | int32 | int64 | bool
shape  ::= (shape <dim>*)          ; rank = number of dims
dim    ::= <int>                   ; v0: static, non-negative
```

- **dtype** is a closed enum of exactly ten names (above). ResNet18 uses
  `float32` and `int64`.
- **rank** is implicit in the number of dims (not stored separately).
- **dimension** in v0 is a static non-negative integer. No dynamic, symbolic, or
  negative dims.

### 5.1 Dynamic / symbolic shape — excluded from v0

v0 does **not** support dynamic or symbolic shapes. `dim` is an integer, full
stop. A future version would add a tagged dim form (e.g. `?` or `(sym …)`)
without altering v0; a v0 reader must reject any non-integer `dim` (fail-safe).

---

## 6. Graph / SSA Model

- Exactly one `graph` per document in v0 (a document == a model == a graph).
- Value namespace is **flat and graph-scoped**: input names, parameter names,
  and node-output names share one namespace; no shadowing; single definition.
- **Node order is topological.** In `(nodes …)`, every node that defines a value
  appears before any node that references that value. Listing order is therefore
  meaningful and enforced: a parser may do a single forward pass, registering
  each node's output types before any later node consumes them.
- A value is live from its single definition point to the graph output (or a
  consumer). No scoping, no blocks, no dominance requirement beyond SSA.

---

## 7. Tensor / Parameter / Constant

### 7.1 Decision (v0): unify weights and constants into `parameter`

```
parameter ::= (parameter <string> <type> <data>)
```

Rationale:

- ONNX itself does **not** distinguish weights from constants — both are
  `initializer`s. ResNet18's `fc.weight` (learned) and `val_187` (axes) are
  identical in structure: named typed tensor + data.
- One concept is sufficient. A separate `constant` form or a `role` tag would be
  speculative and is **not** in v0.
- If a weight/constant distinction is ever needed, it must be a **additive
  extension** (e.g. a future `(parameter … (role weight))` tag), never a change
  to the existing `parameter` form.

### 7.2 How a parameter is referenced

By name, in the same flat value namespace, in a node's `(inputs …)`. Example:
`(node reduce_mean (inputs "relu_16" "val_187") …)`.

### 7.3 Data storage

`data` supports exactly two layouts in v0:

| Form | Use |
|---|---|
| `(data :external <file> <offset> <len>)` | large weights in a flat binary sidecar (46.7 MB for ResNet18). Keeps the text file small and diff-friendly. |
| `(data :values <atom>*)` | small inline constants (`val_187` = `-1 -2`). |

`offset`/`length` are byte offsets/lengths into the sidecar file. The sidecar is
a flat concatenation; no container metadata lives inside the S-Expr. (The blob's
own byte layout is agreed between writer and reader; the S-Expr records only
file, offset, length.)

---

## 8. Node / Operator

```
node ::= (node <op-name> (inputs <string>*) (outputs <value-def>*)
                (attrs <attr>*) (domain <string>)? (version <int>)?)
```

An **operator identity** is the triple `(domain, op-name, version)`:

- `op-name` — canonical lowercase symbol (`conv`, `gemm`, …). Its *meaning* is
  defined by an **operator registry**, which is a separate SonicBoom
  compiler/runtime component, **not part of this format**. The format only
  carries the name and does not copy ONNX's operator representation (no
  `OperatorSchema`, no attribute registry).
- `domain` — default `"default"`. Custom domains are a future extension point.
- `version` — default: the graph's `(opset <domain> <version>)`. This is the
  **operator semantic version** (§11).

S-Expr records *which operator, with which inputs/outputs/attributes*, and leaves
*what the operator computes* to the registry + lower layers (MLIR/runtime).

---

## 9. Attributes

```
attr        ::= (<name> <typed-value>)
typed-value ::= (int <int>) | (float <float>) | (string <string>) | (bool #t|#f)
              | (ints <int>*) | (floats <float>*) | (strings <string>*) | (bools #t|#f*)
```

- Explicit type tags remove ambiguity (`1` vs `1.0` vs `"1"` vs `(ints 1)`).
- The four scalar types + their vectors are exactly what ResNet18 uses:
  `group`, `transB`, `keepdims`, `ceil_mode`, `allowzero` (int);
  `pads`, `strides`, `dilations`, `kernel_shape` (ints);
  `auto_pad` (string); `alpha`, `beta` (float).
- Attribute values are **not** tensors and are **not** values in the graph.

---

## 10. Input / Output

- `(input <string> <type>)` — declares a graph input and its type. Type is
  **required**.
- `(output <string>)` — names an existing value as the graph result. No type is
  attached here (it is the referenced value's definition-site type).

ResNet18: one input `"input"` (float32 [1,3,224,224]); one output `"output"`
(float32 [1,1000]).

---

## 11. Versioning

Three independent versions:

1. **S-Expr format version** — `(version <major> <minor>)`. v0 = `(version 0 1)`.
   - A reader MUST reject a document whose `major` it does not support.
   - A reader MUST accept a document whose `major` matches and `minor` is
     ≤ its own (minor increments are additive).
2. **Operator semantic version** — `(opset <domain> <version>)`. Defines the
   meaning of operators in that domain. v0 example uses `("default", 20)`.
3. **Model version / provenance** — intentionally **not** represented (ONNX's
   `model_version`, `producer`, `ir_version` are serialization provenance, not
   model semantics; drop them).

---

## 12. Extensibility

The format is a set of **named forms**. v0 freezes the forms in §3. Future
features are **additive** — new forms, new dtype names, new dim forms, new data
layouts — and never mutate v0 forms. Concretely, the anticipated extensions are:

| Future need | Extension (new form, non-breaking) |
|---|---|
| dynamic / symbolic shape | new `dim` forms (`?`, `(sym …)`) |
| control flow (If/Loop/Scan) | a `graph`-valued value type + `if`/`loop` node forms |
| subgraph / functions | `(graph …)` referenced as a value or a `function` section |
| custom operator / domain | `(domain …)` / `(version …)` on a node; extra `(opset …)` |
| quantization | new dtype names (`quint8`, `qint8`, …) and/or a `(quant …)` type form |
| metadata | a top-level `(meta …)` section with arbitrary key/value |
| new data layout | new `data` tags (`:external-safetensors`, `:raw`, …) |
| weight/constant distinction | an additive `role` tag on `parameter` |

**Fail-safe rule:** a v0 reader that encounters an unknown form MUST report
"unsupported feature" and refuse, rather than silently mis-parse. This is what
makes v0 files forward-stable.

---

## 13. Complete ResNet18 Example

See `s-expr-v0-resnet18.example.sx` (full 49-node, 44-parameter serialization,
every value explicitly typed, topologically ordered). A schematic head:

```lisp
(sonicboom-s-expr
  (version 0 1)
  (graph
    (name "main")
    (opset "default" 20)
    (inputs
      (input "input" (tensor float32 (shape 1 3 224 224))))
    (outputs
      (output "output"))
    (parameters
      (parameter "conv1.weight" (tensor float32 (shape 64 3 7 7))
        (data :external "resnet18.weights.bin" 0 37632))
      (parameter "val_187" (tensor int64 (shape 2)) (data :values -1 -2)))
    (nodes
      (node conv (inputs "input" "conv1.weight" "conv1.weight_bias")
                 (outputs ("getitem" (tensor float32 (shape 1 64 112 112))))
                 (attrs (group (int 1)) (strides (ints 2 2)) (pads (ints 3 3 3 3))))
      ;; … remaining nodes, in topological order …
      )))
```

---

## 14. Validation Rules

A valid v0 document satisfies:

1. `(sonicboom-s-expr (version 0 1) …)` — exactly one, as the sole top-level form.
2. Exactly one `(graph …)`.
3. **Single definition**: no value name is introduced more than once (input,
   parameter, or node output).
4. **Name resolution**: every name in a node `inputs` and every `(output …)`
   refers to a value defined in the document (input, parameter, or node output);
   the def-use relation is acyclic.
5. **Explicit typing**: every graph `input`, every `parameter`, and every node
   `output` (`value-def`) carries a `(tensor <dtype> <shape>)`. No value lacks a
   type, and there is no annotation section that a parser must fall back on.
6. **Topological order**: for every node N at index i, each input value that is a
   node output must have its producing node at index < i. (Inputs that are graph
   inputs or parameters have no producer and are always valid.) A validator MUST
   reject a document that violates this order.
7. `data` layout is `:external` (with `file`/`offset`/`length`, all present) or
   `:values`. No other layout.
8. `dtype` ∈ the ten-name enum; every `dim` is a non-negative integer.
9. `(opset …)` domains referenced by nodes are declared (or `"default"`).
10. Attribute `typed-value` tags are only the eight forms in §9.
11. The document contains no unknown top-level/nested form (fail-safe, §12).

---

## 15. Resolved Questions (formerly Open)

All v0-scope questions are resolved and frozen:

1. **dtype** — fixed to the ten-name set in §5.
2. **Intermediate value type** — every value carries an explicit type at its
   definition site (§4); no inference, no `(types …)` section.
3. **External weight** — flat binary sidecar `(data :external <file> <offset> <length>)`;
   no container format.
4. **Operator registry** — a separate compiler/runtime component; not part of
   the S-Expr format. S-Expr carries only `domain`/`name`/`version`.
5. **Parameter / Constant** — unified into `parameter`; no `role`. Any future
   distinction is additive.
6. **Dynamic / symbolic shape** — not in v0; `dim` is a non-negative integer.
7. **Node ordering** — `nodes` must be topologically ordered.

No open issues remain that affect v0 semantics.

---

## Final Classification

### FROZEN (v0 must include, exactly as specified)

- Top-level form `(sonicboom-s-expr (version 0 1) (graph …))`.
- One graph; flat SSA value namespace; named-value dataflow.
- **Every value explicitly typed at its definition site** (input / parameter /
  node output); no `(types …)` section.
- `input` (typed) / `output` (name-only reference).
- `parameter` (unified weight+constant) with `:external` / `:values` data only.
- `node` with `(inputs …)` `(outputs (<name> <type>)…)` `(attrs …)`, optional
  `domain`/`version`.
- Attribute typed-value forms: `int/float/string/bool` + `ints/floats/strings/bools`.
- Type `(tensor <dtype> <shape>)`; static non-negative dims; ten-name dtype enum.
- **Topological node ordering**, enforced.
- `opset` as the operator semantic version.
- Fail-safe unknown-form rejection.

### NOT IN v0 (explicitly excluded)

- MLIR dialect / operation / pass; LLVM IR.
- CPU/GPU backend, memory placement, kernel selection, scheduling.
- Dynamic Roofline, JIT, runtime resource state.
- Control flow (If/Loop/Scan), subgraphs, functions.
- Quantization, sparse tensors, non-tensor value types (tuple/list/optional).
- Dynamic / symbolic dims.
- Python / ONNX runtime artifacts (`ir_version`, `producer`, `model_version`).
- Operator *semantics* registry (out of scope for the format itself).
- Weight/constant `role` distinction; container data formats (safetensors, …).

### FUTURE EXTENSION (reserved, not implemented)

- Dynamic/symbolic dims; control flow; subgraph; custom domain/op; quantization;
  metadata `(meta …)`; new data layouts; non-tensor types; a weight `role` tag.

### OPEN QUESTIONS

None. All v0-scope decisions are frozen (see §15).
