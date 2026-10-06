# SonicBoom + Guile AI Framework — Local Progress Report

Mission: a minimal real end-to-end Guile AI framework on SonicBoom — define/compose
models in Guile, execute ops through native C++ backends, train with real
forward/autograd/backprop/loss/optimizer, export model+params, and load/run from
standalone C++.

This is a local, uncommitted record. Each step records what was built and the
actual test result (never an assumption).

---

## Step 1.1 — Unified C++ execution entry — DONE

**What was built**

- `core/include/sonicboom/runtime.h` — `sonicboom::Model` facade. `compile` takes
  an S-Expr v0.1 graph (text or parsed `sx::Document`) and runs the full existing
  pipeline (adapt_graph → CPU resource snapshot → analytical cost model →
  `plan_execution` → region partitioning → backend compilation → `RuntimeExecutor`);
  `run(inputs)` executes over raw little-endian buffers. One error type
  (`ModelError{stage, message}`) collapses parse/plan/resource/weight/compile/execute
  failures. No native-torch/MLIR/ATen/c10 type is exposed.
- `core/src/planner/runtime.cpp` — implementation. Reuses `route_op` (Softmax →
  NativeTorch, else MLIR), `slice_region` for per-MLIR-region sub-documents, one
  `NativeTorchBackend` for all native nodes, a `CpuBackend` per MLIR region
  (whole `Document` for the homogeneous `WholeGraph` path), wired through the
  executor's tag map + per-task backends.
- `tests/core/test_runtime.cpp` — integration test through the single public API.
- `tests/core/CMakeLists.txt` — added `test_runtime` to the planner test list.

**Verified (actually executed)**

- `test_runtime` passes: mixed softmax→relu, homogeneous relu, two separate MLIR
  regions, and error paths (unknown op → stage `plan`; byte-count mismatch →
  stage `execute`; malformed text → stage `parse`) all behave correctly.
- Full test suite: **34/34 pass** (33 pre-existing + `test_runtime`).

**Notes / decisions (not assumptions)**

- Backends are owned by `Model` (`std::vector<std::unique_ptr<Backend>>`) with raw
  pointers handed to the `RuntimeExecutor`; the executor outlives nothing the
  model does not also own, so `run` is safe after move.
- External-weight sidecars are loaded once via `sx::load_external_weights` and
  passed to every `CpuBackend::compile`; the native-torch path currently has no
  parameter/constant inputs (no NT nodes carry weights in v0), consistent with
  the existing co-execution tests.

## Step 1.2 — Model export / persistence — DONE

**What was built**

- `core/include/sonicboom/sx/serialize.h` + `core/src/sx/serialize.cpp` —
  `sx::serialize_document(Document) -> std::expected<std::string, Error>`, the
  inverse of `parse_document`. Losslessly prints every v0.1 form (version, name,
  opsets, inputs, outputs, `:external`/`:values` parameters, nodes with all eight
  attribute typed-value forms, optional domain/version). Float literals are
  formatted (shortest round-trip + explicit `.0`) so the lexer re-reads them as
  floats, never ints.
- `core/include/sonicboom/runtime.h` + `runtime.cpp` — two persistence entry points:
  - `sonicboom::export_model(doc, params, dir)`: writes `<dir>/model.sx` +
    `<dir>/weights.bin`. Parameters present in `params` are rewritten to
    `:external` references into `weights.bin` (byte count validated against the
    declared shape/dtype); others keep their inline/external data. Trained values
    replace inline `:values` initial values.
  - `sonicboom::Model::load(dir, graph_file = "model.sx")`: reads the text,
    parses, compiles with `base_dir = dir` so sidecars resolve — the standalone
    C++ load path (no Guile/Python).
- `tests/core/test_persistence.cpp` + `tests/core/CMakeLists.txt` entry.

**Verified (actually executed)**

- `test_persistence` passes: lossless idempotent round-trip across every form;
  export → `Model::load` → run produces `x + trained_w` correctly (proving the
  trained parameter value actually replaced the inline initial); byte-count
  mismatch and missing-file error paths behave.
- Full test suite: **35/35 pass** (34 prior + `test_persistence`).

**Notes / decisions (not assumptions)**

- The on-disk artifact is the *same* form the ONNX importer emits (S-Expr text +
  flat binary sidecar), so export and import share one `load_external_weights`
  reader — no second format introduced.
- A parameter referenced as a node input is baked by the MLIR lowering as a
  dense constant (`:external` via `dense_from_bytes`, `:values` via
  `dense_constant`), so `add(x, w)` executes correctly through the same path.

## Step 2.2 — Layer/Model composition API — DONE

**What was built**

- `bindings/guile/scheme/sonicboom/ffi.scm` — `(sonicboom ffi)`, the only path
  Guile reaches the native backend. Lazy-loads `libsonicboom.so` via
  `(system foreign)`, wraps `sb_parse`/`sb_compile`/`sb_output_info`/
  `sb_bind_input`/`sb_execute`/`sb_retrieve_output`/`sb_error_*`, and exposes
  `run-s-expr` (compile a graph text + run one input → output bytes or an error
  message). Handles out-params with pointer slots, always frees document/
  executable/error handles, and surfaces C ABI failures as `(values #f msg)`
  (parse/compile/binding/runtime errors are never swallowed).
- `bindings/guile/scheme/sonicboom/nn.scm` — `(sonicboom nn)`, a minimal
  PyTorch/Keras-like layer API. `linear` (dense → native `gemm(x, Wᵀ, b)` with
  `transB`, bias broadcast along the trailing axis) and `relu` (→ native `relu`),
  each a named bundle of parameters plus a forward step that emits S-Expr v0.1
  node text. `sequential` composes layers into a `<model>`; `model-forward`
  assembles one topologically-ordered graph (unique SSA value + parameter names)
  and runs it through `run-s-expr`, returning the output `<tensor>`. No tensor
  arithmetic happens in Scheme — every forward is native `gemm`/`relu`.
- `bindings/guile/tests/test-nn.scm` — SRFI-64 suite (12 assertions) over real
  native execution: linear structure/params, single-sample and batch `x·Wᵀ+b`
  numerics, relu clamping standalone and inside a composed model, 3-layer
  composition, per-layer parameter uniqueness, and error paths (non-float32
  input; malformed graph text surfaces a message).
- `bindings/guile/CMakeLists.txt` — added `guile-nn` ctest with
  `LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>` so the FFI finds the .so.

**Verified (actually executed)**

- `ctest --test-dir build -R guile` → **2/2 pass** (guile-tensor, guile-nn).
- `guile ... test-nn.scm` → **12/12 pass**; hand-computed references matched
  (e.g. linear 2→1 weight [2,3] bias 10 on [4,5] → 33.0; 3-layer ones-model on
  [2,3] → 20.0; relu clamps a composed [-20,-20] to 0).
- Full C++ regression suite → **35/35 pass** (unchanged).

**Notes / decisions (not assumptions)**

- Three real bugs found and fixed during this step, each confirmed by a
  hand-computed reference:
  1. The Guile `sizeof` FFI primitive only accepts pointer types, not scalar
     symbols — hardcoded the 8-byte `uint64`/`size_t` slots (64-bit target).
  2. `(sonicboom nn)` emitted one extra `)` for the `relu` node and one for the
     document tail — balanced by hand-counting against the frozen grammar.
  3. `run-s-expr`'s original `unless`-based error handling did not stop
     execution after a failure (it masked the parse error); replaced with a
     `throw`/`catch` that frees handles and returns the message.
- Value names in node inputs/outputs are strings (quoted), matching the frozen
  spec; parameter SSA names are `p<i>.<key>` and value names `v<i>`, all
  generated by the walker so composition never collides.
- Forward is recompiled per call in v0 (fine for a single forward; the training
  loop in 2.3/3.1 will need a caching strategy or an autograd execution model —
  flagged, not yet designed).

## Step 2.1 — Guile tensor/parameter API — DONE

**What was built**

- `bindings/guile/scheme/sonicboom/tensor.scm` — the `(sonicboom tensor)` module.
  A `<tensor>` is a SRFI-9 record holding a frozen dtype + static shape + a raw
  little-endian bytevector (plus `requires-grad`/`grad` slots); a `<parameter>` is
  a tensor with `requires-grad` set. Pure Scheme data — no C++ object backs a
  tensor, and no arithmetic runs in Scheme loops. Exports: `make-tensor`,
  `%make-tensor`, `bytevector->tensor`, `tensor->bytevector`, `tensor-dtype/shape/
  data/numel`, `tensor-ref`, `tensor-set!`, `tensor->list`, `list->tensor`,
  `tensor-copy`, `zeros`, `ones`, `rand`, `randn`, `parameter`, `parameter?`,
  `tensor-requires-grad?`, `tensor-grad`/`set-tensor-grad!`, `parameter-grad`/
  `parameter-grad-set!`, `dtype-size`, `valid-dtype?`, `valid-shape?`.
  Dtype vocabulary matches the frozen S-Expr v0.1 type system (10 names).
- `bindings/guile/tests/test-tensor.scm` — SRFI-64 suite (19 assertions):
  construction/access round-trips on exactly-representable float32 values,
  int64/float64 dtypes, in-place mutation, zeros/ones, randn shape+variation,
  bytevector round-trip, copy independence, parameter trainability + grad
  get/set, and error paths (bad dtype / shape mismatch / negative dim).
- `bindings/guile/CMakeLists.txt` + root `CMakeLists.txt` — `find_program(GUILE_EXECUTABLE
  guile)` + `add_test(guile-tensor ...)` under `enable_testing()`, so `ctest -R guile`
  runs the suite.

**Verified (actually executed)**

- `guile -L bindings/guile/scheme bindings/guile/tests/test-tensor.scm` → **19/19 pass**.
- `ctest --test-dir build -R guile` → **1/1 pass**.
- Full C++ regression suite → **35/35 pass** (unchanged; no C++ touched).

**Notes / decisions (not assumptions)**

- Two test-script bugs fixed, not module bugs: (1) the test file needed `(srfi
  srfi-1)` for `every`; (2) the grad round-trip used `0.1/0.2/0.3`, which are not
  exactly representable in float32 — switched to `0.5/0.25/-1.0`.
- Guile warns that `(sonicboom tensor)`'s `parameter?` overrides the core SRFI-39
  `parameter?` binding. Benign for the framework; noted, not renamed (the
  PyTorch-like name is intentional).
- `rand`/`randn` seed `default-random-source` once at module load, so samples vary
  across runs.

---

## Acceptance review — lightweight, 2026-10-06 (read-only)

**Workspace state (verified, not assumed)**

- Branch `native-torch/refactor`, HEAD `b8f0b34`. All Step 1.1–2.3 work is still
  uncommitted; the modified tracked files and untracked new files (autograd core,
  `runtime.h`/`runtime.cpp`, `serialize.*`, C ABI training surface, 4 Guile
  modules + 3 test scripts) are all present and intact. Nothing was cleaned,
  reset, committed, pushed, or branched.

**Tests actually run this session + results**

| Command | Result |
| --- | --- |
| `ctest --test-dir build --output-on-failure` | **3/3 pass** (guile-tensor, guile-nn, guile-train) |
| `./build/tests/core/test_autograd` | **exit 0** — "autograd: 3/3 checks passed" |
| `./build/tests/core/test_persistence` | **exit 0** — "test_persistence OK" |
| `./build/tests/core/test_runtime` | **exit 0** — "test_runtime OK" |
| sweep of all 36 `build/tests/core/test_*` executables | **PASS=36 FAIL=0** |

`test_autograd` prints the concrete evidence: finite-difference gradient checks
pass for `w` (2 el), `b` (1 el), `x` (6 el), `relu-x` (4 el); XOR initial loss
`0.428775` → final `1.20792e-15`. The gradient checks are central differences
(`(f(x+ε)−f(x−ε))/2ε`, ε=1e-2, tolerance 5e-3), **not** hardcoded expected
values.

**Step status**

- **Step 1.1** C++ `Model` facade — **PASS** (`test_runtime`).
- **Step 1.2** export / persistence — **PASS** (`test_persistence`: export →
  `Model::load` → run yields `x + trained_w` = `{11,22,33}`).
- **Step 2.1** Guile tensor/parameter — **PASS** (guile-tensor, 19 assertions).
- **Step 2.2** layer/model composition — **PASS** (guile-nn, 12 assertions).
- **Step 2.3** autograd / backprop / loss / optimizer / training — **PASS**
  (`test_autograd` + guile-train).

**Call chain confirmed (Guile → C ABI → C++)**

`train.scm` → `ffi.scm` `proc` (C ABI `sb_linear`/`sb_relu`/`sb_mse_loss`/
`sb_backward`/`sb_sgd_step`/`sb_grad`) → `capi.cpp` → `sonicboom::linear`… →
`nt::find_operator` boxed dispatch into native-torch aten ops. Tensors cross the
boundary as little-endian float32 bytevectors (`sb_tensor_from_f32`/
`sb_tensor_bytes`); `train.scm` performs no element-wise tensor arithmetic — only
byte encode/decode and loop counters. Verified non-zero grads via `sb_grad`:
`gw=(-0.45 -0.8375)`, `gb=(-0.8)`.

**Gaps (report only, not implemented)**

1. **Full closed loop is NOT wired.** Training (2.3) yields `nt::Tensor` params
   (Guile can read them back only as bytevectors); export/load (1.2) is
   C++-only (`export_model` + `Model::load` over `sx::Document` + a params map).
   There is no C ABI export/load surface and no bridge from a trained autograd
   tape's params to the `model.sx`+`weights.bin` artifact. Each half is tested
   independently, but "Guile train → export → standalone C++ load inference" has
   not been run end to end.
2. **Two parallel model representations.** `nn.scm`'s `<model>`/`<layer>` (2.2)
   lowers forward to the S-Expr graph path; `train.scm` (2.3) trains a
   hardcoded MLP over the autograd tape. The 2.2 layer API is not yet connected
   to 2.3 training.
3. **C++ tests are not in ctest.** `tests/core/CMakeLists.txt` uses
   `add_executable` only; the 36 C++ suites run as standalone binaries, while
   only the 3 Guile suites are registered via `add_test`. Not a functional
   blocker, but the C++ suites are easy to forget to run.

**Recommended next step (not started):** expose export/load through the C ABI and
add a bridge from trained `nt::Tensor` params (named) to `export_model`, then add
one Guile test that trains XOR, exports, and verifies a standalone C++ load
reproduces the trained output — closing the loop in the acceptance criteria.

---

## Acceptance report — full verification run, 2026-10-06

**Workspace (recorded at start of run)**

- Branch `native-torch/refactor`, HEAD `b8f0b34`.
- Working tree unchanged by this run: 6 modified tracked files (`CMakeLists.txt`,
  `capi/include/sonicboom/capi.h`, `capi/src/capi.cpp`, `core/CMakeLists.txt`,
  `core/include/sonicboom/tensor.h`, `tests/core/CMakeLists.txt`) plus untracked
  new files (autograd core, `runtime.h`/`runtime.cpp`, `serialize.*`, C ABI
  training surface, 4 Guile modules + 3 test scripts, this progress doc, 3 `*.log`).
- Nothing switched, committed, pushed, cleaned, reset, or deleted.

**Environment**

- Build current: `cmake --build build` → exit 0, all targets up-to-date.
- CUDA is enabled and present: `test_cuda` executed real GPU kernels
  (`to_device` round-trip, `aten::add.Tensor`, `aten::mul.Tensor`) and the
  expected `cudnn_convolution` excluded-op error path.

**Test commands + actual results (this run)**

| Group | Command | Result |
| --- | --- | --- |
| runtime | `./build/tests/core/test_runtime` | exit 0 — "test_runtime OK" |
| persistence | `./build/tests/core/test_persistence` | exit 0 — "test_persistence OK" |
| autograd | `./build/tests/core/test_autograd` | exit 0 — "autograd: 3/3 checks passed" |
| cuda | `./build/tests/core/test_cuda` | exit 0 — real GPU kernels executed |
| all C++ | sweep of 36 `build/tests/core/test_*` | **PASS=36 FAIL=0** |
| Guile | `ctest --test-dir build --output-on-failure` | **3/3 pass** (guile-tensor, guile-nn, guile-train) |

`test_autograd` detail: finite-difference gradient checks pass (w/b/x/relu-x);
XOR loss `0.428775 → 1.20792e-15`.

**Step status**

- 1.1 C++ `Model` facade — **PASS** (`test_runtime`).
- 1.2 export / persistence / `Model::load` — **PASS** (`test_persistence`: external
  parameter restored; inference reflects the trained value `{11,22,33}`).
- 2.1 Guile tensor/parameter — **PASS** (`guile-tensor`).
- 2.2 layer/model composition — **PASS** (`guile-nn`).
- 2.3 autograd / backward / loss / optimizer / training — **PASS**
  (`test_autograd` + `guile-train`).

No regressions; no BLOCKED or FAIL entries.

**Confirmed by execution (this run) vs source-only**

- Executed: `Model` run; serialize → load → run with external weights; real
  finite-difference gradients; SGD loss reduction; Guile forward/training
  crossing the C ABI; CUDA kernels.
- Source-only (not executed as an integration test): the full
  Guile-train → export → standalone-C++-load loop (see gap), and specifically
  `gemm` with `:external` weights (only `add`+external and inline-`:values`
  `gemm` are separately proven).

**Remaining integration gap (not a regression)**

The three APIs remain separate: `nn.scm` (S-Expr composition/forward),
`train.scm` (autograd training over native tensors), and export/load
(`export_model` / `Model::load`, C++ only). "Guile model → train → export →
standalone C++ inference" is not wired. Recommended next task: add
`sb_export_model` (parse S-Expr text + named trained tensors → `export_model`),
name the training parameters, and add one e2e test (Guile trains XOR, exports;
a standalone C++ `Model::load` reproduces ≈ `{0,1,1,0}`).

---

## Step 2.4 — Guile → train → export → standalone C++ inference — DONE

This closes the "Remaining integration gap" above: the full mission loop is now
wired and proven end-to-end.

**What was built**

- `capi/include/sonicboom/capi.h` + `capi/src/capi.cpp` — `sb_export_model`:
  parse S-Expr v0.1 text + `names[]`/`params[]` (named trained `sb_tensor`s) →
  `sonicboom::export_model` → `<dir>/model.sx` + `<dir>/weights.bin`. Maps the
  exporter's `ModelError` (`stage == "export"`) to `SB_ERR_INTERNAL`; parse
  failures to `SB_ERR_PARSE`. Byte-count/shape mismatch is rejected inside
  `export_model`.
- `bindings/guile/scheme/sonicboom/ffi.scm` — registered `sb_export_model`
  `(list '* size_t '* '* size_t '* '*)`.
- `bindings/guile/scheme/sonicboom/nn.scm` — `<layer>` gained a `kind` field
  (`'linear`/`'relu`); added `model-graph-text` (returns the graph text); and
  **unified parameter naming**: `build-graph`'s parameter SSA prefix changed from
  `p<idx>.` to `<layer-name>.`, so graph parameter names now match
  `sequential`'s `model-parameters` keys (both `<layer-name>.<key>`). This is the
  seam that lets `export_model` match trained tensors to graph parameters.
- `bindings/guile/scheme/sonicboom/model.scm` (new) — the unified trainable model:
  `make-model` (builds one native `sb_tensor` per parameter — the single
  authoritative store — seeded from the nn layers, with optional deterministic
  `#:init-values`), `model-forward-native` (autograd forward over the layer list,
  freeing intermediates), `model-train!` (plain-SGD loop), and `model-export!`
  (graph text + native tensors → `sb_export_model`). Native tensors are the only
  mutable parameter store; no independently-mutable Scheme copy.
- `bindings/guile/scheme/sonicboom/train.scm` — exported `list->f32-bv` /
  `f32-bv->list` (needed to write the golden `expected.bin`).
- `tests/core/test_e2e_load.cpp` (new) — the independent inference half: loads an
  exported artifact directory, asserts the graph has a `gemm` node whose trained
  weights are `:external "weights.bin"`, runs the XOR inputs through
  `Model::load`/`run`, and compares to `expected.bin` within 1e-2.
- `bindings/guile/tests/test-e2e-export.scm` (new) — the training half: builds a
  2→4→1 relu MLP (deterministic `#:init-values`), trains on XOR, writes the
  trained model's forward outputs to `<dir>/expected.bin`, then exports.
- `bindings/guile/tests/run-e2e.sh` (new) — orchestrates Guile exporter → C++
  loader against one scratch dir.
- CMake: `test_e2e_load` added to `tests/core/CMakeLists.txt`; `guile-cpp-e2e`
  ctest added to `bindings/guile/CMakeLists.txt`.

**Verified (actually executed)**

- Build: `cmake --build build` → exit 0 (`capi.cpp` rebuilt, `libsonicboom.so`
  relinked, `test_e2e_load` linked).
- `ctest --test-dir build --output-on-failure` → **4/4 Guile tests pass**
  (guile-tensor, guile-nn, guile-train, **guile-cpp-e2e**).
- `guile-cpp-e2e` runs the full loop: Guile trains XOR (`model-train!` lowers
  loss below 0.05), exports the artifact, and the standalone C++
  `test_e2e_load` reloads it — asserting the gemm node + `:external` weights and
  reproducing the trained predictions within 1e-2.
- Full C++ sweep: **36/36 `build/tests/core/test_*` pass** (`test_e2e_load`
  standalone prints its `<export-dir>` usage, exit 2, as designed; it is
  exercised through `guile-cpp-e2e`).

**Confirmed by execution (this run) vs source-only**

- Executed: the entire `Guile model → train → export → standalone C++ load/run`
  loop; `gemm` nodes with `:external` trained weights (the loader asserts both
  the gemm node and the `:external "weights.bin"` reference, then compares
  numeric outputs against the trained model).
- The two engines (native-torch autograd for training, S-Expr → MLIR for
  deployment) agree on structure, parameter naming, shapes, and math semantics:
  the MLIR path reproduces the native-torch predictions from the exported
  weights.

**Step status (cumulative)**

- 2.4 Guile → train → export → standalone C++ inference — **PASS**
  (`guile-cpp-e2e` + `test_e2e_load` via `run-e2e.sh`).

No regressions; no BLOCKED or FAIL entries. The mission loop is complete.

---

## Step 2.5 — C++ suites registered in CTest — DONE

Closes the last item of the "Remaining integration gap" list above ("C++ tests
are not in ctest").

**What was built**

- `tests/core/CMakeLists.txt` — added a `sb_add_cpp_test(name)` helper
  (`add_test` + `LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>` +
  `$<TARGET_FILE:name>`) and registered every C++ suite with it. `test_e2e_load`
  stays built but unregistered (it needs an `<artifact-dir>` arg; driven by
  `guile-cpp-e2e`); `test_cuda` is registered only in the CUDA build.
- `CMakeLists.txt` — moved `enable_testing()` to the top so it is active before
  `add_subdirectory(tests/core)`, letting the C++ suites register under the same
  `ctest` invocation as the Guile suites.

**Verified (actually executed)**

- `ctest -N` → **40 tests** (36 C++ incl. `test_cuda`, + 4 Guile).
- `ctest -j$(nproc) --output-on-failure` → **40/40 pass** in a single run
  (previously the 36 C++ suites ran only as a manual standalone sweep).

**Step status (cumulative)**

- 2.5 C++ suites in CTest — **PASS**.

All three items of the prior "Remaining integration gap" are now closed.
