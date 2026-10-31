# SonicBoom Quality Gate

One entry point for the checks run after a large development task and for
GitHub Actions. Local runs and CI call the same scripts; there is no second
CI-only check implementation.

```sh
tools/quality/run.sh            # default CPU gate
tools/quality/run.sh ci         # same checks, CI environment
tools/quality/run.sh static
tools/quality/run.sh sanitizers
tools/quality/run.sh tests
tools/quality/run.sh concurrency # optional TSan
tools/quality/run.sh memory      # optional Valgrind
tools/quality/run.sh cuda        # optional Compute Sanitizer
tools/quality/run.sh all         # default + optional checks
```

The scripts resolve the repository from their own location, so they can be
invoked from any working directory:

```sh
cd /tmp
/path/to/SonicBoom/tools/quality/run.sh
```

## Default CPU gate

`default` and `ci` run, in this order:

| Check | What it runs |
|---|---|
| `tests` | Private CPU configure/build (Release, CUDA off) + the complete CTest suite. |
| `sanitizers` | Private ASan+UBSan build; full CTest suite with leak detection off, then LeakSanitizer over the native C++ suites. |
| `static` | GCC 13 `-Wall -Wextra` (warnings in SonicBoom files fail the check), GCC `-fanalyzer`, Clang Static Analyzer, and `clang-tidy`, all driven by the real `compile_commands.json`. |

`sanitizers` is one private build with `SONICBOOM_SANITIZE=address;undefined`
because that is the combination the project supports. TSan is a separate build
and a separate mode.

The default gate fails if a required tool is missing, a build fails, a test
fails, a sanitizer reports an error, or an analyzer reports a diagnostic in
SonicBoom-owned code. It does not skip tests to obtain a green result.

## Optional checks

| Mode | Tool | Notes |
|---|---|---|
| `concurrency` | ThreadSanitizer | Private TSan build. `native-torch`, MLIR/LLVM, and libguile are uninstrumented; see limits below. |
| `memory` | Valgrind Memcheck | Focused native suites; complementary to ASan. |
| `cuda` | CUDA Compute Sanitizer | Requires a GPU, `compute-sanitizer`, and an existing CUDA-enabled build supplied with `SONICBOOM_QUALITY_CUDA_BUILD`. It is never part of `ci`. |

Optional checks report `SKIP` with the missing prerequisite and do not fail the
gate. They never report `PASS` without running the tool.

## PASS / FAIL / SKIP

- `PASS` — the command ran and produced no diagnostics/errors.
- `FAIL` — the command ran and found a problem, or a required prerequisite is
  missing. The gate exits non-zero.
- `SKIP` — an optional check could not run. The reason is printed and stored in
  `results.tsv`; it is not counted as a pass.

Each run writes:

```text
.quality/logs/<timestamp>-<mode>/
├── environment.txt     # tool versions, git revision, build configuration, disk
├── commands.log        # exact configure/build/ctest commands
├── results.tsv         # STATUS<TAB>CHECK<TAB>DETAIL
├── tests-*.log
├── sanitizers-*.log
└── static*.log
```

Build trees live under `.quality/build/` and are never an existing developer
build directory (`build/`, `build-asan/`, ...). Both `.quality/` and the CI
cache directories are gitignored.

## Requirements

Ubuntu 24.04 / Debian:

```sh
sudo apt-get install -y \
  g++-13 gcc-13 cmake ninja-build ccache \
  guile-3.0 guile-3.0-dev autoconf automake libtool pkg-config \
  clang-19 clang-tidy-19 valgrind
```

- **Clang 19 or newer is required** for the Clang Static Analyzer and
  clang-tidy passes. Clang 18's frontend cannot parse libstdc++ 13's
  `<expected>`: the header is gated on `__cpp_concepts >= 202002L`, while
  clang 18 reports `201907L`, and `core/` uses `std::expected` throughout.
  Clang 19 reports the required value. The gate pins both tools to the GCC
  installation of the formal compiler (`g++-13`), so run with the `-19`
  package names above.
- GCC's `-fanalyzer` runs one translation unit per process and can use
  several GiB of memory on MLIR-heavy files. `SONICBOOM_QUALITY_ANALYZER_JOBS`
  (default `min(nproc, 2)`) bounds concurrency.

- **MLIR/LLVM** is required by CMake. The gate uses an existing install at
  `SONICBOOM_MLIR_PREFIX` (default `build/llvm-mlir-install`) and never
  downloads LLVM. Build it from the in-tree submodule once:

  ```sh
  git submodule update --init --depth 1 third_party/llvm-project
  tools/mlir/build-mlir.sh
  ```

  The four native-torch submodules are also needed:

  ```sh
  git submodule update --init --depth 1 --jobs 4 -- \
    third_party/native-torch/third_party/cpuinfo \
    third_party/native-torch/third_party/sleef \
    third_party/native-torch/third_party/FP16 \
    third_party/native-torch/third_party/fmt
  ```

- **SonicCross (`guild`)** is a required build-time dependency. Install
  v0.0.2 as described in the top-level `README.md`, or set
  `SONICCROSS_GUILD=/path/to/guild`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SONICBOOM_QUALITY_DIR` | `.quality` | Root for gate artifacts. |
| `SONICBOOM_QUALITY_BUILD_ROOT` | `.quality/build` | Private build trees. |
| `SONICBOOM_QUALITY_LOG_ROOT` | `.quality/logs` | Run logs. |
| `SONICBOOM_QUALITY_JOBS` | `nproc` | Parallel build jobs. |
| `SONICBOOM_QUALITY_ANALYZER_JOBS` | `min(nproc, 2)` | Parallel analyzer jobs (GCC `-fanalyzer` is memory-hungry). |
| `SONICBOOM_QUALITY_CC`, `..._CXX` | `gcc-13`, `g++-13` | Compilers. |
| `SONICBOOM_MLIR_PREFIX` | `build/llvm-mlir-install` | MLIR install prefix. |
| `SONICCROSS_GUILD` | `guild` on PATH | SonicCross executable. |
| `SONICBOOM_PRIVATE_TESTS_DIR` | empty | Optional private test suites. |
| `SONICBOOM_QUALITY_CUDA_BUILD` | empty | Existing CUDA build for `cuda` mode. |
| `SONICBOOM_QUALITY_CCACHE` | `1` | Set to `0` to disable ccache. |

## CI

`.github/workflows/quality.yml` runs `tools/quality/run.sh ci` on
`ubuntu-24.04` for pushes to `master` and for pull requests. It is CPU-only:
GitHub-hosted runners have no GPU, and GPU checks are not faked.

- MLIR is built from the in-tree `third_party/llvm-project` submodule when the
  cache misses; the install is cached by the pinned submodule revision.
- All submodule checkouts use `--depth 1`.
- SonicCross v0.0.2 is cached after its first build.
- ccache is cached with a bounded 1 GiB budget.
- Logs are uploaded as an artifact when the gate fails.
- The static check currently fails on the pre-existing baseline recorded below.
  CI therefore fails at that step until the baseline is cleaned up; the
  failure is real and must not be turned into a skip or a non-blocking step.

GitHub Actions usage is not unlimited. Public repositories currently do not
consume paid minutes; private repositories draw on the account's included
minutes and storage and can be billed. See the official documentation rather
than relying on a number in this file:

- <https://docs.github.com/en/billing/concepts/product-billing/github-actions>
- <https://docs.github.com/en/actions/reference/limits>

## Scope and known limitations

- Sanitizer instrumentation covers SonicBoom-owned targets only
  (`core`, `core/compiler`, `capi`, tests). `third_party/native-torch`,
  MLIR/LLVM, and libguile are not instrumented. ASan's allocator interceptors
  still see the whole process heap; TSan does not see accesses entirely inside
  uninstrumented code.
- LeakSanitizer is run over native C++ suites only. Guile tests run under
  ASan/UBSan with leak detection off because libguile is uninstrumented and
  guardian finalization is not guaranteed at process exit.
- `compute-sanitizer racecheck` finds shared-memory hazards; it is not a
  general global-memory race detector.
- TSan test processes run under `setarch -R` (ASLR disabled) on kernels whose
  address-space randomization exceeds TSan's shadow-memory layout; without it
  TSan aborts with `unexpected memory mapping` before the test starts.
- The GCC analyzer runs with `--param=analyzer-bb-explosion-factor=1
  --param=analyzer-max-enodes-per-program-point=2`. With GCC 13's defaults its
  state exceeded 20 GiB on `core/compiler/lib/BoomDialect.cpp`; with these
  parameters one TU still needs several GiB and several minutes, so the pass
  is serialized by default.
- Static analysis covers SonicBoom-owned translation units (`core`, `capi`,
  and, for the warnings pass, `tests`). Diagnostics in native-torch or MLIR
  headers are logged but are not SonicBoom's to triage.
- The CUDA mode uses an existing CUDA build tree; it does not build one and is
  not part of the default or CI gate.
- No performance benchmark is run by this gate.

## Adding a check

1. Add `checks/<name>.sh` and register the mode in `run.sh`.
2. Record `PASS`/`FAIL`/`SKIP` through `sb_record`.
3. Use a private build directory (never a developer's `build/`).
4. Update this README and `.github/workflows/quality.yml` in the same change.

## Implementation status

Every mode below was executed on this host on 2026-10-31 (Ubuntu 24.04,
GCC 13.3, Clang/LLVM 19.1.1, Valgrind 3.22, CUDA 13.4 compute-sanitizer,
RTX 3050). Results come from `.quality/logs/` of that date.

| Mode | Result | Evidence |
|---|---|---|
| `tests` | PASS | `100% tests passed, 0 tests failed out of 59` (CPU-only Release). |
| `sanitizers` | PASS | ASan+UBSan: all 59 tests; LSan over the native C++ suites. |
| `concurrency` | PASS | TSan: seven native suites and the Guile single-flight test, no report. |
| `memory` | PASS | Valgrind Memcheck: six native suites, no definite errors or leaks. |
| `cuda` | **FAIL** | memcheck/initcheck/synccheck pass on both suites; racecheck reports two shared-memory hazards in `rmsnorm_bwd_dx_kernel`, and the test's own `dX <= 1e-5` check fails under racecheck. This is a real finding in kernel code, not a gate defect, and is deliberately not suppressed. |
| `static` | **FAIL** | All four tools ran; the current tree is not clean. Counts are recorded below. |

`cuda` reports `SKIP` when `SONICBOOM_QUALITY_CUDA_BUILD` is not supplied; the
FAIL above used an existing developer CUDA build. The mode never builds a CUDA
tree, so CI never runs it.

### Static-analysis baseline (2026-10-31)

These are pre-existing diagnostics in the current tree. They are not
suppressed, baselined, or filtered: the static check reports `FAIL` until they
are triaged. Counts over the SonicBoom-owned translation units listed in the
real `compile_commands.json`:

| Pass | Files with diagnostics | Diagnostics | TUs that could not be parsed |
|---|---|---|---|
| GCC `-Wall -Wextra` | 14 | 55 | 0 |
| GCC `-fanalyzer` | 11 | 27 | 0 |
| Clang Static Analyzer | 25 | 111 | 7 |
| clang-tidy | 45 | 433 | 6 |

Notes on the baseline:

- The GCC `-fanalyzer` reports are dominated by library-internal false
  positives (`__cur` from libstdc++ string iterators, `__location` from
  source-location support, `__dest` from `std::copy`) plus one real
  `-Woverflow` that also appears in the warnings pass.
- The Clang analyzer and clang-tidy parse failures are toolchain
  incompatibilities, not analyzer findings: libstdc++ 13's fixed-width
  `std::float16_t`, `namespace boom = ...` alias redefinition, recursive JSON
  value types used through `std::vector` before completion, and `ids.h`
  relying on a transitive `<compare>` include. The last one also produces 98
  of the clang-tidy diagnostics.
- The warnings pass is mechanical debt (unused parameters/variables,
  `-Wswitch` over native-torch's `c10::DeviceType`, missing field
  initializers, sign comparisons).
- None of this is a runtime, compiler, IR, or C ABI change; fixing it is a
  separate code-cleanup task.

### Not verified here

- The complete `run.sh all` invocation in a single process: each mode was run
  and recorded separately.
- The GitHub Actions workflow has not executed on GitHub from this host; only
  the local command it invokes (`run.sh ci`) was exercised.

Planned work is the static-baseline cleanup above. Broader coverage (for
example instrumented native-torch/MLIR builds) is a separate decision, not an
unfinished item.
