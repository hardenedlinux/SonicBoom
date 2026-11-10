# SonicBoom

<img src="assets/sonic-boom-logo.webp" height="600">

**A native tensor runtime. No Python. One `.so`.**

On a desktop RTX 3050, SonicBoom edits code with Qwen2.5-7B at
**471 tok/s, 12.2× its own plain decode**. llama.cpp's fastest speculative
mode reaches 408 tok/s, and its output then differs from its own
non-speculative output on some turns. SonicBoom's output is identical, bit for
bit, with speculation on or off.

SonicBoom is written from the ground up in C++ and CUDA, exposed through a
stable C ABI, with GNU Guile as its first-class language. No Python
interpreter, pip or wheel tree is involved in inference.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg">
  <img alt="SonicBoom: fast, exact and lean. The stack runs from Guile, C or an OpenAI-compatible server through one libsonicboom.so to the runtime, the Boom IR compiler, SonicBoom's own kernels and the GPU. Each speculative step drafts tokens, verifies them in one pass with every row bit-identical to a plain decode step, and commits them unchanged; kernel choices are tuned at load and sealed." src="assets/architecture.svg" width="100%">
</picture>

## Numbers

**Qwen2.5-7B-Instruct Q4_K_M** on an RTX 3050 8 GB

| Workload | SonicBoom | llama.cpp | Ratio |
|---|---:|---:|---:|
| **Code editing, speculative decoding** | **471.1 tok/s** | 407.7 tok/s ¹ | **1.16×** |
| **Same output with speculation on and off** | **10 of 10 turns** | 8 of 10 turns ¹ | |
| Decode, speculation off, context 512 | 40.6 tok/s | 43.3 tok/s | 0.94× |
| Decode, speculation off, context 4K | 38.3 tok/s | 41.4 tok/s | 0.92× |
| Prefill, 512-token prompt | 1600 tok/s | 1594 tok/s | 1.00× |
| Prefill, 8K-token prompt | 1008 tok/s | 1425 tok/s | 0.71× |

Through the OpenAI-compatible server, the same kind of edit request streams at
523.7 tok/s with speculation on and 38.2 tok/s with it off, with
the same text.

Both runtimes run the same GGUF on the same machine: RTX 3050 8 GB (sm_86),
Ryzen 5 5500, llama.cpp `43fe9c64281` with `-ngl 99 -fa on`. Each figure is the
median of 3 process runs. Prefill is wall-clock time for the whole prompt.

¹ The code-editing workload is ten turns, each asking for an edit of a
100–300-line source file; generation is greedy, as long as the file.
llama.cpp runs `llama-server -ngl 99 -fa on -np 1 --spec-type ngram-mod`, its
fastest of the eight speculative modes on this workload. Its identity count
compares each speculative run with its non-speculative run, turn by turn.

Every row comes from the two commands under
[Reproduce the table](#reproduce-the-table). Model files, checksums and the
long-context tiers are in [docs/benchmark.md](docs/benchmark.md). Models under
2B parameters are no longer benchmarked, because they have little practical
use. Gemma 4 E4B, the DeepSeek-R1-Distill-Qwen family and the other Qwen2.5
sizes run as well.

## Reproducible

- **Same input, same output.** This holds with the same version, the same GPU
  and the same warm-up table.
- **Speculation doesn't change the result.** Turning speculative decoding on
  or off gives bit-identical token sequences. It is on by default.
- **Seeded sampling.** The same seed and configuration give the same output.

The warm-up table is measured on your own GPU when a model loads. You can
export it and import it elsewhere with `sb-tune`, which carries a run to
another machine with the same GPU. Batched execution is not covered yet.

## What's new in v0.0.6

- Speculative decoding runs through every entry point: the C ABI, Guile and
  the OpenAI-compatible server, which makes one decode call per reply and
  streams verified tokens as they are accepted.
- `tools/bench.sh --spec` reproduces the code-editing comparison against
  llama-server on your own GPU, and `--tune-export` / `--tune-import` carry the
  warm-up table between machines.
- Speculative decoding is on by default and verifies up to 64 drafted tokens
  per step; a long-match drafter targets code and document edits.
- Grammar- and JSON-constrained decoding computes only the LM-head rows the
  grammar allows.

## Quick start

Requirements: Linux x86-64, g++-13, CMake, Ninja, CUDA 12.0+, an NVIDIA GPU
(tested on sm_86), GNU Guile 3.0 with
[guile-json](https://github.com/aconchillo/guile-json) 4.7.3, and
[SonicCross](https://github.com/hardenedlinux/SonicCross) v0.0.2 at build
time. Full details are in [docs/build.md](docs/build.md).

```sh
git clone --recursive https://github.com/hardenedlinux/SonicBoom
cd SonicBoom
tools/mlir/build-mlir.sh                 # one-time: pinned LLVM/MLIR
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONICBOOM_USE_CUDA=ON
cmake --build build

mkdir -p models
curl -L -o models/qwen2.5-7b-instruct-1m-q4_k_m.gguf \
  https://huggingface.co/ngxson/Qwen2.5-7B-Instruct-1M-Q4_K_M-GGUF/resolve/main/qwen2.5-7b-instruct-1m-q4_k_m.gguf
LD_LIBRARY_PATH=build/core guile -L bindings/guile/scheme
```

```scheme
(use-modules (sonicboom inference))

(define model   (load-model "models/qwen2.5-7b-instruct-1m-q4_k_m.gguf"))
(define session (make-session model 4096))
(display (generate model session "The three laws of thermodynamics are" 64))
```

The same calls are available from C (`sb_llm_*` in
[`capi.h`](capi/include/sonicboom/capi.h)).

### Serve an OpenAI-compatible API

The server is a GNU Artanis app; its dependencies are listed in
[docs/serve-openai.md](docs/serve-openai.md#7-running).

```sh
LD_LIBRARY_PATH=build/core guile -L bindings/guile/scheme \
  bindings/guile/scheme/sonicboom/serve/app.scm \
  --model models/qwen2.5-7b-instruct-1m-q4_k_m.gguf --model-id qwen2.5-7b \
  --context-length 16384 --max-sessions 1 --port 8000
```

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "qwen2.5-7b", "stream": true,
       "messages": [{"role": "user", "content": "Write the opening paragraph of a mystery novel."}]}'
```

Any OpenAI client, or a writing tool that speaks the OpenAI API, can use
`http://127.0.0.1:8000/v1` as its base URL; no API key is checked. A
conversation reuses its shared prefix automatically, so asking for a revision
of a long chapter only prefills the new instruction. `--max-sessions 1` keeps
one KV cache on an 8 GB card; raise it on larger GPUs to serve several
conversations at once.

### Reproduce the table

```sh
LLAMA_BIN=/path/to/llama.cpp/build/bin tools/bench.sh models/qwen2.5-7b-instruct-1m-q4_k_m.gguf --runs 3 --spec
LLAMA_BIN=/path/to/llama.cpp/build/bin tools/bench.sh models/qwen2.5-7b-instruct-1m-q4_k_m.gguf --runs 3 --long
```

The first command prints the code-editing rows, and the second the decode and
prefill rows, for your GPU. The protocol is described in
[docs/benchmark.md](docs/benchmark.md#code-editing). Post your run in the
[v0.0.6 results issue]({{ISSUE_URL}}); earlier runs are in issues
[#1](https://github.com/hardenedlinux/SonicBoom/issues/1),
[#2](https://github.com/hardenedlinux/SonicBoom/issues/2) and
[#3](https://github.com/hardenedlinux/SonicBoom/issues/3).

### Thanks

Community runs keep these numbers honest beyond one card. Thanks to
[@svenha](https://github.com/svenha) for the RTX 4060 Ti report in issue #1,
and to [@mmssix](https://github.com/mmssix) for the four-GPU sweep in issue #2
(RTX 3090 Ti, RTX 3090, RTX 4090 Laptop and RTX PRO 6000 Blackwell), which
also surfaced the CUDA 12.9/13.x build and benchmark-harness fixes.

## What's inside

- **Runtime.** Each decode step is one CUDA Graph replay, with an on-device KV
  cache and exact prefix reuse: a continued prompt gives the same tokens as
  recomputing it from scratch.
- **Kernels.** SonicBoom's own attention kernels (`boom_fattn` for prefill,
  `boom_fdec` for decode and verification) and ggml-derived quantized
  matmuls. The ggml code is MIT-attributed in [NOTICE](NOTICE) and is not
  linked as a dependency.
- **Compiler.** An MLIR dialect, Boom IR, that lowers fused glue kernels
  straight to PTX. Every pass has an on/off switch.
- **Beyond LLMs.** MOSS-Image text-to-image through the same C ABI, and a
  research route that traces Scheme models into graphs, with autograd and
  AdamW.

Architecture, scope and roadmap: [docs/architecture.md](docs/architecture.md),
[docs/scope.md](docs/scope.md), [docs/roadmap.md](docs/roadmap.md).

## How this is built

- Human architectured
- Agent: Claude Code
- Worker: Deepseek-flash-v4.1

## Commercial support

SonicBoom is developed by [HardenedVault](https://hardenedvault.com). We
offer consulting on private or air-gapped LLM inference, performance tuning on
your hardware, and hardened, auditable inference stacks. Contact
[consulting@hardenedvault.com](mailto:consulting@hardenedvault.com).

## License

GPL-3.0-or-later (see [`LICENSE`](LICENSE)). ggml/llama.cpp-derived code
keeps its MIT notice (see [`NOTICE`](NOTICE)).
