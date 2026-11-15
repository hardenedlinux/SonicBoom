# Provenance measurement

`ggml-share.py` measures how much of SonicBoom's source is copied from upstream
llama.cpp. The reference is llama.cpp commit `43fe9c642`; any other checkout is
refused (exit code 2).

The unit is the **substantive line**: a source line after all whitespace has
been removed, with blank lines, lines made only of `{ } ( ) ; ,`, and lines
starting with `//`, `/*` or `*` discarded. Comment and formatting noise is
therefore ignored, but a line that differs from upstream by even one character
is not counted.

Matching is by **run**, not by single line. Every run of `W` consecutive
substantive lines (default `W`=4, set with `--window`) is taken as a tuple;
the upstream side collects every such tuple in llama.cpp's `ggml/src/` and
`src/` sources, and a SonicBoom line counts as upstream only when it sits
inside a run whose tuple occurs upstream. A single line is not enough because
generic statements such as `return false;` occur in both trees by coincidence
and would count as copied code.

The product side covers `core/`, `capi/` and `bindings/`: C, C++ and CUDA
sources plus TableGen (`.td`) files, and Guile Scheme (`.scm`) files under
`bindings/` (where lines starting with `;` are discarded as well). Top-level
`tests/`, `tools/` and `third_party/` are not counted.

## Running

```sh
python3 tools/provenance/ggml-share.py --upstream /path/to/llama.cpp \
    --window 4 --tsv /tmp/provenance.tsv
```

The TSV has one row per file: `path`, substantive lines `n`, matched lines `u`,
and `u / n`. Standard output reports the totals for the `ggml_derived/` trees,
for the engine (everything under `core/`, including `ggml_derived/`, with the
matched lines of `ggml_derived/` and of the rest of `core/` listed separately),
and for the whole product, plus the threshold lists. The first line states the
run length the totals were measured with. Files with fewer than `2 * W`
substantive lines are listed as too small to judge and take no part in the 60%
decision.

## The `ggml_derived/` rule

A file under a `ggml_derived/` directory is code taken from upstream
llama.cpp. It keeps its place there even when SonicBoom has modified it
heavily: the directory records where the code came from, not how much of it
still sits in upstream runs.

The 60% line decides only whether a file outside `ggml_derived/` should be
moved in: at least 60% of its substantive lines sitting in upstream runs (`u /
n` with the default `--window 4`) means the file is upstream code. A file
inside `ggml_derived/` below 60% is reported for information only; the list is
not a move list. Files with fewer than `2 * W` substantive lines are listed as
too small to judge and take no part in the 60% decision.

The ratio is counted per line: which directory a file sits in does not change
its `n`, `u` or `u / n`.
