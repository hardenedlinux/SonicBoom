# Provenance measurement

`ggml-share.py` measures how much of SonicBoom's source is copied from upstream
llama.cpp. The reference is llama.cpp commit `43fe9c642`; any other checkout is
refused (exit code 2).

The unit is the **substantive line**: a source line after all whitespace has
been removed, with blank lines, lines made only of `{ } ( ) ; ,`, and lines
starting with `//`, `/*` or `*` discarded. Comment and formatting noise is
therefore ignored, but a line that differs from upstream by even one character
is not counted. A line counts as upstream if the identical normalized line
occurs anywhere in llama.cpp's `ggml/src/` or `src/` sources.

The product side covers `core/`, `capi/` and `bindings/`: C, C++ and CUDA
sources plus TableGen (`.td`) files, and Guile Scheme (`.scm`) files under
`bindings/` (where lines starting with `;` are discarded as well). Top-level
`tests/`, `tools/` and `third_party/` are not counted.

## Running

```sh
python3 tools/provenance/ggml-share.py --upstream /path/to/llama.cpp \
    --tsv /tmp/provenance.tsv
```

The TSV has one row per file: `path`, substantive lines `n`, matched lines `u`,
and `u / n`. Standard output reports the totals for the `ggml_derived/` trees,
for the engine (everything under `core/` outside `ggml_derived/`), and for the
whole product, plus the two threshold exceptions.

## The `ggml_derived/` rule

A file may live under a `ggml_derived/` directory only if at least 60% of its
substantive lines are identical to upstream. Files below that line are moved to
the ordinary source tree and keep their attribution and license notice in the
file header; files at or above it stay where they are. The tool reports both
violations of this rule, so a reorganization can be checked mechanically.
