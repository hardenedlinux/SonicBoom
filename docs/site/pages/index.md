# SonicBoom

**A native tensor runtime. No Python. One `.so`.**

SonicBoom is written from the ground up in C++ and CUDA, exposed through a
stable C ABI, with GNU Guile as its first-class language. No Python
interpreter, pip or wheel tree is involved in inference.

<div class="search search-large">
  <input id="index-search" class="site-search-input" type="search"
         placeholder="Search the site" aria-label="Search the site"
         autocomplete="off">
  <div class="search-results" hidden></div>
</div>

## Where to start

- [Getting started](getting-started.html) — build SonicBoom and run a model
  from Guile.
- [The Guile binding](api/index.html) — the API reference, generated from the
  modules' documentation.
- [Architecture](architecture.html) — how the pieces fit together.
- [Coming from PyTorch](pytorch.html) — look up a PyTorch name and find the
  SonicBoom function or the few lines that do the same.
- [Benchmarks](benchmark.html) — measured numbers.
- [Roadmap](roadmap.html) — what is done and what comes next.
