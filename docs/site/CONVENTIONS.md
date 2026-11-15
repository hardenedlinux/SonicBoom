# Documentation conventions for the Guile sources

This document is the contract between the Guile sources and the documentation
site generator (`docs/site/extract.scm`, `docs/site/build.scm`). The sources are
the only source of the API reference: whatever is written here is what appears
on the site.

## Procedures

Procedures defined with `define`, `define*`, or `define` whose value is a
`lambda`/`lambda*` carry a docstring in a fixed format:

    "One-sentence summary ending with a period.

    Optional longer description, one or more paragraphs.

    Args:
      name: what it is, type and shape if relevant.
      name: ...
    Returns: what comes back.
    Example:
      (use-modules (sonicboom tensor))
      (tensor->list (tensor-add a b))
      => (2.0 4.0)
    PyTorch: torch.add
    Note: how it differs from the PyTorch function, if it does."

The rules:

- The first line is one complete sentence ending with a period.
- `Args:`, `Returns:`, `Example:`, `PyTorch:`, and `Note:` are fixed labels.
  They start in the first column. A procedure without arguments omits `Args:`.
  `Note:` is optional.
- Each `Args:` entry is indented two spaces and reads `name: description`.
- The `Example:` body is indented two spaces. It ends at the first blank line or
  at the next label. A line `=> value` gives the expected value of the last
  expression of the example, compared with `write`. When the result is a
  tensor, the example converts it to a list before comparing.
- `PyTorch:` is mandatory. Its value is one or more comma-separated full
  PyTorch names (`torch.nn.functional.relu`), or `none` when there is no
  counterpart.

Docstrings describe the behaviour of the interface only. They do not describe
engine internals, tuning, scheduling, or performance, and they never refer to
private repositories.

## Macros, record types, and variables

Macros, record types, and variables have no docstring. Instead, the `;;`
comment block immediately above the definition carries the documentation, in
the format above: each line, with the leading semicolons and one space
removed, is a line of the documentation.

A record type gets one comment block above its `define-record-type` (or
`define-condition-type`) that describes the type and its fields. Its
constructor, predicate, and accessors are documented by that block and are not
documented separately.

## Modules

A module's description is the `;;;` comment block at the top of its file, with
the SPDX and copyright lines removed. The first line of that block names the
module; the rest describes it.

## Adding or changing an exported symbol

When a commit adds or changes an exported symbol, it updates the documentation
(the docstring or comment block) and `docs/site/pytorch-map.scm` in the same
commit. The four documentation tests (`ctest -L docs`: coverage, examples, map,
and site) are part of the basic checks.
