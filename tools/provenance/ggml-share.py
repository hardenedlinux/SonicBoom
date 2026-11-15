#!/usr/bin/env python3
"""Measure how much of SonicBoom's source is line-identical to upstream llama.cpp.

The unit of measurement is the *substantive line*: a source line after all
whitespace has been removed, with blank lines, punctuation-only lines and
comment-only lines discarded. A SonicBoom line is counted as upstream if the
same substantive line occurs anywhere in llama.cpp's sources at the pinned
commit.

Usage:
    python3 tools/provenance/ggml-share.py --upstream /path/to/llama.cpp \
        [--tsv OUT.tsv]

Only the Python standard library is used.
"""

import argparse
import os
import subprocess
import sys

UPSTREAM_REV = "43fe9c642"

UPSTREAM_EXTS = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
PRODUCT_EXTS = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".td"}

# Directory trees that never count towards the product totals.
SCAN_ROOTS = ("core", "capi", "bindings")
EXCLUDED_DIRS = {"tests", "tools", "third_party"}

DERIVED_DIRS = ("core/src/quant/ggml_derived", "core/src/vocab/ggml_derived")

PUNCTUATION_ONLY = set("{}();,")

THRESHOLD = 0.60


def probe(path):
    """Return one normalized source line, or None if it is not substantive."""
    line = "".join(path.split())
    if not line:
        return None
    if all(ch in PUNCTUATION_ONLY for ch in line):
        return None
    if line.startswith("//") or line.startswith("/*") or line.startswith("*"):
        return None
    return line


def substantive_lines(path, scheme):
    """Yield substantive lines of a file. `scheme` is "c" or "scm"."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = probe(raw)
            if line is None:
                continue
            if scheme == "scm" and line.startswith(";"):
                continue
            yield line


def walk_files(root, extensions):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1] in extensions:
                yield os.path.join(dirpath, name)


def relative(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


def excluded(relpath):
    return any(part in EXCLUDED_DIRS for part in relpath.split("/")[:-1])


def upstream_lines(upstream):
    lines = set()
    for sub in ("ggml/src", "src"):
        root = os.path.join(upstream, sub)
        if not os.path.isdir(root):
            continue
        for path in walk_files(root, UPSTREAM_EXTS):
            lines.update(substantive_lines(path, "c"))
    return lines


def repo_files(repo):
    for top in SCAN_ROOTS:
        root = os.path.join(repo, top)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                relpath = relative(repo, os.path.join(dirpath, name))
                if excluded(relpath):
                    continue
                ext = os.path.splitext(name)[1]
                if ext in PRODUCT_EXTS:
                    yield os.path.join(dirpath, name), "c"
                elif ext == ".scm":
                    yield os.path.join(dirpath, name), "scm"


def is_derived(relpath):
    return any(relpath == d or relpath.startswith(d + "/") for d in DERIVED_DIRS)


def is_engine(relpath):
    return relpath.startswith("core/") and not is_derived(relpath)


def percent(part, whole):
    if whole == 0:
        return "0.0%"
    return "%.1f%%" % (100.0 * part / whole)


def main():
    parser = argparse.ArgumentParser(
        description="Measure SonicBoom source lines identical to upstream llama.cpp."
    )
    parser.add_argument("--upstream", required=True,
                        help="path to the llama.cpp checkout")
    parser.add_argument("--tsv", help="write per-file results to this TSV file")
    args = parser.parse_args()

    try:
        head = subprocess.check_output(
            ["git", "-C", args.upstream, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        print("error: cannot read git HEAD of %s" % args.upstream, file=sys.stderr)
        return 2

    if not head.startswith(UPSTREAM_REV):
        print("error: %s is at %s, expected %s"
              % (args.upstream, head[:12], UPSTREAM_REV), file=sys.stderr)
        return 2

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reference = upstream_lines(args.upstream)

    rows = []
    for path, scheme in repo_files(repo):
        relpath = relative(repo, path)
        lines = list(substantive_lines(path, scheme))
        total = len(lines)
        matched = sum(1 for line in lines if line in reference)
        ratio = (float(matched) / total) if total else 0.0
        rows.append((relpath, total, matched, ratio))
    rows.sort()

    derived = [r for r in rows if is_derived(r[0])]
    engine = [r for r in rows if is_engine(r[0])]
    product = rows

    def totals(group):
        return sum(r[1] for r in group), sum(r[2] for r in group)

    derived_n, derived_u = totals(derived)
    engine_n, engine_u = totals(engine)
    product_n, product_u = totals(product)

    print("ggml_derived: n=%d u=%d (%s)"
          % (derived_n, derived_u, percent(derived_u, derived_n)))
    print("engine (core/, excluding ggml_derived): n=%d u=%d (%s)"
          % (engine_n, engine_u, percent(engine_u, engine_n)))
    print("product (core/, capi/, bindings/): n=%d u=%d (%s)"
          % (product_n, product_u, percent(product_u, product_n)))

    high_outside = [r for r in rows
                    if r[3] >= THRESHOLD and not is_derived(r[0])]
    low_inside = [r for r in derived if r[3] < THRESHOLD]

    print("")
    print("files >= %d%% outside ggml_derived (%d):" % (THRESHOLD * 100, len(high_outside)))
    for relpath, total, matched, ratio in high_outside:
        print("  %s\tn=%d\tu=%d\t%.4f" % (relpath, total, matched, ratio))

    print("")
    print("files < %d%% inside ggml_derived (%d):" % (THRESHOLD * 100, len(low_inside)))
    for relpath, total, matched, ratio in low_inside:
        print("  %s\tn=%d\tu=%d\t%.4f" % (relpath, total, matched, ratio))

    if args.tsv:
        with open(args.tsv, "w", encoding="utf-8") as handle:
            for relpath, total, matched, ratio in rows:
                handle.write("%s\t%d\t%d\t%.4f\n" % (relpath, total, matched, ratio))

    return 0


if __name__ == "__main__":
    sys.exit(main())
