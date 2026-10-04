#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# Development-time test oracle for the SonicBoom ResNet-18 CPU execution path.
# This is NOT a SonicBoom runtime dependency and never runs as part of the
# build. It generates three byte-level artifacts from models/resnet18.onnx:
#
#   1. design/resnet18.weights.bin   — the flat float32 weight sidecar that the
#      frozen fixture design/s-expr-v0-resnet18.example.sx references by
#      (:external file offset length). It is the concatenation, in the exact
#      order and byte offsets declared by the fixture, of the 42 float32 ONNX
#      initializers (20 conv weights, fc.weight, fc.bias, 20 conv biases). The
#      two int64 constants (val_187 / val_191) are inlined with :values in the
#      fixture and are therefore absent from the sidecar.
#
#   2. design/resnet18.input.bin     — one deterministic float32 input
#      [1,3,224,224] (NCHW, row-major), drawn from a fixed seed.
#
#   3. design/resnet18.reference.bin — the [1,1000] float32 output produced by
#      ONNX's own reference evaluator (onnx.reference.ReferenceEvaluator) for
#      that input. This is the independent ground truth the C++ test compares
#      against.
#
# Usage:  models/.venv/bin/python tools/reference/gen_resnet18_reference.py
#
# The script is self-checking: it parses the frozen fixture for the exact
# (:external file offset length) layout and asserts the concatenated weights
# match it byte-for-byte (offset, length, and dtype/size).

import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "models")
DESIGN = os.path.join(REPO, "design")
FIXTURE = os.path.join(DESIGN, "s-expr-v0-resnet18.example.sx")
ONNX_PATH = os.path.join(MODELS, "resnet18.onnx")

# Deterministic input seed (documented in the lowering/execution docs).
SEED = 1234


# --- minimal S-expression reader, just enough to pull the parameter list ----

def _tokenize(text):
    # Strip `;;` line comments, then split parens into their own tokens.
    lines = [ln.split(";;", 1)[0] for ln in text.splitlines()]
    text = "\n".join(lines)
    return text.replace("(", " ( ").replace(")", " ) ").split()


def _parse(tokens, i=0):
    tok = tokens[i]
    if tok == "(":
        i += 1
        items = []
        while tokens[i] != ")":
            item, i = _parse(tokens, i)
            items.append(item)
        return items, i + 1
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        tok = tok[1:-1]  # strip string-literal quotes
    return tok, i + 1


def parse_sx(text):
    tokens = _tokenize(text)
    doc, _ = _parse(tokens)
    return doc


def find_form(tree, head):
    """Return the first (head ...) form nested anywhere under `tree`."""
    if isinstance(tree, list):
        if tree and tree[0] == head:
            return tree
        for child in tree:
            r = find_form(child, head)
            if r is not None:
                return r
    return None


def find_all(tree, head):
    out = []
    if isinstance(tree, list):
        if tree and tree[0] == head:
            out.append(tree)
        for child in tree:
            out.extend(find_all(child, head))
    return out


def parse_parameters(tree):
    """Return [(name, dtype, dims, file, offset, length)] for :external params."""
    params_form = find_form(tree, "parameters")
    result = []
    for p in find_all(params_form, "parameter"):
        # (parameter NAME (tensor DTYPE (shape DIMS...)) (data :external F O L))
        name = p[1]
        tensor = p[2]
        dtype = tensor[1]
        dims = [int(d) for d in tensor[2][1:]]
        data = p[3]
        if data[1] != ":external":
            continue  # :values constants are inlined in the fixture, not in the bin
        fname, offset, length = data[2], int(data[3]), int(data[4])
        result.append((name, dtype, dims, fname, offset, length))
    return result


def main():
    assert os.path.exists(ONNX_PATH), f"missing {ONNX_PATH} (run models/resnet18.py first)"
    with open(FIXTURE) as f:
        tree = parse_sx(f.read())

    params = parse_parameters(tree)
    onnx_model = onnx.load(ONNX_PATH)
    inits = {i.name: numpy_helper.to_array(i) for i in onnx_model.graph.initializer}

    # Build the weight sidecar in fixture order and verify the layout.
    blob = bytearray()
    for name, dtype, dims, fname, offset, length in params:
        assert fname == "resnet18.weights.bin", fname
        arr = inits[name]
        assert list(arr.shape) == dims, (name, arr.shape, dims)
        assert arr.dtype == np.float32, (name, arr.dtype)
        assert arr.nbytes == length, (name, arr.nbytes, length)
        assert len(blob) == offset, (name, len(blob), offset)
        blob.extend(arr.tobytes(order="C"))

    weights_path = os.path.join(DESIGN, "resnet18.weights.bin")
    with open(weights_path, "wb") as f:
        f.write(blob)
    print(f"wrote {weights_path} ({len(blob)} bytes, {len(params)} tensors)")

    # Deterministic input and independent reference output.
    rng = np.random.RandomState(SEED)
    x = rng.randn(1, 3, 224, 224).astype(np.float32)
    input_path = os.path.join(DESIGN, "resnet18.input.bin")
    x.tofile(input_path)
    print(f"wrote {input_path} ({x.nbytes} bytes, seed={SEED})")

    evaluator = ReferenceEvaluator(ONNX_PATH)
    y = evaluator.run(None, {"input": x})[0]
    assert y.shape == (1, 1000) and y.dtype == np.float32, (y.shape, y.dtype)
    assert np.isfinite(y).all()
    ref_path = os.path.join(DESIGN, "resnet18.reference.bin")
    np.ascontiguousarray(y).tofile(ref_path)
    print(f"wrote {ref_path} ({y.nbytes} bytes, argmax={int(y.argmax())})")


if __name__ == "__main__":
    main()
