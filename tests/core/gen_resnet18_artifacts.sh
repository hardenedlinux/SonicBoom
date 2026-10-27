#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# CTest setup fixture for the two end-to-end ResNet-18 tests. It regenerates
# every artifact they read into the build tree, none of which is committed:
#
#   <build>/import/resnet18.sx            importer output (S-Expr)
#   <build>/import/resnet18.weights.bin   weight sidecar (both tests)
#   <build>/import/resnet18.input.bin     deterministic input (seed 1234)
#   <build>/import/resnet18.reference.bin ONNX reference logits
#
# The model-derived sidecars are never written into the source tree.
#
# Usage: gen_resnet18_artifacts.sh <source-root> <python> <import-dir>
#
# The model itself is not in the repository (models/ points at an external
# store), so without models/resnet18.onnx the fixture exits 77 and CTest
# reports the dependent tests as skipped rather than failed.

set -eu

root=$1
python=$2
import_dir=$3

onnx="$root/models/resnet18.onnx"
if [ ! -f "$onnx" ]; then
    echo "SKIP: $onnx is missing; the ResNet-18 artifacts cannot be generated"
    exit 77
fi

mkdir -p "$import_dir"

# A fixture setup runs on every ctest invocation; skip the ~46 MB of sidecars
# when they are already newer than the model they derive from. Only build-tree
# outputs are checked: the source tree holds no generated sidecars.
fresh=1
for out in "$import_dir/resnet18.sx" "$import_dir/resnet18.weights.bin" \
           "$import_dir/resnet18.input.bin" \
           "$import_dir/resnet18.reference.bin"; do
    if [ ! -f "$out" ] || [ "$onnx" -nt "$out" ]; then
        fresh=0
    fi
done
if [ "$fresh" -eq 1 ]; then
    echo "resnet18 artifacts are newer than $onnx; nothing to do"
    exit 0
fi

cd "$root"
# The reference generator writes the frozen-fixture weight sidecar (plus the
# input and reference logits); the importer then writes its own sidecar, which
# the two tests consume. The importer runs last so that the end-to-end ONNX
# test still exercises the importer's weight emission rather than the
# generator's — the two layouts are byte-identical.
"$python" "$root/tools/reference/gen_resnet18_reference.py" "$import_dir"
PYTHONPATH="$root/tools" "$python" -m onnx2sx "$onnx" \
    -o "$import_dir/resnet18.sx" --weights-dir "$import_dir"
