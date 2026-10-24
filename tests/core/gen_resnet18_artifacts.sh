#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# CTest setup fixture for the two end-to-end ResNet-18 tests. It regenerates
# every artifact they read, none of which is committed:
#
#   <build>/import/resnet18.sx            importer output (S-Expr)
#   <build>/import/resnet18.weights.bin   importer weight sidecar
#   tests/data/resnet18.weights.bin       frozen-fixture weight sidecar
#   tests/data/resnet18.input.bin         deterministic input (seed 1234)
#   tests/data/resnet18.reference.bin     ONNX reference logits
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
# when they are already newer than the model they derive from.
fresh=1
for out in "$import_dir/resnet18.sx" "$import_dir/resnet18.weights.bin" \
           "$root/tests/data/resnet18.weights.bin" \
           "$root/tests/data/resnet18.input.bin" \
           "$root/tests/data/resnet18.reference.bin"; do
    if [ ! -f "$out" ] || [ "$onnx" -nt "$out" ]; then
        fresh=0
    fi
done
if [ "$fresh" -eq 1 ]; then
    echo "resnet18 artifacts are newer than $onnx; nothing to do"
    exit 0
fi

cd "$root"
PYTHONPATH="$root/tools" "$python" -m onnx2sx "$onnx" \
    -o "$import_dir/resnet18.sx" --weights-dir "$import_dir"
"$python" "$root/tools/reference/gen_resnet18_reference.py"
