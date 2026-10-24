#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Compare two MOSSHST1 hidden-state dumps (reference vs SonicBoom).

Reports the maximum and mean relative error per element and the per-token
cosine similarity, plus the row with the largest relative error.
"""

import argparse
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"MOSSHST1":
            raise SystemExit(f"{path}: bad magic {magic!r}")
        tokens, width = struct.unpack("<qq", f.read(16))
        data = np.frombuffer(f.read(), dtype="<f4")
    if data.size != tokens * width:
        raise SystemExit(f"{path}: {data.size} floats for {tokens}x{width}")
    return data.reshape(tokens, width).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    args = ap.parse_args()

    ref = load(args.reference)
    got = load(args.candidate)
    if ref.shape != got.shape:
        raise SystemExit(f"shape mismatch: {ref.shape} vs {got.shape}")

    diff = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-12)
    rel = diff / denom
    scale = max(np.abs(ref).max(), 1e-12)
    # Elementwise relative error is dominated by near-zero entries; also report
    # a scale-free measure: |diff| / RMS(ref) per row.
    rms = np.sqrt((ref**2).mean(axis=1, keepdims=True))
    rel_scaled = diff / np.maximum(rms, 1e-12)
    cos = (ref * got).sum(axis=1) / np.maximum(
        np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1), 1e-12)

    print(f"tokens={ref.shape[0]} hidden={ref.shape[1]} max|ref|={scale:.4g}")
    print(f"rel   max={rel.max():.6g} mean={rel.mean():.6g} p99={np.percentile(rel, 99):.6g}")
    print(f"scaled max={rel_scaled.max():.6g} mean={rel_scaled.mean():.6g}")
    print(f"cos   min={cos.min():.9f} mean={cos.mean():.9f}")
    row = int(np.argmax(rel_scaled.max(axis=1)))
    print(f"worst row={row} cos={cos[row]:.9f} maxrel={rel[row].max():.6g} "
          f"max|diff|={diff[row].max():.6g} rms={rms[row, 0]:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
