#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Compare two MOSSDIT1 files (S3 gate).

usage: compare_dit.py REFERENCE OURS [--latent-channels 128] [--text-dim 4096] [--dim 3072]

The inputs must be identical; the three output arrays are compared by relative
RMS, max absolute difference, and per-row cosine (rows = tokens for the
residual dumps, channels for the sample).
"""

import argparse
import struct
import sys

import numpy as np


def read(path, channels, text_dim, dim):
    data = open(path, "rb").read()
    if data[:8] != b"MOSSDIT1":
        sys.exit(f"{path}: not a MOSSDIT1 file")
    off = 8
    height, width, text_len = struct.unpack_from("<qqq", data, off)
    off += 24
    (timestep,) = struct.unpack_from("<f", data, off)
    off += 4
    hw = height * width
    tokens = hw + text_len
    sizes = [channels * hw, text_len * text_dim, channels * hw,
             tokens * dim, hw * dim]
    names = ["latent", "text", "out", "block0", "prehead"]
    arrays = {}
    for name, size in zip(names, sizes):
        arrays[name] = np.frombuffer(data, dtype="<f4", count=size, offset=off).copy()
        off += size * 4
    return height, width, text_len, timestep, arrays


def report(name, a, b, rows):
    diff = a.astype(np.float64) - b.astype(np.float64)
    ref = np.linalg.norm(b.astype(np.float64))
    denom = ref if ref > 0 else 1.0
    ra = a.astype(np.float64).reshape(rows, -1)
    rb = b.astype(np.float64).reshape(rows, -1)
    dots = (ra * rb).sum(1)
    norms = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
    cos = dots / np.where(norms > 0, norms, 1.0)
    print(f"  {name:8s} rel_rms={np.linalg.norm(diff) / denom:.6e} "
          f"max_abs={np.abs(diff).max():.6e} "
          f"cos min={cos.min():.6f} mean={cos.mean():.6f} worst_row={cos.argmin()}")
    return cos.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("ours")
    ap.add_argument("--latent-channels", type=int, default=128)
    ap.add_argument("--text-dim", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=3072)
    args = ap.parse_args()

    ha, wa, sa, ta, ref = read(args.reference, args.latent_channels, args.text_dim, args.dim)
    hb, wb, sb, tb, ours = read(args.ours, args.latent_channels, args.text_dim, args.dim)
    if (ha, wa, sa, ta) != (hb, wb, sb, tb):
        sys.exit("input headers differ")
    for name in ("latent", "text"):
        if not np.array_equal(ref[name], ours[name]):
            sys.exit(f"input array {name} differs")
    print(f"h={ha} w={wa} text={sa} timestep={ta}  inputs identical")
    hw = ha * wa
    rows = {
        "out": args.latent_channels,
        "block0": hw + sa,
        "prehead": hw,
    }
    worst = min(report(name, ours[name], ref[name], rows[name]) for name in rows)
    print(f"worst per-row cos {worst:.6f}")


if __name__ == "__main__":
    main()
