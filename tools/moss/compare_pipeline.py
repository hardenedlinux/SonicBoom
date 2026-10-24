#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Compare a SonicBoom MOSS-Image pipeline run with the reference (S5).

usage: compare_pipeline.py --ref-latents A --got-latents B
                           [--ref-image X --got-image Y]
                           [--ref-trace T --got-trace U]

All files are the MOSLAT01 / MOSIMG01 / MOSLTR01 containers. Reports per-step
and final rel_rms / cosine.
"""

import argparse
import os
import struct
import sys

import numpy as np


def read(path, magic, extra=0):
    with open(path, "rb") as f:
        assert f.read(8) == magic, path
        dims = struct.unpack("<%dq" % extra, f.read(8 * extra))
        # The header shape is (batch, steps) for a trace; the payload fills the
        # rest of the file.
        n = (os.path.getsize(path) - 8 - 8 * extra) // 4
        a = np.fromfile(f, dtype="<f4", count=n)
    return dims, a


def stats(got, ref):
    g = got.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    d = g - r
    denom = np.sqrt(np.mean(r * r)) + 1e-30
    cos = float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    return (float(np.sqrt(np.mean(d * d)) / denom), float(np.max(np.abs(d))),
            cos, float(np.sqrt(np.mean(g * g))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-latents")
    ap.add_argument("--got-latents")
    ap.add_argument("--ref-image")
    ap.add_argument("--got-image")
    ap.add_argument("--ref-trace")
    ap.add_argument("--got-trace")
    args = ap.parse_args()

    if args.ref_trace and args.got_trace:
        rd, rt = read(args.ref_trace, b"MOSLTR01", 2)
        gd, gt = read(args.got_trace, b"MOSLTR01", 2)
        # One writer emits (batch, steps), the other (steps, batch); the data
        # is step-major in both, so only the product matters.
        assert rd[0] * rd[1] == gd[0] * gd[1], (rd, gd)
        steps = rd[0] * rd[1]
        # trace layout: [steps][N] values, N = C*H*W
        n = rt.size // steps if steps else 0
        for s in range(steps):
            rel, mx, cos, rms = stats(gt[s * n:(s + 1) * n], rt[s * n:(s + 1) * n])
            print(f"step {s}: latent rel_rms {rel:.3e} max_abs {mx:.3e} "
                  f"cos {cos:.9f} rms {rms:.6f}")

    if args.ref_latents and args.got_latents:
        rdims, ra = read(args.ref_latents, b"MOSLAT01", 4)
        gdims, ga = read(args.got_latents, b"MOSLAT01", 4)
        assert rdims == gdims, (rdims, gdims)
        rel, mx, cos, rms = stats(ga, ra)
        print(f"final latent: rel_rms {rel:.3e} max_abs {mx:.3e} "
              f"cos {cos:.9f} rms {rms:.6f}")

    if args.ref_image and args.got_image:
        rdims, ra = read(args.ref_image, b"MOSIMG01", 4)
        gdims, ga = read(args.got_image, b"MOSIMG01", 4)
        assert rdims == gdims, (rdims, gdims)
        rel, mx, cos, rms = stats(ga, ra)
        print(f"image: rel_rms {rel:.3e} max_abs {mx:.3e} cos {cos:.9f} "
              f"rms {rms:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
