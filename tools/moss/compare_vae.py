#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Compare a SonicBoom VAE decode against the diffusers reference (S4).

usage: compare_vae.py --ref ref.bin --got ours.bin [--ref-stages f --got-stages f]
"""

import argparse
import struct
import sys

import numpy as np


def read_vae(path):
    with open(path, "rb") as f:
        assert f.read(7) == b"MOSVAE1", path
        c, h, w, oc = struct.unpack("<qqqq", f.read(32))
        latent = np.fromfile(f, dtype="<f4", count=c * h * w)
        out = np.fromfile(f, dtype="<f4", count=oc * 64 * h * w)
    return (c, h, w, oc), latent, out


def read_stages(path):
    stages = {}
    with open(path, "rb") as f:
        assert f.read(8) == b"MOSVSG01", path
        (n,) = struct.unpack("<q", f.read(8))
        for _ in range(n):
            (ln,) = struct.unpack("<i", f.read(4))
            name = f.read(ln).decode()
            (nd,) = struct.unpack("<i", f.read(4))
            shape = struct.unpack("<%dq" % nd, f.read(8 * nd))
            a = np.frombuffer(f.read(4 * int(np.prod(shape))), dtype="<f4")
            stages[name] = a.reshape(shape)
    return stages


def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    diff = a - b
    denom = np.sqrt(np.mean(b * b)) + 1e-30
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {
        "rel_rms": float(np.sqrt(np.mean(diff * diff)) / denom),
        "max_abs": float(np.max(np.abs(diff))),
        "cos": cos,
        "rms": float(np.sqrt(np.mean(a * a))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got", required=True)
    ap.add_argument("--ref-stages")
    ap.add_argument("--got-stages")
    ap.add_argument("--tol", type=float, default=2e-4)
    args = ap.parse_args()

    _, lat_r, out_r = read_vae(args.ref)
    _, lat_g, out_g = read_vae(args.got)
    assert np.array_equal(lat_r, lat_g), "latents differ"
    s = stats(out_g, out_r)
    print(f"conv_out image: rel_rms {s['rel_rms']:.3e} max_abs {s['max_abs']:.3e} "
          f"cos {s['cos']:.9f} rms {s['rms']:.6f}")
    worst = s["rel_rms"]

    if args.ref_stages and args.got_stages:
        rs = read_stages(args.ref_stages)
        gs = read_stages(args.got_stages)
        missing = [k for k in rs if k not in gs]
        if missing:
            print("missing stages:", missing)
            return 1
        for name in rs:
            a, b = gs[name], rs[name]
            if a.shape != b.shape:
                print(f"  {name:28s} SHAPE {a.shape} vs {b.shape}")
                return 1
            st = stats(a, b)
            worst = max(worst, st["rel_rms"])
            print(f"  {name:28s} rel_rms {st['rel_rms']:.3e} "
                  f"max_abs {st['max_abs']:.3e} cos {st['cos']:.9f}")
    print(f"worst rel_rms {worst:.3e} (tol {args.tol:.1e}) "
          f"{'PASS' if worst <= args.tol else 'FAIL'}")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
