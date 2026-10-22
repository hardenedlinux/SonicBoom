#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Dump a FLUX.2 VAE decode reference (S4).

usage: vae_reference.py --vae DIR --out FILE [--height 8] [--width 8] [--seed 7]

Runs AutoencoderKLFlux2.decode on a seeded fp32 latent and writes "MOSVAE1":
  int64 C, H, W, OC, then latent[C*H*W] f32, out[OC*(4H)*(4W)] f32.
With --stages FILE it also writes named intermediate tensors as float32
(shape-prefixed) for block-by-block parity checks.
"""

import argparse
import struct
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2


def write_array(f, name, t):
    a = t.detach().float().cpu().numpy().astype("<f4")
    f.write(struct.pack("<i", len(name)))
    f.write(name.encode())
    f.write(struct.pack("<i", a.ndim))
    f.write(struct.pack("<%dq" % a.ndim, *a.shape))
    f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stages")
    args = ap.parse_args()

    model = AutoencoderKLFlux2.from_pretrained(args.vae, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    c = cfg.latent_channels
    g = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, c, args.height, args.width, generator=g, dtype=torch.float32)

    stages = []
    handles = []
    if args.stages:
        def hook(name):
            def fn(_m, _i, out):
                stages.append((name, out))
            return fn
        named = dict(model.decoder.named_modules())
        targets = ["conv_in", "mid_block.resnets.0", "mid_block.attentions.0",
                   "mid_block.resnets.1"]
        for i in range(len(model.decoder.up_blocks)):
            for j in range(len(model.decoder.up_blocks[i].resnets)):
                targets.append(f"up_blocks.{i}.resnets.{j}")
            if model.decoder.up_blocks[i].upsamplers is not None:
                targets.append(f"up_blocks.{i}.upsamplers.0")
        targets += ["conv_norm_out", "conv_out"]
        for name in targets:
            handles.append(named[name].register_forward_hook(hook(name)))

    with torch.no_grad():
        out = model.decode(latent, return_dict=False)[0]
    for h in handles:
        h.remove()

    with open(args.out, "wb") as f:
        f.write(b"MOSVAE1")
        f.write(struct.pack("<qqqq", c, args.height, args.width, out.shape[1]))
        f.write(latent.reshape(-1).numpy().astype("<f4").tobytes())
        f.write(out.reshape(-1).numpy().astype("<f4").tobytes())
    print(f"latent [{c},{args.height},{args.width}] -> out {list(out.shape)} "
          f"rms {out.pow(2).mean().sqrt().item():.6f}")

    if args.stages:
        with open(args.stages, "wb") as f:
            f.write(b"MOSVSG01")
            f.write(struct.pack("<q", len(stages)))
            for name, t in stages:
                write_array(f, name, t)
        for name, t in stages:
            print(f"  {name:28s} {list(t.shape)} rms {t.pow(2).mean().sqrt().item():.6f}")

    # Parameter inventory, so the executor's loader can be checked against it.
    if args.stages:
        for name, p in model.decoder.named_parameters():
            print(f"  W {name:44s} {list(p.shape)}")
        print(f"  W post_quant_conv.weight {list(model.post_quant_conv.weight.shape)}")
        print(f"  W post_quant_conv.bias {list(model.post_quant_conv.bias.shape)}")


if __name__ == "__main__":
    main()
