#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Reference denoising run for the MOSS-Image pipeline (S5 gate).

Loads only the components the denoising loop needs -- the transformer and the
VAE -- so the (16.8 GiB) text encoder does not have to fit alongside them.
Conditioning comes from MOSSHST1 files (tools/moss/encoder_reference.py), the
initial latent from a MOSLAT01 file, both formats sb-moss reads and writes.

Writes the final latent (MOSLAT01), the decoded image (MOSIMG01) and, with
--trace, the per-step latents (MOSLTR01).
"""

import argparse
import importlib.util
import struct
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2


def load_pipeline_module(model_root):
    path = f"{model_root}/pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_hidden(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"MOSSHST1", path
        tokens, width = struct.unpack("<qq", f.read(16))
        a = np.fromfile(f, dtype="<f4", count=tokens * width)
    return torch.from_numpy(a.reshape(tokens, width)).float()


def read_latents(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"MOSLAT01", path
        b, c, h, w = struct.unpack("<qqqq", f.read(32))
        a = np.fromfile(f, dtype="<f4", count=b * c * h * w)
    return torch.from_numpy(a.reshape(b, c, h, w)).float()


def parse_layers(text):
    if not text:
        return ()
    return tuple(int(v) for v in text.split(",") if v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16",
                    help="transformer dtype (bf16 is the checkpoint's; f32 "
                         "separates executor numerics from the f32 executor)")
    ap.add_argument("--pos-embeds", required=True)
    ap.add_argument("--neg-embeds", required=True)
    ap.add_argument("--latent-in", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--slg-layers", default="")
    ap.add_argument("--slg-scale", type=float, default=0.0)
    ap.add_argument("--slg-start", type=float, default=0.0)
    ap.add_argument("--slg-stop", type=float, default=1.0)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--out-latents", required=True)
    ap.add_argument("--out-image", required=True)
    ap.add_argument("--trace")
    args = ap.parse_args()

    mod = load_pipeline_module(args.model)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    transformer = mod.MOSSImageTransformer.from_pretrained(
        f"{args.model}/transformer", torch_dtype=dtype).eval()
    vae = AutoencoderKLFlux2.from_pretrained(
        f"{args.model}/vae", torch_dtype=torch.float32).eval()
    pipe = mod.MOSSImagePipeline(
        transformer, mod.MOSSImageScheduler(), None, None, vae)

    pos = read_hidden(args.pos_embeds)
    neg = read_hidden(args.neg_embeds)
    latents = read_latents(args.latent_in)
    b, c, h, w = latents.shape
    pos_b = pos.unsqueeze(0).repeat(b, 1, 1)
    neg_b = neg.unsqueeze(0).repeat(b, 1, 1)
    pos_mask = torch.ones(pos_b.shape[:2], dtype=torch.bool)
    neg_mask = torch.ones(neg_b.shape[:2], dtype=torch.bool)

    trace = []
    step_fn = pipe.scheduler.step

    def traced(*a, **k):
        out = step_fn(*a, **k)
        trace.append(out[0].detach().float().clone())
        return out

    pipe.scheduler.step = traced

    scheduler_timesteps = []
    with torch.no_grad():
        images = pipe(
            prompt=None,
            height=16 * h,
            width=16 * w,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            prompt_embeds=pos_b,
            prompt_attention_mask=pos_mask,
            negative_prompt_embeds=neg_b,
            negative_prompt_attention_mask=neg_mask,
            latents=latents,
            output_type="pt",
            uncond_skip_layers=() if args.no_skip else (22,),
            slg_layers=parse_layers(args.slg_layers),
            slg_scale=args.slg_scale,
            slg_start=args.slg_start,
            slg_stop=args.slg_stop,
        )[0]

    with open(args.out_latents, "wb") as f:
        f.write(b"MOSLAT01")
        f.write(struct.pack("<qqqq", b, c, h, w))
        f.write(trace[-1].numpy().astype("<f4").tobytes())
    # postprocess() maps clamp(-1,1) to [0,1] for output_type='pt'; undo it so
    # the dump is the pipeline's clamped image in [-1, 1], like sb-moss writes.
    images = images * 2 - 1
    with open(args.out_image, "wb") as f:
        f.write(b"MOSIMG01")
        ib, ic, ih, iw = images.shape
        f.write(struct.pack("<qqqq", ib, ic, ih, iw))
        f.write(images.numpy().astype("<f4").tobytes())
    if args.trace:
        with open(args.trace, "wb") as f:
            f.write(b"MOSLTR01")
            f.write(struct.pack("<qq", len(trace[0]), len(trace)))
            for t in trace:
                f.write(t.numpy().astype("<f4").tobytes())

    print(f"latent [{b},{c},{h},{w}] steps {len(trace)} "
          f"image {list(images.shape)} rms {images.pow(2).mean().sqrt().item():.6f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
