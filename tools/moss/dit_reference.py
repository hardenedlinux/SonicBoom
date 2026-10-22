#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Reference MOSS-Image DiT forward for the S3 numerics gate.

Loads the released transformer (the modeling code ships inside the weights
repo as pipeline_moss_image.py) and runs one deterministic sample, writing the
inputs and the reference outputs as "MOSSDIT1":

    char[8]  magic
    i64      height, width, text_len
    f32      timestep
    f32      latent[latent_channels * height * width]   (row-major [C][H][W])
    f32      text[text_len * text_dim]
    f32      out[latent_channels * height * width]      (the sample)
    f32      x_block0[(h*w + text_len) * dim]           (after layer 0)
    f32      prehead[h * w * dim]                       (before the head)

sb-moss dit reads the file, echoes the inputs and writes its own outputs; the
comparison script diffs the arrays.  Inputs are generated here (not by a
formula shared with C++) so both sides see the identical bytes.

--dtype float32 casts the bf16 checkpoint before the forward: that is the
apples-to-apples reference for SonicBoom's f32-from-bf16-weights executor.
--dtype bfloat16 is the shipped configuration (what the pipeline runs).
"""

import argparse
import importlib.util
import json
import struct
import sys

import torch


def load_pipeline_module(path):
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_moss_image"] = module
    spec.loader.exec_module(module)
    return module


def load_state_dict(directory, torch_dtype):
    from safetensors.torch import load_file

    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        tensors = load_file(str(directory / shard))
        for key, value in tensors.items():
            # rope.bands is a derived constant; the module recomputes it from
            # the config on load, so its stored value is irrelevant.
            state[key] = value.to(torch_dtype) if torch_dtype != torch.bfloat16 else value
        del tensors
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="transformer directory")
    ap.add_argument("--pipeline", default=None, help="pipeline_moss_image.py")
    ap.add_argument("--height", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--text-len", type=int, default=11)
    ap.add_argument("--timestep", type=float, default=500.0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pathlib import Path

    model_dir = Path(args.model)
    pipeline = Path(args.pipeline) if args.pipeline else model_dir / "pipeline_moss_image.py"
    mod = load_pipeline_module(pipeline)

    config = json.loads((model_dir / "config.json").read_text())
    config.pop("_class_name", None)

    torch_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = mod.MOSSImageTransformer(**config)
    state = load_state_dict(model_dir, torch_dtype)
    model.load_training_state_dict(state)
    del state
    model.to(torch_dtype)
    model.eval()

    # Deterministic inputs (bf16-representable-free, small integers/decimals).
    g = torch.Generator().manual_seed(20261021)
    latent = torch.randn(1, config["latent_channels"], args.height, args.width,
                         generator=g, dtype=torch.float32)
    text = torch.randn(1, args.text_len, config["text_dim"],
                       generator=g, dtype=torch.float32)

    captured = {}

    def make_hook(name, video_rows=None):
        def hook(_module, _inputs, output):
            value = output[:, :video_rows] if video_rows else output
            captured[name] = value.detach().float().clone()
        return hook

    handles = [
        model.layers[0].register_forward_hook(make_hook("x_block0")),
        model.layers[-1].register_forward_hook(
            make_hook("prehead", args.height * args.width)),
    ]

    with torch.no_grad():
        out = model(latent.to(torch_dtype), args.timestep, text.to(torch_dtype)).sample
    for handle in handles:
        handle.remove()
    out = out.float()

    x_block0 = captured["x_block0"][0].contiguous().reshape(-1)
    prehead_x = captured["prehead"][0].contiguous().reshape(-1)
    out_flat = out.reshape(-1).contiguous()

    with open(args.out, "wb") as f:
        f.write(b"MOSSDIT1")
        f.write(struct.pack("<qqq", args.height, args.width, args.text_len))
        f.write(struct.pack("<f", args.timestep))
        for tensor in (latent, text, out_flat, x_block0, prehead_x):
            f.write(tensor.numpy().astype("<f4").tobytes())

    print(f"wrote {args.out}: h={args.height} w={args.width} s={args.text_len} "
          f"dtype={args.dtype} out_rms={out_flat.pow(2).mean().sqrt().item():.6f} "
          f"prehead_rms={prehead_x.pow(2).mean().sqrt().item():.6f}")


if __name__ == "__main__":
    main()
