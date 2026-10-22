#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""Build a small synthetic MOSS-Image DiT fixture and its f32 reference (S3).

The released checkpoint is bf16, so a full-size f32 reference needs 20 GB and
does not fit this machine. A small model with the same module structure does
fit, and it lets the parity gate compare two f32 forwards: any disagreement
there is an implementation bug, not bf16 noise.

Writes <out>/config.json, <out>/diffusion_pytorch_model-00001-of-00001.safetensors
(bf16, the dtype the real checkpoint uses) and a sharded index JSON, which is
what sb-moss dit loads; prints the path of the MOSSDIT1 reference it also
writes for those bf16-rounded weights run in f32.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch


def load_pipeline_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_moss_image"] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out", required=True, help="fixture directory")
    ap.add_argument("--ref", required=True, help="MOSSDIT1 reference path")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--ffn-dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--text-dim", type=int, default=128)
    ap.add_argument("--freq-dim", type=int, default=32)
    ap.add_argument("--latent-channels", type=int, default=16)
    ap.add_argument("--height", type=int, default=3)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--text-len", type=int, default=5)
    ap.add_argument("--timestep", type=float, default=437.0)
    args = ap.parse_args()

    mod = load_pipeline_module(args.pipeline)
    config = {
        "dim": args.dim,
        "ffn_dim": args.ffn_dim,
        "num_heads": args.heads,
        "num_layers": args.layers,
        "text_dim": args.text_dim,
        "freq_dim": args.freq_dim,
        "latent_channels": args.latent_channels,
        "rope_base": 10000.0,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261021)
    model = mod.MOSSImageTransformer(**config).eval()
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}

    from safetensors.torch import save_file

    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(state, str(out / shard), metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in state.values())},
        "weight_map": {k: shard for k in state},
    }
    (out / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # The reference forward: the same bf16-rounded weights, computed in f32.
    ref_model = mod.MOSSImageTransformer(**config)
    ref_model.load_training_state_dict({k: v.float() for k, v in state.items()})
    ref_model.to(torch.float32).eval()

    g = torch.Generator().manual_seed(424242)
    latent = torch.randn(1, args.latent_channels, args.height, args.width,
                         generator=g, dtype=torch.float32)
    text = torch.randn(1, args.text_len, args.text_dim, generator=g, dtype=torch.float32)

    captured = {}

    def make_hook(name, video_rows=None):
        def hook(_module, _inputs, output):
            value = output[:, :video_rows] if video_rows else output
            captured[name] = value.detach().float().clone()
        return hook

    handles = [
        ref_model.layers[0].register_forward_hook(make_hook("x_block0")),
        ref_model.layers[-1].register_forward_hook(
            make_hook("prehead", args.height * args.width)),
    ]
    with torch.no_grad():
        sample = ref_model(latent, args.timestep, text).sample
    for handle in handles:
        handle.remove()

    with open(args.ref, "wb") as f:
        f.write(b"MOSSDIT1")
        f.write(struct.pack("<qqq", args.height, args.width, args.text_len))
        f.write(struct.pack("<f", args.timestep))
        for tensor in (latent.reshape(-1), text.reshape(-1), sample.reshape(-1),
                       captured["x_block0"].reshape(-1), captured["prehead"].reshape(-1)):
            f.write(tensor.contiguous().numpy().astype("<f4").tobytes())

    print(f"fixture {out} ({len(state)} tensors, bf16) and reference {args.ref}: "
          f"out_rms={sample.pow(2).mean().sqrt().item():.6f}")


if __name__ == "__main__":
    main()
