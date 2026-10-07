#!/usr/bin/env python3
# cuBLAS (PyTorch) matmul microbenchmark at the Gemma 4 E4B-it matmul shapes,
# for the operator-level comparison against tools/baseline/gemv_bench (which
# times SonicBoom's dequantize-on-the-fly Q3_K gemv kernel in isolation).
#
# Measures pure device kernel time (CUDA events, warm-up + adaptive reps, no
# host/device copy in the timed region) for:
#   - gemv   n=1   y = W @ x        (the decode shape; memory-bound)
#   - gemm   n=32  Y = W @ X        (the prefill shape; compute-bound)
# in fp16 (tensor-core, best case) and fp32 (matches SonicBoom's f32-accumulate
# arithmetic). W is [rows, cols], x/X are [cols] / [cols, n], exactly the
# SonicBoom QuantizedTensor [cols, rows] matvec/matmul orientation.
#
# Run with the models/.venv python (torch 2.14.1+cu130, CUDA available):
#   models/.venv/bin/python tools/baseline/cublas_bench.py
import torch

torch.manual_seed(0)

# (name, rows, cols) — same as gemv_bench.cpp kShapes.
SHAPES = [
    ("ffn_gate_up", 10240, 2560),
    ("ffn_down",    2560, 10240),
    ("attn_q",      2048,  2560),
    ("attn_o",      2560,  2048),
    ("attn_kv",      512,  2560),
]


def bench(W, X):
    """Return steady-state us/op of torch.matmul(W, X) on the default stream."""
    for _ in range(5):
        torch.matmul(W, X)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.matmul(W, X)
    end.record()
    torch.cuda.synchronize()
    one_us = start.elapsed_time(end) * 1000.0

    reps = int(max(20, min(5000, 20000.0 / max(one_us, 0.5))))
    start.record()
    for _ in range(reps):
        torch.matmul(W, X)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def gflops(flops, us):
    return flops / us / 1e3  # flops/1e-6s / 1e9 = GFLOP/s


if not torch.cuda.is_available():
    print("cublas_bench SKIP (no CUDA device)")
    raise SystemExit(0)

print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"dev {torch.cuda.get_device_name(0)}")
print(f"{'shape':14s} {'dtype':>8s} | {'gemv n=1':>18s} | {'gemm n=32':>18s}")

for name, rows, cols in SHAPES:
    for dt in (torch.float16, torch.float32):
        W = torch.randn(rows, cols, device="cuda", dtype=dt)
        x = torch.randn(cols, device="cuda", dtype=dt)
        X = torch.randn(cols, 32, device="cuda", dtype=dt)

        us1 = bench(W, x)
        us32 = bench(W, X)
        flops1 = 2 * rows * cols
        flops32 = flops1 * 32
        dts = "fp16" if dt == torch.float16 else "fp32"
        print(f"{name:14s} {dts:>8s} | {us1:9.2f}us {gflops(flops1, us1):7.1f}GF "
              f"| {us32:9.2f}us {gflops(flops32, us32):7.1f}GF")
