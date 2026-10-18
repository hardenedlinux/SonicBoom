#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# pp-categories.sh — prefill kernel category table, SonicBoom vs llama.cpp.
#
# The comparison is pp512: one timed 512-token prefill pass on each side, with
# the kernels grouped by role (MMQ per quant type, attention, glue regions,
# norms, elementwise, dense GEMMs, quantize, copies, other), reporting GPU
# kernel time and launch count per category. This is the table the prefill
# parity work is measured against; it is not the canonical metric (that is
# tools/bench.sh).
#
# Method (the boom-day3 method):
#   - SonicBoom: prefill_bench --prompt 512 --reps 1 under nsys (cuda+nvtx,
#     --cuda-graph-trace=node, SONICBOOM_NVTX=1). prefill_bench runs, in
#     order, a 512-token warm-up pass, a 1-token warm-up, the timed 512-token
#     pass, then the timed 1-token pass. The warm-up 512 pass is the longest
#     (JIT, plan build, and first-use autotune all land in it), so the timed
#     512 pass is the second-longest `prefill:segment` range; the prompt-1
#     ranges are an order of magnitude shorter.
#   - llama.cpp: llama-bench -p 512 -n 0 -r 1 -ngl 99 under nsys. llama-bench
#     runs a warm-up pass and then the timed pass; the timed pass is the last
#     burst of >= 500 kernels with a > 20 ms gap before it.
#   - A kernel belongs to a window when its [start,end] lies inside it. Kernel
#     time is end-start summed per category; the wall time of the window is
#     also reported (kernel time / wall is the GPU-busy fraction).
#
# Usage: tools/pp-categories.sh [--gemma PATH] [--out DIR] [--no-run]
#   --gemma     model GGUF (default $ROOT/models/gemma-4-E4B-it-Q3_K_M.gguf)
#   --out       artifact directory (default /tmp/pp-cat)
#   --no-run    reuse existing nsys traces in --out (skip profiling)
# Env: LLAMA_BENCH=/path/to/llama-bench  (default ~/Project/llama.cpp/build-cuda/bin/llama-bench)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEMMA="$ROOT/models/gemma-4-E4B-it-Q3_K_M.gguf"
OUT=/tmp/pp-cat
RUN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --gemma) GEMMA="$2"; shift 2 ;;
    --out)   OUT="$2";   shift 2 ;;
    --no-run) RUN=0; shift ;;
    -h|--help) sed -n '3,34p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
GEMMA="$(readlink -f "$GEMMA")"
[ -f "$GEMMA" ] || { echo "model not found: $GEMMA" >&2; exit 2; }
LLAMA_BENCH="${LLAMA_BENCH:-$HOME/Project/llama.cpp/build-cuda/bin/llama-bench}"
mkdir -p "$OUT"

command -v nsys >/dev/null || { echo "nsys not found" >&2; exit 2; }
command -v sqlite3 >/dev/null || { echo "sqlite3 not found" >&2; exit 2; }

if [ "$RUN" = 1 ]; then
  [ -x "$LLAMA_BENCH" ] || { echo "llama-bench not executable: $LLAMA_BENCH" >&2; exit 2; }
  [ -x "$ROOT/tools/baseline/prefill_bench" ] || \
    bash "$ROOT/tools/baseline/build_prefill_bench.sh" >/dev/null

  echo "=== nsys: SonicBoom pp512 ===" >&2
  ( export LD_LIBRARY_PATH="$ROOT/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export SONICBOOM_GEMMA_GGUF="$GEMMA" SONICBOOM_NVTX=1
    nsys profile --force-overwrite true -o "$OUT/sb-pp512" \
      --trace=cuda,nvtx --cuda-graph-trace=node \
      "$ROOT/tools/baseline/prefill_bench" --prompt 512 --reps 1 \
      >"$OUT/sb-pp512.stdout" 2>&1 )
  grep -E "RESULT" "$OUT/sb-pp512.stdout" >&2 || true

  echo "=== nsys: llama.cpp pp512 ===" >&2
  nsys profile --force-overwrite true -o "$OUT/lb-pp512" \
    --trace=cuda --cuda-graph-trace=node \
    "$LLAMA_BENCH" -m "$GEMMA" -p 512 -n 0 -r 1 -ngl 99 \
    >"$OUT/lb-pp512.stdout" 2>&1
  grep -E "pp512" "$OUT/lb-pp512.stdout" | tail -1 >&2 || true
fi

for f in sb-pp512 lb-pp512; do
  [ -f "$OUT/$f.nsys-rep" ] || { echo "missing trace $OUT/$f.nsys-rep" >&2; exit 2; }
  [ -f "$OUT/$f.sqlite" ] || \
    nsys export --force-overwrite true --type sqlite -o "$OUT/$f.sqlite" \
      "$OUT/$f.nsys-rep" >/dev/null 2>&1
done

python3 - "$OUT" <<'PY'
import re, sqlite3, sys

out = sys.argv[1]

def window_sb(db):
    ev = db.execute("""select start,end from NVTX_EVENTS
                       where text='prefill:segment' order by start""").fetchall()
    if not ev:
        sys.exit("no prefill:segment NVTX ranges (is SONICBOOM_NVTX=1 set?)")
    dur = [e - s for s, e in ev]
    # The timed 512 pass is the second-longest range: the longest is the
    # warm-up 512 pass (cold JIT, plan build, first-use autotune), and the
    # prompt-1 passes are far shorter. A duration-ratio threshold is not
    # usable because the warm-up can be more than 2.5x the timed pass once
    # first-use autotune is included in it.
    order = sorted(range(len(dur)), key=lambda i: dur[i])
    i = order[-2] if len(order) >= 2 else order[-1]
    return ev[i][0], ev[i][1], len(ev)

def window_llama(db):
    rows = db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL "
                      "order by start").fetchall()
    bursts, cur = [], [rows[0]]
    for prev, nxt in zip(rows, rows[1:]):
        if nxt[0] - prev[1] > 20_000_000:
            bursts.append(cur); cur = []
        cur.append(nxt)
    bursts.append(cur)
    bursts = [b for b in bursts if len(b) >= 500]
    b = bursts[-1]
    return b[0][0], max(e for _, e in b), len(bursts)

LLAMA_T = {2: 'Q4_0', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K',
           11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K'}
SB_T = {0: 'Q3_K', 1: 'Q4_K', 2: 'Q5_K', 3: 'Q6_K'}

def classify(name, side):
    n = name
    if 'stream_k_fixup' in n: return 'MMQ fixup'
    if 'mul_mat_q' in n:
        t = None
        m = re.search(r'mul_mat_q<\(ggml_type\)(\d+)', n)
        if m: t = LLAMA_T.get(int(m.group(1)), 'type' + m.group(1))
        m = re.search(r'mul_mat_q<\(int\)(\d+)', n)
        if m: t = SB_T.get(int(m.group(1)), 'type' + m.group(1))
        return 'MMQ ' + (t or '?')
    if n.strip() == 'fa' or 'flash_attn' in n or 'attention' in n:
        return 'attention'
    # Generated glue regions first: several contain "norm"/"gelu" substrings.
    if any(k in n for k in ('gelu_mul_quantize', 'add_rms_norm_q8_1_mmq',
                            'scale_quantize_mmq', 'heads_norm_rope',
                            'gelu_mul', 'slice_scale')):
        return 'glue (generated)'
    if 'rms_norm' in n or 'rms_' in n or 'norm' in n:
        return 'norms'
    if ('gemm' in n or 'cutlass' in n or 'gemv' in n or 'gather_q' in n
            or 'mul_mat_vec_q' in n or 'splitKreduce' in n
            or 'k_set_rows' in n):
        return 'dense'
    if 'quantize' in n:
        return 'quantize'
    if 'cpy_' in n or 'memcpy' in n or 'copy' in n:
        return 'copies'
    if any(k in n for k in ('scale_kernel', 'cast_bf16_rows', 'kv_append',
                            'unary_gated_op', 'unary_op', 'k_bin_bcast',
                            'elementwise')):
        return 'elementwise'
    return 'other'

def table(path, w1, w2, side):
    db = sqlite3.connect(path)
    q = """select s.value, k.start, k.end
           from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on k.demangledName = s.id
           where k.start >= ? and k.end <= ?"""
    cat = {}
    tot_n = tot_ms = 0
    for name, s, e in db.execute(q, (w1, w2)):
        c = classify(name, side)
        d = cat.setdefault(c, [0, 0.0])
        d[0] += 1; d[1] += (e - s) / 1e6
        tot_n += 1; tot_ms += (e - s) / 1e6
    db.close()
    return cat, tot_n, tot_ms

sdb = sqlite3.connect(out + '/sb-pp512.sqlite')
sb_w1, sb_w2, nseg = window_sb(sdb)
ldb = sqlite3.connect(out + '/lb-pp512.sqlite')
lb_w1, lb_w2, nburst = window_llama(ldb)
sdb.close(); ldb.close()

sb, sbn, sbm = table(out + '/sb-pp512.sqlite', sb_w1, sb_w2, 'sb')
lb, lbn, lbm = table(out + '/lb-pp512.sqlite', lb_w1, lb_w2, 'lb')

order = ['MMQ Q3_K', 'MMQ Q4_K', 'MMQ Q5_K', 'MMQ Q6_K', 'MMQ fixup',
         'attention', 'glue (generated)', 'norms', 'elementwise', 'dense',
         'quantize', 'copies', 'other']
keys = [k for k in order if k in sb or k in lb]
for k in sorted(set(sb) | set(lb)):
    if k not in keys: keys.append(k)

print('# pp512 category table — SonicBoom vs llama.cpp')
print()
print(f'SB window:    NVTX prefill:segment {sb_w1}..{sb_w2} ns '
      f'({(sb_w2-sb_w1)/1e6:.2f} ms wall; {nseg} segment ranges)')
print(f'llama window: last kernel burst {lb_w1}..{lb_w2} ns '
      f'({(lb_w2-lb_w1)/1e6:.2f} ms wall; {nburst} bursts >= 500 kernels)')
print()
print('| category | SB ms | SB launches | llama ms | llama launches |')
print('|---|---|---|---|---|')
for k in keys:
    a = sb.get(k, [0, 0.0]); b = lb.get(k, [0, 0.0])
    print(f'| {k} | {a[1]:.2f} | {a[0]} | {b[1]:.2f} | {b[0]} |')
print(f'| **total** | **{sbm:.2f}** | **{sbn}** | **{lbm:.2f}** | **{lbn}** |')

def merged(cat):
    n = sum(v[0] for k, v in cat.items()
            if k in ('glue (generated)', 'norms', 'elementwise'))
    ms = sum(v[1] for k, v in cat.items()
             if k in ('glue (generated)', 'norms', 'elementwise'))
    return n, ms
sm, lm = merged(sb), merged(lb)
print()
print(f'glue+norm+elementwise: SB {sm[1]:.2f} ms / {sm[0]} launches, '
      f'llama {lm[1]:.2f} ms / {lm[0]} launches')
print(f'launch ratio: {sbn/lbn:.3f}x; kernel-time ratio: {sbm/lbm:.3f}x; '
      f'SB GPU-busy {(sbm/((sb_w2-sb_w1)/1e6))*100:.1f}%, '
      f'llama {(lbm/((lb_w2-lb_w1)/1e6))*100:.1f}%')
PY
