#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# bench-serve.sh — the server's streamed code-edit rate.
#
# The README quotes the OpenAI-compatible server's streamed code-edit rate;
# this script is the command that produces it. Ten frozen edit requests
# (tools/bench/codeedit) run through the server's streaming
# /v1/chat/completions (this tree has no raw /v1/completions) as ten separate
# requests, greedy, each generating the file's own token count. The token
# counts come from codeedit_bench, the same source tools/bench.sh --spec uses.
#
# Arms:
#   spec off   SONICBOOM_SPEC=off
#   default    SONICBOOM_SPEC unset (the startup probe decides)
#
# Each arm runs N independent fresh server processes (default 3). A rep is:
# start the server, wait for /health, run tools/bench/serve-codeedit.scm once
# over the ten turns, stop the server, confirm the process exited and the port
# was released. The rate is total generated tokens over the wall time from each
# request to its [DONE], summed over the ten turns; the arm's number is the
# median over its reps.
#
# The per-turn texts are compared with the off arm's first rep, turn by turn:
# "identical" counts how many of the ten turns produced the same text. The
# texts and each rep's log stay in the run directory this script prints.
#
# Usage: tools/bench-serve.sh <model.gguf> [--runs N] [--port P]
#                                         [--context-length C]
#        --runs N            fresh server processes per arm (default 3).
#        --port P            server port (default 18080).
#        --context-length C  server context (default: codeedit_bench's --ctx-only).
#
# A missing server dependency (guile, its Artanis/JSON modules, the server
# sources, libsonicboom.so, codeedit_bench, the code-edit manifest) is
# reported item by item and stops the run with a non-zero status before any
# measurement. A server that fails to start or a port that stays bound after
# shutdown also stops the run.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS=3
PORT=18080
CTX=""
GGUF=""

usage() {
  sed -n '/^# Usage:/,/^#$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --context-length) CTX="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "bench-serve: unknown option $1" >&2; usage ;;
    *)
      if [ -z "$GGUF" ]; then GGUF="$1"; else
        echo "bench-serve: unexpected argument $1" >&2; usage
      fi
      shift ;;
  esac
done

case "$RUNS" in
  ''|*[!0-9]*) echo "bench-serve: --runs must be a positive integer" >&2; exit 2 ;;
esac
[ "$RUNS" -ge 1 ] || { echo "bench-serve: --runs must be >= 1" >&2; exit 2; }
case "$PORT" in
  ''|*[!0-9]*) echo "bench-serve: --port must be a positive integer" >&2; exit 2 ;;
esac
if [ -n "$CTX" ]; then
  case "$CTX" in
    *[!0-9]*) echo "bench-serve: --context-length must be a positive integer" >&2
              exit 2 ;;
  esac
fi
[ -n "$GGUF" ] || usage
[ -f "$GGUF" ] || { echo "bench-serve: model not found: $GGUF" >&2; exit 2; }
GGUF="$(readlink -f "$GGUF")"

APP="$ROOT/bindings/guile/scheme/sonicboom/serve/app.scm"
CLIENT="$ROOT/tools/bench/serve-codeedit.scm"
DATA="$ROOT/tools/bench/codeedit"
LIB="$ROOT/build/core/libsonicboom.so"
CB="$ROOT/build/tools/bench/codeedit_bench"

# Every dependency is checked before anything is measured, and each missing
# item is named.
missing=0
need() { # WHAT PATH
  [ -e "$2" ] || { echo "bench-serve: missing $1: $2" >&2; missing=1; }
}
command -v guile >/dev/null 2>&1 ||
  { echo "bench-serve: missing guile on PATH" >&2; missing=1; }
if command -v guile >/dev/null 2>&1; then
  deps=$(guile -c '(use-modules (artanis artanis) (json))' 2>&1) || {
    echo "bench-serve: guile cannot load its server modules:" >&2
    echo "$deps" | sed 's/^/  /' >&2
    missing=1
  }
fi
need "the server app" "$APP"
need "the server client" "$CLIENT"
need "the code-edit manifest" "$DATA/manifest.tsv"
need "libsonicboom.so (build first)" "$LIB"
need "codeedit_bench (build first)" "$CB"
[ "$missing" = 0 ] || { echo "bench-serve: not starting any server" >&2; exit 2; }

MODEL_ID="$(basename "$GGUF" .gguf)"
WORK="$(mktemp -d /tmp/sb-serve.XXXXXX)"
mkdir -p "$WORK/spec"
GEN_FILE="$WORK/spec/gen_tokens.txt"

echo "bench-serve: generating the per-turn token budgets with codeedit_bench" >&2
if [ -z "$CTX" ]; then
  CTX="$("$CB" --model "$GGUF" --data "$DATA" --ctx-only | sed -n 's/^ctx=//p')"
  case "$CTX" in
    ''|*[!0-9]*) echo "bench-serve: codeedit_bench --ctx-only gave no ctx" >&2
                 exit 1 ;;
  esac
fi
SONICBOOM_SPEC=off "$CB" --model "$GGUF" --data "$DATA" --reps 1 >"$GEN_FILE"

# health: 0 when the server answers /health with 200.
# The status is decided after the catch, not with exit inside it: `exit` raises
# a quit exception that a #t handler catches, which would turn every 200 into
# the handler's failure code.
health() {
  guile --no-auto-compile -c "
    (use-modules (web client) (web response) (web uri))
    (define code
      (catch #t
        (lambda ()
          (call-with-values
            (lambda () (http-get (build-uri 'http #:host \"127.0.0.1\"
                                            #:port $PORT #:path \"/health\")))
            (lambda (resp body) (response-code resp))))
        (lambda _ 0)))
    (exit (if (= code 200) 0 1))" >/dev/null 2>&1
}

port_free() {
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    exec 3>&-
    return 1
  fi
  return 0
}

port_free || { echo "bench-serve: port $PORT is already in use" >&2; exit 2; }

# start_server ARM LOG -> pid on stdout
start_server() { # ARM LOG
  local arm="$1" log="$2"
  local spec_env=()
  [ "$arm" = off ] && spec_env=(SONICBOOM_SPEC=off)
  env -u SONICBOOM_SPEC "${spec_env[@]}" \
      LD_LIBRARY_PATH="$ROOT/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      guile -L "$ROOT/bindings/guile/scheme" "$APP" \
        --port "$PORT" --model "$GGUF" --model-id "$MODEL_ID" \
        --context-length "$CTX" --max-sessions 1 >"$log" 2>&1 &
  echo $!
}

# wait_ready PID LOG -> 0 when healthy, 1 when the process died or timed out
wait_ready() { # PID LOG
  local pid="$1" log="$2" waited=0
  while [ "$waited" -lt 1500 ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "bench-serve: server exited during start-up; log $log" >&2
      tail -20 "$log" >&2 || true
      return 1
    fi
    health && return 0
    sleep 0.2
    waited=$((waited + 1))
  done
  echo "bench-serve: server did not become healthy; log $log" >&2
  return 1
}

# stop_server PID -> 0 when the process is gone and the port is free
stop_server() { # PID
  local pid="$1" waited=0
  kill -TERM "$pid" 2>/dev/null || true
  while [ "$waited" -lt 300 ] && kill -0 "$pid" 2>/dev/null; do
    sleep 0.1
    waited=$((waited + 1))
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  waited=0
  while [ "$waited" -lt 300 ] && ! port_free; do
    sleep 0.1
    waited=$((waited + 1))
  done
  if ! port_free; then
    echo "bench-serve: port $PORT stayed bound after the server exited" >&2
    return 1
  fi
  return 0
}

ARMS=(off default)
for arm in "${ARMS[@]}"; do
  mkdir -p "$WORK/$arm"
  for ((rep = 1; rep <= RUNS; rep++)); do
    log="$WORK/$arm/rep$rep.server.log"
    outdir="$WORK/$arm/rep$rep"
    echo "bench-serve: arm $arm rep $rep/$RUNS" >&2
    pid="$(start_server "$arm" "$log")"
    if ! wait_ready "$pid" "$log"; then
      stop_server "$pid" || true
      exit 1
    fi
    if ! guile "$CLIENT" --port "$PORT" --model-id "$MODEL_ID" --data "$DATA" \
         --gen-tokens "$GEN_FILE" --arm "$arm" --out "$outdir" \
         >"$WORK/$arm/rep$rep.txt" 2>&1; then
      echo "bench-serve: the client failed; log $WORK/$arm/rep$rep.txt" >&2
      stop_server "$pid" || true
      exit 1
    fi
    stop_server "$pid" || exit 1
  done
done

# k=v field of a rep's RESULT line.
field() { # FILE KEY
  awk -v k="$2" '
    $1 == "RESULT" && $2 == "serve" {
      for (i = 3; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == k) { print a[2]; exit }
      }
    }' "$1"
}

# hash of turn NN in a rep's RESULT line.
turn_hash() { # FILE NN
  awk -v k="hash$2" '
    $1 == "RESULT" && $2 == "serve" {
      for (i = 3; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == k) { print a[2]; exit }
      }
    }' "$1"
}

median() {
  sort -n | awk '{a[NR]=$1} END{
    if(NR==0){print "nan"} else if(NR%2){print a[(NR+1)/2]} else {printf "%.3f", (a[NR/2]+a[NR/2+1])/2}}'
}

stdev() {
  sort -n | awk '{a[NR]=$1; s+=$1} END{
    if(NR<2){print "nan"; exit}
    m=s/NR; t=0
    for(i=1;i<=NR;i++) t+=(a[i]-m)^2
    printf "%.2f", sqrt(t/(NR-1))}'
}

echo
echo "SonicBoom server, streamed. Listed separately: not a ratio against llama.cpp."
echo
echo "model: $GGUF"
echo "server model id: $MODEL_ID, context: $CTX, port: $PORT"
echo "reps per arm: $RUNS (each a fresh server process)"
echo "run directory: $WORK"
echo
printf '%-10s %12s %12s %12s %10s %10s\n' \
  "arm" "tok/s med" "min" "max" "stdev" "identity"
for arm in "${ARMS[@]}"; do
  vals=""
  idmin=10
  for ((rep = 1; rep <= RUNS; rep++)); do
    f="$WORK/$arm/rep$rep.txt"
    v="$(field "$f" tok_s)"
    case "$v" in ''|*[!0-9.]*) continue ;; esac
    vals="$vals$v\n"
    same=0
    for ((t = 0; t < 10; t++)); do
      nn="$(printf '%02d' "$t")"
      ha="$(turn_hash "$f" "$nn")"
      hb="$(turn_hash "$WORK/off/rep1.txt" "$nn")"
      [ -n "$ha" ] && [ "$ha" = "$hb" ] && same=$((same + 1))
    done
    echo "rep $rep: tok_s=$v identical=$same/10" >&2
    [ "$same" -lt "$idmin" ] && idmin="$same"
  done
  med="$(printf '%b' "$vals" | median)"
  sd="$(printf '%b' "$vals" | stdev)"
  lo="$(printf '%b' "$vals" | sort -g | head -1)"
  hi="$(printf '%b' "$vals" | sort -g | tail -1)"
  case "$med" in
    ''|nan)
      printf '%-10s %12s %12s %12s %10s %10s\n' \
        "$arm" "n/a" "n/a" "n/a" "n/a" "n/a"
      printf 'RESULT bench_serve model=%s arm=%s reps=%s tok_s_median=nan tok_s_min=nan tok_s_max=nan tok_s_sd=nan identical_turns=n/a\n' \
        "$MODEL_ID" "$arm" "$RUNS"
      ;;
    *)
      printf '%-10s %12.1f %12.1f %12.1f %10s %10s\n' \
        "$arm" "$med" "$lo" "$hi" "$sd" "$idmin/10"
      printf 'RESULT bench_serve model=%s arm=%s reps=%s tok_s_median=%.1f tok_s_min=%.1f tok_s_max=%.1f tok_s_sd=%s identical_turns=%s/10\n' \
        "$MODEL_ID" "$arm" "$RUNS" "$med" "$lo" "$hi" "$sd" "$idmin"
      ;;
  esac
done
echo
echo "the identity reference is the off arm's first rep, turn by turn."
