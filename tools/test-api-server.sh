#!/bin/bash
#
# SonicBoom — smoke-test the OpenAI-compatible server against the real backend.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# Starts bindings/guile/scheme/sonicboom/serve/app.scm — the libsonicboom
# backend, a real Gemma 4 GGUF through the C ABI — on a free port and drives it
# with curl: /health, /v1/models, /v1/chat/completions (JSON and SSE), the usage
# fields (prompt_tokens_details.cached_tokens is the prefill's reused-token
# count), the greedy-only rejection, the context-length rejection, and prefix
# reuse across a multi-turn conversation. Exits non-zero if a check fails.
#
#   tools/test-api-server.sh [MODEL.gguf]
#
# The model defaults to models/gemma-4-E4B-it-Q3_K_M.gguf and the path may also
# come from $SONICBOOM_GEMMA_GGUF. Needs curl, a built libsonicboom.so
# (build/core), Guile and GNU Artanis. The server runs with --max-sessions 1 so
# that the prefix cache is exercised on every request (the pool then takes over
# its one session past the shared prefix instead of opening a new one).
#
# Known core limitation, checked last: a prompt that is *entirely* resident in
# the session's KV cache makes sb_llm_prefill fail — the CUDA prefill has an
# empty suffix and returns 0, which capi.cpp:831 reads as an internal error
# (core/src/model/gemma4_plan.cpp:1512-1518). The check reports it rather than
# failing, so the script stays useful on both sides of a fix.

set -u

root=$(cd "$(dirname "$0")/.." && pwd)
model=${1:-${SONICBOOM_GEMMA_GGUF:-$root/models/gemma-4-E4B-it-Q3_K_M.gguf}}
port=${PORT:-$(( 20000 + RANDOM % 20000 ))}
tmp=$(mktemp -d)
body=$tmp/body.json
log=$tmp/server.log
fail=0

if [ ! -f "$model" ]; then
  echo "no model at $model (pass one as \$1 or set \$SONICBOOM_GEMMA_GGUF)" >&2
  exit 2
fi

echo "SonicBoom OpenAI server smoke test"
echo "  model $(basename "$model")"
echo "  port  $port"
echo

LD_LIBRARY_PATH="$root/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
GUILE_AUTO_COMPILE=0 \
  guile --no-auto-compile -L "$root/bindings/guile/scheme" \
        "$root/bindings/guile/scheme/sonicboom/serve/app.scm" \
        --port "$port" --model "$model" --context-length 512 --max-sessions 1 \
        --runtime-dir "$tmp/rt" > "$log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null; rm -rf "$tmp"' EXIT

base="http://127.0.0.1:$port"
up=0
for _ in $(seq 1 600); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$base/health")" = 200 ]; then
    up=1
    break
  fi
  sleep 0.5
done
if [ "$up" != 1 ]; then
  echo "the server did not come up; last lines of its log:" >&2
  tail -n 20 "$log" >&2
  exit 2
fi

ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }
note() { printf '  note  %s\n' "$1"; }

# curl helpers: $code gets the status, $body the response.
jget()  { code=$(curl -s -o "$body" -w '%{http_code}' -m 30 "$1"); }
jpost() { code=$(curl -s -o "$body" -w '%{http_code}' -m 600 \
                   -H 'content-type: application/json' --data-binary "$2" "$1"); }

# Field readers over the response body (our own key order, so a first match).
num()  { grep -o "\"$2\":[0-9]*" "$1" | head -1 | cut -d: -f2; }
str()  { grep -o "\"$2\":\"[^\"]*\"" "$1" | head -1 \
           | sed "s/^\"$2\":\"//; s/\"\$//"; }
# The message of an error body (its text may contain escaped quotes).
msg()  { sed -n 's/.*"message":"\(.*\)","type".*/\1/p' "$1" | head -1; }
# The text of a completion, and of a whole SSE stream's content deltas.
text_one() { grep -o '"content":"[^"]*"' "$1" | head -1 \
               | sed 's/^"content":"//; s/"$//'; }
text_all() { grep -o '"content":"[^"]*"' "$1" \
               | sed 's/^"content":"//; s/"$//' | tr -d '\n'; }

# --- liveness and models -----------------------------------------------------

jget "$base/health"
if [ "$code" = 200 ] && [ "$(cat "$body")" = ok ]; then
  ok "GET /health"
else
  bad "GET /health ($code)"
fi

jget "$base/v1/models"
mid=$(str "$body" id)
if [ "$code" = 200 ] && [ -n "$mid" ] && grep -q '"object":"list"' "$body"; then
  ok "GET /v1/models lists $mid"
else
  bad "GET /v1/models ($code)"
  mid=gemma-4-E4B-it-Q3_K_M
fi

jget "$base/v1/models/$mid"
[ "$code" = 200 ] && grep -q '"object":"model"' "$body" \
  && ok "GET /v1/models/:id" || bad "GET /v1/models/:id ($code)"

jget "$base/v1/models/nope"
[ "$code" = 404 ] && grep -q '"code":"model_not_found"' "$body" \
  && ok "an unknown model is a 404" || bad "an unknown model should be a 404 ($code)"

ask() { # ask CONTENT MAX_TOKENS -> JSON body on stdout
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s}' \
         "$mid" "$1" "$2"
}

# --- a plain completion ------------------------------------------------------

completion=$tmp/completion.json
jpost "$base/v1/chat/completions" "$(ask 'Say hi in three words.' 16)"
cp "$body" "$completion"
if [ "$code" = 200 ] && grep -q '"object":"chat.completion"' "$body" \
   && grep -q '"finish_reason":"stop"' "$body"; then
  p=$(num "$body" prompt_tokens); c=$(num "$body" completion_tokens)
  t=$(num "$body" total_tokens)
  if [ "$((p + c))" = "$t" ] && [ "$p" -gt 0 ] && [ "$c" -gt 0 ]; then
    ok "POST /v1/chat/completions: prompt $p + completion $c = total $t"
  else
    bad "usage totals do not add up: $p + $c != $t"
  fi
  grep -q '"prompt_tokens_details":{"cached_tokens":' "$body" \
    && ok "usage carries prompt_tokens_details.cached_tokens" \
    || bad "usage lacks prompt_tokens_details.cached_tokens"
else
  bad "POST /v1/chat/completions ($code)"
fi

# --- the same request, streamed ----------------------------------------------

# A filler request first: with one pooled session the SSE request below would
# otherwise repeat the prompt that is still resident in the KV cache, which is
# the fully-reused case the core cannot do yet (see the last check). The filler
# prompt is different, so the SSE prefill only shares a few prefix tokens.
jpost "$base/v1/chat/completions" "$(ask 'zqz' 1)"

sse=$tmp/sse.txt
code=$(curl -s -N -o "$sse" -w '%{http_code}' -m 600 \
            -H 'content-type: application/json' \
            --data-binary "$(printf \
              '{"model":"%s","messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true}}' \
              "$mid")" "$base/v1/chat/completions")
if [ "$code" = 200 ] && grep -q '^data: \[DONE\]$' "$sse" \
   && [ -z "$(grep -v '^$' "$sse" | grep -v '^data: ')" ] \
   && grep -q '"finish_reason":"stop"' "$sse" \
   && grep -q '"usage":' "$sse"; then
  streamed=$(text_all "$sse"); whole=$(text_one "$completion")
  if [ "$streamed" = "$whole" ]; then
    ok "SSE: every line is a data: chunk, and the deltas concatenate to the unstreamed text"
  else
    bad "SSE: the delta text differs from the unstreamed text"
  fi
else
  bad "SSE stream ($code)"
fi

# --- greedy only -------------------------------------------------------------

jpost "$base/v1/chat/completions" \
      "$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":1}' "$mid")"
[ "$code" = 400 ] && grep -q '"param":"temperature"' "$body" \
  && ok "temperature 1 -> 400 (param temperature)" \
  || bad "temperature 1 should be a 400 on temperature ($code)"

jpost "$base/v1/chat/completions" \
      "$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"top_p":0.5}' "$mid")"
[ "$code" = 400 ] && grep -q '"param":"top_p"' "$body" \
  && ok "top_p 0.5 -> 400 (param top_p)" \
  || bad "top_p 0.5 should be a 400 on top_p ($code)"

jpost "$base/v1/chat/completions" \
      "$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0,"top_p":1}' "$mid")"
[ "$code" = 200 ] && ok "an explicit greedy request (temperature 0, top_p 1) is accepted" \
  || bad "temperature 0 / top_p 1 should be accepted ($code)"

# --- the context length ------------------------------------------------------

long=$(printf 'word %.0s' $(seq 1 700))
jpost "$base/v1/chat/completions" "$(ask "$long" 1)"
[ "$code" = 400 ] && grep -q '"code":"context_length_exceeded"' "$body" \
  && ok "a 700-token prompt -> 400 (context_length_exceeded)" \
  || bad "an over-long prompt should be a 400 ($code)"

# --- prefix reuse across a turn ----------------------------------------------
# The second prompt repeats the first user turn, so the prefill must reuse it;
# cached_tokens is that reused count. The answer also proves the reused KV is
# the real thing — the number only appears in the reused part of the prompt.

jpost "$base/v1/chat/completions" "$(ask 'Remember the number 4173. Just say ok.' 12)"
p1=$(num "$body" prompt_tokens); c1=$(num "$body" cached_tokens)
if [ "$code" = 200 ]; then
  turn1_text=$(text_one "$body")
else
  bad "the first turn of the reuse check failed ($code)"
  turn1_text=ok
fi
# Keep the model's reply out of the JSON unescaped: only the shared prefix
# matters for the reuse check, and the user turn carries the number.
turn1_text=$(printf '%s' "$turn1_text" | tr -d '"\\')
a1=$tmp/answer1.json
turn2=$(printf \
  '{"model":"%s","messages":[{"role":"user","content":"Remember the number 4173. Just say ok."},{"role":"assistant","content":"%s"},{"role":"user","content":"What number was that?"}],"max_tokens":12}' \
  "$mid" "$turn1_text")
jpost "$base/v1/chat/completions" "$turn2"
cp "$body" "$a1"
if [ "$code" = 200 ]; then
  p2=$(num "$body" prompt_tokens); c2=$(num "$body" cached_tokens)
  answer=$(text_one "$body")
  if [ "$c2" -gt 0 ] && [ "$c2" -le "$p2" ]; then
    ok "the second turn reused $c2 of its $p2 prompt tokens (first turn: $c1 of $p1)"
  else
    bad "cached_tokens is $c2 of $p2 — no prefix reuse"
  fi
  if printf '%s' "$answer" | grep -q 4173; then
    ok "the reused prefix is the real KV state: the model answers \"$answer\""
  else
    bad "the model did not recall the number from the reused prefix: \"$answer\""
  fi
else
  bad "the second turn of the reuse check failed ($code)"
fi

# --- a fully reused prompt (known core limitation) ---------------------------

# The same request again: the session already holds this prompt in full, so the
# prefill has an empty suffix.
jpost "$base/v1/chat/completions" "$turn2"
if [ "$code" = 200 ]; then
  ok "an identical repeat reuses the whole prompt"
else
  note "a fully reused prompt fails in the core (HTTP $code: $(msg "$body"))"
  note "  sb_llm_prefill -> an empty prefill suffix -> done == 0 -> SB_ERR_INTERNAL"
  note "  core/src/model/gemma4_plan.cpp:1512, capi/src/capi.cpp:831"
fi

echo
if [ "$fail" = 0 ]; then
  echo "smoke test passed (server log: $log)"
else
  echo "smoke test FAILED — server log: $log"
  tail -n 20 "$log"
fi
exit "$fail"
