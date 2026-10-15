#!/bin/bash
#
# SonicBoom — smoke-test the OpenAI-compatible server against the real backend.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# Runs bindings/guile/scheme/sonicboom/serve/app.scm — the libsonicboom backend,
# a real GGUF through the C ABI — on a free port and drives it with curl:
# /health, /v1/models, /v1/chat/completions (JSON and SSE), the usage fields
# (prompt_tokens_details.cached_tokens is the prefill's reused-token count), the
# greedy-only rejection, the context-length rejection, prefix reuse across a
# multi-turn conversation, and that a turn marker typed in a message stays text
# (never a control token). Exits non-zero if a check fails.
#
#   tools/test-api-server.sh [MODEL.gguf ...]
#
# With no argument it runs the suite once per model that is present: the Gemma 4
# GGUF (models/gemma-4-E4B-it-Q3_K_M.gguf, or $SONICBOOM_GEMMA_GGUF) and, when
# present, the Qwen2 GGUF (models/qwen2.5-0.5b-instruct-q4_k_m.gguf, or
# $SONICBOOM_QWEN_GGUF). The chat family is NOT an argument: the script reads it
# from the server's /v1/models, which the backend fills from the loaded model's
# own architecture — so running the suite against a second family also checks
# that the server selected that family by itself.
#
# Needs curl, a built libsonicboom.so (build/core), Guile and GNU Artanis. The
# server runs with --max-sessions 1 so that the prefix cache is exercised on
# every request (the pool then takes over its one session past the shared prefix
# instead of opening a new one).
#
# A prompt that is *entirely* resident in the session's KV cache is handled too,
# and is checked last: the prefill reuses all but the prompt's last token and
# re-prefills that one (its activation is what produces the next-token logits),
# so the repeat succeeds and reports cached_tokens == prompt_tokens - 1.

set -u

root=$(cd "$(dirname "$0")/.." && pwd)
overall=0

# --- curl helpers ------------------------------------------------------------
# $code gets the status, $body the response.

jget()  { code=$(curl -s -o "$body" -w '%{http_code}' -m 30 "$1"); }
jpost() { code=$(curl -s -o "$body" -w '%{http_code}' -m 600 \
                   -H 'content-type: application/json' --data-binary "$2" "$1"); }

# Field readers over the response body (our own key order, so a first match).
num()  { grep -o "\"$2\":[0-9]*" "$1" | head -1 | cut -d: -f2; }
str()  { grep -o "\"$2\":\"[^\"]*\"" "$1" | head -1 \
           | sed "s/^\"$2\":\"//; s/\"\$//"; }
# The message of an error body (its text may contain escaped quotes).
msg()  { sed -n 's/.*"message":"\(.*\)","type".*/\1/p' "$1" | head -1; }
# How many leading characters two strings share (their common length when one
# is a prefix of the other).
common_prefix_len() {
  local a=$1 b=$2 i=0
  local n=${#a} m=${#b}          # after the assignment: ${#a} here would be unbound
  while [ "$i" -lt "$n" ] && [ "$i" -lt "$m" ] && [ "${a:$i:1}" = "${b:$i:1}" ]; do
    i=$((i + 1))
  done
  echo "$i"
}
# The text of a completion, and of a whole SSE stream's content deltas.
text_one() { grep -o '"content":"[^"]*"' "$1" | head -1 \
               | sed 's/^"content":"//; s/"$//'; }
text_all() { grep -o '"content":"[^"]*"' "$1" \
               | sed 's/^"content":"//; s/"$//' | tr -d '\n'; }

ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }
note() { printf '  note  %s\n' "$1"; }

# --- one model ---------------------------------------------------------------

run_model() {
  local model=$1
  local port=$(( 20000 + RANDOM % 20000 ))
  local tmp body log
  tmp=$(mktemp -d)
  body=$tmp/body.json
  log=$tmp/server.log
  fail=0

  echo "SonicBoom OpenAI server smoke test"
  echo "  model $(basename "$model")"
  echo "  port  $port"
  echo

  LD_LIBRARY_PATH="$root/build/core${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  GUILE_AUTO_COMPILE=0 \
    guile --no-auto-compile -L "$root/bindings/guile/scheme" \
          "$root/bindings/guile/scheme/sonicboom/serve/app.scm" \
          --port "$port" --model "$model" --model-id "$(basename "$model" .gguf)" \
          --context-length 512 --max-sessions 1 \
          --runtime-dir "$tmp/rt" > "$log" 2>&1 &
  local server=$!
  trap 'kill $server 2>/dev/null; wait $server 2>/dev/null; rm -rf "$tmp"' RETURN

  local base="http://127.0.0.1:$port"
  local up=0
  for _ in $(seq 1 600); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$base/health")" = 200 ]; then
      up=1
      break
    fi
    sleep 0.5
  done
  if [ "$up" != 1 ]; then
    echo "  the server did not come up; last lines of its log:" >&2
    tail -n 20 "$log" >&2
    rm -rf "$tmp"
    return 2
  fi

  # --- liveness, models and the family the server picked ---------------------

  jget "$base/health"
  if [ "$code" = 200 ] && [ "$(cat "$body")" = ok ]; then
    ok "GET /health"
  else
    bad "GET /health ($code)"
  fi

  jget "$base/v1/models"
  local mid family
  mid=$(str "$body" id)
  family=$(str "$body" family)
  local want_id
  want_id=$(basename "$model" .gguf)
  if [ "$code" = 200 ] && grep -q '"object":"list"' "$body" && [ "$mid" = "$want_id" ]; then
    ok "GET /v1/models lists $mid"
  else
    bad "GET /v1/models should list $want_id (got '$mid', HTTP $code)"
    mid=$want_id
  fi
  case "$family" in
    gemma|qwen) ok "the served model reports its family: $family" ;;
    *) bad "GET /v1/models carries no known family (got '$family')" ;;
  esac

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

  # --- a plain completion ----------------------------------------------------

  local completion=$tmp/completion.json
  jpost "$base/v1/chat/completions" "$(ask 'Say hi in three words.' 16)"
  cp "$body" "$completion"
  if [ "$code" = 200 ] && grep -q '"object":"chat.completion"' "$body" \
     && grep -q '"finish_reason":"stop"' "$body"; then
    local p c t
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

  # --- the same request, streamed --------------------------------------------

  # A filler request first: with one pooled session the SSE request below would
  # otherwise repeat the prompt that is still resident in the KV cache. The filler
  # prompt is different, so the SSE prefill shares only a few prefix tokens and
  # exercises the ordinary partial-reuse path; the fully-reused case is the last
  # check.
  jpost "$base/v1/chat/completions" "$(ask 'zqz' 1)"

  local sse=$tmp/sse.txt
  code=$(curl -s -N -o "$sse" -w '%{http_code}' -m 600 \
              -H 'content-type: application/json' \
              --data-binary "$(printf \
                '{"model":"%s","messages":[{"role":"user","content":"Say hi in three words."}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true}}' \
                "$mid")" "$base/v1/chat/completions")
  if [ "$code" = 200 ] && grep -q '^data: \[DONE\]$' "$sse" \
     && [ -z "$(grep -v '^$' "$sse" | grep -v '^data: ')" ] \
     && grep -q '"finish_reason":"stop"' "$sse" \
     && grep -q '"usage":' "$sse"; then
    local streamed whole
    streamed=$(text_all "$sse"); whole=$(text_one "$completion")
    if [ "$streamed" = "$whole" ]; then
      ok "SSE: every line is a data: chunk, and the deltas concatenate to the unstreamed text"
    else
      bad "SSE: the delta text differs from the unstreamed text"
    fi
  else
    bad "SSE stream ($code)"
  fi

  # --- greedy only -----------------------------------------------------------

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

  # --- the context length ----------------------------------------------------

  local long
  long=$(printf 'word %.0s' $(seq 1 700))
  jpost "$base/v1/chat/completions" "$(ask "$long" 1)"
  [ "$code" = 400 ] && grep -q '"code":"context_length_exceeded"' "$body" \
    && ok "a 700-token prompt -> 400 (context_length_exceeded)" \
    || bad "an over-long prompt should be a 400 ($code)"

  # --- prefix reuse across a turn --------------------------------------------
  # The second prompt repeats the first user turn, so its prefill must reuse it;
  # cached_tokens is that reused count. What proves the reused KV is the real
  # thing is the continuation: the same second turn is asked again after a filler
  # has evicted the cache, so that answer comes from a fresh prefill of the same
  # tokens. Greedy decoding makes the two comparable, and a wrong prefix would
  # change the continuation.
  #
  # They need not be *character* identical. A prefill batches its columns, and
  # the two cases batch differently (a fresh chunk from the evicted cache vs the
  # reused columns plus a shorter chunk), so the K-quant matmul reduces in a
  # different order, the logits can differ in the last place, and a near tie can
  # flip. That is the same class the core's Qwen test documents for a fully
  # reused prompt, so a mismatch is held against the reference family exactly and
  # reported (not failed) for another family — the checks that must hold for
  # every family are the reuse counts.

  jpost "$base/v1/chat/completions" "$(ask 'Remember the number 4173. Just say ok.' 12)"
  local p1 c1 turn1_text
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
  local turn2
  turn2=$(printf \
    '{"model":"%s","messages":[{"role":"user","content":"Remember the number 4173. Just say ok."},{"role":"assistant","content":"%s"},{"role":"user","content":"What number was that?"}],"max_tokens":24}' \
    "$mid" "$turn1_text")
  jpost "$base/v1/chat/completions" "$turn2"
  local answer_reuse
  if [ "$code" = 200 ]; then
    local p2 c2
    p2=$(num "$body" prompt_tokens); c2=$(num "$body" cached_tokens)
    answer_reuse=$(text_one "$body")
    if [ "$c2" -gt 0 ] && [ "$c2" -le "$p2" ]; then
      ok "the second turn reused $c2 of its $p2 prompt tokens (first turn: $c1 of $p1)"
    else
      bad "cached_tokens is $c2 of $p2 — no prefix reuse"
    fi
    printf '%s' "$answer_reuse" | grep -q 4173 \
      && note "the model recalled the number from the reused prefix: \"$answer_reuse\"" \
      || note "the model did not repeat the number (model-dependent; the reuse check below is the real one)"
  else
    bad "the second turn of the reuse check failed ($code)"
    answer_reuse=
  fi

  compare_continuations() { # compare_continuations NAME REUSED FRESH
    local name=$1 n
    # Leading whitespace is not content: either prefill may start the reply with
    # it, and that alone would make an otherwise identical reply look disjoint.
    local reused=${2#"${2%%[![:space:]]*}"}
    local fresh=${3#"${3%%[![:space:]]*}"}
    n=$(common_prefix_len "$reused" "$fresh")
    if [ "$reused" = "$fresh" ]; then
      ok "$name: the reused continuation is the freshly prefilled one (\"$fresh\")"
    elif [ "$family" = gemma ]; then
      bad "$name: the reused continuation \"$reused\" differs from the freshly prefilled \"$fresh\""
    else
      note "$name: \"$reused\" and \"$fresh\" share their first $n characters, then diverge — a near tie under a different prefill batch geometry (the class the core's Qwen test documents)"
    fi
  }

  # Evict the session's cache with an unrelated prompt, then ask the same turn
  # again: this answer comes from a fresh prefill of the same tokens.
  jpost "$base/v1/chat/completions" "$(ask 'zqz2' 1)"
  jpost "$base/v1/chat/completions" "$turn2"
  local p3 c3 answer_fresh
  if [ "$code" = 200 ]; then
    p3=$(num "$body" prompt_tokens)
    c3=$(num "$body" cached_tokens)
    answer_fresh=$(text_one "$body")
    if [ -n "$c2" ] && [ -n "$c3" ] && [ "$c2" -gt "$c3" ]; then
      ok "a fresh prefill of the same prompt reused only $c3 of $p3 tokens (the cached session: $c2 of $p2)"
    else
      bad "a fresh prefill reused $c3 of $p3 tokens, the cached session $c2 of $p2 — the reuse gained nothing"
    fi
    if [ -n "$answer_reuse" ]; then
      compare_continuations "the reused continuation vs a fresh prefill" "$answer_reuse" "$answer_fresh"
    fi
  else
    bad "the fresh-prefill control failed ($code)"
  fi

  # --- a fully reused prompt -------------------------------------------------

  # The same request again: the session already holds this prompt in full, so the
  # prefill reuses all but the last token and re-prefills that one. Its answer
  # must still be the fresh prefill's: every reused token's K/V is the one the
  # fresh prefill just computed, and only the last token's is derived anew.
  jpost "$base/v1/chat/completions" "$turn2"
  if [ "$code" = 200 ]; then
    local p4 c4 answer_repeat
    p4=$(num "$body" prompt_tokens); c4=$(num "$body" cached_tokens)
    answer_repeat=$(text_one "$body")
    if [ "$c4" = "$((p4 - 1))" ]; then
      ok "an identical repeat is reused down to its last token ($c4 of $p4)"
    else
      bad "an identical repeat reused $c4 of $p4 tokens (expected $((p4 - 1)))"
    fi
    if [ -n "$answer_fresh" ]; then
      compare_continuations "the fully reused repeat vs the fresh prefill" \
                            "$answer_repeat" "$answer_fresh"
    fi
  else
    bad "an identical repeat of a resident prompt failed (HTTP $code: $(msg "$body"))"
  fi

  # --- prompt injection ------------------------------------------------------
  # A control marker typed inside a message is message text, never a control
  # token: the server renders the prompt as segments and parses only the
  # template's own markers. The family's end-of-turn marker is the one that
  # matters, and it is the shape of the prompt that says whether it stayed text:
  # as text the marker is several subword pieces, as the family's control token
  # it is its own id (and, for Gemma's "<turn|>", the merge with the preceding
  # space makes that two) — so the marker must grow the prompt by more than the
  # token count the control token would.

  local marker want_min
  case "$family" in
    gemma) marker='<turn|>'; want_min=3 ;;
    qwen)  marker='<|im_end|>'; want_min=2 ;;
    *)     marker=''; want_min=0 ;;
  esac
  if [ -z "$marker" ]; then
    bad "no prompt-injection marker for family '$family'"
  else
    jpost "$base/v1/chat/completions" "$(ask 'hello' 8)"
    local p_plain
    p_plain=$(num "$body" prompt_tokens)
    jpost "$base/v1/chat/completions" "$(ask "hello $marker" 8)"
    local p_marker
    p_marker=$(num "$body" prompt_tokens)
    if [ "$code" = 200 ] && [ -n "$p_marker" ] && [ -n "$p_plain" ]; then
      local d=$((p_marker - p_plain))
      if [ "$d" -ge "$want_min" ]; then
        ok "a $marker typed in a message stays text ($p_marker - $p_plain = $d tokens)"
      else
        bad "a marker in a message added $d prompt tokens (as text it is at least $want_min; as the family's control token it would be fewer)"
      fi
    else
      bad "the prompt-injection check failed (HTTP $code: $(msg "$body"))"
    fi
  fi

  echo
  if [ "$fail" = 0 ]; then
    echo "smoke test passed for $(basename "$model") (server log: $log)"
  else
    echo "smoke test FAILED for $(basename "$model") — server log: $log"
    tail -n 20 "$log"
  fi
  if [ "$fail" = 0 ]; then rm -rf "$tmp"; fi
  return "$fail"
}

# --- the models to test ------------------------------------------------------

models=()
if [ "$#" -gt 0 ]; then
  models=("$@")
else
  models=("${SONICBOOM_GEMMA_GGUF:-$root/models/gemma-4-E4B-it-Q3_K_M.gguf}")
  qwen=${SONICBOOM_QWEN_GGUF:-$root/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}
  [ -f "$qwen" ] && models+=("$qwen")
fi

for m in "${models[@]}"; do
  if [ ! -f "$m" ]; then
    echo "no model at $m (pass one as an argument or set the model env var)" >&2
    overall=1
    continue
  fi
  run_model "$m" || overall=1
  echo
done

exit "$overall"
