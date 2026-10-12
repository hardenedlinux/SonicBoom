#!/usr/bin/env python3
# SonicBoom — inference service: the official openai Python SDK against the app
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# Run from the repository root, with the SDK installed (pip install openai):
#   python3 bindings/guile/tests/serve/test-openai-sdk.py
#
# Starts bindings/guile/scheme/sonicboom/serve/app.scm on a free port and
# checks that the SDK parses what it returns: non-streaming, streaming read
# chunk by chunk, tool calls (both forms, and the SDK's stream helper that
# accumulates them), usage with cached_tokens, and errors.

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import openai
from openai import OpenAI

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {info}"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


PORT = free_port()
LOG = tempfile.NamedTemporaryFile(prefix="sonicboom-serve-sdk-", suffix=".log", delete=False)
server = subprocess.Popen(
    ["guile", "-L", "bindings/guile/scheme",
     "bindings/guile/scheme/sonicboom/serve/app.scm", "--port", str(PORT),
     "--runtime-dir", os.path.join(tempfile.gettempdir(), f"sonicboom-serve-sdk-{os.getpid()}")],
    stdout=LOG, stderr=subprocess.STDOUT)


def wait_for_server():
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.25)
    return False


GEMMA = "sonicboom-mock-gemma"
QWEN = "sonicboom-mock-qwen"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "The weather of a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}
QUESTION = {"role": "user", "content": "What is the weather in Paris?"}


def run():
    check("server starts", wait_for_server())
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="unused", max_retries=0)

    # Models
    ids = [m.id for m in client.models.list()]
    check("models.list", ids == [GEMMA, QWEN], ids)
    check("models.retrieve", client.models.retrieve(QWEN).id == QWEN)

    # Non-streaming
    r = client.chat.completions.create(
        model=GEMMA, temperature=0,
        messages=[{"role": "system", "content": "You are the SDK test."},
                  {"role": "user", "content": "Hello from Python"}])
    check("non-stream: content", r.choices[0].message.content == "You said: Hello from Python",
          r.choices[0].message.content)
    check("non-stream: finish_reason", r.choices[0].finish_reason == "stop")
    check("non-stream: usage parsed",
          r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens)
    check("non-stream: cached_tokens parsed",
          isinstance(r.usage.prompt_tokens_details.cached_tokens, int))
    r2 = client.chat.completions.create(
        model=GEMMA, temperature=0,
        messages=[{"role": "system", "content": "You are the SDK test."},
                  {"role": "user", "content": "Another question"}])
    check("prefix reuse: cached_tokens > 0", r2.usage.prompt_tokens_details.cached_tokens > 0,
          r2.usage)

    r = client.chat.completions.create(model=GEMMA, temperature=0, max_tokens=4, stop=["xyz"],
                                       messages=[{"role": "user", "content": "Hi"}])
    check("max_tokens", (r.choices[0].message.content, r.choices[0].finish_reason)
          == ("You ", "length"))
    a = client.chat.completions.create(model=GEMMA, temperature=1.5, top_p=0.9, seed=5,
                                       messages=[{"role": "user", "content": "the lazy dog"}])
    b = client.chat.completions.create(model=GEMMA, temperature=1.5, top_p=0.9, seed=5,
                                       messages=[{"role": "user", "content": "the lazy dog"}])
    check("seed: reproducible", a.choices[0].message.content == b.choices[0].message.content)

    # Streaming, read chunk by chunk
    chunks = []
    with client.chat.completions.create(
            model=QWEN, temperature=0, stream=True, stream_options={"include_usage": True},
            messages=[{"role": "user", "content": "Stream to Python"}]) as stream:
        for chunk in stream:
            chunks.append(chunk)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    check("stream: chunks", len(chunks) > 10, len(chunks))
    check("stream: first delta role", chunks[0].choices[0].delta.role == "assistant")
    check("stream: content", text == "You said: Stream to Python", text)
    check("stream: finish_reason",
          [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop")
    check("stream: usage chunk", chunks[-1].usage is not None and chunks[-1].choices == []
          and chunks[-1].usage.completion_tokens == len("You said: Stream to Python") + 1)

    # Tool calls, non-streaming, then the result goes back
    r = client.chat.completions.create(model=QWEN, temperature=0, messages=[QUESTION],
                                       tools=[WEATHER])
    m = r.choices[0].message
    call = m.tool_calls[0] if m.tool_calls else None
    check("tools: finish_reason", r.choices[0].finish_reason == "tool_calls")
    check("tools: parsed call", call is not None and call.type == "function"
          and call.function.name == "get_weather"
          and json.loads(call.function.arguments) == {"city": "Paris"}, m)
    r2 = client.chat.completions.create(
        model=QWEN, temperature=0, tools=[WEATHER],
        messages=[QUESTION, m.model_dump(exclude_none=True),
                  {"role": "tool", "tool_call_id": call.id, "content": "sunny"}])
    check("tools: result read", r2.choices[0].message.content == "Tool result: sunny",
          r2.choices[0].message.content)
    check("tools: call turn cached",
          r2.usage.prompt_tokens_details.cached_tokens
          == r.usage.prompt_tokens + r.usage.completion_tokens, r2.usage)

    # Tool calls, streaming: deltas by hand
    calls = {}
    finish = None
    for chunk in client.chat.completions.create(model=GEMMA, temperature=0, stream=True,
                                                messages=[QUESTION], tools=[WEATHER]):
        if not chunk.choices:
            continue
        for d in chunk.choices[0].delta.tool_calls or []:
            c = calls.setdefault(d.index, {"id": None, "name": "", "arguments": ""})
            c["id"] = d.id or c["id"]
            if d.function:
                c["name"] += d.function.name or ""
                c["arguments"] += d.function.arguments or ""
        finish = chunk.choices[0].finish_reason or finish
    check("stream tools: deltas", calls.get(0, {}).get("name") == "get_weather"
          and json.loads(calls[0]["arguments"]) == {"city": "Paris"}
          and calls[0]["id"].startswith("call_"), calls)
    check("stream tools: finish_reason", finish == "tool_calls")

    # The SDK's stream helper accumulates the tool call itself
    with client.chat.completions.stream(model=QWEN, temperature=0, messages=[QUESTION],
                                        tools=[WEATHER]) as s:
        final = s.get_final_completion()
    fc = final.choices[0].message.tool_calls
    check("stream helper: final tool call", fc and fc[0].function.name == "get_weather"
          and json.loads(fc[0].function.arguments) == {"city": "Paris"}, final)

    # Errors map to the SDK's exceptions
    try:
        client.chat.completions.create(model=GEMMA, n=2,
                                       messages=[{"role": "user", "content": "x"}])
        check("n=2 -> BadRequestError", False)
    except openai.BadRequestError as e:
        check("n=2 -> BadRequestError", e.status_code == 400 and e.body.get("param") == "n",
              e.body)
    try:
        client.chat.completions.create(model="nope", messages=[{"role": "user", "content": "x"}])
        check("unknown model -> NotFoundError", False)
    except openai.NotFoundError as e:
        check("unknown model -> NotFoundError", e.body.get("code") == "model_not_found", e.body)


try:
    run()
finally:
    server.kill()
    server.wait()

print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
