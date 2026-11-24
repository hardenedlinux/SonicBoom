# Output Decay (OD)

Every accelerated decode number is published with its Output Decay: how much
of the engine's own answer the acceleration changes.

## Definition

For one engine, one model file, one prompt set, greedy decoding (temperature
0) and a fixed max-token budget:

- **Reference:** the engine's output with acceleration **off**, from a fresh
  process.
- **Compared:** the same engine's output with acceleration **on**, from a
  fresh process.
- For each prompt *s*, `p_s` is the length in bytes of the longest common
  prefix of the compared and reference text, and `N_s` is the length in bytes
  of the reference text.
- **OD = 1 − p_s / N_s**, as a percentage.

OD 0% means the accelerated answer is byte-for-byte the answer without
acceleration. OD 30% means that, on average, the last 30% of the answer is no
longer what the model gives without acceleration: once two greedy streams
diverge, everything after the divergence point is a different answer, so the
shared prefix is what survives.

**Run-to-run OD** uses the same formula with both sides taken under the same
settings, from two fresh processes. A non-zero value means the engine is not
reproducible even without changing a setting.

The metric is measured per engine against itself, over bytes of text (not
token ids), so it is comparable across engines and tokenizers.

## Tool

[`tools/output-decay`](../tools/output-decay) reads two directories of
per-prompt outputs, pairs the files by name, and prints the per-prompt OD with
the first divergence offset, the overall OD, and a one-line `RESULT` summary.
It links nothing from SonicBoom, so it can be run on any engine's saved
outputs:

```sh
build/tools/output-decay/output-decay reference-dir compared-dir
```

A run-to-run OD is the same command with two same-setting runs as the two
directories.

## Prompt sets

- `codeedit` — [`tools/bench/codeedit`](../tools/bench/codeedit): the ten
  code-edit turns, with the turns' own budgets.
- `chat` — [`tools/bench/chat`](../tools/bench/chat): ten fixed general
  prompts (zh/en writing and Q&A), 512-token budget.
