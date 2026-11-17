# Reproducibility

SonicBoom gives the same output for the same input when three things are the
same: the SonicBoom version, the machine and its GPU, and the warm-up table.

## What is covered

- Greedy decoding, and sampling with the same seed and settings.
- Speculative decoding on or off: the token sequences are bit-identical.
- Prefill segmentation: the last rows of a prompt run as their own segment
  (`SONICBOOM_PREFILL_SPLIT_LAST`, on by default). The rows held back are the
  smallest segment the batched kernels take, so every prompt row is computed by
  the same kernels as in the unsplit segment; the first-token logits and the
  greedy tokens are bitwise identical either way. The split is a static
  function of the model and the prompt length, so with the same version, the
  same machine and the same input the output is still the same every run.
- Continuing a session, as described under [Continuing a session](#continuing-a-session).

## The warm-up table

When a model loads, SonicBoom measures several kernel variants on your GPU and
records which one to use. Some of these choices change the order of
floating-point operations, so a different choice can give different output.
Once a choice is recorded, every later run reuses it, and the output stays
the same.

The table is measured again, and the output can change, when:

- the table is deleted;
- SonicBoom is upgraded (each version keeps its own records);
- SonicBoom starts with an empty cache directory, for example in a container
  without a persistent volume.

Loading a new model adds records for that model only; existing records are
not touched. Records that only affect speed are not saved if the machine was
busy while they were measured; they are measured again on a later run and
never change the output.

The table lives under `~/.cache/sonicboom/tune` (or
`$XDG_CACHE_HOME/sonicboom/tune`), or under `$SONICBOOM_TUNE_DIR` when that is
set.

## Keeping the output fixed

- **Keep the cache directory.** In a container, mount it on a persistent volume.
- **Export and import the table.** `sb-tune export --out FILE` writes it to a
  file; `sb-tune import --in FILE` installs it, on the same machine or on
  another machine with the same GPU.
- **Canonical numerics.** `SONICBOOM_TUNE=canonical`, or loading a model with
  `BOOM_LLM_LOAD_CANONICAL`, uses a fixed reference choice for everything that
  affects arithmetic. The same model and prompt then give the same tokens
  with an empty or a populated cache. It can be slower.

## Continuing a session

By default, a request that continues a session resumes from the session's
state, and the same sequence of calls always gives the same output.

For models with recurrent layers (for example Qwen3.6), set
`SONICBOOM_REPRO_CONTINUE=replay`, or load the model with
`BOOM_LLM_LOAD_CONTINUE_REPLAY`, to make a continued session give exactly the
bits that replaying the whole conversation from the start would give. This
can recompute up to one prefill segment per turn.

## Not covered

- Different machines or GPUs.
- Different SonicBoom versions.
- Batched execution of several requests together.
