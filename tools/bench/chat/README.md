# chat — the fixed chat prompt set

Ten general prompts, five Chinese and five English, writing and Q&A, one
prompt per file (`00.txt`–`09.txt`, UTF-8, newline-terminated). They are the
`chat` set of the Output Decay metric
([`docs/output-decay.md`](../../../docs/output-decay.md)): greedy decoding on a
fresh process, a 512-token budget, acceleration off for the reference and on
for the compared run.

The prompts never change within a version of the metric. A new set is a new
directory, not an edit here, so numbers recorded against this set stay
reproducible.
