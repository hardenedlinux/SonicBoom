# Test data

Fixtures used by the public tests and tools.

The public tree carries the contract, format and guarantee tests; additional
engine regression suites are maintained separately.

| file | used by | origin |
|---|---|---|
| `s-expr-v0-resnet18.example.sx` | S-Expr parser/executor and ONNX-importer tests | hand-written S-Expr rendering of ResNet-18 |
| `train/tiny-train.sx` | `tools/bench_train.cpp` (training throughput) | small hand-written training graph |
| `qwen2/p1.standard.golden` | `tests/core/test_capi_sampling.c` (the greedy stream the seeded-sampling test pins) | recorded from the Qwen2.5-0.5B reference run |

The rest of the model-specific goldens and their prompt fixtures are engine
regression data and live with the private suites, not here.
