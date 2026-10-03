# native-torch

Unmodified native PyTorch implementation, staged from `~/Project/pytorch`.

Staged scope (see `design/native-torch-migration.md` — authoritative):

- `c10/core`, `c10/util`, `c10/macros` — c10 core library (types, dispatcher keys, allocator, TensorImpl/StorageImpl).
- `aten/src/ATen/core` — ATen core (Tensor, IValue/Stack, Dispatcher, KernelFunction, op registration).

This is **Layer 1** implementation only. It is not the public SonicBoom API.
The SonicBoom Layer 2 headers (`core/include/sonicboom/`) must not include
these headers directly.

License: original PyTorch BSD-3-Clause (headers retained in each file).
