# Weight formats in the graph — design note (Blade side first)

Status: **proposal, under revision.** Nothing in this document is implemented.

The document exists in the tree because two bodies of code point at it:
`TODO(weight-format)` in the ported Boom IR passes
(`core/compiler/lib/BoomPasses.cpp:820`, `:987`) and the private `ds` predicate
those passes keep from the Graph rewrites
(`core/src/planner/rewrites.cpp:362-377`). Both say the same thing: a weight's
quant format is an emitter-baked integer today, and it should be a property of a
type.

All `file:line` references are to `core/` unless stated, and are checked against
the `boom-passes` tip. Statements that are inference rather than observation are
marked **(inference)**.

## 1. Why the graph cannot say what a weight is quantized as

`TensorDesc` (`include/sonicboom/planner/graph.h:41-53`) has `shape`, `dtype`,
`size_bytes` and four flags — no format. `sx::DType` (`sx/ir.h`) has ten
members, none quantized. So the only quantized value the dialect can name for a
*graph tensor* is `q8_1`, and it is named by *provenance* (the first operand of
`quantized_matmul_q8`), not by a declaration. Design §6.2 wants the opposite:
`!boom.q4_k` … as first-class types for weights too, so the `ds`-consistency
precondition of `fuse_rms_norm_mmq_pass` becomes a verifier check instead of a
private predicate.

## 2. Where weights live today, and where their sizes come from

Weights are **not graph tensors** in the model path. The emitter bakes
`(layer, slot)` onto each node and the backend reads `model.weights`
(`model/plan.h:22-27`; `SonicBackend::weight` at `planner/sonic_backend.cpp:153-177`).
No `Gemma4Model`/`Qwen2` emitter node carries `is_constant` — verified by grep:
`core/src/model/*.cpp` never sets it. So **no `TensorDesc::size_bytes` in any
model graph describes a weight**; every tensor in those graphs is an f32
activation.

Weights live in two shapes (`model/weights.h`):

- `quant::QuantizedTensor` — packed quantized bytes, with `numel()`,
  `num_blocks()`, `byte_size() == data().size()` (`quant/quantized_tensor.h:102-107`).
  The **packed** size is what the format implies; `numel()` is the logical
  element count, and by the existing convention the logical element type is f32.
- `std::span<const float>` / `std::span<const uint16_t>` — norm weights, biases,
  RoPE table, per-layer model projection (bf16).

On the device: `upload_weight` (`quant/cuda_quant_matmul.cu:612-629`) sizes the
device copy from `W.byte_size()` (+256 B tail pad, ggml's own padding) and caches
it by `(host pointer, byte size)`. So the device footprint already follows the
**packed** format. Nothing accounts for it: `device_local.capacity_bytes = 0
// unknown; not budget-checked in v0` (`planner/resource.cpp:312`).

The one path where a weight *is* a graph tensor is the Blade/SX one:
`is_constant` / `is_external` (`graph.h:49-50`), set in `adapt_graph`
(`planner/graph_adapter.cpp:167`) from `sx::tensor_byte_size(type)`. That
function derives the size from `dtype` alone (`sx/ir.cpp:74`), so it cannot
produce a packed size for a quantized tensor — this is exactly the place a
format field is needed.

## 3. Does any planning, budgeting or allocation logic use a weight's `size_bytes`?

**The short answer, per path.**

- **Model path (Gemma 4, Qwen2.5): no — vacuously.** Weights are not graph
  tensors (§2), so no `TensorDesc` in those graphs describes a weight and no
  memory-planning or budget code ever sees a weight size. The one number that
  follows the packed format is the device upload's own `byte_size()`
  (`cuda_quant_matmul.cu:612-629`), which is a private cache in the `.cu` with
  no plan visibility and no budget (`resource.cpp:312`).
- **Blade / SX path: yes — constants are tensors like any other.** `adapt_graph`
  makes an SX constant a `TensorDesc` with `is_constant` set
  (`graph_adapter.cpp:165-169`), and the memory planner's loop has **no
  `is_constant` filter**, so a constant gets a buffer and counts toward the peak:

  | reader | what it does with the size |
  |---|---|
  | `planner/memory_planner.cpp:163,178,189` | `align_up(size_bytes, align)` → the buffer's size, `peak += size` |
  | `planner/memory_planner.cpp:~205` → `plan.memory_summary.peak_bytes` | checked against the budget at `planner/plan_validator.cpp:211-221` (`effective_budget_bytes`, `planner/resource.cpp:70`) |
  | `planner/cost_model.cpp:147` | `size_bytes % alignment_bytes` |
  | `planner/runtime_executor.cpp:123,127,325` | buffer size, then `device::acquire(al->size_bytes)` |
  | `planner/transfer_scheduler.cpp:80,88` | payload bytes and transfer-time estimate |
  | `planner/planner.cpp:183-189` | `in_bytes`/`out_bytes` per node, summed over every input/output, constants included |

  The consequence for this proposal: if a Blade parameter ever carries a packed
  size, **all of the above are already correct as written**, because they all
  mean "bytes this tensor occupies" — provided that stays `size_bytes`' one
  meaning (§4.1).

**Layout as capacity — the existing precedent, where the two meanings are
already equated:**

- `planner/rewrites.cpp:65-74` `mmq_q8_fits(dim, n, size_bytes)` — the gate is
  `size_bytes >= blocks * kMmqBlockBytes`, i.e. an f32 buffer's byte count is
  used as the capacity test for a *q8_1 packed* buffer; `:115` stores an
  activation's f32 `size_bytes` as the q8_1 descriptor's budget. This works only
  because a q8_1 activation never exceeds its f32 capacity.
- `planner/sonic_cuda_backend.cpp:481,624,629,651,766` — `in[0].device.size_bytes
  / sizeof(float)` recovers a dimension, i.e. these kernels assume the byte count
  is `n * 4`. A weight typed `q4_K` must not reach them (it does not today).

**Identity and records (this is what breaks first):**

- `planner/graph_adapter.cpp:107` — `model_fingerprint` hashes `size_bytes` for
  every tensor, in order. Any change to how a size is computed changes every
  fingerprint.
- `planner/plan_record.cpp:235` — the plan dump prints each buffer's
  `size_bytes`; the four committed goldens (`tests/data/plans/*.plan`) pin those
  numbers, including the `fingerprint` line.
- `planner/graph_adapter.cpp:107,165-169` and the Boom IR `#boom.tensor_desc`
  attribute — the round trip carries `size_bytes` and would have to carry the
  format too, or the round trip stops being an identity.

**Not affected:** the KV cache geometry (`planner/kv_cache.*`) is format-free.

## 4. Proposal

### 4.1 Blade / planner side

Add the format to the tensor, keeping today's convention that a quantized
tensor's *logical* element type is f32:

```cpp
struct TensorDesc {
  ...
  sx::DType dtype;        // logical element type (f32 for quantized tensors)
  std::optional<sx::QuantFormat> quant;  // packed format; nullopt = plain
  uint64_t size_bytes;    // bytes the tensor occupies (packed when quant set)
};
```

- `size_bytes` keeps its one meaning — bytes occupied — so §3's allocation,
  peak, budget, transfer and cost readers stay correct as written. The packed
  size is `num_blocks * bytes_per_block` from `QuantTypeInfo`
  (`quant/quantized_tensor.h:61-67`), which is exactly `QuantizedTensor::byte_size()`.
- A separate `logical_elems` should **not** be added: `shape` gives it, and a
  second source of truth is how the two drift.
- `sx::tensor_byte_size(type)` becomes format-aware (blocks × bytes_per_block),
  and the Blade parser must be able to say the format in text. That is the
  Blade-side change; the S-Expr v0.1 grammar has no such form, and v0.1 is
  frozen, so it belongs in the v0.2 increment list.
- `sx::Attribute` vocabulary: the format can ride as an attribute on the
  *parameter* declaration rather than a new dtype, which avoids touching
  `sx::DType`'s ten members.

Cost of the change: `model_fingerprint` gains a field → every fingerprint moves →
all four plan goldens must be re-recorded once, in one commit, with
`tools/plan_diff` showing only fingerprint/format-affected lines. That is
expected, not a regression.

### 4.2 Boom IR side

The dialect's half of this is **already in place** as of the `boom-dialect`
work, and it is worth stating exactly what exists before saying what is missing,
because the two halves of the review's ask land in different places:

- Quant formats are already element types, for weights and activations in **one**
  mechanism: `Boom_QuantType` (`compiler/include/sonicboom/compiler/BoomTypes.td:98-118`)
  declares six weight formats (`!boom.q3_k<[cols,rows]>` … `!boom.q8_0<[…]>`,
  block size 256 or 32) plus `!boom.q8_1<[n]>` for the activation layout, each
  carrying `formatName()` (spelled as `quant::QuantTypeInfo::name`) and
  `blockSize()`, with a generated `verify()` that the block-contiguous
  dimension is a multiple of the block size. The type set is hand-written and
  **drift-checked against `QuantType` by `test_boom_dialect`**, not generated
  from it.
- What is missing is not the mechanism but the two ends of it:
  1. **No weight is ever a graph value.** A weight is a `(layer, slot)`
     attribute plus a `model.weights` lookup (§2), so no operand of any matmul
     carries a weight format type, and `#boom.tensor_desc` has no format field
     to carry one through the round trip.
  2. **No op declares the formats it accepts.** Every operand is constrained by
     `Boom_ValueType` (any Boom type), so the generated verifier cannot reject a
     mismatched pair; what the hand-written verifier checks instead is the
     `q8_1`-by-provenance rule (`compiler/lib/BoomVerify.cpp:389-414`) and the
     legal values of `ds` (`:375-380`).

  The change: declare per-op format constraints in ODS (a weight operand becomes
  `AnyQuantOrF32` on the matmul ops; the q8_1 activation operand of
  `quantized_matmul_q8` stays exactly what it is), and add the verifier checks
  that expression cannot carry:
  1. operand/result format matches the op's declaration — rule (2)'s `ds`
     predicate becomes this check (§4.3);
  2. a quantized tensor's logical shape is a whole number of blocks
     (`numel % block_size == 0`), which the runtime already relies on
     (`quant/quantized_tensor.cpp:80-84`);
  3. a quantized operand must be produced by a producer of that format — no
     format is invented by a rewrite (the q8_1 rule at `BoomVerify.cpp:101-104`
     and `:389-414` is this check, for the one format that has producers today);
  4. mixed-format ops state their pair (`quantized_matmul_q8`: q8_1 ×
     weight-format → f32).

### 4.3 The `ds` predicate — the first check that goes away

`ds` is the mmq activation layout a weight format selects, and it is the
concrete reason this proposal exists:

- `fuse_rms_norm_mmq_pass` (`planner/rewrites.cpp:362-377`, ported as
  `RmsNormMmqPass`, `compiler/lib/BoomPasses.cpp:820`) fuses a norm into the
  matmuls that read it **only if** those matmuls agree on `batch` and `ds`, and
  only if none of them carries `ds < 0`.
- `ds < 0` is a hard-coded answer meaning "this weight type has no mmq
  activation layout" — Q6_K, whose mmq weight kernel is disabled. The fused
  activation producers (`fuse_act_mul_mmq_pass`,
  `compiler/lib/BoomPasses.cpp:987`) test the same thing.
- The verifier already bounds where `ds` may appear (`BoomVerify.cpp:110-118`,
  `:375-380`): only the matmuls and the fused activation quantizers may carry
  it, and only as `{-1, 0, 1}`.

Once a weight carries its format as a type, all three facts are derivable: `ds`
is "the activation layout this weight format's mmq kernel reads", the
"no mmq layout" answer is a property of the format (not of an integer's sign),
and the agreement test between two matmuls is a type equality. The predicate
then moves from a private function in two implementations of the same pass into
the verifier, which is the acceptance criterion for §5 step 3 — a plan goldens
diff that shows no change, plus the predicate's deletion.

## 5. Suggested order (each step independently verifiable)

1. `TensorDesc` + serialization + fingerprint get the field, always empty. No
   behaviour change; the four goldens are re-recorded once here, and only the
   fingerprint lines move.
2. `sx::tensor_byte_size` becomes format-aware; Blade v0.2 can name a format for
   a parameter. First step where a real packed weight size exists.
3. Boom IR: weights become graph values that carry their format type, per-op
   format declarations and the verifier rules of §4.2 — the step that removes
   the `ds` predicate (§4.3).
4. Re-record the goldens; the round-trip test then proves the format survives.

Steps 1 and 2 are separable from 3; step 3 is where the dialect's type story
becomes what design §6.2 describes. Step 3 is also the largest: weights entering
the graph changes the emitter and the backend's `model.weights` lookup, and is
**inference** — not costed here.

## 6. Decisions needed

1. Do quantized tensors keep `dtype = f32` as their logical type (this
   document's proposal, matching `QuantizedTensor`), or does `sx::DType` gain
   quantized members? The first keeps the ten dtypes and the existing kernels;
   the second makes the format a dtype everywhere.
2. Should Blade be able to type a *quantized intermediate* (a quantized
   activation other than q8_1, a quantized KV), or is a quantized tensor only
   ever a parameter plus the q8_1 activation? This decides whether the verifier
   needs a general "producer of format F" rule (§4.2 rule 3) or just the two
   producers that exist.
3. Should device weight residency be accounted for at all? Today it is a private
   cache in the `.cu` with no budget and no plan visibility (`resource.cpp:312`).
4. Should step 3 make weights graph values outright, or keep `(layer, slot)` and
   attach the format to the *node* attribute — a smaller change that fixes the
   verifier but leaves `#boom.tensor_desc` without a format? (§4.2 gap 1; the
   two differ in whether the round trip can carry a weight at all.)

## Revision

- **Rev 2** — the first version was written for review
  (`Status: proposal for review`). This revision: states the answer to the
  "does planning/budget use a weight's size" question as a conclusion, per path
  (§3); re-states the Boom IR side against what the dialect already has, so the
  gap is named (`Boom_QuantType` exists; weights are not graph values and ops do
  not declare formats) instead of the mechanism being re-proposed (§4.2); gives
  the `ds` predicate its own section as the removal target (§4.3); and
  re-verifies every citation against the `boom-passes` tip.
