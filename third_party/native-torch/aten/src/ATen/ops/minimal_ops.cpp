// ATen/ops/minimal_ops.cpp — SonicBoom M1 hand-authored minimal generated
// operator-method definitions.
//
// Upstream, each of these methods is emitted by torchgen into a per-op
// ATen/ops/<op>.cpp as a thin dispatcher call. SonicBoom v0 generates only the
// minimal subset the aten/core compile closure requires (manifest M1): the
// comparison / sparse-accessor / copy methods referenced by ivalue.cpp, plus
// TensorBase::to. Each method looks up its schema and invokes the kernel boxed;
// if the operator is not registered, the lookup throws (correct v0 behavior —
// these operators are outside the M1 proof set and are not registered yet).

#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>

namespace at {

namespace {

c10::OperatorHandle findOp(const char* name) {
  return c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
}

// One tensor in, one tensor out (unary, no extra arguments).
Tensor unary(const Tensor& self, const char* op_name) {
  c10::Stack stack;
  stack.emplace_back(self);
  findOp(op_name).callBoxed(stack);
  return std::move(stack[0]).toTensor();
}

// Two tensors in, one tensor out.
Tensor binary(const Tensor& self, const Tensor& other, const char* op_name) {
  c10::Stack stack;
  stack.emplace_back(self);
  stack.emplace_back(other);
  findOp(op_name).callBoxed(stack);
  return std::move(stack[0]).toTensor();
}

} // namespace

bool Tensor::is_nonzero() const {
  c10::Stack stack;
  stack.emplace_back(*this);
  findOp("aten::is_nonzero").callBoxed(stack);
  return stack[0].toBool();
}

Tensor Tensor::eq(const Tensor& other) const {
  return binary(*this, other, "aten::eq");
}

Tensor Tensor::lt(const Tensor& other) const {
  return binary(*this, other, "aten::lt");
}

Tensor Tensor::clone(std::optional<c10::MemoryFormat> memory_format) const {
  c10::Stack stack;
  stack.emplace_back(*this);
  if (memory_format.has_value()) {
    stack.emplace_back(*memory_format);
  }
  findOp("aten::clone").callBoxed(stack);
  return std::move(stack[0]).toTensor();
}

Tensor Tensor::indices() const {
  return unary(*this, "aten::indices");
}

Tensor Tensor::coalesce() const {
  return unary(*this, "aten::coalesce");
}

Tensor Tensor::values() const {
  return unary(*this, "aten::values");
}

Tensor Tensor::_values() const {
  return unary(*this, "aten::_values");
}

Tensor Tensor::_indices() const {
  return unary(*this, "aten::_indices");
}

Tensor Tensor::crow_indices() const {
  return unary(*this, "aten::crow_indices");
}

Tensor Tensor::col_indices() const {
  return unary(*this, "aten::col_indices");
}

// TensorBase::to is the catch-all `to` overload (TensorOptions + flags). It is
// referenced by ivalue.cpp's deepcopy path but is outside the M1 proof set; v0
// has no registered `to` kernel, so it is a defined-but-throwing placeholder.
TensorBase TensorBase::to(at::TensorOptions options,
                          bool non_blocking,
                          bool copy,
                          std::optional<at::MemoryFormat> memory_format) const {
  (void)options;
  (void)non_blocking;
  (void)copy;
  (void)memory_format;
  TORCH_CHECK(false, "aten::to is not in the SonicBoom v0 minimal operator set");
}

} // namespace at
