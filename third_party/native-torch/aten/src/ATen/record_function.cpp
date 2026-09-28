#include <ATen/record_function.h>

#include <utility>

// ATen/record_function.cpp — SonicBoom M1 ADAPT: no-op profiler hooks.
//
// Upstream, this file implements the profiler/observer callback runtime that
// RecordFunction drives from the dispatcher's boxed call path (Dispatcher::
// callBoxed -> runRecordFunction). SonicBoom v0 has no profiler, so every hook
// is a no-op: getStepCallbacksUnlessEmpty returns nullopt, which makes the
// dispatcher skip record-function bookkeeping entirely. When the profiler is
// migrated (a later M-phase), this file is replaced by the real implementation.

namespace at {

RecordFunction::RecordFunction(RecordScope scope) {
  // v0: no callbacks are registered, so step_callbacks_ stays empty and
  // isActive() is false.
  (void)scope;
}

RecordFunction::RecordFunction(StepCallbacks&& step_callbacks)
    : step_callbacks_(std::move(step_callbacks)) {}

RecordFunction::~RecordFunction() = default;

void RecordFunction::before(RecordFunction::FunctionDescriptor fn,
                            int64_t sequence_nr) {
  // v0: no active callbacks, so no start callbacks are run.
  (void)fn;
  (void)sequence_nr;
}

const char* RecordFunction::name() const {
  return "";
}

const char* RecordFunction::overload_name() const {
  return "";
}

size_t RecordFunction::num_inputs() const {
  return 0;
}

size_t RecordFunction::num_outputs() const {
  return 0;
}

uint64_t RecordFunction::currentThreadId() {
  return 0;
}

void RecordFunction::setDefaultNodeId(int64_t defaultNodeId) {
  (void)defaultNodeId;
}

void RecordFunction::end() {}

void RecordFunction::_setAsync() {}

bool RecordFunction::isAsync() const {
  return false;
}

void RecordFunction::_setStaticRuntimeOutVariant() {}

bool RecordFunction::isStaticRuntimeOutVariant() const {
  return false;
}

std::optional<OperatorName> RecordFunction::operator_name() const {
  return std::nullopt;
}

void RecordFunction::runStartCallbacks() {}

StepCallbacks getStepCallbacks(RecordScope scope) {
  (void)scope;
  return StepCallbacks();
}

std::optional<StepCallbacks> getStepCallbacksUnlessEmpty(RecordScope scope) {
  (void)scope;
  return std::nullopt;
}

void enableRecordFunction(bool enable) {
  (void)enable;
}

bool isRecordFunctionEnabled() {
  return false;
}

} // namespace at
