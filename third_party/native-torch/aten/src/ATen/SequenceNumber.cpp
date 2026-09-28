#include <ATen/SequenceNumber.h>

#include <atomic>

// SonicBoom M1 — migrated as-is (COPY). A simple global sequence counter used
// by the dispatcher's record-function bookkeeping to correlate forward/backward
// ops. v0 does not run autograd, but the dispatcher core references peek().

namespace at::sequence_number {

namespace {
std::atomic<uint64_t> sequence_nr_{0};
} // namespace

uint64_t peek() {
  return sequence_nr_.load();
}

uint64_t get_and_increment() {
  return sequence_nr_++;
}

} // namespace at::sequence_number
