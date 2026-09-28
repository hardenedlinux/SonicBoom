#pragma once

#include <ostream>
#include <string>

#include <c10/core/Scalar.h>
#include <ATen/core/Tensor.h>

namespace c10 {
TORCH_API std::ostream& operator<<(std::ostream& out, Backend b);
TORCH_API std::ostream& operator<<(std::ostream & out, const Scalar& s);
TORCH_API std::string toString(const Scalar& s);
}
namespace at {

TORCH_API std::ostream& operator<<(std::ostream& out, const DeprecatedTypeProperties& t);
TORCH_API std::ostream& print(
    std::ostream& stream,
    const Tensor& tensor,
    int64_t linesize);
inline std::ostream& operator<<(std::ostream & out, const Tensor & t) {
  // SonicBoom M1 ADAPT: upstream delegates to at::print (fmt-based tensor
  // pretty-printing, deferred). v0 emits a minimal placeholder so the
  // dispatcher core stays fmt-free (manifest M4).
  (void)t;
  return out << "Tensor";
}
TORCH_API void print(const Tensor & t, int64_t linesize=80);

// API to control scientific notation in tensor printing
TORCH_API void set_printoption_sci_mode(bool enabled);

} // namespace at
