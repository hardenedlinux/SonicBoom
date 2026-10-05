// SonicBoom ADAPT: quantized linear prepack (fbgemm) is deferred in v0
// (CLAUDE.md §4.1.5 — quantized dtypes are deferred). These symbols are
// referenced by the generated CPU registration (RegisterCPU) and by
// native/RNN.cpp's static init, but their real implementations live in
// native/quantized/cpu/*.cpp behind `#ifdef USE_FBGEMM`. v0 supplies throwing
// stubs that mirror upstream's own "not built with FBGEMM" fallback branch.
//
//   _saturate_weight_to_fp16             (qlinear_prepack.cpp)
//   _wrapped_linear_prepack              (qlinear_prepack.cpp)
//   _wrapped_quantized_linear_prepacked  (qlinear_prepack.cpp)
//   register_linear_params               (fbgemm_utils.cpp; declared at global
//                                         scope in native/quantized/library.h)

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/FakeQuantAffine.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/custom_class.h>

namespace at::native {

// The quantized dispatch stubs below are DECLARE_DISPATCH'd in the boundary
// headers and DEFINE_DISPATCH'd in the boundary .cpp files (AffineQuantizer.cpp,
// FakeQuantPer*Affine.cpp, TensorAdvancedIndexing.cpp), but their DEFAULT
// kernels are registered in native/quantized/cpu/kernels/QuantizedOpKernels.cpp
// — a monolithic file that pulls in fbgemm and ARM NEON (both outside v0).
// v0 defers quantized compute (§4.1.5), so these stubs are registered as
// "no CPU dispatch" (DEFAULT = nullptr), mirroring mkl/SpectralOps.cpp's
// REGISTER_NO_CPU_DISPATCH for the deferred FFT stub. Any invocation through
// these dispatch paths fails fast, which is the correct behavior for a
// deferred feature.

REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_tensor_affine_stub)
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_channel_affine_stub)
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_channel_float_qparams_stub)
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_tensor_affine_stub)
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_channel_affine_stub)
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub)
REGISTER_NO_CPU_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub)
REGISTER_NO_CPU_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub)

REGISTER_NO_CPU_DISPATCH(fake_quant_tensor_cachemask_stub)
REGISTER_NO_CPU_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_stub)
REGISTER_NO_CPU_DISPATCH(fake_quant_grad_learnable_tensor_stub)
REGISTER_NO_CPU_DISPATCH(fake_quant_per_channel_cachemask_stub)
REGISTER_NO_CPU_DISPATCH(fake_quant_grad_learnable_channel_stub)

REGISTER_NO_CPU_DISPATCH(masked_fill_kernel_quantized_stub)
REGISTER_NO_CPU_DISPATCH(index_put_kernel_quantized_stub)

at::Tensor _saturate_weight_to_fp16(const Tensor&) {
  TORCH_CHECK(
      false,
      "This build was not compiled with FBGEMM operators "
      "(quantized linear prepack deferred)");
}

at::Tensor _wrapped_linear_prepack(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  TORCH_CHECK(
      false,
      "This build was not compiled with FBGEMM operators "
      "(quantized linear prepack deferred)");
}

at::Tensor _wrapped_quantized_linear_prepacked(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const int64_t) {
  TORCH_CHECK(
      false,
      "This build was not compiled with FBGEMM operators "
      "(quantized linear prepack deferred)");
}

} // namespace at::native

// register_linear_params() is declared (TORCH_API) at global scope in
// native/quantized/library.h; upstream defines it in fbgemm_utils.cpp. v0
// registers the LinearPackedParamsBase custom-class TYPE only (no def_pickle /
// methods): RNN.cpp calls register_linear_params() immediately before
// registering CellParamsBase, whose def_pickle references this type via
// CellParamsSerializationType, so the type must be in the custom-class type map
// for getTypePtr<intrusive_ptr<LinearPackedParamsBase>>() to resolve. Upstream's
// def_pickle/methods reference the fbgemm/qnnpack/mkldnn packed-weight
// subclasses — all deferred (CLAUDE.md §4.1.5). The bare type registration is
// sufficient for the type-erasure path; actual (de)serialization is never
// invoked in v0 because quantized compute is deferred.
int register_linear_params() {
  [[maybe_unused]] static auto register_linear_params =
      torch::selective_class_<LinearPackedParamsBase>(
          "quantized", TORCH_SELECTIVE_CLASS("LinearPackedParamsBase"));
  return 0;
}
