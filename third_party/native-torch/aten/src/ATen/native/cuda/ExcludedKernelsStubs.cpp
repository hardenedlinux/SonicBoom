// SonicBoom ADAPT: throwing stubs for CUDA kernels whose backing dependency is
// deliberately excluded from the v0 build scope.
//
// The generated CUDA registration files (RegisterCUDA_0.cpp, RegisterSparseCUDA_0,
// RegisterSparseCsrCUDA_0, RegisterNestedTensorCUDA_0) reference these symbols
// unconditionally — upstream gates them with `#ifdef` on optional backend
// libraries, but those guards are resolved to "absent" at registration time, so
// the references remain. The real implementations live behind dependencies that
// are not part of v0 (see CLAUDE.md §4.1.5 and the CUDA migration manifest):
//
//   cuDNN      (AT_CUDNN_ENABLED() == 0)          -> cudnn_*, _cudnn_*, _use_cudnn_*
//   MIOpen     (ROCm-only, irrelevant to CUDA)     -> miopen_*, _use_miopen_*
//   CUTLASS    (not vendored)                      -> at::cuda::detail grouped/rowwise
//                                                     MM, _sparse_semi_structured_*,
//                                                     _cslt_* (cuSPARSELt),
//                                                     _mixed_dtypes_linear (int4mm)
//   Flash/SDPA (cuDNN/CUTLASS/FA optional deps)    -> _flash_attention_*,
//                                                     _efficient_attention_*,
//                                                     _scaled_dot_product_*,
//                                                     _fused_sdp_choice_cuda,
//                                                     native_multi_head_attention_cuda,
//                                                     transform_bias_rescale_qkv_cuda,
//                                                     triton_scaled_dot_attention
//   quantized  (deferred dtypes)                   -> *_quantized, fused_moving_avg_*,
//                                                     make_per_*_quantized_tensor_cuda
//   fbgemm     (jagged/nested)                     -> _fbgemm_*_jagged_*
//   sparse CSR (deferred)                          -> structured__convert_indices_*::impl,
//                                                     nested_from_padded_cuda
//
// Each stub mirrors the upstream declaration's signature exactly and throws a
// NotImplementedError via TORCH_CHECK_NOT_IMPLEMENTED, so a controlled call to an
// excluded operator fails with a readable exception rather than a link error,
// null-pointer dispatch, or segfault. This matches the existing stub pattern in
// native/quantized/QuantizedStubs.cpp (FBGEMM-off) and native/cuda/LinearAlgebraStubs.cpp
// (MAGMA-off). It does NOT touch the generated registration files, and it does
// NOT pull in cuDNN / CUTLASS / cuSPARSELt / Flash Attention.

#include <ATen/ATen.h>
#include <ATen/native/cuda/GroupMM.h>
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>

namespace at::cuda::detail {

// CUTLASS grouped / rowwise scaled GEMM kernels (not vendored). The generated
// aten::group_mm / aten::_scaled_mm registrations reference these as their CUDA
// kernel; v0 has no CUTLASS, so fail fast on invocation.
void bf16bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias,
    at::Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "bf16bf16_grouped_mm: excluded from v0 build — CUTLASS not vendored");
}

void f8f8bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    at::Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "f8f8bf16_grouped_mm: excluded from v0 build — CUTLASS not vendored");
}

void f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    at::Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "f8f8bf16_rowwise: excluded from v0 build — CUTLASS not vendored");
}

} // namespace at::cuda::detail

namespace at::native {

// --- CUTLASS / cuSPARSELt sparse-semi-structured + int4mm --------------------

at::Tensor _cslt_compress(const at::Tensor& input) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cslt_compress: excluded from v0 build — cuSPARSELt not linked");
}

at::Tensor _cslt_sparse_mm(
    const at::Tensor& compressed_A,
    const at::Tensor& dense_B,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& alpha,
    std::optional<at::ScalarType> out_dtype,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    int64_t split_k_mode) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cslt_sparse_mm: excluded from v0 build — cuSPARSELt not linked");
}

int64_t _cslt_sparse_mm_search(
    const at::Tensor& compressed_A,
    const at::Tensor& dense_B,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& alpha,
    std::optional<at::ScalarType> out_dtype,
    bool transpose_result) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cslt_sparse_mm_search: excluded from v0 build — cuSPARSELt not linked");
}

at::Tensor _mixed_dtypes_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::string_view> activation) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_mixed_dtypes_linear: excluded from v0 build — CUTLASS (int4mm) not vendored");
}

at::Tensor _sparse_semi_structured_addmm(
    const at::Tensor& input,
    const at::Tensor& mat1,
    const at::Tensor& mat1_meta,
    const at::Tensor& mat2,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::optional<at::ScalarType> out_dtype) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_sparse_semi_structured_addmm: excluded from v0 build — CUTLASS not vendored");
}

std::tuple<at::Tensor, at::Tensor> _sparse_semi_structured_apply(
    const at::Tensor& input, const at::Tensor& thread_masks) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_sparse_semi_structured_apply: excluded from v0 build — CUTLASS not vendored");
}

at::Tensor _sparse_semi_structured_apply_dense(
    const at::Tensor& input, const at::Tensor& thread_masks) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_sparse_semi_structured_apply_dense: excluded from v0 build — CUTLASS not vendored");
}

at::Tensor _sparse_semi_structured_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& meta,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::string_view> activation,
    std::optional<at::ScalarType> out_dtype) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_sparse_semi_structured_linear: excluded from v0 build — CUTLASS not vendored");
}

at::Tensor _sparse_semi_structured_mm(
    const at::Tensor& mat1,
    const at::Tensor& mat1_meta,
    const at::Tensor& mat2,
    std::optional<at::ScalarType> out_dtype) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_sparse_semi_structured_mm: excluded from v0 build — CUTLASS not vendored");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_sparse_semi_structured_tile(
    const at::Tensor& input, c10::string_view algorithm, bool use_cutlass) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_sparse_semi_structured_tile: excluded from v0 build — CUTLASS not vendored");
}

std::tuple<at::Tensor, at::Tensor> _to_sparse_semi_structured(
    const at::Tensor& dense) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_to_sparse_semi_structured: excluded from v0 build — CUTLASS not vendored");
}

// --- cuDNN (AT_CUDNN_ENABLED == 0) ------------------------------------------

at::Tensor cudnn_affine_grid_generator_backward(
    const at::Tensor& grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_affine_grid_generator_backward: excluded from v0 build — cuDNN disabled");
}

at::Tensor cudnn_affine_grid_generator_forward(
    const at::Tensor& theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_affine_grid_generator_forward: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _cudnn_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const at::Tensor& attn_bias,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_attention_backward: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
_cudnn_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    const std::optional<at::Tensor>& cum_seq_q,
    const std::optional<at::Tensor>& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    const std::optional<at::Tensor>& seqused_k,
    const std::optional<at::Tensor>& block_table) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_attention_forward: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cudnn_batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    bool training,
    double exponential_average_factor,
    double epsilon) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_batch_norm: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cudnn_batch_norm_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    const std::optional<at::Tensor>& save_mean,
    const std::optional<at::Tensor>& save_var,
    double epsilon,
    const at::Tensor& reserveSpace) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_batch_norm_backward: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> cudnn_batch_norm_out(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    bool training,
    double exponential_average_factor,
    double epsilon,
    at::Tensor& out0,
    at::Tensor& out1,
    at::Tensor& out2,
    at::Tensor& out3) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_batch_norm_out: excluded from v0 build — cuDNN disabled");
}

at::Tensor cudnn_convolution_add_relu(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& z,
    const std::optional<at::Scalar>& alpha,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_convolution_add_relu: excluded from v0 build — cuDNN disabled");
}

at::Tensor cudnn_convolution(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_convolution: excluded from v0 build — cuDNN disabled");
}

at::Tensor& cudnn_convolution_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    at::Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_convolution_out: excluded from v0 build — cuDNN disabled");
}

at::Tensor cudnn_convolution_relu(
    const at::Tensor& self,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_convolution_relu: excluded from v0 build — cuDNN disabled");
}

at::Tensor cudnn_convolution_transpose(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_convolution_transpose: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor> _cudnn_ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t blank,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_ctc_loss: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor> _cudnn_ctc_loss_tensor(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_ctc_loss_tensor: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor> cudnn_grid_sampler_backward(
    const at::Tensor& self, const at::Tensor& grid, const at::Tensor& grad_output) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_grid_sampler_backward: excluded from v0 build — cuDNN disabled");
}

at::Tensor cudnn_grid_sampler_forward(const at::Tensor& self, const at::Tensor& grid) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cudnn_grid_sampler_forward: excluded from v0 build — cuDNN disabled");
}

at::Tensor _cudnn_init_dropout_state(
    double dropout,
    bool train,
    int64_t dropout_seed,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_init_dropout_state: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> _cudnn_rnn(
    const at::Tensor& input,
    at::TensorList weight,
    int64_t weight_stride0,
    const std::optional<at::Tensor>& weight_buf,
    const at::Tensor& hx,
    const std::optional<at::Tensor>& cx,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    const std::optional<at::Tensor>& dropout_state) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_rnn: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>
_cudnn_rnn_backward(
    const at::Tensor& input,
    at::TensorList weight,
    int64_t weight_stride0,
    const at::Tensor& weight_buf,
    const at::Tensor& hx,
    const std::optional<at::Tensor>& cx,
    const at::Tensor& output,
    const std::optional<at::Tensor>& grad_output,
    const std::optional<at::Tensor>& grad_hy,
    const std::optional<at::Tensor>& grad_cy,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    const std::optional<at::Tensor>& dropout_state,
    const at::Tensor& reserve,
    std::array<bool, 4> output_mask) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_rnn_backward: excluded from v0 build — cuDNN disabled");
}

at::Tensor _cudnn_rnn_flatten_weight(
    at::TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    bool bidirectional) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_cudnn_rnn_flatten_weight: excluded from v0 build — cuDNN disabled");
}

bool _use_cudnn_ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t blank) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_use_cudnn_ctc_loss: excluded from v0 build — cuDNN disabled");
}

bool _use_cudnn_ctc_loss_tensor(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_use_cudnn_ctc_loss_tensor: excluded from v0 build — cuDNN disabled");
}

// --- MIOpen (ROCm-only; irrelevant to the CUDA build) -----------------------

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_batch_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    bool training,
    double exponential_average_factor,
    double epsilon) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_batch_norm: excluded from v0 build — MIOpen is ROCm-only");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_batch_norm_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    const std::optional<at::Tensor>& save_mean,
    const std::optional<at::Tensor>& save_var,
    double epsilon) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_batch_norm_backward: excluded from v0 build — MIOpen is ROCm-only");
}

at::Tensor miopen_convolution_add_relu(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& z,
    const std::optional<at::Scalar>& alpha,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_convolution_add_relu: excluded from v0 build — MIOpen is ROCm-only");
}

at::Tensor miopen_convolution(
    const at::Tensor& self,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_convolution: excluded from v0 build — MIOpen is ROCm-only");
}

at::Tensor miopen_convolution_relu(
    const at::Tensor& self,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_convolution_relu: excluded from v0 build — MIOpen is ROCm-only");
}

at::Tensor miopen_convolution_transpose(
    const at::Tensor& self,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_convolution_transpose: excluded from v0 build — MIOpen is ROCm-only");
}

std::tuple<at::Tensor, at::Tensor> miopen_ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t blank,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_ctc_loss: excluded from v0 build — MIOpen is ROCm-only");
}

std::tuple<at::Tensor, at::Tensor> miopen_ctc_loss_tensor(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_ctc_loss_tensor: excluded from v0 build — MIOpen is ROCm-only");
}

at::Tensor miopen_depthwise_convolution(
    const at::Tensor& self,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_depthwise_convolution: excluded from v0 build — MIOpen is ROCm-only");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> miopen_rnn(
    const at::Tensor& input,
    at::TensorList weight,
    int64_t weight_stride0,
    const at::Tensor& hx,
    const std::optional<at::Tensor>& cx,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    const std::optional<at::Tensor>& dropout_state) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_rnn: excluded from v0 build — MIOpen is ROCm-only");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>
miopen_rnn_backward(
    const at::Tensor& input,
    at::TensorList weight,
    int64_t weight_stride0,
    const at::Tensor& weight_buf,
    const at::Tensor& hx,
    const std::optional<at::Tensor>& cx,
    const at::Tensor& output,
    const std::optional<at::Tensor>& grad_output,
    const std::optional<at::Tensor>& grad_hy,
    const std::optional<at::Tensor>& grad_cy,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    const std::optional<at::Tensor>& dropout_state,
    const at::Tensor& reserve,
    std::array<bool, 4> output_mask) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "miopen_rnn_backward: excluded from v0 build — MIOpen is ROCm-only");
}

bool _use_miopen_ctc_loss(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t blank) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_use_miopen_ctc_loss: excluded from v0 build — MIOpen is ROCm-only");
}

bool _use_miopen_ctc_loss_tensor(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_use_miopen_ctc_loss_tensor: excluded from v0 build — MIOpen is ROCm-only");
}

// --- Flash Attention / SDPA (optional cuDNN/CUTLASS/FA dependencies) ---------

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& out,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    const at::Tensor& logsumexp,
    double dropout_p,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    int64_t custom_mask_type,
    bool bias_requires_grad,
    std::optional<double> scale,
    std::optional<int64_t> num_splits_key,
    std::optional<int64_t> window_size,
    bool shared_storage_dqdkdv) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_efficient_attention_backward: excluded from v0 build — SDPA backend not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt>
_efficient_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const std::optional<at::Tensor>& cu_seqlens_k,
    std::optional<int64_t> max_seqlen_q,
    std::optional<int64_t> max_seqlen_k,
    double dropout_p,
    int64_t custom_mask_type,
    bool compute_log_sumexp,
    std::optional<double> scale,
    const std::optional<at::Tensor>& seqlen_k,
    std::optional<int64_t> window_size) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_efficient_attention_forward: excluded from v0 build — SDPA backend not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& rng_state,
    const at::Tensor& unused,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_flash_attention_backward: excluded from v0 build — Flash Attention not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& cum_seq_q,
    const std::optional<at::Tensor>& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<at::Tensor>& seqused_k,
    const std::optional<at::Tensor>& alibi_slopes,
    const std::optional<at::Tensor>& block_table,
    std::optional<int64_t> num_splits) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_flash_attention_forward: excluded from v0 build — Flash Attention not in scope");
}

at::Tensor _flash_attention_forward_no_dropout_inplace(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& cum_seq_q,
    const std::optional<at::Tensor>& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<at::Tensor>& seqused_k,
    const std::optional<at::Tensor>& alibi_slopes,
    const std::optional<at::Tensor>& block_table,
    std::optional<int64_t> num_splits) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_flash_attention_forward_no_dropout_inplace: excluded from v0 build — Flash Attention not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_flash_attention_forward_quantized(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& cum_seq_q,
    const std::optional<at::Tensor>& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    const std::optional<at::Tensor>& q_descale,
    const std::optional<at::Tensor>& k_descale,
    const std::optional<at::Tensor>& v_descale,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<at::Tensor>& seqused_k,
    const std::optional<at::Tensor>& alibi_slopes) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_flash_attention_forward_quantized: excluded from v0 build — Flash Attention not in scope");
}

at::Tensor& _fill_mem_eff_dropout_mask_(
    at::Tensor& self, double dropout_p, int64_t seed, int64_t offset) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_fill_mem_eff_dropout_mask_: excluded from v0 build — SDPA backend not in scope");
}

int64_t _fused_sdp_choice_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_fused_sdp_choice_cuda: excluded from v0 build — SDPA backend not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_cudnn_attention_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_scaled_dot_product_cudnn_attention_cuda: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_cudnn_attention_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const at::Tensor& attn_bias,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_scaled_dot_product_cudnn_attention_backward_cuda: excluded from v0 build — cuDNN disabled");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_scaled_dot_product_efficient_attention_cuda: excluded from v0 build — SDPA backend not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_efficient_attention_backward_cuda(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool is_causal,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_scaled_dot_product_efficient_attention_backward_cuda: excluded from v0 build — SDPA backend not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_scaled_dot_product_flash_attention_cuda: excluded from v0 build — Flash Attention not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_scaled_dot_product_flash_attention_backward_cuda: excluded from v0 build — Flash Attention not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_cuda_quantized(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& q_descale,
    const std::optional<at::Tensor>& k_descale,
    const std::optional<at::Tensor>& v_descale,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_scaled_dot_product_flash_attention_cuda_quantized: excluded from v0 build — Flash Attention not in scope");
}

std::tuple<at::Tensor, at::Tensor> native_multi_head_attention_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    int64_t embed_dim,
    int64_t num_head,
    const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight,
    const at::Tensor& proj_bias,
    const std::optional<at::Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    std::optional<int64_t> mask_type) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "native_multi_head_attention_cuda: excluded from v0 build — SDPA backend not in scope");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> transform_bias_rescale_qkv_cuda(
    const at::Tensor& qkv, const at::Tensor& qkv_bias, int64_t num_heads) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "transform_bias_rescale_qkv_cuda: excluded from v0 build — SDPA backend not in scope");
}

at::Tensor triton_scaled_dot_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double dropout_p) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "triton_scaled_dot_attention: excluded from v0 build — Triton not in scope");
}

// --- quantized (deferred dtypes, CLAUDE.md §4.1.5) ---------------------------

std::tuple<at::Tensor, at::Tensor> fused_moving_avg_obs_fake_quant_cuda(
    const at::Tensor& self,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    double averaging_const,
    int64_t quant_min,
    int64_t quant_max,
    int64_t ch_axis,
    bool per_row_fake_quant,
    bool symmetric_quant) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "fused_moving_avg_obs_fake_quant_cuda: excluded from v0 build — quantized dtypes deferred");
}

at::Tensor make_per_channel_quantized_tensor_cuda(
    const at::Tensor& self,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t axis) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "make_per_channel_quantized_tensor_cuda: excluded from v0 build — quantized dtypes deferred");
}

at::Tensor make_per_tensor_quantized_tensor_cuda(
    const at::Tensor& self, double scale, int64_t zero_point) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "make_per_tensor_quantized_tensor_cuda: excluded from v0 build — quantized dtypes deferred");
}

// --- fbgemm (jagged / nested, deferred) --------------------------------------

at::Tensor _fbgemm_dense_to_jagged_forward_symint(
    const at::Tensor& dense,
    at::TensorList offsets,
    std::optional<c10::SymInt> total_L) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_fbgemm_dense_to_jagged_forward_symint: excluded from v0 build — FBGEMM jagged deferred");
}

at::Tensor _fbgemm_jagged_to_padded_dense_forward(
    const at::Tensor& values,
    at::TensorList offsets,
    at::IntArrayRef max_lengths,
    double padding_value) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_fbgemm_jagged_to_padded_dense_forward: excluded from v0 build — FBGEMM jagged deferred");
}

// --- sparse CSR / nested (deferred) ------------------------------------------

at::Tensor nested_from_padded_cuda(
    const at::Tensor& padded,
    const at::Tensor& cpu_nested_shape_example,
    bool fuse_transform_0213) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "nested_from_padded_cuda: excluded from v0 build — nested tensor deferred");
}

void structured__convert_indices_from_csr_to_coo_structured_cuda::impl(
    const at::Tensor& crow_indices,
    const at::Tensor& col_indices,
    bool out_int32,
    bool transpose,
    const at::Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_convert_indices_from_csr_to_coo: excluded from v0 build — sparse CSR deferred");
}

void structured__convert_indices_from_coo_to_csr_structured_cuda::impl(
    const at::Tensor& self,
    int64_t size,
    bool out_int32,
    const at::Tensor& out) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "_convert_indices_from_coo_to_csr: excluded from v0 build — sparse CSR deferred");
}

} // namespace at::native
