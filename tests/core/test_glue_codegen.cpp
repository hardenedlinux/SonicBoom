// Compiler-generated glue kernel test (Stage 5a).
//
// The MLIR -> NVVM -> PTX pipeline (compiler::generate_glue_ptx) and the driver
// JIT (planner::glue::launch) must reproduce the hand-written kernel the region
// replaces, byte for byte. The region tested is the decode RMS-norm + q8_1
// quantize: for random x, w the generated kernel's (d, qs[32]) blocks must
// memcmp-equal quant::cuda::rms_norm_quantize_q8_1_dev's output — d included, so
// the reduction order and the quantize sequence both have to match, not just
// the int8 quants.
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include <sonicboom/compiler/GlueCodegen.h>

#include "../../core/src/planner/glue_jit.h"
#include "../../core/src/quant/cuda_resident.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool have_cuda() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int device_arch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) !=
          cudaSuccess)
    return 0;
  return major * 10 + minor;
}

// One region launch, compared byte for byte with the hand-written kernel.
void test_rms_norm_quantize_q8_1(std::mt19937& rng, int n) {
  const std::string tag = "rms_norm_quantize_q8_1 n=" + std::to_string(n);

  std::vector<float> x(n), w(n);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);
  const float eps = 1e-6f;

  const int nblocks = n / 32;
  const size_t yq_bytes =
      size_t(nblocks) * sizeof(sonicboom::quant::cuda::BlockQ8_1);

  float *xd = nullptr, *wd = nullptr;
  void *yref = nullptr, *ygen = nullptr;
  check(cudaMalloc(&xd, n * sizeof(float)) == cudaSuccess, "cudaMalloc x");
  check(cudaMalloc(&wd, n * sizeof(float)) == cudaSuccess, "cudaMalloc w");
  check(cudaMalloc(&yref, yq_bytes) == cudaSuccess, "cudaMalloc yref");
  check(cudaMalloc(&ygen, yq_bytes) == cudaSuccess, "cudaMalloc ygen");
  if (g_failures) return;

  cudaMemcpy(xd, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(yref, 0, yq_bytes);
  cudaMemset(ygen, 0, yq_bytes);

  // The reference: the kernel the region replaces.
  check(sonicboom::quant::cuda::rms_norm_quantize_q8_1_dev(
            std::span<const float>(w), xd, eps,
            reinterpret_cast<sonicboom::quant::cuda::BlockQ8_1 *>(yref), n),
        (tag + ": reference kernel").c_str());

  // The generated kernel, loaded and launched through the JIT.
  void *xarg = xd;
  void *warg = wd;
  void *yarg = ygen;
  int narg = n;
  float earg = eps;
  void *args[] = {&xarg, &warg, &yarg, &narg, &earg};
  check(sonicboom::planner::glue::launch(
            "rms_norm_quantize_q8_1", args, 1,
            sonicboom::compiler::kGlueBlockThreads,
            sonicboom::compiler::kGlueDynamicSharedBytes),
        (tag + ": generated kernel launch").c_str());

  check(cudaDeviceSynchronize() == cudaSuccess, "device synchronize");

  std::vector<uint8_t> got(yq_bytes), ref(yq_bytes);
  cudaMemcpy(got.data(), ygen, yq_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(ref.data(), yref, yq_bytes, cudaMemcpyDeviceToHost);

  const bool equal = std::memcmp(got.data(), ref.data(), yq_bytes) == 0;
  if (!equal) {
    // Report the first differing block and whether it is the scale or a quant,
    // so a reduction-order and a quantization mismatch are distinguishable.
    for (int b = 0; b < nblocks; ++b) {
      const size_t off = size_t(b) * sizeof(sonicboom::quant::cuda::BlockQ8_1);
      if (std::memcmp(got.data() + off, ref.data() + off,
                      sizeof(sonicboom::quant::cuda::BlockQ8_1)) == 0)
        continue;
      float dg = 0, dr = 0;
      std::memcpy(&dg, got.data() + off, sizeof(float));
      std::memcpy(&dr, ref.data() + off, sizeof(float));
      std::cerr << "    first differing block " << b << ": d got=" << dg
                << " ref=" << dr << "\n";
      break;
    }
  }
  check(equal, (tag + ": byte-identical output").c_str());

  cudaFree(xd);
  cudaFree(wd);
  cudaFree(yref);
  cudaFree(ygen);
}

} // namespace

int main() {
  if (!have_cuda()) {
    std::cout << "no CUDA device: skipping\n";
    return 0;
  }
  const int arch = device_arch();
  std::cout << "glue codegen: sm_" << arch << "\n";

  // The PTX is generated once per region and cached on disk by glue_jit; this
  // call exercises the MLIR pipeline itself.
  auto ptx = sonicboom::compiler::generate_glue_ptx(
      sonicboom::compiler::GlueRegionKind::RmsNormQuantizeQ8_1, arch);
  check(ptx.has_value(), "generate_glue_ptx returned PTX");
  if (ptx) {
    check(ptx->find(sonicboom::compiler::glue_kernel_entry(
              sonicboom::compiler::GlueRegionKind::RmsNormQuantizeQ8_1)) !=
              std::string::npos,
          "PTX contains the kernel entry symbol");
  }

  std::mt19937 rng(0x5eed5a);
  for (int n : {512, 2048, 2560}) test_rms_norm_quantize_q8_1(rng, n);

  if (g_failures) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "glue codegen: all checks passed\n";
  return 0;
}
