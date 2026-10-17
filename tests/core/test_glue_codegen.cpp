// Compiler-generated glue kernel test (Stage 5).
//
// The MLIR -> NVVM -> PTX pipeline (compiler::generate_glue_ptx) and the driver
// JIT (planner::glue::launch) must reproduce the hand-written kernel the region
// replaces, byte for byte.
//
// Decode (5a): RMS-norm + q8_1 quantize. For random x, w the generated kernel's
// (d, qs[32]) blocks must memcmp-equal quant::cuda::rms_norm_quantize_q8_1_dev's
// output — d included, so the reduction order and the quantize sequence both
// have to match, not just the int8 quants.
//
// Prefill (5b): residual add + RMS-norm + mmq q8_1 quantize. The generated
// kernel's residual sum and 144-byte mmq blocks (d, optional ds half2, qs) must
// memcmp-equal nn::cuda::add_dev followed by
// quant::cuda::rms_norm_quantize_mmq_q8_1_dev over the same rows.
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include <sonicboom/compiler/GlueCodegen.h>

#include "../../core/src/nn/cuda_resident.h"
#include "../../core/src/planner/glue_jit.h"
#include "../../core/src/quant/cuda_resident.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
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

// A norm-weight buffer that stays alive for the whole run. The reference
// kernels upload their weight through upload_bytes, which caches by (host
// pointer, byte size); a local std::vector whose address gets recycled by the
// allocator would make a later call silently reuse an earlier test's weights.
std::vector<float>& persistent_weight(size_t n) {
  static std::vector<std::unique_ptr<std::vector<float>>> keep;
  keep.push_back(std::make_unique<std::vector<float>>(n));
  return *keep.back();
}

// One region launch, compared byte for byte with the hand-written kernel.
void test_rms_norm_quantize_q8_1(std::mt19937& rng, int n) {
  const std::string tag = "rms_norm_quantize_q8_1 n=" + std::to_string(n);

  std::vector<float> x(n);
  std::vector<float>& w = persistent_weight(n);
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

// One prefill add+norm+quantize region launch, compared byte for byte with the
// reference pair. `ds` is the mmq scale layout (0 = D4, 1 = DS4 half2).
void test_add_rms_norm_quantize_mmq_q8_1(std::mt19937& rng, int tokens, int dim,
                                         int ds) {
  const std::string tag = "add_rms_norm_quantize_mmq_q8_1 tokens=" +
                          std::to_string(tokens) + " dim=" +
                          std::to_string(dim) + " ds=" + std::to_string(ds);

  const size_t elems = size_t(tokens) * size_t(dim);
  std::vector<float> a(elems), b(elems);
  std::vector<float>& w = persistent_weight(dim);
  for (auto& v : a) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : b) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);
  const float eps = 1e-6f;

  const int nblocks = dim / 128;
  const size_t yq_bytes =
      size_t(tokens) * size_t(nblocks) *
      sizeof(sonicboom::quant::cuda::BlockQ8_1Mmq);
  const size_t xy_bytes = elems * sizeof(float);

  float *ad = nullptr, *bd = nullptr, *wd = nullptr;
  float *resref = nullptr, *resgen = nullptr;
  void *yref = nullptr, *ygen = nullptr;
  check(cudaMalloc(&ad, xy_bytes) == cudaSuccess, "cudaMalloc a");
  check(cudaMalloc(&bd, xy_bytes) == cudaSuccess, "cudaMalloc b");
  check(cudaMalloc(&wd, dim * sizeof(float)) == cudaSuccess, "cudaMalloc w");
  check(cudaMalloc(&resref, xy_bytes) == cudaSuccess, "cudaMalloc resref");
  check(cudaMalloc(&resgen, xy_bytes) == cudaSuccess, "cudaMalloc resgen");
  check(cudaMalloc(&yref, yq_bytes) == cudaSuccess, "cudaMalloc yref");
  check(cudaMalloc(&ygen, yq_bytes) == cudaSuccess, "cudaMalloc ygen");
  if (g_failures) return;

  cudaMemcpy(ad, a.data(), xy_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b.data(), xy_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(wd, w.data(), dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(resref, 0, xy_bytes);
  cudaMemset(resgen, 0, xy_bytes);
  cudaMemset(yref, 0, yq_bytes);
  cudaMemset(ygen, 0, yq_bytes);

  // The reference: the fused pair the region replaces.
  auto* yq_ref = reinterpret_cast<sonicboom::quant::cuda::BlockQ8_1Mmq*>(yref);
  check(sonicboom::nn::cuda::add_dev(ad, bd, resref, int(elems)),
        (tag + ": reference add").c_str());
  check(sonicboom::quant::cuda::rms_norm_quantize_mmq_q8_1_dev(
            std::span<const float>(w), resref, eps, yq_ref, dim, tokens, ds),
        (tag + ": reference quantize").c_str());

  // The generated kernel, loaded and launched through the JIT: one 256-thread
  // block per token row.
  void *aarg = ad;
  void *barg = bd;
  void *warg = wd;
  void *yarg = ygen;
  void *rarg = resgen;
  int dimarg = dim;
  int nbarg = tokens;
  int dsarg = ds;
  float earg = eps;
  void *args[] = {&aarg,  &barg, &warg,  &yarg,     &rarg,
                  &dimarg, &nbarg, &dsarg, &earg};
  check(sonicboom::planner::glue::launch(
            "add_rms_norm_quantize_mmq_q8_1", args, tokens,
            sonicboom::compiler::kAddRmsNormBlockThreads,
            sonicboom::compiler::kAddRmsNormSharedBytes),
        (tag + ": generated kernel launch").c_str());

  check(cudaDeviceSynchronize() == cudaSuccess, "device synchronize");

  std::vector<uint8_t> got(yq_bytes), ref(yq_bytes);
  cudaMemcpy(got.data(), ygen, yq_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(ref.data(), yref, yq_bytes, cudaMemcpyDeviceToHost);
  check(std::memcmp(got.data(), ref.data(), yq_bytes) == 0,
        (tag + ": byte-identical mmq blocks").c_str());
  if (std::memcmp(got.data(), ref.data(), yq_bytes) != 0) {
    for (int t = 0; t < tokens; ++t) {
      for (int blk = 0; blk < nblocks; ++blk) {
        const size_t off = (size_t(t) * size_t(nblocks) + size_t(blk)) *
                           sizeof(sonicboom::quant::cuda::BlockQ8_1Mmq);
        if (std::memcmp(got.data() + off, ref.data() + off,
                        sizeof(sonicboom::quant::cuda::BlockQ8_1Mmq)) == 0)
          continue;
        float dg = 0, dr = 0;
        std::memcpy(&dg, got.data() + off, sizeof(float));
        std::memcpy(&dr, ref.data() + off, sizeof(float));
        std::cerr << "    first differing block t=" << t << " blk=" << blk
                  << ": d got=" << dg << " ref=" << dr << "\n";
        break;
      }
    }
  }

  std::vector<float> rgot(elems), rref(elems);
  cudaMemcpy(rgot.data(), resgen, xy_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(rref.data(), resref, xy_bytes, cudaMemcpyDeviceToHost);
  const bool res_equal =
      std::memcmp(rgot.data(), rref.data(), xy_bytes) == 0;
  if (!res_equal) {
    for (size_t i = 0; i < elems; ++i) {
      if (std::memcmp(&rgot[i], &rref[i], sizeof(float)) != 0) {
        std::cerr << "    first differing residual " << i
                  << ": got=" << rgot[i] << " ref=" << rref[i] << "\n";
        break;
      }
    }
  }
  check(res_equal, (tag + ": byte-identical residual sum").c_str());

  cudaFree(ad);
  cudaFree(bd);
  cudaFree(wd);
  cudaFree(resref);
  cudaFree(resgen);
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

  auto ptx_mmq = sonicboom::compiler::generate_glue_ptx(
      sonicboom::compiler::GlueRegionKind::AddRmsNormQuantizeMmqQ8_1, arch);
  check(ptx_mmq.has_value(), "generate_glue_ptx returned prefill PTX");
  if (ptx_mmq) {
    check(
        ptx_mmq->find(sonicboom::compiler::glue_kernel_entry(
                  sonicboom::compiler::GlueRegionKind::
                      AddRmsNormQuantizeMmqQ8_1)) != std::string::npos,
        "prefill PTX contains the kernel entry symbol");
  }

  std::mt19937 rng(0x5eed5a);
  for (int n : {512, 2048, 2560}) test_rms_norm_quantize_q8_1(rng, n);
  // The generated prefill region over Gemma's and Qwen's hidden sizes, both mmq
  // scale layouts, and a single-block shape (only warp 0 has quantize work).
  test_add_rms_norm_quantize_mmq_q8_1(rng, 8, 2560, 0);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 8, 2560, 1);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 5, 896, 0);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 5, 896, 1);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 3, 128, 0);
  test_add_rms_norm_quantize_mmq_q8_1(rng, 3, 256, 1);

  if (g_failures) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "glue codegen: all checks passed\n";
  return 0;
}
