// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "glue_jit.h"

#include "reduce_mode.h"

#include <sonicboom/compiler/GemmCodegen.h>
#include <sonicboom/compiler/GlueCodegen.h>
#include <sonicboom/planner/plan_record.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#if SONICBOOM_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "../nn/cuda_stream.h"
#endif

namespace sonicboom::planner::glue {
namespace {

#if SONICBOOM_USE_CUDA

struct LoadedKernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  // Second entry of a decode module (fa_combine); null for prefill kernels.
  CUfunction function2 = nullptr;
};

// Loaded modules live for the process: a module unload is not needed for a
// 3-7 KiB PTX kernel, and keeping it avoids a lifetime race with a captured
// graph that references the function.
std::mutex &cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, LoadedKernel> &kernel_cache() {
  static std::unordered_map<std::string, LoadedKernel> cache;
  return cache;
}

int sm_arch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) !=
          cudaSuccess)
    return 0;
  return major * 10 + minor;
}

std::filesystem::path cache_path(const std::string &region, int arch) {
  const char *home = std::getenv("HOME");
  std::filesystem::path dir =
      home != nullptr ? std::filesystem::path(home) / ".cache" / "sonicboom" /
                            "kernels"
                      : std::filesystem::path("/tmp") / "sonicboom-kernels";
  return dir / (region + "-sm" + std::to_string(arch) + ".ptx");
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

void write_file(const std::filesystem::path &path, const std::string &text) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return;
  out.write(text.data(), std::streamsize(text.size()));
}

// FNV-1a 64 over the entry body. The cache is a local disk cache, not a trust
// boundary: this detects a truncated, corrupted, or stale body cheaply (the
// alternative — regenerating to compare — is exactly what the cache exists to
// avoid). The compiler version is validated separately below.
uint64_t fnv1a64(std::string_view text) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : text) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

std::string hex64(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx",
                static_cast<unsigned long long>(v));
  return std::string(buf);
}

// A cache entry is one header line plus the kernel text:
//
//   # sb-glue-ptx 1 region=<name> arch=<cc> codegen=<id> hash=<fnv1a64>
//   <ptx>
//
// The hash covers the body. The file name (region-sm<cc>.ptx) stays stable so
// a warm entry is found without generating first; the header is what makes a
// stale entry detectable — an entry written by another compiler, for another
// arch, in the old headerless format, or with a body that does not match its
// hash is ignored and regenerated.
std::string make_entry(const std::string &region, int arch,
                       const std::string &codegen, const std::string &body) {
  std::string text = "# sb-glue-ptx 1 region=" + region +
                     " arch=" + std::to_string(arch) + " codegen=" + codegen +
                     " hash=" + hex64(fnv1a64(body)) + "\n";
  text += body;
  return text;
}

std::string validate_entry(const std::string &text, const std::string &region,
                           int arch, const std::string &codegen) {
  constexpr std::string_view kMagic = "# sb-glue-ptx 1 ";
  if (text.compare(0, kMagic.size(), kMagic) != 0) return {};
  const size_t nl = text.find('\n');
  if (nl == std::string::npos) return {};
  const std::string header = text.substr(0, nl);
  const std::string body = text.substr(nl + 1);

  auto field = [&](const char *key) -> std::string {
    const std::string k = std::string(" ") + key + "=";
    size_t p = header.find(k);
    if (p == std::string::npos) return {};
    p += k.size();
    const size_t e = header.find(' ', p);
    return header.substr(p, e == std::string::npos ? std::string::npos : e - p);
  };
  if (field("region") != region) return {};
  if (field("arch") != std::to_string(arch)) return {};
  if (field("codegen") != codegen) return {};
  if (field("hash") != hex64(fnv1a64(body))) return {};
  return body;
}

// The PTX for the region on this device: the disk cache first (a warm file is
// the whole point — the MLIR pipeline is ~100 ms), the compiler on a miss.
std::string region_ptx(compiler::GlueRegionKind kind, int arch) {
  if (arch <= 0) return {};
  // The reduction mode is part of the kernel's identity: it changes the PTX, so
  // it selects a distinct cache entry (the F64 name is unchanged, keeping the
  // historical entries valid).
  const compiler::ReduceMode mode = device_reduce_mode();
  std::string region = compiler::glue_region_name(kind);
  if (mode != compiler::ReduceMode::F64)
    region += std::string("-") + compiler::reduce_mode_name(mode);
  // SONICBOOM_GLUE_CACHE=0 bypasses the disk cache (read and write): the
  // ablation for "is this the cached kernel or the one this compiler builds".
  static const bool use_cache = [] {
    const char *e = std::getenv("SONICBOOM_GLUE_CACHE");
    return !(e != nullptr && std::string_view(e) == "0");
  }();
  const std::filesystem::path path = cache_path(region, arch);
  if (use_cache) {
    const std::string cached = read_file(path);
    if (!cached.empty()) {
      std::string body = validate_entry(cached, region, arch,
                                        compiler::glue_codegen_version());
      if (!body.empty()) return body;
      std::fprintf(stderr, "glue JIT: ignoring stale PTX cache entry %s\n",
                   path.c_str());
    }
  }
  auto generated = compiler::generate_glue_ptx(kind, arch, mode);
  if (!generated) return {};
  if (use_cache)
    write_file(path, make_entry(region, arch, compiler::glue_codegen_version(),
                                *generated));
  return std::move(*generated);
}

bool use_disk_cache() {
  // SONICBOOM_GLUE_CACHE=0 bypasses the disk cache (read and write): the
  // ablation for "is this the cached kernel or the one this compiler builds".
  static const bool use = [] {
    const char *e = std::getenv("SONICBOOM_GLUE_CACHE");
    return !(e != nullptr && std::string_view(e) == "0");
  }();
  return use;
}

std::string attention_region_name(const compiler::AttentionShape &shape) {
  const std::string mask = shape.window_mask ? "-window" : "-causal";
  if (shape.decode)
    return "attn-dec-d" + std::to_string(shape.head_dim) + "-g" +
           std::to_string(shape.gqa) + mask;
  return "attn-d" + std::to_string(shape.head_dim) + "-g" +
         std::to_string(shape.gqa) + mask + "-b" + std::to_string(shape.br) +
         "x" + std::to_string(shape.bc) + "-dps" + std::to_string(shape.dps) +
         (shape.warp_head ? "-wh" : "") +
         (shape.async_stages >= 2 ? "-as" + std::to_string(shape.async_stages)
                                  : "") +
         (shape.gqa_pack > 1 ? "-p" + std::to_string(shape.gqa_pack) : "") +
         (shape.q_in_reg ? "" : "-qs") +
         (shape.head_fast ? "-hf" : "") +
         (shape.exp2 ? "-x2" : "");
}

// The generated attention kernel's PTX: disk cache first, generator on a miss.
// Same entry format and trust model as region_ptx.
std::string attention_ptx(const compiler::AttentionShape &shape, int arch) {
  if (arch <= 0 || !compiler::attention_shape_supported(shape)) return {};
  const std::string region = attention_region_name(shape);
  const std::string codegen = compiler::attention_codegen_version();
  const std::filesystem::path path = cache_path(region, arch);
  if (use_disk_cache()) {
    const std::string cached = read_file(path);
    if (!cached.empty()) {
      std::string body = validate_entry(cached, region, arch, codegen);
      if (!body.empty()) return body;
      std::fprintf(stderr, "glue JIT: ignoring stale PTX cache entry %s\n",
                   path.c_str());
    }
  }
  auto generated = compiler::generate_attention_ptx(shape, arch);
  if (!generated) return {};
  if (use_disk_cache())
    write_file(path, make_entry(region, arch, codegen, *generated));
  return std::move(*generated);
}

const char *cu_error(CUresult r);

std::string dense_gemm_region_name(const compiler::DenseGemmShape &shape) {
  return "gemm-m" + std::to_string(shape.bm) + "n" + std::to_string(shape.bn) +
         "k" + std::to_string(shape.bk);
}

// The generated dense GEMM's PTX: disk cache first, generator on a miss. Same
// entry format and trust model as region_ptx.
std::string dense_gemm_ptx(const compiler::DenseGemmShape &shape, int arch) {
  if (arch <= 0 || !compiler::dense_gemm_shape_supported(shape)) return {};
  const std::string region = dense_gemm_region_name(shape);
  const std::string codegen = compiler::dense_gemm_codegen_version();
  const std::filesystem::path path = cache_path(region, arch);
  if (use_disk_cache()) {
    const std::string cached = read_file(path);
    if (!cached.empty()) {
      std::string body = validate_entry(cached, region, arch, codegen);
      if (!body.empty()) return body;
      std::fprintf(stderr, "glue JIT: ignoring stale PTX cache entry %s\n",
                   path.c_str());
    }
  }
  auto generated = compiler::generate_dense_gemm_ptx(shape, arch);
  if (!generated) return {};
  if (use_disk_cache())
    write_file(path, make_entry(region, arch, codegen, *generated));
  return std::move(*generated);
}

std::string mmq_tc_region_name(const compiler::MmqTcShape &shape) {
  return "mmqtc-m" + std::to_string(shape.bm) + "n" + std::to_string(shape.bn) +
         "d" + std::to_string(shape.ds);
}

// The generated q8_1 tensor-core kernel's PTX: disk cache first, generator on
// a miss. Same entry format and trust model as region_ptx.
std::string mmq_tc_ptx(const compiler::MmqTcShape &shape, int arch) {
  if (arch <= 0 || !compiler::mmq_tc_shape_supported(shape)) return {};
  const std::string region = mmq_tc_region_name(shape);
  const std::string codegen = compiler::mmq_tc_codegen_version();
  const std::filesystem::path path = cache_path(region, arch);
  if (use_disk_cache()) {
    const std::string cached = read_file(path);
    if (!cached.empty()) {
      std::string body = validate_entry(cached, region, arch, codegen);
      if (!body.empty()) return body;
      std::fprintf(stderr, "glue JIT: ignoring stale PTX cache entry %s\n",
                   path.c_str());
    }
  }
  auto generated = compiler::generate_mmq_tc_ptx(shape, arch);
  if (!generated) return {};
  if (use_disk_cache())
    write_file(path, make_entry(region, arch, codegen, *generated));
  return std::move(*generated);
}

bool load_mmq_tc(const compiler::MmqTcShape &shape, std::string_view key,
                 int shared_bytes, LoadedKernel &out) {
  const int arch = sm_arch();
  const std::string ptx = mmq_tc_ptx(shape, arch);
  if (ptx.empty()) {
    std::fprintf(stderr, "glue JIT: no generated mmq-tc PTX for '%.*s'\n",
                 int(key.size()), key.data());
    return false;
  }
  CUresult r = cuModuleLoadData(&out.module, ptx.c_str());
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "glue JIT: cuModuleLoadData failed for '%.*s': %s\n",
                 int(key.size()), key.data(), cu_error(r));
    return false;
  }
  r = cuModuleGetFunction(&out.function, out.module,
                          compiler::kMmqTcKernelEntry);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "glue JIT: cuModuleGetFunction failed for '%.*s': %s\n",
                 int(key.size()), key.data(), cu_error(r));
    cuModuleUnload(out.module);
    out = {};
    return false;
  }
  if (shared_bytes >= 48 * 1024)
    cuFuncSetAttribute(out.function,
                       CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                       shared_bytes);
  return true;
}

bool load_dense_gemm(const compiler::DenseGemmShape &shape,
                     std::string_view key, int shared_bytes,
                     LoadedKernel &out) {
  const int arch = sm_arch();
  const std::string ptx = dense_gemm_ptx(shape, arch);
  if (ptx.empty()) {
    std::fprintf(stderr, "glue JIT: no generated dense GEMM PTX for '%.*s'\n",
                 int(key.size()), key.data());
    return false;
  }
  CUresult r = cuModuleLoadData(&out.module, ptx.c_str());
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "glue JIT: cuModuleLoadData failed for '%.*s': %s\n",
                 int(key.size()), key.data(), cu_error(r));
    return false;
  }
  r = cuModuleGetFunction(&out.function, out.module,
                          compiler::kDenseGemmKernelEntry);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "glue JIT: cuModuleGetFunction failed for '%.*s': %s\n",
                 int(key.size()), key.data(), cu_error(r));
    cuModuleUnload(out.module);
    out = {};
    return false;
  }
  if (shared_bytes >= 48 * 1024)
    cuFuncSetAttribute(out.function,
                       CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                       shared_bytes);
  return true;
}

const char *cu_error(CUresult r) {
  const char *s = nullptr;
  if (cuGetErrorString(r, &s) == CUDA_SUCCESS && s != nullptr) return s;
  return "unknown CUDA driver error";
}

bool load(std::string_view region, LoadedKernel &out) {
  const auto kind = compiler::glue_region_kind_from_name(region);
  if (!kind) return false;
  const int arch = sm_arch();
  const std::string ptx = region_ptx(*kind, arch);
  if (ptx.empty()) {
    std::fprintf(stderr, "glue JIT: no PTX for region '%.*s'\n",
                 int(region.size()), region.data());
    return false;
  }

  CUresult r = cuModuleLoadData(&out.module, ptx.c_str());
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuModuleLoadData failed for '%.*s': %s\n",
                 int(region.size()), region.data(), cu_error(r));
    return false;
  }
  r = cuModuleGetFunction(&out.function, out.module,
                          compiler::glue_kernel_entry(*kind));
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuModuleGetFunction failed for '%.*s': %s\n",
                 int(region.size()), region.data(), cu_error(r));
    cuModuleUnload(out.module);
    out = {};
    return false;
  }
  return true;
}

bool load_attention(const compiler::AttentionShape &shape,
                    std::string_view key, int shared_bytes,
                    LoadedKernel &out) {
  const int arch = sm_arch();
  const std::string ptx = attention_ptx(shape, arch);
  if (ptx.empty()) {
    std::fprintf(stderr, "glue JIT: no generated attention PTX for '%.*s'\n",
                 int(key.size()), key.data());
    return false;
  }
  CUresult r = cuModuleLoadData(&out.module, ptx.c_str());
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "glue JIT: cuModuleLoadData failed for '%.*s': %s\n",
                 int(key.size()), key.data(), cu_error(r));
    return false;
  }
  if (shape.decode) {
    r = cuModuleGetFunction(&out.function, out.module,
                            compiler::kAttentionDecodeKernelEntry);
    if (r == CUDA_SUCCESS)
      r = cuModuleGetFunction(&out.function2, out.module,
                              compiler::kAttentionCombineKernelEntry);
  } else {
    r = cuModuleGetFunction(&out.function, out.module,
                            compiler::kAttentionKernelEntry);
  }
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "glue JIT: cuModuleGetFunction failed for '%.*s': %s\n",
                 int(key.size()), key.data(), cu_error(r));
    cuModuleUnload(out.module);
    out = {};
    return false;
  }
  // 48 KiB and above needs the per-function dynamic shared-memory limit
  // raised: the driver reserves part of the 48 KiB default on sm_86, so a
  // kernel that asks for exactly 49152 B is rejected without the attribute
  // (the runtime tuner's b16x32-dps4 candidates hit this).
  if (shared_bytes >= 48 * 1024)
    cuFuncSetAttribute(out.function,
                       CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                       shared_bytes);
  return true;
}

#endif // SONICBOOM_USE_CUDA

} // namespace

bool known_region(std::string_view region) {
  return compiler::glue_region_kind_from_name(region).has_value();
}

bool launch(std::string_view region, void *const *args, int blocks, int threads,
            int shared_bytes) {
#if SONICBOOM_USE_CUDA
  const std::string key(region);
  CUfunction function = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cache = kernel_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
      LoadedKernel loaded;
      if (!load(region, loaded)) return false;
      it = cache.emplace(key, loaded).first;
    }
    function = it->second.function;
  }
  if (function == nullptr) return false;

  // The launch hook other kernels carry (plan_hook.h): the plan record must see
  // a fused region's launch like any other, or a recorded plan undercounts the
  // launches the run actually performed.
  if (auto *rec = sonicboom::planner::recorder()) {
    const auto kind = compiler::glue_region_kind_from_name(region);
    const std::string_view kernel =
        kind ? compiler::glue_kernel_entry(*kind) : region;
    rec->add_launch(kernel, "", unsigned(blocks), 1, 1, unsigned(threads), 1, 1,
                    static_cast<uint64_t>(shared_bytes),
                    reinterpret_cast<const void *>(
                        sonicboom::cuda::sb_current_stream()));
  }

  const CUresult r = cuLaunchKernel(
      function, unsigned(blocks), 1, 1, unsigned(threads), 1, 1,
      unsigned(shared_bytes), CUstream(sonicboom::cuda::sb_current_stream()),
      const_cast<void **>(args), nullptr);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuLaunchKernel failed for '%.*s': %s\n",
                 int(region.size()), region.data(), cu_error(r));
    return false;
  }
  return true;
#else
  (void)region;
  (void)args;
  (void)blocks;
  (void)threads;
  (void)shared_bytes;
  return false;
#endif
}

bool launch_attention(const compiler::AttentionShape &shape,
                      void *const *args, int grid_x, int grid_y) {
#if SONICBOOM_USE_CUDA
  if (grid_x <= 0 || grid_y <= 0) return false;
  const int threads = compiler::attention_block_threads(shape);
  const int shared_bytes = compiler::attention_shared_bytes(shape);
  const std::string key = attention_region_name(shape);
  CUfunction function = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cache = kernel_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
      LoadedKernel loaded;
      if (!load_attention(shape, key, shared_bytes, loaded)) return false;
      it = cache.emplace(key, loaded).first;
    }
    function = it->second.function;
  }
  if (function == nullptr) return false;

  if (auto *rec = sonicboom::planner::recorder()) {
    rec->add_launch(key, "", unsigned(grid_x), unsigned(grid_y), 1,
                    unsigned(threads), 1, 1, static_cast<uint64_t>(shared_bytes),
                    reinterpret_cast<const void *>(
                        sonicboom::cuda::sb_current_stream()));
  }
  const CUresult r = cuLaunchKernel(
      function, unsigned(grid_x), unsigned(grid_y), 1, unsigned(threads), 1, 1,
      unsigned(shared_bytes), CUstream(sonicboom::cuda::sb_current_stream()),
      const_cast<void **>(args), nullptr);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuLaunchKernel failed for '%s': %s\n",
                 key.c_str(), cu_error(r));
    return false;
  }
  return true;
#else
  (void)shape;
  (void)args;
  (void)grid_x;
  (void)grid_y;
  return false;
#endif
}

bool launch_attention_decode(const compiler::AttentionShape &shape,
                             void *const *args, int n_chunks, int n_heads_q) {
#if SONICBOOM_USE_CUDA
  if (n_chunks <= 0 || n_heads_q <= 0) return false;
  const int shared_bytes = compiler::attention_shared_bytes(shape);
  const std::string key = attention_region_name(shape);
  CUfunction partial_fn = nullptr;
  CUfunction combine_fn = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cache = kernel_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
      LoadedKernel loaded;
      if (!load_attention(shape, key, shared_bytes, loaded)) return false;
      it = cache.emplace(key, loaded).first;
    }
    partial_fn = it->second.function;
    combine_fn = it->second.function2;
  }
  if (partial_fn == nullptr || combine_fn == nullptr) return false;

  // fa_decode: q, kc, vc, startp, nslots, nhq, nhkv, window, scale, part,
  // chunk. args[11] is out and args[12] the chunk count for fa_combine.
  void *part_args[11];
  for (int i = 0; i < 11; ++i) part_args[i] = args[i];
  void *comb_args[4] = {args[9], args[11], args[5], args[12]};

  if (auto *rec = sonicboom::planner::recorder()) {
    rec->add_launch(key + ":decode", "", unsigned(n_chunks),
                    unsigned(n_heads_q), 1, 32, 1, 1,
                    static_cast<uint64_t>(shared_bytes),
                    reinterpret_cast<const void *>(
                        sonicboom::cuda::sb_current_stream()));
    rec->add_launch(key + ":combine", "", unsigned(n_heads_q), 1, 1,
                    unsigned(shape.head_dim), 1, 1, 0,
                    reinterpret_cast<const void *>(
                        sonicboom::cuda::sb_current_stream()));
  }
  CUresult r = cuLaunchKernel(
      partial_fn, unsigned(n_chunks), unsigned(n_heads_q), 1, 32, 1, 1,
      unsigned(shared_bytes), CUstream(sonicboom::cuda::sb_current_stream()),
      part_args, nullptr);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuLaunchKernel failed for '%s': %s\n",
                 key.c_str(), cu_error(r));
    return false;
  }
  r = cuLaunchKernel(
      combine_fn, unsigned(n_heads_q), 1, 1, unsigned(shape.head_dim), 1, 1, 0,
      CUstream(sonicboom::cuda::sb_current_stream()), comb_args, nullptr);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuLaunchKernel failed for '%s': %s\n",
                 key.c_str(), cu_error(r));
    return false;
  }
  return true;
#else
  (void)shape;
  (void)args;
  (void)n_chunks;
  (void)n_heads_q;
  return false;
#endif
}

bool launch_dense_gemm(const compiler::DenseGemmShape &shape, const void *x,
                       const void *w, void *y, int m, int k, int n,
                       float alpha) {
#if SONICBOOM_USE_CUDA
  if (m <= 0 || k <= 0 || n <= 0 || x == nullptr || w == nullptr ||
      y == nullptr)
    return false;
  if (!compiler::dense_gemm_shape_supported(shape)) return false;
  const int threads = compiler::dense_gemm_block_threads(shape);
  const int shared_bytes = compiler::dense_gemm_shared_bytes(shape);
  const std::string key = dense_gemm_region_name(shape);
  CUfunction function = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cache = kernel_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
      LoadedKernel loaded;
      if (!load_dense_gemm(shape, key, shared_bytes, loaded)) return false;
      it = cache.emplace(key, loaded).first;
    }
    function = it->second.function;
  }
  if (function == nullptr) return false;

  // A memref<?x?xbf16> kernel argument is passed as its flattened 7-field
  // descriptor: allocated ptr, aligned ptr, offset, size0, size1, stride0,
  // stride1. Both operands are K-contiguous, so stride = (k, 1); x is indexed
  // [token][k] and w [row][k] by the kernel.
  struct MemRef2D {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t size0, size1, stride0, stride1;
  };
  MemRef2D xd{const_cast<void *>(x), const_cast<void *>(x), 0, n, k, k, 1};
  MemRef2D wd{const_cast<void *>(w), const_cast<void *>(w), 0, m, k, k, 1};
  int mi = m, ki = k, ni = n;
  float a = alpha;
  void *args[19] = {&xd.allocated, &xd.aligned, &xd.offset, &xd.size0,
                    &xd.size1,     &xd.stride0, &xd.stride1,
                    &wd.allocated, &wd.aligned, &wd.offset, &wd.size0,
                    &wd.size1,     &wd.stride0, &wd.stride1,
                    &y,            &mi,         &ki,        &ni,
                    &a};
  const int gx = (n + shape.bn - 1) / shape.bn;
  const int gy = (m + shape.bm - 1) / shape.bm;

  if (auto *rec = sonicboom::planner::recorder()) {
    rec->add_launch(key, "", unsigned(gx), unsigned(gy), 1, unsigned(threads),
                    1, 1, static_cast<uint64_t>(shared_bytes),
                    reinterpret_cast<const void *>(
                        sonicboom::cuda::sb_current_stream()));
  }
  const CUresult r = cuLaunchKernel(
      function, unsigned(gx), unsigned(gy), 1, unsigned(threads), 1, 1,
      unsigned(shared_bytes), CUstream(sonicboom::cuda::sb_current_stream()),
      args, nullptr);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuLaunchKernel failed for '%s': %s\n",
                 key.c_str(), cu_error(r));
    return false;
  }
  return true;
#else
  (void)shape;
  (void)x;
  (void)w;
  (void)y;
  (void)m;
  (void)k;
  (void)n;
  (void)alpha;
  return false;
#endif
}

bool launch_mmq_tc(const compiler::MmqTcShape &shape, const void *xq,
                   const void *wq, const void *ws, void *y, int m, int k,
                   int n) {
#if SONICBOOM_USE_CUDA
  if (m <= 0 || k <= 0 || n <= 0 || k % 32 != 0 || xq == nullptr ||
      wq == nullptr || ws == nullptr || y == nullptr)
    return false;
  if (!compiler::mmq_tc_shape_supported(shape)) return false;
  const int threads = compiler::mmq_tc_block_threads(shape);
  const int shared_bytes = compiler::mmq_tc_shared_bytes(shape);
  const std::string key = mmq_tc_region_name(shape);
  CUfunction function = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex());
    auto &cache = kernel_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
      LoadedKernel loaded;
      if (!load_mmq_tc(shape, key, shared_bytes, loaded)) return false;
      it = cache.emplace(key, loaded).first;
    }
    function = it->second.function;
  }
  if (function == nullptr) return false;

  const void *a = xq, *w = wq, *s = ws;
  void *args[7] = {&a, &w, &s, &y, &m, &k, &n};
  const int gx = (n + shape.bn - 1) / shape.bn;
  const int gy = (m + shape.bm - 1) / shape.bm;

  if (auto *rec = sonicboom::planner::recorder()) {
    rec->add_launch(key, "", unsigned(gx), unsigned(gy), 1, unsigned(threads),
                    1, 1, static_cast<uint64_t>(shared_bytes),
                    reinterpret_cast<const void *>(
                        sonicboom::cuda::sb_current_stream()));
  }
  const CUresult r = cuLaunchKernel(
      function, unsigned(gx), unsigned(gy), 1, unsigned(threads), 1, 1,
      unsigned(shared_bytes), CUstream(sonicboom::cuda::sb_current_stream()),
      args, nullptr);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "glue JIT: cuLaunchKernel failed for '%s': %s\n",
                 key.c_str(), cu_error(r));
    return false;
  }
  return true;
#else
  (void)shape;
  (void)xq;
  (void)wq;
  (void)ws;
  (void)y;
  (void)m;
  (void)k;
  (void)n;
  return false;
#endif
}

} // namespace sonicboom::planner::glue
