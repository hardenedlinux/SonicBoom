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
  const std::string region = compiler::glue_region_name(kind);
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
  auto generated = compiler::generate_glue_ptx(kind, arch);
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
  return "attn-d" + std::to_string(shape.head_dim) + "-g" +
         std::to_string(shape.gqa) +
         (shape.window_mask ? "-window-" : "-causal-") + "b" +
         std::to_string(shape.br) + "x" + std::to_string(shape.bc);
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
  r = cuModuleGetFunction(&out.function, out.module,
                          compiler::kAttentionKernelEntry);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "glue JIT: cuModuleGetFunction failed for '%.*s': %s\n",
                 int(key.size()), key.data(), cu_error(r));
    cuModuleUnload(out.module);
    out = {};
    return false;
  }
  // Above 48 KiB the dynamic shared-memory limit has to be raised per function.
  if (shared_bytes > 48 * 1024)
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

} // namespace sonicboom::planner::glue
