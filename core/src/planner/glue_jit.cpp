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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
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

// The PTX for the region on this device: the disk cache first (a warm file is
// the whole point — the MLIR pipeline is ~100 ms), the compiler on a miss.
std::string region_ptx(compiler::GlueRegionKind kind, int arch) {
  if (arch <= 0) return {};
  const std::string region = compiler::glue_region_name(kind);
  const std::filesystem::path path = cache_path(region, arch);
  std::string text = read_file(path);
  if (!text.empty()) return text;
  auto generated = compiler::generate_glue_ptx(kind, arch);
  if (!generated) return {};
  write_file(path, *generated);
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

} // namespace sonicboom::planner::glue
