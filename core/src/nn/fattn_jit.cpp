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

// NVRTC half of the boom_fattn candidate JIT. See fattn_jit.h for the flow.

#include "fattn_jit.h"

// Generated at build time from fattn_boom_device.cuh (the same text the AOT
// translation unit includes): `kFattnDeviceSource` is the whole device header.
#include "fattn_device_src.inc"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sonicboom::nn::cuda::fattn_jit {
namespace {

// ---------------------------------------------------------------------------
// NVRTC, loaded lazily. The library version does not have to match the driver:
// NVRTC only emits PTX, and the driver's JIT compiles it for the device.
// ---------------------------------------------------------------------------
struct Nvrtc {
  void* lib = nullptr;
  nvrtcResult (*create)(nvrtcProgram*, const char*, const char*, int,
                        const char* const*, const char* const*) = nullptr;
  nvrtcResult (*compile)(nvrtcProgram, int, const char* const*) = nullptr;
  nvrtcResult (*ptx_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*ptx)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*log_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*log)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*destroy)(nvrtcProgram*) = nullptr;
  nvrtcResult (*version)(int*, int*) = nullptr;
  std::string error;
};

Nvrtc& nvrtc() {
  static Nvrtc n = [] {
    Nvrtc r;
    const char* names[] = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"};
    for (const char* name : names) {
      r.lib = ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (r.lib != nullptr) break;
    }
    if (r.lib == nullptr) {
      r.error = "libnvrtc not found";
      return r;
    }
    auto sym = [&](const char* name) { return ::dlsym(r.lib, name); };
    r.create = reinterpret_cast<decltype(r.create)>(sym("nvrtcCreateProgram"));
    r.compile = reinterpret_cast<decltype(r.compile)>(sym("nvrtcCompileProgram"));
    r.ptx_size = reinterpret_cast<decltype(r.ptx_size)>(sym("nvrtcGetPTXSize"));
    r.ptx = reinterpret_cast<decltype(r.ptx)>(sym("nvrtcGetPTX"));
    r.log_size =
        reinterpret_cast<decltype(r.log_size)>(sym("nvrtcGetProgramLogSize"));
    r.log = reinterpret_cast<decltype(r.log)>(sym("nvrtcGetProgramLog"));
    r.destroy =
        reinterpret_cast<decltype(r.destroy)>(sym("nvrtcDestroyProgram"));
    // Optional: nvrtcVersion exists in every release that has the rest, but a
    // missing symbol only costs the version in the key, not the JIT itself.
    r.version = reinterpret_cast<decltype(r.version)>(sym("nvrtcVersion"));
    if (!r.create || !r.compile || !r.ptx_size || !r.ptx || !r.log_size ||
        !r.log || !r.destroy) {
      r.error = "libnvrtc is missing an entry point";
      ::dlclose(r.lib);
      r.lib = nullptr;
    }
    return r;
  }();
  return n;
}

// ---------------------------------------------------------------------------
// Device facts (queried once).
// ---------------------------------------------------------------------------
struct DeviceInfo {
  bool ok = false;
  int cc_major = 0;
  int cc_minor = 0;
  std::string error;

  int arch() const { return cc_major * 10 + cc_minor; }
};

const DeviceInfo& device_info() {
  static const DeviceInfo info = [] {
    DeviceInfo d;
    int dev = 0;
    if (::cudaGetDevice(&dev) != cudaSuccess) {
      d.error = "no CUDA device";
      return d;
    }
    cudaDeviceProp p{};
    if (::cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
      d.error = "cudaGetDeviceProperties failed";
      return d;
    }
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.ok = true;
    return d;
  }();
  return info;
}

// ---------------------------------------------------------------------------
// Source and cache key.
// ---------------------------------------------------------------------------
int threads_for(const Spec& s) { return (s.br / 16) * s.gqa * s.nw * 32; }

std::string spec_string(const Spec& s) {
  char buf[192];
  std::snprintf(buf, sizeof(buf), "hd%d/gqa%d/br%d/bk%d/nw%d/k%d/v%d/s%d/p%d",
                s.head_dim, s.gqa, s.br, s.bk, s.nw, s.kpad, s.vpad, s.spw,
                s.pipe);
  return buf;
}

std::string wrapper_source(const Spec& s) {
  std::string src = device_source();
  src += "\nextern \"C\" __global__ void __launch_bounds__(" +
         std::to_string(threads_for(s)) + ") sb_fattn_jit_" +
         std::to_string(s.id) +
         "(const __half* q, const __half* k_ws, const __half* v_ws,\n"
         " const uint64_t* start_ptr, uint64_t n_slots, uint64_t window,\n"
         " int n, int n_heads_q, int n_heads_kv, int ne_pad, float scale,\n"
         " float* out) {\n"
         "  sonicboom::nn::cuda::boom_fattn_qreg_body<" +
         std::to_string(s.head_dim) + ", " + std::to_string(s.gqa) + ", " +
         std::to_string(s.br) + ", " + std::to_string(s.bk) + ", " +
         std::to_string(s.nw) + ", true, true, " + std::to_string(s.kpad) +
         ", " + std::to_string(s.vpad) + ", " + std::to_string(s.spw) + ", " +
         (s.pipe ? "true" : "false") +
         ">(q, k_ws, v_ws, start_ptr, n_slots, window, n, n_heads_q,\n"
         "   n_heads_kv, ne_pad, scale, out);\n}\n";
  return src;
}

uint64_t fnv1a(const std::string& text, uint64_t h = 1469598103934665603ull) {
  for (unsigned char c : text) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

std::string cache_dir() {
  if (const char* e = std::getenv("XDG_CACHE_HOME"); e != nullptr && *e != '\0')
    return std::string(e) + "/sonicboom/kernels";
  if (const char* h = std::getenv("HOME"); h != nullptr && *h != '\0')
    return std::string(h) + "/.cache/sonicboom/kernels";
  return "/tmp/sonicboom-kernels";
}

bool read_file(const std::string& path, std::string* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
  std::fclose(f);
  if (text.empty()) return false;
  *out = std::move(text);
  return true;
}

bool write_file_atomic(const std::string& path, const std::string& text) {
  std::string dir;
  const size_t slash = path.find_last_of('/');
  if (slash != std::string::npos) dir = path.substr(0, slash);
  if (!dir.empty()) {
    std::string cmd = "mkdir -p '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) return false;
  }
  char tmp[512];
  std::snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path.c_str(),
                static_cast<int>(::getpid()));
  FILE* f = std::fopen(tmp, "wb");
  if (f == nullptr) return false;
  const bool wrote = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  const bool closed = std::fclose(f) == 0;
  if (!wrote || !closed) {
    ::unlink(tmp);
    return false;
  }
  if (::rename(tmp, path.c_str()) != 0) {
    ::unlink(tmp);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Per-process modules.
// ---------------------------------------------------------------------------
struct Loaded {
  CUmodule module = nullptr;
  CUfunction fn = nullptr;
  int smem_set = -1;
  int regs = 0;
  int local_bytes = 0;
  int max_threads = 0;
};

std::mutex& mu() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, Loaded>& modules() {
  static std::unordered_map<std::string, Loaded> m;
  return m;
}

int g_compiles = 0;

// NVRTC has no default search path. The build bakes in the toolkit it was
// configured against (FATTN_JIT_CUDA_INCLUDE); the standard locations are
// probed as well so a library moved to a machine with a differently installed
// toolkit still finds <cuda_fp16.h>.
std::vector<std::string> include_dirs() {
  std::vector<std::string> dirs;
  auto add = [&dirs](std::string d) {
    if (d.empty()) return;
    for (const std::string& e : dirs)
      if (e == d) return;
    dirs.push_back(std::move(d));
  };
#ifdef FATTN_JIT_CUDA_INCLUDE
  std::string_view baked = FATTN_JIT_CUDA_INCLUDE;
  while (!baked.empty()) {
    const size_t colon = baked.find(':');
    add(std::string(baked.substr(0, colon)));
    if (colon == std::string_view::npos) break;
    baked.remove_prefix(colon + 1);
  }
#endif
  for (const char* d : {"/usr/local/cuda/include", "/usr/include"})
    if (::access(d, R_OK) == 0) add(d);
  return dirs;
}

// The options NVRTC is invoked with. One list, used both for the compile and
// for the artifact key: an option that changes the PTX but not the key (or the
// reverse) would let a stale image be loaded or force a needless recompile.
std::vector<std::string> compile_options() {
  std::vector<std::string> options;
  char arch[32];
  std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d",
                device_info().cc_major, device_info().cc_minor);
  options.emplace_back(arch);
  options.emplace_back("--std=c++17");
  for (const std::string& dir : include_dirs())
    options.push_back("--include-path=" + dir);
  return options;
}

// The NVRTC version, or "" when the library (or its version entry point) is
// missing. NVRTC is dlopen'd, so this can differ from the toolkit the library
// was built with; it decides which compiler produced a cached PTX.
std::string nvrtc_version() {
  Nvrtc& n = nvrtc();
  if (n.lib == nullptr || n.version == nullptr) return {};
  int major = 0, minor = 0;
  if (n.version(&major, &minor) != NVRTC_SUCCESS) return {};
  return std::to_string(major) + "." + std::to_string(minor);
}

// Binary compatibility of a cached PTX: the source text (which embeds the
// layout and schedule parameters), every compile option, and the compiler
// version. All three must match for the file to be the artifact this build
// would produce; anything else is a different kernel or a different code
// generator, never a reusable image.
std::string artifact_identity(const std::string& source) {
  std::string id = source;
  id += "\n--options--\n";
  for (const std::string& o : compile_options()) id += o + "\n";
  id += "--nvrtc--\n";
  id += nvrtc_version().empty() ? std::string("unknown") : nvrtc_version();
  return id;
}

// The PTX cache is keyed by artifact_identity(), and the file name also names
// the device architecture, so a kernel change, a compile-option change, an
// NVRTC update or a different device misses instead of loading a stale
// kernel. Written through a unique temporary file and rename: a concurrent
// process either sees the old file or the complete new one.
std::string cache_file(const Spec& s, const std::string& source) {
  const uint64_t h = fnv1a(artifact_identity(source));
  char name[160];
  std::snprintf(name, sizeof(name), "fattn-%016llx-sm%d.ptx",
                static_cast<unsigned long long>(h), device_info().arch());
  return cache_dir() + "/" + name;
}

std::string compile_ptx(const std::string& source, std::string* err) {
  Nvrtc& n = nvrtc();
  if (n.lib == nullptr) {
    *err = n.error;
    return {};
  }
  const std::vector<std::string> options = compile_options();
  std::vector<const char*> opts;
  opts.reserve(options.size());
  for (const std::string& o : options) opts.push_back(o.c_str());
  nvrtcProgram prog = nullptr;
  if (n.create(&prog, source.c_str(), "fattn_jit.cu", 0, nullptr, nullptr) !=
      NVRTC_SUCCESS) {
    *err = "nvrtcCreateProgram failed";
    return {};
  }
  const nvrtcResult rc = n.compile(prog, int(opts.size()), opts.data());
  std::string log;
  size_t log_size = 0;
  if (n.log_size(prog, &log_size) == NVRTC_SUCCESS && log_size > 1) {
    log.resize(log_size);
    if (n.log(prog, log.data()) == NVRTC_SUCCESS) {
      while (!log.empty() && (log.back() == '\0' || log.back() == '\n'))
        log.pop_back();
    }
  }
  std::string ptx;
  if (rc == NVRTC_SUCCESS) {
    size_t size = 0;
    if (n.ptx_size(prog, &size) == NVRTC_SUCCESS && size > 0) {
      ptx.resize(size);
      if (n.ptx(prog, ptx.data()) != NVRTC_SUCCESS) ptx.clear();
    }
  }
  n.destroy(&prog);
  if (ptx.empty()) {
    *err = "nvrtc compile failed";
    if (!log.empty()) *err += ": " + log;
  }
  return ptx;
}

}  // namespace

bool available() { return nvrtc().lib != nullptr && device_info().ok; }

std::string unavailable_reason() {
  if (nvrtc().lib == nullptr) return nvrtc().error;
  if (!device_info().ok) return device_info().error;
  return {};
}

std::string device_source() { return kFattnDeviceSource; }

uint64_t source_hash() { return fnv1a(device_source()); }

std::string nvrtc_version_string() {
  const std::string v = nvrtc_version();
  return v.empty() ? std::string("unavailable") : v;
}

bool prepare(const Spec& spec, std::string* err) {
  if (!available()) {
    *err = unavailable_reason();
    return false;
  }
  const std::string key = spec_string(spec) + "/sm" + std::to_string(device_info().arch());
  std::lock_guard<std::mutex> lock(mu());
  if (modules().count(key) != 0) return modules()[key].fn != nullptr;

  Loaded entry;
  const std::string source = wrapper_source(spec);
  const std::string path = cache_file(spec, source);
  std::string ptx;
  if (!read_file(path, &ptx)) {
    ptx = compile_ptx(source, err);
    ++g_compiles;
    if (ptx.empty()) {
      modules()[key] = entry;  // remembered as failed; not retried
      return false;
    }
    if (!write_file_atomic(path, ptx)) {
      // A cache write failure is not a compile failure: run from memory.
      std::fprintf(stderr,
                   "[boom_fattn] JIT: could not persist %s (running from "
                   "memory)\n",
                   path.c_str());
    }
  }
  // The runtime API owns the primary context; make sure it exists before the
  // driver API is used.
  ::cudaFree(nullptr);
  CUresult r = ::cuModuleLoadData(&entry.module, ptx.c_str());
  if (r != CUDA_SUCCESS) {
    *err = "cuModuleLoadData failed (" + std::to_string(int(r)) + ")";
    modules()[key] = Loaded{};
    return false;
  }
  const std::string name = "sb_fattn_jit_" + std::to_string(spec.id);
  r = ::cuModuleGetFunction(&entry.fn, entry.module, name.c_str());
  if (r != CUDA_SUCCESS) {
    *err = "cuModuleGetFunction(" + name + ") failed";
    ::cuModuleUnload(entry.module);
    modules()[key] = Loaded{};
    return false;
  }
  ::cuFuncGetAttribute(&entry.regs, CU_FUNC_ATTRIBUTE_NUM_REGS, entry.fn);
  ::cuFuncGetAttribute(&entry.local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                       entry.fn);
  ::cuFuncGetAttribute(&entry.max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                       entry.fn);
  modules()[key] = entry;
  return true;
}

bool ready(const Spec& spec) {
  const std::string key =
      spec_string(spec) + "/sm" + std::to_string(device_info().arch());
  std::lock_guard<std::mutex> lock(mu());
  auto it = modules().find(key);
  return it != modules().end() && it->second.fn != nullptr;
}

bool launch(const Spec& spec, dim3 grid, void* const args[12],
            cudaStream_t stream, size_t smem, int block) {
  const std::string key =
      spec_string(spec) + "/sm" + std::to_string(device_info().arch());
  std::lock_guard<std::mutex> lock(mu());
  auto it = modules().find(key);
  if (it == modules().end() || it->second.fn == nullptr) return false;
  Loaded& l = it->second;
  if (l.smem_set != int(smem)) {
    if (::cuFuncSetAttribute(l.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             int(smem)) != CUDA_SUCCESS)
      return false;
    l.smem_set = int(smem);
  }
  return ::cuLaunchKernel(l.fn, grid.x, grid.y, grid.z, unsigned(block), 1, 1,
                          unsigned(smem), stream,
                          const_cast<void**>(args), nullptr) == CUDA_SUCCESS;
}

bool attributes(const Spec& spec, int* regs, int* local_bytes,
                int* max_threads) {
  const std::string key =
      spec_string(spec) + "/sm" + std::to_string(device_info().arch());
  std::lock_guard<std::mutex> lock(mu());
  auto it = modules().find(key);
  if (it == modules().end() || it->second.fn == nullptr) return false;
  if (regs) *regs = it->second.regs;
  if (local_bytes) *local_bytes = it->second.local_bytes;
  if (max_threads) *max_threads = it->second.max_threads;
  return true;
}

std::string cache_path(const Spec& spec) {
  if (!available()) return {};
  return cache_file(spec, wrapper_source(spec));
}

int compile_count() {
  std::lock_guard<std::mutex> lock(mu());
  return g_compiles;
}

}  // namespace sonicboom::nn::cuda::fattn_jit
