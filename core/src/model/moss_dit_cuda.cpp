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

// Host side of the MOSS-Image CUDA DiT: weight loading (pread with two pinned
// staging buffers, or mmap fallback with madvise), Q8_0 quantization at load,
// and the per-block forward orchestration. The kernels and the CUDA API live
// in moss_dit_kernels.cu (nvcc, C++20); this TU is C++23 like the rest of the
// core. Every projection goes through the shared Q8_0 device matmul, whose
// device weight cache is populated here at load time.

#include <sonicboom/model/moss_dit_cuda.h>

#include <sonicboom/dtype.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/quant/quantized_tensor.h>
#include <sonicboom/safetensors/reader.h>

#include "../quant/ggml_derived/legacy_quants_q5_0_q8_0.h"
#include "../safetensors/json.h"
#include "moss_dit_kernels.h"
#include "weight_cache.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <vector>

// The cache lives in its own internal namespace; this TU reaches it by the
// bare name everywhere.
namespace weight_cache = sonicboom::model::weight_cache;

namespace sonicboom::moss {
namespace {

namespace cdit = cuda_dit;

constexpr float kRmsEps = 1e-6f;

std::string shape_string(std::span<const uint64_t> shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

std::expected<std::string, std::string> read_text(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return std::unexpected("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in.good() && !in.eof()) return std::unexpected("cannot read " + p.string());
  return ss.str();
}

bool env_is(const char* name, const char* value) {
  const char* got = std::getenv(name);
  return got != nullptr && std::strcmp(got, value) == 0;
}

// Convert a row range of a BF16 matrix to Q8_0. Row ranges are disjoint, so
// conversion fans out across cores with no synchronization: this is the 15 s
// single-threaded phase of the load.
void convert_q8_rows(const uint16_t* src, uint64_t cols,
                     quant::ggml_derived::block_q8_0* dst, uint64_t bpr,
                     uint64_t begin, uint64_t end) {
  std::vector<float> rowf(cols);
  for (uint64_t r = begin; r < end; ++r) {
    const uint16_t* in = src + r * cols;
    for (uint64_t i = 0; i < cols; ++i) rowf[i] = bf16_to_f32(in[i]);
    quant::ggml_derived::quantize_q8_0(rowf.data(), cols, dst + r * bpr);
  }
}

unsigned convert_threads() {
  const char* env = std::getenv("SONICBOOM_DIT_CONVERT_THREADS");
  if (env != nullptr && *env != '\0') {
    const long v = std::strtol(env, nullptr, 10);
    if (v >= 1 && v <= 64) return unsigned(v);
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return std::clamp(hw == 0 ? 1u : hw, 1u, 8u);
}

}  // namespace

struct CudaDiffusionTransformer::Impl {
  DitConfig cfg;
  std::unique_ptr<safetensors::ShardedReader> reader;
  int64_t weight_bytes = 0;

  struct QMat {
    std::vector<std::byte> bytes;  // permanent Q8_0 storage (pageable)
    quant::QuantizedTensor tensor;
    uint64_t rows = 0, cols = 0;
    std::vector<float> bias;
    void* d_bias = nullptr;
    bool loaded = false;
  };

  struct VideoLayer {
    QMat q, k, v, o, ffn_up_gate, ffn_down;
    std::vector<float> norm1, norm2, q_norm, k_norm;
    void* d_norm1 = nullptr;
    void* d_norm2 = nullptr;
    void* d_qn = nullptr;
    void* d_kn = nullptr;
  };

  struct TextLayer {
    QMat qkv, proj, ffn_up, ffn_gate, ffn_down;
    std::vector<float> norm1, norm2, q_norm, k_norm;
    void* d_norm1 = nullptr;
    void* d_norm2 = nullptr;
    void* d_qn = nullptr;
    void* d_kn = nullptr;
  };

  QMat time0, time2, shared_mod, head_mod;
  QMat video_embedder, text_proj, final_linear;
  std::vector<float> final_norm;
  void* d_final_norm = nullptr;
  std::vector<VideoLayer> layers;
  std::vector<TextLayer> text_layers;
  std::vector<float> rope_bands;

  struct DevBuf {
    void* p = nullptr;
    size_t n = 0;
    bool ensure(size_t bytes) {
      if (n >= bytes) return true;
      cdit::device_free(p);
      p = cdit::device_alloc(bytes);
      n = p != nullptr ? bytes : 0;
      return p != nullptr;
    }
    void release() {
      cdit::device_free(p);
      p = nullptr;
      n = 0;
    }
    ~DevBuf() { cdit::device_free(p); }
  };
  DevBuf d_x, d_y, d_qkv, d_q, d_k, d_v, d_attn, d_proj, d_ug, d_act, d_tx,
      d_textin, d_hidden, d_out, d_mod, d_headmod, d_rcos, d_rsin;

  // Device-resident weights hold ~5.4 GiB at Q8_0, which does not leave room
  // for the VAE decoder's peak at 512x512 on an 8 GiB part. decode() in the
  // pipeline calls release_device() first; the next forward re-uploads from the
  // host Q8_0 mirror (no re-quantization), so the cost is one H2D pass.
  bool weights_resident = true;

  bool upload_qmat(QMat& m) {
    const size_t bytes = m.bytes.size();
    void* d = cdit::device_alloc(bytes + 256);
    if (d == nullptr) return false;
    static const std::vector<std::byte> zeros(256, std::byte{0});
    if (!cdit::upload(static_cast<std::byte*>(d) + bytes, zeros.data(),
                      zeros.size()) ||
        !cdit::upload(d, m.bytes.data(), bytes) ||
        !quant::cuda::register_weight_device(m.tensor, d)) {
      cdit::device_free(d);
      return false;
    }
    m.loaded = true;
    return true;
  }

  void release_device() {
    if (!weights_resident) return;
    for (Task& t : tasks) {
      if (t.dst->loaded) {
        quant::cuda::unregister_weight_device(t.dst->tensor);
        t.dst->loaded = false;
      }
    }
    for (DevBuf* b : {&d_x, &d_y, &d_qkv, &d_q, &d_k, &d_v, &d_attn,
                      &d_proj, &d_ug, &d_act, &d_tx, &d_textin, &d_hidden,
                      &d_out, &d_mod, &d_headmod, &d_rcos, &d_rsin})
      b->release();
    cdit::sdpa_release_scratch();
    weights_resident = false;
  }

  std::expected<void, std::string> ensure_resident() {
    if (weights_resident) return {};
    for (Task& t : tasks) {
      if (!upload_qmat(*t.dst)) return std::unexpected("re-upload failed for " +
                                                      t.name);
    }
    weights_resident = true;
    return {};
  }

  static std::unexpected<std::string> fail(const char* what) {
    const char* err = cdit::last_error();
    return std::unexpected(std::string(what) +
                           (err[0] ? std::string(": ") + err : std::string()));
  }

  // --- configuration -------------------------------------------------------

  std::expected<void, std::string> load_config(const std::filesystem::path& dir) {
    auto text = read_text(dir / "config.json");
    if (!text) return std::unexpected(text.error());
    std::string error;
    auto root = safetensors::detail::parse_json(*text, &error);
    if (!root) return std::unexpected("config.json: " + error);
    auto int_of = [&](const char* key) -> std::expected<int64_t, std::string> {
      const auto* v = root->find(key);
      if (!v || !v->is_integer) {
        return std::unexpected(std::string("config.json: ") + key +
                               " missing or not an integer");
      }
      return v->integer;
    };
    for (auto [key, dest] :
         {std::pair{"dim", &cfg.dim},
          std::pair{"ffn_dim", &cfg.ffn_dim},
          std::pair{"num_heads", &cfg.num_heads},
          std::pair{"num_layers", &cfg.num_layers},
          std::pair{"text_dim", &cfg.text_dim},
          std::pair{"freq_dim", &cfg.freq_dim},
          std::pair{"latent_channels", &cfg.latent_channels}}) {
      auto got = int_of(key);
      if (!got) return std::unexpected(got.error());
      *dest = *got;
    }
    const auto* base = root->find("rope_base");
    if (base && base->type == safetensors::detail::JsonValue::Type::Number) {
      cfg.rope_base = base->number;
    }
    // The tiled SDPA supports head_dim 32..256 in steps of 32; this checkpoint
    // uses 128 (3072/24).
    if (cfg.dim % cfg.num_heads != 0 || (cfg.head_dim() % 32) != 0 ||
        cfg.head_dim() > 256) {
      return std::unexpected("config.json: bad head geometry");
    }
    if (cfg.freq_dim % 2 != 0) return std::unexpected("config.json: odd freq_dim");
    return {};
  }

  // --- small weights -------------------------------------------------------

  std::expected<std::vector<float>, std::string> f32_tensor(const std::string& name,
                                                            uint64_t n) {
    auto t = reader->tensor(name);
    if (!t) return std::unexpected("missing tensor " + name);
    auto d = reader->tensor_data(name);
    if (!d) return std::unexpected("unreadable tensor " + name);
    if (t->info.dtype != safetensors::DType::BF16 || t->info.n_elements != n) {
      return std::unexpected(name + ": expected BF16 [" + std::to_string(n) + "]");
    }
    const auto* p = reinterpret_cast<const uint16_t*>(d->data());
    std::vector<float> out(n);
    for (uint64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
    return out;
  }

  static void* upload_f32(const std::vector<float>& v) {
    if (v.empty()) return nullptr;
    void* p = cdit::device_alloc(v.size() * sizeof(float));
    if (p == nullptr) return nullptr;
    if (!cdit::upload(p, v.data(), v.size() * sizeof(float))) {
      cdit::device_free(p);
      return nullptr;
    }
    return p;
  }

  std::expected<void, std::string> load_bias(const std::string& name,
                                             uint64_t rows, QMat& m) {
    auto b = f32_tensor(name, rows);
    if (!b) return std::unexpected(b.error());
    m.bias = std::move(*b);
    m.d_bias = upload_f32(m.bias);
    if (m.d_bias == nullptr)
      return std::unexpected("bias upload failed for " + name);
    return {};
  }

  std::expected<void, std::string> load_norms() {
    const uint64_t d = uint64_t(cfg.dim), hd = uint64_t(cfg.head_dim());
    struct Field {
      const char* suffix;
      std::vector<float>* dst;
      void** device;
      uint64_t n;
    };
    for (int64_t l = 0; l < cfg.num_layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      VideoLayer& v = layers[uint64_t(l)];
      for (Field fld : {Field{"norm1.weight", &v.norm1, &v.d_norm1, d},
                        Field{"norm2.weight", &v.norm2, &v.d_norm2, d},
                        Field{"q_norm.weight", &v.q_norm, &v.d_qn, hd},
                        Field{"k_norm.weight", &v.k_norm, &v.d_kn, hd}}) {
        auto got = f32_tensor(p + fld.suffix, fld.n);
        if (!got) return std::unexpected(got.error());
        *fld.dst = std::move(*got);
        *fld.device = upload_f32(*fld.dst);
        if (*fld.device == nullptr) return std::unexpected("norm upload failed");
      }
    }
    for (int b = 0; b < 2; ++b) {
      const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
      TextLayer& t = text_layers[uint64_t(b)];
      for (Field fld : {Field{"norm1.weight", &t.norm1, &t.d_norm1, d},
                        Field{"norm2.weight", &t.norm2, &t.d_norm2, d},
                        Field{"q_norm.weight", &t.q_norm, &t.d_qn, hd},
                        Field{"k_norm.weight", &t.k_norm, &t.d_kn, hd}}) {
        auto got = f32_tensor(p + fld.suffix, fld.n);
        if (!got) return std::unexpected(got.error());
        *fld.dst = std::move(*got);
        *fld.device = upload_f32(*fld.dst);
        if (*fld.device == nullptr)
          return std::unexpected("text norm upload failed");
      }
    }
    auto fn = f32_tensor("final_norm_video.weight", d);
    if (!fn) return std::unexpected(fn.error());
    final_norm = std::move(*fn);
    d_final_norm = upload_f32(final_norm);
    if (d_final_norm == nullptr)
      return std::unexpected("final norm upload failed");

    const int64_t bands = cfg.head_dim() / 8;
    rope_bands.resize(uint64_t(bands));
    for (int64_t j = 0; j < bands; ++j) {
      rope_bands[uint64_t(j)] =
          1.0f / float(std::pow(cfg.rope_base, double(j) / double(bands)));
    }
    return {};
  }

  // --- weight loading ------------------------------------------------------

  struct Task {
    std::string name;
    uint64_t rows = 0, cols = 0;
    QMat* dst = nullptr;
    std::string bias;
    bool has_bias = false;
  };
  std::vector<Task> tasks;

  void add_task(std::string name, uint64_t rows, uint64_t cols, QMat& dst,
                std::string bias = {}) {
    Task t;
    t.name = std::move(name);
    t.rows = rows;
    t.cols = cols;
    t.dst = &dst;
    t.bias = std::move(bias);
    t.has_bias = !t.bias.empty();
    tasks.push_back(std::move(t));
  }

  void plan_tasks() {
    const uint64_t d = uint64_t(cfg.dim), f = uint64_t(cfg.ffn_dim);
    const uint64_t td = uint64_t(cfg.text_dim), ch = uint64_t(cfg.latent_channels);
    const uint64_t freq = uint64_t(cfg.freq_dim);
    add_task("time_embedding.0.weight", d, freq, time0, "time_embedding.0.bias");
    add_task("time_embedding.2.weight", d, d, time2, "time_embedding.2.bias");
    add_task("shared_block_modulation.weight", 6 * d, d, shared_mod,
             "shared_block_modulation.bias");
    add_task("video_head_modulation.weight", 2 * d, d, head_mod,
             "video_head_modulation.bias");
    add_task("video_embedder.weight", d, ch, video_embedder, "video_embedder.bias");
    add_task("text_embedder.proj.weight", d, td, text_proj,
             "text_embedder.proj.bias");
    text_layers.resize(2);
    for (int b = 0; b < 2; ++b) {
      const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
      TextLayer& t = text_layers[uint64_t(b)];
      add_task(p + "qkv.weight", 3 * d, d, t.qkv, p + "qkv.bias");
      add_task(p + "proj.weight", d, d, t.proj, p + "proj.bias");
      add_task(p + "ffn_up.weight", f, d, t.ffn_up, p + "ffn_up.bias");
      add_task(p + "ffn_gate.weight", f, d, t.ffn_gate, p + "ffn_gate.bias");
      add_task(p + "ffn_down.weight", d, f, t.ffn_down, p + "ffn_down.bias");
    }
    layers.resize(uint64_t(cfg.num_layers));
    for (int64_t l = 0; l < cfg.num_layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".proj.";
      VideoLayer& v = layers[uint64_t(l)];
      add_task(p + "q.weight", d, d, v.q);
      add_task(p + "k.weight", d, d, v.k);
      add_task(p + "v.weight", d, d, v.v);
      add_task(p + "o.weight", d, d, v.o);
      add_task(p + "ffn_up_gate.weight", 2 * f, d, v.ffn_up_gate);
      add_task(p + "ffn_down.weight", d, f, v.ffn_down);
    }
    add_task("final_linear_video.weight", ch, d, final_linear);
  }

  // Two pinned Q8 staging buffers: the async H2D copy out of one overlaps the
  // quantization of the next tensor into the other. A buffer is reused only
  // after the upload stream has drained (every other tensor), so an in-flight
  // copy never reads a buffer that is being overwritten.
  struct Q8Stage {
    void* buf[2] = {nullptr, nullptr};
    size_t cap[2] = {0, 0};
    int next = 0;
    int issued = 0;
    void* get(void* up, size_t bytes) {
      if (issued >= 2) {
        if (!cdit::stream_sync(up)) return nullptr;
        issued = 0;
      }
      const int idx = next;
      if (cap[idx] < bytes) {
        cdit::pinned_free(buf[idx]);
        buf[idx] = cdit::pinned_alloc(bytes);
        cap[idx] = buf[idx] != nullptr ? bytes : 0;
      }
      next ^= 1;
      return cap[idx] >= bytes ? buf[idx] : nullptr;
    }
    void note_issued() { ++issued; }
    ~Q8Stage() {
      cdit::pinned_free(buf[0]);
      cdit::pinned_free(buf[1]);
    }
  } q8_stage;

  std::expected<void, std::string> ingest_q8(const uint16_t* src,
                                             uint64_t elements, Task& task,
                                             void* up) {
    QMat& m = *task.dst;
    if (elements != task.rows * task.cols)
      return std::unexpected(task.name + ": element count mismatch");
    const uint64_t bpr = task.cols / 32;
    const size_t bytes = size_t(task.rows * bpr * 34);
    void* stage = q8_stage.get(up, bytes);
    if (stage == nullptr)
      return std::unexpected("pinned staging allocation for " + task.name);
    auto* dst = static_cast<quant::ggml_derived::block_q8_0*>(stage);
    const unsigned threads = convert_threads();
    std::vector<std::thread> workers;
    if (threads > 1 && task.rows > 1) {
      const uint64_t chunk = (task.rows + threads - 1) / threads;
      for (unsigned t = 1; t < threads; ++t) {
        const uint64_t begin = std::min(task.rows, uint64_t(t) * chunk);
        if (begin >= task.rows) break;
        const uint64_t end = std::min(task.rows, begin + chunk);
        workers.emplace_back(convert_q8_rows, src, task.cols, dst, bpr, begin,
                             end);
      }
      convert_q8_rows(src, task.cols, dst, bpr, 0,
                      std::min(task.rows, chunk));
      for (auto& w : workers) w.join();
    } else {
      convert_q8_rows(src, task.cols, dst, bpr, 0, task.rows);
    }
    // Permanent Q8 storage mirrors the staged bytes; the device copy is taken
    // from the pinned stage so it can proceed asynchronously.
    m.bytes.assign(static_cast<const std::byte*>(stage),
                   static_cast<const std::byte*>(stage) + bytes);
    m.rows = task.rows;
    m.cols = task.cols;
    m.tensor = quant::QuantizedTensor(quant::QuantType::Q8_0,
                                      {task.cols, task.rows}, m.bytes);
    if (!m.tensor.valid()) return std::unexpected(task.name + ": invalid Q8_0");
    // 256-byte zeroed tail: the MMQ kernel reads past the last row.
    void* d = cdit::device_alloc(bytes + 256);
    if (d == nullptr)
      return std::unexpected("device alloc failed for " + task.name);
    static const std::vector<std::byte> zeros(256, std::byte{0});
    if (!cdit::upload_async(static_cast<std::byte*>(d) + bytes, zeros.data(),
                            zeros.size(), up) ||
        !cdit::upload_async(d, stage, bytes, up)) {
      cdit::device_free(d);
      return std::unexpected("upload failed for " + task.name);
    }
    q8_stage.note_issued();
    if (!quant::cuda::register_weight_device(m.tensor, d)) {
      cdit::device_free(d);
      return std::unexpected("could not register " + task.name);
    }
    m.loaded = true;
    if (task.has_bias) {
      if (auto r = load_bias(task.bias, task.rows, m); !r)
        return std::unexpected(r.error());
    }
    return {};
  }

  std::expected<void, std::string> load_weights_pread(void* up) {
    // The converted bytes are device-independent: a cache hit replaces the
    // read-and-convert phases with a read of the cache file.
    if (auto cached = load_weights_cache(up)) {
      if (*cached) {
        if (!cdit::stream_sync(up))
          return std::unexpected("weight upload stream failed");
        return {};
      }
    } else {
      return std::unexpected(cached.error());
    }
    // A miss converts and then writes the cache. Hashing the checkpoint is a
    // full read of the same files the producer is reading; run it across the
    // conversion instead of adding it to the critical path.
    std::future<std::string> checkpoint_sha;
    if (weight_cache::directory())
      checkpoint_sha = std::async(std::launch::async, [this] {
        return weight_cache::checkpoint_sha256(shard_paths());
      });

    struct Entry {
      const safetensors::ShardedTensor* t;
      uint64_t offset;
      uint64_t bytes;
      Task* task;
    };
    std::vector<Entry> entries;
    entries.reserve(tasks.size());
    for (Task& task : tasks) {
      const safetensors::ShardedTensor* t = reader->tensor(task.name);
      if (!t) return std::unexpected("missing tensor " + task.name);
      if (t->info.dtype != safetensors::DType::BF16)
        return std::unexpected(task.name + ": expected BF16");
      if (t->info.shape.size() != 2 || t->info.shape[0] != task.rows ||
          t->info.shape[1] != task.cols) {
        return std::unexpected(task.name + ": shape " +
                               shape_string(t->info.shape) + " != [" +
                               std::to_string(task.rows) + ", " +
                               std::to_string(task.cols) + "]");
      }
      const auto& shard = reader->shards()[t->shard];
      entries.push_back(Entry{t, shard.file_offset(t->info),
                              t->info.n_elements * 2, &task});
    }

    std::vector<int> fds(reader->shards().size(), -1);
    for (size_t i = 0; i < fds.size(); ++i) {
      fds[i] = ::open(reader->shards()[i].path().c_str(), O_RDONLY | O_CLOEXEC);
      if (fds[i] < 0)
        return std::unexpected("open " + reader->shards()[i].path().string());
    }

    struct Stage {
      std::mutex m;
      std::condition_variable cv;
      int state[2] = {0, 0};  // 0 = producer may fill, 1 = consumer may read
      void* buf[2] = {nullptr, nullptr};
      size_t cap[2] = {0, 0};
      size_t bytes[2] = {0, 0};
      bool quit = false;
      std::string err;
    } stage;

    std::thread producer([&] {
      for (size_t i = 0; i < entries.size(); ++i) {
        const int idx = int(i & 1);
        {
          std::unique_lock<std::mutex> lk(stage.m);
          stage.cv.wait(lk, [&] { return stage.state[idx] == 0 || stage.quit; });
          if (stage.quit) return;
        }
        const Entry& e = entries[i];
        if (stage.cap[idx] < e.bytes) {
          cdit::pinned_free(stage.buf[idx]);
          stage.buf[idx] = cdit::pinned_alloc(e.bytes);
          stage.cap[idx] = stage.buf[idx] != nullptr ? e.bytes : 0;
          if (stage.buf[idx] == nullptr) {
            std::lock_guard<std::mutex> lk(stage.m);
            stage.err = "pinned allocation failed";
            stage.state[idx] = 1;
            stage.cv.notify_all();
            return;
          }
        }
        stage.bytes[idx] = e.bytes;
        const int fd = fds[e.t->shard];
        size_t done = 0;
        auto* dst = static_cast<std::byte*>(stage.buf[idx]);
        while (done < e.bytes) {
          const ssize_t got =
              ::pread(fd, dst + done, e.bytes - done, off_t(e.offset + done));
          if (got < 0) {
            if (errno == EINTR) continue;
            std::lock_guard<std::mutex> lk(stage.m);
            stage.err = "pread failed for " + e.task->name;
            stage.state[idx] = 1;
            stage.cv.notify_all();
            return;
          }
          if (got == 0) {
            std::lock_guard<std::mutex> lk(stage.m);
            stage.err = "short read for " + e.task->name;
            stage.state[idx] = 1;
            stage.cv.notify_all();
            return;
          }
          done += size_t(got);
        }
        {
          std::lock_guard<std::mutex> lk(stage.m);
          stage.state[idx] = 1;
        }
        stage.cv.notify_all();
      }
      std::lock_guard<std::mutex> lk(stage.m);
      stage.quit = true;
      stage.cv.notify_all();
    });

    for (size_t i = 0; i < entries.size(); ++i) {
      const int idx = int(i & 1);
      std::string stage_err;
      {
        std::unique_lock<std::mutex> lk(stage.m);
        stage.cv.wait(lk, [&] { return stage.state[idx] == 1; });
        stage_err = stage.err;
      }
      std::string err;
      if (!stage_err.empty()) {
        err = stage_err;
      } else {
        Entry& e = entries[i];
        const auto* src = static_cast<const uint16_t*>(stage.buf[idx]);
        if (auto r = ingest_q8(src, e.bytes / 2, *e.task, up); !r) err = r.error();
      }
      if (!err.empty()) {
        {
          std::lock_guard<std::mutex> lk(stage.m);
          stage.quit = true;
        }
        stage.cv.notify_all();
        producer.join();
        return std::unexpected(err);
      }
      {
        std::lock_guard<std::mutex> lk(stage.m);
        stage.state[idx] = 0;
      }
      stage.cv.notify_all();
    }
    producer.join();
    for (int fd : fds) {
      if (fd >= 0) ::close(fd);
    }
    {
      std::lock_guard<std::mutex> lk(stage.m);
      if (!stage.err.empty()) return std::unexpected(stage.err);
    }
    for (void* p : stage.buf) cdit::pinned_free(p);
    if (!cdit::stream_sync(up)) return std::unexpected("weight upload stream failed");
    if (checkpoint_sha.valid()) write_weight_cache(checkpoint_sha.get());
    return {};
  }

  // Fallback: read through the reader's mapping, dropping pages as we go.
  std::expected<void, std::string> load_weights_mmap(void* up) {
    const long page = ::sysconf(_SC_PAGESIZE);
    for (size_t i = 0; i < tasks.size(); ++i) {
      Task& task = tasks[i];
      if (i + 1 < tasks.size()) {
        if (auto next = reader->tensor_data(tasks[i + 1].name)) {
          const uintptr_t nb = reinterpret_cast<uintptr_t>(next->data());
          void* na = reinterpret_cast<void*>(nb & ~uintptr_t(page - 1));
          ::madvise(na, next->size(), MADV_WILLNEED);
        }
      }
      auto data = reader->tensor_data(task.name);
      if (!data) return std::unexpected("unreadable tensor " + task.name);
      if (data->size() != task.rows * task.cols * 2)
        return std::unexpected(task.name + ": size mismatch");
      const auto* src = reinterpret_cast<const uint16_t*>(data->data());
      if (auto r = ingest_q8(src, data->size() / 2, task, up); !r)
        return std::unexpected(r.error());
      const uintptr_t base = reinterpret_cast<uintptr_t>(data->data());
      void* aligned = reinterpret_cast<void*>(base & ~uintptr_t(page - 1));
      size_t span = size_t(((base + data->size() + page - 1) &
                            ~uintptr_t(page - 1)) -
                           uintptr_t(aligned));
      ::madvise(aligned, span, MADV_DONTNEED);
    }
    if (!cdit::stream_sync(up)) return std::unexpected("weight upload stream failed");
    return {};
  }

  // --- converted-weight cache ----------------------------------------------

  std::vector<std::filesystem::path> shard_paths() const {
    std::vector<std::filesystem::path> out;
    out.reserve(reader->shards().size());
    for (const auto& s : reader->shards()) out.push_back(s.path());
    return out;
  }

  std::expected<void, std::string> ingest_cached(
      const weight_cache::CacheFile& file,
      const weight_cache::TensorRecord& rec, Task& task, void* up) {
    QMat& m = *task.dst;
    const size_t bytes = size_t(rec.q8_bytes);
    void* stage = q8_stage.get(up, bytes);
    if (stage == nullptr)
      return std::unexpected("pinned staging allocation for " + task.name);
    std::span<std::byte> staged(static_cast<std::byte*>(stage), bytes);
    m.bias.clear();
    if (!weight_cache::read_tensor(file, rec, staged,
                                   task.has_bias ? &m.bias : nullptr))
      return std::unexpected(task.name +
                             ": cache payload failed verification (remove " +
                             file.path.string() + " to rebuild)");
    m.bytes.assign(staged.begin(), staged.end());
    m.rows = task.rows;
    m.cols = task.cols;
    m.tensor = quant::QuantizedTensor(quant::QuantType::Q8_0,
                                      {task.cols, task.rows}, m.bytes);
    if (!m.tensor.valid()) return std::unexpected(task.name + ": invalid Q8_0");
    void* d = cdit::device_alloc(bytes + 256);
    if (d == nullptr)
      return std::unexpected("device alloc failed for " + task.name);
    static const std::vector<std::byte> zeros(256, std::byte{0});
    if (!cdit::upload_async(static_cast<std::byte*>(d) + bytes, zeros.data(),
                            zeros.size(), up) ||
        !cdit::upload_async(d, stage, bytes, up)) {
      cdit::device_free(d);
      return std::unexpected("upload failed for " + task.name);
    }
    q8_stage.note_issued();
    if (!quant::cuda::register_weight_device(m.tensor, d)) {
      cdit::device_free(d);
      return std::unexpected("could not register " + task.name);
    }
    m.loaded = true;
    if (task.has_bias) {
      m.d_bias = upload_f32(m.bias);
      if (m.d_bias == nullptr)
        return std::unexpected("bias upload failed for " + task.name);
    }
    return {};
  }

  // Try the converted-weight cache. Returns true when every matrix was loaded,
  // false when there is no usable cache (the conversion path then runs). A
  // non-empty error means a cache entry failed verification; that is reported
  // rather than silently re-converted.
  std::expected<bool, std::string> load_weights_cache(void* up) {
    if (!weight_cache::directory()) return false;
    const auto shards = shard_paths();
    auto file = weight_cache::open_existing(shards);
    if (!file) return false;
    for (const Task& task : tasks) {
      const auto* rec = file->find(task.name);
      if (rec == nullptr || rec->rows != task.rows || rec->cols != task.cols ||
          (task.has_bias && rec->bias_count != task.rows))
        return false;  // format or geometry mismatch: rebuild
    }
    for (Task& task : tasks) {
      const auto* rec = file->find(task.name);
      if (auto r = ingest_cached(*file, *rec, task, up); !r)
        return std::unexpected(r.error());
    }
    return true;
  }

  void write_weight_cache(const std::string& sha) {
    if (sha.size() != 64) return;
    const auto shards = shard_paths();
    std::vector<weight_cache::WriteTensor> tensors;
    tensors.reserve(tasks.size());
    for (const Task& task : tasks) {
      weight_cache::WriteTensor w;
      w.name = task.name;
      w.rows = task.rows;
      w.cols = task.cols;
      w.q8 = std::span<const std::byte>(task.dst->bytes);
      if (task.has_bias) w.bias = std::span<const float>(task.dst->bias);
      tensors.push_back(std::move(w));
    }
    auto r = weight_cache::store(shards, tensors, sha);
    if (r)
      std::fprintf(stderr, "sonicboom: weight cache: %s\n", r->c_str());
    else
      std::fprintf(stderr, "sonicboom: weight cache not written: %s\n",
                   r.error().c_str());
  }

  // --- forward -------------------------------------------------------------

  bool mm(const QMat& w, const float* X, float* Y, int n) {
    return quant::cuda::matmul_f32_dev(w.tensor, X, Y, n);
  }

  bool add_bias(const QMat& w, float* y, int rows) {
    return cdit::bias_add(y, static_cast<const float*>(w.d_bias), int(w.rows),
                          long(rows) * long(w.rows));
  }

  bool ensure_buffers(int64_t tokens, int64_t text_len) {
    const size_t d = size_t(cfg.dim), f = size_t(cfg.ffn_dim);
    const size_t half = size_t(cfg.head_dim() / 2);
    const size_t tsz = size_t(tokens);
    return d_x.ensure(tsz * d * 4) && d_y.ensure(tsz * d * 4) &&
           d_qkv.ensure(tsz * 3 * d * 4) && d_q.ensure(tsz * d * 4) &&
           d_k.ensure(tsz * d * 4) && d_v.ensure(tsz * d * 4) &&
           d_attn.ensure(tsz * d * 4) &&
           d_proj.ensure(tsz * d * 4) && d_ug.ensure(tsz * 2 * f * 4) &&
           d_act.ensure(tsz * f * 4) &&
           d_tx.ensure(size_t(text_len) * d * 4) &&
           d_textin.ensure(size_t(text_len) * size_t(cfg.text_dim) * 4) &&
           d_hidden.ensure(tsz * size_t(cfg.latent_channels) * 4) &&
           d_out.ensure(tsz * size_t(cfg.latent_channels) * 4) &&
           d_mod.ensure(6 * d * 4) && d_headmod.ensure(2 * d * 4) &&
           d_rcos.ensure(tsz * half * 4) && d_rsin.ensure(tsz * half * 4);
  }

  bool text_blocks(int64_t text_len, int64_t rope_offset) {
    const int d = int(cfg.dim), hd = int(cfg.head_dim());
    const int heads = int(cfg.num_heads), f = int(cfg.ffn_dim);
    const int n = int(text_len);
    const long total = long(n) * d;
    const float scale = 1.0f / std::sqrt(float(hd));
    float* qkv = static_cast<float*>(d_qkv.p);
    // The split writes three separate buffers: splitting in place over `qkv`
    // would let one thread's write race another thread's read of the source.
    float* q = static_cast<float*>(d_q.p);
    float* k = static_cast<float*>(d_k.p);
    float* v = static_cast<float*>(d_v.p);
    const float* rcos = static_cast<const float*>(d_rcos.p) + rope_offset * (hd / 2);
    const float* rsin = static_cast<const float*>(d_rsin.p) + rope_offset * (hd / 2);
    const long htotal = long(n) * heads;
    const long rtotal = long(n) * heads * (hd / 2);
    for (int b = 0; b < 2; ++b) {
      TextLayer& t = text_layers[uint64_t(b)];
      if (!cdit::rms_norm_rows(static_cast<const float*>(d_tx.p),
                               static_cast<const float*>(t.d_norm1), kRmsEps,
                               static_cast<float*>(d_y.p), d, n))
        return false;
      if (!mm(t.qkv, static_cast<const float*>(d_y.p), qkv, n)) return false;
      if (!add_bias(t.qkv, qkv, n)) return false;
      if (!cdit::qkv_split(qkv, q, k, v, d, total)) return false;
      if (!cdit::rms_norm_heads(q, static_cast<const float*>(t.d_qn), kRmsEps, hd,
                                htotal) ||
          !cdit::rms_norm_heads(k, static_cast<const float*>(t.d_kn), kRmsEps, hd,
                                htotal))
        return false;
      if (!cdit::rope_table(q, rcos, rsin, heads, hd, rtotal) ||
          !cdit::rope_table(k, rcos, rsin, heads, hd, rtotal))
        return false;
      if (!cdit::sdpa_full(q, k, v, n, heads, hd, scale,
                           static_cast<float*>(d_attn.p)))
        return false;
      if (!mm(t.proj, static_cast<const float*>(d_attn.p),
              static_cast<float*>(d_proj.p), n))
        return false;
      if (!add_bias(t.proj, static_cast<float*>(d_proj.p), n)) return false;
      if (!cdit::add(static_cast<float*>(d_tx.p),
                     static_cast<const float*>(d_proj.p), total))
        return false;

      if (!cdit::rms_norm_rows(static_cast<const float*>(d_tx.p),
                               static_cast<const float*>(t.d_norm2), kRmsEps,
                               static_cast<float*>(d_y.p), d, n))
        return false;
      if (!mm(t.ffn_up, static_cast<const float*>(d_y.p),
              static_cast<float*>(d_ug.p), n))
        return false;
      if (!add_bias(t.ffn_up, static_cast<float*>(d_ug.p), n)) return false;
      if (!mm(t.ffn_gate, static_cast<const float*>(d_y.p),
              static_cast<float*>(d_act.p), n))
        return false;
      if (!add_bias(t.ffn_gate, static_cast<float*>(d_act.p), n)) return false;
      if (!cdit::situ_glu_split(static_cast<const float*>(d_ug.p),
                                static_cast<const float*>(d_act.p),
                                static_cast<float*>(d_act.p), long(n) * f))
        return false;
      if (!mm(t.ffn_down, static_cast<const float*>(d_act.p),
              static_cast<float*>(d_proj.p), n))
        return false;
      if (!add_bias(t.ffn_down, static_cast<float*>(d_proj.p), n)) return false;
      if (!cdit::add(static_cast<float*>(d_tx.p),
                     static_cast<const float*>(d_proj.p), total))
        return false;
    }
    return true;
  }

  bool video_block(int64_t l, int64_t tokens) {
    const int d = int(cfg.dim), hd = int(cfg.head_dim());
    const int heads = int(cfg.num_heads), f = int(cfg.ffn_dim);
    const int n = int(tokens);
    const long total = long(n) * d;
    const float scale = 1.0f / std::sqrt(float(hd));
    VideoLayer& v = layers[uint64_t(l)];
    float* x = static_cast<float*>(d_x.p);
    float* y = static_cast<float*>(d_y.p);
    float* qkv = static_cast<float*>(d_qkv.p);
    float* q = qkv;
    float* k = qkv + long(n) * d;
    float* vv = k + long(n) * d;
    const float* mod = static_cast<const float*>(d_mod.p);
    const float* rcos = static_cast<const float*>(d_rcos.p);
    const float* rsin = static_cast<const float*>(d_rsin.p);
    const long htotal = long(n) * heads;
    const long rtotal = long(n) * heads * (hd / 2);

    if (!cdit::rms_norm_rows(x, static_cast<const float*>(v.d_norm1), kRmsEps, y,
                             d, n))
      return false;
    if (!cdit::modulate(y, mod, mod + d, d, total)) return false;
    if (!mm(v.q, y, q, n) || !mm(v.k, y, k, n) || !mm(v.v, y, vv, n))
      return false;
    if (!cdit::rms_norm_heads(q, static_cast<const float*>(v.d_qn), kRmsEps, hd,
                              htotal) ||
        !cdit::rms_norm_heads(k, static_cast<const float*>(v.d_kn), kRmsEps, hd,
                              htotal))
      return false;
    if (!cdit::rope_table(q, rcos, rsin, heads, hd, rtotal) ||
        !cdit::rope_table(k, rcos, rsin, heads, hd, rtotal))
      return false;
    if (!cdit::sdpa_full(q, k, vv, n, heads, hd, scale,
                         static_cast<float*>(d_attn.p)))
      return false;
    if (!mm(v.o, static_cast<const float*>(d_attn.p),
            static_cast<float*>(d_proj.p), n))
      return false;
    if (!cdit::gated_add(x, static_cast<const float*>(d_proj.p), mod + 2 * d, d,
                         total))
      return false;

    if (!cdit::rms_norm_rows(x, static_cast<const float*>(v.d_norm2), kRmsEps, y,
                             d, n))
      return false;
    if (!cdit::modulate(y, mod + 3 * d, mod + 4 * d, d, total)) return false;
    if (!mm(v.ffn_up_gate, y, static_cast<float*>(d_ug.p), n)) return false;
    if (!cdit::situ_glu_fused(static_cast<const float*>(d_ug.p),
                              static_cast<float*>(d_act.p), f, long(n) * f))
      return false;
    if (!mm(v.ffn_down, static_cast<const float*>(d_act.p),
            static_cast<float*>(d_proj.p), n))
      return false;
    return cdit::gated_add(x, static_cast<const float*>(d_proj.p), mod + 5 * d,
                           d, total);
  }

  std::expected<DitSample, std::string> forward(
      std::span<const float> hidden, int64_t height, int64_t width,
      float timestep, std::span<const float> text, int64_t text_len,
      const std::vector<int64_t>& skip_layers) {
    const int64_t dim = cfg.dim, hd = cfg.head_dim();
    const int64_t channels = cfg.latent_channels;
    if (height <= 0 || width <= 0)
      return std::unexpected("forward: bad latent grid");
    if (text_len <= 0) return std::unexpected("forward: empty text sequence");
    const int64_t hw = height * width;
    const int64_t tokens = hw + text_len;
    if (hidden.size() < size_t(channels * hw))
      return std::unexpected("forward: hidden_states shorter than [C][H][W]");
    if (text.size() < size_t(text_len * cfg.text_dim))
      return std::unexpected("forward: encoder_hidden_states too short");
    for (int64_t l : skip_layers) {
      if (l < 0 || l >= cfg.num_layers)
        return std::unexpected("forward: skip_layers out of range");
    }
    if (auto r = ensure_resident(); !r) return std::unexpected(r.error());
    if (!ensure_buffers(tokens, text_len))
      return fail("device scratch allocation");

    // Time embedding: fp64 sinusoid, f32 MLP (Linear -> SiLU -> Linear), all on
    // the host: four tiny matvecs through the same Q8_0 weights.
    const int64_t half = cfg.freq_dim / 2;
    std::vector<float> temb(size_t(cfg.freq_dim));
    for (int64_t i = 0; i < half; ++i) {
      const double freq = std::pow(10000.0, -double(i) / double(half));
      const double phase = double(timestep) * freq;
      temb[size_t(i)] = float(std::cos(phase));
      temb[size_t(half + i)] = float(std::sin(phase));
    }
    std::vector<float> t1(static_cast<size_t>(dim), 0.0f),
        t2(static_cast<size_t>(dim), 0.0f);
    std::vector<float> mod_host(size_t(6 * dim)), headmod_host(size_t(2 * dim));
    if (!quant::cuda::matvec_f32(time0.tensor, temb, t1))
      return std::unexpected("time_embedding.0 matvec failed");
    if (time0.d_bias != nullptr) {
      for (int64_t i = 0; i < dim; ++i) t1[size_t(i)] += time0.bias[size_t(i)];
    }
    for (int64_t i = 0; i < dim; ++i) {
      const float v = t1[size_t(i)];
      t1[size_t(i)] = v * (1.0f / (1.0f + std::exp(-v)));
    }
    if (!quant::cuda::matvec_f32(time2.tensor, t1, t2))
      return std::unexpected("time_embedding.2 matvec failed");
    if (time2.d_bias != nullptr) {
      for (int64_t i = 0; i < dim; ++i) t2[size_t(i)] += time2.bias[size_t(i)];
    }
    if (!quant::cuda::matvec_f32(shared_mod.tensor, t2, mod_host))
      return std::unexpected("shared_block_modulation matvec failed");
    if (shared_mod.d_bias != nullptr) {
      for (int64_t i = 0; i < 6 * dim; ++i)
        mod_host[size_t(i)] += shared_mod.bias[size_t(i)];
    }
    if (!quant::cuda::matvec_f32(head_mod.tensor, t2, headmod_host))
      return std::unexpected("video_head_modulation matvec failed");
    if (head_mod.d_bias != nullptr) {
      for (int64_t i = 0; i < 2 * dim; ++i)
        headmod_host[size_t(i)] += head_mod.bias[size_t(i)];
    }

    // 4-axis half-split RoPE tables, host fp64 -> f32, as the reference.
    const int64_t bands = hd / 8;
    const int64_t stride = hd / 2;
    std::vector<float> rcos(size_t(tokens * stride)),
        rsin(size_t(tokens * stride));
    for (int64_t t = 0; t < tokens; ++t) {
      double coord[4] = {0.0, 0.0, 0.0, 0.0};
      if (t < hw) {
        const int64_t ih = t / width, iw = t % width;
        coord[1] = double(text_len);
        coord[2] = double(ih) - double(height - 1) / 2.0;
        coord[3] = double(iw) - double(width - 1) / 2.0;
      } else {
        coord[1] = double(t - hw);
      }
      for (int64_t axis = 0; axis < 4; ++axis) {
        for (int64_t j = 0; j < bands; ++j) {
          const double angle = coord[axis] * double(rope_bands[size_t(j)]);
          rcos[size_t(t * stride + axis * bands + j)] = float(std::cos(angle));
          rsin[size_t(t * stride + axis * bands + j)] = float(std::sin(angle));
        }
      }
    }
    if (!cdit::upload(d_mod.p, mod_host.data(), mod_host.size() * 4) ||
        !cdit::upload(d_headmod.p, headmod_host.data(), headmod_host.size() * 4) ||
        !cdit::upload(d_rcos.p, rcos.data(), rcos.size() * 4) ||
        !cdit::upload(d_rsin.p, rsin.data(), rsin.size() * 4) ||
        !cdit::upload(d_hidden.p, hidden.data(), size_t(channels * hw) * 4) ||
        !cdit::upload(d_textin.p, text.data(),
                      size_t(text_len * cfg.text_dim) * 4))
      return fail("input upload");

    // Video tokens [hw, channels] -> y[0:hw] (a separate buffer: the matmul
    // reads its input while writing x); text tokens into tx.
    if (!cdit::video_to_tokens(static_cast<const float*>(d_hidden.p),
                               static_cast<float*>(d_y.p), int(channels), int(hw)))
      return fail("video embed transpose");
    if (!mm(video_embedder, static_cast<const float*>(d_y.p),
            static_cast<float*>(d_x.p), int(hw)))
      return std::unexpected("video_embedder matmul failed");
    if (!add_bias(video_embedder, static_cast<float*>(d_x.p), int(hw)))
      return fail("video_embedder bias");
    if (!mm(text_proj, static_cast<const float*>(d_textin.p),
            static_cast<float*>(d_tx.p), int(text_len)))
      return std::unexpected("text_embedder.proj matmul failed");
    if (!add_bias(text_proj, static_cast<float*>(d_tx.p), int(text_len)))
      return fail("text_embedder.proj bias");
    if (!text_blocks(text_len, hw)) return fail("text blocks");
    if (!cdit::copy_d2d(static_cast<float*>(d_x.p) + hw * dim, d_tx.p,
                        size_t(text_len * dim) * 4))
      return fail("text token copy");

    for (int64_t l = 0; l < cfg.num_layers; ++l) {
      const bool skip =
          std::find(skip_layers.begin(), skip_layers.end(), l) != skip_layers.end();
      if (!skip && !video_block(l, tokens)) return fail("video block");
    }

    if (!cdit::rms_norm_rows(static_cast<const float*>(d_x.p),
                             static_cast<const float*>(d_final_norm), kRmsEps,
                             static_cast<float*>(d_y.p), int(dim), hw))
      return fail("final norm");
    if (!cdit::modulate(static_cast<float*>(d_y.p),
                        static_cast<const float*>(d_headmod.p),
                        static_cast<const float*>(d_headmod.p) + dim, int(dim),
                        long(hw * dim)))
      return fail("final modulation");
    if (!mm(final_linear, static_cast<const float*>(d_y.p),
            static_cast<float*>(d_out.p), int(hw)))
      return std::unexpected("final_linear_video matmul failed");

    DitSample sample;
    sample.height = height;
    sample.width = width;
    sample.values.resize(size_t(channels * hw));
    if (!cdit::tokens_to_video(static_cast<const float*>(d_out.p),
                               static_cast<float*>(d_hidden.p), int(channels),
                               int(hw)))
      return fail("latent transpose");
    if (!cdit::download(sample.values.data(), d_hidden.p, sample.values.size() * 4))
      return fail("latent download");
    return sample;
  }

  ~Impl() {
    for (Task& t : tasks) {
      if (t.dst->loaded) quant::cuda::unregister_weight_device(t.dst->tensor);
    }
    for (QMat* p : {&time0, &time2, &shared_mod, &head_mod, &video_embedder,
                    &text_proj, &final_linear}) {
      cdit::device_free(p->d_bias);
    }
    for (VideoLayer& v : layers) {
      for (QMat* p : {&v.q, &v.k, &v.v, &v.o, &v.ffn_up_gate, &v.ffn_down})
        cdit::device_free(p->d_bias);
      for (void* p : {v.d_norm1, v.d_norm2, v.d_qn, v.d_kn})
        cdit::device_free(p);
    }
    for (TextLayer& t : text_layers) {
      for (QMat* p : {&t.qkv, &t.proj, &t.ffn_up, &t.ffn_gate, &t.ffn_down})
        cdit::device_free(p->d_bias);
      for (void* p : {t.d_norm1, t.d_norm2, t.d_qn, t.d_kn})
        cdit::device_free(p);
    }
    cdit::device_free(d_final_norm);
  }
};

bool dit_cuda_available() noexcept { return cdit::device_available(); }

std::expected<std::unique_ptr<CudaDiffusionTransformer>, std::string>
CudaDiffusionTransformer::load(const std::filesystem::path& dir) {
  if (!dit_cuda_available()) return std::unexpected("CUDA is not available");
  auto impl = std::unique_ptr<Impl>(new Impl());
  if (auto r = impl->load_config(dir); !r) return std::unexpected(r.error());

  auto reader = safetensors::ShardedReader::load(
      dir / "diffusion_pytorch_model.safetensors.index.json");
  if (!reader) return std::unexpected("safetensors: " + reader.error().message);
  impl->reader = std::make_unique<safetensors::ShardedReader>(std::move(*reader));
  impl->plan_tasks();
  // Report the Q8_0 storage the object actually holds, not the checkpoint's
  // bf16 size: rows * (cols/32) blocks of 34 bytes, plus f32 biases.
  for (const auto& t : impl->tasks) {
    impl->weight_bytes += int64_t(t.rows * (t.cols / 32) * 34);
    if (t.has_bias) impl->weight_bytes += int64_t(t.rows * 4);
  }
  if (auto r = impl->load_norms(); !r) return std::unexpected(r.error());

  void* up = cdit::stream_create();
  if (up == nullptr) return std::unexpected("could not create upload stream");
  std::expected<void, std::string> r =
      env_is("SONICBOOM_MOSS_LOAD", "mmap") ? impl->load_weights_mmap(up)
                                           : impl->load_weights_pread(up);
  cdit::stream_destroy(up);
  if (!r) return std::unexpected(r.error());
  return std::unique_ptr<CudaDiffusionTransformer>(
      new CudaDiffusionTransformer(std::move(impl)));
}

CudaDiffusionTransformer::CudaDiffusionTransformer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CudaDiffusionTransformer::~CudaDiffusionTransformer() = default;

const DitConfig& CudaDiffusionTransformer::config() const noexcept {
  return impl_->cfg;
}

int64_t CudaDiffusionTransformer::weight_bytes() const noexcept {
  return impl_->weight_bytes;
}

void CudaDiffusionTransformer::release_device() { impl_->release_device(); }

std::expected<DitSample, std::string> CudaDiffusionTransformer::forward(
    std::span<const float> hidden, int64_t height, int64_t width, float timestep,
    std::span<const float> text, int64_t text_len,
    const std::vector<int64_t>& skip_layers) {
  return impl_->forward(hidden, height, width, timestep, text, text_len,
                        skip_layers);
}

}  // namespace sonicboom::moss
