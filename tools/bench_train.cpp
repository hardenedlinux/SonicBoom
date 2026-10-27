// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// bench_train — time one training step of the traced tiny transformer through
// the stable C ABI, in the two call shapes the tracer can use:
//
//   full      every input uploaded, every output downloaded, every step
//             (the executor's original behaviour; the tracer writes the
//             updated leaves back and binds them again)
//   resident  the optimizer's parameters/moments stay on the device; only the
//             batch is uploaded and only the loss is downloaded
//
// Both shapes run the same compiled graph from the same initial state with the
// same per-step batch, so their loss sequences must be bit-identical. A CUDA
// graph is captured by the core when SONICBOOM_TRAIN_GRAPH is not "0"; the
// numbers below therefore separate resident state from graph replay.
//
//   bench_train [--fixture PATH] [--gate-steps N] [--timed-steps N] [--cpu]
//               [--loss-trace]
//
// Prints, per mode: ms/step, host->device and device->host bytes per step, and
// an FNV-1a hash of the loss sequence. `--loss-trace` prints each step's loss
// on stderr, which is what the CPU/device numerical comparison reads.

#include <sonicboom/capi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Err {
  sb_error *e = nullptr;
  ~Err() { if (e) sb_error_free(e); }
  std::string text() const {
    return e ? sb_error_message(e) : std::string("(no message)");
  }
};

[[noreturn]] void die(const std::string &msg) {
  std::fprintf(stderr, "bench_train: %s\n", msg.c_str());
  std::exit(1);
}

std::string read_file(const char *path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die(std::string("cannot open ") + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct TensorInfo {
  std::string name;
  sb_dtype dtype = SB_DTYPE_FLOAT32;
  std::uint64_t bytes = 0;
  std::vector<std::int64_t> dims;
};

std::vector<TensorInfo> query(sb_executable *exe, bool inputs) {
  size_t n = 0;
  Err err;
  const sb_status st = inputs ? sb_input_count(exe, &n) : sb_output_count(exe, &n);
  if (st != SB_OK) die("count: " + err.text());
  std::vector<TensorInfo> out;
  for (size_t i = 0; i < n; ++i) {
    TensorInfo t;
    sb_dtype dtype;
    std::int64_t rank = 0;
    const std::int64_t *dims = nullptr;
    std::uint64_t bytes = 0;
    const sb_status si = inputs
        ? sb_input_info_at(exe, i, &dtype, &rank, &dims, &bytes)
        : sb_output_info_at(exe, i, &dtype, &rank, &dims, &bytes);
    if (si != SB_OK) die("info: " + err.text());
    const char *name = nullptr;
    const sb_status sn = inputs ? sb_input_name_at(exe, i, &name)
                                : sb_output_name_at(exe, i, &name);
    if (sn != SB_OK) die("name: " + err.text());
    t.name = name ? name : "";
    t.dtype = dtype;
    t.bytes = bytes;
    if (dims) t.dims.assign(dims, dims + rank);
    out.push_back(std::move(t));
  }
  return out;
}

// xorshift32 — a fixed, portable sequence so two runs see the same numbers.
struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
  std::uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float uniform(float lo, float hi) {
    const float u = static_cast<float>(next() >> 8) / 16777216.0f;
    return lo + (hi - lo) * u;
  }
};

std::uint32_t name_seed(const std::string &name) {
  std::uint32_t h = 2166136261u;
  for (unsigned char c : name) h = (h ^ c) * 16777619u;
  return h;
}

bool is_batch(const TensorInfo &t) {
  // The traced step's batch: token ids and targets, [T] int32.
  return t.dtype == SB_DTYPE_INT32 && t.dims.size() == 1;
}

bool is_step_counter(const TensorInfo &t) {
  return t.dtype == SB_DTYPE_INT32 && t.dims.empty();
}

void fill_initial(const std::vector<TensorInfo> &ins,
                  std::vector<std::vector<std::uint8_t>> &bufs) {
  bufs.resize(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    bufs[i].assign(ins[i].bytes, 0);
    if (is_batch(ins[i]) || is_step_counter(ins[i])) continue;  // per step
    if (ins[i].dtype != SB_DTYPE_FLOAT32) continue;
    Rng rng(name_seed(ins[i].name));
    float *p = reinterpret_cast<float *>(bufs[i].data());
    const size_t n = ins[i].bytes / sizeof(float);
    for (size_t k = 0; k < n; ++k) p[k] = rng.uniform(-0.05f, 0.05f);
  }
}

void set_batch(const std::vector<TensorInfo> &ins,
               std::vector<std::vector<std::uint8_t>> &bufs,
               std::uint32_t step, std::uint32_t vocab) {
  for (size_t i = 0; i < ins.size(); ++i) {
    if (is_step_counter(ins[i])) {
      const std::int32_t v = static_cast<std::int32_t>(step);
      std::memcpy(bufs[i].data(), &v, sizeof(v));
    } else if (is_batch(ins[i])) {
      Rng rng(name_seed(ins[i].name) ^ (step * 2654435761u));
      std::int32_t *p = reinterpret_cast<std::int32_t *>(bufs[i].data());
      const size_t n = ins[i].bytes / sizeof(std::int32_t);
      for (size_t k = 0; k < n; ++k) p[k] = static_cast<std::int32_t>(rng.next() % vocab);
    }
  }
}

std::uint64_t fnv1a(const std::vector<std::uint8_t> &b, std::uint64_t h = 1469598103934665603ull) {
  for (std::uint8_t c : b) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

bool g_loss_trace = false;

struct Result {
  std::vector<std::vector<std::uint8_t>> last_loss;   // one vector
  std::vector<std::vector<std::uint8_t>> final_outputs;
  std::uint64_t loss_hash = 0;
  double ms_per_step = 0.0;
  std::uint64_t h2d_per_step = 0;
  std::uint64_t d2h_per_step = 0;
};

// Which output carries the next value of an input. The tracer's step is the
// traced function's argument list followed by the AdamW nodes: each AdamW
// reads (p, grad, m1, m2, step) and writes the updated (p, m1, m2), so the
// graph text gives the leaf -> node-result correspondence while the compiled
// outputs keep the node-result names. Output 0 is the loss.
std::vector<int> output_to_input(const std::vector<TensorInfo> &ins,
                                 const std::vector<TensorInfo> &outs,
                                 const std::string &text) {
  std::vector<int> map(outs.size(), -1);
  // A CPU executable does not retain value names (sb_compile compiles the CPU
  // interpreter without them), so the document's (inputs ...) and (outputs ...)
  // sections, which list them in graph order, supply the names the compiled
  // tensors lack.
  auto doc_names = [&text](const char *section, const char *entry) {
    std::vector<std::string> names;
    std::istringstream in(text);
    std::string l;
    bool inside = false;
    while (std::getline(in, l)) {
      if (!inside) {
        if (l.find(section) == std::string::npos) continue;
        inside = true;
        continue;
      }
      if (l.find(entry) == std::string::npos) break;
      const size_t b = l.find('"');
      const size_t e = l.find('"', b + 1);
      if (e == std::string::npos) break;
      names.push_back(l.substr(b + 1, e - b - 1));
    }
    return names;
  };
  const std::vector<std::string> doc_inputs = doc_names("(inputs", "(input \"");
  const std::vector<std::string> doc_outputs = doc_names("(outputs", "(output \"");
  if (!doc_inputs.empty() && doc_inputs.size() != ins.size())
    die("document lists " + std::to_string(doc_inputs.size()) +
        " inputs but the executable has " + std::to_string(ins.size()));
  if (!doc_outputs.empty() && doc_outputs.size() != outs.size())
    die("document lists " + std::to_string(doc_outputs.size()) +
        " outputs but the executable has " + std::to_string(outs.size()));
  std::unordered_map<std::string, int> in_by_name;
  for (size_t i = 0; i < ins.size(); ++i) {
    const std::string name = !ins[i].name.empty()
                                 ? ins[i].name
                                 : (i < doc_inputs.size() ? doc_inputs[i]
                                                          : std::string());
    in_by_name[name] = int(i);
  }
  std::unordered_map<std::string, std::string> out_to_leaf;
  std::istringstream lines(text);
  std::string line;
  bool saw_adamw = false;
  while (std::getline(lines, line)) {
    if (line.find("(node adamw") == std::string::npos) continue;
    saw_adamw = true;
    std::vector<std::string> q;
    for (size_t p = 0; (p = line.find('"', p)) != std::string::npos;) {
      const size_t e = line.find('"', p + 1);
      if (e == std::string::npos) break;
      q.push_back(line.substr(p + 1, e - p - 1));
      p = e + 1;
    }
    // p, grad, m1, m2, step, then the three results p', m1', m2'.
    if (q.size() != 8) continue;
    out_to_leaf[q[5]] = q[0];
    out_to_leaf[q[6]] = q[2];
    out_to_leaf[q[7]] = q[3];
  }
  if (!saw_adamw) die("the fixture has no AdamW nodes to map outputs by");
  for (size_t o = 0; o < outs.size(); ++o) {
    const std::string name = !outs[o].name.empty()
                                 ? outs[o].name
                                 : (o < doc_outputs.size() ? doc_outputs[o]
                                                           : std::string());
    auto it = out_to_leaf.find(name);
    if (it == out_to_leaf.end()) continue;
    auto in = in_by_name.find(it->second);
    if (in == in_by_name.end())
      die("AdamW leaf '" + it->second + "' is not a graph input");
    if (ins[in->second].bytes != outs[o].bytes)
      die("output/input size mismatch for leaf " + it->second);
    map[o] = in->second;
  }
  if (map[0] >= 0) die("output 0 is not the loss");
  size_t mapped = 0;
  for (int i : map) mapped += i >= 0 ? 1 : 0;
  std::fprintf(stderr,
               "bench_train: %zu of %zu outputs carry leaf state (%zu inputs)\n",
               mapped, outs.size(), ins.size());
  return map;
}

Result run(sb_executable *exe, bool resident, const std::string &text,
           int gate_steps, int timed_steps, int repeats) {
  auto ins = query(exe, true);
  auto outs = query(exe, false);
  // Residency is a device-graph property; the full shape and the CPU executor
  // never consult it.
  std::vector<bool> res(ins.size(), false);
  if (resident) {
    for (size_t i = 0; i < ins.size(); ++i) {
      int r = 0;
      Err err;
      if (sb_input_is_resident(exe, i, &r, &err.e) != SB_OK)
        die("residency: " + err.text());
      res[i] = r != 0;
    }
  }
  auto map = output_to_input(ins, outs, text);
  if (map.empty() || map[0] >= 0 || outs[0].bytes != 4)
    die("output 0 is not the loss");

  std::vector<std::vector<std::uint8_t>> bufs;
  fill_initial(ins, bufs);

  std::uint64_t h2d = 0, d2h = 0;
  bool uploaded = false;
  std::uint64_t loss_hash = 1469598103934665603ull;
  Result result;

  auto step_once = [&](std::uint32_t step, bool download_all, bool account) {
    set_batch(ins, bufs, step, 256);
    std::vector<const void *> in_ptr(ins.size(), nullptr);
    std::vector<size_t> in_len(ins.size(), 0);
    for (size_t i = 0; i < ins.size(); ++i) {
      in_len[i] = ins[i].bytes;
      if (resident && res[i] && uploaded) continue;  // state is on the device
      in_ptr[i] = bufs[i].data();
      if (account) h2d += ins[i].bytes;
    }
    std::vector<std::vector<std::uint8_t>> out_bufs;
    std::vector<size_t> out_idx;
    std::vector<void *> out_ptr;
    std::vector<size_t> out_cap, out_len;
    if (download_all) {
      for (size_t o = 0; o < outs.size(); ++o) {
        out_bufs.emplace_back(outs[o].bytes, 0);
        out_idx.push_back(o);
      }
    } else {
      out_bufs.emplace_back(outs[0].bytes, 0);
      out_idx.push_back(0);
    }
    for (size_t k = 0; k < out_bufs.size(); ++k) {
      out_ptr.push_back(out_bufs[k].data());
      out_cap.push_back(out_bufs[k].size());
      out_len.push_back(0);
    }
    Err err;
    sb_status st;
    if (resident) {
      st = sb_run_resident(exe, in_ptr.data(), in_len.data(), ins.size(),
                           out_idx.data(), out_idx.size(), out_ptr.data(),
                           out_cap.data(), out_len.data(), &err.e);
    } else {
      std::vector<void *> all_ptr;
      std::vector<size_t> all_cap, all_len;
      std::vector<std::vector<std::uint8_t>> all;
      for (size_t o = 0; o < outs.size(); ++o) {
        all.emplace_back(outs[o].bytes, 0);
        all_ptr.push_back(all[o].data());
        all_cap.push_back(all[o].size());
        all_len.push_back(0);
      }
      st = sb_run_multi(exe, in_ptr.data(), in_len.data(), ins.size(),
                        all_ptr.data(), all_cap.data(), outs.size(),
                        all_len.data(), &err.e);
      if (st == SB_OK) {
        out_bufs = std::move(all);
        out_idx.clear();
        for (size_t o = 0; o < outs.size(); ++o) out_idx.push_back(o);
      }
    }
    if (st != SB_OK) die("step failed: " + err.text());
    uploaded = true;
    if (account && !download_all) {
      for (size_t k = 0; k < out_bufs.size(); ++k) d2h += out_bufs[k].size();
    } else if (account) {
      for (auto &b : out_bufs) d2h += b.size();
    }
    // The full shape is the tracer's: an updated leaf is written back and
    // bound again on the next step.
    if (!resident) {
      for (size_t o = 0; o < out_idx.size(); ++o) {
        const int i = map[out_idx[o]];
        if (i >= 0) bufs[i] = out_bufs[o];
      }
    }
    for (size_t k = 0; k < out_bufs.size(); ++k) {
      if (out_idx[k] == 0) {
        loss_hash = fnv1a(out_bufs[k], loss_hash);
        result.last_loss = {out_bufs[k]};
        if (g_loss_trace && out_bufs[k].size() == sizeof(float)) {
          float v = 0.0f;
          std::memcpy(&v, out_bufs[k].data(), sizeof(v));
          std::fprintf(stderr, "loss-trace step=%u loss=%.9g\n", step, v);
        }
      }
    }
    return out_bufs;
  };

  // Gate: the first gate_steps steps, loss bit-identical between the shapes.
  for (int s = 0; s < gate_steps; ++s) step_once(static_cast<std::uint32_t>(s + 1), false, false);
  {
    // Everything the step produced, for the end-state comparison.
    const auto all = step_once(static_cast<std::uint32_t>(gate_steps + 1), true, false);
    result.final_outputs = all;
  }
  result.loss_hash = loss_hash;

  // Timing: warm up, then `repeats` runs of `timed_steps` steps; keep the
  // median rate.
  for (int s = 0; s < 2; ++s) step_once(static_cast<std::uint32_t>(gate_steps + 2 + s), false, false);
  std::vector<double> rates;
  std::uint32_t step = static_cast<std::uint32_t>(gate_steps + 4);
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < timed_steps; ++s) step_once(step++, false, false);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    rates.push_back(ms / timed_steps);
  }
  std::sort(rates.begin(), rates.end());
  result.ms_per_step = rates[rates.size() / 2];

  // Steady-state traffic for one step.
  h2d = d2h = 0;
  step_once(step++, false, true);
  result.h2d_per_step = h2d;
  result.d2h_per_step = d2h;
  return result;
}

} // namespace

// The last gate step's loss value (a one-element f32 output).
float loss_value(const Result &r) {
  float v = 0.0f;
  if (r.last_loss.size() == 1 && r.last_loss[0].size() == sizeof(v))
    std::memcpy(&v, r.last_loss[0].data(), sizeof(v));
  return v;
}

int main(int argc, char **argv) {
#ifdef SONICBOOM_PRIVATE_DATA_DIR
  const std::string default_fixture =
      std::string(SONICBOOM_PRIVATE_DATA_DIR) + "/train/tiny-train.sx";
  const char *fixture = default_fixture.c_str();
#else
  const char *fixture = nullptr;
#endif
  int gate_steps = 50;
  int timed_steps = 20;
  int repeats = 3;
  bool cpu = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--fixture") && i + 1 < argc) fixture = argv[++i];
    else if (!std::strcmp(argv[i], "--gate-steps") && i + 1 < argc) gate_steps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--timed-steps") && i + 1 < argc) timed_steps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--cpu")) cpu = true;
    else if (!std::strcmp(argv[i], "--loss-trace")) g_loss_trace = true;
    else die(std::string("unknown argument ") + argv[i]);
  }
  if (fixture == nullptr)
    die("no --fixture and SONICBOOM_PRIVATE_DATA_DIR is not set; the bench "
        "input lives in the private suite's tests/data");
  const std::string text = read_file(fixture);

  const char *g = std::getenv("SONICBOOM_TRAIN_GRAPH");
  const bool graph_on = !(g && g[0] == '0' && g[1] == '\0');

  auto compile = [&]() {
    sb_document *doc = nullptr;
    sb_executable *exe = nullptr;
    Err err;
    if (sb_parse(text.data(), text.size(), &doc, &err.e) != SB_OK)
      die("parse: " + err.text());
    Err err2;
    const sb_status st = cpu ? sb_compile(doc, "", &exe, &err2.e)
                             : sb_compile_device(doc, &exe, &err2.e);
    sb_document_free(doc);
    if (st != SB_OK) die("compile: " + err2.text());
    return exe;
  };

  // CPU executor: the tracer's full call shape only (there is no resident
  // state and no CUDA graph on the host path). The loss value printed here is
  // the cross-backend gate against the device executor's.
  if (cpu) {
    sb_executable *exe = compile();
    const Result r = run(exe, false, text, gate_steps, timed_steps, repeats);
    std::fprintf(stdout, "backend=cpu mode=full graph=off steps=%d ms/step=%.4f "
                         "h2d/step=%llu d2h/step=%llu loss_hash=%016llx "
                         "loss=%.9g\n",
                 gate_steps, r.ms_per_step,
                 static_cast<unsigned long long>(r.h2d_per_step),
                 static_cast<unsigned long long>(r.d2h_per_step),
                 static_cast<unsigned long long>(r.loss_hash), loss_value(r));
    sb_executable_free(exe);
    return 0;
  }

  sb_executable *full = compile();
  sb_executable *resident = compile();
  const Result full_r = run(full, false, text, gate_steps, timed_steps, repeats);
  const Result res_r = run(resident, true, text, gate_steps, timed_steps, repeats);

  bool loss_same = full_r.last_loss == res_r.last_loss;
  std::uint64_t lh_full = 0, lh_res = 0;
  for (const auto &b : full_r.last_loss) lh_full = fnv1a(b, lh_full);
  for (const auto &b : res_r.last_loss) lh_res = fnv1a(b, lh_res);
  (void)lh_full; (void)lh_res;
  std::fprintf(stdout, "mode=full      graph=%s steps=%d ms/step=%.4f "
                       "h2d/step=%llu d2h/step=%llu loss_hash=%016llx "
                       "loss=%.9g\n",
               graph_on ? "on" : "off", gate_steps, full_r.ms_per_step,
               static_cast<unsigned long long>(full_r.h2d_per_step),
               static_cast<unsigned long long>(full_r.d2h_per_step),
               static_cast<unsigned long long>(full_r.loss_hash),
               loss_value(full_r));
  std::fprintf(stdout, "mode=resident  graph=%s steps=%d ms/step=%.4f "
                       "h2d/step=%llu d2h/step=%llu loss_hash=%016llx "
                       "loss=%.9g\n",
               graph_on ? "on" : "off", gate_steps, res_r.ms_per_step,
               static_cast<unsigned long long>(res_r.h2d_per_step),
               static_cast<unsigned long long>(res_r.d2h_per_step),
               static_cast<unsigned long long>(res_r.loss_hash),
               loss_value(res_r));

  // Cross-shape gate: the loss bytes of the last gate step, and every output
  // of the step after it.
  bool final_same = full_r.final_outputs.size() == res_r.final_outputs.size();
  if (final_same)
    for (size_t i = 0; i < full_r.final_outputs.size(); ++i)
      if (full_r.final_outputs[i] != res_r.final_outputs[i]) { final_same = false; break; }
  std::fprintf(stdout, "gate: loss_%s all_outputs_%s\n",
               loss_same ? "identical" : "DIFFER",
               final_same ? "identical" : "DIFFER");
  if (!loss_same || !final_same) return 1;

  sb_executable_free(full);
  sb_executable_free(resident);
  return 0;
}
