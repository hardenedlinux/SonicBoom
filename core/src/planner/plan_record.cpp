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

#include <sonicboom/planner/plan_record.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace sonicboom::planner {

namespace plan_detail {

thread_local PlanRecorder* tls_recorder_storage = nullptr;

} // namespace plan_detail

namespace {

// Render one attribute value deterministically. Integers and bools are exact;
// doubles use the shortest round-trippable form ("%.17g") so the text never
// depends on the default stream precision.
std::string attr_value_text(const sx::Attribute& a) {
  char buf[64];
  switch (a.kind()) {
    case sx::AttrValueKind::Int:
      return std::to_string(std::get<int64_t>(a.value));
    case sx::AttrValueKind::Bool:
      return std::get<bool>(a.value) ? "true" : "false";
    case sx::AttrValueKind::Float: {
      std::snprintf(buf, sizeof buf, "%.17g", std::get<double>(a.value));
      return buf;
    }
    case sx::AttrValueKind::String:
      return std::get<std::string>(a.value);
    case sx::AttrValueKind::Ints: {
      std::string s;
      for (int64_t v : std::get<std::vector<int64_t>>(a.value)) {
        if (!s.empty()) s += ',';
        s += std::to_string(v);
      }
      return s;
    }
    case sx::AttrValueKind::Bools: {
      std::string s;
      for (bool v : std::get<std::vector<bool>>(a.value)) {
        if (!s.empty()) s += ',';
        s += v ? "true" : "false";
      }
      return s;
    }
    case sx::AttrValueKind::Floats: {
      std::string s;
      for (double v : std::get<std::vector<double>>(a.value)) {
        if (!s.empty()) s += ',';
        std::snprintf(buf, sizeof buf, "%.17g", v);
        s += buf;
      }
      return s;
    }
    case sx::AttrValueKind::Strings: {
      std::string s;
      for (const std::string& v : std::get<std::vector<std::string>>(a.value)) {
        if (!s.empty()) s += ',';
        s += v;
      }
      return s;
    }
  }
  return {};
}

} // namespace

std::string shape_text(std::span<const int64_t> shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ',';
    s += std::to_string(shape[i]);
  }
  s += ']';
  return s;
}

// recorder() is defined inline in plan_record.h (one thread-local load).

void set_thread_recorder(PlanRecorder* rec) noexcept {
  plan_detail::tls_recorder_storage = rec;
}

void PlanRecorder::clear() {
  steps_.clear();
  streams_.clear();
  run_ = PlanRunRecord{};
}

uint32_t PlanRecorder::stream_ordinal(const void* stream) {
  for (const auto& [s, id] : streams_)
    if (s == stream) return id;
  const uint32_t id = static_cast<uint32_t>(streams_.size());
  streams_.emplace_back(stream, id);
  return id;
}

void PlanRecorder::begin_step(uint32_t index, uint64_t task_id, uint32_t node_id,
                              std::string_view op, std::string_view backend,
                              std::string_view name,
                              std::span<const sx::Attribute> attrs,
                              const void* stream) {
  StepRecord s;
  s.index = index;
  s.task_id = task_id;
  s.node_id = node_id;
  s.op = op;
  s.backend = backend;
  s.name = name;
  // A caller that knows the stream (a device-side recorder) sets it here; the
  // host-side executor passes null and end_step() derives it from the launches.
  if (stream != nullptr) s.stream = stream_ordinal(stream);
  s.attrs.reserve(attrs.size());
  for (const auto& a : attrs) s.attrs.push_back(AttrRecord{a.name, attr_value_text(a)});
  // Canonical attribute order: by name, so the record does not depend on the
  // order the emitter (or a rewrite pass) happened to append them in.
  std::sort(s.attrs.begin(), s.attrs.end(),
            [](const AttrRecord& a, const AttrRecord& b) { return a.name < b.name; });
  steps_.push_back(std::move(s));
}

void PlanRecorder::end_step() {
  if (steps_.empty()) return;
  StepRecord& s = steps_.back();
  // Derive the step's stream from its launches when the caller could not supply
  // one. Every launch in a step runs on one stream, so the first is the step's.
  if (s.stream == kStreamUnknown && !s.launches.empty())
    s.stream = s.launches.front().stream;
}

void PlanRecorder::add_buffer(BufferRecord b) {
  if (steps_.empty()) return;
  steps_.back().buffers.push_back(std::move(b));
}

void PlanRecorder::set_weight_quant(std::string_view q) {
  if (!steps_.empty()) steps_.back().weight_quant = q;
}

void PlanRecorder::set_activation_quant(std::string_view q) {
  if (!steps_.empty()) steps_.back().activation_quant = q;
}

void PlanRecorder::add_launch(std::string_view kernel, std::string_view tparams,
                              unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                              unsigned by, unsigned bz, uint64_t smem_bytes,
                              const void* stream) {
  if (steps_.empty()) return;
  LaunchRecord l;
  l.kernel = kernel;
  l.tparams = tparams;
  l.grid[0] = gx; l.grid[1] = gy; l.grid[2] = gz;
  l.block[0] = bx; l.block[1] = by; l.block[2] = bz;
  l.smem_bytes = smem_bytes;
  l.stream = stream_ordinal(stream);
  steps_.back().launches.push_back(std::move(l));
}

void PlanRecorder::set_run(std::string_view model, std::string_view mode,
                           uint64_t n_ctx, uint64_t n, uint64_t plan_id,
                           uint64_t model_fingerprint, std::string_view backend) {
  run_.model = model;
  run_.mode = mode;
  run_.n_ctx = n_ctx;
  run_.n = n;
  run_.plan_id = plan_id;
  run_.model_fingerprint = model_fingerprint;
  run_.backend = backend;
}

void PlanRecorder::set_device(std::string_view device) {
  run_.device = device;
}

PlanRunRecord PlanRecorder::finish() const {
  PlanRunRecord out = run_;
  out.steps = steps_;
  return out;
}

std::string serialize_plan_run(const PlanRunRecord& run) {
  std::string s;
  s.reserve(1024 + run.steps.size() * 512);
  auto line = [&](const std::string& t) { s += t; s += '\n'; };

  line("# sonicboom plan record");
  line("schema " + std::to_string(run.schema_version));
  line("model " + run.model);
  line("mode " + run.mode);
  line("n_ctx " + std::to_string(run.n_ctx));
  line("n " + std::to_string(run.n));
  line("plan_id " + std::to_string(run.plan_id));
  line("model_fingerprint " + std::to_string(run.model_fingerprint));
  line("backend " + run.backend);
  if (!run.device.empty()) line("device " + run.device);
  line("steps " + std::to_string(run.steps.size()));

  for (const StepRecord& st : run.steps) {
    std::string head = "step " + std::to_string(st.index) + " task " +
                       std::to_string(st.task_id) + " node " +
                       std::to_string(st.node_id) + " op " + st.op + " backend " +
                       st.backend;
    if (st.stream != kStreamUnknown) head += " stream s" + std::to_string(st.stream);
    if (!st.name.empty()) head += " name " + st.name;
    line(head);
    for (const AttrRecord& a : st.attrs)
      line("  attr " + a.name + " " + a.value);
    for (const BufferRecord& b : st.buffers) {
      std::string bl = "  buf " + b.role + " " + std::to_string(b.index) +
                       " t" + std::to_string(b.tensor) + " buffer b" +
                       std::to_string(b.buffer) + " offset " +
                       std::to_string(b.offset) + " size " +
                       std::to_string(b.size_bytes) + " shape " + b.shape +
                       " dtype " + b.dtype;
      line(bl);
    }
    if (!st.weight_quant.empty()) line("  weight_quant " + st.weight_quant);
    if (!st.activation_quant.empty())
      line("  activation_quant " + st.activation_quant);
    for (const LaunchRecord& l : st.launches) {
      std::string ll = "  launch " + l.kernel;
      if (!l.tparams.empty()) ll += "<" + l.tparams + ">";
      ll += " grid " + std::to_string(l.grid[0]) + "x" +
            std::to_string(l.grid[1]) + "x" + std::to_string(l.grid[2]);
      ll += " block " + std::to_string(l.block[0]) + "x" +
            std::to_string(l.block[1]) + "x" + std::to_string(l.block[2]);
      ll += " smem " + std::to_string(l.smem_bytes);
      if (l.stream != kStreamUnknown) ll += " stream s" + std::to_string(l.stream);
      line(ll);
    }
  }
  return s;
}

} // namespace sonicboom::planner
