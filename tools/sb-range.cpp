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

// sb-range — the facts the Boom range/error analysis needs (boom-r4 F2).
//
//   sb-range calibrate MODEL OUT_FACTS
//       Scan the loaded weights (per (slot, layer): the dequantized minimum and
//       maximum), then run the CPU spine over fixed prompts and fold every
//       executed activation's measured extrema in. Writes OUT_FACTS in the
//       RangeFacts text format plus OUT_FACTS.prompts.txt naming the exact
//       prompts and token counts the calibration saw.
//
//   sb-range analyze MODEL FACTS OUT_REPORT
//       Build the plan with SONICBOOM_BOOM_ANALYSIS_FACTS=FACTS and
//       SONICBOOM_BOOM_ANALYSIS_UNITS=all, which makes the lowering run the
//       analyses on every graph it lowers and on the lifted boom.program, then
//       write the report to OUT_REPORT. The plan itself is unchanged: the
//       analyses write no attribute.
//
// The weight error term is zero: the interval describes the stored weights, not
// an approximation of them. A calibrated activation likewise carries err = 0 —
// it is the value the arithmetic produced.

#include <sonicboom/compiler/BoomAnalysis.h>
#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/quant/dequant.h>
#include <sonicboom/vocab/vocab.h>

#include "../core/src/planner/calibration.h"

#include "analysis_driver.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbvocab = sonicboom::vocab;
namespace quant = sonicboom::quant;
namespace compiler = sonicboom::compiler;

namespace {

using Slot = sbmodel::WeightSlot;

struct Loaded {
  std::optional<sbgguf::Reader> reader;
  sbmodel::Gemma4Model model;
  sbvocab::Vocab vocab;
};

bool load(const std::string &path, Loaded &out) {
  auto r = sbgguf::Reader::load(path);
  if (!r) {
    std::cerr << "cannot read " << path << "\n";
    return false;
  }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) {
    std::cerr << "cannot load model weights from " << path << "\n";
    return false;
  }
  auto v = sbvocab::load_vocab(*r);
  if (!v) {
    std::cerr << "cannot load vocabulary from " << path << "\n";
    return false;
  }
  out.reader.emplace(std::move(*r));
  out.model = std::move(*m);
  out.vocab = std::move(*v);
  return true;
}

// --- weight scan -------------------------------------------------------------

void scanSpan(compiler::RangeFacts &facts, Slot slot, int64_t layer,
              std::span<const float> w) {
  double lo = std::numeric_limits<double>::infinity();
  double hi = -lo;
  for (const float v : w) {
    if (!std::isfinite(v)) continue;
    lo = std::min(lo, double(v));
    hi = std::max(hi, double(v));
  }
  if (lo <= hi)
    facts.setWeight(int64_t(slot), layer, compiler::RangeError::bounded(lo, hi, 0));
}

void scanQuant(compiler::RangeFacts &facts, Slot slot, int64_t layer,
               const quant::QuantizedTensor &t) {
  if (!t.valid() || t.numel() == 0) return;
  const auto &dims = t.dims();
  const uint64_t row = dims.empty() ? t.numel() : dims.front();
  if (row == 0) return;
  const uint64_t rows = t.numel() / row;
  std::vector<float> buf(row);
  double lo = std::numeric_limits<double>::infinity();
  double hi = -lo;
  for (uint64_t r = 0; r < rows; ++r) {
    if (!quant::dequantize_row_f32(t, r, buf.data(), buf.size())) return;
    for (const float v : buf) {
      if (!std::isfinite(v)) continue;
      lo = std::min(lo, double(v));
      hi = std::max(hi, double(v));
    }
  }
  if (lo <= hi)
    facts.setWeight(int64_t(slot), layer, compiler::RangeError::bounded(lo, hi, 0));
}

// Every weight the plan can name by (slot, layer). A tied lm head is the token
// embedding (the emitter records no Output weight), so Output stays absent.
// A model-wide weight is recorded at layer -1, the layer a node that names no
// `layer` attribute looks up (the analysis reads the attribute, not the plan).
void scanWeights(const sbmodel::Gemma4Weights &w, compiler::RangeFacts &facts) {
  constexpr int64_t kGlobal = -1;
  scanQuant(facts, Slot::TokenEmbd, kGlobal, w.token_embd);
  scanQuant(facts, Slot::PerLayerTokenEmbd, kGlobal, w.per_layer_token_embd);
  scanSpan(facts, Slot::PerLayerProjNorm, kGlobal, w.per_layer_proj_norm);
  scanSpan(facts, Slot::OutputNorm, kGlobal, w.output_norm);
  scanQuant(facts, Slot::Output, kGlobal, w.output);
  // per_layer_model_proj is raw bf16 halves; widen them one at a time.
  if (!w.per_layer_model_proj.empty()) {
    std::vector<float> tmp(w.per_layer_model_proj.size());
    for (size_t i = 0; i < tmp.size(); ++i)
      tmp[i] = sonicboom::bf16_to_f32(w.per_layer_model_proj[i]);
    scanSpan(facts, Slot::PerLayerModelProj, kGlobal, tmp);
  }

  for (size_t l = 0; l < w.blocks.size(); ++l) {
    const auto &b = w.blocks[l];
    const int64_t layer = int64_t(l);
    scanSpan(facts, Slot::AttnNorm, layer, b.attn_norm);
    scanSpan(facts, Slot::FfnNorm, layer, b.ffn_norm);
    scanSpan(facts, Slot::AttnQNorm, layer, b.attn_q_norm);
    scanSpan(facts, Slot::AttnKNorm, layer, b.attn_k_norm);
    scanSpan(facts, Slot::PostAttentionNorm, layer, b.post_attention_norm);
    scanSpan(facts, Slot::PostFfwNorm, layer, b.post_ffw_norm);
    scanSpan(facts, Slot::PostNorm, layer, b.post_norm);
    scanQuant(facts, Slot::AttnQ, layer, b.attn_q);
    scanQuant(facts, Slot::AttnK, layer, b.attn_k);
    scanQuant(facts, Slot::AttnV, layer, b.attn_v);
    scanQuant(facts, Slot::AttnOutput, layer, b.attn_output);
    scanQuant(facts, Slot::FfnGate, layer, b.ffn_gate);
    scanQuant(facts, Slot::FfnUp, layer, b.ffn_up);
    scanQuant(facts, Slot::FfnDown, layer, b.ffn_down);
    scanSpan(facts, Slot::AttnQBias, layer, b.attn_q_bias);
    scanSpan(facts, Slot::AttnKBias, layer, b.attn_k_bias);
    scanSpan(facts, Slot::AttnVBias, layer, b.attn_v_bias);
    scanSpan(facts, Slot::AttnOutputBias, layer, b.attn_output_bias);
    scanSpan(facts, Slot::InpGate, layer, b.inp_gate);
    scanSpan(facts, Slot::Proj, layer, b.proj);
    scanSpan(facts, Slot::LayerOutputScale, layer, b.layer_output_scale);
  }
}

// --- calibration -------------------------------------------------------------

// The recorded calibration corpus: one paragraph repeated to roughly 200
// tokens, then a couple of short prompts, for a mix of prefill lengths.
const char *const kCalibText =
    "A tensor runtime executes a graph whose nodes name typed values, and every "
    "value flows from an input to an output through a kernel. The planner "
    "chooses a kernel for each shape class, and the autotuner measures the "
    "choice once per device. A range analysis propagates an interval and an "
    "error bound from the weights and the inputs so a reduction's accumulation "
    "order can be judged against a stated tolerance. ";
const char *const kShortPrompts[] = {
    "The capital of France is",
    "List the first five prime numbers.",
};

std::vector<uint64_t> encode(const Loaded &l, const std::string &text) {
  const auto ids = sbvocab::tokenize(l.vocab, text);
  std::vector<uint64_t> out;
  out.reserve(ids.size());
  for (int32_t id : ids) out.push_back(uint64_t(id));
  return out;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage:\n"
                 "  sb-range calibrate MODEL OUT_FACTS\n"
                 "  sb-range analyze MODEL FACTS OUT_REPORT\n";
    return 2;
  }
  const std::string mode = argv[1];
  const std::string model_path = argv[2];

  if (mode == "calibrate") {
    const std::string out_path = argv[3];
    Loaded l;
    if (!load(model_path, l)) return 1;

    compiler::RangeFacts facts;
    scanWeights(l.model.weights, facts);
    std::cout << "weights: " << facts.weightCount() << " facts\n";

    const std::string calib_path = out_path + ".calib.tmp";
    setenv("SONICBOOM_BOOM_CALIBRATION", calib_path.c_str(), 1);

    std::string prompts;
    auto run = [&](const std::string &text, const char *label) {
      const auto prompt = encode(l, text);
      if (prompt.empty()) return;
      std::vector<int64_t> tokens(16, -1);
      const uint64_t n = sbmodel::generate_prefill_spine(
          l.model, prompt, 0, tokens, 0, /*use_prefill=*/true);
      prompts += std::string(label) + " tokens=" + std::to_string(prompt.size()) +
                 " generated=" + std::to_string(n) + "\n";
      std::cout << label << ": prompt " << prompt.size() << " tokens, " << n
                << " generated\n";
    };

    std::string long_text;
    for (int i = 0; i < 3; ++i) long_text += kCalibText;
    run(long_text, "corpus");
    run(kShortPrompts[0], "short0");
    run(kShortPrompts[1], "short1");

    if (!sonicboom::planner::calibration::flush(calib_path.c_str())) {
      std::cerr << "cannot write calibration file " << calib_path << "\n";
      return 1;
    }
    compiler::RangeFacts calib;
    std::string error;
    if (!calib.load(calib_path, &error)) {
      std::cerr << error << "\n";
      return 1;
    }
    facts.mergeFrom(calib);
    std::filesystem::remove(calib_path);

    if (!facts.save(out_path, &error)) {
      std::cerr << error << "\n";
      return 1;
    }
    std::ofstream p(out_path + ".prompts.txt");
    p << prompts;
    std::cout << "facts: " << facts.weightCount() << " weights, "
              << facts.tensorCount() << " tensors -> " << out_path << "\n";
    return 0;
  }

  if (mode == "analyze") {
    if (argc < 5) {
      std::cerr << "usage: sb-range analyze MODEL FACTS OUT_REPORT\n";
      return 2;
    }
    const std::string facts_path = argv[3];
    const std::string report_path = argv[4];
    // The driver marks a clone with BufferReusePass (the CPU path lowers
    // without the phase's passes) and lifts a decode clone into a boom.program,
    // so both the marked graphs and the step are reported.
    std::string error;
    if (!sb_tools::run_range_analysis(model_path, facts_path, report_path,
                                      &error)) {
      std::cerr << error << "\n";
      return 1;
    }
    std::cout << "analysis written to " << report_path << "\n";
    return 0;
  }

  std::cerr << "unknown command " << mode << "\n";
  return 2;
}
