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

// Phase 5A: Gemma 4 model loader + explicit execution plan + per-layer
// embedding / single-block reference execution.
//
// Two tiers, mirroring the frozen facts in design/gemma4-model-map.md:
//   1. Pure bf16 conversion checks (always run, no model needed).
//   2. Real-model checks (skipped cleanly when the target GGUF is absent):
//      - config + plan structural validation against the frozen facts
//      - weight binding (42 blocks, every tensor bound, shapes/dtypes checked)
//      - embed_token == dequantize_row_f32 * sqrt(n_embd) (wiring cross-check)
//      - embed_per_layer / run_block: shape + finiteness + determinism
//        (the numerical oracle comparison is Phase 5C-2)

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/exec.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/weights.h>
#include <sonicboom/quant/dequant.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbquant = sonicboom::quant;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool all_finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

// --- tier 1: bf16 -> f32 (pure, no model) ---------------------------------

void test_bf16_conversion() {
  check(sbmodel::bf16_to_f32(0x3F80) == 1.0f, "bf16: 0x3F80 -> 1.0");
  check(sbmodel::bf16_to_f32(0x4000) == 2.0f, "bf16: 0x4000 -> 2.0");
  check(sbmodel::bf16_to_f32(0x3F00) == 0.5f, "bf16: 0x3F00 -> 0.5");
  check(sbmodel::bf16_to_f32(0xC000) == -2.0f, "bf16: 0xC000 -> -2.0");
  check(sbmodel::bf16_to_f32(0x0000) == 0.0f, "bf16: 0x0000 -> 0.0");
  check(std::signbit(sbmodel::bf16_to_f32(0x8000)), "bf16: 0x8000 -> -0.0");
}

// --- tier 2: real model (skipped when absent) ------------------------------

std::string find_model() {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF")) candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/models/gemma-4-E4B-it-Q3_K_M.gguf");
#endif
  candidates.push_back("models/gemma-4-E4B-it-Q3_K_M.gguf");
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

void test_model(const sbmodel::Gemma4Model& m) {
  const sbmodel::Gemma4Config& c = m.config;

  // --- frozen config facts (design/gemma4-model-map.md) ---------------------
  check(c.block_count == 42, "config: block_count 42");
  check(c.embedding_length == 2560, "config: embedding_length 2560");
  check(c.feed_forward_length == 10240, "config: feed_forward_length 10240");
  check(c.head_count == 8, "config: head_count 8");
  check(c.head_count_kv == 2, "config: head_count_kv 2");
  check(c.head_dim_global == 512, "config: head_dim_global 512");
  check(c.head_dim_swa == 256, "config: head_dim_swa 256");
  check(c.per_layer_input == 256, "config: per_layer_input 256");
  check(c.context_length == 131072, "config: context_length 131072");
  check(c.sliding_window == 512, "config: sliding_window 512");
  check(c.vocab_size == 262144, "config: vocab_size 262144");
  check(c.rms_epsilon == 1e-6f, "config: rms_epsilon 1e-6");
  check(c.final_logit_softcapping == 30.0f, "config: final_logit_softcapping 30");
  check(c.rope_base == 1e6f, "config: rope_base 1e6");
  check(c.rope_base_swa == 10000.0f, "config: rope_base_swa 10000");
  check(c.shared_kv_layers == 18, "config: shared_kv_layers 18");
  check(c.valid(), "config: valid()");

  // --- shared-KV structure (Phase 5D) --------------------------------------
  check(c.n_layer_kv() == 24, "config: 24 KV layers (42 - 18)");
  check(c.has_kv(0) && c.has_kv(22) && c.has_kv(23), "config: layers 0/22/23 have KV");
  check(!c.has_kv(24) && !c.has_kv(29) && !c.has_kv(41), "config: layers 24/29/41 shared");
  check(c.kv_donor_layer(0) == 0 && c.kv_donor_layer(23) == 23,
        "config: KV layer donor is identity");
  check(c.kv_donor_layer(24) == 22, "config: SWA shared layer 24 -> donor 22");
  check(c.kv_donor_layer(29) == 23, "config: global shared layer 29 -> donor 23");
  check(c.kv_donor_layer(41) == 23, "config: global shared layer 41 -> donor 23");

  // --- sliding window pattern (42 bits) ------------------------------------
  const std::string want_pattern = "111110111110111110111110111110111110111110";
  check(c.sliding_window_pattern.size() == 42, "config: pattern has 42 entries");
  if (c.sliding_window_pattern.size() == 42) {
    std::string got;
    for (bool b : c.sliding_window_pattern) got += b ? '1' : '0';
    check(got == want_pattern, "config: sliding_window_pattern matches frozen bits");
  }

  // --- plan (42 layers, correct per-layer kind/head_dim/rope) ---------------
  check(m.plan.size() == 42, "plan: 42 layers");
  if (m.plan.size() == 42) {
    const sbmodel::LayerConfig& l0 = m.plan[0];
    check(l0.attention == sbmodel::AttentionKind::SlidingWindow,
          "plan[0]: sliding-window");
    check(l0.head_dim == 256 && l0.rope_base == 10000.0f && l0.sliding_window == 512,
          "plan[0]: head_dim 256 / rope 10000 / window 512");

    const sbmodel::LayerConfig& l5 = m.plan[5];
    check(l5.attention == sbmodel::AttentionKind::Global, "plan[5]: global");
    check(l5.head_dim == 512 && l5.rope_base == 1e6f && l5.sliding_window == 0,
          "plan[5]: head_dim 512 / rope 1e6 / no window");

    const sbmodel::LayerConfig& l41 = m.plan[41];
    check(l41.attention == sbmodel::AttentionKind::Global, "plan[41]: global");
  }

  // --- weight binding -------------------------------------------------------
  check(m.weights.blocks.size() == 42, "weights: 42 blocks bound");
  check(!m.weights.empty(), "weights: non-empty");
  const auto& b0 = m.weights.blocks[0];
  check(b0.attn_q.valid() && b0.attn_q.dims() == std::vector<uint64_t>({2560, 2048}),
        "weights: blk.0 attn_q [2560,2048]");
  check(b0.attn_v.valid() && b0.attn_v.dims() == std::vector<uint64_t>({2560, 512}),
        "weights: blk.0 attn_v [2560,512]");
  check(b0.ffn_gate.valid() && b0.ffn_gate.dims() == std::vector<uint64_t>({2560, 10240}),
        "weights: blk.0 ffn_gate [2560,10240]");
  const auto& b5 = m.weights.blocks[5];
  check(b5.attn_q.dims() == std::vector<uint64_t>({2560, 4096}),
        "weights: blk.5 attn_q [2560,4096] (global)");
  // Mixed precision: attn_v / ffn_down are q5_K in blk.{0,1}, q4_K elsewhere.
  check(b0.attn_v.type() == sbquant::QuantType::Q5_K, "weights: blk.0 attn_v q5_K");
  check(b0.ffn_down.type() == sbquant::QuantType::Q5_K, "weights: blk.0 ffn_down q5_K");
  // Explicit blk.1 (Q5_K) -> blk.2 (Q4_K) mixed-precision boundary: the
  // last q5_K block is blk.1, the first q4_K block is blk.2.
  check(m.weights.blocks[1].attn_v.type() == sbquant::QuantType::Q5_K,
        "weights: blk.1 attn_v q5_K (last q5_K block)");
  check(m.weights.blocks[1].ffn_down.type() == sbquant::QuantType::Q5_K,
        "weights: blk.1 ffn_down q5_K (last q5_K block)");
  check(m.weights.blocks[2].attn_v.type() == sbquant::QuantType::Q4_K,
        "weights: blk.2 attn_v q4_K (mixed boundary)");
  check(m.weights.blocks[2].ffn_down.type() == sbquant::QuantType::Q4_K,
        "weights: blk.2 ffn_down q4_K (mixed boundary)");
  check(b5.attn_v.type() == sbquant::QuantType::Q4_K, "weights: blk.5 attn_v q4_K");
  check(b0.attn_norm.size() == 2560 && b0.ffn_norm.size() == 2560,
        "weights: blk.0 norm weights [2560]");
  check(b0.layer_output_scale.size() == 1, "weights: blk.0 layer_output_scale [1]");

  const uint32_t d = c.embedding_length;

  // --- Phase 5C-1: all 17 per-block tensor groups bound ---------------------
  // The 7 groups added in Phase 5C-1 are bound for every block, with the
  // per-layer head_dim (256 SWA / 512 global) and per-layer gate dim (256).
  {
    const uint32_t per = c.per_layer_input;
    int missing = 0;
    for (uint32_t l = 0; l < c.block_count; ++l) {
      const sbmodel::BlockWeights& bw = m.weights.blocks[l];
      const sbmodel::LayerConfig& lc = m.plan[l];
      const bool ok =
          !bw.attn_q_norm.empty() && bw.attn_q_norm.size() == lc.head_dim &&
          !bw.attn_k_norm.empty() && bw.attn_k_norm.size() == lc.head_dim &&
          !bw.post_attention_norm.empty() && bw.post_attention_norm.size() == d &&
          !bw.post_ffw_norm.empty() && bw.post_ffw_norm.size() == d &&
          !bw.post_norm.empty() && bw.post_norm.size() == d &&
          !bw.inp_gate.empty() && bw.inp_gate.size() == uint64_t(d) * per &&
          !bw.proj.empty() && bw.proj.size() == uint64_t(d) * per;
      if (!ok) ++missing;
    }
    check(missing == 0, "weights: 7 added groups bound for all 42 blocks");
  }

  // Representative shape checks for both head dims + the per-layer gate dims.
  check(b0.attn_q_norm.size() == 256 && b0.attn_k_norm.size() == 256,
        "weights: blk.0 q/k norm [256] (SWA)");
  check(b5.attn_q_norm.size() == 512 && b5.attn_k_norm.size() == 512,
        "weights: blk.5 q/k norm [512] (global)");
  check(b0.inp_gate.size() == 2560 * 256 && b0.proj.size() == 256 * 2560,
        "weights: blk.0 inp_gate [2560,256] / proj [256,2560]");
  check(b0.post_attention_norm.size() == 2560 && b0.post_ffw_norm.size() == 2560 &&
            b0.post_norm.size() == 2560,
        "weights: blk.0 post-* norms [2560]");

  // Empirical fact (read from raw weights, independent of run_block): the q/k
  // norm tensors are scalar broadcasts — every entry in the span is identical.
  {
    bool q_const = !b0.attn_q_norm.empty();
    bool k_const = !b0.attn_k_norm.empty();
    for (size_t i = 1; i < b0.attn_q_norm.size(); ++i)
      if (b0.attn_q_norm[i] != b0.attn_q_norm[0]) q_const = false;
    for (size_t i = 1; i < b0.attn_k_norm.size(); ++i)
      if (b0.attn_k_norm[i] != b0.attn_k_norm[0]) k_const = false;
    check(q_const, "weights: blk.0 attn_q_norm is a scalar broadcast");
    check(k_const, "weights: blk.0 attn_k_norm is a scalar broadcast");
  }

  // --- embed_token == dequantize_row_f32 * sqrt(n_embd) ---------------------
  {
    std::vector<float> emb(d), direct(d);
    check(sbmodel::embed_token(m, 0, emb), "embed_token: succeeds");
    check(sbquant::dequantize_row_f32(m.weights.token_embd, 0, direct.data(), d),
          "embed_token: direct row dequant succeeds");
    const float s = std::sqrt(float(d));
    bool scaled = true;
    for (uint32_t i = 0; i < d; ++i)
      if (emb[i] != direct[i] * s) scaled = false;
    check(scaled, "embed_token: == dequantize_row_f32 * sqrt(n_embd)");
    check(all_finite(emb), "embed_token: finite");
  }

  // --- per-layer embedding (length per_layer_input) -------------------------
  {
    const uint32_t per = c.per_layer_input;
    std::vector<float> per0(per), per5(per);
    check(sbmodel::embed_per_layer(m, 0, 0, per0, false), "embed_per_layer: layer 0 ok");
    check(sbmodel::embed_per_layer(m, 5, 0, per5, false), "embed_per_layer: layer 5 ok");
    check(all_finite(per0), "embed_per_layer: layer 0 finite");
    check(all_finite(per5), "embed_per_layer: layer 5 finite");
    check(per0 != per5, "embed_per_layer: layer 0 != layer 5 (distinct projections)");

    std::vector<float> again(per);
    sbmodel::embed_per_layer(m, 0, 0, again, false);
    check(again == per0, "embed_per_layer: deterministic");

    std::vector<float> short_out(per - 1);
    check(!sbmodel::embed_per_layer(m, 0, 0, short_out, false),
          "embed_per_layer: rejects short output");
  }

  // --- single-block execution (KV layers only; shared layers need forward()) -
  {
    for (uint32_t layer : {0u, 5u, 22u, 23u}) {
      std::vector<float> out(d), again(d);
      check(sbmodel::run_block(m, layer, 0, 0, out), "run_block: succeeds");
      check(all_finite(out), "run_block: finite");
      bool nonzero = false;
      for (float x : out)
        if (x != 0.0f) nonzero = true;
      check(nonzero, "run_block: non-degenerate output");
      check(sbmodel::run_block(m, layer, 0, 0, again) && again == out,
            "run_block: deterministic");
    }

    // Validation: out-of-range layer rejected; shared layer needs forward().
    std::vector<float> bad(d);
    check(!sbmodel::run_block(m, 42, 0, 0, bad), "run_block: rejects layer >= 42");
    check(!sbmodel::run_block(m, 24, 0, 0, bad), "run_block: shared layer needs forward()");
  }

  // --- full forward (Phase 5D: chaining + shared-KV) ------------------------
  {
    std::vector<float> out(d), again(d);
    check(sbmodel::forward(m, 0, 0, out), "forward: succeeds");
    check(all_finite(out), "forward: finite");
    bool nonzero = false;
    for (float x : out)
      if (x != 0.0f) nonzero = true;
    check(nonzero, "forward: non-degenerate output");
    check(sbmodel::forward(m, 0, 0, again) && again == out, "forward: deterministic");

    // The chained output must differ from any single-block (isolated) output:
    // the residual stream flows through 42 blocks, and shared layers (24..41)
    // read the donor V instead of computing their own.
    std::vector<float> blk0(d);
    sbmodel::run_block(m, 0, 0, 0, blk0);
    check(out != blk0, "forward: chained output != isolated layer-0 output");
  }
}

} // namespace

int main() {
  test_bf16_conversion();

  const std::string path = find_model();
  if (path.empty()) {
    std::cout << "  (skipped: Gemma 4 model not found)\n";
    std::cout << "test_gemma4 OK\n";
    return 0;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    check(false, "model: failed to load GGUF");
    std::cerr << "test_gemma4 FAILED: 1 check(s)\n";
    return 1;
  }

  auto m = sbmodel::load_gemma4(*r);
  if (!m) {
    std::cerr << "  load_gemma4 error: " << m.error() << "\n";
    check(false, "model: load_gemma4 succeeded");
    std::cerr << "test_gemma4 FAILED: 1 check(s)\n";
    return 1;
  }

  test_model(*m);

  if (g_failures == 0) {
    std::cout << "test_gemma4 OK\n";
    return 0;
  }
  std::cerr << "test_gemma4 FAILED: " << g_failures << " check(s)\n";
  return 1;
}
