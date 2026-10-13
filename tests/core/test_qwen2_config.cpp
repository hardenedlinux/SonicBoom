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

// Section C.6 / Qwen generality: arch-generic config loader. Proves the loader
// generalizes beyond Gemma 4 by reading a Qwen2.5 GGUF's `general.architecture`
// ("qwen2") and deriving a Qwen2 Gemma4Config from the qwen2.* metadata — the
// standard decoder block (SwiGLU, no SWA/shared-KV/per-layer-input/softcap,
// head_dim = embedding_length / head_count). Config-level only: it binds no
// weights, because Qwen2.5 Q4_K_M uses quant types (Q5_0/Q8_0/Q6_K) and
// attn_q/k/v biases SonicBoom does not yet model (recorded as blocked in the
// post-v1 list).

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

std::string find_model() {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("SONICBOOM_QWEN_GGUF")) candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
#endif
  candidates.push_back("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

void test_qwen2_config(const sbmodel::Gemma4Config& c) {
  // --- frozen Qwen2.5-0.5B-Instruct facts (dump of the Q4_K_M GGUF) ----------
  check(c.arch == sbmodel::ModelArch::Qwen2, "config: arch Qwen2");
  check(c.block_count == 24, "config: block_count 24");
  check(c.embedding_length == 896, "config: embedding_length 896");
  check(c.feed_forward_length == 4864, "config: feed_forward_length 4864");
  check(c.head_count == 14, "config: head_count 14");
  check(c.head_count_kv == 2, "config: head_count_kv 2");
  check(c.head_dim_global == 64, "config: head_dim_global 64 (896 / 14)");
  check(c.head_dim_swa == 0, "config: head_dim_swa 0 (no SWA)");
  check(c.per_layer_input == 0, "config: per_layer_input 0 (no per-layer input)");
  check(c.context_length == 32768, "config: context_length 32768");
  check(c.sliding_window == 0, "config: sliding_window 0");
  check(c.vocab_size == 151936, "config: vocab_size 151936");
  check(std::fabs(c.rms_epsilon - 1e-6f) < 1e-7f, "config: rms_epsilon 1e-6");
  check(c.final_logit_softcapping == 0.0f, "config: final_logit_softcapping 0");
  check(c.rope_base == 1e6f, "config: rope_base 1e6");
  check(c.rope_base_swa == 0.0f, "config: rope_base_swa 0");
  check(c.shared_kv_layers == 0, "config: shared_kv_layers 0");
  check(c.sliding_window_pattern.empty(), "config: sliding_window_pattern empty");
  check(c.valid(), "config: valid()");

  // --- every layer is global (no SWA), no shared KV --------------------------
  check(c.n_layer_kv() == 24, "config: 24 KV layers (no sharing)");
  const sbmodel::LayerConfig l0 = c.layer_config(0);
  check(l0.attention == sbmodel::AttentionKind::Global, "layer 0: global");
  check(l0.head_dim == 64 && l0.rope_base == 1e6f && l0.sliding_window == 0,
        "layer 0: head_dim 64 / rope 1e6 / no window");
  check(c.has_kv(0) && c.has_kv(23), "config: layers 0/23 have KV");
  check(c.kv_donor_layer(7) == 7, "config: kv_donor_layer identity (no sharing)");
}

} // namespace

int main() {
  const std::string path = find_model();
  if (path.empty()) {
    std::cout << "  (skipped: Qwen2.5 model not found)\n";
    std::cout << "test_qwen2_config OK\n";
    return 0;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    check(false, "model: failed to load GGUF");
    std::cerr << "test_qwen2_config FAILED: 1 check(s)\n";
    return 1;
  }

  // detect_arch must report qwen2 before load_config is even attempted.
  auto arch = sbmodel::detect_arch(*r);
  check(arch && *arch == sbmodel::ModelArch::Qwen2, "detect_arch: qwen2");
  if (!arch) {
    std::cerr << "  detect_arch error: " << arch.error() << "\n";
    std::cerr << "test_qwen2_config FAILED: 1 check(s)\n";
    return 1;
  }

  auto cfg = sbmodel::load_config(*r);
  if (!cfg) {
    std::cerr << "  load_config error: " << cfg.error() << "\n";
    check(false, "model: load_config succeeded");
    std::cerr << "test_qwen2_config FAILED: 1 check(s)\n";
    return 1;
  }

  test_qwen2_config(*cfg);

  if (g_failures == 0) {
    std::cout << "test_qwen2_config OK\n";
    return 0;
  }
  std::cerr << "test_qwen2_config FAILED: " << g_failures << " check(s)\n";
  return 1;
}
