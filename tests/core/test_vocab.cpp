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

// GGUF vocabulary tokenizer acceptance. Loads the Gemma 4 tokenizer from the
// GGUF (skips cleanly when the model is absent) and checks two properties over
// 20 Chinese / English / code samples:
//
//   1. tokenize(x) matches llama.cpp's llama-tokenize exactly. The golden
//      token ids below were produced by tools/baseline/tokenize_text (llama.cpp
//      commit 43fe9c64281ef735046adc025e9e7559a1f659a5, add_special=true).
//   2. detokenize(tokenize(x)) == x  (round-trip; <bos> is control and is
//      stripped, ▁/byte escapes are undone).

#include <sonicboom/gguf/reader.h>
#include <sonicboom/vocab/vocab.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbvocab = sonicboom::vocab;

namespace {

int g_failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

std::string find_model() {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF"))
    candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/models/gemma-4-E4B-it-Q3_K_M.gguf");
#endif
  candidates.push_back("models/gemma-4-E4B-it-Q3_K_M.gguf");
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

struct Sample {
  std::string text;
  std::vector<int32_t> tokens;
};

const std::vector<Sample> kSamples = {
    {"Hello, world!", {2, 9259, 236764, 1902, 236888}},
    {"What's the weather like today? It's sunny.",
     {2, 3689, 236789, 236751, 506, 7606, 1133, 3124, 236881, 1030, 236789,
      236751, 17635, 236761}},
    {"你好，世界！", {2, 144626, 236900, 12811, 237354}},
    {"今天天气真不错，我们去公园散步吧。",
     {2, 22601, 122362, 237783, 56033, 236900, 6932, 237648, 95361, 239057,
      238269, 238702, 236924}},
    {"def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
     {2, 2063, 10779, 236769, 236749, 1473, 107, 140, 2060, 538, 768, 538, 655,
      236743, 236778, 1663, 10779, 236769, 236749, 236772, 236770, 236768, 900,
      10779, 236769, 236749, 236772, 236778, 236768, 107}},
    {"int main() {\n    printf(\"hello\\n\");\n    return 0;\n}\n",
     {2, 720, 1689, 825, 642, 107, 140, 8641, 885, 23391, 236785, 236749, 1443,
      107, 140, 2060, 236743, 236771, 236793, 107, 236783, 107}},
    {"template <typename T>\nT add(T a, T b) { return a + b; }\n",
     {2, 9944, 655, 43217, 558, 236813, 107, 236774, 1138, 236769, 236774, 496,
      236764, 558, 518, 236768, 642, 994, 496, 900, 518, 236793, 682, 107}},
    {"机器学习和深度学习：Transformer、Attention、BERT。",
     {2, 202589, 237206, 70126, 23486, 237184, 86313, 236951, 82839, 236951,
      53208, 236924}},
    {"The quick brown fox jumps over the lazy dog.",
     {2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761}},
    {"// 这是一个注释\nint x = 42; // 赋值\n",
     {2, 715, 236743, 120509, 138730, 107, 720, 1123, 578, 236743, 236812,
      236778, 236793, 973, 236743, 92454, 107}},
    {"3.14159, 2.71828, 100%, #42 @test",
     {2, 236800, 236761, 236770, 236812, 236770, 236810, 236819, 236764, 236743,
      236778, 236761, 236832, 236770, 236828, 236778, 236828, 236764, 236743,
      236770, 236771, 236771, 10014, 997, 236812, 236778, 1392, 2181}},
    {"Don't stop believing, it's a wonderful life.",
     {2, 8993, 236789, 236745, 4721, 44454, 236764, 625, 236789, 236751, 496,
      10455, 1972, 236761}},
    {"我在用SonicBoom做LLM推理，速度很快！",
     {2, 75190, 237105, 69771, 122196, 237893, 2182, 236792, 198749, 236900,
      27690, 64455, 237354}},
    {"{\"name\": \"SonicBoom\", \"version\": \"1.0\", \"tags\": [\"ml\", \"cuda\"]}",
     {2, 14937, 1201, 1083, 623, 69771, 122196, 827, 623, 4406, 1083, 623,
      236770, 236761, 236771, 827, 623, 19112, 1083, 11058, 1528, 827, 623,
      46089, 99471}},
    {"Hello \U0001F600 world \U0001F680 test",
     {2, 9259, 163543, 1902, 236743, 242015, 1594}},
    {"line1\nline2\n\nline4\n",
     {2, 1257, 236770, 107, 1257, 236778, 108, 1257, 236812, 107}},
    {"自然语言处理是人工智能的一个重要分支。",
     {2, 22841, 37557, 19157, 237026, 136119, 55399, 16918, 132286, 236924}},
    {"#!/bin/bash\necho \"$HOME\"\nexit 0\n",
     {2, 38752, 6264, 236786, 42422, 107, 17454, 6610, 30903, 236775, 107,
      15560, 236743, 236771, 107}},
    {"How many tokens does this sentence contain?",
     {2, 3910, 1551, 21739, 1677, 672, 13315, 3014, 236881}},
    {"col1\tcol2\tcol3",
     {2, 1475, 236770, 255968, 1475, 236778, 255968, 1475, 236800}},
};

}  // namespace

int main() {
  const std::string path = find_model();
  if (path.empty()) {
    std::cout << "  (skipped: Gemma 4 model not found)\n";
    std::cout << "test_vocab OK\n";
    return 0;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    std::cerr << "  failed to load GGUF\n";
    std::cerr << "test_vocab FAILED: 1 check(s)\n";
    return 1;
  }
  auto v = sbvocab::load_vocab(*r);
  if (!v) {
    std::cerr << "  load_vocab failed: " << v.error() << "\n";
    std::cerr << "test_vocab FAILED: 1 check(s)\n";
    return 1;
  }

  for (size_t i = 0; i < kSamples.size(); ++i) {
    const Sample& s = kSamples[i];

    // 1. tokenize matches llama-tokenize exactly.
    const std::vector<int32_t> toks = sbvocab::tokenize(*v, s.text);
    if (toks != s.tokens) {
      check(false, "sample " + std::to_string(i) + ": tokenize mismatch");
      std::cerr << "    got  ";
      for (int32_t t : toks) std::cerr << t << " ";
      std::cerr << "\n    want ";
      for (int32_t t : s.tokens) std::cerr << t << " ";
      std::cerr << "\n";
    }

    // 2. detokenize(tokenize(x)) == x.
    const std::string back =
        sbvocab::detokenize(*v, toks.data(), toks.size());
    if (back != s.text) {
      check(false, "sample " + std::to_string(i) + ": round-trip mismatch");
      std::cerr << "    got  \"" << back << "\"\n";
      std::cerr << "    want \"" << s.text << "\"\n";
    }
  }

  if (g_failures == 0) {
    std::cout << "test_vocab OK (" << kSamples.size() << " samples)\n";
    return 0;
  }
  std::cerr << "test_vocab FAILED: " << g_failures << " check(s)\n";
  return 1;
}
