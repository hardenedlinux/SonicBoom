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

// sb-moss — MOSS-Image bring-up driver.
//
//   sb-moss encode --model DIR (--text TEXT | --tokens "1,2,3") --out FILE
//                  [--raw] [--dump-tokens]
//
// Runs the Qwen3.5 text encoder over a prompt and writes the last hidden state
// as "MOSSHST1", int64 tokens, int64 hidden, then float32[tokens * hidden].
// --raw tokenizes TEXT exactly as given; the default wraps it in the encoder's
// chat template (the same rendering the MOSS-Image pipeline's encode_prompt
// uses with enable_thinking=True). --dump-tokens prints the ids only and does
// not run the model.

#include <sonicboom/model/moss_text.h>
#include <sonicboom/vocab/vocab.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: sb-moss encode --model DIR (--text TEXT | --tokens LIST) --out FILE\n"
         "                      [--raw] [--dump-tokens]\n"
         "       sb-moss tokens --model DIR (--text TEXT | --chat) [--raw]\n"
         "\n"
         "  --model DIR    a text_encoder directory (config.json + index + shards)\n"
         "  --text TEXT    prompt text\n"
         "  --tokens LIST  comma-separated token ids (skips the tokenizer)\n"
         "  --raw          do not apply the chat template\n"
         "  --dump-tokens  print the token ids and exit\n"
         "  --out FILE     hidden-state output (`encode`)\n";
}

// The pipeline's chat rendering for one user turn with enable_thinking=True
// (MOSS-Image's encode_prompt): the Qwen3.5 template emits no system block for
// a single-user conversation, so the text is the user turn plus the assistant
// generation prompt and its empty thinking block.
std::string chat_prompt(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 64);
  out += "<|im_start|>user\n";
  out += text;
  out += "<|im_end|>\n<|im_start|>assistant\n<think>\n";
  return out;
}

std::vector<int32_t> parse_ids(const std::string& list) {
  std::vector<int32_t> ids;
  std::stringstream ss(list);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    ids.push_back(static_cast<int32_t>(std::stol(item)));
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string_view command = argv[1];
  if (command != "encode" && command != "tokens") {
    usage();
    return 2;
  }

  std::string model_dir;
  std::string text;
  std::string ids_arg;
  std::string out_path;
  bool have_text = false;
  bool raw = false;
  bool dump_tokens = false;
  for (int i = 2; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "sb-moss: " << what << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") {
      model_dir = next("--model");
    } else if (a == "--text") {
      text = next("--text");
      have_text = true;
    } else if (a == "--tokens") {
      ids_arg = next("--tokens");
    } else if (a == "--out") {
      out_path = next("--out");
    } else if (a == "--raw") {
      raw = true;
    } else if (a == "--dump-tokens") {
      dump_tokens = true;
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::cerr << "sb-moss: unknown option " << a << "\n";
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::cerr << "sb-moss: --model is required\n";
    return 2;
  }

  std::vector<int32_t> ids;
  if (!ids_arg.empty()) {
    ids = parse_ids(ids_arg);
  } else {
    if (!have_text) {
      std::cerr << "sb-moss: one of --text or --tokens is required\n";
      return 2;
    }
    // The tokenizer sits beside the text encoder in the MOSS-Image layout, so
    // fall back to the sibling directory when the encoder directory has none.
    std::filesystem::path tokenizer_dir =
        std::filesystem::path(model_dir) / "tokenizer.json";
    if (!std::filesystem::exists(tokenizer_dir)) {
      tokenizer_dir = std::filesystem::path(model_dir).parent_path() /
                      "tokenizer" / "tokenizer.json";
    }
    auto vocab = sonicboom::vocab::load_tokenizer_json(tokenizer_dir);
    if (!vocab) {
      std::cerr << "sb-moss: " << vocab.error() << "\n";
      return 1;
    }
    const std::string prompt = raw ? text : chat_prompt(text);
    ids = sonicboom::vocab::tokenize(*vocab, prompt, /*parse_special=*/true,
                                     /*add_special=*/false);
  }

  if (dump_tokens || command == "tokens") {
    for (size_t i = 0; i < ids.size(); ++i) {
      std::cout << (i ? " " : "") << ids[i];
    }
    std::cout << "\n";
    if (dump_tokens || command == "tokens") return 0;
  }

  auto encoder = sonicboom::moss::TextEncoder::load(model_dir);
  if (!encoder) {
    std::cerr << "sb-moss: " << encoder.error() << "\n";
    return 1;
  }
  const auto& cfg = (*encoder)->config();
  std::cerr << "encoder: " << cfg.num_layers << " layers, hidden "
            << cfg.hidden_size << ", weights " << (*encoder)->weight_bytes()
            << " B\n";

  const auto start = std::chrono::steady_clock::now();
  auto hidden = (*encoder)->forward(ids);
  if (!hidden) {
    std::cerr << "sb-moss: " << hidden.error() << "\n";
    return 1;
  }
  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start);
  std::cerr << "forward: " << ids.size() << " tokens in " << elapsed.count()
            << " s (" << ids.size() / elapsed.count() << " tok/s)\n";

  if (out_path.empty()) {
    std::cerr << "sb-moss: --out is required for encode\n";
    return 2;
  }
  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "sb-moss: cannot write " << out_path << "\n";
    return 1;
  }
  out.write("MOSSHST1", 8);
  out.write(reinterpret_cast<const char*>(&hidden->tokens), sizeof(int64_t));
  out.write(reinterpret_cast<const char*>(&hidden->hidden_size), sizeof(int64_t));
  out.write(reinterpret_cast<const char*>(hidden->values.data()),
            static_cast<std::streamsize>(hidden->values.size() * sizeof(float)));
  if (!out) {
    std::cerr << "sb-moss: write failed for " << out_path << "\n";
    return 1;
  }
  std::cerr << "wrote " << out_path << " (" << hidden->tokens << "x"
            << hidden->hidden_size << " f32)\n";
  return 0;
}
