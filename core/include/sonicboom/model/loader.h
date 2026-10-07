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

#pragma once

#include <expected>
#include <string>
#include <vector>

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/config.h>
#include <sonicboom/model/weights.h>

// Gemma 4 model loader (Phase 5A): bind the frozen model facts (design/
// gemma4-model-map.md) to a validated in-memory model — explicit config, bound
// weights, and a 42-layer execution plan. Pure C++23; depends only on the GGUF
// reader and the quant representation, never on ggml/llama.cpp.

namespace sonicboom::model {

// A loaded, validated Gemma 4 model. All spans borrow the reader's buffer.
struct Gemma4Model {
  Gemma4Config config;
  Gemma4Weights weights;
  std::vector<LayerConfig> plan;  // one per block, index == layer
};

// Load and validate `r` as a Gemma 4 E4B model. Returns an error string when
// the metadata is inconsistent, a required tensor is missing, or a tensor's
// type/shape disagrees with the config (see the model map for the expected
// shapes). Never throws; never reads past the reader's buffer.
std::expected<Gemma4Model, std::string> load_gemma4(const gguf::Reader& r);

} // namespace sonicboom::model
