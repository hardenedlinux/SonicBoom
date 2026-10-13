// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace sonicboom::quant {
std::expected<std::vector<float>, std::string> dequant_foreign_to_f32(
    uint32_t ggml_type_id, const uint8_t* data, uint64_t n_elems);
}
