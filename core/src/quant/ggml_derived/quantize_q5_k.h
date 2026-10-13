// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#pragma once

#include <cstdint>
#include <vector>

namespace sonicboom::quant {
std::vector<uint8_t> quantize_f32_to_q5_k(const float* src, uint64_t n_elems);
}
