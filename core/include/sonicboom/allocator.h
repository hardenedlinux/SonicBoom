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

#include <cstddef>

namespace nt {

// Layer 2 allocator boundary (manifest M2). Define the interface only; the
// complete allocator system is deferred. The ABI-level deleter is a plain C
// function pointer (ctx, data) — NOT std::function — so it can cross the C ABI
// boundary later.
class Allocator {
 public:
  using DeleterFn = void (*)(void* ctx, void* data);

  virtual ~Allocator() = default;

  virtual void* allocate(size_t nbytes) = 0;
  virtual void deallocate(void* data) = 0;

  // C-compatible deleter + context for memory returned by allocate().
  virtual DeleterFn deleter() const = 0;
  virtual void* context() const = 0;
};

} // namespace nt
