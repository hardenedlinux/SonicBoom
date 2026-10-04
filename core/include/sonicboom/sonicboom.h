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

// Umbrella header for the Native Torch (nt) Layer 2 public API. No c10/ATen
// type is reachable from any of these headers.

#include <sonicboom/backend.h>
#include <sonicboom/scalar_type.h>
#include <sonicboom/device.h>
#include <sonicboom/layout.h>
#include <sonicboom/memory_format.h>
#include <sonicboom/scalar.h>
#include <sonicboom/tensor.h>
#include <sonicboom/value.h>
#include <sonicboom/argument.h>
#include <sonicboom/schema.h>
#include <sonicboom/argument_list.h>
#include <sonicboom/operator_handle.h>
#include <sonicboom/registration.h>
#include <sonicboom/allocator.h>
