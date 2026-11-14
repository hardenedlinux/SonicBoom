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

// Optional per-node output dump for numerical diagnosis (debug only).
//
// SONICBOOM_DUMP_NODES is a comma-separated list of node-name prefixes and
// SONICBOOM_DUMP_DIR an output directory. With both set, a backend writes the
// first output of every node whose name starts with a prefix to
// <dir>/<step:06d>-<node>.bin, with float32 outputs as float32 and integer
// outputs as int32 (the device backend copies back synchronously first). The
// step is the index of the graph execution, from 0 (begin_graph() marks one).
//
// A node's runtime name is its operator (`rms_norm`, `moe_route`, ...; the
// lowering records the op kind, not the emitter's per-node name), so several
// nodes of one op share it. The dump therefore names a node by its first
// output's tensor name, which the emitter made unique (`x212`, `moe_ids12`,
// `logits`): that is the name a prefix selects and the file carries.
//
// create() reads the environment once and returns null when either variable is
// unset or empty, so a backend pays one pointer test per node and nothing
// else while the dump is off. Nothing here reads or changes the graph.

#include <sonicboom/sx/ir.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::planner {

class NodeDump {
public:
  // Null when the dump is off.
  static std::unique_ptr<NodeDump> create();

  // One graph execution started.
  void begin_graph() noexcept { ++step_; }

  // Write the first output of `node`: `bytes` host bytes of `dtype` elements.
  // A name that does not match, or a dtype outside float32/integer, writes
  // nothing.
  void write(std::string_view node, const void *data, std::size_t bytes,
             sx::DType dtype) const;

private:
  NodeDump(std::string dir, std::vector<std::string> prefixes);

  bool matches(std::string_view node) const;

  std::string dir_;
  std::vector<std::string> prefixes_;
  uint64_t step_ = 0;
};

} // namespace sonicboom::planner
