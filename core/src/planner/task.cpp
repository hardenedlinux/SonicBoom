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

#include <sonicboom/planner/task.h>

namespace sonicboom::planner {

const char* task_kind_name(TaskKind k) noexcept {
  switch (k) {
    case TaskKind::Compute: return "compute";
    case TaskKind::Transfer: return "transfer";
    case TaskKind::Allocate: return "allocate";
    case TaskKind::Release: return "release";
    case TaskKind::Synchronize: return "synchronize";
  }
  return "?";
}

const ComputeTaskDesc* TaskDesc::as_compute() const noexcept {
  return std::get_if<ComputeTaskDesc>(&payload);
}
const TransferTaskDesc* TaskDesc::as_transfer() const noexcept {
  return std::get_if<TransferTaskDesc>(&payload);
}
const AllocateTaskDesc* TaskDesc::as_allocate() const noexcept {
  return std::get_if<AllocateTaskDesc>(&payload);
}
const ReleaseTaskDesc* TaskDesc::as_release() const noexcept {
  return std::get_if<ReleaseTaskDesc>(&payload);
}

} // namespace sonicboom::planner
