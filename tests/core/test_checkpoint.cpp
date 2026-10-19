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

// test_checkpoint.cpp — R7: named parameter checkpoint save/load through the
// C ABI.
//
// sb_save_params writes a self-describing binary file of named float32
// tensors; sb_load_params restores values by name into existing tensors. The
// test covers:
//   - bit-exact round-trip of tensors of different shapes (including a
//     zero-numel tensor), with load order different from save order;
//   - a load → save round-trip reproducing the original file byte for byte;
//   - idempotent reloading;
//   - all-or-nothing failure semantics: a missing name or a shape mismatch
//     leaves every target untouched, including those that did match;
//   - corrupt, truncated and trailing-byte files are rejected;
//   - duplicate save names are rejected without touching an existing file;
//   - invalid arguments and missing files report the documented status.
//
// This runs entirely on the CPU; sb_tensor handles are host tensors.

#include <sonicboom/capi.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
}

// Run one call, print the diagnostic on failure, free the error handle and
// clear the slot (the caller never reuses a dangling handle).
void expect(sb_status got, sb_status want, sb_error*& err, const char* what) {
  check(got == want, what);
  if (err) {
    if (got != want)
      std::fprintf(stderr, "  %s: %s\n", what, sb_error_message(err));
    sb_error_free(err);
    err = nullptr;
  }
}

sb_tensor* make_tensor(const std::vector<int64_t>& dims,
                       const std::vector<float>& values, sb_error*& err,
                       const char* what) {
  sb_tensor* t = nullptr;
  const sb_status st =
      sb_tensor_from_f32(dims.data(), static_cast<int64_t>(dims.size()),
                         values.empty() ? nullptr : values.data(),
                         values.size() * sizeof(float), &t, &err);
  expect(st, SB_OK, err, what);
  return t;
}

std::vector<float> read_tensor(const sb_tensor* t, std::size_t numel,
                               sb_error*& err, const char* what) {
  std::vector<float> v(numel);
  std::size_t written = 0;
  const sb_status st =
      sb_tensor_bytes(t, v.empty() ? nullptr : v.data(), v.size() * sizeof(float),
                      &written, &err);
  expect(st, SB_OK, err, what);
  check(written == v.size() * sizeof(float), what);
  return v;
}

std::vector<char> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
}

void write_file(const fs::path& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

} // namespace

int main() {
  sb_error* err = nullptr;
  const fs::path dir = fs::temp_directory_path() / "sonicboom_checkpoint_test";
  const fs::path ckpt = dir / "params.ckpt";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  if (!fs::exists(dir)) {
    std::fprintf(stderr, "FAIL: cannot create %s\n", dir.c_str());
    return 1;
  }

  // --- source tensors --------------------------------------------------------
  std::vector<float> emb_values(35), norm_values(5), bias_values(1);
  for (std::size_t i = 0; i < emb_values.size(); ++i)
    emb_values[i] = 0.125f * static_cast<float>(i) - 1.0f;
  for (std::size_t i = 0; i < norm_values.size(); ++i)
    norm_values[i] = 0.75f + 0.0625f * static_cast<float>(i);
  bias_values[0] = -0.375f;

  sb_tensor* src_emb = make_tensor({7, 5}, emb_values, err, "source embedding");
  sb_tensor* src_norm = make_tensor({5}, norm_values, err, "source norm");
  sb_tensor* src_bias = make_tensor({1}, bias_values, err, "source bias");
  sb_tensor* src_empty = make_tensor({0, 3}, {}, err, "source empty");
  if (g_failures) {
    std::fprintf(stderr, "FAIL: tensor setup\n");
    return 1;
  }

  const char* save_names[] = {"embedding", "norm", "bias", "empty"};
  const sb_tensor* save_tensors[] = {src_emb, src_norm, src_bias, src_empty};
  expect(sb_save_params(ckpt.c_str(), save_names, save_tensors, 4, &err), SB_OK,
         err, "sb_save_params");
  check(fs::exists(ckpt), "checkpoint file exists");
  check(fs::file_size(ckpt) > 8 + 8, "checkpoint has entries");

  // --- round-trip, load order independent of save order ----------------------
  sb_tensor* dst_emb = make_tensor({7, 5}, std::vector<float>(35, 9.0f), err,
                                   "target embedding");
  sb_tensor* dst_norm = make_tensor({5}, std::vector<float>(5, -9.0f), err,
                                    "target norm");
  sb_tensor* dst_bias = make_tensor({1}, std::vector<float>(1, 7.0f), err,
                                    "target bias");
  sb_tensor* dst_empty = make_tensor({0, 3}, {}, err, "target empty");

  const char* load_names[] = {"bias", "empty", "embedding", "norm"};
  sb_tensor* load_tensors[] = {dst_bias, dst_empty, dst_emb, dst_norm};
  expect(sb_load_params(ckpt.c_str(), load_names, load_tensors, 4, &err), SB_OK,
         err, "sb_load_params");
  check(same_bits(read_tensor(dst_emb, 35, err, "read embedding"), emb_values),
        "embedding round-trips bit-exactly");
  check(same_bits(read_tensor(dst_norm, 5, err, "read norm"), norm_values),
        "norm round-trips bit-exactly");
  check(same_bits(read_tensor(dst_bias, 1, err, "read bias"), bias_values),
        "bias round-trips bit-exactly");

  // Idempotent: a second load produces the same values.
  expect(sb_load_params(ckpt.c_str(), load_names, load_tensors, 4, &err), SB_OK,
         err, "sb_load_params again");
  check(same_bits(read_tensor(dst_emb, 35, err, "read embedding again"),
                  emb_values),
        "second load is idempotent");

  // --- load → save reproduces the file byte for byte -------------------------
  const fs::path ckpt2 = dir / "params2.ckpt";
  const char* rt_names[] = {"embedding", "norm", "bias", "empty"};
  const sb_tensor* rt_tensors[] = {dst_emb, dst_norm, dst_bias, dst_empty};
  expect(sb_save_params(ckpt2.c_str(), rt_names, rt_tensors, 4, &err), SB_OK,
         err, "sb_save_params round-trip");
  check(read_file(ckpt) == read_file(ckpt2),
        "save → load → save is byte-identical");

  // --- failure paths leave every target untouched ----------------------------
  sb_tensor* probe = make_tensor({7, 5}, std::vector<float>(35, 3.25f), err,
                                 "probe embedding");
  sb_tensor* probe_ok_shape = make_tensor({5}, std::vector<float>(5, 4.5f), err,
                                          "probe norm");

  const char* missing_names[] = {"embedding", "not_there"};
  sb_tensor* missing_targets[] = {probe, probe_ok_shape};
  expect(sb_load_params(ckpt.c_str(), missing_names, missing_targets, 2, &err),
         SB_ERR_BINDING, err, "missing name is rejected");
  check(same_bits(read_tensor(probe, 35, err, "probe embedding after missing"),
                  std::vector<float>(35, 3.25f)),
        "missing name leaves a matching target untouched");

  sb_tensor* wrong_shape = make_tensor({6}, std::vector<float>(6, 8.0f), err,
                                       "wrong-shape norm");
  const char* mismatched_names[] = {"embedding", "norm"};
  sb_tensor* mismatched_targets[] = {probe, wrong_shape};
  expect(
      sb_load_params(ckpt.c_str(), mismatched_names, mismatched_targets, 2, &err),
      SB_ERR_BINDING, err, "shape mismatch is rejected");
  check(same_bits(read_tensor(probe, 35, err, "probe embedding after mismatch"),
                  std::vector<float>(35, 3.25f)),
        "shape mismatch leaves an earlier matching target untouched");

  // Corrupt magic.
  const fs::path bad_magic = dir / "bad-magic.ckpt";
  std::vector<char> bytes = read_file(ckpt);
  bytes[0] = 'X';
  write_file(bad_magic, bytes);
  expect(sb_load_params(bad_magic.c_str(), missing_names, missing_targets, 2,
                        &err),
         SB_ERR_WEIGHT, err, "bad magic is rejected");

  // Truncated file.
  const fs::path truncated = dir / "truncated.ckpt";
  std::vector<char> cut = read_file(ckpt);
  cut.resize(cut.size() - 5);
  write_file(truncated, cut);
  expect(sb_load_params(truncated.c_str(), missing_names, missing_targets, 2,
                        &err),
         SB_ERR_WEIGHT, err, "truncated file is rejected");

  // Trailing bytes are corruption too.
  const fs::path trailing = dir / "trailing.ckpt";
  std::vector<char> extra = read_file(ckpt);
  extra.push_back('\0');
  write_file(trailing, extra);
  expect(sb_load_params(trailing.c_str(), missing_names, missing_targets, 2,
                        &err),
         SB_ERR_WEIGHT, err, "trailing bytes are rejected");

  // Duplicate save names are rejected before the destination is touched: the
  // existing good file must still load.
  const char* dup_names[] = {"norm", "norm"};
  const sb_tensor* dup_tensors[] = {src_norm, src_bias};
  expect(sb_save_params(ckpt.c_str(), dup_names, dup_tensors, 2, &err),
         SB_ERR_BINDING, err, "duplicate save names are rejected");
  expect(sb_load_params(ckpt.c_str(), load_names, load_tensors, 4, &err), SB_OK,
         err, "rejected save left the existing file intact");

  // Missing file, invalid arguments.
  expect(sb_load_params((dir / "absent.ckpt").c_str(), load_names, load_tensors,
                        4, &err),
         SB_ERR_WEIGHT, err, "missing file is rejected");
  expect(sb_save_params(nullptr, save_names, save_tensors, 4, &err),
         SB_ERR_INTERNAL, err, "null path is rejected");
  expect(sb_load_params(ckpt.c_str(), nullptr, load_tensors, 1, &err),
         SB_ERR_INTERNAL, err, "null name list is rejected");

  // --- cleanup ---------------------------------------------------------------
  sb_tensor_free(src_emb);
  sb_tensor_free(src_norm);
  sb_tensor_free(src_bias);
  sb_tensor_free(src_empty);
  sb_tensor_free(dst_emb);
  sb_tensor_free(dst_norm);
  sb_tensor_free(dst_bias);
  sb_tensor_free(dst_empty);
  sb_tensor_free(probe);
  sb_tensor_free(probe_ok_shape);
  sb_tensor_free(wrong_shape);
  fs::remove_all(dir, ec);

  std::printf("checkpoint: 4 named tensors, bit-exact round-trip, %s\n",
              g_failures ? "FAILURES" : "all checks passed");
  return g_failures ? 1 : 0;
}
