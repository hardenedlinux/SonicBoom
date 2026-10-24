// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Converted-weight disk cache (internal).
//
// The MOSS-Image DiT loader reads BF16 safetensors and converts every matrix
// to Q8_0 on the CPU: 15 s of a 20 s load on this machine. The converted bytes
// are device-independent, so they can be persisted once and read back on the
// next process start.
//
// Identity and validation are deliberately split by cost:
//
//   - The cache file name carries the checkpoint's SHA-256, a content hash
//     computed once (at conversion time) and recorded in a sidecar.
//   - Later opens never rehash 9.5 GiB. The sidecar keys on the shard paths and
//     records (device, inode, size, mtime_ns) for each; a matching stat set is
//     what admits the cache. A shard replaced in place with the same size and
//     mtime is outside that cheap validation, exactly as for any
//     mtime-validated build cache; delete the cache directory to force a
//     rebuild.
//   - Every tensor record carries an XXH64 of its payload; reads verify it.
//     That catches a truncated or torn file (a crash before rename cannot
//     produce one, but a disk error or a hand-edited file can).
//
// Files are written to a temporary name and renamed into place while holding
// an exclusive flock, so a concurrent process either sees the old state or
// the complete new file, never a partial one. `SONICBOOM_WEIGHT_CACHE=0`
// disables both lookup and writing.

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::model::weight_cache {

inline constexpr const char *kFormatTag = "q8_0-v1";

struct TensorRecord {
  std::string name;
  uint64_t rows = 0;
  uint64_t cols = 0;
  uint64_t q8_offset = 0;  // absolute file offset
  uint64_t q8_bytes = 0;
  uint64_t q8_hash = 0;    // XXH64 of the payload
  uint64_t bias_offset = 0;
  uint64_t bias_count = 0;  // f32 elements; 0 = no bias
  uint64_t bias_hash = 0;
};

struct CacheFile {
  std::filesystem::path path;
  std::vector<TensorRecord> tensors;

  const TensorRecord *find(std::string_view name) const;
};

// Cache directory: $SONICBOOM_WEIGHT_CACHE, else $XDG_CACHE_HOME/sonicboom/
// weights, else ~/.cache/sonicboom/weights. nullopt when the cache is
// disabled by the environment (SONICBOOM_WEIGHT_CACHE=0 or =off).
std::optional<std::filesystem::path> directory();

// Open an existing cache for these shard files. Cheap: never reads the shard
// contents. Returns nullopt on a miss, a sidecar mismatch, a bad header, or a
// disabled cache.
std::optional<CacheFile> open_existing(
    std::span<const std::filesystem::path> shards);

// Read one tensor's converted bytes and bias out of an open cache. `q8_out`
// must be exactly the record's q8_bytes and `bias_out` sized to bias_count
// (ignored when the record has no bias). Verifies both hashes.
bool read_tensor(const CacheFile &file, const TensorRecord &record,
                 std::span<std::byte> q8_out, std::vector<float> *bias_out);

// One converted tensor handed to store().
struct WriteTensor {
  std::string name;
  uint64_t rows = 0;
  uint64_t cols = 0;
  std::span<const std::byte> q8;
  std::span<const float> bias;  // empty = no bias
};

// SHA-256 of the checkpoint: the per-shard digests chained in shard order. The
// caller computes this once, concurrently with the conversion (it is a full
// read of the checkpoint), and hands the result to store().
std::string checkpoint_sha256(std::span<const std::filesystem::path> shards);

// Persist a freshly converted checkpoint. `checkpoint_sha` is the digest from
// checkpoint_sha256(); an empty value aborts. Writes a temporary file, renames
// it into place, and updates the sidecar. A failure is reported to the caller
// and never leaves a partial cache file.
std::expected<std::filesystem::path, std::string> store(
    std::span<const std::filesystem::path> shards,
    std::span<const WriteTensor> tensors, const std::string &checkpoint_sha);

// SHA-256 of a file's contents, lowercase hex. Exposed for tests.
std::string sha256_file(const std::filesystem::path &path);
// Canonical XXH64 (seed 0) of a buffer. Exposed for tests.
uint64_t xxh64(std::span<const std::byte> data);

// FNV-1a 64 over the shard consumer; exposed for tests.
uint64_t path_fingerprint(std::span<const std::filesystem::path> shards);

} // namespace sonicboom::model::weight_cache
