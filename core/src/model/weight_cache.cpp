// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Converted-weight disk cache; see weight_cache.h for the contract.

#include "weight_cache.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string_view>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sonicboom::model::weight_cache {
namespace {

// --- SHA-256 (FIPS 180-4), streaming ---------------------------------------

constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t block[64] = {};
  uint64_t block_len = 0;
  uint64_t total = 0;

  void compress(const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5],
             g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + s1 + ch + kSha256K[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  void update(const uint8_t *p, size_t n) {
    total += n;
    while (n > 0) {
      const size_t take = std::min(n, size_t(64) - block_len);
      std::memcpy(block + block_len, p, take);
      block_len += take;
      p += take;
      n -= take;
      if (block_len == 64) {
        compress(block);
        block_len = 0;
      }
    }
  }

  std::array<uint8_t, 32> finish() {
    const uint64_t bits = total * 8;
    const uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (block_len != 56) update(&zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = uint8_t(bits >> (56 - 8 * i));
    update(len, 8);
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 4; ++j)
        out[4 * i + j] = uint8_t(h[i] >> (24 - 8 * j));
    return out;
  }
};

std::string hex(const uint8_t *p, size_t n) {
  static const char *d = "0123456789abcdef";
  std::string s;
  s.reserve(2 * n);
  for (size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 0xf];
  }
  return s;
}

// --- XXH64 ------------------------------------------------------------------

constexpr uint64_t kP1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t kP2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t kP3 = 0x165667B19E3779F9ULL;
constexpr uint64_t kP4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t kP5 = 0x27D4EB2F165667C5ULL;

uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
uint64_t rd64(const uint8_t *p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
uint32_t rd32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
uint64_t round(uint64_t acc, uint64_t in) {
  acc += in * kP2;
  acc = rotl(acc, 31);
  return acc * kP1;
}
uint64_t merge(uint64_t acc, uint64_t val) {
  acc ^= round(0, val);
  return acc * kP1 + kP4;
}

uint64_t xxh64_impl(const uint8_t *p, size_t n) {
  const uint8_t *end = p + n;
  uint64_t h;
  if (n >= 32) {
    uint64_t v1 = kP1 + kP2, v2 = kP2, v3 = 0, v4 = uint64_t(0) - kP1;
    const uint8_t *limit = end - 32;
    do {
      v1 = round(v1, rd64(p)); p += 8;
      v2 = round(v2, rd64(p)); p += 8;
      v3 = round(v3, rd64(p)); p += 8;
      v4 = round(v4, rd64(p)); p += 8;
    } while (p <= limit);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
  } else {
    h = kP5;
  }
  h += uint64_t(n);
  while (p + 8 <= end) {
    h ^= round(0, rd64(p));
    h = rotl(h, 27) * kP1 + kP4;
    p += 8;
  }
  if (p + 4 <= end) {
    h ^= uint64_t(rd32(p)) * kP1;
    h = rotl(h, 23) * kP2 + kP3;
    p += 4;
  }
  while (p < end) {
    h ^= uint64_t(*p) * kP5;
    h = rotl(h, 11) * kP1;
    ++p;
  }
  h ^= h >> 33; h *= kP2;
  h ^= h >> 29; h *= kP3;
  h ^= h >> 32;
  return h;
}

// --- identity ---------------------------------------------------------------

struct ShardId {
  uint64_t size = 0;
  uint64_t dev = 0;
  uint64_t ino = 0;
  uint64_t mtime_ns = 0;
  bool valid = false;
};

ShardId stat_shard(const std::filesystem::path &p) {
  ShardId s;
  struct stat st;
  if (::stat(p.c_str(), &st) != 0) return s;
  s.size = uint64_t(st.st_size);
  s.dev = uint64_t(st.st_dev);
  s.ino = uint64_t(st.st_ino);
  s.mtime_ns = uint64_t(st.st_mtim.tv_sec) * 1000000000ull +
               uint64_t(st.st_mtim.tv_nsec);
  s.valid = true;
  return s;
}

std::string fingerprint_hex(uint64_t fp) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)fp);
  return buf;
}

std::filesystem::path sidecar_path(const std::filesystem::path &dir,
                                   uint64_t fp) {
  return dir / ("moss-image-" + fingerprint_hex(fp) + "-" + kFormatTag + ".id");
}

std::filesystem::path cache_path(const std::filesystem::path &dir,
                                 const std::string &sha) {
  return dir / ("moss-image-" + sha.substr(0, 16) + "-" + kFormatTag + ".sbw");
}

void write_le(uint8_t *p, uint64_t v, size_t n) {
  for (size_t i = 0; i < n; ++i) p[i] = uint8_t(v >> (8 * i));
}
uint64_t read_le(const uint8_t *p, size_t n) {
  uint64_t v = 0;
  for (size_t i = 0; i < n; ++i) v |= uint64_t(p[i]) << (8 * i);
  return v;
}

constexpr char kMagic[8] = {'S', 'B', 'W', 'Q', '8', 'C', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 64;
constexpr uint64_t kAlign = 64;

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

bool pread_exact(int fd, uint64_t off, std::byte *dst, uint64_t n) {
  auto *p = reinterpret_cast<uint8_t *>(dst);
  while (n > 0) {
    const ssize_t got = ::pread(fd, p, size_t(n), off_t(off));
    if (got < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (got == 0) return false;
    p += got;
    off += uint64_t(got);
    n -= uint64_t(got);
  }
  return true;
}

// Record layout, after the 64-byte header:
//   u16 name_len, name bytes, u64 rows, cols, q8_offset, q8_bytes, q8_hash,
//   bias_offset, bias_count, bias_hash
constexpr size_t kFixedRecordBytes = 2 + 8 * 8;

std::vector<uint8_t> encode_record(const TensorRecord &r) {
  std::vector<uint8_t> b(kFixedRecordBytes + r.name.size());
  uint8_t *p = b.data();
  write_le(p, r.name.size(), 2); p += 2;
  std::memcpy(p, r.name.data(), r.name.size()); p += r.name.size();
  for (uint64_t v : {r.rows, r.cols, r.q8_offset, r.q8_bytes, r.q8_hash,
                     r.bias_offset, r.bias_count, r.bias_hash}) {
    write_le(p, v, 8);
    p += 8;
  }
  return b;
}

bool decode_record(const uint8_t *p, const uint8_t *end, TensorRecord &r) {
  if (size_t(end - p) < kFixedRecordBytes) return false;
  const size_t name_len = size_t(read_le(p, 2));
  p += 2;
  if (size_t(end - p) < name_len + 8 * 8) return false;
  r.name.assign(reinterpret_cast<const char *>(p), name_len);
  p += name_len;
  uint64_t v[8];
  for (uint64_t &x : v) {
    x = read_le(p, 8);
    p += 8;
  }
  r.rows = v[0]; r.cols = v[1]; r.q8_offset = v[2]; r.q8_bytes = v[3];
  r.q8_hash = v[4]; r.bias_offset = v[5]; r.bias_count = v[6]; r.bias_hash = v[7];
  return true;
}

} // namespace

std::optional<std::filesystem::path> directory() {
  const char *env = std::getenv("SONICBOOM_WEIGHT_CACHE");
  if (env != nullptr) {
    const std::string_view v(env);
    if (v == "0" || v == "off" || v == "false") return std::nullopt;
    if (!v.empty()) return std::filesystem::path(v);
  }
  const char *xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg != nullptr && *xdg != '\0')
    return std::filesystem::path(xdg) / "sonicboom" / "weights";
  const char *home = std::getenv("HOME");
  if (home != nullptr && *home != '\0')
    return std::filesystem::path(home) / ".cache" / "sonicboom" / "weights";
  return std::nullopt;
}

std::string sha256_file(const std::filesystem::path &path) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return {};
  Sha256 s;
  std::vector<uint8_t> buf(1u << 20);
  for (;;) {
    const size_t got = std::fread(buf.data(), 1, buf.size(), f);
    if (got > 0) s.update(buf.data(), got);
    if (got < buf.size()) break;
  }
  const bool bad = std::ferror(f) != 0;
  std::fclose(f);
  if (bad) return {};
  return hex(s.finish().data(), 32);
}

uint64_t xxh64(std::span<const std::byte> data) {
  return xxh64_impl(reinterpret_cast<const uint8_t *>(data.data()), data.size());
}

uint64_t path_fingerprint(std::span<const std::filesystem::path> shards) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (const auto &p : shards) {
    for (char c : p.string()) {
      h ^= uint8_t(c);
      h *= 0x100000001b3ull;
    }
    h ^= 0;
    h *= 0x100000001b3ull;
  }
  return h;
}

const TensorRecord *CacheFile::find(std::string_view name) const {
  for (const auto &r : tensors)
    if (r.name == name) return &r;
  return nullptr;
}

std::optional<CacheFile> open_existing(
    std::span<const std::filesystem::path> shards) {
  const auto dir = directory();
  if (!dir) return std::nullopt;
  if (shards.empty()) return std::nullopt;
  const auto side = sidecar_path(*dir, path_fingerprint(shards));
  std::ifstream in(side);
  if (!in) return std::nullopt;

  std::string sha;
  std::vector<ShardId> ids;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("sha256 ", 0) == 0) {
      sha = line.substr(7);
    } else if (line.rfind("shard ", 0) == 0) {
      ShardId id;
      unsigned long long dev = 0, ino = 0, mtime = 0, size = 0;
      if (std::sscanf(line.c_str(), "shard %llu %llu %llu %llu", &size, &dev,
                      &ino, &mtime) == 4) {
        id.size = size;
        id.dev = dev;
        id.ino = ino;
        id.mtime_ns = mtime;
        id.valid = true;
        ids.push_back(id);
      }
    }
  }
  if (sha.size() != 64 || ids.size() != shards.size()) return std::nullopt;
  for (size_t i = 0; i < shards.size(); ++i) {
    const ShardId now = stat_shard(shards[i]);
    if (!now.valid || now.size != ids[i].size || now.dev != ids[i].dev ||
        now.ino != ids[i].ino || now.mtime_ns != ids[i].mtime_ns)
      return std::nullopt;
  }

  const auto path = cache_path(*dir, sha);
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return std::nullopt;
  struct stat st;
  if (::fstat(fd, &st) != 0 || uint64_t(st.st_size) < kHeaderBytes) {
    ::close(fd);
    return std::nullopt;
  }
  const uint64_t file_size = uint64_t(st.st_size);
  uint8_t header[kHeaderBytes];
  bool ok = pread_exact(fd, 0, reinterpret_cast<std::byte *>(header), kHeaderBytes);
  uint32_t count = 0;
  uint64_t data_offset = 0;
  if (ok) {
    ok = std::memcmp(header, kMagic, sizeof(kMagic)) == 0 &&
         read_le(header + 8, 4) == kVersion;
    count = uint32_t(read_le(header + 12, 4));
    data_offset = read_le(header + 16, 8);
    ok = ok && count > 0 && count <= (1u << 20) &&
         data_offset >= kHeaderBytes && data_offset <= file_size &&
         hex(header + 24, 32) == sha;
  }
  std::vector<uint8_t> recs;
  if (ok) {
    recs.resize(size_t(data_offset - kHeaderBytes));
    ok = pread_exact(fd, kHeaderBytes, reinterpret_cast<std::byte *>(recs.data()),
                     recs.size());
  }
  CacheFile out;
  out.path = path;
  if (ok) {
    const uint8_t *p = recs.data();
    const uint8_t *end = recs.data() + recs.size();
    out.tensors.reserve(count);
    for (uint32_t i = 0; i < count && ok; ++i) {
      TensorRecord r;
      ok = decode_record(p, end, r);
      p += kFixedRecordBytes + r.name.size();
      if (ok) {
        ok = r.q8_bytes > 0 && r.q8_offset + r.q8_bytes <= file_size &&
             (r.bias_count == 0 ||
              r.bias_offset + r.bias_count * 4 <= file_size);
        out.tensors.push_back(std::move(r));
      }
    }
    ok = ok && out.tensors.size() == count;
  }
  ::close(fd);
  if (!ok) return std::nullopt;
  return out;
}

bool read_tensor(const CacheFile &file, const TensorRecord &record,
                 std::span<std::byte> q8_out, std::vector<float> *bias_out) {
  if (q8_out.size() != record.q8_bytes) return false;
  const int fd = ::open(file.path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  bool ok = pread_exact(fd, record.q8_offset, q8_out.data(), q8_out.size());
  if (ok) ok = xxh64(q8_out) == record.q8_hash;
  if (ok && record.bias_count > 0 && bias_out != nullptr) {
    std::vector<std::byte> raw(size_t(record.bias_count) * 4);
    ok = pread_exact(fd, record.bias_offset, raw.data(), raw.size()) &&
         xxh64(raw) == record.bias_hash;
    if (ok) {
      bias_out->resize(size_t(record.bias_count));
      std::memcpy(bias_out->data(), raw.data(), raw.size());
    }
  }
  ::close(fd);
  return ok;
}

std::string checkpoint_sha256(std::span<const std::filesystem::path> shards) {
  std::string sha;
  for (const auto &s : shards) {
    const std::string part = sha256_file(s);
    if (part.empty()) return {};
    if (sha.empty()) {
      sha = part;
      continue;
    }
    // Chain per-shard digests: deterministic and content-derived, and it does
    // not need one pass over the concatenated file.
    Sha256 chain;
    const std::string both = sha + part;
    chain.update(reinterpret_cast<const uint8_t *>(both.data()), both.size());
    sha = hex(chain.finish().data(), 32);
  }
  return sha;
}

std::expected<std::filesystem::path, std::string> store(
    std::span<const std::filesystem::path> shards,
    std::span<const WriteTensor> tensors, const std::string &sha) {
  const auto dir = directory();
  if (!dir) return std::unexpected("weight cache disabled");
  std::error_code ec;
  std::filesystem::create_directories(*dir, ec);
  if (ec) return std::unexpected("cannot create " + dir->string() + ": " + ec.message());
  if (tensors.empty()) return std::unexpected("no tensors to cache");
  if (sha.size() != 64) return std::unexpected("no checkpoint digest");

  const auto path = cache_path(*dir, sha);
  const auto side = sidecar_path(*dir, path_fingerprint(shards));

  const auto lock_path = *dir / ".sbw.lock";
  const int lock = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (lock < 0) return std::unexpected("cannot open cache lock");
  if (::flock(lock, LOCK_EX) != 0) {
    ::close(lock);
    return std::unexpected("cannot lock the weight cache");
  }
  struct LockGuard {
    int fd;
    ~LockGuard() { ::flock(fd, LOCK_UN); ::close(fd); }
  } guard{lock};

  // Another process may have completed the same cache while we hashed.
  if (std::filesystem::exists(path) && std::filesystem::exists(side))
    return path;

  // Records and data offsets.
  std::vector<TensorRecord> recs(tensors.size());
  uint64_t cursor = kHeaderBytes;
  for (size_t i = 0; i < tensors.size(); ++i) cursor += kFixedRecordBytes + tensors[i].name.size();
  cursor = align_up(cursor);
  const uint64_t data_offset = cursor;
  for (size_t i = 0; i < tensors.size(); ++i) {
    TensorRecord &r = recs[i];
    r.name = tensors[i].name;
    r.rows = tensors[i].rows;
    r.cols = tensors[i].cols;
    r.q8_bytes = tensors[i].q8.size();
    r.q8_hash = xxh64(tensors[i].q8);
    r.q8_offset = cursor;
    cursor = align_up(cursor + r.q8_bytes);
    if (!tensors[i].bias.empty()) {
      r.bias_count = tensors[i].bias.size();
      r.bias_offset = cursor;
      r.bias_hash =
          xxh64({reinterpret_cast<const std::byte *>(tensors[i].bias.data()),
                 tensors[i].bias.size() * sizeof(float)});
      cursor = align_up(cursor + r.bias_count * 4);
    }
  }

  const std::string tmp = path.string() + ".tmp." + std::to_string(::getpid());
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) return std::unexpected("cannot create " + tmp);
  auto pwrite_all = [&](uint64_t off, const void *p, size_t n) {
    const auto *b = static_cast<const uint8_t *>(p);
    while (n > 0) {
      const ssize_t got = ::pwrite(fd, b, n, off_t(off));
      if (got < 0) {
        if (errno == EINTR) continue;
        return false;
      }
      b += got;
      off += uint64_t(got);
      n -= size_t(got);
    }
    return true;
  };

  uint8_t header[kHeaderBytes] = {};
  std::memcpy(header, kMagic, sizeof(kMagic));
  write_le(header + 8, kVersion, 4);
  write_le(header + 12, recs.size(), 4);
  write_le(header + 16, data_offset, 8);
  for (int i = 0; i < 32; ++i) {
    const char *h = sha.c_str() + 2 * i;
    auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    header[24 + i] = uint8_t((nibble(h[0]) << 4) | nibble(h[1]));
  }
  bool ok = pwrite_all(0, header, sizeof(header));
  uint64_t pos = kHeaderBytes;
  for (const TensorRecord &r : recs) {
    if (!ok) break;
    const auto rec = encode_record(r);
    ok = pwrite_all(pos, rec.data(), rec.size());
    pos += rec.size();
  }
  for (size_t i = 0; i < tensors.size() && ok; ++i) {
    const TensorRecord &r = recs[i];
    ok = pwrite_all(r.q8_offset, tensors[i].q8.data(), r.q8_bytes);
    if (ok && r.bias_count > 0)
      ok = pwrite_all(r.bias_offset, tensors[i].bias.data(), r.bias_count * 4);
  }
  ok = ok && ::fsync(fd) == 0;
  ::close(fd);
  if (!ok) {
    std::filesystem::remove(tmp, ec);
    return std::unexpected("failed writing " + tmp);
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return std::unexpected("cannot rename cache into place");
  }

  // Sidecar: content hash plus the cheap stat identity used by later opens.
  const std::string side_tmp = side.string() + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream out(side_tmp, std::ios::trunc);
    if (!out) return std::unexpected("cannot write the cache sidecar");
    out << "sha256 " << sha << "\n";
    for (const auto &s : shards) {
      const ShardId id = stat_shard(s);
      if (!id.valid) return std::unexpected("cannot stat " + s.string());
      out << "shard " << id.size << " " << id.dev << " " << id.ino << " "
          << id.mtime_ns << "\n";
    }
    out.flush();
    if (!out) return std::unexpected("cannot write the cache sidecar");
  }
  std::filesystem::rename(side_tmp, side, ec);
  if (ec) {
    std::filesystem::remove(side_tmp, ec);
    return std::unexpected("cannot rename the cache sidecar");
  }
  return path;
}

} // namespace sonicboom::model::weight_cache
