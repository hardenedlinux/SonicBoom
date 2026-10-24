// Known-answer test for sb_image_write_png: a 2x2 RGB image must produce the
// exact 82-byte PNG below. The expected bytes were computed independently
// (Python: struct + zlib.crc32 + zlib.adler32), so this checks the signature,
// the IHDR fields, the stored-block deflate framing, every CRC and the
// Adler-32 rather than re-deriving them from the implementation.

#include <sonicboom/capi.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

void test_2x2_known_answer() {
  const std::vector<uint8_t> rgb = {
      255, 0,   0,   0,   255, 0,     // row 0: red, green
      0,   0,   255, 255, 255, 255,   // row 1: blue, white
  };
  const std::vector<uint8_t> expected = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,  // signature
      0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,  // IHDR, 13 bytes
      0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,  // 2 x 2
      0x08, 0x02, 0x00, 0x00, 0x00,                    // 8-bit RGB, no
      0xfd, 0xd4, 0x9a, 0x73,                          // interlace; CRC
      0x00, 0x00, 0x00, 0x19, 0x49, 0x44, 0x41, 0x54,  // IDAT, 25 bytes
      0x78, 0x01, 0x01, 0x0e, 0x00, 0xf1, 0xff,        // zlib + stored block
      0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00, 0x00,  // filter 0 + pixels
      0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
      0x1f, 0xee, 0x05, 0xfb,                          // Adler-32
      0xde, 0xdd, 0xec, 0x2b,                          // CRC
      0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,  // IEND, 0 bytes
      0xae, 0x42, 0x60, 0x82,                          // CRC
  };

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("sb-test-png-" + std::to_string(::getpid()) + ".png");
  sb_error* err = nullptr;
  const sb_status st =
      sb_image_write_png(path.c_str(), rgb.data(), 2, 2, &err);
  check(st == SB_OK, "sb_image_write_png returns SB_OK");
  if (st != SB_OK && err != nullptr) {
    std::cerr << "    error: " << sb_error_message(err) << "\n";
    sb_error_free(err);
  }
  std::ifstream in(path, std::ios::binary);
  const std::vector<uint8_t> got((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  std::error_code ec;
  std::filesystem::remove(path, ec);

  check(got.size() == expected.size(), "the PNG is 82 bytes");
  if (got.size() == expected.size()) {
    check(got == expected, "the PNG bytes match the known answer");
  } else {
    std::printf("    got %zu bytes, expected %zu\n", got.size(),
                expected.size());
  }
}

void test_rejects_null() {
  sb_error* err = nullptr;
  check(sb_image_write_png(nullptr, nullptr, 2, 2, &err) != SB_OK,
        "NULL path/rgb is rejected");
  if (err != nullptr) sb_error_free(err);
}

}  // namespace

int main() {
  test_2x2_known_answer();
  test_rejects_null();
  if (g_failures != 0) {
    std::cerr << g_failures << " PNG failure(s)\n";
    return 1;
  }
  std::cout << "test_png: all checks passed\n";
  return 0;
}
