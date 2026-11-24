// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// output-decay: Output Decay between two directories of per-prompt outputs.
//
//   output-decay <reference-dir> <compared-dir>
//
// The reference directory is the engine's acceleration-off outputs, the
// compared directory its acceleration-on outputs (or, for a run-to-run OD,
// two fresh processes with the same settings). Each prompt is one regular file
// in both directories, paired by name. The tool reads the files as bytes and
// prints one tab-separated line per prompt (reference bytes, common-prefix
// bytes, OD %, first divergent byte offset) and an overall line, followed by a
// one-line RESULT summary for scripts.
//
// Exit status: 0 on success, 1 on a directory/read error, 2 on usage.

#include "output_decay.h"

#include <cstdio>
#include <exception>
#include <string>

namespace od = sonicboom::tools::od;

namespace {

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s <reference-dir> <compared-dir>\n"
               "\n"
               "Reference: the engine's acceleration-off outputs; compared: its\n"
               "acceleration-on outputs (or a second same-setting run for a\n"
               "run-to-run OD). One regular file per prompt, paired by file name.\n",
               argv0);
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    usage(argv[0]);
    return 2;
  }
  if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    usage(argv[0]);
    return 0;
  }

  od::Summary summary;
  try {
    summary = od::compare_directories(argv[1], argv[2]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "output-decay: %s\n", e.what());
    return 1;
  }

  std::printf("prompt\tref_bytes\tprefix_bytes\tod%%\tfirst_divergence\n");
  for (const od::PromptResult& p : summary.prompts) {
    std::printf("%s\t%llu\t%llu\t%.4f%%\t", p.name.c_str(),
                static_cast<unsigned long long>(p.reference_bytes),
                static_cast<unsigned long long>(p.prefix_bytes),
                p.od_percent);
    if (p.identical)
      std::printf("none\n");
    else
      std::printf("%llu\n",
                  static_cast<unsigned long long>(p.first_divergence));
  }
  std::printf("overall\t-\t-\t%.4f%%\t-\n", summary.od_mean_percent);
  std::printf("RESULT prompts=%zu od_mean_percent=%.4f "
              "od_pooled_percent=%.4f\n",
              summary.prompts.size(), summary.od_mean_percent,
              summary.od_pooled_percent);
  return 0;
}
