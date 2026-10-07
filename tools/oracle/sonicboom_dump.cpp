// sonicboom_dump — export SonicBoom's native Gemma 4 block intermediates in the
// same machine-readable format as tools/oracle/gemma4_dump.cpp (the llama.cpp
// oracle), for differential comparison via tools/oracle/compare_dumps.
//
// This links against libsonicboom.so (the SHARED core) and uses only public
// Layer 2 headers (<sonicboom/...>). It computes the token embedding and a
// single block's forward pass, invoking the run_block_traced dev-time hook to
// emit each named intermediate with llama.cpp's cb() name convention.
//
// The dump format matches gemma4_dump exactly (see that file's header):
//   #  <comment>
//   @ <name> <dtype> <d0> <d1> <d2> <d3>
//   <v0> ... <vn-1>   (%.9g, one per line)
//
// Build: tools/oracle/build_sonicboom.sh   Run: sonicboom_dump -m <model> ...
//
// Phase 5D: `--all` dumps the full 42-block forward (chained residual stream +
// Gemma 4 shared-KV) via forward_traced, emitting every block's intermediates.
// Without `--all` it dumps a single block in isolation (layer 0 default), which
// is only correct for layer 0 (residual input == token embedding, has_kv true).

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/exec.h>
#include <sonicboom/model/loader.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;

namespace {

FILE* g_fp = nullptr;

void emit(const char* name, std::span<const float> v) {
  std::fprintf(g_fp, "@ %s f32 %zu 1 1 1\n", name, v.size());
  for (float x : v) std::fprintf(g_fp, "%.9g\n", x);
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* out_path = "sonicboom.dump";
  int token = 0;
  int pos = 0;
  int layer = 0;
  bool all_layers = false;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-m")) {
      model_path = argv[++i];
    } else if (!std::strcmp(argv[i], "-o")) {
      out_path = argv[++i];
    } else if (!std::strcmp(argv[i], "--token")) {
      token = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--pos")) {
      pos = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--layer")) {
      layer = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--all")) {
      all_layers = true;
    }
  }
  if (!model_path) {
    std::fprintf(stderr,
                 "usage: %s -m <model.gguf> [-o out] [--token id] [--pos p] [--layer l] [--all]\n",
                 argv[0]);
    return 2;
  }

  auto r = sbgguf::Reader::load(model_path);
  if (!r) {
    std::fprintf(stderr, "failed to load GGUF: %s\n", model_path);
    return 1;
  }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) {
    std::fprintf(stderr, "load_gemma4 error: %s\n", m.error().c_str());
    return 1;
  }

  g_fp = std::fopen(out_path, "w");
  if (!g_fp) {
    std::fprintf(stderr, "failed to open output: %s\n", out_path);
    return 1;
  }
  std::fprintf(g_fp, "# sonicboom gemma4 dump: model=%s token=%d pos=%d layer=%d all=%d\n",
               model_path, token, pos, layer, all_layers);

  const uint32_t d = m->config.embedding_length;

  // Token embedding, scaled by sqrt(n_embd) — llama.cpp's "inp_scaled" (global,
  // no layer suffix).
  std::vector<float> inpL(d);
  if (!sbmodel::embed_token(*m, uint64_t(token), inpL)) {
    std::fprintf(stderr, "embed_token failed\n");
    std::fclose(g_fp);
    return 1;
  }
  emit("inp_scaled", inpL);

  // Per-layer embedding for every layer, in llama.cpp's "inp_per_layer" layout
  // [n_embd_per_layer, n_layer] (layer l's slice at offset l*per, element j at
  // l*per + j). This is the tensor the per-layer gate multiplies with, and it
  // carries the bf16 projection + q4_K token-embd path — a common source of a
  // small-but-amplified residual-stream discrepancy.
  if (all_layers) {
    const uint32_t per = m->config.per_layer_input;
    std::vector<float> ple(uint64_t(m->config.block_count) * per);
    for (uint32_t l = 0; l < m->config.block_count; ++l) {
      std::vector<float> one(per);
      if (!sbmodel::embed_per_layer(*m, l, uint64_t(token), one, false)) {
        std::fprintf(stderr, "embed_per_layer failed (layer %u)\n", l);
        std::fclose(g_fp);
        return 1;
      }
      std::copy(one.begin(), one.end(), ple.begin() + uint64_t(l) * per);
    }
    emit("inp_per_layer", ple);
  }

  std::vector<float> out(d);
  if (all_layers) {
    if (!sbmodel::forward_traced(
            *m, uint64_t(token), uint64_t(pos), out,
            [](const char* name, std::span<const float> v) { emit(name, v); })) {
      std::fprintf(stderr, "forward_traced failed\n");
      std::fclose(g_fp);
      return 1;
    }
  } else {
    if (!sbmodel::run_block_traced(
            *m, uint32_t(layer), uint64_t(token), uint64_t(pos), out,
            [](const char* name, std::span<const float> v) { emit(name, v); })) {
      std::fprintf(stderr, "run_block_traced failed (layer %d)\n", layer);
      std::fclose(g_fp);
      return 1;
    }
  }

  std::fclose(g_fp);
  return 0;
}
