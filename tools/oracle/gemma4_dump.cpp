// gemma4_dump — export named f32 intermediate tensors from a Gemma 4 decode.
//
// Dev-time reference/oracle only. This tool links against llama.cpp (not
// SonicBoom) and is used for differential validation of SonicBoom's native
// Gemma 4 block math. It is NOT part of libsonicboom.so and adds no llama.cpp
// dependency to the SonicBoom runtime.
//
// It decodes a single token at an arbitrary position and, via the public
// `cb_eval` scheduler callback (llama.h), dumps every tensor that llama.cpp's
// Gemma 4 graph builder named (ggml_set_name), in GGUF/f32 machine-readable
// form. Because a single token is decoded, attention collapses to `out == v`,
// but the Q/K/V norms, RoPE, FFN, and per-layer gate intermediates are all
// still computed and dumped — which is exactly what the block math needs to be
// validated against.
//
// Dump format (text, deterministic):
//   #  <comment>
//   @ <name> <dtype> <d0> <d1> <d2> <d3>
//   <v0>
//   <v1>
//   ... (d0*d1*d2*d3 float values, %.9g, one per line)
//
// Build: tools/oracle/build.sh   Run: tools/oracle/gemma4_dump -m <model> ...

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Read a GGML_TYPE_F32 tensor into a row-major vector of its nelements floats,
// honouring the tensor's byte strides (safe for views).
void read_f32(const ggml_tensor* t, std::vector<float>& out) {
  const int64_t n0 = t->ne[0], n1 = t->ne[1], n2 = t->ne[2], n3 = t->ne[3];
  out.resize(size_t(n0) * size_t(n1) * size_t(n2) * size_t(n3));
  const char* base = static_cast<const char*>(t->data);
  size_t idx = 0;
  for (int64_t i3 = 0; i3 < n3; ++i3)
    for (int64_t i2 = 0; i2 < n2; ++i2)
      for (int64_t i1 = 0; i1 < n1; ++i1)
        for (int64_t i0 = 0; i0 < n0; ++i0) {
          const size_t off = size_t(i0) * t->nb[0] + size_t(i1) * t->nb[1] +
                             size_t(i2) * t->nb[2] + size_t(i3) * t->nb[3];
          out[idx++] = *reinterpret_cast<const float*>(base + off);
        }
}

// Scheduler eval callback: dump every named f32 tensor once it is computed.
bool dump_cb(ggml_tensor* t, bool ask, void* user_data) {
  FILE* fp = static_cast<FILE*>(user_data);
  if (ask) {
    // "needed" == named f32 tensor we want dumped (others are batched + skipped).
    return t->name[0] != '\0' && t->type == GGML_TYPE_F32;
  }
  std::vector<float> vals;
  read_f32(t, vals);
  fprintf(fp, "@ %s %s %lld %lld %lld %lld\n", t->name, ggml_type_name(t->type),
          (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
          (long long)t->ne[3]);
  for (float v : vals) fprintf(fp, "%.9g\n", v);
  return true;  // continue evaluating
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* out_path = "gemma4_oracle.dump";
  int token = 0;
  int pos = 0;
  int n_ctx = 512;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-m")) {
      model_path = argv[++i];
    } else if (!strcmp(argv[i], "-o")) {
      out_path = argv[++i];
    } else if (!strcmp(argv[i], "--token")) {
      token = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--pos")) {
      pos = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--ctx")) {
      n_ctx = atoi(argv[++i]);
    }
  }
  if (!model_path) {
    fprintf(stderr,
            "usage: %s -m <model.gguf> [-o out] [--token id] [--pos p] [--ctx n]\n",
            argv[0]);
    return 2;
  }
  if (pos >= n_ctx) {
    fprintf(stderr, "pos (%d) must be < ctx (%d)\n", pos, n_ctx);
    return 2;
  }

  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;  // CPU only
  llama_model* model = llama_model_load_from_file(model_path, mparams);
  if (!model) {
    fprintf(stderr, "failed to load model: %s\n", model_path);
    return 1;
  }

  FILE* fp = fopen(out_path, "w");
  if (!fp) {
    fprintf(stderr, "failed to open output: %s\n", out_path);
    return 1;
  }
  fprintf(fp, "# gemma4 oracle dump: model=%s token=%d pos=%d\n", model_path,
          token, pos);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = n_ctx;
  cparams.cb_eval = dump_cb;
  cparams.cb_eval_user_data = fp;

  llama_context* ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    fprintf(stderr, "failed to create context\n");
    fclose(fp);
    return 1;
  }

  // Single token at `pos` (the position controls RoPE; n_kv stays 1 so the
  // attention softmax is over one logit).
  llama_batch batch = llama_batch_init(1, 0, 1);
  batch.n_tokens = 1;
  batch.token[0] = token;
  batch.pos[0] = pos;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = 0;
  batch.logits[0] = 1;

  const int rc = llama_decode(ctx, batch);
  if (rc != 0) {
    fprintf(stderr, "decode failed: rc=%d\n", rc);
    llama_batch_free(batch);
    fclose(fp);
    return 1;
  }

  llama_batch_free(batch);
  fclose(fp);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
