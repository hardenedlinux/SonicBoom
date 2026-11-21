// A full-context prefill must fit next to the decode runtimes the session pays
// for at construction. The failure this guards (pp32768 under an unset
// SONICBOOM_SPEC): the load-time verify-set warm-up claimed so much device
// memory at a 32k context that the prefill's own segments could no longer
// allocate, and the run died inside the first segment ("out of memory" under
// the embedding node) instead of prefilling.
//
// The check is the public flow: load Qwen2.5-7B, create a 32768-token session
// with the speculation mode left at its default (unset -> the startup probe
// resolves it), and prefill 32768 tokens. Nothing here is numeric; success is
// the guarantee. Skips (exit 77) without a CUDA device or the model GGUF.

#include <sonicboom/capi.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifndef SONICBOOM_SRC_DIR
#define SONICBOOM_SRC_DIR "."
#endif

namespace {

constexpr size_t kContext = 32768;

const char* const kSeedText =
    "Once upon a time, in a small village at the edge of a deep forest, there "
    "lived a careful engineer who measured everything twice and wrote down "
    "what she saw. Each morning she walked down to the river, counted the "
    "stones in the shallow water, and told the baker what the water would do "
    "before the afternoon rain arrived.";

std::string model_path(const char* env_name, const char* file) {
  if (const char* env = std::getenv(env_name); env && *env) return env;
  return std::string(SONICBOOM_SRC_DIR) + "/models/" + file;
}

}  // namespace

int main() {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    std::fprintf(stderr, "SKIP: no CUDA device\n");
    return 77;
  }
  const std::string path =
      model_path("SONICBOOM_QWEN_GGUF", "qwen2.5-7b-instruct-1m-q4_k_m.gguf");
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    std::fprintf(stderr, "SKIP: model absent (%s)\n", path.c_str());
    return 77;
  }
  // The regression is the default speculation policy: an unset
  // SONICBOOM_SPEC lets the probe resolve auto, whose envelope is what the
  // warm-up built. A caller's explicit mode must not change the check.
  ::unsetenv("SONICBOOM_SPEC");
  ::unsetenv("SONICBOOM_SPEC_MAX_DRAFTS");

  boom_error* err = nullptr;
  boom_llm_model* model = nullptr;
  if (boom_llm_model_load(path.c_str(), &model, &err) != BOOM_OK) {
    std::fprintf(stderr, "model_load: %s\n",
                 err ? boom_error_message(err) : "(none)");
    if (err) boom_error_free(err);
    return 1;
  }

  boom_token seed[256];
  size_t n_seed = 0;
  if (boom_llm_tokenize(model, kSeedText, std::strlen(kSeedText), seed, 256,
                       &n_seed, &err) != BOOM_OK ||
      n_seed == 0) {
    std::fprintf(stderr, "tokenize: %s\n",
                 err ? boom_error_message(err) : "empty prompt");
    if (err) boom_error_free(err);
    boom_llm_model_free(model);
    return 1;
  }
  std::vector<boom_token> prompt(kContext);
  for (size_t i = 0; i < kContext; ++i) prompt[i] = seed[i % n_seed];

  // One slot past the prompt, as a request needs: the prefill samples the
  // continuation token from the last prompt position.
  const uint64_t n_ctx = kContext + 1;
  boom_llm_session* session = nullptr;
  if (boom_llm_session_new(model, n_ctx, &session, &err) != BOOM_OK) {
    std::fprintf(stderr, "session_new: %s\n",
                 err ? boom_error_message(err) : "(none)");
    if (err) boom_error_free(err);
    boom_llm_model_free(model);
    return 1;
  }

  size_t reused = 0;
  const boom_status st =
      boom_llm_prefill(session, prompt.data(), prompt.size(), &reused, &err);
  if (st != BOOM_OK) {
    std::fprintf(stderr, "prefill(%zu): %s\n", kContext,
                 err ? boom_error_message(err) : "(none)");
    if (err) boom_error_free(err);
    boom_llm_session_free(session);
    boom_llm_model_free(model);
    return 1;
  }
  // `reused` is the prefix-reuse count, 0 on a fresh session; the guarantee is
  // that the whole prompt prefilled next to the session's warm runtimes.
  std::printf("prefill %zu tokens at n_ctx %llu OK (prefix reuse %zu)\n",
              kContext, static_cast<unsigned long long>(n_ctx), reused);
  boom_llm_session_free(session);
  boom_llm_model_free(model);
  return 0;
}
