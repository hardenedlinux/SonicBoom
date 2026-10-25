// The published inference guarantees, on qwen2.5-0.5b and Gemma 4 E4B through
// the stable C ABI, greedy, with the numerics-bearing choices pinned the way
// the golden tests pin them:
//
//   (1) determinism: the same prompt generated twice in fresh sessions gives
//       the same 32 tokens;
//   (2) prefix reuse: a full prefill of a 40-token prompt and the same prompt
//       prefilled as 24 + 16 tokens on one session give the same 32
//       continuation tokens;
//   (3) KV-capacity independence: n_ctx 1024 and n_ctx 4096 give the same 32
//       tokens;
//   (4) speculative decoding: SONICBOOM_SPEC=off and =on give the same 32
//       tokens.
//
// Each run is a forked child of this executable: the runtime feature switches
// (SONICBOOM_SPEC) are parsed once per process, and a child per mode keeps one
// model's device allocations from accumulating next to another run's. The child
// writes its token stream to a file; the parent waits and compares.
//
// Skips (exit 77) without a CUDA device or either model GGUF.

#include <sonicboom/capi.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef SONICBOOM_SRC_DIR
#define SONICBOOM_SRC_DIR "."
#endif

namespace {

constexpr uint32_t kMaxTokens = 32;
constexpr size_t kPromptTokens = 40;
constexpr size_t kReuseFirst = 24;

const char* const kPromptText =
    "Once upon a time, in a small village at the edge of a deep forest, there "
    "lived a careful engineer who measured everything twice and wrote down "
    "what she saw. Each morning she walked down to the river, counted the "
    "stones in the shallow water, and told the baker what the water would do "
    "before the afternoon rain arrived.";

int g_failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void die(const char* what, sb_error* err) {
  std::fprintf(stderr, "  %s: %s\n", what,
               err ? sb_error_message(err) : "(none)");
  if (err) sb_error_free(err);
  g_failures = 1;
}

void check_status(sb_status st, sb_error* err, const char* what) {
  if (st != SB_OK) die(what, err);
}

// The greedy sampling contract: temperature 0 is the historical argmax path.
sb_llm_sampling greedy() {
  sb_llm_sampling s;
  std::memset(&s, 0, sizeof s);
  s.struct_size = sizeof s;
  s.max_tokens = kMaxTokens;
  return s;
}

int collect_tokens(void* user, sb_token token) {
  auto* out = static_cast<std::vector<sb_token>*>(user);
  out->push_back(token);
  return 0;
}

std::string model_path(const char* env_name, const char* file) {
  if (const char* env = std::getenv(env_name); env && *env) return env;
  return std::string(SONICBOOM_SRC_DIR) + "/models/" + file;
}

// --- child side -------------------------------------------------------------

int child_main(const std::string& path, const std::string& mode,
               const std::string& out_path) {
  uint64_t n_ctx = 512;
  bool spec = false;
  if (mode == "ctx1024") {
    n_ctx = 1024;
  } else if (mode == "ctx4096") {
    n_ctx = 4096;
  } else if (mode == "spec_on") {
    spec = true;
  } else if (mode != "plain" && mode != "full" && mode != "reuse" &&
             mode != "spec_off") {
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
  }
  if (mode == "spec_on" || mode == "spec_off") {
    ::setenv("SONICBOOM_SPEC", spec ? "on" : "off", 1);
  }

  sb_error* err = nullptr;
  sb_llm_model* model = nullptr;
  if (sb_llm_model_load(path.c_str(), &model, &err) != SB_OK) {
    die("model_load", err);
    return 1;
  }

  sb_token prompt[1024];
  size_t n_prompt = 0;
  if (sb_llm_tokenize(model, kPromptText, std::strlen(kPromptText), prompt,
                      1024, &n_prompt, &err) != SB_OK) {
    die("tokenize", err);
    sb_llm_model_free(model);
    return 1;
  }
  if (n_prompt < kPromptTokens) {
    std::fprintf(stderr, "  prompt tokenized to %zu tokens, need %zu\n",
                 n_prompt, kPromptTokens);
    sb_llm_model_free(model);
    return 1;
  }

  sb_llm_session* session = nullptr;
  if (sb_llm_session_new(model, n_ctx, &session, &err) != SB_OK) {
    die("session_new", err);
    sb_llm_model_free(model);
    return 1;
  }

  uint64_t reused = 0;
  if (mode == "reuse") {
    // A prefill sets the session's prompt to tokens[0, n): the reuse case
    // prefills the first 24 tokens, then the full 40-token prompt, so the
    // second call processes only the 16 new tokens (the last prompt token is
    // always processed, hence at most 23 reused).
    if (sb_llm_prefill(session, prompt, kReuseFirst, &reused, &err) != SB_OK) {
      die("prefill (first 24)", err);
      sb_llm_session_free(session);
      sb_llm_model_free(model);
      return 1;
    }
    if (sb_llm_prefill(session, prompt, kPromptTokens, &reused, &err) != SB_OK) {
      die("prefill (rest)", err);
      sb_llm_session_free(session);
      sb_llm_model_free(model);
      return 1;
    }
    std::fprintf(stderr, "  reuse: %llu of %zu leading tokens reused\n",
                 static_cast<unsigned long long>(reused), kReuseFirst);
  } else {
    if (sb_llm_prefill(session, prompt, kPromptTokens, &reused, &err) != SB_OK) {
      die("prefill", err);
      sb_llm_session_free(session);
      sb_llm_model_free(model);
      return 1;
    }
  }

  std::vector<sb_token> tokens;
  const sb_llm_sampling s = greedy();
  uint32_t n_out = 0;
  if (sb_llm_decode(session, &s, collect_tokens, &tokens, &n_out, &err) !=
      SB_OK) {
    die("decode", err);
    sb_llm_session_free(session);
    sb_llm_model_free(model);
    return 1;
  }

  sb_llm_session_free(session);
  sb_llm_model_free(model);

  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  out.write(reinterpret_cast<const char*>(tokens.data()),
            static_cast<std::streamsize>(tokens.size() * sizeof(sb_token)));
  if (!out) {
    std::fprintf(stderr, "short write %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(stderr, "  %s: %u tokens\n", mode.c_str(), n_out);
  return 0;
}

// --- parent side ------------------------------------------------------------

std::string g_self;

std::vector<sb_token> run_child(const std::string& model, const std::string& mode,
                                bool* ok) {
  char tmpl[] = "/tmp/sb-guarantees-XXXXXX";
  const int fd = ::mkstemp(tmpl);
  *ok = false;
  if (fd < 0) {
    std::perror("mkstemp");
    return {};
  }
  ::close(fd);

  const pid_t pid = ::fork();
  if (pid < 0) {
    std::perror("fork");
    ::unlink(tmpl);
    return {};
  }
  if (pid == 0) {
    ::execl(g_self.c_str(), g_self.c_str(), "--child", model.c_str(),
            mode.c_str(), tmpl, static_cast<char*>(nullptr));
    std::perror("execl");
    ::_exit(127);
  }

  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) {
    std::perror("waitpid");
    ::unlink(tmpl);
    return {};
  }
  std::vector<sb_token> tokens;
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    std::ifstream in(tmpl, std::ios::binary);
    sb_token t = 0;
    while (in.read(reinterpret_cast<char*>(&t), sizeof t)) tokens.push_back(t);
    *ok = true;
  } else {
    std::fprintf(stderr, "  child (%s) failed (status %d)\n", mode.c_str(),
                 WIFEXITED(status) ? WEXITSTATUS(status) : -1);
  }
  ::unlink(tmpl);
  return tokens;
}

void compare(const std::string& label, const std::string& a_mode,
             const std::vector<sb_token>& a, const std::string& b_mode,
             const std::vector<sb_token>& b) {
  check(a.size() == kMaxTokens && b.size() == kMaxTokens,
        label + ": both runs produced " + std::to_string(kMaxTokens) +
            " tokens (" + a_mode + " " + std::to_string(a.size()) + ", " +
            b_mode + " " + std::to_string(b.size()) + ")");
  bool same = a.size() == b.size();
  size_t diff = 0;
  for (size_t i = 0; same && i < a.size(); ++i) {
    if (a[i] != b[i]) {
      same = false;
      diff = i;
    }
  }
  if (!same && a.size() == b.size()) {
    std::fprintf(stderr, "  first difference at token %zu: %d vs %d\n", diff,
                 int(a[diff]), int(b[diff]));
  }
  check(same, label + ": identical token stream (" + a_mode + " vs " + b_mode +
                  ")");
  if (same && a.size() == kMaxTokens) {
    std::printf("  %-22s %u tokens identical\n", label.c_str(),
                unsigned(a.size()));
  }
}

std::string tokens_text(const std::vector<sb_token>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ' ';
    s += std::to_string(v[i]);
  }
  return s;
}

void run_model(const char* env_name, const char* file, const char* label) {
  const std::string path = model_path(env_name, file);
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    std::fprintf(stderr, "  model not found: %s\n", path.c_str());
    ++g_failures;
    return;
  }
  std::printf("[%s]\n", label);

  bool ok = false;

  // (1) determinism: two fresh processes, each with a fresh session.
  const auto plain_a = run_child(path, "plain", &ok);
  check(ok, std::string(label) + ": determinism run 1");
  const auto plain_b = run_child(path, "plain", &ok);
  check(ok, std::string(label) + ": determinism run 2");
  compare("determinism", "run 1", plain_a, "run 2", plain_b);

  // (2) prefix reuse: 24 + 16 on one session vs a full prefill.
  const auto reuse = run_child(path, "reuse", &ok);
  check(ok, std::string(label) + ": prefix-reuse run");
  compare("prefix reuse", "full", plain_a, "24+16", reuse);

  // (3) KV-capacity independence.
  const auto small = run_child(path, "ctx1024", &ok);
  check(ok, std::string(label) + ": n_ctx 1024 run");
  const auto large = run_child(path, "ctx4096", &ok);
  check(ok, std::string(label) + ": n_ctx 4096 run");
  compare("KV capacity", "n_ctx 1024", small, "n_ctx 4096", large);

  // (4) speculative decoding must not change the stream.
  const auto spec_off = run_child(path, "spec_off", &ok);
  check(ok, std::string(label) + ": SONICBOOM_SPEC=off run");
  const auto spec_on = run_child(path, "spec_on", &ok);
  check(ok, std::string(label) + ": SONICBOOM_SPEC=on run");
  compare("speculative decode", "SONICBOOM_SPEC=off", spec_off,
          "SONICBOOM_SPEC=on", spec_on);

  // The reference stream stays visible for a failure report.
  std::printf("  reference          %s\n", tokens_text(plain_a).c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 5 && std::strcmp(argv[1], "--child") == 0)
    return child_main(argv[2], argv[3], argv[4]);

  // Pin the numerics-bearing choices the golden tests pin (overwrite=0: an
  // explicit environment wins).
  ::setenv("SONICBOOM_MMQ_DECOMP", "tiled", 0);
  ::setenv("SONICBOOM_ATTN_DECODE_CHUNK", "256", 0);

  char self[4096];
  const ssize_t n = ::readlink("/proc/self/exe", self, sizeof self - 1);
  if (n <= 0) {
    std::fprintf(stderr, "cannot resolve /proc/self/exe\n");
    return 1;
  }
  self[n] = '\0';
  g_self = self;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    std::fprintf(stderr, "SKIP: no CUDA device\n");
    return 77;
  }

  const std::string qwen =
      model_path("SONICBOOM_QWEN_GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf");
  const std::string gemma =
      model_path("SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf");
  std::error_code ec;
  const bool have_qwen = std::filesystem::exists(qwen, ec);
  const bool have_gemma = std::filesystem::exists(gemma, ec);
  if (!have_qwen || !have_gemma) {
    std::fprintf(stderr, "SKIP: model absent (%s%s%s)\n",
                 have_qwen ? "" : qwen.c_str(),
                 (!have_qwen && !have_gemma) ? ", " : "",
                 have_gemma ? "" : gemma.c_str());
    return 77;
  }

  run_model("SONICBOOM_QWEN_GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "qwen2.5-0.5b");
  run_model("SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf",
            "gemma-4-E4B");

  if (g_failures) {
    std::fprintf(stderr, "test_guarantees FAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("test_guarantees OK\n");
  return 0;
}
