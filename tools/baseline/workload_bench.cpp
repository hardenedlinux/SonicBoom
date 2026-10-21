// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// workload_bench — the interactive traces the canonical metric set does not
// cover. tg64/d512/pp512/reuse measure one request shape each; a session's cost
// is a *sequence* of requests, and what a session spends between them is what
// its planner rebuilds and its KV cache re-prefills.
//
//   (A) agent    a multi-turn session: a shared ~1000-token system prompt, then
//                eight user turns of 57/312/140/33/401/96/210/75 tokens and 128
//                generated tokens each, with the prefix cache on — every turn
//                is a prefix reuse of everything before it, so its cost is one
//                suffix prefill plus a short decode. Reported per turn: TTFT,
//                decode rate, total. `--fresh` runs the same trace with a fresh
//                session per turn (the whole history re-prefilled); the ratio
//                of the two is what prefix reuse is worth at this depth.
//
//   (B) coding   one 2048-token prompt and a 256-token generation: TTFT (the
//                prompt prefill) and the decode rate that follows at that
//                depth — the trace pp512 does not reach, since prefill cost is
//                not linear in prompt length past the sliding-window ring and
//                the decode at depth is not the decode at depth 0.
//
//   (C) codeedit six independent edit requests, each pasting one 150-400 line
//                repo source file and asking for a small edit that returns the
//                whole file; a turn generates the pasted file's own token count.
//                Copy-heavy by construction, so it is the arm prompt lookup is
//                aimed at; each turn is a fresh session (a stateless request API
//                re-sends the file), so nothing is reused across turns.
//
//   (D) chat     eight short unrelated user turns and 96 generated tokens each —
//                free-form text with little repetition in the answer, the arm
//                where the drafter rarely fires and speculation must be free.
//
// Every trace reports its emitted-token FNV-1a hash, so the spec on/off arms —
// two processes, since SONICBOOM_SPEC is parsed once — can be checked
// bit-identical.
//
// The text is fixed and model-side: one in-binary corpus, tokenized with the
// loaded model's own tokenizer (vocab/tokenize), sliced to the trace's token
// counts. Same text for both models, each model's own ids — the point of a
// fixed trace is that its *lengths* are reproducible, and a real tokenizer is
// what makes the ids the kernels see the ones a session would produce.
//
// Per turn, TTFT is one call that prefills the turn's suffix and decodes a
// single token; the decode phase is a second call whose prefix is that token,
// so it prefills one token and decodes `--gen` of them (greedy, so the stream
// is the same as if the turn had been one call). Both calls extend the session
// monotonically, which is what keeps the prefix cache honest.
//
// Usage: workload_bench --workload agent|coding|codeedit|chat
//          [--turns T] [--turn-tokens L1,L2,...] [--system S] [--gen G]
//          [--prompt P] [--n-ctx N] [--reps R] [--fresh] [--text PATH]
// Build: tools/baseline/build_workload_bench.sh

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/vocab/vocab.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;
namespace sbvocab = sonicboom::vocab;

// The fixed corpus: prose, no markup, nothing a tokenizer treats specially.
// Repeated `kCorpusRepeats` times it is comfortably longer than the longest
// trace (1000 system + 1324 user + 128 gen for agent, 2048 + 256 for coding),
// and the tool fails loudly if a model's tokenizer cuts it shorter than that.
static const char* kCorpus = R"TEXT(
A runtime that plans its work before it does it can afford to be explicit about
what it plans. The planner turns a model description into a sequence of steps,
each step naming one operator, the buffers it reads and writes, and the kernels
it launches; the executor then walks that sequence without asking the model or
the plan any further questions. Keeping those two phases apart is what makes a
plan worth recording: it can be read, diffed, counted, and compared against the
plan of a later build of the same runtime, and a change in behaviour shows up as
a change in the record rather than as a number that moved.

The same separation is what makes a plan worth caching. Building a plan costs
real time — the operators have to be lowered, their buffers sized, their launch
geometries chosen — and a session that asks one question after another keeps
asking for plans of the same shape. A cache keyed on that shape answers most of
those requests with a lookup, and the work that remains is the work the request
actually asked for: the tokens it added, the rows it appended, the blocks it
read.

Memory is the other half of the story. A plan allocates buffers for the values
that live between operators, and the size of those buffers is a property of the
shapes in the plan, not of the arithmetic inside the kernels. A buffer that is
declared wider than the values that fill it costs address space and, on a device
with a fixed budget, the difference between a working set that fits and one that
does not. Sizing it from the values that actually flow through it is the same
kind of decision as choosing a kernel: it is a decision about the whole plan,
made once, and best made where the whole plan is visible.

Attention is where a long context is paid for. Every token attends to the tokens
before it, and the cost of that grows with the number of positions, so a session
that reaches depth carries a cost its first request never showed. A sliding
window bounds the work per layer by making the oldest positions fall out of the
window as the newest arrive; the ring that holds the window is then a fixed-size
buffer written in place, and the positions that fall out are gone rather than
kept. What remains is the part that cannot be bounded away: the current chunk of
positions, which is processed against everything the window still holds.

Quantization is the third lever. Weights stored in four or three bits per value
buy a working set that fits in cache and in device memory, and the arithmetic
that consumes them is built around the layout those bits are stored in: blocks
of values sharing one scale, unpacked as they are read, multiplied against
activations that were quantized the same way so the inner product can be done a
half register at a time. The alternative — unpacking to floating point and
letting a general matrix kernel do the work — is simpler and often slower, and
which of the two wins depends on the shape of the request, not on the size of
the model.

None of this is specific to one model. The operators are ordinary: a projection,
a normalization, a rotation of positions, a product of two halves, a sum with
what came before. What differs between models is how those are arranged and how
often each is needed, and that is exactly the part a planner can measure and a
record can show. A description that names the operators and their shapes is
portable in a way a hand-written forward pass is not, and it leaves the choice
of kernels where it belongs: below, where the hardware is known, and open to
being changed without changing the description above it.
)TEXT";
static constexpr int kCorpusRepeats = 4;

// The agent trace's shape, from the trace the harness is specified against.
static const uint64_t kTurnTokens[] = {57, 312, 140, 33, 401, 96, 210, 75};
static constexpr uint64_t kTurnCount = sizeof(kTurnTokens) / sizeof(kTurnTokens[0]);
static constexpr uint64_t kSystemTokens = 1000;
static constexpr uint64_t kAgentGen = 128;
static constexpr uint64_t kCodingPrompt = 2048;
static constexpr uint64_t kCodingGen = 256;

// (C) the code-edit trace: six turns, each pasting one repo source file (150 to
// 400 lines, checked when it is read) and asking for a small edit that returns
// the whole file. The generation length of a turn is the pasted file's own
// token count, so the drafter has the source text to copy — the trace is the
// arm where prompt lookup should pay. Each turn is its own session: an edit
// request is stateless, so the file is re-sent rather than kept in context.
static const char* kEditFiles[] = {
    "core/src/planner/cost_model.cpp",
    "core/src/planner/partition.cpp",
    "core/src/planner/device_pool.cpp",
    "core/src/planner/transfer_scheduler.cpp",
    "core/src/planner/runtime.cpp",
    "core/src/autograd/autograd.cpp",
};
static constexpr std::size_t kEditTurns =
    sizeof(kEditFiles) / sizeof(kEditFiles[0]);
static const char* kEditInstructions[] = {
    "Apply this edit to the file below and return the complete file: rename "
    "the first local variable to `outcome` and leave every other line "
    "unchanged.\n\n",
    "Apply this edit to the file below and return the complete file: add a "
    "bounds check before the first indexed write and leave every other line "
    "unchanged.\n\n",
    "Apply this edit to the file below and return the complete file: replace "
    "the first early return with an error status and leave every other line "
    "unchanged.\n\n",
    "Apply this edit to the file below and return the complete file: hoist the "
    "first loop bound into a local constant and leave every other line "
    "unchanged.\n\n",
    "Apply this edit to the file below and return the complete file: change "
    "the first log message's level to warning and leave every other line "
    "unchanged.\n\n",
    "Apply this edit to the file below and return the complete file: make the "
    "first helper static and leave every other line unchanged.\n\n",
};

// (D) the free-form chat trace: eight short unrelated user turns and 96
// generated tokens each, with nothing in the prompt repeated in the answer —
// the arm where the drafter rarely fires, so speculation must not cost
// anything. The user turns are fixed token slices of the corpus.
static const uint64_t kChatTurnTokens[] = {32, 48, 40, 24, 56, 36, 44, 28};
static constexpr uint64_t kChatTurns =
    sizeof(kChatTurnTokens) / sizeof(kChatTurnTokens[0]);
static constexpr uint64_t kChatGen = 96;

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count() * 1000.0;
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// FNV-1a over the emitted token ids. SONICBOOM_SPEC is parsed once per process,
// so the off and auto arms are two processes: comparing their stream hashes is
// how every trace run is checked bit-identical.
static constexpr uint64_t kHashInit = 1469598103934665603ull;
static uint64_t hash_token(uint64_t h, uint64_t token) {
  h ^= token;
  return h * 1099511628211ull;
}

// The corpus, tokenized once with the model's own tokenizer and repeated until
// it is longer than any trace asks for.
static std::vector<uint64_t> corpus_tokens(const sbvocab::Vocab& vocab,
                                           const std::string& text,
                                           uint64_t need) {
  std::vector<uint64_t> ids;
  for (int rep = 0; rep < kCorpusRepeats && ids.size() < need; ++rep) {
    // add_special prepends <bos> for the first piece only, as a session would.
    const std::vector<int32_t> t =
        sbvocab::tokenize(vocab, text, /*parse_special=*/false,
                          /*add_special=*/rep == 0);
    for (int32_t id : t) ids.push_back(static_cast<uint64_t>(id));
  }
  return ids;
}

struct Turn {
  double ttft_ms = 0.0;   // suffix prefill + first token
  double decode_ms = 0.0; // `gen` further tokens
  double tok_s = 0.0;     // decode rate of the above
  uint64_t user_tokens = 0;
  uint64_t reused = 0;    // prefix tokens the session already held
};

// One call of the spine: prompt `hist` at position 0, `out` greedy tokens from
// whatever session state has been built. Returns the tokens produced, 0 on
// failure.
static uint64_t call(const sbmodel::Gemma4Model& m,
                     const std::vector<uint64_t>& hist, uint64_t n_ctx,
                     sbplanner::KvCacheSession& session,
                     std::vector<int64_t>& out, uint64_t* reused) {
  std::fill(out.begin(), out.end(), -1);
  return sbmodel::generate_prefill_spine_cuda_prefix(
      m, hist, 0, out, session, /*use_prefill=*/true, reused);
}

// (A) the multi-turn trace. `fresh` gives every turn its own session, which is
// the same prompt sequence with nothing reused.
static bool agent_trace(const sbmodel::Gemma4Model& m,
                        const std::vector<uint64_t>& ids, uint64_t system,
                        const std::vector<uint64_t>& turns, uint64_t gen,
                        uint64_t n_ctx, bool fresh, std::vector<Turn>& out,
                        double* trace_ms, uint64_t* hash = nullptr) {
  std::vector<uint64_t> hist(ids.begin(), ids.begin() + system);
  std::size_t cursor = system;
  out.clear();
  double total = 0.0;
  uint64_t h = kHashInit;

  sbplanner::KvCacheSession session;
  session.n_ctx = n_ctx;
  // A carried token: the first call of a turn produces one, the second starts
  // its decode from it. Kept out of `hist` until both calls have run.
  std::vector<int64_t> one(1), gen_out(gen);

  for (std::size_t t = 0; t < turns.size(); ++t) {
    if (fresh) {
      session.caches.reset();
      session.filled_tokens.clear();
    }
    if (cursor + turns[t] > ids.size()) {
      fprintf(stderr, "agent trace: corpus too short for turn %zu\n", t + 1);
      return false;
    }
    hist.insert(hist.end(), ids.begin() + cursor, ids.begin() + cursor + turns[t]);
    cursor += turns[t];
    if (hist.size() + gen + 1 > n_ctx) {
      fprintf(stderr, "agent trace: turn %zu exceeds n_ctx\n", t + 1);
      return false;
    }

    Turn turn;
    turn.user_tokens = turns[t];
    uint64_t reused = 0;
    const Clock::time_point a0 = Clock::now();
    const uint64_t first = call(m, hist, n_ctx, session, one, &reused);
    turn.ttft_ms = ms_since(a0);
    turn.reused = reused;
    if (first == 0) {
      fprintf(stderr, "agent trace: turn %zu prefill failed\n", t + 1);
      return false;
    }
    // The decode phase starts from the token the prefill produced: same history
    // plus that one token, so the session only grows (never rewinds) and the
    // turn's cost is measured in the two phases a session actually sees.
    hist.push_back(static_cast<uint64_t>(one[0]));
    h = hash_token(h, static_cast<uint64_t>(one[0]));
    const Clock::time_point b0 = Clock::now();
    const uint64_t done = call(m, hist, n_ctx, session, gen_out, nullptr);
    turn.decode_ms = ms_since(b0);
    if (done == 0) {
      fprintf(stderr, "agent trace: turn %zu decode failed\n", t + 1);
      return false;
    }
    turn.tok_s = double(done) * 1000.0 / turn.decode_ms;
    // The history the next turn extends is what the turn generated: the carried
    // token plus the rest of the decode's stream.
    for (uint64_t i = 0; i < done; ++i) {
      h = hash_token(h, static_cast<uint64_t>(gen_out[i]));
      if (i + 1 < done) hist.push_back(static_cast<uint64_t>(gen_out[i]));
    }
    out.push_back(turn);
    total += turn.ttft_ms + turn.decode_ms;
  }
  *trace_ms = total;
  if (hash) *hash = h;
  return true;
}

// One code-edit turn: the instruction and pasted file, already tokenized with
// the model's tokenizer, and the generation length (the file's token count).
struct EditTurn {
  std::vector<uint64_t> user;
  uint64_t gen = 0;
};

// (C) the code-edit trace: six independent edit requests, each pasting one
// repo file into a fresh session and generating the file's own token count
// back, the way a stateless request API re-sends the file it wants edited.
// Same two-call turn structure as agent_trace (TTFT prefills the paste and
// decodes one token; the decode call starts from that token).
static bool codeedit_trace(const sbmodel::Gemma4Model& m,
                           const std::vector<EditTurn>& turns, uint64_t n_ctx,
                           std::vector<Turn>& out, double* trace_ms,
                           uint64_t* hash) {
  std::vector<int64_t> one(1), gen_out;
  out.clear();
  uint64_t h = kHashInit;
  double total = 0.0;

  for (std::size_t t = 0; t < turns.size(); ++t) {
    std::vector<uint64_t> hist = turns[t].user;
    // Each edit is its own request: the file is re-sent, so the session does not
    // carry the previous turn's paste or generation.
    sbplanner::KvCacheSession session;
    session.n_ctx = n_ctx;
    if (hist.size() + turns[t].gen + 1 > n_ctx) {
      fprintf(stderr, "code-edit trace: turn %zu exceeds n_ctx (%zu > %llu)\n",
              t + 1, hist.size() + turns[t].gen + 1,
              (unsigned long long)n_ctx);
      return false;
    }
    gen_out.assign(turns[t].gen, -1);
    Turn turn;
    turn.user_tokens = turns[t].user.size();
    uint64_t reused = 0;
    const Clock::time_point a0 = Clock::now();
    const uint64_t first = call(m, hist, n_ctx, session, one, &reused);
    turn.ttft_ms = ms_since(a0);
    turn.reused = reused;
    if (first == 0) {
      fprintf(stderr, "code-edit trace: turn %zu prefill failed\n", t + 1);
      return false;
    }
    hist.push_back(static_cast<uint64_t>(one[0]));
    h = hash_token(h, static_cast<uint64_t>(one[0]));
    const Clock::time_point b0 = Clock::now();
    const uint64_t done = call(m, hist, n_ctx, session, gen_out, nullptr);
    turn.decode_ms = ms_since(b0);
    if (done == 0) {
      fprintf(stderr, "code-edit trace: turn %zu decode failed\n", t + 1);
      return false;
    }
    turn.tok_s = double(done) * 1000.0 / turn.decode_ms;
    for (uint64_t i = 0; i < done; ++i) {
      h = hash_token(h, static_cast<uint64_t>(gen_out[i]));
      if (i + 1 < done) hist.push_back(static_cast<uint64_t>(gen_out[i]));
    }
    out.push_back(turn);
    total += turn.ttft_ms + turn.decode_ms;
  }
  *trace_ms = total;
  *hash = h;
  return true;
}

// (D) the free-form chat trace: short unrelated user turns (corpus slices at
// an offset that moves faster than the slices advance) and a fixed generation,
// a persistent session. Same two-call turn structure as agent_trace.
static bool chat_trace(const sbmodel::Gemma4Model& m,
                       const std::vector<uint64_t>& ids,
                       const std::vector<uint64_t>& turns, uint64_t gen,
                       uint64_t n_ctx, std::vector<Turn>& out,
                       double* trace_ms, uint64_t* hash) {
  std::vector<uint64_t> hist;
  std::size_t cursor = 0;
  std::vector<int64_t> one(1), gen_out(gen);
  sbplanner::KvCacheSession session;
  session.n_ctx = n_ctx;
  out.clear();
  uint64_t h = kHashInit;
  double total = 0.0;

  for (std::size_t t = 0; t < turns.size(); ++t) {
    if (cursor + turns[t] > ids.size()) {
      fprintf(stderr, "chat trace: corpus too short for turn %zu\n", t + 1);
      return false;
    }
    hist.insert(hist.end(), ids.begin() + cursor, ids.begin() + cursor + turns[t]);
    // The next message is a different part of the corpus, not the continuation.
    cursor += turns[t] + 137;
    if (hist.size() + gen + 1 > n_ctx) {
      fprintf(stderr, "chat trace: turn %zu exceeds n_ctx\n", t + 1);
      return false;
    }
    std::fill(gen_out.begin(), gen_out.end(), -1);
    Turn turn;
    turn.user_tokens = turns[t];
    uint64_t reused = 0;
    const Clock::time_point a0 = Clock::now();
    const uint64_t first = call(m, hist, n_ctx, session, one, &reused);
    turn.ttft_ms = ms_since(a0);
    turn.reused = reused;
    if (first == 0) {
      fprintf(stderr, "chat trace: turn %zu prefill failed\n", t + 1);
      return false;
    }
    hist.push_back(static_cast<uint64_t>(one[0]));
    h = hash_token(h, static_cast<uint64_t>(one[0]));
    const Clock::time_point b0 = Clock::now();
    const uint64_t done = call(m, hist, n_ctx, session, gen_out, nullptr);
    turn.decode_ms = ms_since(b0);
    if (done == 0) {
      fprintf(stderr, "chat trace: turn %zu decode failed\n", t + 1);
      return false;
    }
    turn.tok_s = double(done) * 1000.0 / turn.decode_ms;
    for (uint64_t i = 0; i < done; ++i) {
      h = hash_token(h, static_cast<uint64_t>(gen_out[i]));
      if (i + 1 < done) hist.push_back(static_cast<uint64_t>(gen_out[i]));
    }
    out.push_back(turn);
    total += turn.ttft_ms + turn.decode_ms;
  }
  *trace_ms = total;
  *hash = h;
  return true;
}

// The speculation counters after a trace, for the RESULT line. They accumulate
// per process across all calls, so the caller resets them before the measured
// run; zero means the plain loop ran.
static void print_turn_table(const std::vector<Turn>& turns) {
  printf("  turn  user  reused      ttft_ms    decode_tok_s   turn_ms\n");
  for (std::size_t t = 0; t < turns.size(); ++t) {
    printf("  %4zu %5llu %7llu %12.1f %15.1f %9.1f\n", t + 1,
           (unsigned long long)turns[t].user_tokens,
           (unsigned long long)turns[t].reused, turns[t].ttft_ms,
           turns[t].tok_s, turns[t].ttft_ms + turns[t].decode_ms);
  }
}

static void result_spec_fields() {
  const sbmodel::SpecCounters c = sbmodel::spec_counters();
  printf("spec_iter=%llu spec_verify=%llu spec_plain=%llu spec_drafts=%llu "
         "spec_accepted=%llu spec_emitted=%llu",
         (unsigned long long)c.iterations, (unsigned long long)c.verify_steps,
         (unsigned long long)c.plain_steps, (unsigned long long)c.draft_tokens,
         (unsigned long long)c.accepted_tokens,
         (unsigned long long)c.emitted_tokens);
}

int main(int argc, char** argv) {
  std::string workload = "agent";
  std::string text_path;
  uint64_t system = kSystemTokens, gen = 0, prompt = kCodingPrompt;
  uint64_t n_ctx = 0, reps = 1;
  bool fresh = false;
  // --warmup: pay the decode-runtime construction at load (the
  // sb_llm_session_new behaviour) before the trace; --no-warm: skip the
  // in-tool warm-up trace so the first measured turnover is genuinely cold.
  bool warmup = false, no_warm = false;
  std::vector<uint64_t> turn_tokens(kTurnTokens, kTurnTokens + kTurnCount);

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto next = [&]() -> const char* { return argv[++i]; };
    if (!strcmp(a, "--workload")) workload = next();
    else if (!strcmp(a, "--turn-tokens")) {
      turn_tokens.clear();
      std::istringstream ss(next());
      std::string item;
      while (std::getline(ss, item, ',')) turn_tokens.push_back(std::stoull(item));
    } else if (!strcmp(a, "--turns")) {
      const uint64_t want = std::stoull(next());
      if (want > turn_tokens.size()) {
        fprintf(stderr, "--turns %llu exceeds the %zu turn lengths given\n",
                (unsigned long long)want, turn_tokens.size());
        return 2;
      }
      turn_tokens.resize(want);
    } else if (!strcmp(a, "--system")) system = std::stoull(next());
    else if (!strcmp(a, "--gen")) gen = std::stoull(next());
    else if (!strcmp(a, "--prompt")) prompt = std::stoull(next());
    else if (!strcmp(a, "--n-ctx")) n_ctx = std::stoull(next());
    else if (!strcmp(a, "--reps")) reps = std::stoull(next());
    else if (!strcmp(a, "--fresh")) fresh = true;
    else if (!strcmp(a, "--text")) text_path = next();
    else if (!strcmp(a, "--warmup")) warmup = true;
    else if (!strcmp(a, "--no-warm")) no_warm = true;
    else {
      fprintf(stderr,
              "usage: workload_bench --workload agent|coding|codeedit|chat "
              "[--turns T] [--turn-tokens L1,L2,...] [--system S] [--gen G] "
              "[--prompt P] [--n-ctx N] [--reps R] [--fresh] [--text PATH] "
              "[--warmup] [--no-warm]\n");
      return 2;
    }
  }
  if (workload != "agent" && workload != "coding" &&
      workload != "codeedit" && workload != "chat") {
    fprintf(stderr, "unknown workload: %s\n", workload.c_str());
    return 2;
  }
  if (gen == 0) {
    if (workload == "agent") gen = kAgentGen;
    else if (workload == "coding") gen = kCodingGen;
    else if (workload == "chat") gen = kChatGen;
    // codeedit: 0 means each turn generates the pasted file's own token count.
  }

  // The repo root: --workload codeedit pastes real source files from it.
  const char* root_env = std::getenv("SONICBOOM_ROOT");
  const std::filesystem::path repo_root(root_env ? root_env : ".");

  std::string path;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF")) path = env;
  if (path.empty()) {
    const char* root = std::getenv("SONICBOOM_ROOT");
    std::string base = root ? std::string(root) : std::string(".");
    path = base + "/models/gemma-4-E4B-it-Q3_K_M.gguf";
  }
  if (!std::filesystem::exists(path)) {
    fprintf(stderr, "model not found: %s\n", path.c_str());
    return 2;
  }
  auto r = sbgguf::Reader::load(path);
  if (!r) { fprintf(stderr, "failed to load GGUF\n"); return 1; }
  // Arch-dispatched: the model struct and every spine entry below serve both
  // architectures this runtime models.
  auto m = sbmodel::load_gemma4(*r);
  if (!m) { fprintf(stderr, "model load error: %s\n", m.error().c_str()); return 1; }
  auto vocab = sbvocab::load_vocab(*r);
  if (!vocab) { fprintf(stderr, "vocab load error: %s\n", vocab.error().c_str()); return 1; }

  std::string text;
  if (text_path.empty()) {
    for (int i = 0; i < kCorpusRepeats; ++i) text += kCorpus;
  } else {
    std::ifstream in(text_path);
    if (!in) { fprintf(stderr, "text not found: %s\n", text_path.c_str()); return 2; }
    std::ostringstream ss;
    ss << in.rdbuf();
    text = ss.str();
  }

  const uint64_t agent_need = system + std::accumulate(turn_tokens.begin(),
                                                       turn_tokens.end(), uint64_t(0));
  const uint64_t chat_need =
      std::accumulate(kChatTurnTokens, kChatTurnTokens + kChatTurns,
                      uint64_t(0)) +
      kChatTurns * 137 + kChatGen + 1;
  const uint64_t need =
      std::max(std::max(agent_need, prompt), chat_need) + gen + 1;
  std::vector<uint64_t> ids = corpus_tokens(*vocab, text, need);
  if (ids.size() < need) {
    fprintf(stderr,
            "corpus tokenizes to %zu ids, the trace needs %llu: pass --text "
            "with a longer file\n",
            ids.size(), (unsigned long long)need);
    return 2;
  }

  // (C) the code-edit turns: read each repo file, check its size, tokenize the
  // instruction plus the file, and set the turn's generation to the file's own
  // token count (the whole file back), capped by --gen when given.
  std::vector<EditTurn> edits;
  if (workload == "codeedit") {
    for (std::size_t i = 0; i < kEditTurns; ++i) {
      const std::filesystem::path file = repo_root / kEditFiles[i];
      std::ifstream in(file);
      if (!in) {
        fprintf(stderr, "code-edit trace: cannot read %s\n", file.c_str());
        return 2;
      }
      std::ostringstream ss;
      ss << in.rdbuf();
      const std::string source = ss.str();
      const uint64_t lines = std::count(source.begin(), source.end(), '\n') +
                             (!source.empty() && source.back() != '\n' ? 1 : 0);
      if (lines < 150 || lines > 400) {
        fprintf(stderr,
                "code-edit trace: %s has %llu lines (the trace uses files of "
                "150-400 lines)\n",
                file.c_str(), (unsigned long long)lines);
        return 2;
      }
      const std::vector<int32_t> file_ids =
          sbvocab::tokenize(*vocab, source, /*parse_special=*/false,
                            /*add_special=*/false);
      if (file_ids.empty()) {
        fprintf(stderr, "code-edit trace: %s tokenized to nothing\n",
                file.c_str());
        return 2;
      }
      const std::string prompt_text = std::string(kEditInstructions[i]) + source;
      const std::vector<int32_t> user_ids =
          sbvocab::tokenize(*vocab, prompt_text, /*parse_special=*/false,
                            /*add_special=*/i == 0);
      EditTurn turn;
      turn.user.assign(user_ids.begin(), user_ids.end());
      turn.gen = gen ? std::min<uint64_t>(gen, file_ids.size())
                     : uint64_t(file_ids.size());
      edits.push_back(std::move(turn));
    }
  }
  const std::vector<uint64_t> chat_turns(kChatTurnTokens,
                                         kChatTurnTokens + kChatTurns);

  const bool ok = [&]() -> bool {
    if (workload == "agent") {
      // Every token the trace ever holds: the system prompt, every user turn,
      // and every turn's generation (the carried token plus the decode's).
      const uint64_t history = agent_need + turn_tokens.size() * gen;
      if (n_ctx == 0) n_ctx = history + gen + 1;
      if (warmup) {
        const Clock::time_point w0 = Clock::now();
        const bool ok = sbmodel::warmup_spine_cuda(*m, n_ctx);
        printf("[warmup] %s %.1f ms at load (n_ctx %llu)\n",
               ok ? "built" : "failed", ms_since(w0),
               (unsigned long long)n_ctx);
      }
      std::vector<Turn> warm;
      double warm_ms = 0.0;
      // Warm up the plan/capture caches on the exact trace shape; the measured
      // runs then see a session starting from the same state every time.
      // --no-warm skips it so the first measured turn is genuinely cold.
      if (!no_warm &&
          !agent_trace(*m, ids, system, turn_tokens, gen, n_ctx, fresh, warm,
                       &warm_ms))
        return false;

      std::vector<Turn> best;
      double best_ms = 0.0;
      uint64_t best_hash = kHashInit;
      sbmodel::spec_counters_reset();
      for (uint64_t rep = 0; rep < reps; ++rep) {
        std::vector<Turn> turns;
        double trace_ms = 0.0;
        uint64_t hash = kHashInit;
        if (!agent_trace(*m, ids, system, turn_tokens, gen, n_ctx, fresh, turns,
                         &trace_ms, &hash))
          return false;
        if (best.empty() || trace_ms < best_ms) {
          best = turns;
          best_ms = trace_ms;
          best_hash = hash;
        }
      }

      printf("agent trace%s: system %llu, %zu turns, gen %llu, n_ctx %llu\n",
             fresh ? " (fresh session per turn)" : " (prefix reuse)",
             (unsigned long long)system, turn_tokens.size(),
             (unsigned long long)gen, (unsigned long long)n_ctx);
      printf("  turn  user  reused      ttft_ms    decode_tok_s   turn_ms\n");
      for (std::size_t t = 0; t < best.size(); ++t) {
        printf("  %4zu %5llu %7llu %12.1f %15.1f %9.1f\n", t + 1,
               (unsigned long long)best[t].user_tokens,
               (unsigned long long)best[t].reused, best[t].ttft_ms,
               best[t].tok_s, best[t].ttft_ms + best[t].decode_ms);
      }
      std::vector<double> ttfts, rates, turnms;
      for (const Turn& t : best) {
        ttfts.push_back(t.ttft_ms);
        rates.push_back(t.tok_s);
        turnms.push_back(t.ttft_ms + t.decode_ms);
      }
      printf("  trace wall: %.1f ms over %zu turns, stream hash %016llx\n",
             best_ms, best.size(), (unsigned long long)best_hash);
      printf("RESULT workload_agent%s turns=%zu gen=%llu system=%llu ttft_ms=%.3f "
             "decode_tok_s=%.3f turn_ms=%.3f trace_ms=%.3f tokens_hash=%016llx ",
             fresh ? "_fresh" : "", best.size(), (unsigned long long)gen,
             (unsigned long long)system, median(ttfts), median(rates),
             median(turnms), best_ms, (unsigned long long)best_hash);
      result_spec_fields();
      printf("\n");
      return true;
    }

    // (C) the code-edit trace: six turns, each pasting one repo file and
    // asking for the whole file back after a small edit.
    if (workload == "codeedit") {
      if (n_ctx == 0) {
        uint64_t per_turn = 0;
        for (const EditTurn& t : edits)
          per_turn = std::max(per_turn, t.user.size() + t.gen + 1);
        n_ctx = per_turn + 2;
      }
      if (warmup) {
        const Clock::time_point w0 = Clock::now();
        const bool ok = sbmodel::warmup_spine_cuda(*m, n_ctx);
        printf("[warmup] %s %.1f ms at load (n_ctx %llu)\n",
               ok ? "built" : "failed", ms_since(w0),
               (unsigned long long)n_ctx);
      }
      if (!no_warm) {
        // Warm the plan/capture caches (and, with speculation, the draft-length
        // cost cache) on the first turn, generation shortened to 64 tokens.
        std::vector<EditTurn> warm_turn(1, edits[0]);
        warm_turn[0].gen = std::min<uint64_t>(edits[0].gen, 64);
        std::vector<Turn> warm;
        double warm_ms = 0.0;
        uint64_t warm_hash = kHashInit;
        if (!codeedit_trace(*m, warm_turn, n_ctx, warm, &warm_ms, &warm_hash))
          return false;
      }
      std::vector<Turn> best;
      double best_ms = 0.0;
      uint64_t best_hash = kHashInit;
      sbmodel::spec_counters_reset();
      for (uint64_t rep = 0; rep < reps; ++rep) {
        std::vector<Turn> turns;
        double trace_ms = 0.0;
        uint64_t hash = kHashInit;
        if (!codeedit_trace(*m, edits, n_ctx, turns, &trace_ms, &hash))
          return false;
        if (best.empty() || trace_ms < best_ms) {
          best = turns;
          best_ms = trace_ms;
          best_hash = hash;
        }
      }
      uint64_t gen_total = 0;
      for (const EditTurn& t : edits) gen_total += t.gen;
      printf("code-edit trace: %zu turns, %llu pasted tokens, %llu generated, "
             "n_ctx %llu\n",
             edits.size(),
             (unsigned long long)std::accumulate(
                 edits.begin(), edits.end(), uint64_t(0),
                 [](uint64_t a, const EditTurn& t) {
                   return a + t.user.size();
                 }),
             (unsigned long long)gen_total, (unsigned long long)n_ctx);
      print_turn_table(best);
      std::vector<double> ttfts, rates, turnms;
      for (const Turn& t : best) {
        ttfts.push_back(t.ttft_ms);
        rates.push_back(t.tok_s);
        turnms.push_back(t.ttft_ms + t.decode_ms);
      }
      printf("  trace wall: %.1f ms, stream hash %016llx\n", best_ms,
             (unsigned long long)best_hash);
      printf("RESULT workload_codeedit turns=%zu gen=%llu ttft_ms=%.3f "
             "decode_tok_s=%.3f turn_ms=%.3f trace_ms=%.3f tokens_hash=%016llx ",
             best.size(), (unsigned long long)gen_total, median(ttfts),
             median(rates), median(turnms), best_ms,
             (unsigned long long)best_hash);
      result_spec_fields();
      printf("\n");
      return true;
    }

    // (D) the free-form chat trace: short unrelated turns, a fixed generation.
    if (workload == "chat") {
      if (n_ctx == 0) {
        const uint64_t history =
            std::accumulate(chat_turns.begin(), chat_turns.end(), uint64_t(0)) +
            chat_turns.size() * 137;
        n_ctx = history + gen + 1;
      }
      if (warmup) {
        const Clock::time_point w0 = Clock::now();
        const bool ok = sbmodel::warmup_spine_cuda(*m, n_ctx);
        printf("[warmup] %s %.1f ms at load (n_ctx %llu)\n",
               ok ? "built" : "failed", ms_since(w0),
               (unsigned long long)n_ctx);
      }
      std::vector<Turn> warm;
      double warm_ms = 0.0;
      uint64_t warm_hash = kHashInit;
      if (!no_warm && !chat_trace(*m, ids, chat_turns, gen, n_ctx, warm,
                                  &warm_ms, &warm_hash))
        return false;
      std::vector<Turn> best;
      double best_ms = 0.0;
      uint64_t best_hash = kHashInit;
      sbmodel::spec_counters_reset();
      for (uint64_t rep = 0; rep < reps; ++rep) {
        std::vector<Turn> turns;
        double trace_ms = 0.0;
        uint64_t hash = kHashInit;
        if (!chat_trace(*m, ids, chat_turns, gen, n_ctx, turns, &trace_ms,
                        &hash))
          return false;
        if (best.empty() || trace_ms < best_ms) {
          best = turns;
          best_ms = trace_ms;
          best_hash = hash;
        }
      }
      printf("chat trace: %zu turns, gen %llu, n_ctx %llu\n", chat_turns.size(),
             (unsigned long long)gen, (unsigned long long)n_ctx);
      print_turn_table(best);
      std::vector<double> ttfts, rates, turnms;
      for (const Turn& t : best) {
        ttfts.push_back(t.ttft_ms);
        rates.push_back(t.tok_s);
        turnms.push_back(t.ttft_ms + t.decode_ms);
      }
      printf("  trace wall: %.1f ms, stream hash %016llx\n", best_ms,
             (unsigned long long)best_hash);
      printf("RESULT workload_chat turns=%zu gen=%llu ttft_ms=%.3f "
             "decode_tok_s=%.3f turn_ms=%.3f trace_ms=%.3f tokens_hash=%016llx ",
             best.size(), (unsigned long long)gen, median(ttfts), median(rates),
             median(turnms), best_ms, (unsigned long long)best_hash);
      result_spec_fields();
      printf("\n");
      return true;
    }

    // (B) one long prompt and a long generation, the same two phases as a turn.
    if (n_ctx == 0) n_ctx = prompt + gen + 1;
    if (warmup) {
      const Clock::time_point w0 = Clock::now();
      const bool ok = sbmodel::warmup_spine_cuda(*m, n_ctx);
      printf("[warmup] %s %.1f ms at load (n_ctx %llu)\n",
             ok ? "built" : "failed", ms_since(w0),
             (unsigned long long)n_ctx);
    }
    std::vector<uint64_t> hist(ids.begin(), ids.begin() + prompt);
    if (!no_warm) {
      std::vector<int64_t> warm(1);
      sbplanner::KvCacheSession session;
      session.n_ctx = n_ctx;
      if (call(*m, hist, n_ctx, session, warm, nullptr) == 0) {
        fprintf(stderr, "coding warm-up failed\n");
        return false;
      }
    }
    double best_ttft = 1e18, best_decode = 1e18;
    uint64_t done = 0;
    uint64_t best_hash = kHashInit;
    sbmodel::spec_counters_reset();
    for (uint64_t rep = 0; rep < reps; ++rep) {
      sbplanner::KvCacheSession session;
      session.n_ctx = n_ctx;
      std::vector<int64_t> one(1), out(gen);
      const Clock::time_point a0 = Clock::now();
      if (call(*m, hist, n_ctx, session, one, nullptr) == 0) {
        fprintf(stderr, "coding ttft failed\n");
        return false;
      }
      const double ttft = ms_since(a0);
      hist.push_back(static_cast<uint64_t>(one[0]));
      const Clock::time_point b0 = Clock::now();
      done = call(*m, hist, n_ctx, session, out, nullptr);
      const double decode = ms_since(b0);
      hist.pop_back();
      if (done == 0) {
        fprintf(stderr, "coding decode failed\n");
        return false;
      }
      if (ttft + decode < best_ttft + best_decode) {
        best_ttft = ttft;
        best_decode = decode;
        uint64_t hash = hash_token(kHashInit, static_cast<uint64_t>(one[0]));
        for (uint64_t i = 0; i < done; ++i)
          hash = hash_token(hash, static_cast<uint64_t>(out[i]));
        best_hash = hash;
      }
    }
    const double rate = double(done) * 1000.0 / best_decode;
    printf("coding trace: prompt %llu, gen %llu, n_ctx %llu\n",
           (unsigned long long)prompt, (unsigned long long)gen,
           (unsigned long long)n_ctx);
    printf("  time to first token: %.1f ms\n", best_ttft);
    printf("  decode after it     : %.1f tok/s (%llu tokens in %.1f ms)\n",
           rate, (unsigned long long)done, best_decode);
    printf("RESULT workload_coding prompt=%llu gen=%llu ttft_ms=%.3f "
           "decode_tok_s=%.3f total_ms=%.3f tokens_hash=%016llx ",
           (unsigned long long)prompt, (unsigned long long)gen, best_ttft, rate,
           best_ttft + best_decode, (unsigned long long)best_hash);
    result_spec_fields();
    printf("\n");
    return true;
  }();

  return ok ? 0 : 1;
}
