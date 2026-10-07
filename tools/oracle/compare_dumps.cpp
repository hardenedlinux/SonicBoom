// compare_dumps — differential comparison of two SonicBoom/oracle tensor dumps.
//
// Dev-time reference tool only. Reads two dumps in the format written by
// gemma4_dump (and by SonicBoom's dump tool):
//
//   #  <comment>
//   @ <name> <dtype> <d0> <d1> <d2> <d3>
//   <v0>
//   <v1>
//   ... (d0*d1*d2*d3 float values, %.9g, one per line)
//
// and reports, per common tensor name (in reference dump order — which is the
// computation order), the element count, max absolute error, max relative
// error, and RMS error; then flags the first "mismatching stage" (the first
// reference tensor whose error exceeds the tolerance), and lists names present
// in only one of the two dumps.
//
// Relative error is |a-b| / max(|a|, |b|, 1e-12), so it degrades gracefully to
// absolute error at/around zero.
//
// Build: tools/oracle/build_compare.sh   Run: compare_dumps <ref> <cand> [tol]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Tensor {
  std::vector<float> v;
};

bool read_dump(const char* path, std::vector<std::pair<std::string, Tensor>>& out) {
  FILE* fp = std::fopen(path, "r");
  if (!fp) {
    std::fprintf(stderr, "compare_dumps: cannot open %s\n", path);
    return false;
  }
  char line[8192];
  std::string cur_name;
  std::vector<float> cur;
  size_t idx = 0;
  bool in_tensor = false;
  while (std::fgets(line, sizeof line, fp)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    if (line[0] == '@') {
      if (in_tensor) {
        out.emplace_back(cur_name, Tensor{cur});
        cur.clear();
      }
      // @ <name> <dtype> <d0> <d1> <d2> <d3>, where <name> may contain spaces
      // (e.g. "Qcur-0 (reshaped)"). Tokenize; the last 5 tokens are dtype + 4
      // dims; everything between "@" and those is the name.
      std::vector<std::string> toks;
      char* save = nullptr;
      for (char* p = strtok_r(line + 1, " \t\r\n", &save); p;
           p = strtok_r(nullptr, " \t\r\n", &save))
        toks.emplace_back(p);
      if (toks.size() < 6) {
        std::fprintf(stderr, "compare_dumps: bad header in %s: %s", path, line);
        std::fclose(fp);
        return false;
      }
      const long long d0 = std::atoll(toks[toks.size() - 4].c_str());
      const long long d1 = std::atoll(toks[toks.size() - 3].c_str());
      const long long d2 = std::atoll(toks[toks.size() - 2].c_str());
      const long long d3 = std::atoll(toks[toks.size() - 1].c_str());
      std::string name = toks[0];
      for (size_t i = 1; i + 5 < toks.size(); ++i) name += " " + toks[i];
      cur_name = name;
      const size_t n = size_t(d0) * size_t(d1) * size_t(d2) * size_t(d3);
      cur.assign(n, 0.0f);
      idx = 0;
      in_tensor = true;
    } else if (in_tensor) {
      // value line (one float per tensor; sequential fill)
      if (idx < cur.size()) cur[idx++] = std::strtof(line, nullptr);
    }
  }
  if (in_tensor) out.emplace_back(cur_name, Tensor{cur});
  std::fclose(fp);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <ref.dump> <cand.dump> [tol]\n", argv[0]);
    return 2;
  }
  const double tol = argc >= 4 ? std::atof(argv[3]) : 1e-4;

  std::vector<std::pair<std::string, Tensor>> ref, cand;
  if (!read_dump(argv[1], ref)) return 1;
  if (!read_dump(argv[2], cand)) return 1;

  // Index each dump by name, keeping occurrence order so duplicate names (the
  // ggml auto-names " (view)", " (reshaped)", ...) match positionally.
  std::map<std::string, std::vector<size_t>> ref_idx, cand_idx;
  for (size_t i = 0; i < ref.size(); ++i) ref_idx[ref[i].first].push_back(i);
  for (size_t i = 0; i < cand.size(); ++i) cand_idx[cand[i].first].push_back(i);

  std::vector<std::string> only_cand;
  std::string first_mismatch;
  int n_common = 0, n_mismatch = 0;
  size_t n_only_ref = 0;
  double worst_rms = 0.0;

  std::printf("reference: %s (%zu tensors, %zu unique names)\n", argv[1], ref.size(),
              ref_idx.size());
  std::printf("candidate: %s (%zu tensors, %zu unique names)\n", argv[2], cand.size(),
              cand_idx.size());
  std::printf("tolerance: %g\n\n", tol);
  std::printf("%-28s %10s %14s %14s %14s\n", "tensor", "elems", "max_abs", "max_rel",
              "rms");

  // Iterate in candidate order (the checkpoint emission order). For the k-th
  // occurrence of each name, compare against the k-th occurrence in reference.
  std::map<std::string, size_t> cand_occ;
  for (const auto& [name, ct] : cand) {
    const size_t k = cand_occ[name]++;
    auto it = ref_idx.find(name);
    if (it == ref_idx.end()) {
      only_cand.push_back(name);
      continue;
    }
    if (k >= it->second.size()) {
      std::printf("%-28s OCCURRENCE MISMATCH ref has %zu\n", name.c_str(),
                  it->second.size());
      ++n_mismatch;
      if (first_mismatch.empty()) first_mismatch = name;
      continue;
    }
    const Tensor& rt = ref[it->second[k]].second;
    if (rt.v.size() != ct.v.size()) {
      std::printf("%-28s COUNT MISMATCH ref=%zu cand=%zu\n", name.c_str(),
                  rt.v.size(), ct.v.size());
      ++n_mismatch;
      if (first_mismatch.empty()) first_mismatch = name;
      continue;
    }
    ++n_common;
    double max_abs = 0.0, max_rel = 0.0, rms = 0.0;
    for (size_t i = 0; i < rt.v.size(); ++i) {
      const double a = rt.v[i], b = ct.v[i];
      const double d = std::fabs(a - b);
      if (d > max_abs) max_abs = d;
      const double rel = d / std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1e-12);
      if (rel > max_rel) max_rel = rel;
      rms += d * d;
    }
    rms = std::sqrt(rms / double(rt.v.size()));
    if (rms > worst_rms) worst_rms = rms;
    const bool bad = max_abs > tol || max_rel > tol;
    if (bad) {
      ++n_mismatch;
      if (first_mismatch.empty()) first_mismatch = name;
    }
    std::printf("%-28s %10zu %14.6g %14.6g %14.6g%s\n", name.c_str(), rt.v.size(),
                max_abs, max_rel, rms, bad ? "  <-- MISMATCH" : "");
  }

  // Names present in reference but never in candidate (expected to be large:
  // the oracle dump contains far more tensors than the checkpoints we check).
  for (const auto& [name, idxs] : ref_idx)
    if (cand_idx.find(name) == cand_idx.end()) n_only_ref += idxs.size();

  std::printf("\n--- summary ---\n");
  std::printf("common tensors: %d\n", n_common);
  std::printf("mismatching:    %d\n", n_mismatch);
  std::printf("worst rms:      %.6g\n", worst_rms);
  std::printf("first mismatch: %s\n",
              first_mismatch.empty() ? "(none)" : first_mismatch.c_str());
  std::printf("only in reference: %zu tensors\n", n_only_ref);
  std::printf("only in candidate: %zu\n", only_cand.size());
  for (const auto& n : only_cand) std::printf("    %s\n", n.c_str());

  return n_mismatch == 0 ? 0 : 1;
}
