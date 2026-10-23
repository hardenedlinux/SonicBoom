// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Constrained decoding: GBNF -> NFA -> per-position token mask, plus the JSON
// Schema -> GBNF compiler. See core/include/sonicboom/vocab/grammar.h.
//
// Structure:
//   1. GBNF parser (rules -> AST)
//   2. NFA builder. A rule reference becomes a call edge, so a configuration
//      carries a return-address stack (a pushdown automaton); recursive
//      grammars such as JSON's value/object/array work without inlining.
//   3. The grammar state: a canonical set of configurations. Transitions are
//      memoized per (set, byte), and a set's token mask is computed once by
//      walking the vocabulary trie only through allowed bytes.
//   4. JSON Schema -> GBNF text.

#include <sonicboom/vocab/grammar.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml_derived/bpe_tokenizer.h"
#include "ggml_derived/unicode_byte_map.h"

namespace sonicboom::vocab {
namespace {

// --- token text ------------------------------------------------------------

// The bytes `token` contributes to the generated text. This is the per-token
// piece detokenize() applies (▁ -> space, byte tokens expanded, byte-level
// decode for GPT2-style vocabularies), with control/unknown/unused tokens
// contributing nothing.
std::string token_piece(const Vocab& v, int32_t token) {
  if (token < 0 || static_cast<size_t>(token) >= v.tokens.size() ||
      static_cast<size_t>(token) >= v.type.size())
    return {};
  switch (v.type[static_cast<size_t>(token)]) {
    case TokenType::Control:
    case TokenType::Unknown:
    case TokenType::Unused:
      return {};
    case TokenType::Byte:
      return std::string(1, static_cast<char>(
          ggml_bpe::token_to_byte(v.tokens[static_cast<size_t>(token)])));
    case TokenType::UserDefined:
      return v.tokens[static_cast<size_t>(token)];
    case TokenType::Normal:
      break;
  }
  std::string piece = v.tokens[static_cast<size_t>(token)];
  if (v.escape_whitespaces) {
    ggml_bpe::unescape_whitespace(piece);
  } else {
    // The stored spelling is byte-encoded; a piece outside the byte table has
    // no text form (detokenize prints a placeholder), so it never matches.
    std::string decoded;
    if (!ggml_unicode::byte_decode(piece, decoded)) return {};
    piece = std::move(decoded);
  }
  return piece;
}

// --- GBNF AST --------------------------------------------------------------

struct Ast;
using AstPtr = std::shared_ptr<Ast>;

struct Ast {
  enum class Kind { Empty, Lit, Class, Seq, Alt, Star, Plus, Opt, Rep, Ref };
  Kind kind = Kind::Empty;
  std::string lit;                 // Lit bytes; Ref rule name
  std::array<bool, 256> cls{};     // Class byte set
  std::vector<AstPtr> kids;        // Seq / Alt
  AstPtr kid;                      // Star / Plus / Opt / Rep
  int min = 0;
  int max = -1;                    // -1 = unbounded

  static AstPtr make(Kind k) {
    auto a = std::make_shared<Ast>();
    a->kind = k;
    return a;
  }
};

int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

struct Rule {
  std::string name;
  AstPtr body;
};

// --- GBNF parser -----------------------------------------------------------

class GbnfParser {
public:
  explicit GbnfParser(std::string_view src) : s_(src) {}

  std::expected<std::vector<Rule>, std::string> parse() {
    std::vector<Rule> rules;
    while (true) {
      skip();
      if (eof()) break;
      const size_t rule_at = i_;
      std::string name;
      if (!ident(name)) return fail("rule name");
      skip();
      if (s_.substr(i_, 3) != "::=") return fail("'::=' after rule name");
      i_ += 3;
      AstPtr body = parse_alt();
      if (!body) return std::unexpected(err_.empty() ? "rule body" : err_);
      rules.push_back(Rule{std::move(name), std::move(body)});
      (void)rule_at;
    }
    if (rules.empty()) return std::unexpected("grammar has no rules");
    return rules;
  }

private:
  std::string_view s_;
  size_t i_ = 0;
  std::string err_;

  std::unexpected<std::string> fail(const std::string& what) {
    if (err_.empty())
      err_ = what + " expected at offset " + std::to_string(i_);
    return std::unexpected(err_);
  }

  bool eof() const { return i_ >= s_.size(); }
  char peek() const { return eof() ? '\0' : s_[i_]; }

  void skip() {
    while (!eof()) {
      const char c = s_[i_];
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        ++i_;
      } else if (c == '#') {
        while (!eof() && s_[i_] != '\n') ++i_;
      } else {
        break;
      }
    }
  }

  bool scan_ident(size_t& p, std::string& out) const {
    auto is_start = [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    auto is_cont = [&](char c) {
      return is_start(c) || (c >= '0' && c <= '9') || c == '-';
    };
    if (p >= s_.size() || !is_start(s_[p])) return false;
    const size_t b = p;
    while (p < s_.size() && is_cont(s_[p])) ++p;
    out.assign(s_.substr(b, p - b));
    return true;
  }

  bool ident(std::string& out) {
    std::string tmp;
    if (!scan_ident(i_, tmp)) return false;
    out = std::move(tmp);
    return true;
  }

  bool lookahead_rule_start() const {
    size_t p = i_;
    while (p < s_.size()) {
      const char c = s_[p];
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '#') {
        if (c == '#')
          while (p < s_.size() && s_[p] != '\n') ++p;
        else
          ++p;
      } else {
        break;
      }
    }
    std::string name;
    if (!scan_ident(p, name)) return false;
    while (p < s_.size() && (s_[p] == ' ' || s_[p] == '\t' || s_[p] == '\r' ||
                             s_[p] == '\n'))
      ++p;
    return s_.substr(p, 3) == "::=";
  }

  static bool parse_escape(std::string& out, std::string_view s, size_t& p) {
    // Caller consumed the backslash.
    if (p >= s.size()) return false;
    const char c = s[p++];
    switch (c) {
      case 'n': out.push_back('\n'); return true;
      case 't': out.push_back('\t'); return true;
      case 'r': out.push_back('\r'); return true;
      case 'b': out.push_back('\b'); return true;
      case 'f': out.push_back('\f'); return true;
      case '0': out.push_back('\0'); return true;
      case '"': out.push_back('"'); return true;
      case '\'': out.push_back('\''); return true;
      case '\\': out.push_back('\\'); return true;
      case '/': out.push_back('/'); return true;
      case 'x': {
        if (p + 1 >= s.size()) return false;
        const int h = hex_digit(s[p]), l = hex_digit(s[p + 1]);
        if (h < 0 || l < 0) return false;
        out.push_back(static_cast<char>((h << 4) | l));
        p += 2;
        return true;
      }
      case 'u': {
        if (p + 3 >= s.size()) return false;
        uint32_t cp = 0;
        for (int k = 0; k < 4; ++k) {
          const int h = hex_digit(s[p + size_t(k)]);
          if (h < 0) return false;
          cp = (cp << 4) | uint32_t(h);
        }
        p += 4;
        utf8_append(out, cp);
        return true;
      }
      default:
        out.push_back(c);
        return true;
    }
  }

  AstPtr parse_alt() {
    AstPtr first = parse_seq();
    if (!first) return nullptr;
    std::vector<AstPtr> kids;
    kids.push_back(std::move(first));
    while (true) {
      skip();
      if (peek() != '|') break;
      ++i_;
      AstPtr next = parse_seq();
      if (!next) return nullptr;
      kids.push_back(std::move(next));
    }
    if (kids.size() == 1) return kids.front();
    auto a = Ast::make(Ast::Kind::Alt);
    a->kids = std::move(kids);
    return a;
  }

  AstPtr parse_seq() {
    std::vector<AstPtr> kids;
    while (true) {
      skip();
      if (eof() || peek() == '|' || peek() == ')') break;
      if (lookahead_rule_start()) break;
      AstPtr item = parse_item();
      if (!item) return nullptr;
      kids.push_back(std::move(item));
    }
    if (kids.size() == 1) return kids.front();
    auto a = Ast::make(Ast::Kind::Seq);
    a->kids = std::move(kids);
    return a;
  }

  AstPtr parse_item() {
    AstPtr p = parse_primary();
    if (!p) return nullptr;
    while (true) {
      skip();
      const char c = peek();
      if (c == '*') {
        ++i_;
        auto a = Ast::make(Ast::Kind::Star);
        a->kid = std::move(p);
        p = std::move(a);
      } else if (c == '+') {
        ++i_;
        auto a = Ast::make(Ast::Kind::Plus);
        a->kid = std::move(p);
        p = std::move(a);
      } else if (c == '?') {
        ++i_;
        auto a = Ast::make(Ast::Kind::Opt);
        a->kid = std::move(p);
        p = std::move(a);
      } else if (c == '{') {
        const size_t save = i_;
        ++i_;
        int lo = -1, hi = -1;
        if (!parse_uint(lo)) {
          i_ = save;  // not a repetition; let the caller see '{'
          break;
        }
        if (peek() == ',') {
          ++i_;
          if (peek() != '}') {
            if (!parse_uint(hi)) {
              i_ = save;
              break;
            }
          }
        } else {
          hi = lo;
        }
        if (peek() != '}') {
          i_ = save;
          break;
        }
        ++i_;
        if (lo < 0 || (hi >= 0 && hi < lo) || lo > 4096 ||
            (hi > 4096)) {
          err_ = "bad repetition bounds at offset " + std::to_string(save);
          return nullptr;
        }
        auto a = Ast::make(Ast::Kind::Rep);
        a->kid = std::move(p);
        a->min = lo;
        a->max = hi;
        p = std::move(a);
      } else {
        break;
      }
    }
    return p;
  }

  bool parse_uint(int& out) {
    if (peek() < '0' || peek() > '9') return false;
    long v = 0;
    while (!eof() && s_[i_] >= '0' && s_[i_] <= '9') {
      v = v * 10 + (s_[i_] - '0');
      ++i_;
      if (v > 100000) return false;
    }
    out = static_cast<int>(v);
    return true;
  }

  AstPtr parse_primary() {
    skip();
    const char c = peek();
    if (c == '\0') {
      fail("item");
      return nullptr;
    }
    if (c == '(') {
      ++i_;
      AstPtr alt = parse_alt();
      if (!alt) return nullptr;
      skip();
      if (peek() != ')') {
        fail("')'");
        return nullptr;
      }
      ++i_;
      return alt;
    }
    if (c == '.') {
      ++i_;
      auto a = Ast::make(Ast::Kind::Class);
      for (int b = 0; b < 256; ++b) a->cls[size_t(b)] = (b != '\n');
      return a;
    }
    if (c == '"') {
      ++i_;
      std::string lit;
      while (true) {
        if (eof()) {
          fail("closing '\"'");
          return nullptr;
        }
        const char ch = s_[i_++];
        if (ch == '"') break;
        if (ch == '\\') {
          if (!parse_escape(lit, s_, i_)) {
            fail("string escape");
            return nullptr;
          }
        } else {
          lit.push_back(ch);
        }
      }
      auto a = Ast::make(Ast::Kind::Lit);
      a->lit = std::move(lit);
      return a;
    }
    if (c == '[') return parse_class();
    std::string name;
    if (ident(name)) {
      auto a = Ast::make(Ast::Kind::Ref);
      a->lit = std::move(name);
      return a;
    }
    fail("item");
    return nullptr;
  }

  // One class atom: an escaped byte, a UTF-8 lead byte's raw bytes, or a plain
  // byte. Ranges are expanded by the caller.
  bool class_atom(std::vector<uint8_t>& bytes) {
    if (eof()) return false;
    const char c = s_[i_++];
    if (c == '\\') {
      std::string tmp;
      const size_t before = i_;
      if (!parse_escape(tmp, s_, i_)) {
        i_ = before;
        return false;
      }
      for (char b : tmp) bytes.push_back(static_cast<uint8_t>(b));
      return true;
    }
    bytes.push_back(static_cast<uint8_t>(c));
    return true;
  }

  AstPtr parse_class() {
    ++i_;  // '['
    bool negate = false;
    if (peek() == '^') {
      negate = true;
      ++i_;
    }
    std::array<bool, 256> set{};
    bool any = false;
    while (true) {
      if (eof()) {
        fail("closing ']'");
        return nullptr;
      }
      if (peek() == ']') {
        ++i_;
        break;
      }
      std::vector<uint8_t> lo;
      if (!class_atom(lo)) {
        fail("class atom");
        return nullptr;
      }
      if (lo.size() == 1 && peek() == '-' && i_ + 1 < s_.size() &&
          s_[i_ + 1] != ']') {
        ++i_;
        std::vector<uint8_t> hi;
        if (!class_atom(hi) || hi.size() != 1) {
          fail("class range end");
          return nullptr;
        }
        if (hi[0] < lo[0]) {
          fail("reversed class range");
          return nullptr;
        }
        for (int b = lo[0]; b <= hi[0]; ++b) set[size_t(b)] = true;
        any = true;
        continue;
      }
      for (uint8_t b : lo) set[b] = true;
      any = true;
    }
    if (!any) {
      fail("empty character class");
      return nullptr;
    }
    if (negate)
      for (bool& b : set) b = !b;
    auto a = Ast::make(Ast::Kind::Class);
    a->cls = set;
    return a;
  }
};

// --- NFA -------------------------------------------------------------------

struct ByteEdge {
  uint8_t lo, hi;
  int32_t to;
};

struct CallEdge {
  int32_t to;    // callee entry
  int32_t cont;  // return address
};

struct NState {
  std::vector<ByteEdge> bytes;
  std::vector<CallEdge> calls;
  std::vector<int32_t> eps;
  bool ret = false;    // rule exit: pop the call stack
  bool accept = false; // start rule exit: the grammar can end here
};

class NfaBuilder {
public:
  std::expected<void, std::string> build(const std::vector<Rule>& rules) {
    root_name_ = rules.front().name;
    for (const Rule& r : rules) {
      if (rules_.count(r.name)) return std::unexpected("duplicate rule '" + r.name + "'");
      rules_[r.name] = RuleRef{add(), add(), r.body.get()};
    }
    for (const Rule& r : rules) {
      RuleRef& ref = rules_.at(r.name);
      // Compile the body starting at the entry state, then route its end to
      // the rule exit.
      const int32_t end = compile(*r.body, ref.entry);
      if (!err_.empty()) return std::unexpected(err_);
      st_[size_t(end)].eps.push_back(ref.exit);
      st_[size_t(ref.exit)].ret = (r.name != root_name_);
      st_[size_t(ref.exit)].accept = (r.name == root_name_);
    }
    if (root_entry() < 0) return std::unexpected("start rule missing");
    return {};
  }

  const std::vector<NState>& states() const { return st_; }
  int32_t root_entry() const {
    auto it = rules_.find(root_name_);
    return it == rules_.end() ? -1 : it->second.entry;
  }

private:
  struct RuleRef {
    int32_t entry;
    int32_t exit;
    const Ast* body;
  };

  std::vector<NState> st_;
  std::map<std::string, RuleRef> rules_;
  std::string root_name_;
  std::string err_;

  int32_t add() {
    st_.emplace_back();
    return static_cast<int32_t>(st_.size() - 1);
  }

  void eps(int32_t a, int32_t b) { st_[size_t(a)].eps.push_back(b); }

  int32_t compile(const Ast& a, int32_t at) {
    if (!err_.empty()) return at;
    switch (a.kind) {
      case Ast::Kind::Empty:
        return at;
      case Ast::Kind::Lit: {
        int32_t cur = at;
        for (unsigned char c : a.lit) {
          const int32_t n = add();
          st_[size_t(cur)].bytes.push_back(ByteEdge{c, c, n});
          cur = n;
        }
        return cur;
      }
      case Ast::Kind::Class: {
        const int32_t n = add();
        int b = 0;
        while (b < 256) {
          if (!a.cls[size_t(b)]) {
            ++b;
            continue;
          }
          int e = b;
          while (e + 1 < 256 && a.cls[size_t(e + 1)]) ++e;
          st_[size_t(at)].bytes.push_back(
              ByteEdge{uint8_t(b), uint8_t(e), n});
          b = e + 1;
        }
        return n;
      }
      case Ast::Kind::Seq: {
        int32_t cur = at;
        for (const AstPtr& k : a.kids) cur = compile(*k, cur);
        return cur;
      }
      case Ast::Kind::Alt: {
        const int32_t join = add();
        for (const AstPtr& k : a.kids) {
          const int32_t start = add();
          eps(at, start);
          const int32_t end = compile(*k, start);
          eps(end, join);
        }
        return join;
      }
      case Ast::Kind::Star: {
        const int32_t end = add();
        const int32_t loop = add();
        eps(at, end);
        eps(at, loop);
        const int32_t body_end = compile(*a.kid, loop);
        eps(body_end, loop);
        // Leave the loop after any iteration: without this edge the star can
        // only be taken zero times or forever.
        eps(body_end, end);
        return end;
      }
      case Ast::Kind::Plus: {
        const int32_t loop = add();
        eps(at, loop);
        const int32_t body_end = compile(*a.kid, loop);
        eps(body_end, loop);
        const int32_t end = add();
        eps(body_end, end);
        return end;
      }
      case Ast::Kind::Opt: {
        const int32_t end = add();
        eps(at, end);
        const int32_t body_end = compile(*a.kid, at);
        eps(body_end, end);
        return end;
      }
      case Ast::Kind::Rep: {
        int32_t cur = at;
        for (int k = 0; k < a.min; ++k) cur = compile(*a.kid, cur);
        if (a.max < 0) {
          const int32_t end = add();
          const int32_t loop = add();
          eps(cur, end);
          eps(cur, loop);
          const int32_t body_end = compile(*a.kid, loop);
          eps(body_end, loop);
          return end;
        }
        for (int k = a.min; k < a.max; ++k) {
          const int32_t end = add();
          eps(cur, end);
          const int32_t body_end = compile(*a.kid, cur);
          eps(body_end, end);
          cur = end;
        }
        return cur;
      }
      case Ast::Kind::Ref: {
        auto it = rules_.find(a.lit);
        if (it == rules_.end()) {
          err_ = "unknown rule '" + a.lit + "'";
          return at;
        }
        const int32_t cont = add();
        st_[size_t(at)].calls.push_back(CallEdge{it->second.entry, cont});
        return cont;
      }
    }
    return at;
  }
};

// --- grammar state ---------------------------------------------------------

using Stack = std::vector<int32_t>;
using Config = std::pair<int32_t, Stack>;
using ConfigKey = std::vector<Config>;

struct TrieEdge {
  uint8_t c;
  int32_t child;
};

struct TrieNode {
  std::vector<TrieEdge> edges;  // sorted by byte after construction
  std::vector<int32_t> ids;     // tokens whose text ends here
};

struct SetNode {
  ConfigKey cfgs;
  bool accepting = false;
  std::array<uint8_t, 32> byte_ok{};   // bit c = some config has a byte edge c
  std::vector<int32_t> trans;          // 256; -1 unknown, -2 dead, else set id
  bool mask_ready = false;
  std::vector<uint8_t> mask;

  SetNode() : trans(256, -1) {}
};

constexpr size_t kMaxStack = 64;
constexpr size_t kMaxConfigs = 4096;

}  // namespace

struct Grammar::Impl {
  const Vocab* vocab = nullptr;
  size_t vocab_size = 0;
  std::vector<int32_t> stops;

  std::vector<NState> nfa;
  int32_t root_entry = -1;

  std::vector<TrieNode> trie;
  std::vector<std::string> pieces;  // id -> piece bytes ("" = never allowed)

  std::deque<SetNode> sets;
  std::map<ConfigKey, int32_t> set_ids;
  int32_t start_set = -1;
  int32_t cur_set = -1;
  bool finished = false;

  bool closure(std::vector<Config> in, ConfigKey& out) const {
    std::set<Config> seen;
    std::vector<Config> work;
    for (Config& c : in) {
      if (seen.insert(c).second) work.push_back(std::move(c));
    }
    while (!work.empty()) {
      Config c = std::move(work.back());
      work.pop_back();
      const NState& s = nfa[size_t(c.first)];
      for (int32_t t : s.eps) {
        Config n{t, c.second};
        if (seen.insert(n).second) work.push_back(std::move(n));
      }
      for (const CallEdge& e : s.calls) {
        if (c.second.size() >= kMaxStack) return false;
        Config n{e.to, c.second};
        n.second.push_back(e.cont);
        if (seen.insert(n).second) work.push_back(std::move(n));
      }
      if (s.ret && !c.second.empty()) {
        Config n{c.second.back(), c.second};
        n.second.pop_back();
        if (seen.insert(n).second) work.push_back(std::move(n));
      }
      if (seen.size() > kMaxConfigs) return false;
    }
    out.assign(seen.begin(), seen.end());
    return true;
  }

  int32_t intern(ConfigKey key, bool accepting) {
    auto it = set_ids.find(key);
    if (it != set_ids.end()) return it->second;
    const int32_t id = static_cast<int32_t>(sets.size());
    sets.emplace_back();
    SetNode& n = sets.back();
    n.cfgs = std::move(key);
    n.accepting = accepting;
    for (const Config& c : n.cfgs) {
      for (const ByteEdge& e : nfa[size_t(c.first)].bytes) {
        for (int b = e.lo; b <= e.hi; ++b)
          n.byte_ok[size_t(b) >> 3] |= uint8_t(1u << (b & 7));
      }
    }
    set_ids.emplace(n.cfgs, id);
    return id;
  }

  int32_t closure_set(std::vector<Config> in) {
    ConfigKey key;
    if (!closure(std::move(in), key)) return -2;
    bool accepting = false;
    for (const Config& c : key) accepting |= nfa[size_t(c.first)].accept;
    return intern(std::move(key), accepting);
  }

  int32_t transition(int32_t set_id, uint8_t c) {
    if (set_id < 0) return -2;
    SetNode& n = sets[size_t(set_id)];
    int32_t& slot = n.trans[c];
    if (slot != -1) return slot;
    std::vector<Config> next;
    for (const Config& cfg : n.cfgs) {
      for (const ByteEdge& e : nfa[size_t(cfg.first)].bytes) {
        if (c >= e.lo && c <= e.hi) next.emplace_back(e.to, cfg.second);
      }
    }
    if (next.empty()) {
      slot = -2;
      return slot;
    }
    slot = closure_set(std::move(next));
    return slot;
  }

  bool byte_ok(int32_t set_id, uint8_t c) const {
    return (sets[size_t(set_id)].byte_ok[size_t(c) >> 3] >> (c & 7)) & 1;
  }

  void mark(int32_t node, int32_t set_id, std::vector<uint8_t>& mask) {
    const TrieNode& t = trie[size_t(node)];
    for (int32_t id : t.ids)
      mask[size_t(id) >> 3] |= uint8_t(1u << (id & 7));
    for (const TrieEdge& e : t.edges) {
      if (!byte_ok(set_id, e.c)) continue;
      const int32_t next = transition(set_id, e.c);
      if (next < 0) continue;
      mark(e.child, next, mask);
    }
  }

  bool is_stop(int32_t token) const {
    return std::find(stops.begin(), stops.end(), token) != stops.end();
  }

  const std::vector<uint8_t>& compute_mask() {
    SetNode& n = sets[size_t(cur_set)];
    if (finished) {
      // Nothing may be sampled after the stop token; the set's cached mask
      // (computed while it was still accepting) must not be handed out again.
      n.mask.clear();
      n.mask_ready = true;
      return n.mask;
    }
    if (!n.mask_ready) {
      n.mask.assign((vocab_size + 7) / 8, 0);
      mark(0, cur_set, n.mask);
      if (n.accepting) {
        for (int32_t t : stops)
          if (t >= 0 && size_t(t) < vocab_size)
            n.mask[size_t(t) >> 3] |= uint8_t(1u << (t & 7));
      }
      n.mask_ready = true;
    }
    return n.mask;
  }
};

Grammar::Grammar() : impl_(std::make_unique<Impl>()) {}
Grammar::~Grammar() = default;

std::expected<std::unique_ptr<Grammar>, std::string> Grammar::parse(
    std::string_view gbnf, const Vocab& vocab,
    std::span<const int32_t> stop_tokens) {
  GbnfParser parser(gbnf);
  auto rules = parser.parse();
  if (!rules) return std::unexpected(rules.error());

  NfaBuilder builder;
  if (auto ok = builder.build(*rules); !ok) return std::unexpected(ok.error());

  auto g = std::unique_ptr<Grammar>(new Grammar());
  Impl& im = *g->impl_;
  im.vocab = &vocab;
  im.vocab_size = vocab.tokens.size();
  im.stops.assign(stop_tokens.begin(), stop_tokens.end());
  im.nfa = builder.states();
  im.root_entry = builder.root_entry();
  im.pieces.assign(im.vocab_size, std::string());

  // Vocabulary trie. Stop tokens are excluded: they are allowed by the
  // accepting-state rule, not as text, and a tokenizer could otherwise expose
  // one whose piece is also a legal literal.
  im.trie.emplace_back();
  for (size_t id = 0; id < im.vocab_size; ++id) {
    const int32_t t = static_cast<int32_t>(id);
    if (im.is_stop(t)) continue;
    std::string piece = token_piece(vocab, t);
    if (piece.empty()) continue;
    im.pieces[id] = piece;
    int32_t node = 0;
    for (unsigned char c : piece) {
      int32_t child = -1;
      for (const TrieEdge& e : im.trie[size_t(node)].edges) {
        if (e.c == c) {
          child = e.child;
          break;
        }
      }
      if (child < 0) {
        child = static_cast<int32_t>(im.trie.size());
        im.trie.emplace_back();
        im.trie[size_t(node)].edges.push_back(TrieEdge{c, child});
      }
      node = child;
    }
    im.trie[size_t(node)].ids.push_back(t);
  }
  for (TrieNode& n : im.trie)
    std::sort(n.edges.begin(), n.edges.end(),
              [](const TrieEdge& a, const TrieEdge& b) { return a.c < b.c; });

  im.start_set = im.closure_set({Config{im.root_entry, Stack{}}});
  if (im.start_set < 0)
    return std::unexpected(
        "grammar has unbounded left recursion or call growth");
  im.cur_set = im.start_set;

  // A concrete first-position check: the start state must allow at least one
  // token, otherwise generation can never produce a single character.
  if (g->impl_->compute_mask().empty() ||
      std::all_of(g->impl_->sets[size_t(g->impl_->cur_set)].mask.begin(),
                  g->impl_->sets[size_t(g->impl_->cur_set)].mask.end(),
                  [](uint8_t b) { return b == 0; })) {
    // An empty vocabulary mask at the start means no token can begin the
    // text; report it while the grammar is being installed rather than as a
    // sampling failure on the first step.
    return std::unexpected("no token can start this grammar");
  }
  return g;
}

const std::vector<uint8_t>& Grammar::mask() { return impl_->compute_mask(); }

bool Grammar::accept(int32_t token) {
  Impl& im = *impl_;
  if (token < 0 || size_t(token) >= im.vocab_size || im.finished) return false;
  if (im.is_stop(token)) {
    if (!im.sets[size_t(im.cur_set)].accepting) return false;
    im.finished = true;
    return true;
  }
  const std::string& piece = im.pieces[size_t(token)];
  if (piece.empty()) return false;
  int32_t cur = im.cur_set;
  for (unsigned char c : piece) {
    cur = im.transition(cur, c);
    if (cur < 0) return false;
  }
  im.cur_set = cur;
  return true;
}

bool Grammar::accepting() {
  return impl_->sets[size_t(impl_->cur_set)].accepting;
}

bool Grammar::finished() const { return impl_->finished; }

void Grammar::reset() {
  impl_->cur_set = impl_->start_set;
  impl_->finished = false;
}

size_t Grammar::vocab_size() const { return impl_->vocab_size; }

// ===========================================================================
// JSON Schema -> GBNF
// ===========================================================================

namespace {

struct JVal {
  enum class K { Null, Bool, Num, Str, Arr, Obj } k = K::Null;
  bool b = false;
  std::string s;  // string value, or a number's literal text
  std::vector<JVal> a;
  std::vector<std::pair<std::string, JVal>> o;

  const JVal* get(std::string_view key) const {
    if (k != K::Obj) return nullptr;
    for (const auto& kv : o)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
};

class JsonParser {
public:
  explicit JsonParser(std::string_view s) : s_(s) {}

  std::expected<JVal, std::string> parse() {
    skip();
    JVal v;
    if (!value(v)) return std::unexpected(err_);
    skip();
    if (i_ != s_.size()) return std::unexpected("trailing JSON text");
    return v;
  }

private:
  std::string_view s_;
  size_t i_ = 0;
  std::string err_;

  void skip() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' ||
                              s_[i_] == '\n' || s_[i_] == '\r'))
      ++i_;
  }

  bool lit(std::string_view word) {
    if (s_.substr(i_, word.size()) != word) return false;
    i_ += word.size();
    return true;
  }

  bool value(JVal& out) {
    skip();
    if (i_ >= s_.size()) return err("value");
    const char c = s_[i_];
    if (c == '{') return object(out);
    if (c == '[') return array(out);
    if (c == '"') {
      out.k = JVal::K::Str;
      return string(out.s);
    }
    if (lit("true")) {
      out.k = JVal::K::Bool;
      out.b = true;
      return true;
    }
    if (lit("false")) {
      out.k = JVal::K::Bool;
      out.b = false;
      return true;
    }
    if (lit("null")) {
      out.k = JVal::K::Null;
      return true;
    }
    return number(out);
  }

  bool number(JVal& out) {
    const size_t b = i_;
    if (i_ < s_.size() && s_[i_] == '-') ++i_;
    if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') return err("number");
    while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    if (i_ < s_.size() && s_[i_] == '.') {
      ++i_;
      if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') return err("number");
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') return err("number");
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    out.k = JVal::K::Num;
    out.s.assign(s_.substr(b, i_ - b));
    return true;
  }

  bool string(std::string& out) {
    ++i_;  // '"'
    while (true) {
      if (i_ >= s_.size()) return err("string");
      const char c = s_[i_++];
      if (c == '"') return true;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (i_ >= s_.size()) return err("escape");
      const char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (i_ + 4 > s_.size()) return err("\\u escape");
          uint32_t cp = 0;
          for (int k = 0; k < 4; ++k) {
            const int h = hex_digit(s_[i_ + size_t(k)]);
            if (h < 0) return err("\\u escape");
            cp = (cp << 4) | uint32_t(h);
          }
          i_ += 4;
          // Surrogate pair.
          if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 6 <= s_.size() &&
              s_[i_] == '\\' && s_[i_ + 1] == 'u') {
            uint32_t lo = 0;
            bool ok = true;
            for (int k = 0; k < 4; ++k) {
              const int h = hex_digit(s_[i_ + 2 + size_t(k)]);
              if (h < 0) ok = false;
              lo = (lo << 4) | uint32_t(h < 0 ? 0 : h);
            }
            if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              i_ += 6;
            }
          }
          utf8_append(out, cp);
          break;
        }
        default:
          return err("escape");
      }
    }
  }

  bool object(JVal& out) {
    out.k = JVal::K::Obj;
    ++i_;  // '{'
    skip();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return true;
    }
    while (true) {
      skip();
      if (i_ >= s_.size() || s_[i_] != '"') return err("object key");
      std::string key;
      if (!string(key)) return false;
      skip();
      if (i_ >= s_.size() || s_[i_] != ':') return err("':'");
      ++i_;
      JVal v;
      if (!value(v)) return false;
      out.o.emplace_back(std::move(key), std::move(v));
      skip();
      if (i_ < s_.size() && s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (i_ < s_.size() && s_[i_] == '}') {
        ++i_;
        return true;
      }
      return err("',' or '}'");
    }
  }

  bool array(JVal& out) {
    out.k = JVal::K::Arr;
    ++i_;  // '['
    skip();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      return true;
    }
    while (true) {
      JVal v;
      if (!value(v)) return false;
      out.a.push_back(std::move(v));
      skip();
      if (i_ < s_.size() && s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (i_ < s_.size() && s_[i_] == ']') {
        ++i_;
        return true;
      }
      return err("',' or ']'");
    }
  }

  bool err(const std::string& what) {
    if (err_.empty())
      err_ = what + " expected at offset " + std::to_string(i_);
    return false;
  }
};

std::string json_quote(std::string_view s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out += "\"";
  return out;
}

std::string json_dump(const JVal& v);

std::string json_dump(const JVal& v) {
  switch (v.k) {
    case JVal::K::Null: return "null";
    case JVal::K::Bool: return v.b ? "true" : "false";
    case JVal::K::Num: return v.s;
    case JVal::K::Str: return json_quote(v.s);
    case JVal::K::Arr: {
      std::string out = "[";
      for (size_t i = 0; i < v.a.size(); ++i) {
        if (i) out += ",";
        out += json_dump(v.a[i]);
      }
      return out + "]";
    }
    case JVal::K::Obj: {
      std::string out = "{";
      for (size_t i = 0; i < v.o.size(); ++i) {
        if (i) out += ",";
        out += json_quote(v.o[i].first);
        out += ":";
        out += json_dump(v.o[i].second);
      }
      return out + "}";
    }
  }
  return "null";
}

// Bytes of `text` written as a GBNF double-quoted literal.
std::string gbnf_quote(std::string_view text) {
  std::string out = "\"";
  for (unsigned char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\x%02x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out += "\"";
  return out;
}

// A GBNF double-quoted literal holding exactly these bytes.
std::string qlit(std::string_view bytes) { return gbnf_quote(bytes); }

class SchemaCompiler {
public:
  std::expected<std::string, std::string> compile(const JVal& root,
                                                  std::string_view prefix,
                                                  std::string_view suffix) {
    root_ = &root;
    rules_.emplace_back("root", "");
    auto body = emit(root, 0);
    if (!body) return std::unexpected(body.error());
    // An envelope (when given) is literal text around the value. Its root
    // matches exactly that text: no whitespace before it, and none after it, so
    // a finished envelope leaves the stop tokens as its only continuation.
    // Whitespace inside the value (an object's members, between array elements)
    // is still the value rule's own ws.
    //
    // A bare value is JSON text (RFC 8259), so up to 16 bytes of whitespace may
    // precede it. It is the same bounded ws the value uses, so a greedy decode
    // cannot pad forever, and a model that opens its reply with a newline keeps
    // that choice. Nothing may follow the value: at the accepting state the
    // stop tokens are then its only continuation. A trailing whitespace option
    // would instead let a model that prefers a newline to its end-of-turn token
    // keep emitting whitespace up to the bound, padding a finished reply.
    std::string text = *body;
    if (!suffix.empty()) text += " " + qlit(suffix);
    if (!prefix.empty()) text = qlit(prefix) + " " + text;
    if (prefix.empty() && suffix.empty()) text = ensure_ws() + " " + text;
    rules_.front().second = text;
    std::string out;
    for (const auto& r : rules_) {
      out += r.first;
      out += " ::= ";
      out += r.second;
      out += "\n";
    }
    return out;
  }

private:
  const JVal* root_ = nullptr;
  std::vector<std::pair<std::string, std::string>> rules_;
  std::map<std::string, std::string> ref_names_;  // $ref path -> rule name
  std::map<std::string, std::string> named_;      // role -> rule name
  int counter_ = 0;

  static std::unexpected<std::string> err(std::string m) {
    return std::unexpected(std::move(m));
  }

  std::string fresh(std::string_view base) {
    return std::string(base) + "_" + std::to_string(counter_++);
  }

  std::string add_rule(std::string name, std::string body) {
    rules_.emplace_back(std::move(name), std::move(body));
    return rules_.back().first;
  }

  std::string ensure_ws() {
    auto it = named_.find("ws");
    if (it != named_.end()) return it->second;
    // Bounded, not `*`: a greedy decode that opens an object and then prefers
    // whitespace would otherwise cycle on it forever (every loop iteration is a
    // valid continuation). Sixteen characters per run is more than pretty-
    // printed indentation needs and forces the next real token.
    const std::string name = add_rule("ws", "[ \\t\\n\\r]{0,16}");
    named_["ws"] = name;
    return name;
  }

  // One character of a JSON string: a raw byte that is neither a quote, a
  // backslash nor a control byte, or an escape sequence. Byte-oriented, so a
  // raw UTF-8 continuation byte is one unit.
  std::string strchar() {
    auto it = named_.find("strchar");
    if (it != named_.end()) return it->second;
    const std::string body = "([^\"\\\\\\x00-\\x1f] | " + qlit("\\") +
                             " (\"u\" [0-9a-fA-F]{4} | [\"\\\\/bfnrt]))";
    const std::string name = add_rule(fresh("strchar"), body);
    named_["strchar"] = name;
    return name;
  }

  std::string string_value() {
    auto it = named_.find("string");
    if (it != named_.end()) return it->second;
    const std::string name = fresh("string");
    named_["string"] = name;
    add_rule(name, "(" + qlit("\"") + " " + strchar() + "* " + qlit("\"") + ")");
    return name;
  }

  std::string number() {
    auto it = named_.find("number");
    if (it != named_.end()) return it->second;
    const std::string body =
        "(\"-\"? (\"0\" | [1-9] [0-9]*) (" + qlit(".") +
        " [0-9]+)? ([eE] [+-]? [0-9]+)?)";
    const std::string name = add_rule(fresh("number"), body);
    named_["number"] = name;
    return name;
  }

  std::string boolean_value() {
    auto it = named_.find("boolean");
    if (it != named_.end()) return it->second;
    const std::string name =
        add_rule(fresh("boolean"), "(\"true\" | \"false\")");
    named_["boolean"] = name;
    return name;
  }

  // The unconstrained JSON value grammar, for schemas that do not pin a type.
  // Names are reserved before the bodies are built so the mutual recursion
  // between value/object/array terminates.
  std::string value() {
    auto it = named_.find("value");
    if (it != named_.end()) return it->second;
    const std::string name = fresh("value");
    named_["value"] = name;
    const std::string body = "(" + any_object() + " | " + any_array() + " | " +
                             string_value() + " | " + number() + " | " +
                             boolean_value() + " | " + qlit("null") + ")";
    add_rule(name, body);
    return name;
  }

  std::string any_object() {
    auto it = named_.find("object");
    if (it != named_.end()) return it->second;
    const std::string name = fresh("object");
    named_["object"] = name;
    const std::string ws = ensure_ws();
    const std::string v = value();
    const std::string k = string_value();
    const std::string body =
        "(" + qlit("{") + " " + ws + " ( " + k + " " + ws + " " + qlit(":") +
        " " + ws + " " + v + " ( " + ws + " " + qlit(",") + " " + ws + " " + k +
        " " + ws + " " + qlit(":") + " " + ws + " " + v + " )* )? " + ws + " " +
        qlit("}") + ")";
    add_rule(name, body);
    return name;
  }

  std::string any_array() {
    auto it = named_.find("array");
    if (it != named_.end()) return it->second;
    const std::string name = fresh("array");
    named_["array"] = name;
    const std::string ws = ensure_ws();
    const std::string v = value();
    const std::string body =
        "(" + qlit("[") + " " + ws + " ( " + v + " ( " + ws + " " + qlit(",") +
        " " + ws + " " + v + " )* )? " + ws + " " + qlit("]") + ")";
    add_rule(name, body);
    return name;
  }

  static const std::set<std::string>& unsupported_keywords() {
    static const std::set<std::string> k = {
        "pattern", "format", "not", "if", "then", "else", "propertyNames",
        "uniqueItems", "patternProperties", "contains", "minProperties",
        "maxProperties", "prefixItems", "multipleOf", "exclusiveMinimum",
        "exclusiveMaximum", "dependentRequired", "dependentSchemas",
        "additionalItems", "unevaluatedProperties", "unevaluatedItems",
        "contentEncoding", "contentMediaType", "maxContains", "minContains"};
    return k;
  }

  static const std::set<std::string>& known_keywords() {
    static const std::set<std::string> k = {
        "type", "const", "enum", "anyOf", "oneOf", "allOf", "$ref",
        "properties", "required", "additionalProperties", "items", "minItems",
        "maxItems", "minLength", "maxLength", "minimum", "maximum", "title",
        "description", "default", "examples", "$schema", "$id", "$comment",
        "$defs", "definitions"};
    return k;
  }

  std::expected<void, std::string> check_keywords(const JVal& s) {
    for (const auto& kv : s.o) {
      if (unsupported_keywords().count(kv.first))
        return err("unsupported keyword '" + kv.first + "'");
      if (!known_keywords().count(kv.first))
        return err("unknown keyword '" + kv.first + "'");
    }
    return {};
  }

  std::expected<std::vector<std::string>, std::string> types_of(const JVal& s) {
    const JVal* t = s.get("type");
    if (!t) return std::vector<std::string>{};
    if (t->k == JVal::K::Str) return std::vector<std::string>{t->s};
    if (t->k == JVal::K::Arr) {
      std::vector<std::string> out;
      for (const JVal& v : t->a) {
        if (v.k != JVal::K::Str) return err("type array element");
        out.push_back(v.s);
      }
      return out;
    }
    return err("unsupported 'type' value");
  }

  static bool has(const JVal& s, std::string_view k) {
    return s.get(k) != nullptr;
  }

  std::string literal_of(const JVal& v) { return gbnf_quote(json_dump(v)); }

  std::string join(const std::vector<std::string>& parts, const char* sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i) out += sep;
      out += parts[i];
    }
    return out;
  }

  std::expected<std::string, std::string> emit(const JVal& s, int depth) {
    if (depth > 24) return err("schema nesting too deep");
    if (s.k == JVal::K::Bool) {
      if (s.b) return value();
      return err("boolean schema false matches nothing");
    }
    if (s.k != JVal::K::Obj) return err("schema must be an object");
    if (auto ok = check_keywords(s); !ok) return std::unexpected(ok.error());

    if (const JVal* c = s.get("const")) return literal_of(*c);
    if (const JVal* e = s.get("enum")) {
      if (e->k != JVal::K::Arr || e->a.empty())
        return err("enum must be a non-empty array");
      std::string out = "(";
      for (size_t i = 0; i < e->a.size(); ++i) {
        if (i) out += " | ";
        out += literal_of(e->a[i]);
      }
      return out + ")";
    }
    for (const char* comb : {"anyOf", "oneOf"}) {
      if (const JVal* a = s.get(comb)) {
        if (a->k != JVal::K::Arr || a->a.empty())
          return err(std::string(comb) + " must be a non-empty array");
        std::string out = "(";
        for (size_t i = 0; i < a->a.size(); ++i) {
          auto sub = emit(a->a[i], depth + 1);
          if (!sub) return sub;
          if (i) out += " | ";
          out += *sub;
        }
        return out + ")";
      }
    }
    if (const JVal* a = s.get("allOf")) {
      if (a->k != JVal::K::Arr || a->a.empty())
        return err("allOf must be a non-empty array");
      // Merge object schemas. Anything else is rejected rather than
      // approximated, so a compiled grammar never claims less than the schema.
      JVal merged;
      merged.k = JVal::K::Obj;
      JVal props;
      props.k = JVal::K::Obj;
      JVal required;
      required.k = JVal::K::Arr;
      bool addl_false = false;
      const JVal* addl_schema = nullptr;
      for (const JVal& sub : a->a) {
        if (sub.k != JVal::K::Obj) return err("allOf branch is not an object schema");
        if (auto ok = check_keywords(sub); !ok) return std::unexpected(ok.error());
        if (sub.get("$ref")) return err("allOf with $ref is unsupported");
        auto ts = types_of(sub);
        if (!ts) return std::unexpected(ts.error());
        bool is_obj = ts->empty();
        for (const auto& t : *ts) is_obj |= (t == "object");
        if (!is_obj) return err("allOf branch is not an object schema");
        if (const JVal* p = sub.get("properties")) {
          if (p->k != JVal::K::Obj) return err("properties must be an object");
          for (const auto& kv : p->o) {
            if (props.get(kv.first))
              return err("allOf merges duplicate property '" + kv.first + "'");
            props.o.emplace_back(kv.first, kv.second);
          }
        }
        if (const JVal* r = sub.get("required")) {
          if (r->k != JVal::K::Arr) return err("required must be an array");
          for (const JVal& v : r->a) {
            if (v.k != JVal::K::Str) return err("required element");
            bool dup = false;
            for (const JVal& q : required.a) dup |= (q.s == v.s);
            if (!dup) required.a.push_back(v);
          }
        }
        if (const JVal* ap = sub.get("additionalProperties")) {
          if (ap->k == JVal::K::Bool && !ap->b) addl_false = true;
          if (ap->k == JVal::K::Obj) addl_schema = ap;
        }
      }
      JVal type_key;
      type_key.k = JVal::K::Str;
      type_key.s = "object";
      merged.o.emplace_back("type", std::move(type_key));
      merged.o.emplace_back("properties", std::move(props));
      merged.o.emplace_back("required", std::move(required));
      if (addl_false) {
        JVal f;
        f.k = JVal::K::Bool;
        f.b = false;
        merged.o.emplace_back("additionalProperties", std::move(f));
      } else if (addl_schema) {
        merged.o.emplace_back("additionalProperties", *addl_schema);
      }
      return emit_object(merged, depth + 1);
    }
    if (const JVal* ref = s.get("$ref")) {
      if (ref->k != JVal::K::Str) return err("$ref must be a string");
      return emit_ref(ref->s, depth + 1);
    }

    auto ts = types_of(s);
    if (!ts) return std::unexpected(ts.error());
    if (ts->empty()) {
      if (has(s, "properties") || has(s, "required") ||
          has(s, "additionalProperties"))
        return emit_object(s, depth + 1);
      return value();
    }
    if (ts->size() > 1) {
      std::string out = "(";
      for (size_t i = 0; i < ts->size(); ++i) {
        auto one = emit_one(s, (*ts)[i], depth + 1);
        if (!one) return one;
        if (i) out += " | ";
        out += *one;
      }
      return out + ")";
    }
    return emit_one(s, ts->front(), depth + 1);
  }

  std::expected<std::string, std::string> emit_one(const JVal& s,
                                                   const std::string& type,
                                                   int depth) {
    if (type == "object") return emit_object(s, depth);
    if (type == "array") return emit_array(s, depth);
    if (type == "string") return emit_string(s, depth);
    if (type == "integer") return emit_number(s, true);
    if (type == "number") return emit_number(s, false);
    if (type == "boolean") return boolean_value();
    if (type == "null") return std::string("\"null\"");
    return err("unsupported type '" + type + "'");
  }

  std::expected<std::string, std::string> emit_ref(const std::string& path,
                                                   int depth) {
    const JVal* target = nullptr;
    if (path == "#") {
      target = root_;
    } else {
      for (const char* prefix : {"#/$defs/", "#/definitions/"}) {
        const std::string_view p(prefix);
        if (path.rfind(p, 0) != 0) continue;
        const std::string key = path.substr(p.size());
        const JVal* defs =
            root_->get(p == "#/$defs/" ? "$defs" : "definitions");
        if (defs) target = defs->get(key);
      }
    }
    if (!target) return err("unsupported $ref '" + path + "'");
    auto it = ref_names_.find(path);
    if (it != ref_names_.end()) return it->second;
    const std::string name = fresh("ref");
    ref_names_[path] = name;
    add_rule(name, "");  // reserve before compiling a recursive reference
    auto body = emit(*target, depth + 1);
    if (!body) return body;
    set_rule(name, *body);
    return name;
  }

  std::expected<std::string, std::string> emit_object(const JVal& s,
                                                      int depth) {
    const std::string ws = ensure_ws();
    const JVal* props = s.get("properties");
    if (props && props->k != JVal::K::Obj)
      return err("properties must be an object");
    const JVal* required = s.get("required");
    if (required && required->k != JVal::K::Arr)
      return err("required must be an array");
    const JVal* addl = s.get("additionalProperties");
    bool extra = true;  // JSON Schema default
    const JVal* extra_schema = nullptr;
    if (addl) {
      if (addl->k == JVal::K::Bool) {
        extra = addl->b;
      } else if (addl->k == JVal::K::Obj) {
        extra_schema = addl;
      } else {
        return err("additionalProperties must be a boolean or a schema");
      }
    }
    const size_t n = props ? props->o.size() : 0;
    if (n == 0) {
      if (extra) return any_object();
      return "(" + qlit("{") + " " + ws + " " + qlit("}") + ")";
    }
    if (n > 8) return err("more than 8 declared properties unsupported");

    std::vector<std::string> keys;
    std::vector<const JVal*> values;
    keys.reserve(n);
    values.reserve(n);
    for (const auto& kv : props->o) {
      keys.push_back(kv.first);
      values.push_back(&kv.second);
    }
    uint32_t required_mask = 0;
    if (required) {
      for (const JVal& r : required->a) {
        if (r.k != JVal::K::Str) return err("required element");
        bool found = false;
        for (size_t i = 0; i < keys.size(); ++i)
          if (keys[i] == r.s) {
            required_mask |= 1u << i;
            found = true;
          }
        if (!found) return err("required property '" + r.s + "' is not declared");
      }
    }
    std::vector<std::string> value_exprs;
    value_exprs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto v = emit(*values[i], depth + 1);
      if (!v) return v;
      value_exprs.push_back(*v);
    }
    std::string extra_value;
    if (extra) {
      if (extra_schema) {
        auto v = emit(*extra_schema, depth + 1);
        if (!v) return v;
        extra_value = *v;
      } else {
        extra_value = value();
      }
    }

    const std::string tag = fresh("obj");
    std::string extra_pair;
    std::string extra_pairs;
    if (extra) {
      extra_pairs = tag + "_xs";
      extra_pair = string_value() + " " + ws + " " + qlit(":") + " " + ws + " " +
                   extra_value;
      // This rule is only ever entered after a comma, so it must consume at
      // least one pair: an empty alternative here would make the comma
      // trailing, which JSON does not allow.
      add_rule(extra_pairs, "( " + extra_pair + " " + ws + " ( " + qlit(",") +
                                " " + ws + " " + extra_pairs + " )? )");
    }

    // pair_state(mask) matches the remaining pairs given the subset of declared
    // keys already emitted. Keys may appear in any order, each at most once;
    // extra keys are only allowed once every required key has appeared.
    //
    // Every state has two rules. `sM` is the sequence of remaining pairs and
    // may match empty when the required keys are satisfied (that is how the
    // object ends). `pM` is the same sequence but consumes at least one pair.
    // A continuation after a comma always targets `pNext`, never `sNext`:
    // `sNext` matching empty after a comma would be a trailing comma.
    std::map<uint32_t, std::string> pair_rules;
    std::vector<uint32_t> todo{0};
    const uint32_t full_mask = uint32_t((uint64_t(1) << n) - 1);
    while (!todo.empty()) {
      const uint32_t mask = todo.back();
      todo.pop_back();
      if (pair_rules.count(mask)) continue;
      const std::string name = tag + "_s" + std::to_string(mask);
      const std::string p_name = tag + "_p_s" + std::to_string(mask);
      const bool complete = (mask & required_mask) == required_mask;
      std::vector<std::string> alts;
      for (size_t i = 0; i < n; ++i) {
        if (mask & (1u << i)) continue;
        const uint32_t next = mask | (1u << i);
        if (!pair_rules.count(next)) todo.push_back(next);
        // The object may only end once every required key has appeared, so the
        // continuation is optional only when `next` already satisfies that;
        // otherwise the remaining required pairs are forced.
        const bool next_complete = (next & required_mask) == required_mask;
        // `pNext` exists only when another pair can follow: a declared key is
        // still available, or the required keys are done and extra keys are
        // allowed. Otherwise this member is the last one.
        const bool has_more = next != full_mask || (extra && next_complete);
        std::string cont;
        if (has_more) {
          const std::string next_p = tag + "_p_s" + std::to_string(next);
          cont = next_complete
                     ? " ( " + qlit(",") + " " + ws + " " + next_p + " )?"
                     : " " + qlit(",") + " " + ws + " " + next_p;
        }
        alts.push_back(gbnf_quote(json_quote(keys[i])) + " " + ws + " " +
                       qlit(":") + " " + ws + " " + value_exprs[i] +
                       (cont.empty() ? "" : " " + ws + " " + cont));
      }
      if (extra && complete) {
        alts.push_back(extra_pair + " " + ws + " ( " + qlit(",") + " " + ws +
                       " " + extra_pairs + " )?");
      }
      if (!alts.empty()) add_rule(p_name, "( " + join(alts, " | ") + " )");
      // May the sequence stop here? Only when every required key is present.
      if (complete) {
        pair_rules[mask] = alts.empty()
                               ? "( " + qlit("") + " )"
                               : "( " + qlit("") + " | " + p_name + " )";
      } else {
        pair_rules[mask] = p_name;
      }
    }
    return add_rule(tag, "(" + qlit("{") + " " + ws + " " + pair_rules[0] +
                             " " + ws + " " + qlit("}") + ")");
  }

  void set_rule(const std::string& name, std::string body) {
    for (auto& r : rules_)
      if (r.first == name) {
        r.second = std::move(body);
        return;
      }
  }

  std::expected<std::string, std::string> emit_array(const JVal& s,
                                                     int depth) {
    const std::string ws = ensure_ws();
    const JVal* items = s.get("items");
    std::string item;
    if (items) {
      if (items->k == JVal::K::Arr) return err("tuple items is unsupported");
      auto v = emit(*items, depth + 1);
      if (!v) return v;
      item = *v;
    } else {
      item = value();
    }
    int min_items = 0;
    int max_items = -1;
    if (const JVal* m = s.get("minItems")) {
      auto n = int_of(*m);
      if (!n) return err("minItems must be a non-negative integer");
      min_items = *n;
    }
    if (const JVal* m = s.get("maxItems")) {
      auto n = int_of(*m);
      if (!n) return err("maxItems must be a non-negative integer");
      max_items = *n;
    }
    if (max_items >= 0 && max_items < min_items)
      return err("maxItems is below minItems");
    if (max_items > 256) return err("maxItems above 256 unsupported");
    if (min_items > 256) return err("minItems above 256 unsupported");
    if (max_items == 0) return "(" + qlit("[") + " " + ws + " " + qlit("]") + ")";
    const std::string more =
        " ( " + qlit(",") + " " + ws + " " + item + " )";
    std::string inner;
    if (min_items == 0) {
      std::string opt;
      if (max_items < 0) {
        opt = item + more + "*";
      } else if (max_items == 1) {
        opt = item;
      } else {
        opt = item + more + "{0," + std::to_string(max_items - 1) + "}";
      }
      inner = "( " + opt + " )?";
    } else {
      inner = item;
      if (min_items > 1)
        inner += more + "{" + std::to_string(min_items - 1) + "}";
      if (max_items < 0) {
        inner += more + "*";
      } else if (max_items > min_items) {
        inner += more + "{0," + std::to_string(max_items - min_items) + "}";
      }
    }
    return "(" + qlit("[") + " " + ws + " " + inner + " " + ws + " " + qlit("]") +
           ")";
  }

  std::expected<std::string, std::string> emit_string(const JVal& s,
                                                      int depth) {
    (void)depth;
    int min_len = 0;
    int max_len = -1;
    if (const JVal* m = s.get("minLength")) {
      auto n = int_of(*m);
      if (!n) return err("minLength must be a non-negative integer");
      min_len = *n;
    }
    if (const JVal* m = s.get("maxLength")) {
      auto n = int_of(*m);
      if (!n) return err("maxLength must be a non-negative integer");
      max_len = *n;
    }
    if (max_len >= 0 && max_len < min_len)
      return err("maxLength is below minLength");
    if (min_len > 256 || max_len > 256)
      return err("string length bounds above 256 unsupported");
    const std::string chars = strchar();
    std::string rep;
    if (max_len < 0) {
      rep = min_len == 0 ? chars + "*"
                         : chars + "{" + std::to_string(min_len) + ",}";
    } else if (min_len == max_len) {
      rep = chars + "{" + std::to_string(min_len) + "}";
    } else {
      rep = chars + "{" + std::to_string(min_len) + "," +
            std::to_string(max_len) + "}";
    }
    return "(" + qlit("\"") + " " + rep + " " + qlit("\"") + ")";
  }

  std::expected<int, std::string> int_of(const JVal& v) {
    if (v.k != JVal::K::Num) return err("expected an integer");
    char* end = nullptr;
    const long n = std::strtol(v.s.c_str(), &end, 10);
    if (!end || *end != '\0' || n < -1000000 || n > 1000000)
      return err("expected an integer");
    return static_cast<int>(n);
  }

  std::expected<std::string, std::string> emit_number(const JVal& s,
                                                      bool integer) {
    const JVal* min = s.get("minimum");
    const JVal* max = s.get("maximum");
    if (min || max) {
      if (!integer) return err("bounds on a non-integer number unsupported");
      if (!min || !max)
        return err("integer needs both minimum and maximum for a bounded grammar");
      auto lo = int_of(*min);
      auto hi = int_of(*max);
      if (!lo) return err("minimum must be an integer");
      if (!hi) return err("maximum must be an integer");
      if (*hi < *lo) return err("maximum is below minimum");
      if (*hi - *lo > 256) return err("integer range wider than 256 unsupported");
      std::string out = "(";
      for (int v = *lo; v <= *hi; ++v) {
        if (v != *lo) out += " | ";
        out += gbnf_quote(std::to_string(v));
      }
      return out + ")";
    }
    if (integer) return std::string("(\"-\"? (\"0\" | [1-9] [0-9]*))");
    return number();
  }
};


}  // namespace

std::expected<std::string, std::string> json_schema_to_gbnf(
    std::string_view schema_json, std::string_view prefix,
    std::string_view suffix) {
  JsonParser parser(schema_json);
  auto root = parser.parse();
  if (!root) return std::unexpected(root.error());
  SchemaCompiler compiler;
  return compiler.compile(*root, prefix, suffix);
}

}  // namespace sonicboom::vocab
