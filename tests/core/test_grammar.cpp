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

// Constrained decoding's vocabulary half, without a model: the GBNF parser,
// the pushdown-NFA mask, and the JSON Schema -> GBNF compiler. The toy
// vocabularies below let each grammar position's allowed-token bitmap and the
// accept/accepting/finished lifecycle be checked exactly.
//
// The end-to-end property (20 schemas x 3 models, 100% schema-valid output, and
// greedy identity whenever the unconstrained token is allowed) needs the real
// GGUFs and lives in the private test suite, which drives the C API.

#include <sonicboom/vocab/grammar.h>
#include <sonicboom/vocab/vocab.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sbvocab = sonicboom::vocab;

namespace {

// The EOS id of every toy vocabulary below: the grammar's only stop token.
constexpr int32_t kStops[] = {1};

int g_failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

// A Gemma-style vocabulary: Normal tokens with ▁ for a leading space, one EOS
// control token. `eos` is the stop-token id.
sbvocab::Vocab make_vocab(std::vector<std::string> tokens) {
  sbvocab::Vocab v;
  v.kind = sbvocab::TokenizerKind::Gemma4;
  v.tokens = std::move(tokens);
  v.type.assign(v.tokens.size(), sbvocab::TokenType::Normal);
  // The first two entries are control tokens (<bos>, <eos>); the rest are the
  // caller's text pieces, stored with ▁ for a leading space.
  v.tokens[0] = "<bos>";
  v.tokens[1] = "<eos>";
  v.type[0] = sbvocab::TokenType::Control;
  v.type[1] = sbvocab::TokenType::Control;
  v.escape_whitespaces = true;
  v.bos = 0;
  v.eos = 1;
  for (size_t i = 0; i < v.tokens.size(); ++i)
    v.token_to_id[v.tokens[i]] = int32_t(i);
  return v;
}

std::vector<int32_t> allowed(const std::vector<uint8_t>& mask, size_t vocab) {
  std::vector<int32_t> out;
  for (size_t i = 0; i < vocab; ++i)
    if ((mask[i >> 3] >> (i & 7)) & 1u) out.push_back(int32_t(i));
  return out;
}

void test_word_grammar() {
  // Text pieces: ids 2..7. "yes"/"no" stand alone; " yes"/" no" carry a space.
  auto vocab = make_vocab({"", "", "yes", "no", "\xE2\x96\x81yes",
                           "\xE2\x96\x81no", "y", "n"});
  check(vocab.tokens[2] == "yes", "toy vocab built");

  auto g = sbvocab::Grammar::parse(R"(root ::= "yes" | "no")", vocab, kStops);
  check(bool(g), "word grammar parses");
  if (!g) return;
  std::unique_ptr<sbvocab::Grammar> gb = std::move(*g);

  // At the start only complete "yes"/"no" may be sampled: "y" and "n" are
  // prefixes of the grammar but are not complete tokens the grammar can accept
  // alone — they ARE allowed (they start a valid path). The mask is over token
  // pieces, so every non-empty piece that the grammar can consume is allowed:
  // ids 2,3 ("yes","no"), 4,5 (leading space — not allowed), 6,7 ("y","n").
  const auto start = allowed(gb->mask(), vocab.tokens.size());
  check(std::find(start.begin(), start.end(), 2) != start.end(), "start: 'yes' allowed");
  check(std::find(start.begin(), start.end(), 3) != start.end(), "start: 'no' allowed");
  check(std::find(start.begin(), start.end(), 6) != start.end(), "start: 'y' allowed");
  check(std::find(start.begin(), start.end(), 4) == start.end(), "start: ' yes' disallowed");
  check(std::find(start.begin(), start.end(), 1) == start.end(), "start: <eos> disallowed");
  check(!gb->accepting(), "start: not accepting");

  check(gb->accept(2), "accept 'yes'");
  check(gb->accepting(), "after 'yes': accepting");
  check(!gb->accept(3), "'no' disallowed after 'yes'");
  const auto after = allowed(gb->mask(), vocab.tokens.size());
  check(std::find(after.begin(), after.end(), 1) != after.end(),
        "after 'yes': <eos> allowed");
  check(gb->accept(1), "accept <eos>");
  check(gb->finished(), "after <eos>: finished");
  check(gb->mask().empty(), "finished: empty mask");

  gb->reset();
  check(!gb->accepting() && !gb->finished(), "reset: start state");
  check(std::find(allowed(gb->mask(), vocab.tokens.size()).begin(),
                  allowed(gb->mask(), vocab.tokens.size()).end(), 3) !=
            allowed(gb->mask(), vocab.tokens.size()).end(),
        "reset: 'no' allowed again");
  check(gb->accept(3), "accept 'no'");
  check(gb->accept(1), "accept <eos> after 'no'");
  check(gb->finished(), "finished again");
}

void test_byte_class() {
  auto vocab = make_vocab({"", "", "yes", "no", "\xE2\x96\x81x", "1"});
  auto g = sbvocab::Grammar::parse(R"(root ::= [a-z]+ )", vocab, kStops);
  check(bool(g), "class grammar parses");
  if (!g) return;
  auto gb = std::move(*g);
  check(gb->accept(2), "class: 'yes' accepted");
  check(gb->accepting(), "class: accepting after one word");
  const auto m = allowed(gb->mask(), vocab.tokens.size());
  check(std::find(m.begin(), m.end(), 1) != m.end(), "class: <eos> allowed");
  check(std::find(m.begin(), m.end(), 3) != m.end(), "class: another word allowed");

  auto bad = sbvocab::Grammar::parse(R"(root ::= [a-z)", vocab, kStops);
  check(!bad, "unterminated class is rejected");

  // A grammar no token can start is a load-time error, not a silent dead loop.
  auto dead = sbvocab::Grammar::parse(R"(root ::= "zzz")", vocab, kStops);
  check(!dead, "unreachable first byte is rejected");
}

// A vocabulary of single-character tokens, enough for the JSON grammars and
// the <tool_call> envelope.
sbvocab::Vocab make_byte_vocab() {
  std::vector<std::string> pieces = {
      "", "", "{", "}", "\"", ":", ",", "n", "a", "m", "e", "o", "k",
      "t", "r", "u", "f", "l", "s", "x", "c", "0", "1", ".", "-", " ",
      "\n", "[", "]", "<", ">", "/", "_", "b", "z", "2", "3"};
  return make_vocab(std::move(pieces));
}

// The id of the one-character token C, or -1 when the vocabulary has none.
int32_t id_of(const sbvocab::Vocab& vocab, char c) {
  for (size_t i = 0; i < vocab.tokens.size(); ++i)
    if (vocab.tokens[i].size() == 1 && vocab.tokens[i][0] == c)
      return int32_t(i);
  return int32_t(-1);
}

void test_json_schema() {
  const char* schema = R"({"type":"object",
    "properties":{"name":{"type":"string"},"ok":{"type":"boolean"}},
    "required":["name","ok"],"additionalProperties":false})";
  auto gbnf = sbvocab::json_schema_to_gbnf(schema);
  check(bool(gbnf), "object schema compiles");
  if (!gbnf) {
    std::cerr << "    " << gbnf.error() << "\n";
    return;
  }

  auto vocab = make_byte_vocab();
  const int32_t stop = 1;
  const int32_t stops[1] = {stop};
  auto g = sbvocab::Grammar::parse(*gbnf, vocab,
                                   std::span<const int32_t>(stops, 1));
  check(bool(g), "compiled schema grammar parses");
  if (!g) {
    std::cerr << "    " << g.error() << "\n";
    return;
  }
  auto gb = std::move(*g);

  // A complete, valid instance: {"name":"x","ok":true}. Every byte must be
  // accepted (whitespace around the value is tested separately below).
  const std::string text = R"({"name":"x","ok":true})";
  bool all = true;
  for (size_t i = 0; i < text.size(); ++i)
    if (!gb->accept(id_of(vocab, text[i]))) {
      std::cerr << "  byte " << i << " rejected: '" << text[i] << "'\n";
      all = false;
      break;
    }
  check(all, "valid instance accepted byte by byte");
  check(gb->accepting(), "complete instance is accepting");
  check(gb->accept(stop), "stop token ends it");
  check(gb->finished(), "finished after stop");

  // A wrong key is rejected at the first stray byte.
  gb->reset();
  const std::string bad = R"({"nope":1)";
  bool ok_path = true;
  for (size_t i = 0; i < bad.size(); ++i) {
    if (!gb->accept(id_of(vocab, bad[i]))) {
      ok_path = (i == bad.size() - 1);  // expected: rejected at 'o' (or later)
      break;
    }
  }
  check(!ok_path || !gb->accepting(), "unknown key is not accepted");

  // Missing required member: closing the object early must not be accepting.
  gb->reset();
  const std::string short_obj = R"({"name":"x"})";
  bool all2 = true;
  for (char c : short_obj) all2 = all2 && gb->accept(id_of(vocab, c));
  check(!all2 || !gb->accepting(), "missing required member is not accepting");

  // Unsupported keywords are rejected by name, not ignored.
  auto rejected = sbvocab::json_schema_to_gbnf(
      R"({"type":"string","pattern":"^a+$"})");
  check(!rejected, "pattern keyword rejected");
  if (!rejected)
    check(rejected.error().find("pattern") != std::string::npos,
          "rejection names the keyword");

  // Union type and enum also compile.
  auto unions = sbvocab::json_schema_to_gbnf(
      R"({"type":["string","null"],"enum":["a","b"]})");
  check(bool(unions), "union + enum compiles");
  auto integer = sbvocab::json_schema_to_gbnf(
      R"({"type":"integer","minimum":1,"maximum":5})");
  check(bool(integer), "bounded integer compiles");
  auto ref = sbvocab::json_schema_to_gbnf(R"({
    "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},
                     "required":["next"]}},
    "$ref":"#/$defs/node"})");
  check(bool(ref), "recursive $ref compiles");
}

// Every byte of `text` must be accepted and the position afterwards must be an
// accepting one. The grammar is reset first.
bool accepts_text(sbvocab::Grammar& gb, const sbvocab::Vocab& vocab,
                  const std::string& text) {
  gb.reset();
  for (char c : text)
    if (!gb.accept(id_of(vocab, c))) return false;
  return gb.accepting();
}

// The root of a bare schema is JSON text (RFC 8259): up to 16 bytes of
// whitespace may precede the value, so a model that opens a reply with a
// newline keeps the tokens it would have chosen unconstrained. Nothing may
// follow the value, so a finished reply can only be followed by a stop token
// (a trailing whitespace option would let a model pad after the value to the
// whitespace bound). A PREFIX/SUFFIX envelope is exact on both sides, so a
// tool-call marker cannot be padded either.
void test_json_text_whitespace() {
  auto vocab = make_byte_vocab();
  const int32_t stops[1] = {1};

  auto s = sbvocab::json_schema_to_gbnf(R"({"type":"boolean"})");
  check(bool(s), "text grammar: schema compiles");
  if (s) {
    auto g = sbvocab::Grammar::parse(*s, vocab,
                                     std::span<const int32_t>(stops, 1));
    check(bool(g), "text grammar parses");
    if (g) {
      check(accepts_text(**g, vocab, "true"), "JSON text: bare value accepted");
      check(accepts_text(**g, vocab, " \ntrue"),
            "JSON text: leading whitespace accepted");
      check(!accepts_text(**g, vocab, "true\n"),
            "JSON text: nothing follows the value");
      check(!accepts_text(**g, vocab, "true false"),
            "JSON text: two values rejected");
      const std::string long_ws(17, ' ');
      check(!accepts_text(**g, vocab, long_ws + "true"),
            "JSON text: a 17-byte whitespace run is rejected");
      check(!accepts_text(**g, vocab, long_ws),
            "JSON text: whitespace alone is not a value");
    }
  }

  auto e = sbvocab::json_schema_to_gbnf(R"({"type":"boolean"})", "<t>", "</t>");
  check(bool(e), "envelope grammar: schema compiles");
  if (e) {
    auto g = sbvocab::Grammar::parse(*e, vocab,
                                     std::span<const int32_t>(stops, 1));
    check(bool(g), "envelope grammar parses");
    if (g) {
      check(accepts_text(**g, vocab, "<t>true</t>"), "envelope accepted");
      check(!accepts_text(**g, vocab, " <t>true</t>"),
            "envelope: no leading padding");
      check(!accepts_text(**g, vocab, "<t>true</t> "),
            "envelope: no trailing padding");
      check(!accepts_text(**g, vocab, "<t> true</t>"),
            "envelope: the value starts right after the marker");
    }
  }
}

// A comma after the last member must not be accepted. The object states are
// entered both at `{` (where "no more pairs" is a valid sequence end, so `{}`
// and a short object work) and after a comma (where it is not): if one state
// serves both, the continuation after a comma can match empty and
// `{"a":1,}` becomes valid JSON that is not JSON.
void test_no_trailing_comma() {
  auto vocab = make_byte_vocab();
  const int32_t stops[1] = {1};

  const char* one = R"({"type":"object","properties":{"a":{"type":"integer"}},
      "required":["a"],"additionalProperties":false})";
  auto g1 = sbvocab::json_schema_to_gbnf(one);
  check(bool(g1), "one-member object schema compiles");
  if (g1) {
    auto parsed = sbvocab::Grammar::parse(*g1, vocab,
                                          std::span<const int32_t>(stops, 1));
    check(bool(parsed), "one-member object grammar parses");
    if (parsed) {
      check(accepts_text(**parsed, vocab, R"({"a":1})"),
            "one-member object accepted");
      check(!accepts_text(**parsed, vocab, R"({"a":1,})"),
            "trailing comma after the last member is rejected");
    }
  }

  // Two members, one required: the continuation after `a` is optional, and
  // after `b` there is nothing left to continue with.
  const char* two = R"({"type":"object",
      "properties":{"a":{"type":"integer"},"b":{"type":"integer"}},
      "required":["a"],"additionalProperties":false})";
  auto g2 = sbvocab::json_schema_to_gbnf(two);
  check(bool(g2), "two-member object schema compiles");
  if (g2) {
    auto parsed = sbvocab::Grammar::parse(*g2, vocab,
                                          std::span<const int32_t>(stops, 1));
    check(bool(parsed), "two-member object grammar parses");
    if (parsed) {
      check(accepts_text(**parsed, vocab, R"({"a":1})"),
            "required member alone is accepted");
      check(accepts_text(**parsed, vocab, R"({"a":1,"b":2})"),
            "both members are accepted");
      check(accepts_text(**parsed, vocab, R"({"b":2,"a":1})"),
            "members may appear in any order");
      check(!accepts_text(**parsed, vocab, R"({"a":1,})"),
            "trailing comma after an optional continuation is rejected");
      check(!accepts_text(**parsed, vocab, R"({"b":2,"a":1,})"),
            "trailing comma after the last declared member is rejected");
    }
  }

  // additionalProperties (the default) uses its own repetition rule, which is
  // also entered only after a comma.
  const char* extra = R"({"type":"object","properties":{"a":{"type":"integer"}},
      "required":["a"]})";
  auto g3 = sbvocab::json_schema_to_gbnf(extra);
  check(bool(g3), "object with extra properties compiles");
  if (g3) {
    auto parsed = sbvocab::Grammar::parse(*g3, vocab,
                                          std::span<const int32_t>(stops, 1));
    check(bool(parsed), "extra-properties grammar parses");
    if (parsed) {
      check(accepts_text(**parsed, vocab, R"({"a":1,"z":2})"),
            "an extra property is accepted");
      check(!accepts_text(**parsed, vocab, R"({"a":1,"z":2,})"),
            "trailing comma after an extra property is rejected");
    }
  }
}

// The compiler's envelope form: the schema value between two literal texts.
// That is how the server constrains a tool call to one <tool_call> block
// without a second grammar.
void test_schema_envelope() {
  auto gbnf = sbvocab::json_schema_to_gbnf(
      R"({"type":"object","properties":{"name":{"const":"cat"}},
          "required":["name"],"additionalProperties":false})",
      "<tool_call>\n", "\n</tool_call>");
  check(bool(gbnf), "envelope schema compiles");
  if (!gbnf) return;

  auto vocab = make_byte_vocab();
  const int32_t stops[1] = {1};
  auto g = sbvocab::Grammar::parse(*gbnf, vocab,
                                   std::span<const int32_t>(stops, 1));
  check(bool(g), "envelope grammar parses");
  if (!g) return;
  auto gb = std::move(*g);

  const std::string text = "<tool_call>\n{\"name\":\"cat\"}\n</tool_call>";
  bool all = true;
  for (char c : text)
    if (!gb->accept(id_of(vocab, c))) {
      all = false;
      break;
    }
  check(all, "tool-call envelope accepted");
  check(gb->accepting(), "tool-call envelope is accepting");
  check(gb->accept(1), "tool-call envelope stop finishes");
  check(gb->finished(), "tool-call envelope finished");
}

}  // namespace

int main() {
  {
    auto vocab = make_vocab({"", "", "yes", "no", "\xE2\x96\x81yes",
                             "\xE2\x96\x81no", "y", "n"});
    check(vocab.tokens.size() == 8, "vocab size");
  }
  test_word_grammar();
  test_byte_class();
  test_json_schema();
  test_json_text_whitespace();
  test_no_trailing_comma();
  test_schema_envelope();
  if (g_failures) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "test_grammar: OK\n";
  return 0;
}
