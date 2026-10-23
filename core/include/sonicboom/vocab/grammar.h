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

#pragma once

// Constrained decoding: a byte-level grammar over the tokenizer's vocabulary.
//
// A `Grammar` is the sampling-time half of constrained decoding (the other half
// is the token mask the sampler applies before its argmax):
//
//   GBNF text -> Grammar::parse() -> per-position allowed-token bitmap
//
// The grammar consumes the same bytes the tokenizer's detokenize() would emit
// for each token, so it constrains the *text* the model produces, not token
// ids. Its state is a set of NFA configurations (state + rule-call stack);
// recursion between rules is therefore supported (JSON values contain objects
// and arrays contain values). Each distinct configuration set caches its mask,
// so a step inside a long string or number costs a bitmap lookup, not a walk
// over the vocabulary.
//
// The grammar is byte-oriented. GBNF character classes are byte sets; a schema
// that constrains non-ASCII text uses UTF-8 byte ranges (the JSON Schema
// compiler in this module emits `\x80`-`\xFF` for unconstrained string
// content, which admits every valid multi-byte sequence).
//
// Layer 2: no native-torch / ATen / c10 types cross this header.

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <sonicboom/vocab/vocab.h>

namespace sonicboom::vocab {

class Grammar {
public:
  // Parse `gbnf` and build the token trie from `vocab`. `stop_tokens` (the
  // model's EOS ids) are not part of the text alphabet: they are allowed only
  // where the grammar can end, and accepting one finishes the grammar.
  //
  // The first rule of the GBNF source is the start rule. An error string names
  // the rule/offset that failed to parse or compile (unsupported syntax,
  // left recursion, unbounded rule-call growth).
  static std::expected<std::unique_ptr<Grammar>, std::string> parse(
      std::string_view gbnf, const Vocab& vocab,
      std::span<const int32_t> stop_tokens = {});

  ~Grammar();
  Grammar(const Grammar&) = delete;
  Grammar& operator=(const Grammar&) = delete;

  // The allowed-token bitmap for the current position: bit t is set when token
  // t may be sampled. Its length is (vocab_size + 7) / 8; ids past vocab_size
  // are always zero. The buffer stays valid until the next mask()/accept().
  const std::vector<uint8_t>& mask();

  // Advance over the text of `token`. Returns false (and changes nothing) when
  // the token is not allowed at this position. A stop token is allowed only in
  // an accepting position and finishes the grammar.
  bool accept(int32_t token);

  // True when the text consumed so far is a complete value of the grammar.
  bool accepting();

  // True after an accepting position sampled one of the stop tokens.
  bool finished() const;

  // Return to the start position.
  void reset();

  size_t vocab_size() const;

private:
  Grammar();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Compile a JSON Schema document (one JSON value) to GBNF text.
//
// Supported: type (single or union), const, enum, properties, required,
// additionalProperties (false/true/schema), patternProperties (rejected),
// items, prefixItems (rejected), minItems, maxItems, minLength, maxLength,
// anyOf, oneOf (as alternation; overlapping branches are the caller's
// problem), allOf over object schemas, local $ref into $defs/definitions, and
// the annotation keywords (title, description, default, examples, $schema,
// $id, $comment). Integer minimum/maximum are accepted when the range is small
// enough to enumerate (at most 256 values); other numeric bounds and all
// pattern/format/not/if-then-else/propertyNames/uniqueItems constraints are
// rejected by name rather than silently ignored, so a compiled grammar always
// implies the constraints it claims.
//
// Without PREFIX/SUFFIX the grammar is JSON text (RFC 8259): up to 16 bytes of
// whitespace may precede the value. The bound keeps a greedy decode from
// padding forever, and a reply that opens with a newline decodes to the same
// tokens it would have chosen unconstrained. Nothing may follow the value, so
// once it is complete the stop tokens are the only continuation: a model that
// would rather continue with a newline ends its reply there instead of padding
// whitespace up to the bound.
//
// When PREFIX or SUFFIX is non-empty the value must also appear between those
// literal texts (UTF-8, quoted for GBNF here). That is how a caller composes a
// schema into a larger envelope, such as a tool-call marker, without re-parsing
// the grammar. That root matches exactly the envelope — no whitespace before or
// after it — so a complete envelope leaves the stop tokens as its only
// continuation. (Whitespace inside the value is still accepted.)
std::expected<std::string, std::string> json_schema_to_gbnf(
    std::string_view schema_json, std::string_view prefix = {},
    std::string_view suffix = {});

}  // namespace sonicboom::vocab
