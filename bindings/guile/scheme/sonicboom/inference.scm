;;; SonicBoom — Guile binding: minimal inference surface
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The (sonicboom inference) module binds the minimal LLM C ABI (capi.h
;;; boom_llm_*) through (sonicboom ffi), which is the only path Guile uses to
;;; reach libsonicboom.so. It exposes load-model / make-session / generate —
;;; a decode inference loop over the GGUF built-in tokenizer (greedy by
;;; default, seeded sampling when temperature > 0).
;;;
;;;   Guile → (sonicboom inference) → (sonicboom ffi) → libsonicboom.so
;;;
;;; Handles are foreign pointers. Freeing is explicit (free-model!/free-session!)
;;; so a test owns its lifetime deterministically; the session holds a C++
;;; shared_ptr to the model so a freed model handle does not dangle a live
;;; session. Three entry points:
;;;   - generate         string -> string   (tokenize + decode + detokenize)
;;;   - generate-tokens  token-ids -> token-ids  (the raw token-ID loop)
;;;   - generate-stream  tokens -> tokens, calling a procedure per token with
;;;                      the text that became available with it

(define-module (sonicboom inference)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom error)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (ice-9 optargs)
  #:export (load-model free-model! model-arch model-eos
            make-session free-session!
            set-grammar! set-json-schema! clear-grammar! constraint-state
            json-schema->gbnf
            prefill! tokenize tokenize-segments detokenize
            generate generate-tokens generate-stream make-sampling-bv
            numerics-profile set-numerics-profile!))

;;; --- numerics profile -------------------------------------------------------

;;; The process-wide numeric contract (capi.h boom_set_numerics_profile): 'strict
;;; keeps every historical reduction, 'standard allows the profile's
;;; substitutions (today: f16 KV storage). It is latched at the first model or
;;; session, so call set-numerics-profile! first. Until either is called the
;;; process uses SONICBOOM_NUMERICS, then 'standard.

(define (numerics-profile)
  "The numerics profile this process is using.

The profile is the process-wide numeric contract: strict keeps every
historical reduction order, standard allows the profile's substitutions.
Args:
Returns: 'strict or 'standard.
Example:
  (use-modules (sonicboom inference))
  (memq (numerics-profile) '(strict standard))
  => (standard)
PyTorch: none
Note: The profile is chosen before the first model or session exists; see
set-numerics-profile!."
  (let ((err (make-slot))
        (out (make-bytevector (sizeof int) 0)))
    (sb-check! ((proc 'boom_get_numerics_profile) (bytevector->pointer out)
                (slot-ptr err))
               err "numerics-profile")
    (case (bytevector-s32-native-ref out 0)
      ((0) 'strict)
      (else 'standard))))

(define (set-numerics-profile! profile)
  "Request a numerics profile for this process.

The first model or session latches the profile, so call this before either.
Requesting the effective profile again is a no-op success; requesting the
other one after latching signals a &sonicboom-error.
Args:
  profile: 'strict or 'standard.
Returns: the effective profile after the request.
Example:
  (use-modules (sonicboom inference))
  (let ((p (numerics-profile))) (eq? p (set-numerics-profile! p)))
  => #t
PyTorch: none"
  (let ((code (case profile
                ((strict) 0)
                ((standard) 1)
                (else (raise-sonicboom-error
                       BOOM_ERR_BINDING 'binding
                       (string-append "unknown numerics profile: "
                                      (object->string profile))))))
        (err (make-slot)))
    (sb-check! ((proc 'boom_set_numerics_profile) code (slot-ptr err))
               err "set-numerics-profile!")
    (numerics-profile)))

;;; --- helpers ----------------------------------------------------------------

;; sb-check! comes from (sonicboom ffi): it raises a &sonicboom-error carrying
;; the C status, the boom_error_kind and the message.

;; A list of token ids → a native-endian int32 bytevector (boom_token[]).
(define (tokens->bv tokens)
  (let* ((n (length tokens))
         (bv (make-bytevector (* n 4) 0)))
    (let loop ((ts tokens) (i 0))
      (unless (null? ts)
        (bytevector-s32-native-set! bv (* i 4) (car ts))
        (loop (cdr ts) (+ i 1))))
    bv))

;; (sonicboom ffi)'s `proc` is the accessor for a C function, but
;; `generate-stream` takes the caller's per-token callback under that same name,
;; which would shadow it. This alias is how the streaming code reaches the C ABI.
(define (sb-fn name) (proc name))

;; Guile's `bytevector-copy` copies the whole vector; this is the range copy.
(define (bytevector-range bv start end)
  (let ((out (make-bytevector (- end start) 0)))
    (bytevector-copy! bv start out 0 (- end start))
    out))

;; A native-endian int32 bytevector of N elements → a list of token ids.
(define (s32-bv->list bv n)
  (let loop ((i 0) (acc '()))
    (if (< i n)
        (loop (+ i 1) (cons (bytevector-s32-native-ref bv (* i 4)) acc))
        (reverse acc))))

;; boom_llm_sampling as a native-layout bytevector (x86-64; see capi.h):
;;   struct_size u64@0, temperature f32@8, top_p f32@12, top_k u32@16,
;;   seed u64@24, max_tokens u32@32, min_p f32@36, repeat_penalty f32@40,
;;   repeat_last_n u32@44, presence_penalty f32@48,
;;   frequency_penalty f32@52  → 56 bytes.
;; The defaults are greedy/neutral: temperature 0, top_p 1 (off), top_k 0 (off),
;; min_p 0 (off), seed 0, repeat_penalty 1 (off), repeat_last_n 64, and the two
;; penalties 0. temperature 0 keeps the historical greedy path bit-identical.
(define* (make-sampling-bv max-tokens
                           #:key
                           (temperature 0.0)
                           (top-k 0)
                           (top-p 1.0)
                           (min-p 0.0)
                           (seed 0)
                           (repeat-penalty 1.0)
                           (repeat-last-n 64)
                           (presence-penalty 0.0)
                           (frequency-penalty 0.0))
  "Build the native sampling-parameters structure for a decode call.

Temperature 0 selects the greedy path; top-k 0, top-p 1, min-p 0,
repeat-penalty 1 and the two penalties 0 are neutral (off). A positive
temperature samples deterministically from seed.
Args:
  max-tokens: the maximum number of tokens to produce.
  temperature: the sampling temperature.
  top-k: keep the k most likely tokens; 0 disables.
  top-p: nucleus threshold; 1 disables.
  min-p: minimum relative probability; 0 disables.
  seed: the sampling seed.
  repeat-penalty: penalty for repeated tokens; 1 disables.
  repeat-last-n: how many recent tokens the repeat penalty looks at.
  presence-penalty: additive presence penalty; 0 disables.
  frequency-penalty: additive frequency penalty; 0 disables.
Returns: a bytevector holding the native structure.
Example:
  (use-modules (sonicboom inference))
  (bytevector-length (make-sampling-bv 8))
  => 56
PyTorch: none
Note: The bytevector is an internal representation; pass it through the
generation procedures rather than decoding it."
  (let ((bv (make-bytevector 56 0)))
    (bytevector-u64-native-set! bv 0 56)
    (bytevector-ieee-single-native-set! bv 8 (exact->inexact temperature))
    (bytevector-ieee-single-native-set! bv 12 (exact->inexact top-p))
    (bytevector-u32-native-set! bv 16 top-k)
    (bytevector-u64-native-set! bv 24 seed)
    (bytevector-u32-native-set! bv 32 max-tokens)
    (bytevector-ieee-single-native-set! bv 36 (exact->inexact min-p))
    (bytevector-ieee-single-native-set! bv 40 (exact->inexact repeat-penalty))
    (bytevector-u32-native-set! bv 44 repeat-last-n)
    (bytevector-ieee-single-native-set! bv 48 (exact->inexact presence-penalty))
    (bytevector-ieee-single-native-set! bv 52 (exact->inexact frequency-penalty))
    bv))

;;; --- model ------------------------------------------------------------------

(define (load-model path)
  "Load a model from a GGUF file.

Args:
  path: the path of the GGUF file.
Returns: an opaque model handle.
PyTorch: none
Note: GGUF is SonicBoom's model-file format; PyTorch core has no GGUF
loader."
  (let* ((model-slot (make-slot))
         (err-slot (make-slot))
         (st ((proc 'boom_llm_model_load) (string->pointer path)
              (slot-ptr model-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "load-model")
    (slot-ref model-slot)))

(define (free-model! model)
  "Release a model handle.

A session keeps the model it was created from alive, so freeing the handle
does not invalidate a live session.
Args:
  model: a model handle from load-model.
Returns: unspecified.
PyTorch: none"
  ((proc 'boom_llm_model_free) model))

;; The model's architecture, as its GGUF declares it in general.architecture
;; ("gemma4", "qwen2", ...). Who formats the prompt reads this from the model
;; rather than guessing the family from the file name.
(define (model-arch model)
  "The model's architecture name, as recorded in its GGUF metadata.

Args:
  model: a model handle.
Returns: a string such as \"gemma4\" or \"qwen2\".
PyTorch: none
Note: The name comes from the file rather than the model family being
guessed from the path."
  (let* ((len-slot (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot)))
    ;; Sizing call (cap 0): fills *len_out (the length, no NUL).
    (let ((st ((proc 'boom_llm_model_arch) model %null-pointer 0
               (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st BOOM_ERR_BINDING))
        (sb-check! st err-slot "model-arch")))
    (let* ((m (bytevector-u64-native-ref len-slot 0))
           (out (make-bytevector (+ m 1) 0))    ; +1 for the NUL terminator
           (st ((proc 'boom_llm_model_arch) model (bytevector->pointer out) (+ m 1)
                (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (sb-check! st err-slot "model-arch")
      (utf8->string (bytevector-range out 0 m)))))

;; The model's own end-of-sequence token ids (a possibly empty list), straight
;; from its metadata: what generation has to stop on, per model rather than per
;; family.
(define (model-eos model)
  "The model's end-of-sequence token ids.

Args:
  model: a model handle.
Returns: a list of token ids; empty when the file declares none.
PyTorch: none
Note: Generation stops on these ids, per model rather than per family."
  (let* ((n-slot (make-bytevector 8 0))        ; size_t
         (err-slot (make-slot)))
    ;; Sizing call (cap 0): fills *n_out (the count).
    (let ((st ((proc 'boom_llm_model_eos) model %null-pointer 0
               (bytevector->pointer n-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st BOOM_ERR_BINDING))
        (sb-check! st err-slot "model-eos")))
    (let* ((n (bytevector-u64-native-ref n-slot 0))
           (bv (make-bytevector (* n 4) 0))
           (st ((proc 'boom_llm_model_eos) model (bytevector->pointer bv) n
                (bytevector->pointer n-slot) (slot-ptr err-slot))))
      (sb-check! st err-slot "model-eos")
      (s32-bv->list bv n))))

;;; --- session ----------------------------------------------------------------

(define (make-session model n-ctx)
  "Create a session over a model.

A session holds the token history and the context window of one
conversation; it keeps the model alive.
Args:
  model: a model handle.
  n-ctx: the context-window size in tokens.
Returns: an opaque session handle.
PyTorch: none"
  (let* ((session-slot (make-slot))
         (err-slot (make-slot))
         (st ((proc 'boom_llm_session_new) model n-ctx
              (slot-ptr session-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "make-session")
    (slot-ref session-slot)))

(define (free-session! session)
  "Release a session handle.

Args:
  session: a session handle from make-session.
Returns: unspecified.
PyTorch: none"
  ((proc 'boom_llm_session_free) session))

;;; --- constrained decoding ---------------------------------------------------

;;; A session may carry a grammar (capi.h "Constrained decoding"): every sampled
;;; token must keep its text acceptable, and the model's EOS ids end the output
;;; where the grammar can end. The grammar restarts at every prefill, so install
;;; it before prefill! — that is what masks the turn's first token too.

(define (set-grammar! session gbnf)
  "Constrain a session's next turn to a grammar.

The grammar restarts at every prefill!, so install it before prefill! to
mask the turn's first token as well. Every sampled token must keep the
generated text acceptable to the grammar, and the model's end-of-sequence
ids end the output where the grammar can end.
Args:
  session: a session handle.
  gbnf: a grammar in the GBNF dialect.
Returns: unspecified.
PyTorch: none
Note: Constrained decoding is not part of PyTorch's generation loop."
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'boom_llm_session_set_grammar) session
                (string->pointer gbnf) (slot-ptr err-slot))
               err-slot "set-grammar!")))

(define (set-json-schema! session schema-json)
  "Constrain a session's next turn to a JSON Schema.

The schema is compiled to a grammar; an unsupported keyword is rejected
here by name rather than ignored.
Args:
  session: a session handle.
  schema-json: the schema as a JSON string.
Returns: unspecified.
PyTorch: none"
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'boom_llm_session_set_json_schema) session
                (string->pointer schema-json) (slot-ptr err-slot))
               err-slot "set-json-schema!")))

(define (clear-grammar! session)
  "Remove a session's grammar, so the next turn decodes unconstrained.

Args:
  session: a session handle.
Returns: unspecified.
PyTorch: none"
  ((proc 'boom_llm_session_clear_grammar) session))

(define (constraint-state session)
  "The state of a session's constrained decoding.

Args:
  session: a session handle.
Returns: two values: whether a grammar is installed, and whether the text
sampled so far is already a complete value of it.
PyTorch: none"
  (let* ((active (make-bytevector (sizeof int) 0))
         (complete (make-bytevector (sizeof int) 0))
         (err-slot (make-slot)))
    (sb-check! ((proc 'boom_llm_session_constraint_state) session
                (bytevector->pointer active) (bytevector->pointer complete)
                (slot-ptr err-slot))
               err-slot "constraint-state")
    (values (not (zero? (bytevector-s32-native-ref active 0)))
            (not (zero? (bytevector-s32-native-ref complete 0))))))

(define* (json-schema->gbnf schema-json #:key (prefix "") (suffix ""))
  "Compile a JSON Schema to GBNF grammar text.

Args:
  schema-json: the schema as a JSON string.
  prefix: literal text that must appear before the schema value; empty by
          default.
  suffix: literal text that must appear after the schema value; empty by
          default.
Returns: the grammar text.
Example:
  (use-modules (sonicboom inference))
  (string? (json-schema->gbnf \"{\\\"type\\\": \\\"string\\\"}\"))
  => #t
PyTorch: none
Note: The prefix and suffix are how a tool-call envelope is built around a
function's arguments schema."
  (let* ((len-slot (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot))
         (head (list (string->pointer schema-json)
                     (string->pointer prefix) (string->pointer suffix))))
    ;; Sizing call (cap 0): fills *len_out (the length, no NUL).
    (let ((st (apply (proc 'boom_llm_json_schema_to_gbnf)
                     (append head
                             (list %null-pointer 0
                                   (bytevector->pointer len-slot)
                                   (slot-ptr err-slot))))))
      (unless (or (sb-status-ok? st) (= st BOOM_ERR_BINDING))
        (sb-check! st err-slot "json-schema->gbnf")))
    (let* ((m (bytevector-u64-native-ref len-slot 0))
           (out (make-bytevector (+ m 1) 0))    ; +1 for the NUL terminator
           (st (apply (proc 'boom_llm_json_schema_to_gbnf)
                      (append head
                              (list (bytevector->pointer out) (+ m 1)
                                    (bytevector->pointer len-slot)
                                    (slot-ptr err-slot))))))
      (sb-check! st err-slot "json-schema->gbnf")
      (utf8->string (bytevector-range out 0 m)))))

;;; --- tokenize / detokenize --------------------------------------------------

;; boom_llm_tokenize_ex flags, from capi.h.
(define sb-tokenize-parse-special 1)
(define sb-tokenize-add-special 2)

;; The two-call sizing path over boom_llm_tokenize_ex: TEXT with the given FLAGS.
(define (tokenize-flags model text flags)
  (let* ((bv (string->utf8 text))
         (len (bytevector-length bv))
         (n-slot (make-bytevector 8 0))          ; size_t
         (err-slot (make-slot))
         (fn (proc 'boom_llm_tokenize_ex))
         (head (list model (bytevector->pointer bv) len flags)))
    ;; Sizing call (cap 0): fills *n_out, returns BOOM_ERR_BINDING.
    (let ((st (apply fn (append head
                                (list %null-pointer 0
                                      (bytevector->pointer n-slot)
                                      (slot-ptr err-slot))))))
      (unless (or (sb-status-ok? st) (= st BOOM_ERR_BINDING))
        (sb-check! st err-slot "tokenize")))
    (let* ((n (bytevector-u64-native-ref n-slot 0))
           (out (make-bytevector (* n 4) 0))
           (st (apply fn (append head
                                 (list (bytevector->pointer out) n
                                       (bytevector->pointer n-slot)
                                       (slot-ptr err-slot))))))
      (sb-check! st err-slot "tokenize")
      (s32-bv->list out n))))

(define* (tokenize model text #:key (special? #f))
  "Encode text to a list of token ids.

A beginning-of-sequence token is prepended, matching the command-line
tokenizer. Special-token parsing is off by default: it turns text into
control structure, so use it only on text the caller generated itself, never
on a user's message.
Args:
  model: a model handle.
  text: the UTF-8 text to encode.
  special?: when true, control pieces of the vocabulary that occur verbatim
            in text become single tokens.
Returns: a list of token ids.
PyTorch: none"
  (tokenize-flags model text
                  (logior sb-tokenize-add-special
                               (if special? sb-tokenize-parse-special 0))))

(define (tokenize-segments model segments)
  "Encode a rendered chat prompt given as segments.

Segments are (kind . text) pairs with kind 'text or 'special, as produced by
a chat template. A 'special segment is tokenized with special parsing, so a
turn marker spelled out in it becomes the vocabulary's control token; a
'text segment is tokenized as plain text, so a marker that came from a user's
message stays text. The beginning-of-sequence token is added once, by the
first segment.
Args:
  model: a model handle.
  segments: the prompt segments.
Returns: a list of token ids.
PyTorch: none"
  (let ((parts (if (null? segments) (list (cons 'text "")) segments)))
    (let lp ((rest parts) (first? #t) (out '()))
      (if (null? rest)
          (apply append (reverse out))
          (let* ((seg (car rest))
                 (flags (logior
                         (if (eq? (car seg) 'special)
                             sb-tokenize-parse-special 0)
                         (if first? sb-tokenize-add-special 0))))
            (lp (cdr rest) #f
                (cons (tokenize-flags model (cdr seg) flags) out)))))))

(define (detokenize-bytes model tokens)
  "Decode TOKENS (a list of token ids) to its UTF-8 bytes, as a bytevector. The
bytes are not necessarily a whole number of characters: a token may carry only
the first bytes of a multi-byte character, which is exactly what streaming has
to hold back until the rest arrives."
  (if (null? tokens)
      (make-bytevector 0 0)
      (let* ((n (length tokens))
             (in (tokens->bv tokens))
             (len-slot (make-bytevector 8 0))        ; size_t
             (err-slot (make-slot)))
        ;; Sizing call (cap 0): fills *len_out (the text length, no NUL).
        (let ((st ((proc 'boom_llm_detokenize) model (bytevector->pointer in) n
                   %null-pointer 0 (bytevector->pointer len-slot)
                   (slot-ptr err-slot))))
          (unless (or (sb-status-ok? st) (= st BOOM_ERR_BINDING))
            (sb-check! st err-slot "detokenize")))
        (let* ((m (bytevector-u64-native-ref len-slot 0))
               (out (make-bytevector (+ m 1) 0))     ; +1 for the NUL terminator
               (st ((proc 'boom_llm_detokenize) model (bytevector->pointer in) n
                     (bytevector->pointer out) (+ m 1) (bytevector->pointer len-slot)
                     (slot-ptr err-slot))))
          (sb-check! st err-slot "detokenize")
          ;; Copy exactly the m text bytes the C ABI wrote (it also NUL-terminated
          ;; out[m], which is why cap is m+1).
          (bytevector-range out 0 m)))))

(define (detokenize model tokens)
  "Decode a list of token ids back to text.

Args:
  model: a model handle.
  tokens: a list of token ids.
Returns: a UTF-8 string.
PyTorch: none"
  (utf8->string (detokenize-bytes model tokens)))

;;; --- prefill / generate -----------------------------------------------------

(define (prefill! session tokens)
  "Set a session's prompt to a list of token ids.

Any tokens the session already holds that are a prefix of the new prompt
are reused rather than recomputed.
Args:
  session: a session handle.
  tokens: the prompt as a list of token ids.
Returns: the number of tokens reused from the previous prompt.
PyTorch: none"
  (let* ((bv (tokens->bv tokens))
         (n (length tokens))
         (reused-slot (make-bytevector 8 0))
         (err-slot (make-slot))
         (st ((proc 'boom_llm_prefill) session (bytevector->pointer bv) n
              (bytevector->pointer reused-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "prefill")
    (bytevector-u64-native-ref reused-slot 0)))

(define* (generate-tokens session prompt max-tokens #:optional (prefill? #t)
                          #:key
                          (temperature 0.0)
                          (top-k 0)
                          (top-p 1.0)
                          (min-p 0.0)
                          (seed 0)
                          (repeat-penalty 1.0)
                          (repeat-last-n 64)
                          (presence-penalty 0.0)
                          (frequency-penalty 0.0))
  "Generate token ids from a prompt given as token ids.

Args:
  session: a session handle.
  prompt: the prompt as a list of token ids.
  max-tokens: the maximum number of tokens to produce.
  prefill?: when false, decode from the prompt the session already holds.
  temperature, top-k, top-p, min-p, seed, repeat-penalty, repeat-last-n,
  presence-penalty, frequency-penalty: sampling parameters, as in
  make-sampling-bv; the default temperature 0 is greedy.
Returns: the list of produced token ids.
PyTorch: none"
  (when prefill? (prefill! session prompt))
  (let* ((tokens '())
         (cb (lambda (user token)
               (set! tokens (cons token tokens))
               0))                 ; 0 = continue
         (cb-ptr (procedure->pointer int cb (list '* int32)))
         (smp-bv (make-sampling-bv max-tokens
                                   #:temperature temperature
                                   #:top-k top-k
                                   #:top-p top-p
                                   #:min-p min-p
                                   #:seed seed
                                   #:repeat-penalty repeat-penalty
                                   #:repeat-last-n repeat-last-n
                                   #:presence-penalty presence-penalty
                                   #:frequency-penalty frequency-penalty))
         (n-slot (make-bytevector 4 0))
         (err-slot (make-slot))
         (st ((proc 'boom_llm_decode) session (bytevector->pointer smp-bv)
              cb-ptr %null-pointer (bytevector->pointer n-slot)
              (slot-ptr err-slot))))
    (sb-check! st err-slot "decode")
    (reverse tokens)))

(define* (generate model session prompt max-tokens #:key
                   (temperature 0.0)
                   (top-k 0)
                   (top-p 1.0)
                   (min-p 0.0)
                   (seed 0)
                   (repeat-penalty 1.0)
                   (repeat-last-n 64)
                   (presence-penalty 0.0)
                   (frequency-penalty 0.0))
  "Generate text from a prompt.

Tokenizes the prompt, prefills the session, decodes up to max-tokens tokens
and detokenizes them.
Args:
  model: a model handle.
  session: a session handle over that model.
  prompt: the UTF-8 prompt text.
  max-tokens: the maximum number of tokens to produce.
  temperature, top-k, top-p, min-p, seed, repeat-penalty, repeat-last-n,
  presence-penalty, frequency-penalty: sampling parameters, as in
  make-sampling-bv; the default temperature 0 is greedy.
Returns: the generated text.
PyTorch: none"
  (detokenize model
              (generate-tokens session (tokenize model prompt) max-tokens #t
                               #:temperature temperature
                               #:top-k top-k
                               #:top-p top-p
                               #:min-p min-p
                               #:seed seed
                               #:repeat-penalty repeat-penalty
                               #:repeat-last-n repeat-last-n
                               #:presence-penalty presence-penalty
                               #:frequency-penalty frequency-penalty)))

;;; --- streaming --------------------------------------------------------------
;;; The C ABI hands each produced token to a callback as it is decoded
;;; (boom_llm_decode's boom_llm_token_cb), so a per-token stream only needs the text
;;; side to be assembled. Two things make that less trivial than "detokenize
;;; this token":
;;;
;;;   - a token's bytes may be an incomplete UTF-8 sequence, so a chunk may have
;;;     to wait for the next token; and
;;;   - detokenization is not additive. The tokenizer's space cleanup runs over
;;;     the whole text and looks across token boundaries, so a later token can
;;;     change text that an earlier token already produced — a per-token
;;;     piecewise concatenation would drift from `generate`'s output.
;;;
;;; So the running text is recomputed through the C ABI for the tokens produced
;;; so far and only the new, character-aligned tail is handed to the caller.
;;; That is O(tokens^2) detokenization for one call, against a decode that is
;;; orders of magnitude more expensive per step.

;; The length of the longest prefix of BV's first END bytes that ends on a UTF-8
;; character boundary: a trailing lead byte whose continuation bytes have not
;; arrived yet is left out (that is the point of the buffering). Bytes that are
;; not UTF-8 at all are passed through — there is nothing better to do with them,
;; and they decode exactly as they would at the end of a whole generation.
(define (utf8-complete-prefix bv end)
  (let ((floor (max 0 (- end 4))))
    (let loop ((i (- end 1)))
      (if (< i floor)
          end
          (let ((b (bytevector-u8-ref bv i)))
            (if (< b 128)                          ; ASCII byte: a boundary here
                (+ i 1)
                (if (< b 192)                      ; continuation byte: look back
                    (loop (- i 1))
                    (let ((need (cond ((< b 224) 2) ((< b 240) 3) (else 4))))
                      (if (<= (+ i need) end) end i)))))))))

(define (emit-upto model tokens emitted proc token)
  "Detokenize TOKENS and hand PROC the bytes past EMITTED that end on a character
boundary. Returns the new emitted length."
  (let* ((whole (detokenize-bytes model tokens))
         (boundary (utf8-complete-prefix whole (bytevector-length whole))))
    (if (> boundary emitted)
        (begin
          (proc token (utf8->string (bytevector-range whole emitted boundary)))
          boundary)
        emitted)))

(define* (generate-stream model session prompt max-tokens proc #:optional (prefill? #t)
                          #:key
                          (temperature 0.0)
                          (top-k 0)
                          (top-p 1.0)
                          (min-p 0.0)
                          (seed 0)
                          (repeat-penalty 1.0)
                          (repeat-last-n 64)
                          (presence-penalty 0.0)
                          (frequency-penalty 0.0))
  "Generate tokens while delivering the text as it becomes available.

Calls (proc token chunk) once per produced token, in order, on the calling
thread. Chunk is the text that became available with that token and may be
empty; chunks never split a UTF-8 character. A final call with token #f
flushes any bytes held back at the end. Concatenating every chunk in order
gives exactly the string generate returns for the same prompt.
Args:
  model: a model handle.
  session: a session handle.
  prompt: the prompt as a list of token ids.
  max-tokens: the maximum number of tokens to produce.
  proc: the per-token procedure (proc token chunk).
  prefill?: when false, decode from the prompt the session already holds.
  temperature, top-k, top-p, min-p, seed, repeat-penalty, repeat-last-n,
  presence-penalty, frequency-penalty: sampling parameters, as in
  make-sampling-bv.
Returns: the list of produced token ids.
PyTorch: none
Note: An exception raised by proc stops the decode and is re-thrown by
generate-stream."
  (when prefill? (prefill! session prompt))
  (let* ((tokens '())          ; reversed: as produced
         (emitted 0)           ; bytes of the running text already handed out
         (failure #f)          ; (key . args) from PROC, re-thrown after decode
         (cb (lambda (user token)
               (set! tokens (cons token tokens))
               (catch #t
                 (lambda ()
                   (set! emitted (emit-upto model (reverse tokens) emitted proc token))
                   0)                 ; 0 = keep decoding
                 (lambda (key . args)
                   (set! failure (cons key args))
                   1))))              ; non-zero = stop the decode
         (cb-ptr (procedure->pointer int cb (list '* int32)))
         (smp-bv (make-sampling-bv max-tokens
                                   #:temperature temperature
                                   #:top-k top-k
                                   #:top-p top-p
                                   #:min-p min-p
                                   #:seed seed
                                   #:repeat-penalty repeat-penalty
                                   #:repeat-last-n repeat-last-n
                                   #:presence-penalty presence-penalty
                                   #:frequency-penalty frequency-penalty))
         (n-slot (make-bytevector 4 0))
         (err-slot (make-slot))
         (st ((sb-fn 'boom_llm_decode) session (bytevector->pointer smp-bv)
              cb-ptr %null-pointer (bytevector->pointer n-slot)
              (slot-ptr err-slot))))
    (if failure
        (apply throw failure)
        (sb-check! st err-slot "decode"))
    ;; Flush: bytes held back only because their sequence was incomplete.
    (set! emitted (emit-upto model (reverse tokens) emitted proc #f))
    (reverse tokens)))
