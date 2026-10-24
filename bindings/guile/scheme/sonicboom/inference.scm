;;; SonicBoom — Guile binding: minimal inference surface
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The (sonicboom inference) module binds the minimal LLM C ABI (capi.h
;;; sb_llm_*) through (sonicboom ffi), which is the only path Guile uses to
;;; reach libsonicboom.so. It exposes load-model / make-session / generate —
;;; a greedy-decode inference loop over the GGUF built-in tokenizer.
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

;;; The process-wide numeric contract (capi.h sb_set_numerics_profile): 'strict
;;; keeps every historical reduction, 'standard allows the profile's
;;; substitutions (today: f16 KV storage). It is latched at the first model or
;;; session, so call set-numerics-profile! first. Until either is called the
;;; process uses SONICBOOM_NUMERICS, then 'standard.

(define (numerics-profile)
  "The effective numerics profile: 'strict or 'standard."
  (let ((err (make-slot))
        (out (make-bytevector (sizeof int) 0)))
    (sb-check! ((proc 'sb_get_numerics_profile) (bytevector->pointer out)
                (slot-ptr err))
               err "numerics-profile")
    (case (bytevector-s32-native-ref out 0)
      ((0) 'strict)
      (else 'standard))))

(define (set-numerics-profile! profile)
  "Request PROFILE ('strict or 'standard) for this process and return the
effective profile. Signals 'sb-inference if the profile is already latched to
the other value; setting the effective profile again is a no-op success."
  (let ((code (case profile
                ((strict) 0)
                ((standard) 1)
                (else (throw 'sb-inference
                             (string-append "unknown numerics profile: "
                                            (object->string profile))))))
        (err (make-slot)))
    (sb-check! ((proc 'sb_set_numerics_profile) code (slot-ptr err))
               err "set-numerics-profile!")
    (numerics-profile)))

;;; --- helpers ----------------------------------------------------------------

(define (sb-check! st err-slot what)
  (unless (sb-status-ok? st)
    (let ((msg (capture-error err-slot)))
      (throw 'sb-inference
             (string-append what ": " msg)))))

;; A list of token ids → a native-endian int32 bytevector (sb_token[]).
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

;; sb_llm_sampling as a native-layout bytevector (x86-64; see capi.h):
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
  "Load a GGUF (a model family SonicBoom supports) and return an opaque model
handle."
  (let* ((model-slot (make-slot))
         (err-slot (make-slot))
         (st ((proc 'sb_llm_model_load) (string->pointer path)
              (slot-ptr model-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "load-model")
    (slot-ref model-slot)))

(define (free-model! model)
  ((proc 'sb_llm_model_free) model))

;; The model's architecture, as its GGUF declares it in general.architecture
;; ("gemma4", "qwen2", ...). Who formats the prompt reads this from the model
;; rather than guessing the family from the file name.
(define (model-arch model)
  (let* ((len-slot (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot)))
    ;; Sizing call (cap 0): fills *len_out (the length, no NUL).
    (let ((st ((proc 'sb_llm_model_arch) model %null-pointer 0
               (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st SB_ERR_BINDING))
        (sb-check! st err-slot "model-arch")))
    (let* ((m (bytevector-u64-native-ref len-slot 0))
           (out (make-bytevector (+ m 1) 0))    ; +1 for the NUL terminator
           (st ((proc 'sb_llm_model_arch) model (bytevector->pointer out) (+ m 1)
                (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (sb-check! st err-slot "model-arch")
      (utf8->string (bytevector-range out 0 m)))))

;; The model's own end-of-sequence token ids (a possibly empty list), straight
;; from its metadata: what generation has to stop on, per model rather than per
;; family.
(define (model-eos model)
  (let* ((n-slot (make-bytevector 8 0))        ; size_t
         (err-slot (make-slot)))
    ;; Sizing call (cap 0): fills *n_out (the count).
    (let ((st ((proc 'sb_llm_model_eos) model %null-pointer 0
               (bytevector->pointer n-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st SB_ERR_BINDING))
        (sb-check! st err-slot "model-eos")))
    (let* ((n (bytevector-u64-native-ref n-slot 0))
           (bv (make-bytevector (* n 4) 0))
           (st ((proc 'sb_llm_model_eos) model (bytevector->pointer bv) n
                (bytevector->pointer n-slot) (slot-ptr err-slot))))
      (sb-check! st err-slot "model-eos")
      (s32-bv->list bv n))))

;;; --- session ----------------------------------------------------------------

(define (make-session model n-ctx)
  "Create a session (token history + context window) over MODEL."
  (let* ((session-slot (make-slot))
         (err-slot (make-slot))
         (st ((proc 'sb_llm_session_new) model n-ctx
              (slot-ptr session-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "make-session")
    (slot-ref session-slot)))

(define (free-session! session)
  ((proc 'sb_llm_session_free) session))

;;; --- constrained decoding ---------------------------------------------------

;;; A session may carry a grammar (capi.h "Constrained decoding"): every sampled
;;; token must keep its text acceptable, and the model's EOS ids end the output
;;; where the grammar can end. The grammar restarts at every prefill, so install
;;; it before prefill! — that is what masks the turn's first token too.

(define (set-grammar! session gbnf)
  "Constrain SESSION's next turn to GBNF (a llama.cpp-dialect grammar)."
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'sb_llm_session_set_grammar) session
                (string->pointer gbnf) (slot-ptr err-slot))
               err-slot "set-grammar!")))

(define (set-json-schema! session schema-json)
  "Constrain SESSION's next turn to the JSON Schema SCHEMA-JSON. An unsupported
keyword is rejected here by name, not silently ignored."
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'sb_llm_session_set_json_schema) session
                (string->pointer schema-json) (slot-ptr err-slot))
               err-slot "set-json-schema!")))

(define (clear-grammar! session)
  "Remove SESSION's grammar: the next turn decodes unconstrained."
  ((proc 'sb_llm_session_clear_grammar) session))

(define (constraint-state session)
  "Two values: whether SESSION has a grammar installed, and whether the text
sampled so far is already a complete value of it."
  (let* ((active (make-bytevector (sizeof int) 0))
         (complete (make-bytevector (sizeof int) 0))
         (err-slot (make-slot)))
    (sb-check! ((proc 'sb_llm_session_constraint_state) session
                (bytevector->pointer active) (bytevector->pointer complete)
                (slot-ptr err-slot))
               err-slot "constraint-state")
    (values (not (zero? (bytevector-s32-native-ref active 0)))
            (not (zero? (bytevector-s32-native-ref complete 0))))))

(define* (json-schema->gbnf schema-json #:key (prefix "") (suffix ""))
  "Compile SCHEMA-JSON to GBNF text. With PREFIX/SUFFIX, the schema value must
appear between those literal texts — how the server builds a tool-call envelope
around a function's arguments schema."
  (let* ((len-slot (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot))
         (head (list (string->pointer schema-json)
                     (string->pointer prefix) (string->pointer suffix))))
    ;; Sizing call (cap 0): fills *len_out (the length, no NUL).
    (let ((st (apply (proc 'sb_llm_json_schema_to_gbnf)
                     (append head
                             (list %null-pointer 0
                                   (bytevector->pointer len-slot)
                                   (slot-ptr err-slot))))))
      (unless (or (sb-status-ok? st) (= st SB_ERR_BINDING))
        (sb-check! st err-slot "json-schema->gbnf")))
    (let* ((m (bytevector-u64-native-ref len-slot 0))
           (out (make-bytevector (+ m 1) 0))    ; +1 for the NUL terminator
           (st (apply (proc 'sb_llm_json_schema_to_gbnf)
                      (append head
                              (list (bytevector->pointer out) (+ m 1)
                                    (bytevector->pointer len-slot)
                                    (slot-ptr err-slot))))))
      (sb-check! st err-slot "json-schema->gbnf")
      (utf8->string (bytevector-range out 0 m)))))

;;; --- tokenize / detokenize --------------------------------------------------

;; sb_llm_tokenize_ex flags, from capi.h.
(define sb-tokenize-parse-special 1)
(define sb-tokenize-add-special 2)

;; The two-call sizing path over sb_llm_tokenize_ex: TEXT with the given FLAGS.
(define (tokenize-flags model text flags)
  (let* ((bv (string->utf8 text))
         (len (bytevector-length bv))
         (n-slot (make-bytevector 8 0))          ; size_t
         (err-slot (make-slot))
         (fn (proc 'sb_llm_tokenize_ex))
         (head (list model (bytevector->pointer bv) len flags)))
    ;; Sizing call (cap 0): fills *n_out, returns SB_ERR_BINDING.
    (let ((st (apply fn (append head
                                (list %null-pointer 0
                                      (bytevector->pointer n-slot)
                                      (slot-ptr err-slot))))))
      (unless (or (sb-status-ok? st) (= st SB_ERR_BINDING))
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
  "Encode TEXT (a UTF-8 string) into a list of token ids (<bos> prepended,
  matching llama-tokenize).

  With #:special? #t, control pieces of the model's vocabulary (e.g. <|turn>,
  <turn|>) that occur verbatim in TEXT become their own token instead of being
  BPE'd as text — what a chat template's turn markers need. It is off by
  default: special parsing turns text into control structure, so it must only be
  used on text the caller generated itself, never on a user's message."
  (tokenize-flags model text
                  (logior sb-tokenize-add-special
                               (if special? sb-tokenize-parse-special 0))))

(define (tokenize-segments model segments)
  "Tokenize a rendered chat prompt given as SEGMENTS: a list of (kind . text)
with kind 'text or 'special (see (sonicboom serve chat-template)). A 'special
segment is tokenized with special parsing, so a turn marker spelled out in it
becomes the vocabulary's one control token; a 'text segment is tokenized as
plain text, so a marker that came from a user's message stays text. <bos> is
added once, by the first segment's call."
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
        (let ((st ((proc 'sb_llm_detokenize) model (bytevector->pointer in) n
                   %null-pointer 0 (bytevector->pointer len-slot)
                   (slot-ptr err-slot))))
          (unless (or (sb-status-ok? st) (= st SB_ERR_BINDING))
            (sb-check! st err-slot "detokenize")))
        (let* ((m (bytevector-u64-native-ref len-slot 0))
               (out (make-bytevector (+ m 1) 0))     ; +1 for the NUL terminator
               (st ((proc 'sb_llm_detokenize) model (bytevector->pointer in) n
                     (bytevector->pointer out) (+ m 1) (bytevector->pointer len-slot)
                     (slot-ptr err-slot))))
          (sb-check! st err-slot "detokenize")
          ;; Copy exactly the m text bytes the C ABI wrote (it also NUL-terminated
          ;; out[m], which is why cap is m+1).
          (bytevector-range out 0 m)))))

(define (detokenize model tokens)
  "Decode TOKENS (a list of token ids) back to a UTF-8 string."
  (utf8->string (detokenize-bytes model tokens)))

;;; --- prefill / generate -----------------------------------------------------

(define (prefill! session tokens)
  "Set SESSION's prompt to TOKENS (a list of token ids). Returns the number of
tokens reused from the previous prompt (informational)."
  (let* ((bv (tokens->bv tokens))
         (n (length tokens))
         (reused-slot (make-bytevector 8 0))
         (err-slot (make-slot))
         (st ((proc 'sb_llm_prefill) session (bytevector->pointer bv) n
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
  "Prefill PROMPT (a list of token ids) and decode up to MAX-TOKENS tokens.
Returns the list of produced token ids. When PREFILL? is false, decode from the
prompt already held by SESSION.

The sampling keywords are those of make-sampling-bv: temperature 0 (the
default) is greedy and bit-identical to the historical path; temperature > 0
samples deterministically from SEED (see sb_llm_sampling in capi.h)."
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
         (st ((proc 'sb_llm_decode) session (bytevector->pointer smp-bv)
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
  "Tokenize PROMPT (a UTF-8 string), prefill, decode up to MAX-TOKENS tokens,
and detokenize. Returns the generated text. The sampling keywords are passed
through to generate-tokens; the default temperature 0 is greedy."
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
;;; (sb_llm_decode's sb_llm_token_cb), so a per-token stream only needs the text
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

(define* (generate-stream model session prompt max-tokens proc #:optional (prefill? #t))
  "Prefill PROMPT (a list of token ids) and greedily decode up to MAX-TOKENS
tokens, calling (PROC TOKEN CHUNK) once per produced token, in order. CHUNK is
the text that became available with that token; it is empty when the token only
completed a character started earlier. Chunks never split a UTF-8 sequence, and
concatenated in order they are exactly the string `generate` returns for the same
prompt. PROC runs on the calling thread; a final call with TOKEN #f flushes any
bytes still held back at the end of the decode (normally none), so concatenating
every CHUNK including that one is the whole text. Returns the list of produced
token ids.

An exception raised by PROC stops the decode and is re-thrown here, rather than
unwinding through the decoder's stack frame."
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
         (smp-bv (make-sampling-bv max-tokens))
         (n-slot (make-bytevector 4 0))
         (err-slot (make-slot))
         (st ((sb-fn 'sb_llm_decode) session (bytevector->pointer smp-bv)
              cb-ptr %null-pointer (bytevector->pointer n-slot)
              (slot-ptr err-slot))))
    (if failure
        (apply throw failure)
        (sb-check! st err-slot "decode"))
    ;; Flush: bytes held back only because their sequence was incomplete.
    (set! emitted (emit-upto model (reverse tokens) emitted proc #f))
    (reverse tokens)))
