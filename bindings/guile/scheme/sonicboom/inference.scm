;;; SonicBoom — Guile binding: minimal inference surface
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The (sonicboom inference) module binds the minimal LLM C ABI (capi.h
;;; sb_llm_*) through (sonicboom ffi), which is the only path Guile uses to
;;; reach libsonicboom.so. It exposes load-model / make-session / generate — a
;;; raw-token-ID inference loop (no tokenizer, greedy decode) sufficient to
;;; reproduce the golden token stream against the C++ spine.
;;;
;;;   Guile → (sonicboom inference) → (sonicboom ffi) → libsonicboom.so
;;;
;;; Handles are foreign pointers. Freeing is explicit (free-model!/free-session!)
;;; so a test owns its lifetime deterministically; the session holds a C++
;;; shared_ptr to the model so a freed model handle does not dangle a live
;;; session. No tokenizer: prompts are lists of vocabulary indices.

(define-module (sonicboom inference)
  #:use-module (sonicboom ffi)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:export (load-model free-model!
            make-session free-session!
            prefill! generate))

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

;; sb_llm_sampling as a native-layout bytevector (x86-64; see capi.h):
;;   struct_size u64@0, temperature f32@8, top_p f32@12, top_k u32@16,
;;   seed u64@24, max_tokens u32@32  → 40 bytes.
;; Greedy only: temperature 0, top_p 1 (off), top_k 0 (off), seed 0.
(define (make-sampling-bv max-tokens)
  (let ((bv (make-bytevector 40 0)))
    (bytevector-u64-native-set! bv 0 40)
    (bytevector-ieee-single-native-set! bv 8 0.0)
    (bytevector-ieee-single-native-set! bv 12 1.0)
    (bytevector-u32-native-set! bv 16 0)
    (bytevector-u64-native-set! bv 24 0)
    (bytevector-u32-native-set! bv 32 max-tokens)
    bv))

;;; --- model ------------------------------------------------------------------

(define (load-model path)
  "Load a Gemma 4 GGUF and return an opaque model handle."
  (let* ((model-slot (make-slot))
         (err-slot (make-slot))
         (st ((proc 'sb_llm_model_load) (string->pointer path)
              (slot-ptr model-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "load-model")
    (slot-ref model-slot)))

(define (free-model! model)
  ((proc 'sb_llm_model_free) model))

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

(define (generate session prompt max-tokens)
  "Prefill PROMPT (a list of token ids) and greedily decode up to MAX-TOKENS
tokens. Returns the list of produced token ids."
  (prefill! session prompt)
  (let* ((tokens '())
         (cb (lambda (user token)
               (set! tokens (cons token tokens))
               0))                 ; 0 = continue
         (cb-ptr (procedure->pointer int cb (list '* int32)))
         (smp-bv (make-sampling-bv max-tokens))
         (n-slot (make-bytevector 4 0))
         (err-slot (make-slot))
         (st ((proc 'sb_llm_decode) session (bytevector->pointer smp-bv)
              cb-ptr %null-pointer (bytevector->pointer n-slot)
              (slot-ptr err-slot))))
    (sb-check! st err-slot "decode")
    (reverse tokens)))
