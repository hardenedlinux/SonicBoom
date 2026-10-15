;;; SonicBoom — Guile inference binding test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Loads the Gemma 4 GGUF through (sonicboom inference) and checks four
;;; things:
;;;
;;;   1. generate-tokens (the token-ID entry point) seeds token 2 and greedily
;;;      decodes 4 tokens, matching the C++ spine golden decode stream
;;;      {236761 107 236769 236776} (the same tokens test_generate checks).
;;;   2. generate (string -> string) tokenizes, decodes, and detokenizes back to
;;;      a non-empty string.
;;;   3. tokenize / detokenize round-trip through the two-call C ABI sizing path.
;;;   4. prefix-reuse bookkeeping on one session: a fresh first prefill reports
;;;      0 reused; extending by 100 new tokens reports 4000; a prompt whose
;;;      first token already differs reports 0.
;;;   5. generate-stream: one callback per produced token, chunks that
;;;      concatenate to exactly generate's output, the same token ids as
;;;      generate-tokens, and a callback exception that stops the decode.
;;;
;;; Skips cleanly when the GGUF is absent, so the suite is green on machines
;;; without the model. The generate-stream chunk-boundary checks need no model
;;; and always run.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom inference))

(define (find-model)
  (let ((cands
         (filter (lambda (p) (and p (file-exists? p)))
                 (list
                  (getenv "SONICBOOM_GEMMA_GGUF")
                  (and=> (getenv "SONICBOOM_SRC_DIR")
                         (lambda (d)
                           (string-append d
                             "/models/gemma-4-E4B-it-Q3_K_M.gguf")))
                  "models/gemma-4-E4B-it-Q3_K_M.gguf"
                  "../../models/gemma-4-E4B-it-Q3_K_M.gguf"))))
    (and (pair? cands) (car cands))))

(test-begin "sonicboom-inference")

(let ((path (find-model)))
  (if (not path)
      (test-skip 11)
      (let* ((model (load-model path))
             (sess1 (make-session model 512)))
        (test-equal "generate-tokens reproduces the golden decode stream"
          '(236761 107 236769 236776)
          (generate-tokens sess1 '(2) 4))
        (free-session! sess1)

        (let ((sess2 (make-session model 512)))
          (test-assert "generate (string -> string) returns text"
            (let ((s (generate model sess2 "Hello" 4)))
              (and (string? s) (> (string-length s) 0))))
          (free-session! sess2))

        ;; Streaming. One decode per session: generate-stream's own run, and a
        ;; separate generate / generate-tokens run of the same prompt for the
        ;; reference (greedy decode is deterministic, so they see the same
        ;; tokens).
        (let* ((prompt (tokenize model "Hello"))
               (chunks '())
               (sess (make-session model 512))
               (ids (generate-stream model sess prompt 8
                                     (lambda (tok chunk)
                                       (set! chunks (cons (cons tok chunk) chunks))))))
          (free-session! sess)
          (let ((ref-tokens (make-session model 512))
                (ref-text (make-session model 512)))
            (test-equal "generate-stream produces the same token ids as generate-tokens"
              (generate-tokens ref-tokens prompt 8)
              ids)
            (test-equal "the concatenated chunks are exactly generate's output"
              (generate model ref-text "Hello" 8)
              (apply string-append (map cdr (reverse chunks))))
            (free-session! ref-tokens)
            (free-session! ref-text))
          ;; A final flush call (token #f) only happens when the last token left
          ;; a character half-arrived; for this prompt there is none, so the
          ;; callback saw exactly one call per token, in order.
          (test-equal "the callback sees exactly one call per produced token, in order"
            ids (map car (reverse chunks)))
          ;; A callback exception must stop the decode and come back out here,
          ;; rather than unwinding through the decoder's frame.
          (let ((seen 0)
                (sess2 (make-session model 512)))
            (test-assert "an exception from the callback stops the decode and propagates"
              (catch #t
                (lambda ()
                  (generate-stream model sess2 prompt 20
                                   (lambda (tok chunk)
                                     (set! seen (+ seen 1))
                                     (when (>= seen 3) (error "stop here"))))
                  #f)
                (lambda (key . args) #t)))
            (test-equal "the decode stopped at the failing token" 3 seen)
            (free-session! sess2)))

        (test-equal "tokenize / detokenize round-trip"
          "Hello, world!"
          (detokenize model (tokenize model "Hello, world!")))

        ;; Prefix-reuse bookkeeping on one session: a fresh first prefill reports
        ;; 0 reused; extending by 100 new tokens reports the shared 4000-token
        ;; prefix; a prompt whose first token already differs reports 0. n_ctx is
        ;; sized for the full 4100-token prompt.
        (let ((sess3 (make-session model 8192)))
          (let ((p4000 (list-tabulate 4000 (lambda (i) (+ 2 (modulo i 500)))))
                (p4100 (list-tabulate 4100 (lambda (i) (+ 2 (modulo i 500))))))
            (test-equal "first prefill on a fresh session reports 0 reused"
              0
              (prefill! sess3 p4000))
            (test-equal "second prefill reports the 4000-token prefix reused"
              4000
              (prefill! sess3 p4100))
            (test-equal "prefill with a different first token reports 0 reused"
              0
              (prefill! sess3 '(3 4 5))))
          (free-session! sess3))

        (free-model! model))))

;;; --- chunk boundaries, without a model --------------------------------------
;;; generate-stream holds a chunk back until its last character has fully
;;; arrived, which is the whole point of the buffering. No natural prompt above
;;; happened to split a character across two tokens, so the boundary rule itself
;;; is checked directly — it is the one piece of streaming that can be pinned
;;; down exactly. (Reached through the module, since it is an internal rule, not
;;; part of the inference API.)

(let ((prefix (module-ref (resolve-module '(sonicboom inference))
                          'utf8-complete-prefix)))
  (test-equal "an empty buffer emits nothing"
    0 (prefix (make-bytevector 0 0) 0))
  (test-equal "ASCII ends on a boundary"
    2 (prefix #vu8(104 105) 2))
  (test-equal "a pending 2-byte lead is held back"
    1 (prefix #vu8(104 195) 2))
  (test-equal "a completed 2-byte character is emitted"
    3 (prefix #vu8(104 195 169) 3))
  (test-equal "a completed 3-byte character is emitted"
    3 (prefix #vu8(226 130 172) 3))
  (test-equal "a pending 4-byte lead is held back"
    1 (prefix #vu8(10 240 159) 3))
  (test-equal "a completed 4-byte character is emitted"
    5 (prefix #vu8(10 240 159 152 128) 5))
  (test-equal "ASCII followed by a partial character emits only the ASCII"
    3 (prefix #vu8(97 98 99 226 130) 5)))

(test-end "sonicboom-inference")
