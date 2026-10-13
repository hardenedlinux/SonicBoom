;;; SonicBoom — Guile inference binding test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Loads the Gemma 4 GGUF through (sonicboom inference) and checks three
;;; things:
;;;
;;;   1. generate-tokens (the token-ID entry point) seeds token 2 and greedily
;;;      decodes 4 tokens, matching the C++ spine golden decode stream
;;;      {236761 107 236769 236776} (the same tokens test_generate checks).
;;;   2. generate (string -> string) tokenizes, decodes, and detokenizes back to
;;;      a non-empty string.
;;;   3. tokenize / detokenize round-trip through the two-call C ABI sizing path.
;;;
;;; Skips cleanly when the GGUF is absent, so the suite is green on machines
;;; without the model.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
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
      (test-skip 3)
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

        (test-equal "tokenize / detokenize round-trip"
          "Hello, world!"
          (detokenize model (tokenize model "Hello, world!")))

        (free-model! model))))

(test-end "sonicboom-inference")
