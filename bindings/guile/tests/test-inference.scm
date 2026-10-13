;;; SonicBoom — Guile inference binding golden-token test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Loads the Gemma 4 GGUF through (sonicboom inference), seeds token 2, and
;;; greedily decodes 4 tokens, asserting they match the C++ spine golden decode
;;; stream {236761 107 236769 236776} (the same tokens test_generate checks).
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
      (test-skip 1)
      (test-equal "generate reproduces the golden decode stream"
        '(236761 107 236769 236776)
        (let* ((model (load-model path))
               (sess  (make-session model 512))
               (out   (generate sess '(2) 4)))
          (free-session! sess)
          (free-model! model)
          out))))

(test-end "sonicboom-inference")
