;;; SonicBoom — Guile e2e exporter (the training half of the Guile → C++ loop)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Trains a small 2→4→1 MLP on XOR through the native-torch autograd surface,
;;; writes the trained model's predictions to <dir>/expected.bin, and exports the
;;; model + trained weights to <dir>/model.sx + <dir>/weights.bin via
;;; sb_export_model. The independent C++ loader (tests/core/test_e2e_load.cpp)
;;; then loads the artifact and must reproduce expected.bin.
;;;
;;; Usage: test-e2e-export.scm <dir>

(use-modules (srfi srfi-64)
             (rnrs io ports)
             (sonicboom nn)
             (sonicboom model)
             (sonicboom train))

(define dir
  (if (>= (length (command-line)) 2)
      (cadr (command-line))
      (error "usage: test-e2e-export.scm <dir>")))

(test-begin "sonicboom-e2e-export")

;; XOR dataset in the order the C++ loader repeats: [0,0] [0,1] [1,0] [1,1].
(define x   (tensor-from-list '(4 2) '(0 0 0 1 1 0 1 1)))
(define tgt (tensor-from-list '(4 1) '(0 1 1 0)))

;; 2→4→1 MLP (relu hidden). Layer names are explicit so the trained parameter
;; names ("fc1.weight", …) are deterministic and match the graph text.
(define structure
  (sequential (linear 2 4 #:name 'fc1)
              (relu #:name 'act1)
              (linear 4 1 #:name 'fc2)))

;; Deterministic seed (same values the train-xor demo uses) so the run always
;; converges to the same trained weights.
(define m (make-model '(4 2) structure
                      #:init-values
                      '(("fc1.weight" . (0.6 -0.3 0.2 0.5 -0.5 0.4 0.3 -0.7))
                        ("fc1.bias"   . (0.1 -0.1 0.2 -0.2))
                        ("fc2.weight" . (0.5 -0.6 0.3 -0.4))
                        ("fc2.bias"   . (0.0)))))

(test-assert "make-model seeds four native parameters"
  (= (length (nn-model-params m)) 4))

(call-with-values
    (lambda () (model-train! m x tgt #:epochs 3000 #:lr 0.5))
  (lambda (initial final)
    (test-assert "XOR training lowers the loss below 0.05"
      (and (real? initial) (real? final)
           (> initial 0.0)
           (< final initial)
           (< final 0.05)))))

;; The trained model's forward outputs are the golden values the C++ loader must
;; reproduce through the independent S-Expr → MLIR path.
(define tape (make-tape))
(define pred (model-forward-native m x tape))
(define pred-list (tensor->list pred 4))

(let ((p (open-file (string-append dir "/expected.bin") "wb")))
  (put-bytevector p (list->f32-bv pred-list))
  (close-port p))

;; Export the graph + trained native weights to a self-contained artifact.
(model-export! m dir)

(test-assert "export wrote model.sx"
  (file-exists? (string-append dir "/model.sx")))
(test-assert "export wrote weights.bin"
  (file-exists? (string-append dir "/weights.bin")))

;; Release the native handles.
(tape-free! tape)
(tensor-free! pred)
(tensor-free! x) (tensor-free! tgt)
(for-each (lambda (p) (tensor-free! (cdr p))) (nn-model-params m))

(test-end "sonicboom-e2e-export")
