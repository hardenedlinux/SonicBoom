;;; SonicBoom — traced-graph lowering to S-Expr + execution through the core
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-trace-lower.scm
;;;
;;; A traced forward graph is lowered onto the S-Expr v0.1 runtime vocabulary
;;; and executed by libsonicboom.so; the result is compared with the tracer's
;;; pure-Scheme reference evaluation of the same graph (float32, relative error
;;; <= 1e-4). A graph using an op with no lowering must raise.

(use-modules (srfi srfi-1)
             (srfi srfi-13)
             (srfi srfi-64)
             (ice-9 match)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace interp)
             (sonicboom trace layers)
             (sonicboom trace lower))

;;; --- 2-layer MLP: linear -> relu -> linear ------------------------------------

(define x (make-ctensor 'float32 '(2 4)
                        '(0.5 -1.25 2.0 0.75
                          -0.5 1.5 -0.25 3.0)))

(define w1 (make-ctensor 'float32 '(4 8)
                         '(0.10 -0.20 0.30 -0.40 0.50 -0.60 0.70 -0.80
                           0.21 0.22 -0.23 0.24 -0.25 0.26 0.27 -0.28
                           -0.31 0.32 0.33 -0.34 0.35 0.36 -0.37 0.38
                           0.41 -0.42 -0.43 0.44 0.45 -0.46 0.47 -0.48)))
(define b1 (make-ctensor 'float32 '(8)
                         '(0.05 -0.15 0.25 -0.35 0.45 -0.55 0.65 -0.75)))
(define w2 (make-ctensor 'float32 '(8 3)
                         '(0.11 0.12 -0.13
                           -0.14 0.15 0.16
                           0.17 -0.18 0.19
                           -0.21 0.22 -0.23
                           0.24 0.25 0.26
                           -0.27 -0.28 0.29
                           0.31 0.32 -0.33
                           -0.34 0.35 0.36)))
(define b2 (make-ctensor 'float32 '(3) '(0.01 -0.02 0.03)))

(define (mlp x) (linear (relu (linear x w1 b1)) w2 b2))

;;; --- helpers ------------------------------------------------------------------

(define (rel-max-err a b)
  "Largest |a-b| / max(1, |b|) over the two tensors' flat elements."
  (let loop ((xs (ctensor->list a)) (ys (ctensor->list b)) (worst 0.0))
    (if (null? xs)
        worst
        (let* ((e (abs (- (car xs) (car ys))))
               (r (max 1.0 (abs (car ys)))))
          (loop (cdr xs) (cdr ys) (max worst (/ e r)))))))

(define (doc-input-name doc)
  (cadr (car (cdr (assq 'inputs (cdr (caddr doc)))))))

;;; --- lowering + execution ------------------------------------------------------

(test-begin "sonicboom-trace-lower")

(define graph (trace-graph mlp x))

(test-equal "linear is fused into a single gemm; relu survives"
  '(gemm relu gemm)
  (match (call-with-input-string (traced->s-expr graph) read)
    (('sonicboom-s-expr _ ('graph . fields))
     (map cadr (cdr (assq 'nodes fields))))))

(let* ((in-name (doc-input-name graph))
       (expected (cdar (interpret-sexpr graph (list (cons in-name x)))))
       (actual (run-traced graph (list (cons in-name x)))))
  (test-equal "output shape matches the reference evaluation"
    (ctensor-shape expected) (ctensor-shape actual))
  (test-assert "run-traced matches the pure-Guile reference (rel err <= 1e-4)"
    (<= (rel-max-err actual expected) 1e-4))
  (test-equal "dtype is float32" 'float32 (ctensor-dtype actual)))

;;; --- an op with no S-Expr lowering must be reported ----------------------------

(define mul-graph
  (trace-graph (lambda (x) (mul x 2.0)) (make-ctensor 'float32 '(2 3)
                                                      '(1.0 2.0 3.0 4.0 5.0 6.0))))

(test-error "an unsupported op raises, naming the op"
  'misc-error (traced->s-expr mul-graph))

(test-assert "the unsupported-op error names the op"
  (let ((msg (catch 'misc-error
               (lambda () (traced->s-expr mul-graph) "")
               (lambda (key . args)
                 (call-with-output-string
                  (lambda (p) (apply display-error #f p args)))))))
    (string-contains msg "mul")))

(test-end "sonicboom-trace-lower")
