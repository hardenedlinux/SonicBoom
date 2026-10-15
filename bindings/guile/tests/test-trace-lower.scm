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

(define (doc-fields doc)
  "The graph's field alist, from the lowered text or the traced datum."
  (match doc
    (('sonicboom-s-expr _ ('graph . fields)) fields)
    (_ (error "not an S-Expr document" doc))))

(define (lowered-ops graph)
  (map cadr (cdr (assq 'nodes (doc-fields (call-with-input-string
                                           (traced->s-expr graph) read))))))

;; Run PROC on X through both paths and check shapes and numerics.
(define (check-lowering name proc x)
  (let* ((graph (trace-graph proc x))
         (in-name (doc-input-name graph))
         (expected (cdar (interpret-sexpr graph (list (cons in-name x)))))
         (actual (run-traced graph (list (cons in-name x)))))
    (test-equal (string-append name ": shape") (ctensor-shape expected)
                (ctensor-shape actual))
    (test-assert (string-append name ": run-traced matches the reference")
      (<= (rel-max-err actual expected) 1e-4))))

;;; --- lowering + execution ------------------------------------------------------

(test-begin "sonicboom-trace-lower")

(define graph (trace-graph mlp x))

(test-equal "linear is fused into a single gemm; relu survives"
  '(gemm relu gemm)
  (lowered-ops graph))

(let* ((in-name (doc-input-name graph))
       (expected (cdar (interpret-sexpr graph (list (cons in-name x)))))
       (actual (run-traced graph (list (cons in-name x)))))
  (test-equal "output shape matches the reference evaluation"
    (ctensor-shape expected) (ctensor-shape actual))
  (test-assert "run-traced matches the pure-Guile reference (rel err <= 1e-4)"
    (<= (rel-max-err actual expected) 1e-4))
  (test-equal "dtype is float32" 'float32 (ctensor-dtype actual)))

;;; --- reshape -------------------------------------------------------------------
;;; A reshape that produces the graph output directly from the graph input
;;; bufferizes to a view of that input; the runtime frees the returned pointer,
;;; which IS the input buffer, and corrupts the heap. That is a core bug (see
;;; NEEDS-CORE in the report), so the executed checks give the reshape an
;;; intermediate operand, which is the normal case inside a graph.

(define m23 (make-ctensor 'float32 '(2 3) '(1.0 2.0 3.0 4.0 5.0 6.0)))

(check-lowering "reshape [2 3] -> [3 2]" (lambda (x) (reshape (relu x) '(3 2))) m23)
(check-lowering "reshape [2 3] -> [1 6]" (lambda (x) (reshape (relu x) '(1 6))) m23)

(test-equal "reshape lowers to relu + reshape"
  '(relu reshape) (lowered-ops (trace-graph (lambda (x) (reshape (relu x) '(3 2))) m23)))

(test-equal "a reshape of the input still lowers to a single reshape node"
  '(reshape) (lowered-ops (trace-graph (lambda (x) (reshape x '(3 2))) m23)))

;;; --- reduce_mean ---------------------------------------------------------------

(define m24 (make-ctensor 'float32 '(2 4) '(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)))

(check-lowering "mean axes=(1) keepdims"
  (lambda (x) (mean x #:axes '(1) #:keepdims #t)) m24)
(check-lowering "mean axes=(1)"
  (lambda (x) (mean x #:axes '(1))) m24)
(check-lowering "mean axes=(0)"
  (lambda (x) (mean x #:axes '(0))) m24)
(check-lowering "mean axes=(0 1) keepdims"
  (lambda (x) (mean x #:axes '(0 1) #:keepdims #t)) m24)
(check-lowering "mean negative axis"
  (lambda (x) (mean x #:axes '(-1))) m24)
(check-lowering "mean of an intermediate value (relu output)"
  (lambda (x) (mean (relu x) #:axes '(1))) m24)

(define w43 (make-ctensor 'float32 '(4 3)
                          '(0.10 -0.20 0.30 -0.40 0.50 -0.60
                            0.70 -0.80 0.90 -1.00 0.11 0.12)))
(check-lowering "keepdims mean feeding a linear"
  (lambda (x) (linear (mean x #:axes '(0) #:keepdims #t) w43 b2)) m24)

(test-equal "keepdims mean lowers to one reduce_mean; axes is a parameter"
  '(reduce_mean)
  (lowered-ops (trace-graph (lambda (x) (mean x #:axes '(1) #:keepdims #t)) m24)))

(test-equal "rank-lowering mean lowers to reduce_mean + reshape"
  '(reduce_mean reshape)
  (lowered-ops (trace-graph (lambda (x) (mean x #:axes '(1))) m24)))

;; A whole-tensor reduction has a rank-0 result, and the core's reshape cannot
;; express one (its shape operand would be a 0-element tensor), so this lowers
;; correctly but cannot execute yet — NEEDS-CORE, see the report.
(test-equal "a whole-axis reduction lowers to reduce_mean + reshape"
  '(reduce_mean reshape)
  (lowered-ops (trace-graph (lambda (x) (mean x)) m24)))

;; The runtime takes no axes attribute: it must come in as a rank-1 int64
;; :values parameter, so the lowering synthesises one per reduction.
(define (axes-params text)
  (filter (lambda (p) (string-suffix? ".axes" (cadr p)))
          (cdr (assq 'parameters (doc-fields (call-with-input-string text read))))))

(let ((ps (axes-params (traced->s-expr
                        (trace-graph (lambda (x) (mean x #:axes '(1))) m24)))))
  (test-equal "one axes parameter is synthesised per reduction" 1 (length ps))
  (test-equal "the parameter is a `(parameter <name> <type> <data>)` form"
    'parameter (car (car ps)))
  (test-assert "the axes parameter is rank-1 int64"
    (equal? (caddr (car ps)) '(tensor int64 (shape 1))))
  (test-equal "the axes parameter carries the reduced axes"
    '(data :values 1) (cadddr (car ps))))

(test-equal "an all-axes reduction synthesises the full axis list"
  '(data :values 0 1)
  (cadddr (car (axes-params (traced->s-expr
                             (trace-graph (lambda (x) (mean x)) m24))))))

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

(test-error "a second unmapped op (exp) also raises, not just mul"
  'misc-error
  (traced->s-expr (trace-graph (lambda (x) (texp x)) m23)))

(test-end "sonicboom-trace-lower")
