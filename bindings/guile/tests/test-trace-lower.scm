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

(define (doc-ops doc)
  (map cadr (cdr (assq 'nodes (doc-fields doc)))))

(define (lowered-ops graph)
  (doc-ops (call-with-input-string (traced->s-expr graph) read)))

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

;;; --- linear fusion: conditions and the conservative fallback -------------------

;; The matmul may be either operand of the commutative add.
(check-lowering "linear with the bias as the first add operand"
  (lambda (x) (add b1 (matmul x w1))) x)
(test-equal "bias first still lowers to a single gemm"
  '(gemm) (lowered-ops (trace-graph (lambda (x) (add b1 (matmul x w1))) x)))

;; A matmul read twice is not fused; both nodes stay, and the graph still runs
;; (the core add broadcasts the rank-1 bias).
(check-lowering "matmul with two consumers is not fused and still computes"
  (lambda (x)
    (let ((y (matmul x w1)))
      (add (relu y) (add y b1))))
  x)
(test-equal "a matmul with two consumers keeps both of its uses"
  '(gemm relu add add)
  (lowered-ops (trace-graph (lambda (x)
                              (let ((y (matmul x w1)))
                                (add (relu y) (add y b1))))
                            x)))

;; A matmul output that is also a graph output is observed outside the
;; graph: fusing would delete its only producer. The lowering must fall back
;; to the un-fused nodes, in a multi-output graph just as in a single-output
;; one.
(define (two-outputs x)
  (let ((y (matmul x w1)))
    (list y (add y b1))))

(define two-graph (trace-graph two-outputs x))

(define (doc-output-names doc)
  (map cadr (cdr (assq 'outputs (doc-fields doc)))))

;; A node is (node <op> (inputs ...) (outputs (<name> <type>) ...) (attrs ...)),
;; so CDDR is the three field forms and the outputs form lists (name type)
;; pairs. Same shape in the traced datum and in the text read back from
;; `traced->s-expr`.
(define (doc-node-output-names n)
  (map car (cdr (assq 'outputs (cddr n)))))

;; Every value the document defines, in order, as (kind . name): INPUT and
;; PARAMETER for the graph sections, the node's op for a node output.
(define (doc-definitions doc)
  (let ((fields (doc-fields doc)))
    (append (map (lambda (i) (cons 'input (cadr i)))
                 (cdr (assq 'inputs fields)))
            (map (lambda (p) (cons 'parameter (cadr p)))
                 (cdr (assq 'parameters fields)))
            (append-map (lambda (n)
                          (map (lambda (o) (cons (cadr n) o))
                               (doc-node-output-names n)))
                        (cdr (assq 'nodes fields))))))

(define (definitions-of defs name)
  (filter (lambda (d) (equal? (cdr d) name)) defs))

(define (duplicates xs)
  (let loop ((xs xs) (seen '()) (dups '()))
    (cond ((null? xs) dups)
          ((member (car xs) seen) (loop (cdr xs) seen (cons (car xs) dups)))
          (else (loop (cdr xs) (cons (car xs) seen) dups)))))

;; The traced graph still names the nodes matmul/add; `lowered-ops` below
;; reads the text `traced->s-expr` emits, so it must see gemm/add.
(test-equal "the traced graph uses matmul; only the lowered graph uses gemm"
  '(matmul add) (doc-ops two-graph))
(test-equal "a matmul output that is also a graph output is not fused"
  '(gemm add) (lowered-ops two-graph))

(define lowered-two (call-with-input-string (traced->s-expr two-graph) read))
(define two-out-names (doc-output-names lowered-two))
(define two-defs (doc-definitions lowered-two))

(test-equal "lowering preserves the graph's outputs in order"
  (doc-output-names two-graph) two-out-names)

(test-equal "every output of the lowered graph is defined exactly once"
  '() (filter (lambda (o) (not (= 1 (length (definitions-of two-defs o)))))
              two-out-names))
(test-equal "no value name in the lowered graph is defined twice"
  '() (duplicates (map cdr two-defs)))
(test-equal "the first output's producer is the gemm (the matmul)"
  '(gemm) (map car (definitions-of two-defs (car two-out-names))))
(test-equal "the second output's producer is the add (the bias add)"
  '(add) (map car (definitions-of two-defs (cadr two-out-names))))

;; The reference interpreter runs the traced graph (matmul + add), not the
;; lowered text: it checks the graph semantics that lowering must preserve.
;; No native execution is claimed for this pair: the S-Expr C ABI binds one
;; input and returns one output, so `run-traced` cannot run a two-output
;; graph (single-output fusion is executed natively above).
(let* ((in-name (doc-input-name two-graph))
       (outs (interpret-sexpr two-graph (list (cons in-name x))))
       (mm (matmul x w1))
       (sum (add mm b1)))
  (test-equal "the reference interpreter returns the traced graph's outputs in order"
    (doc-output-names two-graph) (map car outs))
  (test-assert "the first output is the matmul value"
    (<= (rel-max-err (cdar outs) mm) 1e-5))
  (test-assert "the second output is the bias-added value"
    (<= (rel-max-err (cdadr outs) sum) 1e-5)))

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
