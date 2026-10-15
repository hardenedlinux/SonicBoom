;;; SonicBoom — (sonicboom nn) layers are traceable: one model, two routes
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-nn-trace.scm
;;;
;;; A `sequential` model built from the ordinary layer constructors can be run
;;; two ways: `model-forward`, which emits S-Expr text from the layer structure,
;;; and `model-trace-forward`, which traces the same layers as symbolic ops and
;;; lowers the resulting graph back onto the runtime's node vocabulary. Both
;;; drive libsonicboom.so over the same parameter tensors, so the two routes must
;;; agree (float32, relative error <= 1e-4). The trace must also produce the same
;;; node sequence the emitted text does: a `linear` is a `gemm`.

(use-modules (srfi srfi-1)
             (srfi srfi-13)
             (srfi srfi-64)
             (sonicboom tensor)
             (sonicboom nn)
             (sonicboom trace lower))

;;; --- a deterministic 2-layer MLP ----------------------------------------------

;; Fill a tensor with small deterministic values: non-zero, non-uniform, so the
;; comparison below cannot pass by both routes returning zeros.
(define (fill! t)
  (let ((n (tensor-numel t)))
    (let loop ((i 0))
      (when (< i n)
        (tensor-set! t i (* 0.125 (- (modulo (* 7 (+ i 3)) 11) 5)))
        (loop (+ i 1)))))
  t)

(define (set-model-weights! m)
  (for-each (lambda (p) (fill! (cdr p))) (model-parameters m))
  m)

(define mlp
  (set-model-weights!
   (sequential (linear 4 8 #:init 'zeros #:name 'l1)
               (relu #:name 'act)
               (linear 8 3 #:init 'zeros #:name 'l2))))

;; A model whose linear has no bias exercises the trace's matmul-without-add path.
(define no-bias
  (set-model-weights!
   (sequential (linear 4 3 #:init 'zeros #:bias #f #:name 'w1))))

(define x (list->tensor '(0.5 -1.25 2.0 0.75 -0.5 1.5 -0.25 3.0) '(2 4)))

;;; --- helpers ------------------------------------------------------------------

(define (rel-max-err a b)
  "Largest |a-b| / max(1, |b|) over two tensors' flat elements."
  (let loop ((xs (tensor->list a)) (ys (tensor->list b)) (worst 0.0))
    (if (null? xs)
        worst
        (let* ((e (abs (- (car xs) (car ys))))
               (r (max 1.0 (abs (car ys)))))
          (loop (cdr xs) (cdr ys) (max worst (/ e r)))))))

(define (doc-fields doc)
  "The graph's field alist, from a traced document datum."
  (if (and (pair? doc) (eq? (car doc) 'sonicboom-s-expr))
      (cdr (assq 'graph (cdr doc)))
      (error "not an S-Expr document" doc)))

(define (lowered-ops model dims)
  (map (lambda (n) (cadr n))
       (cdr (assq 'nodes (doc-fields (call-with-input-string
                                      (traced->s-expr (model-trace model dims))
                                      read))))))

(define (traced-params model dims)
  (cdr (assq 'parameters (doc-fields (model-trace model dims)))))

;;; --- tests --------------------------------------------------------------------

(test-begin "sonicboom-nn-trace")

;; 1. the two routes agree, on a model with a relu between two linear layers
(let ((a (model-forward mlp x))
      (b (model-trace-forward mlp x)))
  (test-equal "model-trace-forward: output shape matches model-forward"
    (tensor-shape a) (tensor-shape b))
  (test-equal "model-trace-forward: dtype matches model-forward"
    (tensor-dtype a) (tensor-dtype b))
  (test-assert "model-trace-forward matches model-forward (rel err <= 1e-4)"
    (<= (rel-max-err b a) 1e-4))
  (test-assert "the model's output is not all zeros (the check is not vacuous)"
    (> (apply max (map abs (tensor->list a))) 0.1)))

;; 2. a linear without bias goes through the same two routes
(let ((a (model-forward no-bias x))
      (b (model-trace-forward no-bias x)))
  (test-assert "a bias-free linear traces to the same output (rel err <= 1e-4)"
    (<= (rel-max-err b a) 1e-4)))

;; 3. the traced model lowers to the same node sequence the emitted text has
(test-equal "the traced MLP lowers to gemm / relu / gemm"
  '(gemm relu gemm) (lowered-ops mlp '(2 4)))
(test-equal "a bias-free linear lowers to a single gemm"
  '(gemm) (lowered-ops no-bias '(2 4)))

;; 4. model-trace returns a well-formed S-Expr v0.1 document
(let* ((doc (model-trace mlp '(2 4)))
       (fields (doc-fields doc)))
  (test-assert "model-trace returns an S-Expr v0.1 document datum"
    (and (pair? doc) (eq? (car doc) 'sonicboom-s-expr)))
  (test-equal "the document declares version 0 1"
    '(version 0 1) (cadr doc))
  (test-equal "the graph has one input, one output and a node list"
    '(#t #t #t)
    (list (pair? (assq 'inputs fields))
          (pair? (assq 'outputs fields))
          (pair? (assq 'nodes fields))))
  (test-equal "the traced graph holds the tracer's ops (the lowering fuses them)"
    '(matmul add relu matmul add)
    (map cadr (cdr (assq 'nodes fields)))))

;; 5. every model parameter reaches the graph, under the tracer's own constant
;;    names (c0, c2, ...): the mapping is by shape and value, not by name. A
;;    linear's weight is laid out [out, in] by (sonicboom nn) and [in, out] by
;;    the tracer, so a traced weight is its transpose — the same values.
(define (traced-param-shapes model dims)
  (map (lambda (p) (cdr (caddr (caddr p)))) (traced-params model dims)))

(test-equal "one traced parameter per model parameter"
  (length (model-parameters mlp))
  (length (traced-params mlp '(2 4))))

(test-equal "each model parameter appears in the graph under the traced layout"
  '((3) (8) (3 8) (8 4))
  (sort (map (lambda (s) (if (= 2 (length s)) (reverse s) s))
             (traced-param-shapes mlp '(2 4)))
        (lambda (a b) (< (apply + a) (apply + b)))))

(test-assert "a traced rank-1 parameter carries the model's bias values"
  (let* ((want (cdr (assq 'l1.bias (model-parameters mlp))))
         (got (car (filter (lambda (p) (equal? '(shape 8) (caddr (caddr p))))
                           (traced-params mlp '(2 4))))))
    (and got
         (<= (rel-max-err (list->tensor (cddr (cadddr got)) '(8)) want) 1e-7))))

;; 6. the traced graph is reusable: tracing the same model twice gives the same
;;    document, and a second forward pass gives the same numbers
(test-assert "tracing the same model twice gives the same lowered text"
  (equal? (traced->s-expr (model-trace mlp '(2 4)))
          (traced->s-expr (model-trace mlp '(2 4)))))
(test-assert "a repeated traced forward pass gives the same output"
  (equal? (tensor->list (model-trace-forward mlp x))
          (tensor->list (model-trace-forward mlp x))))

;; 7. the existing text route is unchanged: the emitted graph still uses the
;;    model-parameter names for its parameters, and still runs
(test-assert "model-graph-text still emits one gemm per linear"
  (let ((text (model-graph-text mlp '(2 4))))
    (and (string-contains text "(node gemm")
         (string-contains text "l1.weight")
         (string-contains text "l2.bias"))))

(test-end "sonicboom-nn-trace")
