;;; SonicBoom — tracing prototype tests: same-trace constant identity dedup
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace-dedup.scm
;;;
;;; A concrete ctensor captured by a traced body becomes an inline graph
;;; parameter. Capturing the *same object* again in the same segment must reuse
;;; that parameter (one value list, one name), while distinct objects with equal
;;; contents stay distinct. The checks here also pin the consequences:
;;; parameter references resolve, results and gradients are unchanged, and the
;;; optimization is confined to one graph.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace autodiff)
             (sonicboom trace macro))

(test-begin "sonicboom-trace-dedup")

(define (t x) (list->ctensor x))
(define (stat f key) (assq-ref (trace-stats f) key))

;;; --- document helpers ---------------------------------------------------------

(define (doc-fields text)
  (match (call-with-input-string text read)
    (('sonicboom-s-expr _ ('graph . fields)) fields)
    (_ (error "not an S-Expr document" text))))

(define (segment-params text) (cdr (assq 'parameters (doc-fields text))))
(define (segment-nodes text) (cdr (assq 'nodes (doc-fields text))))
(define (node-inputs n) (cdr (assq 'inputs (cddr n))))
(define (all-inputs text) (append-map node-inputs (segment-nodes text)))

(define (param-names text) (map cadr (segment-params text)))

(define (close-list? a b)
  (and (= (length a) (length b))
       (every (lambda (x y)
                ;; float32 storage: relative tolerance at its precision
                (<= (abs (- x y)) (* 1e-5 (max 1.0 (abs x) (abs y)))))
              a b)))

(define (flat t) (ctensor->list t))

;;; =============================================================================
;;; One object captured several times -> one parameter
;;; =============================================================================

(define c (make-ctensor 'float32 '(2) '(1.5 -0.5)))

(define same-const
  (trace-function (lambda (x) (add (mul x c) c))))

(define x (t '(1.0 2.0)))

(let* ((out (same-const x))
       (text (cdar (trace-segments same-const)))
       (names (param-names text)))
  (test-equal "same object captured twice emits one parameter" 1 (length names))
  (test-assert "the repeated occurrence references the parameter's name"
    (>= (count (lambda (i) (equal? i (car names))) (all-inputs text)) 2))
  (test-equal "dedup does not change the result"
    '(3.0 -1.5) (map (lambda (v) (+ v 0.0)) (flat out)))
  (test-assert "dedup does not change the value list"
    (equal? (cddr (cadddr (car (segment-params text)))) '(1.5 -0.5))))

;;; =============================================================================
;;; Distinct objects with equal contents -> distinct parameters
;;; =============================================================================

(define c1 (make-ctensor 'float32 '(2) '(1.5 -0.5)))
(define c2 (make-ctensor 'float32 '(2) '(1.5 -0.5)))

(define two-consts
  (trace-function (lambda (x) (add (mul x c1) c2))))

(let* ((out (two-consts x))
       (names (param-names (cdar (trace-segments two-consts)))))
  (test-equal "distinct objects are not merged by value equality" 2 (length names))
  (test-assert "their parameter names differ" (not (equal? (car names) (cadr names))))
  (test-equal "independent objects still produce the right result"
    '(3.0 -1.5) (map (lambda (v) (+ v 0.0)) (flat out))))

;;; =============================================================================
;;; No sharing across traces
;;; =============================================================================

(define trace-a (trace-function (lambda (x) (add x c))))
(define trace-b (trace-function (lambda (x) (add x c))))

(trace-a x)
(trace-b x)

(let ((ta (cdar (trace-segments trace-a)))
      (tb (cdar (trace-segments trace-b))))
  (test-equal "each traced function has its own single parameter"
    '(1 1) (list (length (param-names ta)) (length (param-names tb))))
  (test-assert "two traces are two graphs"
    (not (eq? trace-a trace-b))))

;;; =============================================================================
;;; Gradients
;;; =============================================================================

;; d/dx sum(x*c + c) = c
(define vg (value-and-grad (lambda (x) (sum (add (mul x c) c)))))
(let* ((r (vg x))
       (loss (car r))
       (grad (cadr r)))
  (test-assert "dedup: loss unchanged"
    (<= (abs (- (ctensor-item loss) (+ 3.0 -1.5))) 1e-6))
  (test-equal "dedup: gradient is still c" '(1.5 -0.5)
    (map (lambda (v) (+ v 0.0)) (flat grad))))

;;; =============================================================================
;;; Rollback paths (host-loop switch) keep the index consistent with params
;;; =============================================================================

(define cvals '(1.5 -0.5))
(define (l-add xs ys) (map + xs ys))
(define (l-mul xs ys) (map * xs ys))
(define (l-iter a n) (if (zero? n) a (l-iter (l-add (l-mul a cvals) cvals) (- n 1))))

;; The walker (traced-lambda*) is what turns the named let into %loop; a plain
;; lambda handed to trace-function would recurse eagerly during tracing.
(define looped
  (traced-lambda* (#:max-unroll 4) (x)
    (let loop ((i 0) (a x))
      (if (< i 20) (loop (+ i 1) (add (mul a c) c)) a))))
(let ((out (looped x)))
  (test-assert "host-loop rollback keeps the result"
    (close-list? (l-iter '(1.0 2.0) 20) (flat out)))
  (test-assert "the loop really took the host-loop path"
    (positive? (stat looped 'host-loops))))

;;; =============================================================================

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-dedup")
(exit (if (zero? failures) 0 1))
