;;; SonicBoom — tracing prototype tests: the optimizer's guarantees
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile -L bindings/guile/scheme bindings/guile/tests/trace/test-trace-optim.scm
;;;
;;; Regression tests for the optimizations the tracer already performs:
;;;
;;;   - dead-node elimination keeps every effectful node and its whole input
;;;     dependency chain, keeps everything that feeds a graph output, and
;;;     drops the parameters only the dead part referenced;
;;;   - constant folding computes an all-concrete subtree at trace time, with
;;;     the same value plain Scheme arithmetic gives;
;;;   - a branch on a concrete (static) test executes only the chosen arm and
;;;     records no join node, while a symbolic test still joins with `where`;
;;;   - optimizing changes neither the result, repeated calls, cache hits nor
;;;     the gradients.
;;;
;;; Nothing here asserts that a missing optimization stays missing; a future
;;; CSE pass may change node counts as long as these checks hold.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace graph)
             (sonicboom trace dce)
             (sonicboom trace tracer)
             (sonicboom trace macro)
             (sonicboom trace interp)
             (sonicboom trace autodiff))

(test-begin "sonicboom-trace-optim")

(define (t x) (list->ctensor x))
(define (stat f key) (assq-ref (trace-stats f) key))

(define (read-doc text) (call-with-input-string text read))
(define (doc-fields doc)
  (match doc
    (('sonicboom-s-expr _ ('graph . fields)) fields)
    (_ (error "not an S-Expr document" doc))))
(define (doc-nodes doc) (cdr (assq 'nodes (doc-fields doc))))
(define (doc-params doc) (cdr (assq 'parameters (doc-fields doc))))
(define (doc-inputs doc) (cdr (assq 'inputs (doc-fields doc))))
(define (node-ops text) (map cadr (doc-nodes (read-doc text))))

;;; --- plain-Scheme reference (nested lists) -------------------------------------

(define (deep-map f . xs)
  (if (pair? (car xs)) (apply map (lambda ys (apply deep-map f ys)) xs) (apply f xs)))
(define (deep-flatten x) (if (pair? x) (append-map deep-flatten x) (list x)))
(define (l-add xs ys) (deep-map + xs ys))
(define (l-scale x k) (deep-map (lambda (a) (* a k)) x))
(define (l-shift x k) (deep-map (lambda (a) (+ a k)) x))
(define (l-relu x) (deep-map (lambda (a) (max 0 a)) x))
(define (close? a b)
  (let ((xs (deep-flatten a)) (ys (deep-flatten b)))
    (and (= (length xs) (length ys))
         (every (lambda (x y) (<= (abs (- x y)) 1e-5)) xs ys))))
(define (nested t) (ctensor->nested t))

;;; =============================================================================
;;; Dead-node elimination: unit level
;;; =============================================================================

;; a, p0, p1, p2 are the values the nodes below read.
(define p0 (make-gparam "p0" 'float32 '(2) '(1.0 2.0)))
(define p1 (make-gparam "p1" 'float32 '(2) '(3.0 4.0)))
(define p2 (make-gparam "p2" 'float32 '(2) '(5.0 6.0)))

(define n-add (make-gnode 'add '("a" "p0") (list (list "v1" 'float32 '(2))) '()))
(define n-dead (make-gnode 'relu '("v1") (list (list "v2" 'float32 '(2))) '()))
(define n-print (make-gnode 'print '("v1") (list (list "v3" 'float32 '(2)))
                            '((label (string "dbg")))))
(define n-out (make-gnode 'mul '("a" "p1") (list (list "v4" 'float32 '(2))) '()))
(define n-chain (make-gnode 'relu '("v4") (list (list "v5" 'float32 '(2))) '()))
;; consumes the print's output and p2; neither is otherwise needed
(define n-print-use (make-gnode 'mul '("v3" "p2")
                                (list (list "v6" 'float32 '(2))) '()))

(call-with-values
    (lambda ()
      (eliminate-dead-nodes (list n-add n-dead n-print n-out n-chain n-print-use)
                            (list p0 p1 p2) '("v5")))
  (lambda (nodes params)
    (test-equal "DCE keeps effectful and output-feeding nodes in original order"
      '(add print mul relu) (map gnode-op nodes))
    (test-assert "DCE keeps exactly n-add, n-print, n-out, n-chain"
      (equal? nodes (list n-add n-print n-out n-chain)))
    (test-equal "DCE keeps the parameter of the effect and of the graph output"
      '("p0" "p1") (map gparam-name params))))

;; A node with several outputs is live when any one output is live.
(define n-multi (make-gnode 'adamw '("a")
                            (list (list "v7" 'float32 '(2))
                                  (list "v8" 'float32 '(2)))
                            '()))
(call-with-values
    (lambda () (eliminate-dead-nodes (list n-multi) '() '("v8")))
  (lambda (nodes params)
    (test-equal "all outputs of a live multi-output node survive" (list n-multi) nodes)))

;; A graph output may be a parameter: no node needed, the parameter is live.
(call-with-values
    (lambda () (eliminate-dead-nodes (list n-dead) (list p0 p1) '("p1")))
  (lambda (nodes params)
    (test-equal "a graph output that is a parameter drops all nodes" '() nodes)
    (test-equal "a graph output that is a parameter keeps it" '("p1")
                (map gparam-name params))))

;;; =============================================================================
;;; Constant folding at trace time
;;; =============================================================================

(define fold-text
  (trace->sexpr (lambda (x) (add x (mul 2.0 3.0))) (t '((1.0 2.0)))))

(test-equal "a constant subtree becomes one inline parameter, not nodes"
  '(add) (node-ops fold-text))
(test-equal "the folded constant is the value plain arithmetic gives"
  '(data :values 6.0) (cadddr (car (doc-params (read-doc fold-text)))))
(test-assert "the graph with a folded constant computes x + 6"
  (close? (l-add '((1.0 2.0)) '((6.0 6.0)))
          (nested (cdar (interpret-sexpr (read-doc fold-text)
                                         (list (cons (cadr (car (doc-inputs
                                                                 (read-doc fold-text))))
                                                     (t '((1.0 2.0))))))))))

;; A body that is entirely constant has no nodes; its value is a parameter.
(define const-text (trace->sexpr (lambda (x) (mul 2.0 3.0)) (t '((0.0)))))

(test-equal "an all-concrete body records no node" '() (node-ops const-text))
(test-equal "an all-concrete body keeps its value as one parameter"
  1 (length (doc-params (read-doc const-text))))
(test-assert "the all-concrete body evaluates to 6"
  (let ((out (cdar (interpret-sexpr (read-doc const-text) '()))))
    (= 1 (ctensor-numel out)) (= 6.0 (ctensor-item out))))

;; A mixed subtree keeps the symbolic op but folds its constant operand.
(define mixed-text
  (trace->sexpr (lambda (x) (mul x (add 1.0 2.0))) (t '((1.0 2.0)))))

(test-equal "constant folding leaves the symbolic op in place"
  '(mul) (node-ops mixed-text))
(test-equal "the constant operand of a symbolic op is folded"
  '(data :values 3.0) (cadddr (car (doc-params (read-doc mixed-text)))))

;; Folding must not execute an effect early: a fully constant argument to
;; trace-print is folded, but the print itself is a node and runs only when
;; the graph runs.
(define (with-output thunk)
  (let* ((out (open-output-string)) (err (open-output-string))
         (r (parameterize ((current-output-port out) (current-error-port err))
              (thunk))))
    (values r (get-output-string out) (get-output-string err))))

(define folded-print
  (trace-function (lambda (x)
                    (trace-print (mul 2.0 3.0) "folded")
                    (add x 1.0))))

(call-with-values
    (lambda ()
      (with-output (lambda () (folded-print (t '((1.0 2.0)))))))
  (lambda (r out err)
    (test-equal "a folded effectful operand does not print at trace time"
      "folded: 6.0\n" out)
    (test-equal "no trace-time side-effect warning for a folded print" 0
                (stat folded-print 'output-warnings))))
(test-assert "the folded print still records a print node"
  (memq 'print (node-ops (cdar (trace-segments folded-print)))))
(test-assert "the folded print's argument is a constant parameter"
  (any (lambda (p) (equal? (cadddr p) '(data :values 6.0)))
       (doc-params (read-doc (cdar (trace-segments folded-print))))))
(test-equal "the print's operand is not a node of its own"
  '(print add) (node-ops (cdar (trace-segments folded-print))))

;;; =============================================================================
;;; Static branches keep Scheme semantics
;;; =============================================================================

(define-traced (static-pick x n)
  ;; N is a Scheme value, so this is an ordinary Scheme `if`: only the chosen
  ;; arm is evaluated and traced.
  (if (> n 0) (mul x 2.0) (relu x)))

(test-equal "a static true branch records only the then arm"
  '(mul) (node-ops (trace->sexpr static-pick (t '((1.0 -2.0))) 1)))
(test-equal "a static false branch records only the else arm"
  '(relu) (node-ops (trace->sexpr static-pick (t '((1.0 -2.0))) -1)))
(test-assert "a static true branch computes mul"
  (close? '((2.0 -4.0))
          (nested (static-pick (t '((1.0 -2.0))) 1))))
(test-assert "a static false branch computes relu"
  (close? '((1.0 0.0))
          (nested (static-pick (t '((1.0 -2.0))) -1))))

(define-traced (dynamic-pick x)
  ;; A symbolic test joins both arms, as before.
  (if (gt (sum x) 0.0) (mul x 2.0) (relu x)))

(test-assert "a symbolic test still records a where join"
  (memq 'where (node-ops (trace->sexpr dynamic-pick (t '((1.0 -2.0)))))))
(test-assert "a symbolic test computes the chosen arm's value"
  (and (close? '((4.0 -2.0)) (nested (dynamic-pick (t '((2.0 -1.0))))))
       (close? '((1.0 0.0)) (nested (dynamic-pick (t '((1.0 -2.0))))))))

;;; =============================================================================
;;; The optimizations preserve results, repeats, cache and gradients
;;; =============================================================================

(define (clean x) (add (mul (relu x) 3.0) 1.0))
(define (with-dead x)
  ;; The whole matmul chain is unused: DCE must remove it before execution.
  (let ((dead (neg (relu (matmul x x)))))
    (add (mul (relu x) 3.0) 1.0)))

(define (l-clean x) (l-shift (l-scale (l-relu x) 3.0) 1.0))

(let ((f (trace-function clean))
      (g (trace-function with-dead)))
  (define (check label x)
    ;; One call per traced function: the counters below would otherwise count
    ;; the comparison's second call.
    (let ((rf (nested (f (t x)))) (rg (nested (g (t x)))))
      (test-assert (string-append label ": result matches plain Scheme")
        (close? (l-clean x) rf))
      (test-assert (string-append label ": dead code does not change the result")
        (close? rf rg))))
  (check "first call" '((1.0 -2.0) (3.0 0.5)))
  (test-equal "clean: the first call traces, no cache hit yet"
    '(1 1 0 1) (map (lambda (k) (stat f k)) '(calls traces cache-hits segments-traced)))
  (test-equal "with-dead: the first call traces, no cache hit yet"
    '(1 1 0 1) (map (lambda (k) (stat g k)) '(calls traces cache-hits segments-traced)))
  (check "second call, new values" '((-1.0 4.0) (2.0 -3.0)))
  (test-equal "clean: the repeated call is a cache hit and traces nothing new"
    '(2 1 1 1) (map (lambda (k) (stat f k)) '(calls traces cache-hits segments-traced)))
  (test-equal "with-dead: the repeated call is a cache hit and traces nothing new"
    '(2 1 1 1) (map (lambda (k) (stat g k)) '(calls traces cache-hits segments-traced)))
  (test-equal "dead code never reaches the emitted graph"
    (node-ops (trace->sexpr clean (t '((1.0 -2.0) (3.0 0.5)))))
    (node-ops (trace->sexpr with-dead (t '((1.0 -2.0) (3.0 0.5)))))))

;; Gradients: dead code must not add, remove or change a gradient.
(define (loss-clean x w) (sum (mul (matmul x w) (matmul x w))))

(define (loss-with-dead x w)
  (let ((dead (neg (relu (add x 100.0)))))
    (sum (mul (matmul x w) (matmul x w)))))

(define (vg-close? a b)
  (and (close? (nested (car a)) (nested (car b)))
       (= (length (cdr a)) (length (cdr b)))
       (every (lambda (ga gb) (close? (nested ga) (nested gb)))
              (cdr a) (cdr b))))

(let ((vc (value-and-grad loss-clean))
      (vd (value-and-grad loss-with-dead)))
  (define (check label x w)
    (test-assert (string-append label ": dead code does not change loss or grads")
      (vg-close? (vc (t x) (t w)) (vd (t x) (t w)))))
  (define x-args '(((1.0 2.0) (3.0 4.0)) ((-0.5 1.5) (2.0 0.25))))
  (define w-args '(((0.5 -1.0) (2.0 0.25)) ((1.0 0.5) (-2.0 0.75))))
  (check "grad first call" (car x-args) (car w-args))
  (test-equal "grad first call traces, no cache hit yet"
    '(1 1 0) (map (lambda (k) (stat vc k)) '(calls traces cache-hits)))
  (check "grad second call, new values" (cadr x-args) (cadr w-args))
  (test-equal "gradients of the repeated call come from the cache and reuse the trace"
    '(2 1 1) (map (lambda (k) (stat vc k)) '(calls traces cache-hits)))
  (test-equal "gradients of the dead-code graph come from the cache"
    '(2 1 1) (map (lambda (k) (stat vd k)) '(calls traces cache-hits))))

;;; =============================================================================

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-optim")
(exit (if (zero? failures) 0 1))
