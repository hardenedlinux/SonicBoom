;;; SonicBoom — tracing prototype tests (pure Guile, no libsonicboom)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace.scm
;;;
;;; Every traced result is compared with the same computation written
;;; independently in plain Scheme over nested lists. Cache behaviour is checked
;;; through trace-stats / trace-tree.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace interp))

;;; --- plain-Scheme reference (nested lists, no prototype code) ----------------

(define (deep-map f . xs)
  (if (pair? (car xs)) (apply map (lambda ys (apply deep-map f ys)) xs) (apply f xs)))
(define (deep-flatten x) (if (pair? x) (append-map deep-flatten x) (list x)))
(define (l-sum x) (fold + 0 (deep-flatten x)))
(define (l-scale x k) (deep-map (lambda (a) (* a k)) x))
(define (l-shift x k) (deep-map (lambda (a) (+ a k)) x))
(define (l-relu x) (deep-map (lambda (a) (max 0 a)) x))
(define (l-neg x) (deep-map - x))
(define (l-matmul a b)
  (map (lambda (row)
         (apply map (lambda col (fold + 0 (map * row col))) b))
       a))
(define (l-add-row m v) (map (lambda (row) (map + row v)) m))

(define (close? a b)
  (let ((xs (deep-flatten a)) (ys (deep-flatten b)))
    (and (= (length xs) (length ys))
         (every (lambda (x y) (< (abs (- x y)) 1e-9)) xs ys))))

(define (t x) (list->ctensor x))
(define (stat f key) (assq-ref (trace-stats f) key))

(define (node-ops text)
  (match (call-with-input-string text read)
    (('sonicboom-s-expr ('version 0 1) ('graph . fields))
     (map cadr (cdr (assq 'nodes fields))))))

;;; --- functions under test -------------------------------------------------------

;; no branch
(define (mlp x w b) (relu (add (matmul x w) b)))
(define (l-mlp x w b) (l-relu (l-add-row (l-matmul x w) b)))

;; one data-dependent if
(define (step x) (if (bool (gt (sum x) 0.0)) (mul x 2.0) (sub x 1.0)))
(define (l-step x) (if (> (l-sum x) 0) (l-scale x 2) (l-shift x -1)))

;; if inside a loop; N is a static argument
(define (loop-if x n)
  (let loop ((i 0) (acc x))
    (if (= i n)
        acc
        (loop (+ i 1)
              (if (bool (gt (sum acc) 0.0))
                  (sub (mul acc 0.5) 1.0)
                  (add (neg acc) 0.75))))))
(define (l-loop-if x n)
  (let loop ((i 0) (acc x))
    (if (= i n)
        acc
        (loop (+ i 1)
              (if (> (l-sum acc) 0)
                  (l-shift (l-scale acc 0.5) -1)
                  (l-shift (l-neg acc) 0.75))))))

(define x-pos '((1.0 2.0) (3.0 4.0)))
(define x-neg '((-1.0 2.0) (-3.0 -4.0)))
(define x-pos2 '((0.5 0.25) (2.0 1.0)))
(define x-neg2 '((-0.5 -0.25) (-2.0 1.0)))
(define loop-inputs
  (list x-pos x-neg x-pos2 x-neg2 '((0.1 0.1) (0.1 0.1)) '((5.0 -1.0) (0.0 0.0))))

(test-begin "sonicboom-trace")

;;; --- symbolic tensors / mode -------------------------------------------------------

(test-equal "eager mode computes with ctensors"
  (l-step x-pos)
  (ctensor->nested (step (t x-pos))))

(test-assert "not tracing outside a traced call" (not (tracing?)))

(test-assert "tracing inside a traced call"
  (let* ((seen #f)
         (f (trace-function (lambda (x) (set! seen (tracing?)) (relu x)))))
    (f (t x-pos))
    seen))

(test-equal "symbolic tensor records shape and dtype only"
  '(float32 (2 3) "v0")
  (let ((r (make-recorder (make-trace-context 16) '() '() '() 0 '() 0 #f)))
    (parameterize ((current-tracer r))
      (let ((s (matmul (make-sym "a" 'float32 '(2 4))
                       (make-sym "b" 'float32 '(4 3)))))
        (list (sym-dtype s) (sym-shape s) (sym-name s))))))

(test-error "bool on a multi-element symbolic tensor" #t
  ((trace-function (lambda (x) (if (bool x) x x))) (t x-pos)))

;;; --- no branch: S-Expr output -----------------------------------------------------------

(define w '((1.0 -1.0 0.5) (2.0 0.0 -0.5)))
(define b '(0.5 -3.0 1.0))

(let ((text (trace->sexpr mlp (t x-pos) (t w) (t b))))
  (test-equal "no-branch graph: node ops in order" '(matmul add relu) (node-ops text))
  (test-assert "every node output is typed"
    (match (call-with-input-string text read)
      (('sonicboom-s-expr _ ('graph . fields))
       (every (match-lambda
                (('node _ _ ('outputs (name ('tensor dtype ('shape . dims)))) _)
                 (and (string? name) (symbol? dtype) (every integer? dims)))
                (_ #f))
              (cdr (assq 'nodes fields))))))
  (test-assert "reference interpreter on emitted text matches plain Scheme"
    (close? (l-mlp x-pos w b)
            (ctensor->nested
             (cdar (interpret-sexpr-text
                    text `(("arg0" . ,(t x-pos)) ("arg1" . ,(t w))
                           ("arg2" . ,(t b)))))))))

(let ((f (trace-function mlp)))
  (test-assert "no-branch: first call matches plain Scheme"
    (close? (l-mlp x-pos w b) (ctensor->nested (f (t x-pos) (t w) (t b)))))
  (test-assert "no-branch: second call (new values) matches plain Scheme"
    (close? (l-mlp x-neg w b) (ctensor->nested (f (t x-neg) (t w) (t b)))))
  (test-equal "no-branch: second call hits the cache"
    '(2 1 1 0)
    (map (lambda (k) (stat f k)) '(calls cache-hits traces graph-breaks))))

;;; --- dead-node elimination -----------------------------------------------------------------

(test-equal "DCE drops values that do not reach the output"
  '(add)
  (node-ops (trace->sexpr (lambda (x) (let ((unused (mul (relu x) 3.0)))
                                        (add x 1.0)))
                          (t x-pos))))

(test-assert "DCE drops the unused constant parameter"
  (match (call-with-input-string
          (trace->sexpr (lambda (x) (mul x 3.0) (add x 1.0)) (t x-pos)) read)
    (('sonicboom-s-expr _ ('graph . fields))
     (= 1 (length (cdr (assq 'parameters fields)))))))

;;; --- single data-dependent if: graph break + trace tree -------------------------------------

(let ((f (trace-function step)))
  (test-assert "if: first call (true side)"
    (close? (l-step x-pos) (ctensor->nested (f (t x-pos)))))
  (test-equal "if: first call broke the graph once"
    '(1 1 2) (map (lambda (k) (stat f k)) '(traces graph-breaks segments-traced)))
  (test-assert "if: same branch, new values"
    (close? (l-step x-pos2) (ctensor->nested (f (t x-pos2)))))
  (test-equal "if: second call hit the cache" 1 (stat f 'cache-hits))
  (test-assert "if: new branch value"
    (close? (l-step x-neg) (ctensor->nested (f (t x-neg)))))
  (test-equal "if: new branch value re-entered the stored continuation only"
    ;; one more trace, one more segment (the #f suffix), no new break
    '(2 3 1) (map (lambda (k) (stat f k)) '(traces segments-traced graph-breaks)))
  (test-assert "if: false branch again"
    (close? (l-step x-neg2) (ctensor->nested (f (t x-neg2)))))
  (test-equal "if: both branches now cached" 2 (stat f 'cache-hits))
  (test-equal "if: trace tree has both children"
    '(#t #f)
    (match (trace-tree f)
      (((sig 'branch cond (vals . subtrees) ...)) vals)))
  (test-equal "if: segments are pruned to what they compute"
    '((reduce_sum greater) (mul) (sub))
    (map (lambda (s) (node-ops (cdr s))) (trace-segments f))))

;;; --- single if: multi-resume into a where node -----------------------------------------------

(let ((f (trace-function step #:multi-resume? #t)))
  (test-assert "multi-resume: true input"
    (close? (l-step x-pos) (ctensor->nested (f (t x-pos)))))
  (test-assert "multi-resume: false input, same graph"
    (close? (l-step x-neg) (ctensor->nested (f (t x-neg)))))
  (test-equal "multi-resume: one trace, no break, one merge, one cache hit"
    '(1 0 1 1) (map (lambda (k) (stat f k)) '(traces graph-breaks merges cache-hits)))
  (test-equal "multi-resume: single segment with a where node"
    '((reduce_sum greater mul sub where))
    (map (lambda (s) (node-ops (cdr s))) (trace-segments f))))

(let ((f (trace-function step #:multi-resume? #t #:max-side-nodes 0)))
  (f (t x-pos))
  (test-equal "multi-resume: over-budget side falls back to a graph break"
    '(1 0) (map (lambda (k) (stat f k)) '(graph-breaks merges))))

;;; --- if inside a loop -------------------------------------------------------------------------

(define (check-loop-if label f)
  (test-assert (string-append label ": all inputs match plain Scheme")
    (every (lambda (x) (close? (l-loop-if x 3) (ctensor->nested (f (t x) 3))))
           loop-inputs))
  (let ((traces (stat f 'traces)) (hits (stat f 'cache-hits)))
    (test-assert (string-append label ": repeated inputs match plain Scheme")
      (every (lambda (x) (close? (l-loop-if x 3) (ctensor->nested (f (t x) 3))))
             loop-inputs))
    (test-equal (string-append label ": repeated inputs all hit the cache")
      (list traces (+ hits (length loop-inputs)))
      (list (stat f 'traces) (stat f 'cache-hits)))))

(let ((f (trace-function loop-if)))
  (check-loop-if "loop-if (breaks)" f)
  (test-equal "loop-if (breaks): every path is 3 branches deep"
    3
    (match (trace-tree f)
      (((sig . tree))
       ;; depth of a uniform tree, #f if paths differ in length
       (let depth ((n tree))
         (match n
           (('leaf _) 0)
           (('branch _ (v . sub) ...)
            (let ((ds (map depth sub)))
              (and (every number? ds) (apply = ds) (+ 1 (car ds))))))))))
  (test-assert "loop-if (breaks): a new static argument gets its own tree"
    (and (close? (l-loop-if x-pos 2) (ctensor->nested (f (t x-pos) 2)))
         (= 2 (length (trace-tree f))))))

(let ((f (trace-function loop-if #:multi-resume? #t #:max-side-nodes 64
                         #:max-depth 3)))
  (check-loop-if "loop-if (multi-resume)" f)
  (test-equal "loop-if (multi-resume): one graph, no break"
    '(1 0 1)
    (list (stat f 'traces) (stat f 'graph-breaks) (length (trace-segments f)))))

(let ((f (trace-function loop-if #:multi-resume? #t #:max-side-nodes 8
                         #:max-depth 3)))
  (check-loop-if "loop-if (tight budget)" f)
  (test-assert "loop-if (tight budget): mixes breaks and merges"
    (and (> (stat f 'graph-breaks) 0) (> (stat f 'merges) 0))))

;;; --- assumptions, executor, nesting -------------------------------------------------------

(let ((f (trace-function
          (lambda (x)
            (let ((c (gt (sum x) 0.0)))
              (if (bool c)
                  (if (bool c) (mul x 2.0) (neg x))   ; same condition again
                  (sub x 1.0)))))))
  (test-assert "repeated bool on one condition: result"
    (close? (l-scale x-pos 2) (ctensor->nested (f (t x-pos)))))
  (test-equal "repeated bool on one condition: one break" 1 (stat f 'graph-breaks)))

(let* ((seen '())
       (exec (lambda (text inputs)
               (set! seen (cons text seen))
               (interpret-sexpr-text text inputs)))
       (f (trace-function step #:executor exec)))
  (test-assert "pluggable executor (keyword): result"
    (close? (l-step x-neg) (ctensor->nested (f (t x-neg)))))
  (test-equal "pluggable executor (keyword): used for every segment run"
    (stat f 'executor-calls) (length seen)))

(let* ((n 0)
       (f (trace-function step)))
  (parameterize ((current-executor
                  (lambda (text inputs) (set! n (+ n 1))
                          (interpret-sexpr-text text inputs))))
    (f (t x-pos)))
  (test-equal "pluggable executor (parameter)" (stat f 'executor-calls) n))

(let* ((inner (trace-function (lambda (x) (relu (sub x 2.0)))))
       (outer (trace-function (lambda (x) (mul (inner x) 3.0)))))
  (test-assert "traced function called while tracing is inlined"
    (close? (l-scale (l-relu (l-shift x-pos -2)) 3)
            (ctensor->nested (outer (t x-pos)))))
  (test-equal "inlined: inner was never traced on its own" 0 (stat inner 'calls)))

(let ((f (trace-function
          (lambda (x w)
            (let ((h (matmul x w)))
              (if (bool (lt (sum h) 10.0)) (relu h) (neg h)))))))
  (test-assert "matmul + if: both sides"
    (and (close? (let ((h (l-matmul x-pos w))) (if (< (l-sum h) 10) (l-relu h) (l-neg h)))
                 (ctensor->nested (f (t x-pos) (t w))))
         (close? (let ((h (l-matmul x-neg w))) (if (< (l-sum h) 10) (l-relu h) (l-neg h)))
                 (ctensor->nested (f (t x-neg) (t w)))))))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace")
(exit (if (zero? failures) 0 1))
