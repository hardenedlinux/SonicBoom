;;; SonicBoom — tracing prototype tests: syntactic branches, guards, effects
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace-macro.scm
;;;
;;; Covers (sonicboom trace macro) — define-traced / traced-lambda rewriting
;;; if/cond/when/unless into in-graph `where` joins — plus free-variable guards,
;;; output-port side-effect detection, trace-print, and #:retrace-on-miss?.
;;; Results are compared with plain Scheme over nested lists. Prints the node
;;; count table (macro joins vs continuation multi-resume) to stdout.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 format)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace macro))

;;; --- plain-Scheme reference ------------------------------------------------------

(define (deep-map f . xs)
  (if (pair? (car xs)) (apply map (lambda ys (apply deep-map f ys)) xs) (apply f xs)))
(define (deep-flatten x) (if (pair? x) (append-map deep-flatten x) (list x)))
(define (l-sum x) (fold + 0 (deep-flatten x)))
(define (l-scale x k) (deep-map (lambda (a) (* a k)) x))
(define (l-shift x k) (deep-map (lambda (a) (+ a k)) x))
(define (l-neg x) (deep-map - x))

(define (close? a b)
  (let ((xs (deep-flatten a)) (ys (deep-flatten b)))
    (and (= (length xs) (length ys))
         (every (lambda (x y) (< (abs (- x y)) 1e-9)) xs ys))))

(define (t x) (list->ctensor x))
(define (stat f key) (assq-ref (trace-stats f) key))
(define (stats f keys) (map (lambda (k) (stat f k)) keys))

(define (segment-node-count text)
  (match (call-with-input-string text read)
    (('sonicboom-s-expr _ ('graph . fields)) (length (cdr (assq 'nodes fields))))))
(define (total-nodes f)
  (fold + 0 (map (lambda (s) (segment-node-count (cdr s))) (trace-segments f))))
(define (node-ops text)
  (match (call-with-input-string text read)
    (('sonicboom-s-expr _ ('graph . fields)) (map cadr (cdr (assq 'nodes fields))))))

(define (with-output thunk)
  "Run THUNK; returns (values result stdout-text stderr-text)."
  (let* ((out (open-output-string)) (err (open-output-string))
         (r (parameterize ((current-output-port out) (current-error-port err))
              (thunk))))
    (values r (get-output-string out) (get-output-string err))))

(define (error-message thunk)
  "The error THUNK raises, rendered as text (#f if it returns)."
  (catch #t
    (lambda () (thunk) #f)
    (lambda (key . args) (format #f "~a" args))))

(define (tree-shape tree)
  "A trace tree with value names dropped: only kinds and branch values."
  (match tree
    (('leaf _) 'leaf)
    (('branch _ (v . sub) ...) (cons 'branch (map (lambda (v s) (cons v (tree-shape s))) v sub)))))

;;; --- inputs ------------------------------------------------------------------------

(define x-pos '((1.0 2.0) (3.0 4.0)))
(define x-neg '((-1.0 2.0) (-3.0 -4.0)))
(define x-pos2 '((0.5 0.25) (2.0 1.0)))
(define x-neg2 '((-0.5 -0.25) (-2.0 1.0)))
(define loop-inputs
  (list x-pos x-neg x-pos2 x-neg2 '((0.1 0.1) (0.1 0.1)) '((5.0 -1.0) (0.0 0.0))))

(define (l-loop-if x n)
  (let loop ((i 0) (acc x))
    (if (= i n)
        acc
        (loop (+ i 1)
              (if (> (l-sum acc) 0)
                  (l-shift (l-scale acc 0.5) -1)
                  (l-shift (l-neg acc) 0.75))))))

;; Macro version: plain `if` on a tensor condition, no `bool`.
(define-traced (loop-if/macro x n)
  (let loop ((i 0) (acc x))
    (if (= i n)
        acc
        (loop (+ i 1)
              (if (gt (sum acc) 0.0)
                  (sub (mul acc 0.5) 1.0)
                  (add (neg acc) 0.75))))))

;; Continuation version (as in test-trace.scm).
(define (loop-if/bool x n)
  (let loop ((i 0) (acc x))
    (if (= i n)
        acc
        (loop (+ i 1)
              (if (bool (gt (sum acc) 0.0))
                  (sub (mul acc 0.5) 1.0)
                  (add (neg acc) 0.75))))))

(define (step/bool x) (if (bool (gt (sum x) 0.0)) (mul x 2.0) (sub x 1.0)))
(define (l-step x) (if (> (l-sum x) 0) (l-scale x 2) (l-shift x -1)))

(test-begin "sonicboom-trace-macro")

;;; --- if inside a loop, macro version --------------------------------------------------

(test-assert "macro loop-if: all inputs match plain Scheme"
  (every (lambda (x) (close? (l-loop-if x 3) (ctensor->nested (loop-if/macro (t x) 3))))
         loop-inputs))

(test-equal "macro loop-if: one trace, no break, no continuation merge, 3 where joins"
  '(1 0 0 3 1)
  (append (stats loop-if/macro '(traces graph-breaks merges branch-merges))
          (list (length (trace-segments loop-if/macro)))))

(test-equal "macro loop-if: every later call hits the cache"
  (- (length loop-inputs) 1) (stat loop-if/macro 'cache-hits))

(test-equal "macro loop-if: graph is the loop unrolled, one where per iteration"
  (apply append (make-list 3 '(reduce_sum greater mul sub neg add where)))
  (node-ops (cdar (trace-segments loop-if/macro))))

;; Node counts: macro joins vs continuation multi-resume, n = 1..4.
(define (macro-nodes n)
  (let ((f (traced-lambda (x)
             (let loop ((i 0) (acc x))
               (if (= i n)
                   acc
                   (loop (+ i 1)
                         (if (gt (sum acc) 0.0)
                             (sub (mul acc 0.5) 1.0)
                             (add (neg acc) 0.75))))))))
    (f (t x-pos))
    (total-nodes f)))

(define (continuation-nodes n)
  (let ((f (trace-function loop-if/bool #:multi-resume? #t
                           #:max-side-nodes 10000 #:max-depth n)))
    (f (t x-pos) n)
    (and (zero? (stat f 'graph-breaks)) (total-nodes f))))

(define node-table
  (map (lambda (n) (list n (macro-nodes n) (continuation-nodes n))) '(1 2 3 4)))

(format #t "~%if-in-loop node counts (one graph, no graph break):~%")
(format #t "  iterations  macro-where  continuation-multi-resume~%")
(for-each (lambda (row) (apply format #t "  ~10d  ~11d  ~25d~%" row)) node-table)

(test-equal "node count: macro version grows linearly (7 per iteration)"
  '(7 14 21 28) (map cadr node-table))

(test-equal "node count: continuation multi-resume at 3 iterations is 49"
  49 (caddr (assv 3 node-table)))

(test-assert "node count: continuation multi-resume grows faster than linearly"
  (let ((c (map caddr node-table)))
    (< (- (cadr c) (car c)) (- (caddr c) (cadr c)) (- (cadddr c) (caddr c)))))

;;; --- cond / when / unless / value shapes ------------------------------------------------

(define-traced (piecewise x)
  (cond ((gt (sum x) 9.0) (mul x 0.1))
        ((gt (sum x) 0.0) x)
        (else (neg x))))
(define (l-piecewise x)
  (cond ((> (l-sum x) 9) (l-scale x 0.1)) ((> (l-sum x) 0) x) (else (l-neg x))))

(test-assert "cond: three arms, all inputs"
  (every (lambda (x) (close? (l-piecewise x) (ctensor->nested (piecewise (t x)))))
         (list x-pos x-neg x-pos2 '((5.0 5.0) (0.0 0.0)))))
(test-equal "cond: one trace, two where joins, cache hits afterwards"
  '(1 0 2 3) (stats piecewise '(traces graph-breaks branch-merges cache-hits)))

(define-traced (scaled x n)
  ;; concrete (static) tests keep normal Scheme semantics
  (let ((y (when (> n 0) (mul x 2.0))))
    (unless (> n 0) (error "unreachable"))
    (if (= n 1) y (add y 1.0))))
(test-assert "when/unless/if on concrete tests behave like Scheme"
  (and (close? (l-scale x-pos 2) (ctensor->nested (scaled (t x-pos) 1)))
       (close? (l-shift (l-scale x-pos 2) 1) (ctensor->nested (scaled (t x-pos) 2)))))

(define-traced (sign-scale x)
  ;; Scheme numbers in the arms become rank-0 constants joined by where
  (mul x (if (gt (sum x) 0.0) 1.0 -1.0)))
(test-assert "if returning Scheme numbers joins them as constants"
  (and (close? x-pos (ctensor->nested (sign-scale (t x-pos))))
       (close? (l-neg x-neg) (ctensor->nested (sign-scale (t x-neg))))))

(define-traced (swap-sum x y)
  ;; arms returning lists of tensors are joined element by element
  (let ((p (if (gt (sum x) (sum y)) (list x y) (list y x))))
    (sub (car p) (cadr p))))
(test-assert "if returning lists joins element-wise"
  (and (close? (deep-map - x-pos x-neg)
               (ctensor->nested (swap-sum (t x-pos) (t x-neg))))
       (close? (deep-map - x-pos x-neg)
               (ctensor->nested (swap-sum (t x-neg) (t x-pos))))))

(define-traced (nested x)
  (if (gt (sum x) 0.0)
      (if (gt (sum x) 5.0) (mul x 3.0) (mul x 2.0))
      (neg x)))
(test-assert "nested ifs"
  (every (lambda (x)
           (close? (cond ((> (l-sum x) 5) (l-scale x 3)) ((> (l-sum x) 0) (l-scale x 2))
                         (else (l-neg x)))
                   (ctensor->nested (nested (t x)))))
         (list x-pos x-pos2 x-neg)))

;;; --- errors carry the source location ---------------------------------------------------

(define bad-loc (current-source-location)) (define-traced (bad x) (if (gt (sum x) 0.0) x (sum x)))

(test-assert "arms with different shapes: error names file and line"
  (let ((msg (error-message (lambda () (bad (t x-pos))))))
    (and msg
         (string-contains msg "arms differ in dtype/shape")
         (string-contains msg (format #f "test-trace-macro.scm:~a:"
                                      (+ 1 (assq-ref bad-loc 'line)))))))

(define-traced (float-cond x) (if (sum x) x (neg x)))
(test-assert "non-bool symbolic condition is an error"
  (let ((msg (error-message (lambda () (float-cond (t x-pos))))))
    (and msg (string-contains msg "must be a bool tensor"))))

;;; --- continuation fallback --------------------------------------------------------------

(define-syntax my-if            ; defined outside the traced body: core `if`
  (syntax-rules () ((_ c a b) (if c a b))))

(define-traced (fallback x)
  (my-if (bool (gt (sum x) 0.0)) (mul x 2.0) (sub x 1.0)))
(test-assert "if inside a foreign macro falls back to bool + continuation"
  (and (close? (l-step x-pos) (ctensor->nested (fallback (t x-pos))))
       (close? (l-step x-neg) (ctensor->nested (fallback (t x-neg))))))
(test-equal "fallback: graph break taken, continuation re-entered on the miss"
  '(1 2 0) (stats fallback '(graph-breaks traces branch-merges)))

;;; --- guards ---------------------------------------------------------------------------------

(define make-scaler
  (lambda (k0)
    (let ((k k0))
      (values (traced-lambda (x) (if (gt (sum x) 0.0) (mul x k) (neg x)))
              (lambda (v) (set! k v))))))

(call-with-values (lambda () (make-scaler 2.0))
  (lambda (f set-k!)
    (test-assert "closure guard: first value"
      (close? (l-scale x-pos 2) (ctensor->nested (f (t x-pos)))))
    (set-k! 3.0)
    (test-assert "closure guard: changed free variable is not served stale"
      (close? (l-scale x-pos 3) (ctensor->nested (f (t x-pos)))))
    (test-equal "closure guard: change caused a fresh trace (guard miss)"
      '(2 1 0) (stats f '(traces guard-misses cache-hits)))
    (set-k! 2.0)
    (test-assert "closure guard: old value again"
      (close? (l-scale x-pos 2) (ctensor->nested (f (t x-pos)))))
    (test-equal "closure guard: old value hits the first cached trace"
      '(2 1) (stats f '(traces cache-hits)))
    (test-equal "closure guard: one cached trace per guard value"
      '(((k . 2.0)) ((k . 3.0)))
      (map cdar (trace-tree f)))))

(define threshold 0.0)
(define (helper x) (mul x 2.0))    ; procedure-valued: not guarded
(define-traced (above x) (if (gt (sum x) threshold) (helper x) (neg x)))

(test-assert "module-variable guard: before"
  (close? (l-scale x-pos2 2) (ctensor->nested (above (t x-pos2)))))
(set! threshold 100.0)
(test-assert "module-variable guard: after set!"
  (close? (l-neg x-pos2) (ctensor->nested (above (t x-pos2)))))
(test-equal "module-variable guard: data variables listed in the cache key"
  '(threshold) (map car (cdar (car (trace-tree above)))))

;;; --- side effects -----------------------------------------------------------------------------

(define-traced (noisy x)
  (display "tracing!")
  (relu x))

(call-with-values (lambda () (with-output (lambda () (noisy (t x-neg)) (noisy (t x-pos)))))
  (lambda (r out err)
    (test-equal "output while tracing is captured, not written" "" out)
    (test-equal "output while tracing is reported once (cached call is silent)"
      '(1 ("tracing!")) (list (stat noisy 'output-warnings) (trace-warnings noisy)))
    (test-assert "warning goes to current-error-port" (string-contains err "tracing!"))))

(define-traced (printing x)
  (trace-print (mul x 2.0) "dbl")      ; result unused: kept by DCE (effect)
  (add x 1.0))

(call-with-values
    (lambda () (with-output (lambda () (map (lambda (x) (printing (t x)))
                                            (list x-pos x-neg x-pos)))))
  (lambda (r out err)
    (test-equal "trace-print prints at run time, once per call, with run-time values"
      (string-append "dbl: ((2.0 4.0) (6.0 8.0))\n"
                     "dbl: ((-2.0 4.0) (-6.0 -8.0))\n"
                     "dbl: ((2.0 4.0) (6.0 8.0))\n")
      out)
    (test-equal "trace-print: no tracing-time warning" 0 (stat printing 'output-warnings))
    (test-assert "trace-print: print node survives DCE"
      (memq 'print (node-ops (cdar (trace-segments printing)))))))

(let ((f (trace-function (lambda (x)
                           (let ((y (trace-print (mul x 2.0) "before-break")))
                             (if (bool (gt (sum x) 0.0)) y (neg y)))))))
  (call-with-values (lambda () (with-output (lambda () (f (t x-pos)) (f (t x-neg)))))
    (lambda (r out err)
      (test-equal "trace-print across a graph break and a cache miss prints once per call"
        2 (length (filter (lambda (l) (string-prefix? "before-break" l))
                          (string-split out #\newline)))))))

(test-equal "trace-print in eager mode prints immediately"
  "eager: (1.0 2.0)\n"
  (call-with-values (lambda () (with-output (lambda () (trace-print (t '(1.0 2.0)) "eager"))))
    (lambda (r out err) out)))

;;; --- retrace-on-miss --------------------------------------------------------------------------

(let ((f (trace-function step/bool #:retrace-on-miss? #t)))
  (test-assert "retrace: both branches correct"
    (and (close? (l-step x-pos) (ctensor->nested (f (t x-pos))))
         (close? (l-step x-neg) (ctensor->nested (f (t x-neg))))
         (close? (l-step x-pos2) (ctensor->nested (f (t x-pos2))))
         (close? (l-step x-neg2) (ctensor->nested (f (t x-neg2))))))
  (test-equal "retrace: the miss re-traced from the start, then both branches hit"
    '(2 1 0 2) (stats f '(traces retraces retrace-divergences cache-hits)))
  (test-equal "retrace: the grafted tree has the same shape as with continuations"
    '(#t #f)
    (match (trace-tree f) (((key 'branch cond (vals . subtrees) ...)) vals))))

(let ((f (trace-function loop-if/bool #:retrace-on-miss? #t))
      (g (trace-function loop-if/bool)))
  (for-each (lambda (x) (f (t x) 3) (g (t x) 3)) loop-inputs)
  (test-assert "retrace loop-if: repeated inputs match plain Scheme and hit the cache"
    (let ((hits (stat f 'cache-hits)))
      (and (every (lambda (x) (close? (l-loop-if x 3) (ctensor->nested (f (t x) 3))))
                  loop-inputs)
           (= (stat f 'cache-hits) (+ hits (length loop-inputs))))))
  (test-equal "retrace loop-if: same tree as the continuation version"
    (map (compose tree-shape cdr) (trace-tree g))
    (map (compose tree-shape cdr) (trace-tree f)))
  (test-assert "retrace loop-if: re-traces prefixes (more segments traced), no divergence"
    (and (> (stat f 'segments-traced) (stat g 'segments-traced))
         (zero? (stat f 'retrace-divergences)))))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-macro")
(exit (if (zero? failures) 0 1))
