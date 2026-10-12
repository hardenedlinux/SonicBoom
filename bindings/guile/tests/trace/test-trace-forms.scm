;;; SonicBoom — tracing prototype tests: R7RS control forms, strict mode, compat
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace-forms.scm
;;;
;;; Each standard control form the syntax-rules walker rewrites (and, or,
;;; case, do, named let, cond =>) is traced on symbolic conditions and checked
;;; against the same computation in plain Scheme over nested lists, alone and
;;; in nested combinations; its concrete (Scheme) semantics are checked too.
;;; Then strict mode, and that only the compat module touches
;;; implementation-specific mechanisms.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 format)
             (ice-9 rdelim)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace macro)
             (sonicboom trace interp))

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
(define (node-ops text)
  (match (call-with-input-string text read)
    (('sonicboom-s-expr _ ('graph . fields)) (map cadr (cdr (assq 'nodes fields))))))
(define (all-ops f) (append-map (lambda (s) (node-ops (cdr s))) (trace-segments f)))

(define (agrees? traced plain inputs)
  "TRACED on each input (twice) matches PLAIN on the same nested list."
  (every (lambda (x)
           (and (close? (plain x) (ctensor->nested (traced (t x))))
                (close? (plain x) (ctensor->nested (traced (t x))))))
         inputs))

(define (error-message thunk)
  (catch #t (lambda () (thunk) #f) (lambda (key . args) (format #f "~a" args))))

(define (stderr-of thunk)
  (let ((err (open-output-string)))
    (parameterize ((current-error-port err)) (thunk))
    (get-output-string err)))

(define x-pos  '((1.0 2.0) (3.0 4.0)))        ; sum 10
(define x-neg  '((-1.0 2.0) (-3.0 -4.0)))     ; sum -6
(define x-mid  '((0.5 0.5) (0.5 0.5)))        ; sum 2
(define x-ones '((1.0 1.0) (1.0 1.0)))        ; sum 4
(define x-zero '((0.0 0.0) (0.0 0.0)))        ; sum 0
(define inputs (list x-pos x-neg x-mid x-ones x-zero))

(test-begin "sonicboom-trace-forms")

;;; --- and / or --------------------------------------------------------------------------

(define-traced (in-range x)
  (if (and (gt (sum x) 0.0) (lt (sum x) 5.0)) (mul x 2.0) (neg x)))
(define (l-in-range x) (if (and (> (l-sum x) 0) (< (l-sum x) 5)) (l-scale x 2) (l-neg x)))

(test-assert "and: symbolic conditions" (agrees? in-range l-in-range inputs))
(test-equal "and: one graph, an `and` node feeding one where"
  '(1 0 1 1) (stats in-range '(traces graph-breaks logic-merges branch-merges)))
(test-assert "and: node present" (memq 'and (all-ops in-range)))

(define-traced (outside x)
  (if (or (lt (sum x) 1.0) (gt (sum x) 9.0)) (sub x 1.0) x))
(define (l-outside x) (if (or (< (l-sum x) 1) (> (l-sum x) 9)) (l-shift x -1) x))
(test-assert "or: symbolic conditions" (agrees? outside l-outside inputs))
(test-assert "or: node present" (memq 'or (all-ops outside)))

(define-traced (and-of-or x)
  ;; and with a nested or
  (if (and (gt (sum x) -100.0) (or (lt (sum x) 1.0) (gt (sum x) 9.0)))
      (mul x 3.0)
      x))
(define (l-and-of-or x)
  (if (and (> (l-sum x) -100) (or (< (l-sum x) 1) (> (l-sum x) 9))) (l-scale x 3) x))
(test-assert "and with nested or" (agrees? and-of-or l-and-of-or inputs))
(test-equal "and with nested or: one and, one or, one where, no break"
  '(1 2 0) (list (count (lambda (op) (memq op '(and))) (all-ops and-of-or))
                 (stat and-of-or 'logic-merges) (stat and-of-or 'graph-breaks)))

(define-traced (or-of-and x)
  (if (or (and (gt (sum x) 1.0) (lt (sum x) 3.0)) (lt (sum x) -1.0)) (neg x) x))
(define (l-or-of-and x)
  (if (or (and (> (l-sum x) 1) (< (l-sum x) 3)) (< (l-sum x) -1)) (l-neg x) x))
(test-assert "or with nested and" (agrees? or-of-and l-or-of-and inputs))

(define-traced (not-and x)
  (if (logical-not (and (gt (sum x) 0.0) (lt (sum x) 5.0))) x (mul x 0.5)))
(test-assert "logical-not of and"
  (agrees? not-and (lambda (x) (if (not (and (> (l-sum x) 0) (< (l-sum x) 5))) x (l-scale x 0.5)))
           inputs))

(define-traced (bad-and x) (and (gt (sum x) 0.0) (mul x 2.0)))
(test-assert "and on a symbolic condition with a non-bool operand is an error"
  (let ((msg (error-message (lambda () (bad-and (t x-pos))))))
    (and msg (string-contains msg "needs bool operands")
         (string-contains msg "test-trace-forms.scm:"))))

;;; --- case -----------------------------------------------------------------------------------

(define-traced (by-sum x)
  (case (sum x)
    ((10.0) (mul x 0.1))
    ((-6.0 -7.0) => (lambda (k) (mul x k)))
    ((4.0) (add x 1.0))
    (else x)))
(define (l-by-sum x)
  (case (l-sum x)
    ((10.0) (l-scale x 0.1))
    ((-6.0 -7.0) (l-scale x (l-sum x)))
    ((4.0) (l-shift x 1))
    (else x)))
(test-assert "case: symbolic key, =>, else" (agrees? by-sum l-by-sum inputs))
(test-equal "case: one graph, an equal test per datum, a where per clause"
  '(1 0 3 4)
  (list (stat by-sum 'traces) (stat by-sum 'graph-breaks)
        (stat by-sum 'branch-merges)
        (count (lambda (op) (eq? op 'equal)) (all-ops by-sum))))

(define-traced (case-else=> x)
  (case (sum x) ((10.0) x) (else => (lambda (k) (mul x k)))))
(test-assert "case: else =>"
  (agrees? case-else=> (lambda (x) (if (= (l-sum x) 10) x (l-scale x (l-sum x)))) inputs))

(define-traced (case-no-else x) (case (sum x) ((10.0) (mul x 2.0))))
(test-assert "case without else on a symbolic key cannot produce a value"
  (let ((msg (error-message (lambda () (case-no-else (t x-pos))))))
    (and msg (string-contains msg "cannot be joined"))))

;; case inside a loop (static iteration count)
(define-traced (loop-case x n)
  (let loop ((i 0) (acc x))
    (if (= i n)
        acc
        (loop (+ i 1)
              (case (sum acc)
                ((4.0) (mul acc 2.0))
                ((8.0 0.0) => (lambda (k) (sub acc 1.0)))
                (else (add acc 1.0)))))))
(define (l-loop-case x n)
  (let loop ((i 0) (acc x))
    (if (= i n)
        acc
        (loop (+ i 1)
              (case (l-sum acc)
                ((4.0) (l-scale acc 2))
                ((8.0 0.0) (l-shift acc -1))
                (else (l-shift acc 1)))))))
(test-assert "case in a loop"
  (agrees? (lambda (x) (loop-case x 4)) (lambda (x) (l-loop-case x 4)) inputs))
(test-equal "case in a loop: one trace, unrolled (2 clause joins x 4 iterations), no break"
  '(1 0 8) (stats loop-case '(traces graph-breaks branch-merges)))

;;; --- do ----------------------------------------------------------------------------------------

(define-traced (do-static x)
  ;; static iteration count, data-dependent step
  (do ((i 0 (+ i 1))
       (acc x (if (gt (sum acc) 0.0) (mul acc 0.5) (add acc 1.0))))
      ((= i 3) acc)))
(define (l-do-static x)
  (do ((i 0 (+ i 1))
       (acc x (if (> (l-sum acc) 0) (l-scale acc 0.5) (l-shift acc 1))))
      ((= i 3) acc)))
(test-assert "do: static count, symbolic if in the step" (agrees? do-static l-do-static inputs))
(test-equal "do: unrolled into one graph (3 where), no break"
  '(1 0 3) (stats do-static '(traces graph-breaks branch-merges)))

(define-traced (do-until x)
  ;; the termination test itself is symbolic: a data-dependent loop
  (do ((acc x (mul acc 0.5))) ((lt (sum acc) 1.0) acc)))
(define (l-do-until x)
  (do ((acc x (l-scale acc 0.5))) ((< (l-sum acc) 1) acc)))
(define do-inputs (list x-pos x-mid x-ones '((0.1 0.1) (0.1 0.1))))
(test-assert "do: symbolic termination test (data-dependent iteration count)"
  (agrees? do-until l-do-until do-inputs))
(test-assert "do: data-dependent loop falls back to graph breaks, then hits the cache"
  (and (> (stat do-until 'loop-breaks) 0) (> (stat do-until 'graph-breaks) 0)
       (= (stat do-until 'cache-hits) (+ (length do-inputs) 0))))

;;; --- named let ------------------------------------------------------------------------------

(define-traced (halve-until x)
  (let loop ((acc x))
    (if (gt (sum acc) 1.0) (loop (mul acc 0.5)) acc)))
(test-assert "named let: symbolic exit test (data-dependent recursion)"
  (agrees? halve-until
           (lambda (x) (let loop ((acc x)) (if (> (l-sum acc) 1) (loop (l-scale acc 0.5)) acc)))
           do-inputs))
(test-assert "named let: re-entry detected, resolved with breaks"
  (> (stat halve-until 'loop-breaks) 0))

(define-traced (nested-loops x)
  ;; named let around do, with cond => inside the do step
  (let outer ((j 0) (acc x))
    (if (= j 2)
        acc
        (outer (+ j 1)
               (do ((i 0 (+ i 1))
                    (a acc (cond ((gt (sum a) 4.0) => (lambda (c) (where c (mul a 0.5) a)))
                                 ((lt (sum a) 0.0) (neg a))
                                 (else (add a 1.0)))))
                   ((= i 2) a))))))
(define (l-nested-loops x)
  (let outer ((j 0) (acc x))
    (if (= j 2)
        acc
        (outer (+ j 1)
               (do ((i 0 (+ i 1))
                    (a acc (cond ((> (l-sum a) 4) (l-scale a 0.5))
                                 ((< (l-sum a) 0) (l-neg a))
                                 (else (l-shift a 1)))))
                   ((= i 2) a))))))
(test-assert "named let + do + cond => nested" (agrees? nested-loops l-nested-loops inputs))
(test-equal "named let + do + cond =>: one graph, no break"
  '(1 0) (stats nested-loops '(traces graph-breaks)))

;;; --- cond => ----------------------------------------------------------------------------------

(define-traced (cond-arrow x)
  (cond ((gt (sum x) 5.0) => (lambda (c) (where c (mul x 0.1) x)))
        ((lt (sum x) 0.0) => (lambda (c) (neg x)))
        (else x)))
(define (l-cond-arrow x)
  (cond ((> (l-sum x) 5) (l-scale x 0.1)) ((< (l-sum x) 0) (l-neg x)) (else x)))
(test-assert "cond =>: receiver gets the symbolic condition" (agrees? cond-arrow l-cond-arrow inputs))

(define-traced (case-of-and x)
  ;; case inside a cond => receiver, and inside a case clause
  (cond ((gt (sum x) -100.0)
         => (lambda (c)
              (case (sum x)
                ((10.0 2.0) (if (and c (lt (sum x) 5.0)) (mul x 4.0) (mul x 0.5)))
                (else x))))
        (else x)))
(define (l-case-of-and x)
  (case (l-sum x)
    ((10.0 2.0) (if (< (l-sum x) 5) (l-scale x 4) (l-scale x 0.5)))
    (else x)))
(test-assert "cond => around case around if-of-and" (agrees? case-of-and l-case-of-and inputs))

;;; --- concrete (Scheme) semantics are preserved -------------------------------------------

(define-traced (concrete x n)
  (let ((v (list (and 1 n) (or #f n) (and) (or)
                 (case n ((1) 'one) ((3 4) => (lambda (k) (* k 10))) (else 'other))
                 (do ((i 0 (+ i 1)) (s 0 (+ s i))) ((= i n) s))
                 (let loop ((i n) (acc '())) (if (zero? i) acc (loop (- i 1) (cons i acc))))
                 (cond ((assv n '((3 . three))) => cdr) (else 'none)))))
    (if (equal? v (list 3 3 #t #f 30 3 '(1 2 3) 'three)) x (neg x))))
(test-assert "concrete values keep R7RS semantics in every rewritten form"
  (close? x-pos (ctensor->nested (concrete (t x-pos) 3))))

(test-equal "trace-opaque: the form is left as written (core if: a tensor is true)"
  (l-neg x-pos)                         ; sum is 10, yet the `then` arm ran
  (ctensor->nested ((traced-lambda (x) (trace-opaque (if (lt (sum x) 0.0) (neg x) x)))
                    (t x-pos))))

;;; --- strict mode ----------------------------------------------------------------------------

(define (plain-abs x) (if (gt (sum x) 0.0) x (neg x)))   ; plain define: core if
(define-traced (traced-abs x) (if (gt (sum x) 0.0) x (neg x)))
(define (plain-helper x) (mul x 2.0))

(let* ((f (traced-lambda* (#:strict? #t) (x) (add (plain-abs x) 1.0)))
       (err (stderr-of (lambda () (f (t x-neg))))))
  (test-assert "strict: symbolic tensor passed to a plain procedure is reported"
    (and (= 1 (stat f 'strict-warnings))
         (string-contains (car (trace-strict-warnings f)) "plain-abs")
         (string-contains err "sonicboom-trace: strict: procedure plain-abs")))
  (test-assert "strict: the hazard it reports (plain-abs took the tensor as true)"
    (close? (l-shift x-neg 1) (ctensor->nested (f (t x-neg))))))

(let ((f (traced-lambda* (#:strict? #t) (x) (add (traced-abs x) 1.0))))
  (stderr-of (lambda () (f (t x-neg))))
  (test-equal "strict: the same helper written with define-traced is silent and correct"
    (list 0 (l-shift (l-neg x-neg) 1))
    (list (stat f 'strict-warnings) (ctensor->nested (f (t x-neg))))))

(let ((f (traced-lambda* (#:strict? #t) (x)
           (define (inner y) (if (gt (sum y) 0.0) y (neg y)))   ; internal define
           (let loop ((i 0) (acc x))                              ; named let
             (if (= i 2)
                 (car (map (lambda (z) (inner z)) (list acc)))    ; covered lambda via map
                 (loop (+ i 1) (relu (add acc 1.0))))))))         ; ops
  (stderr-of (lambda () (f (t x-neg))))
  (test-equal "strict: ops, internal define, named let, map over a covered lambda: silent"
    0 (stat f 'strict-warnings)))

(let ((f (traced-lambda* (#:strict? #t) (x)
           (car (map plain-helper (list x (plain-helper x)))))))
  (stderr-of (lambda () (f (t x-pos))))
  (test-assert "strict: map over a plain procedure reports the procedure, once"
    (and (= 1 (stat f 'strict-warnings))
         (string-contains (car (trace-strict-warnings f)) "plain-helper"))))

(let ((f (traced-lambda (x) (add (plain-abs x) 1.0))))
  (stderr-of (lambda () (f (t x-neg))))
  (test-equal "strict mode is off by default" 0 (stat f 'strict-warnings)))

(let ((f (traced-lambda (x) (add (plain-abs x) 1.0))))
  (parameterize ((strict-tracing #t))
    (stderr-of (lambda () (f (t x-neg)))))
  (test-equal "strict mode via the strict-tracing parameter" 1 (stat f 'strict-warnings)))

;;; --- loop unroll limit: host loops ------------------------------------------------------

(define w22 (t '((0.5 0.1) (0.2 0.9))))
(define (l-matvec a w)                ; a: (1 2) list-of-lists, w: 2x2
  (list (list (+ (* (caar a) (caar w)) (* (cadar a) (caadr w)))
              (+ (* (caar a) (cadar w)) (* (cadar a) (cadadr w))))))
(define (l-decay x n)
  (do ((i 0 (+ i 1)) (acc x (deep-map (lambda (v) (max 0 (+ v 0.01))) (l-matvec acc '((0.5 0.1) (0.2 0.9))))))
      ((= i n) (l-sum acc))))

(define-traced (decay x w n)
  (do ((i 0 (+ i 1)) (acc x (relu (add (matmul acc w) 0.01)))) ((= i n) (sum acc))))

(let* ((x '((1.0 2.0)))
       (before (interp-body-cache-stats))
       (r1 (decay (t x) w22 100))
       (mid (interp-body-cache-stats))
       (r2 (decay (t x) w22 100))
       (after (interp-body-cache-stats)))
  (test-assert "do: 100 iterations (> max-unroll 64) match plain Scheme"
    (and (close? (l-decay x 100) (ctensor-item r1)) (close? (l-decay x 100) (ctensor-item r2))))
  (test-equal "do: one host loop node, iteration bodies 2..100 matched the cached body"
    '(1 100 99) (stats decay '(host-loops loop-body-traces loop-body-hits)))
  (test-equal "do: the graph is one loop node plus the exit (no unrolled iterations)"
    '(loop reduce_sum) (node-ops (cdar (trace-segments decay))))
  (test-equal "do: the body graph is parsed once, then every iteration reuses it"
    (list 1 99 100)
    (list (- (car mid) (car before)) (- (cdr mid) (cdr before)) (- (cdr after) (cdr mid))))
  (test-equal "do: the second call is a cache hit running one graph" '(1 2)
    (stats decay '(cache-hits executor-calls))))

(test-assert "do: 10 iterations (<= max-unroll) are unrolled"
  (and (close? (l-decay '((1.0 2.0)) 10) (ctensor-item (decay (t '((1.0 2.0))) w22 10)))
       (= 1 (stat decay 'host-loops))))     ; still only the 100-iteration one

(define-traced* (#:max-unroll 8) (halve-n x n)
  ;; named let, static trip count above a small limit
  (let loop ((i 0) (acc x))
    (if (= i n) acc (loop (+ i 1) (mul acc 0.5)))))
(test-assert "named let: 20 iterations with #:max-unroll 8 becomes a host loop"
  (and (close? (l-scale x-pos (expt 0.5 20)) (ctensor->nested (halve-n (t x-pos) 20)))
       (equal? '(1 20 19) (stats halve-n '(host-loops loop-body-traces loop-body-hits)))))

(define-traced* (#:max-unroll 4) (ramp x n)
  ;; the counter enters the tensor computation: bodies differ, one run each
  (do ((i 0 (+ i 1)) (acc x (add acc i))) ((= i n) acc)))
(test-assert "host loop whose body changes every iteration: one loop node per run"
  (and (close? (l-shift x-pos (* 1/2 9 10)) (ctensor->nested (ramp (t x-pos) 10)))
       (equal? '(10 10 0) (stats ramp '(host-loops loop-body-traces loop-body-hits)))))

(define-traced* (#:max-unroll #f) (decay-unrolled x w n)
  (do ((i 0 (+ i 1)) (acc x (relu (add (matmul acc w) 0.01)))) ((= i n) (sum acc))))
(test-assert "#:max-unroll #f always unrolls"
  (and (close? (l-decay '((1.0 2.0)) 100) (ctensor-item (decay-unrolled (t '((1.0 2.0))) w22 100)))
       (= 0 (stat decay-unrolled 'host-loops))
       (= 301 (length (node-ops (cdar (trace-segments decay-unrolled)))))))

;;; --- strict mode: trace-pure -------------------------------------------------------------

(define (plain-scale x) (mul x 3.0))             ; no `if`: safe, but plain
(trace-pure plain-scale)
(let ((f (traced-lambda* (#:strict? #t) (x) (add (plain-scale x) (plain-abs x)))))
  (stderr-of (lambda () (f (t x-pos))))
  (test-assert "strict: a trace-pure procedure is not reported; others still are"
    (and (= 1 (stat f 'strict-warnings))
         (string-contains (car (trace-strict-warnings f)) "plain-abs"))))

;;; --- portability: only compat.scm uses implementation-specific mechanisms -----------

;; Uses (an open paren and the primitive's name; compat's wrappers are
;; %-prefixed, so they do not match).
(define %guile-specific
  '("(call-with-prompt" "(abort-to-prompt" "(make-prompt-tag" "(%make-void-port"
    "(set-procedure-property!" "(procedure-property" "(make-hash-table" "(hash-ref"
    "(hash-set!" "(syntax-case" "(syntax-source" "(syntax-local-binding"
    "(datum->syntax" "(eval-when" "(throw " "(catch "))

(define (module-source name)
  (let ((path (search-path %load-path (string-append "sonicboom/trace/" name ".scm"))))
    (call-with-input-file path
      (lambda (p) (let loop ((acc '()))
                    (let ((line (read-line p)))
                      (if (eof-object? line)
                          (string-concatenate-reverse acc)
                          (loop (cons* "\n" (if (string-prefix? ";" (string-trim line)) "" line)
                                       acc)))))))))

(test-equal "only compat.scm uses prompts, port tricks, tables, procedure properties, syntax-case"
  '()
  (append-map (lambda (m)
                (let ((src (module-source m)))
                  (filter-map (lambda (w) (and (string-contains src w) (list m w)))
                              %guile-specific)))
              '("tensor" "ops" "graph" "dce" "interp" "symbolic" "tracer" "macro"
                "loop" "autodiff" "layers" "optim")))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-forms")
(exit (if (zero? failures) 0 1))
