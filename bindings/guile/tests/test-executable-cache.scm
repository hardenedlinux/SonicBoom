;;; SonicBoom — text-cache identity and ownership contracts (public API)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core GUILE_AUTO_COMPILE=0 \
;;;       guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-executable-cache.scm
;;;
;;; The process-wide cache in (sonicboom executable) is keyed by graph text, so
;;; it may only own executables whose behaviour is fully determined by that text.
;;; These are the externally observable halves of that contract, exercised
;;; through the public procedures only (cached-executable, executable-run,
;;; free-executable!, clear-executable-cache!):
;;;
;;;   * a self-contained (:values) graph is cached and reused;
;;;   * a :bound graph is never shared: two owners backing the same text with
;;;     different bytes keep independent executables and independent results;
;;;   * a :external graph is never reused across working directories or after
;;;     its sidecar changes, because the bytes are copied at compile time;
;;;   * free-executable! evicts the corresponding cache entry, so a later
;;;     lookup compiles a fresh handle instead of returning a released one;
;;;   * repeated compile / evict / release stays correct (leak-checked under
;;;     ASan/LSan in the quality gate).
;;;
;;; Internal cache-table invariants and race orchestration live in the private
;;; suite.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (rnrs bytevectors)
             (rnrs io ports)
             (sonicboom executable))

;;; --- graphs ------------------------------------------------------------------

;; y = x + x: parameter-free, fully determined by the text.
(define values-text
  "(sonicboom-s-expr (version 0 1) (graph (name \"add_self_cache\")
     (inputs (input \"x\" (tensor float32 (shape 2))))
     (outputs (output \"y\"))
     (parameters)
     (nodes (node add (inputs \"x\" \"x\")
                  (outputs (\"y\" (tensor float32 (shape 2))))))))")

;; y = x + b, b supplied after compile: the text does not identify the bytes.
(define bound-text
  "(sonicboom-s-expr (version 0 1) (graph (name \"add_bound_cache\")
     (inputs (input \"x\" (tensor float32 (shape 2))))
     (outputs (output \"y\"))
     (parameters (parameter \"b\" (tensor float32 (shape 2))
                   (data :bound \"b\")))
     (nodes (node add (inputs \"x\" \"b\")
                  (outputs (\"y\" (tensor float32 (shape 2))))))))")

;; y = x + b, b copied from a sidecar resolved against the process cwd.
(define external-text
  "(sonicboom-s-expr (version 0 1) (graph (name \"add_ext_cache\")
     (inputs (input \"x\" (tensor float32 (shape 2))))
     (outputs (output \"y\"))
     (parameters (parameter \"b\" (tensor float32 (shape 2))
                   (data :external \"weights.bin\" 0 8)))
     (nodes (node add (inputs \"x\" \"b\")
                  (outputs (\"y\" (tensor float32 (shape 2))))))))")

;;; --- helpers -----------------------------------------------------------------

(define (f32-bv . xs)
  (let ((bv (make-bytevector (* 4 (length xs)) 0)))
    (let loop ((i 0) (rest xs))
      (unless (null? rest)
        (bytevector-ieee-single-native-set! bv (* 4 i) (exact->inexact (car rest)))
        (loop (+ i 1) (cdr rest))))
    bv))

(define (bv->floats bv)
  (let loop ((i (- (/ (bytevector-length bv) 4) 1)) (acc '()))
    (if (< i 0)
        acc
        (loop (- i 1)
              (cons (bytevector-ieee-single-native-ref bv (* 4 i)) acc)))))

(define (run exe bv)
  (call-with-values (lambda () (executable-run exe bv))
    (lambda (out err) (cons out err))))

(define (run-floats exe bv)
  (let ((r (run exe bv)))
    (and (car r) (bv->floats (car r)))))

(define (clear!)
  (clear-executable-cache!)
  ;; Reaping frees the evicted handles once they are unreachable.
  (gc) (gc)
  (reap-executables!))

(define (dir-name tag)
  (string-append "/tmp/sonicboom-executable-cache-" tag "-"
                 (number->string (getpid))))

(define (write-weights! path . xs)
  (call-with-output-file path
    (lambda (port) (put-bytevector port (apply f32-bv xs)))))

;;; --- tests -------------------------------------------------------------------

(test-begin "sonicboom-executable-cache")

;; 1. Self-contained caching is preserved.
(clear!)
(let ((e1 (cached-executable values-text))
      (e2 (cached-executable values-text)))
  (test-assert ":values text returns one cache-owned executable" (eq? e1 e2))
  (test-assert "a cached handle is not freed" (not (executable-freed? e1)))
  (test-equal "the cached executable computes the right bytes"
              '(2.0 4.0) (run-floats e1 (f32-bv 1.0 2.0))))

;; 2. A :bound graph is not shared between independent owners: each lookup gets
;;    its own executable and its own bound bytes.
(clear!)
(let ((e1 (cached-executable bound-text))
      (e2 (cached-executable bound-text)))
  (test-assert "a :bound graph is not shared between owners" (not (eq? e1 e2)))
  (executable-bind-parameter! e1 "b" (f32-bv 10.0 20.0))
  (executable-bind-parameter! e2 "b" (f32-bv -1.0 -2.0))
  (test-equal "the first owner keeps its own bound bytes"
              '(11.0 22.0) (run-floats e1 (f32-bv 1.0 2.0)))
  (test-equal "the second owner keeps its own bound bytes"
              '(0.0 0.0) (run-floats e2 (f32-bv 1.0 2.0)))
  (free-executable! e1)
  (test-assert "freeing one owner does not free the other's handle"
               (not (executable-freed? e2)))
  (free-executable! e2))

;; 3. The same :external text compiled from two working directories resolves
;;    each directory's sidecar: distinct executables, distinct bytes.
(clear!)
(let ((dir-a (dir-name "a"))
      (dir-b (dir-name "b"))
      (cwd (getcwd)))
  (mkdir dir-a)
  (mkdir dir-b)
  (write-weights! (string-append dir-a "/weights.bin") 1.0 2.0)
  (write-weights! (string-append dir-b "/weights.bin") 3.0 4.0)
  (let ((e-a #f) (e-b #f))
    (dynamic-wind
      (lambda () (chdir dir-a))
      (lambda () (set! e-a (cached-executable external-text)))
      (lambda () (chdir cwd)))
    (dynamic-wind
      (lambda () (chdir dir-b))
      (lambda () (set! e-b (cached-executable external-text)))
      (lambda () (chdir cwd)))
    (test-assert "same text in different directories: distinct executables"
                 (not (eq? e-a e-b)))
    (test-equal "directory A's sidecar bytes are used"
                '(1.5 1.5) (run-floats e-a (f32-bv 0.5 -0.5)))
    (test-equal "directory B's sidecar bytes are used"
                '(3.5 3.5) (run-floats e-b (f32-bv 0.5 -0.5)))
    (free-executable! e-a)
    (free-executable! e-b))
  (delete-file (string-append dir-a "/weights.bin"))
  (delete-file (string-append dir-b "/weights.bin"))
  (rmdir dir-a)
  (rmdir dir-b))

;; 4. A changed sidecar yields a new executable: a cache hit here would keep
;;    the old bytes silently.
(clear!)
(let ((dir (dir-name "mutate"))
      (cwd (getcwd)))
  (mkdir dir)
  (write-weights! (string-append dir "/weights.bin") 1.0 2.0)
  (let ((e1 #f) (e2 #f))
    (dynamic-wind
      (lambda () (chdir dir))
      (lambda ()
        (set! e1 (cached-executable external-text))
        (write-weights! (string-append dir "/weights.bin") 5.0 6.0)
        (set! e2 (cached-executable external-text)))
      (lambda () (chdir cwd)))
    (test-assert "a changed sidecar yields a new executable" (not (eq? e1 e2)))
    (test-equal "the first executable keeps the old bytes"
                '(1.5 1.5) (run-floats e1 (f32-bv 0.5 -0.5)))
    (test-equal "the second executable sees the new bytes"
                '(5.5 5.5) (run-floats e2 (f32-bv 0.5 -0.5)))
    (free-executable! e1)
    (free-executable! e2))
  (delete-file (string-append dir "/weights.bin"))
  (rmdir dir))

;; 5. Freeing a cache-owned handle (a contract violation) evicts the entry: the
;;    next lookup compiles a fresh handle instead of returning a released one.
(clear!)
(let ((e1 (cached-executable values-text)))
  (free-executable! e1)
  (test-assert "the freed handle reports freed" (executable-freed? e1))
  (let ((e2 (cached-executable values-text)))
    (test-assert "a lookup after the free is not the released handle"
                 (not (eq? e1 e2)))
    (test-assert "the replacement handle is live"
                 (not (executable-freed? e2)))
    (test-equal "the replacement handle runs"
                '(2.0 4.0) (run-floats e2 (f32-bv 1.0 2.0)))))

;; 6. Repeated compile / evict / release stays correct and leaves nothing
;;    behind for the guardian or the leak checker.
(clear!)
(test-assert "repeated compile/evict/release keeps working"
             (let loop ((i 0))
               (if (= i 25)
                   #t
                   (let ((e (cached-executable values-text)))
                     (clear-executable-cache!)
                     (let ((outs (run-floats e (f32-bv 1.0 2.0))))
                       (when (not (equal? outs '(2.0 4.0)))
                         (error "repeated cache cycle produced wrong bytes" outs)))
                     (free-executable! e)
                     (loop (+ i 1))))))
(gc) (gc)
(reap-executables!)
(test-assert "the cache still works after the cycles"
             (equal? (run-floats (cached-executable values-text) (f32-bv 1.0 2.0))
                     '(2.0 4.0)))

;; srfi-64 reports counts but does not set the process exit status; CTest has
;; to see a failure, so capture the runner before test-end clears it and turn
;; its failure count into our exit code.
(define %runner (test-runner-current))
(test-end "sonicboom-executable-cache")
(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
