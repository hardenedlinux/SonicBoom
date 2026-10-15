;;; SonicBoom — compile-once executables: correctness, reuse and lifetime
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-executable.scm
;;;
;;; Checks that a graph compiled once and run many times gives the same bytes as
;;; the compile-per-call path, that repeated runs are stable, that the native
;;; handle is released exactly once, and that the text-keyed cache reuses one
;;; executable without changing the numerics.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom ffi)
             (sonicboom executable))

;;; --- a self-contained graph: y = x + x (parameter-free) -----------------------

(define add-graph-text
  "(sonicboom-s-expr
     (version 0 1)
     (graph
       (name \"add-self\")
       (inputs (input \"x\" (tensor float32 (shape 2 2))))
       (outputs (output \"y\"))
       (parameters)
       (nodes
         (node add (inputs \"x\" \"x\")
               (outputs (\"y\" (tensor float32 (shape 2 2))))
               (attrs)))))")

(define (f32-bv . xs)
  (let ((bv (make-bytevector (* 4 (length xs)) 0)))
    (let loop ((i 0) (rest xs))
      (unless (null? rest)
        (bytevector-ieee-single-native-set! bv (* 4 i) (exact->inexact (car rest)))
        (loop (+ i 1) (cdr rest))))
    bv))

(define input (f32-bv 1.0 -2.5 3.25 0.0))

(define (call-run exe bv)
  (call-with-values (lambda () (executable-run exe bv)) (lambda (out err) (cons out err))))

;;; --- tests --------------------------------------------------------------------

(test-begin "sonicboom-executable")

;; 1. compiled-once output is byte-identical to the compile-per-call path
(let* ((uncached (call-with-values (lambda () (run-s-expr add-graph-text input))
                   (lambda (out err) (cons out err))))
       (exe (compile-s-expr add-graph-text))
       (once (call-run exe input)))
  (test-assert "run-s-expr succeeds"
    (and (car uncached) (not (cdr uncached))))
  (test-assert "executable-run succeeds"
    (and (car once) (not (cdr once))))
  (test-assert "compiled-once output equals the uncached output"
    (equal? (car once) (car uncached)))
  (test-assert "add(x,x) doubles the input"
    (equal? (car once) (f32-bv 2.0 -5.0 6.5 0.0)))
  (test-equal "input byte size is reported from the graph" 16
    (executable-input-nbytes exe))
  (test-equal "output byte size is reported from the graph" 16
    (executable-output-nbytes exe))

  ;; 2. 100 repeated runs of one executable give identical outputs
  (test-assert "100 repeated runs of one executable are identical"
    (let loop ((i 0))
      (if (= i 100)
          #t
          (let ((out (car (call-run exe input))))
            (and (equal? out (car once))
                 (loop (+ i 1)))))))

  ;; 3. a mismatched input size is rejected, not silently accepted
  (test-assert "a wrong-sized input is reported as an error"
    (let ((r (call-run exe (make-bytevector 8 0))))
      (and (not (car r)) (string? (cdr r)))))

  ;; 4. lifetime: free is idempotent and a freed executable refuses to run
  (test-assert "fresh executable is not marked freed" (not (executable-freed? exe)))
  (free-executable! exe)
  (test-assert "free-executable! marks the handle freed" (executable-freed? exe))
  (test-assert "free-executable! is idempotent (no error on the second free)"
    (begin (free-executable! exe) (executable-freed? exe)))
  (test-assert "a freed executable returns an error instead of running"
    (let ((r (call-run exe input)))
      (and (not (car r)) (string? (cdr r))))))

;; 5. the cache returns one executable per graph text and keeps the numerics
(let* ((e1 (cached-executable add-graph-text))
       (e2 (cached-executable add-graph-text)))
  (test-assert "the cache returns the same object for the same text"
    (eq? e1 e2))
  (test-assert "cached-executable is not freed by a cache hit"
    (not (executable-freed? e1)))
  (test-assert "run-s-expr/cached matches the uncached bytes"
    (let ((r (call-with-values (lambda () (run-s-expr/cached add-graph-text input))
               (lambda (out err) (cons out err)))))
      (and (car r) (equal? (car r) (f32-bv 2.0 -5.0 6.5 0.0))))))

;; 6. disabling the cache falls back to the compile-per-call path, same bytes
(test-assert "with the cache disabled, run-s-expr/cached still computes the same"
  (parameterize ((executable-cache-enabled #f))
    (let ((r (call-with-values (lambda () (run-s-expr/cached add-graph-text input))
               (lambda (out err) (cons out err)))))
      (and (car r) (equal? (car r) (f32-bv 2.0 -5.0 6.5 0.0))))))

;; 7. clearing the cache releases the handles and leaves it usable
(test-assert "clear-executable-cache! drops freed handles and leaves a usable cache"
  (begin
    (clear-executable-cache!)
    (let ((r (call-with-values (lambda () (run-s-expr/cached add-graph-text input))
               (lambda (out err) (cons out err)))))
      (and (car r) (equal? (car r) (f32-bv 2.0 -5.0 6.5 0.0))))))

;; 8. the GC fallback: an executable that is dropped without an explicit free
;;    is handed back by the guardian and reaped.
(define (leak-one!)
  (let ((e (compile-s-expr add-graph-text)))
    (executable-run e input)            ; e never freed and never retained
    #t))
(leak-one!)
(gc) (gc)
(test-assert "the GC fallback reaps an executable that was never freed"
  (>= (reap-executables!) 1))
(test-equal "reaping again finds nothing new" 0 (reap-executables!))

;; 9. a malformed graph raises at compile time, not at run time
(test-error "a malformed graph raises at compile time"
  'misc-error (compile-s-expr "(this is not an s-expr document)"))

(test-end "sonicboom-executable")
