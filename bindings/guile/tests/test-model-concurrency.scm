;;; SonicBoom — concurrent first forward: per-model single-flight compilation
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core GUILE_AUTO_COMPILE=0 \
;;;       guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-model-concurrency.scm
;;;
;;; Many threads race the first forward of a model. Exactly one compilation per
;;; (model, input shape) may happen; the others wait for it and reuse the
;;; executable. Different models and different input shapes compile
;;; independently, and a failed compilation raises in every caller without
;;; poisoning the model.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 threads)
             (sonicboom tensor)
             (sonicboom nn))

;;; --- helpers -----------------------------------------------------------------

;; A linear layer with explicit weight/bias values (flat lists).
(define (linear-with wvals bvals in out)
  (let* ((l (linear in out #:init 'zeros #:name 'l1))
         (w (layer-parameter-ref l 'weight))
         (b (layer-parameter-ref l 'bias)))
    (for-each (lambda (i v) (tensor-set! w i v))
              (iota (length wvals)) wvals)
    (when b
      (for-each (lambda (i v) (tensor-set! b i v))
                (iota (length bvals)) bvals))
    l))

;; Start N threads, release them together, and return one result per thread:
;; (ok . value) on success, (error key . args) when the thread raised.
(define (run-concurrently n thunk)
  (let* ((lock (make-mutex))
         (condv (make-condition-variable))
         (ready 0)
         (go? #f)
         (results (make-vector n #f))
         (threads
          (map (lambda (i)
                 (call-with-new-thread
                  (lambda ()
                    (lock-mutex lock)
                    (set! ready (+ ready 1))
                    (when (= ready n)
                      (set! go? #t)
                      (broadcast-condition-variable condv))
                    (let wait ()
                      (unless go?
                        (wait-condition-variable condv lock)
                        (wait)))
                    (unlock-mutex lock)
                    (vector-set! results i
                                 (catch #t
                                   (lambda () (cons 'ok (thunk i)))
                                   (lambda (key . args)
                                     (cons 'error (cons key args))))))))
               (iota n))))
    (for-each join-thread threads)
    (vector->list results)))

(define (all-ok? results) (every (lambda (r) (eq? (car r) 'ok)) results))
(define (all-error? results) (every (lambda (r) (eq? (car r) 'error)) results))
(define (values-of results) (map cdr results))

;; Internal per-model counter maintained by (sonicboom nn); see
;; model-executable.
(define (compilations model)
  ((module-ref (resolve-module '(sonicboom nn)) 'model-compilation-count)
   model))

;;; --- tests -------------------------------------------------------------------

(test-begin "sonicboom-model-concurrency")

;; 1. Many threads, one model, one input shape: one compilation, and each run
;;    sees its own input (the shared executable's buffers must not interleave).
(let* ((m (sequential (linear-with '(1.0 0.0) '(0.0) 2 1)))
       (before (compilations m))
       (results
        (run-concurrently
         8
         (lambda (i)
           (tensor->list
            (model-forward m (make-tensor (list (exact->inexact i) 100.0) '(1 2))))))))
  (test-assert "same shape: every thread succeeds" (all-ok? results))
  (test-equal "same shape: each run sees its own input"
              (map (lambda (i) (list (exact->inexact i))) (iota 8))
              (values-of results))
  (test-equal "same shape: exactly one compilation"
              1 (- (compilations m) before)))

;; 2. Two models racing on their own first forward: one compilation each, and
;;    neither model's weights leak into the other.
(let* ((m1 (sequential (linear-with '(1.0 0.0) '(0.0) 2 1)))
       (m2 (sequential (linear-with '(0.0 1.0) '(0.0) 2 1)))
       (models (vector m1 m2))
       (before1 (compilations m1))
       (before2 (compilations m2))
       (results
        (run-concurrently
         8
         (lambda (i)
           (let ((which (modulo i 2)))
             (cons which
                   (tensor->list
                    (model-forward (vector-ref models which)
                                   (make-tensor '(3.0 -4.0) '(1 2))))))))))
  (test-assert "two models: every thread succeeds" (all-ok? results))
  (test-equal "two models: each uses its own weight"
              '((0 . (3.0)) (1 . (-4.0)))
              (delete-duplicates (values-of results)))
  (test-equal "two models: one compilation each"
              '(1 1)
              (list (- (compilations m1) before1)
                    (- (compilations m2) before2))))

;; 3. One model, two input shapes: independent entries, one compilation each.
(let* ((m (sequential (linear-with '(1.0 1.0) '(0.0) 2 1)))
       (before (compilations m))
       (run (lambda (i)
              (if (even? i)
                  (cons 2 (tensor->list
                           (model-forward
                            m (make-tensor '(1.0 2.0 3.0 4.0) '(2 2)))))
                  (cons 4 (tensor->list
                           (model-forward
                            m (make-tensor
                               '(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                               '(4 2))))))))
       (results (run-concurrently 8 run)))
  (test-assert "two shapes: every thread succeeds" (all-ok? results))
  (test-equal "two shapes: each entry computes its own shape"
              '((2 3.0 7.0) (4 3.0 7.0 11.0 15.0))
              (delete-duplicates (values-of results)))
  (test-equal "two shapes: one compilation each"
              2 (- (compilations m) before)))

;; 4. A first forward whose compilation fails: every racing caller sees the
;;    error, the failure is not cached, and other shapes still work.
(let* ((m (sequential (linear-with '(1.0 0.0) '(0.0) 2 1)))
       (bad (make-tensor '(1.0 2.0 3.0) '(1 3)))   ; 3 features vs weight's 2
       (results (run-concurrently 8 (lambda (i) (model-forward m bad)))))
  (test-assert "failing compile: every caller raises" (all-error? results))
  (test-error "failing compile: the failure is not cached" #t
              (model-forward m bad))
  (test-equal "failing compile: the model is still usable"
              '(3.0)
              (tensor->list (model-forward m (make-tensor '(3.0 -4.0) '(1 2))))))

;; srfi-64 reports counts but does not set the process exit status; CTest has
;; to see a failure, so capture the runner before test-end clears it and turn
;; its failure count into our exit code.
(define %runner (test-runner-current))
(test-end "sonicboom-model-concurrency")
(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
