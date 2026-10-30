;;; SonicBoom — deferred (:bound) parameters and per-model executables (Stage 3)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core GUILE_AUTO_COMPILE=0 \
;;;       guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-model-binding.scm
;;;
;;; Checks the executable-level binding contract (compile without bytes, bind
;;; by name, run-completeness, one-time binding), and the model-level rules:
;;; the emitted text carries no weight values, each model instance owns its
;;; bound executable (same structure, different weights never share state),
;;; values are captured at the first forward, and :values / :external graphs
;;; keep their existing behavior.

(use-modules (srfi srfi-1)
             (srfi srfi-11)
             (srfi srfi-13)
             (srfi srfi-64)
             (rnrs bytevectors)
             (rnrs io ports)
             (sonicboom tensor)
             (sonicboom nn)
             (sonicboom executable))

;;; --- helpers -----------------------------------------------------------------

(define (f32-bv . xs)
  (let ((bv (make-bytevector (* 4 (length xs)) 0)))
    (let loop ((i 0) (rest xs))
      (unless (null? rest)
        (bytevector-ieee-single-native-set! bv (* 4 i)
                                            (exact->inexact (car rest)))
        (loop (+ i 1) (cdr rest))))
    bv))

(define b-bytes (f32-bv 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0))
(define x-input (f32-bv 1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0 8.0))
(define expected-z (f32-bv 1.5 1.0 0.0 1.0 0.0 1.0 0.0 1.0))

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

;;; --- the executable-level contract -------------------------------------------

(define bound-text
  (string-append
   "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_bound\")"
   " (inputs (input \"x\" (tensor float32 (shape 8))))"
   " (outputs (output \"z\"))"
   " (parameters (parameter \"b\" (tensor float32 (shape 8))"
   "   (data :bound \"b\")))"
   " (nodes"
   "   (node add (inputs \"x\" \"b\")"
   "     (outputs (\"y\" (tensor float32 (shape 8)))))"
   "   (node relu (inputs \"y\")"
   "     (outputs (\"z\" (tensor float32 (shape 8))))))))"))

(define values-text
  (string-append
   "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_values\")"
   " (inputs (input \"x\" (tensor float32 (shape 8))))"
   " (outputs (output \"z\"))"
   " (parameters (parameter \"b\" (tensor float32 (shape 8))"
   "   (data :values 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)))"
   " (nodes"
   "   (node add (inputs \"x\" \"b\")"
   "     (outputs (\"y\" (tensor float32 (shape 8)))))"
   "   (node relu (inputs \"y\")"
   "     (outputs (\"z\" (tensor float32 (shape 8))))))))"))

(define external-text
  (string-append
   "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_ext\")"
   " (inputs (input \"x\" (tensor float32 (shape 8))))"
   " (outputs (output \"z\"))"
   " (parameters (parameter \"b\" (tensor float32 (shape 8))"
   "   (data :external \"weights.bin\" 0 32)))"
   " (nodes"
   "   (node add (inputs \"x\" \"b\")"
   "     (outputs (\"y\" (tensor float32 (shape 8)))))"
   "   (node relu (inputs \"y\")"
   "     (outputs (\"z\" (tensor float32 (shape 8))))))))"))

(test-begin "sonicboom-model-binding")

;; 1. Compile succeeds without bytes; run fails until every binding is set.
(let ((exe (compile-s-expr bound-text)))
  (test-equal "bound: one parameter is enumerated"
              1 (executable-parameter-count exe))
  (test-equal "bound: the parameter keeps its graph name"
              "b" (executable-parameter-name exe 0))
  (test-assert "bound: the executable starts incomplete"
               (not (executable-bindings-complete? exe)))
  (let-values (((out err) (executable-run exe x-input)))
    (test-assert "bound: run before binding fails" (and (not out) err)))

  ;; 2. Binding failure paths.
  (test-error "an unknown binding name raises" #t
              (executable-bind-parameter! exe "nope" b-bytes))
  (test-error "a wrong byte count raises" #t
              (executable-bind-parameter! exe "b" (f32-bv 1.0)))
  (test-assert "a failed bind leaves the executable incomplete"
               (not (executable-bindings-complete? exe)))

  ;; 3. Binding completes the graph and run produces the :values result.
  (executable-bind-parameter! exe "b" b-bytes)
  (test-assert "bound: the executable is complete after binding"
               (executable-bindings-complete? exe))
  (let-values (((out err) (executable-run exe x-input)))
    (test-assert "bound: run after binding succeeds" (and out (not err)))
    (test-assert "bound: the result equals the :values graph's"
                 (and out (bytevector=? out expected-z))))

  ;; 4. Binding is a one-time initialization.
  (test-error "rebinding after a successful run raises" #t
              (executable-bind-parameter! exe "b" (f32-bv 9 9 9 9 9 9 9 9)))

  ;; 5. A Stage 2 view shows the bound bytes, is read-only, and releases
  ;;    independently of another view.
  (let* ((v1 (executable-parameter-ref exe 0))
         (v2 (executable-parameter-ref exe 0)))
    (test-assert "bound view: bytes equal the bound data"
                 (bytevector=? (tensor-data v1) b-bytes))
    (test-error "bound view: tensor-set! raises" #t (tensor-set! v1 0 99.0))
    (tensor-release! v1)
    (test-assert "bound view: the second view survives the first's release"
                 (bytevector=? (tensor-data v2) b-bytes))
    (tensor-release! v2))

  (free-executable! exe)
  (test-assert "free-executable! after binding is safe"
               (executable-freed? exe)))

;; 6. :values graphs keep their behavior: complete, no binding step.
(let ((exe (compile-s-expr values-text)))
  (test-assert ":values graph is complete without binding"
               (executable-bindings-complete? exe))
  (let-values (((out err) (executable-run exe x-input)))
    (test-assert ":values graph runs unchanged"
                 (and out (not err) (bytevector=? out expected-z))))
  (free-executable! exe))

;; 7. :external graphs keep their behavior (bytes from a sidecar). The Guile
;;    layer compiles with base_dir "", so the sidecar is resolved against the
;;    process cwd; run this block from a private directory.
(let* ((dir (string-append "/tmp/sonicboom-binding-"
                           (number->string (getpid))))
       (cwd (getcwd)))
  (mkdir dir)
  (call-with-output-file (string-append dir "/weights.bin")
    (lambda (port) (put-bytevector port b-bytes)))
  (dynamic-wind
    (lambda () (chdir dir))
    (lambda ()
      (let ((exe (compile-s-expr external-text)))
        (test-assert ":external graph is complete without binding"
                     (executable-bindings-complete? exe))
        (let-values (((out err) (executable-run exe x-input)))
          (test-assert ":external graph runs unchanged"
                       (and out (not err) (bytevector=? out expected-z))))
        (free-executable! exe)))
    (lambda () (chdir cwd)))
  (delete-file (string-append dir "/weights.bin"))
  (rmdir dir))

;; 8. Same structure, different weights: identical text, independent state.
(let* ((m1 (sequential (linear-with '(1.0 0.0) '(0.0) 2 1)))
       (m2 (sequential (linear-with '(0.0 1.0) '(0.0) 2 1)))
       (x (make-tensor '(3.0 -4.0) '(1 2))))
  (test-assert "the emitted text is parameter-independent"
               (string=? (model-graph-text m1 '(1 2))
                         (model-graph-text m2 '(1 2))))
  (test-assert "the text declares parameters :bound, not :values"
               (let ((text (model-graph-text m1 '(1 2))))
                 (and (string-contains text "(data :bound \"l1.weight\")")
                      (string-contains text "(data :bound \"l1.bias\")")
                      (not (string-contains text ":values")))))
  (test-equal "model 1 uses its own weight"
              '(3.0) (tensor->list (model-forward m1 x)))
  (test-equal "model 2 uses its own weight"
              '(-4.0) (tensor->list (model-forward m2 x)))
  (test-equal "model 1 is unaffected by model 2's forward"
              '(3.0) (tensor->list (model-forward m1 x))))

;; 9. Weights are captured at the first forward (bind once, then run).
(let* ((l (linear-with '(1.0 0.0) '(0.0) 2 1))
       (m (sequential l))
       (x (make-tensor '(3.0 -4.0) '(1 2))))
  ;; Update before the first forward: visible.
  (tensor-set! (model-parameter-ref m 'l1.weight) 0 0.0)
  (tensor-set! (model-parameter-ref m 'l1.weight) 1 1.0)
  (test-equal "an update before the first forward is bound"
              '(-4.0) (tensor->list (model-forward m x)))
  ;; Update after the first forward: not re-bound.
  (tensor-set! (model-parameter-ref m 'l1.weight) 0 1.0)
  (tensor-set! (model-parameter-ref m 'l1.weight) 1 0.0)
  (test-equal "an update after the first forward is not re-bound"
              '(-4.0) (tensor->list (model-forward m x))))

;; 10. With the executable cache disabled the graph is compiled per call and
;;     bound from the current tensors each time.
(let* ((m (sequential (linear-with '(1.0 0.0) '(0.0) 2 1)))
       (x (make-tensor '(3.0 -4.0) '(1 2))))
  (test-equal "cached forward uses the current weight"
              '(3.0) (tensor->list (model-forward m x)))
  (executable-cache-enabled #f)
  (tensor-set! (model-parameter-ref m 'l1.weight) 0 0.0)
  (tensor-set! (model-parameter-ref m 'l1.weight) 1 1.0)
  (test-equal "cache off recompiles and sees the updated weight"
              '(-4.0) (tensor->list (model-forward m x)))
  (executable-cache-enabled #t))

;; srfi-64 reports counts but does not set the process exit status; CTest has
;; to see a failure, so capture the runner before test-end clears it and turn
;; its failure count into our exit code.
(define %runner (test-runner-current))
(test-end "sonicboom-model-binding")
(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
