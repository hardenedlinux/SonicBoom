;;; SonicBoom — Guile trainable-parameter (fused AdamW) binding test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; (sonicboom param) drives the C ABI's fused AdamW parameter surface. The
;;; 10-step replay is tests/core/test_adamw.cpp's deterministic sequence:
;;;
;;;   init[i]    = 0.4*sin(0.7*i) + 0.05*(i mod 3)
;;;   grad_t[i]  = 0.5*cos(0.31*t + 0.37*i)*(1 + 0.05*i) + 0.01*t
;;;
;;; compared against an independent double-precision AdamW reference (the
;;; numpy formula, with the moment buffers rounded to float32 each step, as the
;;; kernel stores them). Gate: max-abs <= 1e-6. The lr-input form (a one-element
;;; float32 tensor handle) must agree with the scalar form bit for bit.
;;;
;;; Runs on the host path without a GPU and on the fused device path with one.
;;; Skips cleanly if libsonicboom.so cannot be loaded.

(use-modules (srfi srfi-1)
             (srfi srfi-34)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom error)
             (sonicboom ffi)
             (sonicboom param)
             (sonicboom train))

(define kN 17)                          ; odd: exercises the grid-stride tail

;; The float32 value of X, as a Scheme number: the constant a <param> and the
;; C reference both see.
(define (f32 x) (car (f32-bv->list (list->f32-bv (list x)))))

(define lr (f32 0.02))
(define b1 (f32 0.9))
(define b2 (f32 0.999))
(define eps (f32 1e-8))
(define wd (f32 0.01))

(define (init-values)
  (map (lambda (i)
         (f32 (+ (* 0.4 (sin (* 0.7 i))) (* 0.05 (modulo i 3)))))
       (iota kN)))

(define (grad-values t)
  (map (lambda (i)
         (f32 (+ (* 0.5 (cos (+ (* 0.31 t) (* 0.37 i)))
                    (+ 1 (* 0.05 i)))
                 (* 0.01 t))))
       (iota kN)))

;; A stateful double-precision AdamW reference; each call applies one step and
;; returns the current parameter vector.
(define (make-adamw-ref init)
  (let ((p (list->vector init))
        (m (make-vector (length init) 0.0))
        (v (make-vector (length init) 0.0))
        (n (length init)))
    (lambda (t gstep)
      (let ((bc1 (- 1.0 (expt b1 t)))
            (bc2 (- 1.0 (expt b2 t))))
        (let loop ((i 0))
          (when (< i n)
            (let* ((g (list-ref gstep i))
                   (mi (+ (* b1 (vector-ref m i)) (* (- 1.0 b1) g)))
                   (vi (+ (* b2 (vector-ref v i)) (* (- 1.0 b2) g g)))
                   (mh (/ mi bc1))
                   (vh (/ vi bc2))
                   (pval (vector-ref p i)))
              ;; The moments are float32 state, as the kernel stores them; the
              ;; update itself uses the unrounded doubles, as the reference does.
              (vector-set! m i (f32 mi))
              (vector-set! v i (f32 vi))
              (vector-set! p i
                           (f32 (- pval
                                   (* lr (+ (/ mh (+ (sqrt vh) eps))
                                            (* wd pval))))))
              (loop (+ i 1)))))
        p))))

(define (max-abs-diff bv pvec)
  (let loop ((i 0) (worst 0.0))
    (if (= i kN)
        worst
        (loop (+ i 1)
              (max worst
                   (abs (- (bytevector-ieee-single-native-ref bv (* 4 i))
                           (vector-ref pvec i))))))))

;; The 10-step replay, comparing against the double reference after every step.
(define (replay! p)
  (let ((ref-step (make-adamw-ref (init-values))))
    (let loop ((t 1) (worst 0.0))
      (if (> t 10)
          worst
          (let ((g (grad-values t)))
            (param-set-grad! p (list->f32-bv g))
            (param-step! p lr b1 b2 eps wd)
            (let ((ref (ref-step t g)))
              (loop (+ t 1)
                    (max worst (max-abs-diff (param-read p) ref)))))))))

(define schedule (map f32 '(0.05 0.02 0.01 0.005 0.002
                            0.01 0.04 0.03 0.015 0.0075)))

;; The lr-input form against the scalar form: two fresh parameters see the same
;; gradient sequence, one updated with the rate as a number and one with the
;; rate as a one-element float32 tensor handle. Returns
;; (agree? a-step-count b-step-count).
(define (run-lr-forms! init)
  (let ((a (make-param (list kN) #:init (list->f32-bv init)))
        (b (make-param (list kN) #:init (list->f32-bv init))))
    (let loop ((t 1) (agree #t))
      (if (<= t 10)
          (let* ((g (grad-values t))
                 (lr-t (list-ref schedule (- t 1)))
                 (tensor (tensor-from-list (list 1) (list lr-t))))
            (param-set-grad! a (list->f32-bv g))
            (param-set-grad! b (list->f32-bv g))
            (param-step! a lr-t b1 b2 eps wd)
            (param-step! b tensor b1 b2 eps wd)
            (tensor-free! tensor)
            (loop (+ t 1)
                  (and agree (equal? (param-read a) (param-read b)))))
          (let ((result (list agree
                              (param-step-count a)
                              (param-step-count b))))
            (free-param! a)
            (free-param! b)
            result)))))

;; The GC fallback: a parameter that is dropped without an explicit free is
;; handed back by the guardian and reaped.
(define (leak-one!)
  (let ((q (make-param (list 2))))
    (param-read q)
    #t))

(test-begin "sonicboom-param")

(if (not (libsonicboom-available?))
    (test-skip 17)
    (begin
      (let* ((init (init-values))
             (p (make-param (list kN) #:init (list->f32-bv init))))
        (test-equal "param-numel" kN (param-numel p))
        (test-assert "param-on-device? is a boolean"
          (boolean? (param-on-device? p)))
        (test-equal "the step count starts at 0" 0 (param-step-count p))
        (test-equal "param-read returns numel*4 bytes"
          (* 4 kN) (bytevector-length (param-read p)))

        (let ((worst (replay! p)))
          (format #t "adamw: 10-step max-abs vs double reference = ~,3e~%"
                  worst)
          (test-assert "the 10-step replay matches the double reference"
            (<= worst 1e-6)))
        (test-equal "the step count advanced 10 times"
          10 (param-step-count p))

        ;; zero_grad: a step with a zeroed gradient is pure weight decay.
        (param-zero-grad! p)
        (param-step! p lr b1 b2 eps wd)
        (test-equal "the step count advanced with zero_grad"
          11 (param-step-count p))
        (let ((ref-step (make-adamw-ref init)))
          (let loop ((t 1))
            (when (<= t 10)
              (ref-step t (grad-values t))
              (loop (+ t 1))))
          (let ((ref (ref-step 11 (map (lambda (i) 0.0) (iota kN)))))
            (test-assert "a zero_grad step matches the reference"
              (<= (max-abs-diff (param-read p) ref) 1e-6))))
        (free-param! p))

      (let ((result (run-lr-forms! (init-values))))
        (test-assert "the lr-tensor form is bit-identical to the scalar form"
          (car result))
        (test-assert "both lr forms advanced 10 steps"
          (and (= 10 (cadr result)) (= 10 (caddr result)))))

      ;; Error paths: typed binding errors, never a call on a dangling handle.
      (test-assert "a freed parameter is a binding error"
        (let ((q (make-param (list 2))))
          (free-param! q)
          (guard (c ((sonicboom-error? c)
                     (eq? 'binding (sonicboom-error-kind c))))
            (param-read q)
            #f)))

      (test-assert "free-param! is idempotent"
        (let ((q (make-param (list 2))))
          (free-param! q)
          (free-param! q)
          #t))

      (test-assert "a zero dimension is a binding error"
        (guard (c ((sonicboom-error? c)
                   (eq? 'binding (sonicboom-error-kind c))))
          (make-param (list 0))
          #f))

      (test-assert "a short init is a binding error"
        (guard (c ((sonicboom-error? c)
                   (eq? 'binding (sonicboom-error-kind c))))
          (make-param (list 4) #:init (make-bytevector 12 0))
          #f))

      (test-assert "a short gradient is a binding error"
        (let ((q (make-param (list 4))))
          (guard (c ((sonicboom-error? c)
                     (eq? 'binding (sonicboom-error-kind c))))
            (param-set-grad! q (make-bytevector 12 0))
            #f)))

      (leak-one!)
      (gc) (gc)
      (test-assert "the GC fallback reaps a parameter that was never freed"
        (>= (reap-params!) 1))
      (test-equal "reaping again finds nothing new" 0 (reap-params!))))

(test-end "sonicboom-param")
