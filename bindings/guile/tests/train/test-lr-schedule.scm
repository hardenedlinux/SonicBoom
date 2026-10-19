;;; SonicBoom — Guile learning-rate schedule tests
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/train/test-lr-schedule.scm
;;;
;;; The CS336 A1 gate: warmup starts at zero and rises monotonically to the
;;; base rate; the cosine equals the base rate at the end of warmup and the
;;; minimum at total-steps; the constant and linear-warmup schedules hold
;;; their values.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 format)
             (sonicboom train lr-schedule))

(define BASE 3e-4)
(define LOW 3e-5)
(define WARM 100)
(define TOTAL 1000)

(define (approx= a b) (<= (abs (- a b)) 1e-6))

(define cosine (make-cosine-schedule BASE LOW WARM TOTAL))
(define constant (make-constant-schedule BASE))
(define linear (make-linear-warmup-schedule BASE WARM))

(test-begin "sonicboom-lr-schedule")

;;; --- cosine with warmup -----------------------------------------------------

(test-assert "cosine starts at 0"
  (= 0 (cosine 0)))

(test-assert "cosine rises monotonically through warmup"
  (let loop ((s 1))
    (or (> s WARM)
        (and (> (cosine s) (cosine (- s 1)))
             (loop (+ s 1))))))

(test-assert "cosine equals base-lr at the end of warmup"
  (approx= BASE (cosine WARM)))

(test-assert "cosine equals min-lr at total-steps"
  (approx= LOW (cosine TOTAL)))

(test-assert "cosine stays at min-lr past total-steps"
  (and (approx= LOW (cosine (+ TOTAL 1)))
       (approx= LOW (cosine (* 10 TOTAL)))))

(test-assert "cosine decays monotonically from warmup to total-steps"
  (let loop ((s (+ WARM 1)))
    (or (> s TOTAL)
        (and (<= (cosine s) (cosine (- s 1))) (loop (+ s 1))))))

(test-assert "cosine stays inside [min-lr, base-lr] after warmup"
  (let loop ((s WARM))
    (or (> s TOTAL)
        (let ((lr (cosine s)))
          (and (>= lr (- LOW 1e-12))
               (<= lr (+ BASE 1e-12))
               (loop (+ s 1)))))))

(test-assert "cosine warmup stays inside [0, base-lr]"
  (let loop ((s 0))
    (or (> s WARM)
        (let ((lr (cosine s)))
          (and (>= lr 0) (<= lr (+ BASE 1e-12)) (loop (+ s 1)))))))

(test-assert "cosine is half-way between base and min at the midpoint"
  (approx= (/ (+ BASE LOW) 2)
           (cosine (+ WARM (quotient (- TOTAL WARM) 2)))))

;;; --- constant ---------------------------------------------------------------

(test-assert "constant schedule always returns its rate"
  (every (lambda (s) (approx= BASE (constant s))) '(0 1 100 999999)))

;;; --- linear warmup ----------------------------------------------------------

(test-assert "linear warmup starts at 0"
  (= 0 (linear 0)))

(test-assert "linear warmup rises monotonically to base-lr"
  (let loop ((s 1))
    (or (> s WARM)
        (and (> (linear s) (linear (- s 1))) (loop (+ s 1))))))

(test-assert "linear warmup reaches base-lr and then stays constant"
  (and (approx= BASE (linear WARM))
       (approx= BASE (linear (+ WARM 1)))
       (approx= BASE (linear (* 10 WARM)))))

;;; --- validation -------------------------------------------------------------

(test-error "min-lr above base-lr is rejected" #t
  (make-cosine-schedule 1e-3 2e-3 10 100))

(test-error "a negative warmup is rejected" #t
  (make-linear-warmup-schedule 1e-3 -1))

(test-error "warmup longer than total-steps is rejected" #t
  (make-cosine-schedule 1e-3 1e-4 100 50))

(let ((failures (test-runner-fail-count (test-runner-current))))
  (test-end "sonicboom-lr-schedule")
  (exit (if (zero? failures) 0 1)))
