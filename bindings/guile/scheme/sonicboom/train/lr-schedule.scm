;;; SonicBoom — Guile binding: learning-rate schedules
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Pure Scheme schedule procedures: each constructor returns (lambda (step)
;;; -> lr). They are values, not tensors — the training loop applies the
;;; result itself (e.g. into an sgd or AdamW step), so there is no new op and
;;; no native dependency here.

(define-module (sonicboom train lr-schedule)
  #:use-module (ice-9 format)
  #:export (make-cosine-schedule
            make-constant-schedule
            make-linear-warmup-schedule))

(define %pi (* 4 (atan 1)))

(define (check-lr who name x)
  (unless (and (real? x) (>= x 0))
    (error who name "must be a non-negative real" x)))

(define (check-steps who name x)
  (unless (and (integer? x) (>= x 0))
    (error who name "must be a non-negative integer" x)))

(define (make-constant-schedule lr)
  "A schedule that always returns LR."
  (check-lr "make-constant-schedule" "lr" lr)
  (lambda (step) (exact->inexact lr)))

(define (make-linear-warmup-schedule base-lr warmup-steps)
  "Linear warmup from 0 at step 0 to BASE-LR at step WARMUP-STEPS, constant
from then on."
  (check-lr "make-linear-warmup-schedule" "base-lr" base-lr)
  (check-steps "make-linear-warmup-schedule" "warmup-steps" warmup-steps)
  (lambda (step)
    (cond ((< step 0) 0.0)
          ((< step warmup-steps)
           (* (exact->inexact base-lr) (/ step warmup-steps)))
          (else (exact->inexact base-lr)))))

(define (make-cosine-schedule base-lr min-lr warmup-steps total-steps)
  "Linear warmup from 0 to BASE-LR over WARMUP-STEPS, then a half-cosine decay
from BASE-LR to MIN-LR at TOTAL-STEPS, then MIN-LR."
  (check-lr "make-cosine-schedule" "base-lr" base-lr)
  (check-lr "make-cosine-schedule" "min-lr" min-lr)
  (when (> min-lr base-lr)
    (error "make-cosine-schedule: min-lr must not exceed base-lr"
           min-lr base-lr))
  (check-steps "make-cosine-schedule" "warmup-steps" warmup-steps)
  (check-steps "make-cosine-schedule" "total-steps" total-steps)
  (when (> warmup-steps total-steps)
    (error "make-cosine-schedule: warmup-steps must not exceed total-steps"
           warmup-steps total-steps))
  (let ((base (exact->inexact base-lr))
        (low (exact->inexact min-lr))
        (span (- total-steps warmup-steps)))
    (lambda (step)
      (cond
        ;; Before training starts: no warmup means the schedule starts at its
        ;; peak, a warmup means it starts at zero.
        ((< step 0) (if (> warmup-steps 0) 0.0 base))
        ((< step warmup-steps) (* base (/ step warmup-steps)))
        ((or (>= step total-steps) (= span 0)) low)
        (else
         (let* ((t (/ (- step warmup-steps) span))
                (cos (/ (+ 1.0 (cos (* %pi t))) 2.0)))
           (+ low (* (- base low) cos))))))))
