;;; SonicBoom — Guile training driver tests
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Exercises (sonicboom train): the thin autograd driver over the C ABI. Guile
;;; only composes linear/relu/mse and runs the loop; every gradient and update
;;; is computed natively in libsonicboom.so. These tests assert the acceptance
;;; criteria: backward yields real (non-zero) gradients, and a deterministic
;;; XOR run lowers the loss across epochs.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (sonicboom train))

(define (approx= a b eps) (< (abs (- a b)) eps))

(test-begin "sonicboom-train")

;; --- gradients are real (computed natively, not fabricated) -----------------

(test-assert "backward produces non-zero parameter gradients"
  (let* ((x    (tensor-from-list '(2 2) '(0.5 1.0 -0.5 0.25)))
         (w    (tensor-from-list '(1 2) '(0.3 -0.4)))
         (b    (tensor-from-list '(1) '(0.1)))
         (tgt  (tensor-from-list '(2 1) '(0.7 -0.2)))
         (tape (make-tape))
         (y    (autograd-linear x w b tape))
         (loss (autograd-mse y tgt tape)))
    (autograd-backward loss tape)
    (let* ((gw-h (autograd-grad tape w))
           (gb-h (autograd-grad tape b))
           (gw   (tensor->list gw-h 2))
           (gb   (tensor->list gb-h 1)))
      (tensor-free! gw-h) (tensor-free! gb-h)
      (tensor-free! y) (tensor-free! loss)
      (tensor-free! x) (tensor-free! w) (tensor-free! b) (tensor-free! tgt)
      (tape-free! tape)
      (and (not (every (lambda (v) (approx= v 0.0 1e-9)) gw))
           (not (every (lambda (v) (approx= v 0.0 1e-9)) gb))))))

;; --- end-to-end training loop ------------------------------------------------

(test-assert "XOR training lowers the loss across epochs"
  (call-with-values
      (lambda () (train-xor #:epochs 3000 #:lr 0.5))
    (lambda (initial final)
      (and (real? initial) (real? final)
           (> initial 0.0)
           (< final initial)
           (< final 0.05)))))

(test-end "sonicboom-train")
