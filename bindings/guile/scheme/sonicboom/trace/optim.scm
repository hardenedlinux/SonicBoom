;;; SonicBoom — tracing prototype: optimizers for traced training steps
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Update rules written with the tensor ops, meant to be called inside a
;;; traced training step after `grad`, so the update is part of the step's
;;; graph. Optimizer state (Adam's moments and step counter) is ordinary
;;; tensors: the step takes it as input and returns the new state, which keeps
;;; the graph the same from step to step (a Scheme step counter would be a
;;; static argument and force a retrace every step).

(define-module (sonicboom trace optim)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:export (sgd-update adam-init adam-update))

(define* (sgd-update params grads #:key (lr 0.1))
  "PARAMS - lr * GRADS, element by element of the two lists."
  (map (lambda (p g) (sub p (mul g lr))) params grads))

(define (adam-init params)
  "Initial Adam state for PARAMS (ctensors): zero first and second moments
and the step counter t = 0 as a rank-0 float32 tensor."
  (let ((zeros (lambda (p) (make-ctensor (ctensor-dtype p) (ctensor-shape p)
                                         (make-list (ctensor-numel p) 0)))))
    (list (map zeros params) (map zeros params) (scalar->ctensor 0.0 'float32))))

(define* (adam-update params grads ms vs t #:key (lr 1e-2) (beta1 0.9) (beta2 0.999)
                      (eps 1e-8))
  "One Adam step. MS / VS are the first / second moment tensors, T the step
counter tensor (number of steps already taken). Returns
(new-params new-ms new-vs new-t). The bias corrections 1 - beta^t are
computed in the graph as 1 - exp(t' log beta) with t' = t + 1."
  (let* ((t1 (add t 1.0))
         (bc1 (sub 1.0 (texp (mul t1 (log beta1)))))
         (bc2 (sub 1.0 (texp (mul t1 (log beta2)))))
         (ms* (map (lambda (m g) (add (mul m beta1) (mul g (- 1 beta1)))) ms grads))
         (vs* (map (lambda (v g) (add (mul v beta2) (mul (mul g g) (- 1 beta2)))) vs grads))
         (ps* (map (lambda (p m v)
                     (sub p (mul (div (div m bc1) (add (tsqrt (div v bc2)) eps)) lr)))
                   params ms* vs*)))
    (list ps* ms* vs* t1)))
