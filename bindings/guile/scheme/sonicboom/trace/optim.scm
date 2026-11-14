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
  "One gradient-descent update of a parameter list.

Returns PARAMS - lr * GRADS element by element.

Args:
  params: a list of tensors.
  grads: a list of gradients, in the same order.
  lr: keyword; the learning rate (default 0.1).
Returns: a list of updated tensors.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace optim))
  (map ctensor->list
       (sgd-update (list (list->ctensor '(1.0 2.0)))
                   (list (list->ctensor '(0.5 0.5)))
                   #:lr 0.1))
  => ((0.949999988079071 1.9500000476837158))
PyTorch: torch.optim.SGD
Note: torch.optim.SGD owns the parameter list and updates it in place; this
is a pure function returning the updated values."
  (map (lambda (p g) (sub p (mul g lr))) params grads))

(define (adam-init params)
  "The initial Adam state for PARAMS.

The state holds a zero first moment and a zero second moment for every
parameter, and a rank-0 float32 counter tensor with the value 0.

Args:
  params: a list of ctensors.
Returns: the list (first-moments second-moments step-counter).
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace optim))
  (ctensor->list (caddr (adam-init (list (list->ctensor '(1.0 2.0))))))
  => (0.0)
PyTorch: torch.optim.Adam
Note: torch.optim.Adam keeps its state in the optimizer object; the state
here is a value that a step takes and returns."
  (let ((zeros (lambda (p) (make-ctensor (ctensor-dtype p) (ctensor-shape p)
                                         (make-list (ctensor-numel p) 0)))))
    (list (map zeros params) (map zeros params) (scalar->ctensor 0.0 'float32))))

(define* (adam-update params grads ms vs t #:key (lr 1e-2) (beta1 0.9) (beta2 0.999)
                      (eps 1e-8))
  "One Adam step.

MS and VS are the first and the second moment tensors of PARAMS and T is the
rank-0 step counter. The bias corrections are computed from T inside the
graph, and the new state is returned so the next step takes it as input.

Args:
  params: a list of tensors.
  grads: a list of gradients, in the same order.
  ms: the first moments.
  vs: the second moments.
  t: the step counter tensor.
  lr: keyword; the learning rate (default 1e-2).
  beta1: keyword; the first moment decay (default 0.9).
  beta2: keyword; the second moment decay (default 0.999).
  eps: keyword; added under the root (default 1e-8).
Returns: the list (new-params new-ms new-vs new-t); new-t is T + 1.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace optim))
  (let* ((p (list (list->ctensor '(1.0 2.0))))
         (g (list (list->ctensor '(0.5 0.5))))
         (st (adam-init p)))
    (map ctensor->list
         (car (adam-update p g (car st) (cadr st) (caddr st) #:lr 0.1))))
  => ((0.9000006914138794 1.9000006914138794))
PyTorch: torch.optim.Adam
Note: torch.optim.Adam updates the parameters in place; this is a pure
function whose state is passed in and returned."
  (let* ((t1 (add t 1.0))
         (bc1 (sub 1.0 (texp (mul t1 (log beta1)))))
         (bc2 (sub 1.0 (texp (mul t1 (log beta2)))))
         (ms* (map (lambda (m g) (add (mul m beta1) (mul g (- 1 beta1)))) ms grads))
         (vs* (map (lambda (v g) (add (mul v beta2) (mul (mul g g) (- 1 beta2)))) vs grads))
         (ps* (map (lambda (p m v)
                     (sub p (mul (div (div m bc1) (add (tsqrt (div v bc2)) eps)) lr)))
                   params ms* vs*)))
    (list ps* ms* vs* t1)))
