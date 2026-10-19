;;; SonicBoom — tracing prototype: fused training vocabulary
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; One fused op per training layer, emitted under the S-Expr op names the
;;; lowering resolves to Boom IR operators:
;;;
;;;   fused-softmax        -> softmax          (last axis)
;;;   fused-rms-norm       -> rms_norm
;;;   fused-rope           -> rope
;;;   fused-silu-mul       -> silu_mul
;;;   fused-cross-entropy  -> cross_entropy    (mean NLL, 1-D int targets)
;;;
;;; `backward!` (sonicboom trace backward) recognizes these forward nodes and
;;; emits the matching *_bwd operator from the same recording. Embedding lookup
;;; is the ordinary `index-select` on axis 0. The composite versions in
;;; (sonicboom trace layers) still decompose into primitive ops; these exist
;;; for training graphs that go to the native lowering.
;;;
;;; adamw-step! is the one optimizer update that is itself a graph op: one
;;; node for the whole parameter/moment update, so a training graph with any
;;; number of parameters has one optimizer node per tensor instead of a
;;; decomposed arithmetic chain.

(define-module (sonicboom trace fused)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:export (fused-softmax fused-rms-norm fused-rope fused-silu-mul
            fused-cross-entropy adamw-step! fused-clip-grad-norm
            clip-grad-norm!))

(define* (fused-clip-grad-norm grads #:key (max-norm 1.0))
  "Global-norm gradient clipping over GRADS, a list of gradient tensors.
Returns one clipped tensor per gradient, in order. The global L2 norm is
accumulated in double over every element of every gradient; when it does not
exceed MAX-NORM the results are copies of the inputs (their values unchanged),
otherwise each is scaled by MAX-NORM / (norm + 1e-6). One clip_grad_norm node:
the device executor runs it as one accumulation pass plus one scale kernel per
gradient."
  (unless (and (real? max-norm) (>= max-norm 0))
    (error "fused-clip-grad-norm: max-norm must be a non-negative number"
           max-norm))
  (apply-op-multi 'clip_grad_norm grads
                  `((max_norm (float ,max-norm)))
                  (map tensor-type grads)))

(define* (clip-grad-norm! params #:optional (max-norm 1.0))
  "Clip the accumulated .grad of every leaf in PARAMS (a leaf or a list) by the
global L2 norm of all their gradients, in place. PARAMS is what `backward!`
filled in; after this call each leaf's .grad is the clipped gradient, so the
next `adamw-step!` sees it. Leaves without an accumulated gradient are skipped.
Returns the list of clipped gradient values. Inside a traced step this emits
one clip_grad_norm node (the fused CUDA kernel); in eager mode it runs the
reference kernel."
  (let* ((ps (if (leaf? params) (list params) params))
         (leaves (filter (lambda (p) (leaf-grad p)) ps)))
    (when (null? leaves)
      (error "clip-grad-norm!: no leaf has an accumulated gradient"))
    (let ((clipped (fused-clip-grad-norm (map leaf-grad leaves)
                                         #:max-norm max-norm)))
      (for-each set-leaf-grad! leaves clipped)
      clipped)))

(define (fused-softmax x)
  "Numerically stable softmax over the last axis."
  (apply-op 'softmax (list x) '()))

(define* (fused-rms-norm x w #:key (eps 1e-6))
  "x / sqrt(mean(x^2) + EPS) * W over the last axis."
  (apply-op 'rms_norm (list x w) `((eps (float ,eps)))))

(define* (fused-rope x #:key head-dim (theta-base 10000.0) (freq-scale 1.0)
                     (pos0 0))
  "Rotary position embedding on the last axis, HEAD-DIM values per head; the
row index selects the position. Shape [rows, heads * head_dim]."
  (apply-op 'rope (list x)
            `((head_dim (int ,head-dim))
              (theta_base (float ,theta-base))
              (freq_scale (float ,freq-scale))
              (pos0 (int ,pos0)))))

(define (fused-silu-mul gate up)
  "SwiGLU activation: silu(GATE) * UP."
  (apply-op 'silu_mul (list gate up) '()))

(define (fused-cross-entropy logits targets)
  "Mean negative log-likelihood of 1-D TARGETS (class indices) under LOGITS
[rows, classes]. Returns a rank-0 mean loss (unlike the composite version,
which also keeps the per-row values in the graph)."
  (apply-op 'cross_entropy (list logits targets) '()))

(define (tensor-type x)
  (cond ((sym? x) (cons (sym-dtype x) (sym-shape x)))
        ((leaf? x) (tensor-type (leaf-value x)))
        (else (cons (ctensor-dtype x) (ctensor-shape x)))))

(define* (adamw-step! param grad m1 m2 step
                      #:key (lr 1e-3) (beta1 0.9) (beta2 0.999)
                      (eps 1e-8) (weight-decay 0.01))
  "One fused AdamW update of PARAM from GRAD, with first/second moment state
M1/M2 and the 1-based int32 STEP tensor. When PARAM/M1/M2 are leaves their
value is replaced by the updated tensor, which the tracer syncs back to the
caller's leaf after the segment runs. Returns (new-param new-m1 new-m2)."
  (let ((outs (apply-op-multi 'adamw (list param grad m1 m2 step)
                              `((lr (float ,lr)) (beta1 (float ,beta1))
                                (beta2 (float ,beta2)) (eps (float ,eps))
                                (weight_decay (float ,weight-decay)))
                              (list (tensor-type param) (tensor-type m1)
                                    (tensor-type m2)))))
    (when (leaf? param) (set-leaf-value! param (car outs)))
    (when (leaf? m1) (set-leaf-value! m1 (cadr outs)))
    (when (leaf? m2) (set-leaf-value! m2 (caddr outs)))
    outs))
