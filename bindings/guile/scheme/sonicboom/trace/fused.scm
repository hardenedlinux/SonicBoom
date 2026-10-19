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

(define-module (sonicboom trace fused)
  #:use-module (sonicboom trace symbolic)
  #:export (fused-softmax fused-rms-norm fused-rope fused-silu-mul
            fused-cross-entropy))

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
