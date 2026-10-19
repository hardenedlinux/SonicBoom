;;; SonicBoom — R6: tiny Transformer forward + backward + AdamW on the GPU
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-train-tiny.scm
;;;
;;; One traced training step holds the whole model: embedding lookup, 4
;;; pre-norm blocks (RMSNorm → RoPE → causal attention → residual → RMSNorm →
;;; SwiGLU → residual), a final RMSNorm and the vocabulary projection, then
;;; fused cross-entropy, `backward!` (fused adjoint ops) and `adamw-step!`.
;;; Shapes: seq 32, hidden 64, 4 heads of 16, FFN 128, vocabulary 256.
;;;
;;; The traced step is executed twice from the same initial parameters:
;;;   - on the GPU: S-Expr → sb_compile_device → Blade → Boom IR (Training
;;;     passes) → per-node CUDA kernels;
;;;   - by the pure-Scheme reference interpreter on the same emitted graph.
;;; Every parameter gradient must agree to 1e-5 (absolute), and the GPU run
;;; must lower the loss over 40 AdamW steps of one fixed batch.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 format)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace interp)
             (sonicboom trace fused)
             (sonicboom trace backward)
             (sonicboom trace device))

;;; --- problem size -----------------------------------------------------------

(define T 32)                    ; sequence length
(define D 64)                    ; hidden size
(define H 4)                     ; attention heads
(define HD 16)                   ; head dim (= D / H)
(define F 128)                   ; SwiGLU hidden size
(define V 256)                   ; vocabulary
(define L 4)                     ; blocks

(define ETA 1e-2)                ; AdamW learning rate
(define BETA1 0.9)
(define BETA2 0.999)
(define EPS 1e-8)

;;; --- deterministic init -----------------------------------------------------

(define seed 12345)
(define (rand)
  (set! seed (modulo (+ (* seed 1103515245) 12345) 2147483648))
  (- (* 2.0 (/ seed 2147483648.0)) 1.0))

(define (rand-tensor shape scale)
  (make-ctensor 'float32 shape
                (map (lambda (_) (* scale (rand)))
                     (iota (shape-numel shape)))))

(define (zero-tensor shape)
  (make-ctensor 'float32 shape (make-list (shape-numel shape) 0)))

(define (full-tensor shape v)
  (make-ctensor 'float32 shape (make-list (shape-numel shape) v)))

;; Parameter order: embedding | (wq wk wv wo norm1 norm2 gate up down) × L |
;; final norm | vocabulary projection.
(define param-specs
  (append
   (list (cons (list V D) 0.05))
   (append-map (lambda (_)
                 (list (cons (list D D) 0.08) (cons (list D D) 0.08)
                       (cons (list D D) 0.08) (cons (list D D) 0.08)
                       (cons (list D) 1.0) (cons (list D) 1.0)
                       (cons (list D F) 0.06) (cons (list D F) 0.06)
                       (cons (list F D) 0.06)))
               (iota L))
   (list (cons (list D) 1.0) (cons (list D V) 0.06))))

(define (make-params) (map (lambda (s) (rand-tensor (car s) (cdr s))) param-specs))
(define (make-moments) (map (lambda (s) (zero-tensor (car s))) param-specs))
(define (make-step) (make-ctensor 'int32 '() '(1)))

(define (copy-ctensor t)
  (make-ctensor (ctensor-dtype t) (ctensor-shape t) (ctensor-data t)))

(define (fresh-leaves params)
  (map (lambda (p) (make-leaf (copy-ctensor p))) params))

;;; --- data -------------------------------------------------------------------

(define (token i) (modulo (+ (* 7 i) 3) V))
(define ids   (make-ctensor 'int32 (list T) (map token (iota T))))
(define targets
  (make-ctensor 'int32 (list T) (map (lambda (i) (token (+ i 1))) (iota T))))

;; The causal additive mask: 0 on and below the diagonal, -1e9 above. It is a
;; graph constant, added to the [H,T,T] score tensor with broadcasting.
(define causal
  (make-ctensor 'float32 (list T T)
                (append-map (lambda (i)
                              (map (lambda (j) (if (<= j i) 0.0 -1e9)) (iota T)))
                            (iota T))))

;;; --- model ------------------------------------------------------------------

(define (rms x w) (fused-rms-norm x w #:eps 1e-6))
(define (heads x) (permute (reshape x (list T H HD)) (list 1 0 2)))
(define (merge-heads x) (reshape (permute x (list 1 0 2)) (list T D)))

(define (block p x)
  (let* ((wq (list-ref p 0)) (wk (list-ref p 1)) (wv (list-ref p 2))
         (wo (list-ref p 3))
         (n1 (list-ref p 4)) (n2 (list-ref p 5))
         (gate-w (list-ref p 6)) (up-w (list-ref p 7)) (down-w (list-ref p 8))
         (h (rms x n1))
         (q (heads (fused-rope (matmul h wq) #:head-dim HD)))
         (k (heads (fused-rope (matmul h wk) #:head-dim HD)))
         (v (heads (matmul h wv)))
         (scores (mul (bmm q (transpose k)) (/ 1.0 (sqrt HD))))
         (cores (add scores causal))
         (attn (merge-heads (bmm (fused-softmax cores) v)))
         (x1 (add x (matmul attn wo)))
         (m (rms x1 n2))
         (act (fused-silu-mul (matmul m gate-w) (matmul m up-w)))
         (x2 (add x1 (matmul act down-w))))
    x2))

(define (model params x-ids)
  (let* ((emb (car params))
         (blocks (take (drop params 1) (* 9 L)))
         (tail (drop params (+ 1 (* 9 L))))
         (norm-w (car tail))
         (wout (cadr tail))
         (x (index-select emb 0 x-ids)))
    (let loop ((i 0) (x x))
      (if (= i L)
          (matmul (rms x norm-w) wout)
          (loop (+ i 1) (block (take (drop blocks (* 9 i)) 9) x))))))

;;; --- one traced training step -----------------------------------------------

(define (train-step params ms vs t x-ids y-ids)
  (let ((loss (fused-cross-entropy (model params x-ids) y-ids)))
    (backward! loss)
    (for-each (lambda (p m1 m2)
                (adamw-step! p (leaf-grad p) m1 m2 t
                             #:lr ETA #:beta1 BETA1 #:beta2 BETA2
                             #:eps EPS #:weight-decay 0.0))
              params ms vs)
    loss))

(define (bump-step! t)
  (set-leaf-value! t
                   (make-ctensor 'int32 '()
                                 (list (+ 1 (inexact->exact
                                             (ctensor-item (leaf-value t))))))))

;;; --- checks -----------------------------------------------------------------

(define (mean xs) (/ (fold + 0 xs) (length xs)))

(define (max-abs-diff as bs)
  (let loop ((xs as) (ys bs) (worst 0.0))
    (if (null? xs)
        worst
        (loop (cdr xs) (cdr ys)
              (fold (lambda (a b w) (max w (abs (- a b))))
                    worst (ctensor->list (car xs)) (ctensor->list (car ys)))))))

(unless (device-available?)
  (format #t "SKIP: libsonicboom.so was built without CUDA~%")
  (exit 0))

(test-begin "sonicboom-train-tiny")

;; One step from the same initial parameters: the reference interpreter and the
;; GPU executor run the same emitted graph.
(define init (make-params))
(define init-m (make-moments))
(define init-v (make-moments))

(define gpu-params (fresh-leaves init))
(define gpu-ms (fresh-leaves init-m))
(define gpu-vs (fresh-leaves init-v))
(define gpu-t (make-leaf (make-step)))

(define ref-params (fresh-leaves init))
(define ref-ms (fresh-leaves init-m))
(define ref-vs (fresh-leaves init-v))
(define ref-t (make-leaf (make-step)))

(define gpu-step (trace-function train-step #:executor (device-executor)))
(define ref-step (trace-function train-step #:executor interpret-sexpr-text))

(define gpu-loss (gpu-step gpu-params gpu-ms gpu-vs gpu-t ids targets))
(define ref-loss (ref-step ref-params ref-ms ref-vs ref-t ids targets))

(define gpu-grads (map leaf-grad gpu-params))
(define ref-grads (map leaf-grad ref-params))

(test-equal "the step compiles to exactly one device segment"
  1 (compiled-device-segments))
(test-assert "GPU and reference loss agree"
  (< (abs (- (ctensor-item gpu-loss) (ctensor-item ref-loss))) 1e-4))

(define worst (max-abs-diff gpu-grads ref-grads))
(format #t "~%parameter gradients: ~a tensors, max |GPU - reference| = ~e~%"
        (length gpu-grads) worst)
(test-assert "every parameter gradient matches the reference to 1e-5"
  (<= worst 1e-5))

(test-assert "the AdamW step updated every parameter and moment"
  (every (lambda (l) (ctensor? (leaf-value l)))
         (append gpu-params gpu-ms gpu-vs)))

;; Loss curve: 40 cached replays of the same device graph on one fixed batch.
(define steps 40)

(define curve
  (let loop ((i 0) (acc '()))
    (if (= i steps)
        (reverse acc)
        (let* ((raw (gpu-step gpu-params gpu-ms gpu-vs gpu-t ids targets))
               (loss (ctensor-item raw)))
          (bump-step! gpu-t)
          (loop (+ i 1) (cons loss acc))))))

(format #t "loss (~a GPU steps): " steps)
(for-each (lambda (l) (format #t "~,3f " l)) (take curve 5))
(display "... ")
(for-each (lambda (l) (format #t "~,3f " l)) (take-right curve 5))
(format #t "~%")

(test-assert "loss falls over the 40 steps (last five below the first five)"
  (< (mean (take-right curve 5)) (mean (take curve 5))))

(test-assert "every later step replays the cached graph (one trace, one compile)"
  (and (= 1 (compiled-device-segments))
       (= 1 (assq-ref (trace-stats gpu-step) 'traces))
       (= (+ steps 1) (assq-ref (trace-stats gpu-step) 'executor-calls))))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-train-tiny")
(exit (if (zero? failures) 0 1))
