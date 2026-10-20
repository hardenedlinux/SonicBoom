;;; SonicBoom — D2: AdamW learning rate as a graph input
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-adamw-lr-input.scm
;;;
;;; `adamw-step!` accepts the learning rate as a tensor. A tensor (here a
;;; make-lr-tensor leaf) is a graph input, so one trace and one compiled device
;;; segment serve a whole schedule; a number stays a node attribute, so the old
;;; form needs one trace per rate. The test runs ten steps of a cosine-like
;;; schedule both ways from identical state and requires the two loss curves to
;;; agree, checks the compile counts (1 versus 10), compares against the
;;; pure-Scheme interpreter, and exercises the SONICBOOM_ADAMW_LR_ATTR=1
;;; ablation switch on the device executor.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 format)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace interp)
             (sonicboom trace fused)
             (sonicboom trace backward)
             (sonicboom trace device))

;;; --- problem size -----------------------------------------------------------

(define T 8)                     ; sequence length
(define D 16)                    ; hidden size
(define H 2)                     ; attention heads
(define HD 8)                    ; head dim (= D / H)
(define F 32)                    ; SwiGLU hidden size
(define V 64)                    ; vocabulary
(define L 1)                     ; blocks

(define BETA1 0.9)
(define BETA2 0.999)
(define EPS 1e-8)

(define STEPS 10)
;; A schedule that visits ten distinct rates, as a cosine decay would.
(define LRS '#(0.0100 0.0096 0.0084 0.0064 0.0040
               0.0020 0.0009 0.0004 0.0002 0.0001))

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

;;; --- one traced training step; LR is the last tensor argument ---------------

(define (train-step params ms vs t x-ids y-ids lr)
  (let ((loss (fused-cross-entropy (model params x-ids) y-ids)))
    (backward! loss)
    (for-each (lambda (p m1 m2)
                (adamw-step! p (leaf-grad p) m1 m2 t
                             #:lr lr #:beta1 BETA1 #:beta2 BETA2
                             #:eps EPS #:weight-decay 0.0))
              params ms vs)
    loss))

(define (bump-step! t)
  (set-leaf-value! t
                   (make-ctensor 'int32 '()
                                 (list (+ 1 (inexact->exact
                                             (ctensor-item (leaf-value t))))))))

(define (set-lr! lr v)
  (set-leaf-value! lr (scalar->ctensor v 'float32)))

(define (loss-of x) (ctensor-item x))

;; The step returns the forward loss, which the update does not touch; the
;; ablation below has to look at the parameters the step wrote back.
(define (params-diff ps qs)
  (fold max 0.0
        (map (lambda (p q)
               (fold max 0.0
                     (map (lambda (a b) (abs (- a b)))
                          (ctensor->list (leaf-value p))
                          (ctensor->list (leaf-value q)))))
             ps qs)))

;;; --- checks -----------------------------------------------------------------

(unless (device-available?)
  (format #t "SKIP: libsonicboom.so was built without CUDA~%")
  (exit 0))

(test-begin "sonicboom-adamw-lr-input")

;; Run A: the lr-tensor form. One traced function, one lr leaf, ten rates.
(define init (make-params))
(define init-m (make-moments))
(define init-v (make-moments))

(define a-params (fresh-leaves init))
(define a-ms (fresh-leaves init-m))
(define a-vs (fresh-leaves init-v))
(define a-t (make-leaf (make-step)))
(define a-lr (make-lr-tensor 'cuda))

(define gpu-step (trace-function train-step #:executor (device-executor)))

(define (run-schedule! step params ms vs t lr-leaf)
  (let loop ((i 0) (acc '()))
    (if (= i STEPS)
        (reverse acc)
        (begin
          (set-lr! lr-leaf (vector-ref LRS i))
          (let ((loss (loss-of (step params ms vs t ids targets lr-leaf))))
            (bump-step! t)
            (loop (+ i 1) (cons loss acc)))))))

(define a-losses (run-schedule! gpu-step a-params a-ms a-vs a-t a-lr))
(define segments-after-a (compiled-device-segments))

(format #t "lr-tensor form: ~a steps, ~a device segment(s) compiled~%"
        STEPS segments-after-a)
(format #t "  loss ")
(for-each (lambda (l) (format #t "~,4f " l)) a-losses)
(format #t "~%")

(test-equal "ten different rates share one traced segment"
  STEPS (length (delete-duplicates (vector->list LRS))))
(test-equal "the lr-tensor form compiled exactly one device segment"
  1 segments-after-a)

;; Run B: the attribute form from identical state. Each distinct rate is a
;; static argument, hence its own trace and its own segment.
(define b-params (fresh-leaves init))
(define b-ms (fresh-leaves init-m))
(define b-vs (fresh-leaves init-v))
(define b-t (make-leaf (make-step)))

(define b-losses
  (let loop ((i 0) (acc '()))
    (if (= i STEPS)
        (reverse acc)
        (let* ((rate (vector-ref LRS i))
               (step (trace-function
                      (lambda (params ms vs t x-ids y-ids)
                        (train-step params ms vs t x-ids y-ids rate))
                      #:executor (device-executor)))
               (loss (loss-of (step b-params b-ms b-vs b-t ids targets))))
          (bump-step! b-t)
          (loop (+ i 1) (cons loss acc))))))

(define segments-after-b (compiled-device-segments))
(format #t "attribute form: ~a step(s) compiled in total (~a for this run)~%"
        segments-after-b (- segments-after-b segments-after-a))
(test-equal "the attribute form needed one segment per rate"
  (+ segments-after-a STEPS) segments-after-b)

(define worst (fold max 0.0 (map (lambda (x y) (abs (- x y)))
                                 a-losses b-losses)))
(format #t "loss: max |lr-tensor - attribute| over ~a steps = ~e~%"
        STEPS worst)
(test-assert "the lr-tensor form matches the attribute form"
  (<= worst 1e-6))

;; Run C: the pure-Scheme interpreter executes the same emitted graph (it reads
;; the sixth operand in its own adamw kernel).
(define c-params (fresh-leaves init))
(define c-ms (fresh-leaves init-m))
(define c-vs (fresh-leaves init-v))
(define c-t (make-leaf (make-step)))
(define c-lr (make-lr-tensor 'cuda))
(define ref-step (trace-function train-step #:executor interpret-sexpr-text))
(define c-losses (run-schedule! ref-step c-params c-ms c-vs c-t c-lr))
(define ref-worst (fold max 0.0 (map (lambda (x y) (abs (- x y)))
                                     a-losses c-losses)))
(format #t "loss: max |device - interpreter| over ~a steps = ~e~%"
        STEPS ref-worst)
(test-assert "the device lr-tensor form matches the interpreter"
  (<= ref-worst 1e-4))

;; Ablation: SONICBOOM_ADAMW_LR_ATTR=1 ignores a sixth operand and uses the
;; node's lr attribute (absent here, so the 1e-3 default). A fresh parameter
;; updated with a 0.5 lr tensor must then match the attribute form at 1e-3.
(setenv "SONICBOOM_ADAMW_LR_ATTR" "1")
(define d-params (fresh-leaves init))
(define d-ms (fresh-leaves init-m))
(define d-vs (fresh-leaves init-v))
(define d-t (make-leaf (make-step)))
(define d-lr (make-lr-tensor 'cuda))
(set-lr! d-lr 0.5)
(gpu-step d-params d-ms d-vs d-t ids targets d-lr)
(unsetenv "SONICBOOM_ADAMW_LR_ATTR")

(define e-params (fresh-leaves init))
(define e-ms (fresh-leaves init-m))
(define e-vs (fresh-leaves init-v))
(define e-t (make-leaf (make-step)))
(define e-step
  (trace-function
   (lambda (params ms vs t x-ids y-ids)
     (train-step params ms vs t x-ids y-ids 1e-3))
   #:executor (device-executor)))
(e-step e-params e-ms e-vs e-t ids targets)

(define d-e-diff (params-diff d-params e-params))
(format #t "ablation: max |params after lr-tensor 0.5 with LR_ATTR=1 - params at 1e-3| = ~e~%"
        d-e-diff)
(test-assert "SONICBOOM_ADAMW_LR_ATTR=1 restores the attribute form"
  (<= d-e-diff 1e-6))

;; The switch is read per execution: with it unset again, the same head state
;; must move with the tensor rate, not with the attribute default.
(define f-params (fresh-leaves init))
(define f-ms (fresh-leaves init-m))
(define f-vs (fresh-leaves init-v))
(define f-t (make-leaf (make-step)))
(define f-lr (make-lr-tensor 'cuda))
(set-lr! f-lr 0.5)
(gpu-step f-params f-ms f-vs f-t ids targets f-lr)
(define f-e-diff (params-diff f-params e-params))
(format #t "ablation: max |params at lr 0.5 (no switch) - params at 1e-3| = ~e~%"
        f-e-diff)
(test-assert "without the switch the tensor rate is used again"
  (> f-e-diff 1e-2))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-adamw-lr-input")
(exit (if (zero? failures) 0 1))
