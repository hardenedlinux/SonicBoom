;;; SonicBoom — tracing prototype milestone: backward graph emission
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace-backward.scm
;;;
;;; A two-layer MLP on CPU through the S-Expr interpreter: backward! walks the
;;; recorded tape in reverse, emits the fused adjoints (matmul_bwd,
;;; cross_entropy_bwd), and stores each parameter leaf's gradient in its .grad
;;; field. The graph traced once holds forward and backward nodes and is
;;; replayed by every later call. Gradients are checked against an independent
;;; double-precision reference on the same values; max absolute error 1e-5.
;;; The emitted S-Expr is inspected for the fused operators.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 format)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace macro)
             (sonicboom trace backward)
             (sonicboom trace fused))

(define (stat f key) (assq-ref (trace-stats f) key))

;;; --- model ------------------------------------------------------------------------

(define N 4)        ; rows
(define D 3)        ; input features
(define H 4)        ; hidden
(define C 2)        ; classes

(define seed 11)
(define (rand)
  (set! seed (modulo (+ (* seed 1103515245) 12345) 2147483648))
  (- (* 2.0 (/ seed 2147483648.0)) 1.0))
(define (rand-tensor shape scale)
  (make-ctensor 'float32 shape
                (map (lambda (_) (* scale (rand))) (iota (shape-numel shape)))))
(define (full shape v)
  (make-ctensor 'float32 shape (make-list (shape-numel shape) v)))

(define inputs
  (make-ctensor 'float32 (list N D) (map (lambda (i) (* 0.25 (- i 5))) (iota (* N D)))))
(define targets (make-ctensor 'int64 (list N) '(0 1 0 1)))

(define (params)
  (list (rand-tensor (list D H) 0.6) (full (list H) 0.1)
        (rand-tensor (list H C) 0.6) (full (list C) -0.1)))

(define-traced (mlp-step x y w1 b1 w2 b2)
  (let* ((h (relu (add (matmul x w1) b1)))
         (logits (add (matmul h w2) b2))
         (loss (fused-cross-entropy logits y)))
    (backward! loss)
    loss))

;;; --- double-precision reference ------------------------------------------------------

(define (ref-grads x w1 b1 w2 b2 y)
  "Forward + reverse pass written out in full, on flat row-major lists. Returns
(loss dW1 db1 dW2 db2) with the same layout as the ctensors."
  (let* ((xv (list->vector (ctensor->list x)))
         (yv (list->vector (ctensor->list y)))
         (w1v (list->vector (ctensor->list w1)))
         (b1v (list->vector (ctensor->list b1)))
         (w2v (list->vector (ctensor->list w2)))
         (b2v (list->vector (ctensor->list b2)))
         (pre (make-vector (* N H) 0.0))
         (hh (make-vector (* N H) 0.0))
         (logits (make-vector (* N C) 0.0))
         (lse (make-vector N 0.0))
         (loss 0.0)
         (dlogits (make-vector (* N C) 0.0))
         (dh (make-vector (* N H) 0.0))
         (dpre (make-vector (* N H) 0.0))
         (dw1 (make-vector (* D H) 0.0))
         (db1 (make-vector H 0.0))
         (dw2 (make-vector (* H C) 0.0))
         (db2 (make-vector C 0.0)))
    ;; forward
    (do ((i 0 (+ i 1))) ((= i N))
      (do ((j 0 (+ j 1))) ((= j H))
        (let ((s (vector-ref b1v j)))
          (do ((k 0 (+ k 1))) ((= k D))
            (set! s (+ s (* (vector-ref xv (+ (* i D) k))
                            (vector-ref w1v (+ (* k H) j))))))
          (vector-set! pre (+ (* i H) j) s)
          (vector-set! hh (+ (* i H) j) (max 0.0 s)))))
    (do ((i 0 (+ i 1))) ((= i N))
      (do ((j 0 (+ j 1))) ((= j C))
        (let ((s (vector-ref b2v j)))
          (do ((k 0 (+ k 1))) ((= k H))
            (set! s (+ s (* (vector-ref hh (+ (* i H) k))
                            (vector-ref w2v (+ (* k C) j))))))
          (vector-set! logits (+ (* i C) j) s))))
    (do ((i 0 (+ i 1))) ((= i N))
      (let* ((off (* i C))
             (mx (let loop ((j 0) (m -inf.0))
                   (if (= j C) m (loop (+ j 1) (max m (vector-ref logits (+ off j)))))))
             (sum (let loop ((j 0) (s 0.0))
                    (if (= j C) s
                        (loop (+ j 1) (+ s (exp (- (vector-ref logits (+ off j)) mx))))))))
        (vector-set! lse i (+ mx (log sum)))
        (set! loss (+ loss (- (vector-ref lse i)
                              (vector-ref logits (+ off (vector-ref yv i))))))))
    (set! loss (/ loss N))
    ;; backward
    (do ((i 0 (+ i 1))) ((= i N))
      (let ((t (vector-ref yv i)))
        (do ((j 0 (+ j 1))) ((= j C))
          (vector-set! dlogits (+ (* i C) j)
                       (- (/ (exp (- (vector-ref logits (+ (* i C) j))
                                     (vector-ref lse i)))
                             N)
                          (if (= j t) (/ 1.0 N) 0.0))))))
    (do ((k 0 (+ k 1))) ((= k H))
      (do ((j 0 (+ j 1))) ((= j C))
        (let ((s 0.0))
          (do ((i 0 (+ i 1))) ((= i N))
            (set! s (+ s (* (vector-ref hh (+ (* i H) k))
                            (vector-ref dlogits (+ (* i C) j))))))
          (vector-set! dw2 (+ (* k C) j) s))))
    (do ((j 0 (+ j 1))) ((= j C))
      (let ((s 0.0))
        (do ((i 0 (+ i 1))) ((= i N))
          (set! s (+ s (vector-ref dlogits (+ (* i C) j)))))
        (vector-set! db2 j s)))
    (do ((i 0 (+ i 1))) ((= i N))
      (do ((k 0 (+ k 1))) ((= k H))
        (let ((s 0.0))
          (do ((j 0 (+ j 1))) ((= j C))
            (set! s (+ s (* (vector-ref dlogits (+ (* i C) j))
                            (vector-ref w2v (+ (* k C) j))))))
          (vector-set! dh (+ (* i H) k) s)
          (vector-set! dpre (+ (* i H) k)
                       (if (> (vector-ref pre (+ (* i H) k)) 0.0) s 0.0)))))
    (do ((k 0 (+ k 1))) ((= k D))
      (do ((j 0 (+ j 1))) ((= j H))
        (let ((s 0.0))
          (do ((i 0 (+ i 1))) ((= i N))
            (set! s (+ s (* (vector-ref xv (+ (* i D) k))
                            (vector-ref dpre (+ (* i H) j))))))
          (vector-set! dw1 (+ (* k H) j) s))))
    (do ((j 0 (+ j 1))) ((= j H))
      (let ((s 0.0))
        (do ((i 0 (+ i 1))) ((= i N))
          (set! s (+ s (vector-ref dpre (+ (* i H) j)))))
        (vector-set! db1 j s)))
    (list loss (vector->list dw1) (vector->list db1)
          (vector->list dw2) (vector->list db2))))

(define (max-abs a b)
  (if (null? a) 0.0 (apply max (map (lambda (x y) (abs (- x y))) a b))))

(define (check-grads label refs leaves)
  (let ((diffs (map (lambda (r l) (max-abs r (ctensor->list (leaf-grad l))))
                    refs leaves)))
    (for-each (lambda (name d) (format #t "  ~a: max |error| ~,3e~%" name d))
              '(dW1 db1 dW2 db2) diffs)
    (test-assert (string-append label ": gradients match the fp64 reference")
                 (every (lambda (d) (< d 1e-5)) diffs))
    diffs))

;;; --- tests -------------------------------------------------------------------------

(test-begin "sonicboom-trace-backward")

(define initial (params))
(define leaves (map make-leaf initial))
(define (call! ps)
  (apply mlp-step (append (list inputs targets) ps)))

(define loss1 (call! leaves))
(define ref1 (ref-grads inputs (car initial) (cadr initial)
                        (caddr initial) (cadddr initial) targets))

(format #t "first call: loss ~,6f (reference ~,6f)~%"
        (ctensor-item loss1) (car ref1))
(test-assert "loss matches the fp64 reference"
             (< (abs (- (ctensor-item loss1) (car ref1))) 1e-6))
(format #t "gradient max |error| vs fp64 reference:~%")
(check-grads "first call" (cdr ref1) leaves)

;; The graph that was handed to the interpreter holds both passes.
(define (node-ops text)
  (match (call-with-input-string text read)
    (('sonicboom-s-expr _ ('graph . fields))
     (map cadr (cdr (assq 'nodes fields))))))
(define ops (append-map (lambda (p) (node-ops (cdr p))) (trace-segments mlp-step)))
(format #t "graph ops: ~a~%" ops)
(test-assert "forward ops are in the graph"
             (every (lambda (op) (memq op ops)) '(matmul cross_entropy)))
(test-assert "fused backward ops are in the graph"
             (every (lambda (op) (memq op ops)) '(matmul_bwd cross_entropy_bwd)))
(test-equal "traced once, one segment" '(1 1)
  (list (stat mlp-step 'traces) (length (trace-segments mlp-step))))

;; A later call with different parameter values replays the cached graph and
;; still lands the right gradients in the new leaves' .grad slots.
(set! seed 4242)
(define updated (list (rand-tensor (list D H) 0.6) (cadr initial)
                      (rand-tensor (list H C) 0.6) (cadddr initial)))
(define leaves2 (map make-leaf updated))
(define loss2 (call! leaves2))
(define ref2 (ref-grads inputs (car updated) (cadr updated)
                        (caddr updated) (cadddr updated) targets))
(format #t "second call (cache replay): loss ~,6f~%" (ctensor-item loss2))
(test-assert "replayed loss matches the fp64 reference"
             (< (abs (- (ctensor-item loss2) (car ref2))) 1e-6))
(check-grads "replayed call" (cdr ref2) leaves2)
(test-equal "second call hit the cache" '(2 1 1)
  (map (lambda (k) (stat mlp-step k)) '(calls traces cache-hits)))

;; zero-grad! clears the slots.
(zero-grad! leaves2)
(test-assert "zero-grad! clears .grad"
             (every (lambda (l) (not (leaf-grad l))) leaves2))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-backward")
(exit (if (zero? failures) 0 1))
