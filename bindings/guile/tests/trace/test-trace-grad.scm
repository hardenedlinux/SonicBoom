;;; SonicBoom — tracing prototype tests: reverse-mode autodiff, XOR training
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace-grad.scm
;;;
;;; Every gradient is checked against central finite differences in double
;;; precision: relative error |a - n| / max(|a|, |n|, 1e-3) < 1e-4, h = 1e-5.
;;; The finite differences evaluate the same traced loss function (so `if`,
;;; `case` and loops follow the traced semantics). Prints the maximum relative
;;; error and the XOR loss curve to stdout.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 format)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace macro)
             (sonicboom trace autodiff))

(define (t x) (list->ctensor x))
(define (stat f key) (assq-ref (trace-stats f) key))
(define (node-ops text)
  (match (call-with-input-string text read)
    (('sonicboom-s-expr _ ('graph . fields)) (map cadr (cdr (assq 'nodes fields))))))

;;; --- finite differences ----------------------------------------------------------

(define %h 1e-5)
(define %max-rel 0.0)

(define (perturb tensor i delta)
  (let ((v (vector-copy (ctensor-data tensor))))
    (vector-set! v i (+ (vector-ref v i) delta))
    (make-ctensor (ctensor-dtype tensor) (ctensor-shape tensor) v)))

(define (numeric-grad loss-fn args k)
  "Central differences of LOSS-FN (args -> rank-0 ctensor) w.r.t. argument K."
  (let ((x (list-ref args k)))
    (map (lambda (i)
           (let ((at (lambda (d)
                       (ctensor-item
                        (apply loss-fn (append (take args k)
                                               (list (perturb x i d))
                                               (drop args (+ k 1))))))))
             (/ (- (at %h) (at (- %h))) (* 2 %h))))
         (iota (ctensor-numel x)))))

(define (rel-error a n) (/ (abs (- a n)) (max (abs a) (abs n) 1e-3)))

(define (grad-ok? loss-fn args which)
  "Analytic gradients (value-and-grad of LOSS-FN) agree with finite
differences for the argument positions WHICH."
  (let* ((vg (value-and-grad loss-fn))
         (r (apply vg args))
         (grads (cdr r))
         (tensor-positions (filter (lambda (i) (ctensor? (list-ref args i)))
                                   (iota (length args)))))
    (every (lambda (k)
             (let* ((analytic (ctensor->list (list-ref grads (list-index (lambda (p) (= p k))
                                                                          tensor-positions))))
                    (numeric (numeric-grad loss-fn args k))
                    (errs (map rel-error analytic numeric)))
               (set! %max-rel (apply max %max-rel errs))
               (every (lambda (e) (< e 1e-4)) errs)))
           which)))

;; A fixed weighting so that each output element gets a different gradient.
(define (probe shape)
  (make-ctensor 'float32 shape
                (map (lambda (i) (+ 0.3 (* 0.17 (modulo (* 7 i) 11)))) (iota (shape-numel shape)))))
(define (weighted y) (sum (mul y (probe (shape-of y)))))

(define a23 (t '((0.5 -1.2 2.0) (1.5 0.7 -0.3))))
(define b23 (t '((1.1 0.4 -0.9) (-2.0 0.8 1.3))))
(define pos23 (t '((1.5 2.0 0.8) (3.0 1.2 2.5))))
(define b3  (t '(0.2 -0.5 1.0)))
(define m32 (t '((0.3 -1.0) (0.8 0.5) (-0.6 1.4))))

(test-begin "sonicboom-trace-grad")

;;; --- per-op gradients -----------------------------------------------------------

(test-assert "grad add"       (grad-ok? (lambda (a b) (weighted (add a b))) (list a23 b23) '(0 1)))
(test-assert "grad add (broadcast bias)"
  (grad-ok? (lambda (a b) (weighted (add a b))) (list a23 b3) '(0 1)))
(test-assert "grad sub"       (grad-ok? (lambda (a b) (weighted (sub a b))) (list a23 b23) '(0 1)))
(test-assert "grad mul"       (grad-ok? (lambda (a b) (weighted (mul a b))) (list a23 b23) '(0 1)))
(test-assert "grad div"       (grad-ok? (lambda (a b) (weighted (div a b))) (list a23 pos23) '(0 1)))
(test-assert "grad neg"       (grad-ok? (lambda (a) (weighted (neg a))) (list a23) '(0)))
(test-assert "grad matmul"    (grad-ok? (lambda (a m) (weighted (matmul a m))) (list a23 m32) '(0 1)))
(test-assert "grad relu"      (grad-ok? (lambda (a) (weighted (relu a))) (list a23) '(0)))
(test-assert "grad sum (all)" (grad-ok? (lambda (a) (mul (sum a) (sum a))) (list a23) '(0)))
(test-assert "grad sum (axis 0)"
  (grad-ok? (lambda (a) (weighted (sum a #:axes '(0)))) (list a23) '(0)))
(test-assert "grad sum (axis 1, keepdims)"
  (grad-ok? (lambda (a) (weighted (sum a #:axes '(1) #:keepdims #t))) (list a23) '(0)))
(test-assert "grad where"
  (grad-ok? (lambda (a b) (weighted (where (gt a 0.0) (mul a b) (neg b)))) (list a23 b23) '(0 1)))
(test-assert "grad transpose" (grad-ok? (lambda (a) (weighted (transpose a))) (list a23) '(0)))
(test-assert "grad reshape"   (grad-ok? (lambda (a) (weighted (reshape a '(3 2)))) (list a23) '(0)))
(test-assert "grad expand"    (grad-ok? (lambda (b) (weighted (expand b '(4 3)))) (list b3) '(0)))
(test-assert "comparison and logical ops carry no gradient"
  (let ((r ((value-and-grad (lambda (a) (where (logical-and (gt (sum a) 0.0) (lt (sum a) 9.0))
                                                (sum a) (neg (sum a)))))
            a23)))
    (equal? (ctensor->list (cadr r)) (make-list 6 1.0))))

;;; --- composite functions ---------------------------------------------------------

(define-traced (f-if x)
  (if (gt (sum x) 0.0) (weighted (mul x x)) (weighted (neg x))))
(test-assert "grad through a traced if (true side)"  (grad-ok? f-if (list a23) '(0)))
(test-assert "grad through a traced if (false side)" (grad-ok? f-if (list (t '((-1.0 -2.0) (0.5 -0.25)))) '(0)))

(define-traced (f-loop-case x k)
  ;; case inside a loop; the key comes from K (not differentiated)
  (let loop ((i 0) (acc x))
    (if (= i 3)
        (weighted acc)
        (loop (+ i 1)
              (case (sum k)
                ((1.0) (mul acc acc))
                ((2.0) (relu (add acc 0.5)))
                (else (mul (add acc 1.0) 0.5)))))))
(test-assert "grad through case in a loop (k = 1)"
  (grad-ok? f-loop-case (list (t '((0.9 1.1) (0.7 1.2))) (t '(1.0))) '(0)))
(test-assert "grad through case in a loop (k = 2)"
  (grad-ok? f-loop-case (list (t '((0.9 1.1) (0.7 1.2))) (t '(2.0))) '(0)))
(test-assert "grad through case in a loop (else)"
  (grad-ok? f-loop-case (list (t '((0.9 1.1) (0.7 1.2))) (t '(5.0))) '(0)))

(define (mlp-loss w1 b1 w2 x y)
  (let* ((h (relu (add (matmul x w1) b1)))
         (d (sub (matmul h w2) y)))
    (mul (sum (mul d d)) 0.5)))
(test-assert "grad of an MLP loss (all parameters)"
  (grad-ok? mlp-loss
            (list (t '((0.5 -0.3 0.8) (0.2 0.9 -0.4))) (t '(0.1 0.2 0.3))
                  (t '((0.7) (-0.5) (0.6)))
                  (t '((1.0 2.0) (-1.0 0.5))) (t '((1.0) (0.0))))
            '(0 1 2)))

;;; --- double where: division by zero in the unselected side ------------------------

(define safe-recip (lambda (x) (sum (where (gt x 0.0) (div 1.0 x) 0.0))))
(let* ((x (t '(2.0 0.0 -1.0 4.0)))
       (g (ctensor->list (cadr ((value-and-grad safe-recip) x)))))
  (test-assert "where over 1/x at x = 0: gradient is finite (no NaN / inf)"
    (every (lambda (v) (and (= v v) (< (abs v) 1e10))) g))
  (test-equal "where over 1/x: -1/x^2 where selected, 0 elsewhere"
    '(-0.25 0.0 0.0 -0.0625) (map (lambda (v) (+ v 0.0)) g)))

(define-traced (safe-recip-if x)
  ;; same through the macro `if` (rank-0 condition): x = 0 takes the else side
  (if (gt (sum x) 0.0) (sum (div 1.0 x)) (sum (mul x 0.0))))
(test-equal "if over 1/x at x = 0: gradient is 0, not NaN"
  '(0.0 0.0)
  (map (lambda (v) (+ v 0.0)) (ctensor->list (cadr ((value-and-grad safe-recip-if) (t '(0.0 0.0)))))))

;;; --- graph shape -------------------------------------------------------------------

(let ((f (trace-function (lambda (x) (grad (sum (mul x x)) x)))))
  (test-equal "only what the backward pass needs survives DCE (x*x and the sum are gone)"
    '((2.0 4.0))
    (list (ctensor->list (f (t '(1.0 2.0))))))
  (test-assert "forward nodes not needed by the gradient were eliminated"
    (let ((ops (node-ops (cdar (trace-segments f)))))
      (not (memq 'reduce_sum ops)))))

(test-assert "every emitted node with a Core ATen counterpart carries its name"
  (let* ((f (value-and-grad mlp-loss))
         (run (f (t '((0.5 -0.3) (0.2 0.9))) (t '(0.1 0.2)) (t '((0.7) (-0.5)))
               (t '((1.0 2.0))) (t '((1.0)))))
         (doc (call-with-input-string (cdar (trace-segments f)) read)))
    (match doc
      (('sonicboom-s-expr _ ('graph . fields))
       (every (lambda (node)
                (match node
                  (('node op _ _ ('attrs . attrs))
                   (let ((a (assq 'aten attrs)))
                     (and a (string-prefix? "aten::" (cadr (cadr a))))))))
              (cdr (assq 'nodes fields)))))))

(test-assert "grad across a graph break is refused"
  (catch #t
    (lambda ()
      ((trace-function (lambda (x)
                         (let ((y (mul x 2.0)))
                           (if (bool (gt (sum y) 0.0))
                               (grad (sum (mul y y)) x)
                               x))))
       (t '(1.0 2.0)))
      #f)
    (lambda (key . args) (string-contains (format #f "~a" args) "graph break"))))

;;; --- XOR training -------------------------------------------------------------------

(define seed 12345)
(define (rand)
  (set! seed (modulo (+ (* seed 1103515245) 12345) 2147483648))
  (- (* 2.0 (/ seed 2147483648.0)) 1.0))
(define (rand-tensor shape scale)
  (make-ctensor 'float32 shape (map (lambda (_) (* scale (rand))) (iota (shape-numel shape)))))

(define xor-x (t '((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))))
(define xor-y (t '((0.0) (1.0) (1.0) (0.0))))

(define-traced (train-step w1 b1 w2 b2 x y)
  ;; forward + loss + grad + SGD update, one graph
  (let* ((h (relu (add (matmul x w1) b1)))
         (p (add (matmul h w2) b2))
         (d (sub p y))
         (loss (mul (sum (mul d d)) 0.25))
         (gs (grad loss (list w1 b1 w2 b2))))
    (cons loss (map (lambda (w g) (sub w (mul g 0.1))) (list w1 b1 w2 b2) gs))))

(define curve
  (let loop ((step 0)
             (params (list (rand-tensor '(2 8) 1.0) (rand-tensor '(8) 0.1)
                           (rand-tensor '(8 1) 1.0) (rand-tensor '(1) 0.1)))
             (acc '()))
    (let* ((r (apply train-step (append params (list xor-x xor-y))))
           (loss (ctensor-item (car r)))
           (acc (if (zero? (modulo step 100)) (cons (cons step loss) acc) acc)))
      (if (= step 1000)
          (reverse acc)
          (loop (+ step 1) (cdr r) acc)))))

(format #t "~%max relative gradient error: ~,3e~%" %max-rel)
(format #t "XOR loss (step: loss):~%")
(for-each (lambda (p) (format #t "  ~4d: ~,6f~%" (car p) (cdr p))) curve)

(test-assert "XOR: loss falls below 0.05" (< (cdr (last curve)) 0.05))
(test-assert "XOR: loss decreases over training" (< (cdr (last curve)) (cdr (first curve))))
(test-equal "XOR: traced once, every later step hits the cache"
  '(1001 1 1000 0) (map (lambda (k) (stat train-step k)) '(calls traces cache-hits graph-breaks)))
(test-equal "XOR: one graph per step" '(1001 1)
  (list (stat train-step 'executor-calls) (length (trace-segments train-step))))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-grad")
(exit (if (zero? failures) 0 1))
