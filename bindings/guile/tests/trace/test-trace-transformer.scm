;;; SonicBoom — tracing prototype milestone: a tiny character-level Transformer
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root (about a minute, interpreted):
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace-transformer.scm
;;;
;;; One pre-norm Transformer block (d_model 16, 2 heads of 8, MLP 32 with GELU,
;;; causal mask by where + a constant lower-triangular matrix), embedding +
;;; learned positions, final layer norm and a vocabulary projection; sequence
;;; length 8, vocabulary = the characters of the training text. One traced
;;; training step does forward, cross-entropy, grad and Adam; it is traced
;;; once and every later step runs the same single graph. Prints the loss
;;; every 50 steps and the graph's node count.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 format)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace tracer)
             (sonicboom trace macro)
             (sonicboom trace autodiff)
             (sonicboom trace layers)
             (sonicboom trace optim))

(define (stat f key) (assq-ref (trace-stats f) key))

;;; --- data ------------------------------------------------------------------------

(define text (apply string-append (make-list 30 "hello world. ")))      ; 390 chars
(define vocab (sort (delete-duplicates (string->list text)) char<?))
(define V (length vocab))
(define (char-id c) (list-index (lambda (x) (char=? x c)) vocab))
(define ids (map char-id (string->list text)))

(define T 8) (define D 16) (define H 2) (define HD 8) (define F 32)
(define (i64 xs) (make-ctensor 'int64 (list (length xs)) xs))
(define (window i)
  ;; inputs ids[i .. i+T), targets ids[i+1 .. i+T]
  (list (i64 (take (drop ids i) T)) (i64 (take (drop ids (+ i 1)) T))))
(define n-windows (- (length ids) T 1))

;;; --- parameters ----------------------------------------------------------------------

(define seed 7)
(define (rand)
  (set! seed (modulo (+ (* seed 1103515245) 12345) 2147483648))
  (- (* 2.0 (/ seed 2147483648.0)) 1.0))
(define (randn shape scale)
  (make-ctensor 'float32 shape (map (lambda (_) (* scale (rand))) (iota (shape-numel shape)))))
(define (full shape v) (make-ctensor 'float32 shape (make-list (shape-numel shape) v)))

(define init-params
  ;; emb pos | wq wk wv wo | ln1 gamma beta | ln2 gamma beta | w1 c1 w2 c2 | lnf gamma beta | wout bout
  (list (randn (list V D) 0.3) (randn (list T D) 0.1)
        (randn (list D D) 0.25) (randn (list D D) 0.25) (randn (list D D) 0.25)
        (randn (list D D) 0.25)
        (full (list D) 1.0) (full (list D) 0.0) (full (list D) 1.0) (full (list D) 0.0)
        (randn (list D F) 0.25) (full (list F) 0.0) (randn (list F D) 0.18) (full (list D) 0.0)
        (full (list D) 1.0) (full (list D) 0.0) (randn (list D V) 0.25) (full (list V) 0.0)))

;;; --- model -----------------------------------------------------------------------------

(define mask (causal-mask T))                                   ; constant bool [T T]

(define (heads x) (permute (reshape x (list T H HD)) '(1 0 2)))  ; [T D] -> [H T HD]
(define (merge-heads x) (reshape (permute x '(1 0 2)) (list T D)))

(define-traced (model params x-ids)
  (match params
    ((emb pos wq wk wv wo g1 b1 g2 b2 w1 c1 w2 c2 gf bf wout bout)
     (let* ((x (add (embedding emb x-ids) pos))
            (h (layer-norm x g1 b1))
            (q (heads (matmul h wq))) (k (heads (matmul h wk))) (v (heads (matmul h wv)))
            (scores (mul (bmm q (transpose k)) (/ 1.0 (sqrt HD))))
            (scores (where mask scores -1e9))                     ; causal
            (attn (merge-heads (bmm (softmax scores) v)))
            (x (add x (matmul attn wo)))
            (x (add x (linear (gelu (linear (layer-norm x g2 b2) w1 c1)) w2 c2))))
       (linear (layer-norm x gf bf) wout bout)))))

(define-traced (train-step params ms vs t x-ids y-ids)
  (let* ((loss (cross-entropy (model params x-ids) y-ids))
         (gs (grad loss params)))
    (cons loss (adam-update params gs ms vs t #:lr 0.01))))

(test-begin "sonicboom-trace-transformer")

;;; --- causal mask ------------------------------------------------------------------------

(let* ((a (window 0)) (x1 (car a))
       (x2 (i64 (append (take (ctensor->list x1) (- T 1)) (list (char-id #\.)))))
       (l1 (ctensor->nested (model init-params x1)))
       (l2 (ctensor->nested (model init-params x2))))
  (test-assert "causal: changing the last token leaves earlier positions' logits unchanged"
    (and (equal? (take l1 (- T 1)) (take l2 (- T 1)))
         (not (equal? (last l1) (last l2))))))

;;; --- training ---------------------------------------------------------------------------

(define steps 300)

(define curve        ; list of (step loss window-mean)
  (let loop ((step 0)
             (state (cons init-params (adam-init init-params)))
             (window-sum 0)
             (acc '()))
    (let* ((r (apply train-step (append state (window (modulo (* step 5) n-windows)))))
           (loss (ctensor-item (car r)))
           (window-sum (+ window-sum loss))
           (report? (zero? (modulo step 50)))
           (acc (if report?
                    (cons (list step loss (/ window-sum (if (zero? step) 1 50))) acc)
                    acc)))
      (if (= step steps)
          (reverse acc)
          (loop (+ step 1) (cdr r) (if report? 0 window-sum) acc)))))

(define graph-nodes
  (match (call-with-input-string (cdar (trace-segments train-step)) read)
    (('sonicboom-s-expr _ ('graph . fields)) (length (cdr (assq 'nodes fields))))))

(format #t "~%vocabulary ~a chars, ~a training windows, ~a parameter tensors~%"
        V n-windows (length init-params))
(format #t "loss (step: loss / mean over the previous 50 steps):~%")
(for-each (lambda (r) (apply format #t "  ~4d: ~,4f / ~,4f~%" r)) curve)
(format #t "training-step graph: ~a nodes, ~a graph(s) for ~a steps~%"
        graph-nodes (length (trace-segments train-step)) (stat train-step 'executor-calls))

(test-assert "loss falls clearly: last 50-step mean below a third of the first loss"
  (< (caddr (last curve)) (/ (cadr (first curve)) 3)))
(test-assert "loss falls clearly: last 50-step mean below 0.5 nats"
  (< (caddr (last curve)) 0.5))
(test-equal "traced once; every later step is a cache hit"
  (list (+ steps 1) 1 steps 0)
  (map (lambda (k) (stat train-step k)) '(calls traces cache-hits graph-breaks)))
(test-equal "every step executes the same single graph"
  (list (+ steps 1) 1)
  (list (stat train-step 'executor-calls) (length (trace-segments train-step))))
(test-assert "the step graph holds forward, backward and the Adam update"
  (let ((ops (match (call-with-input-string (cdar (trace-segments train-step)) read)
               (('sonicboom-s-expr _ ('graph . fields))
                (map cadr (cdr (assq 'nodes fields)))))))
    (every (lambda (op) (memq op ops))
           '(index_select index_add bmm gelu reduce_max exp log gather scatter_add
             sqrt where permute))))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-transformer")
(exit (if (zero? failures) 0 1))
