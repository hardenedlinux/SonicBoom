;;; SonicBoom — tracing prototype: reverse-mode automatic differentiation
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; `grad` works on the graph being traced: called inside a traced function on
;;; a symbolic loss, it walks the nodes recorded so far in the current segment
;;; backwards from the loss and appends the backward nodes to the same
;;; recording, using each op's pure gradient rule from (sonicboom trace ops).
;;; The traced function then returns whatever it likes (loss, gradients,
;;; updated parameters) and the whole forward + backward computation is one
;;; S-Expr graph. Forward values the backward nodes do not read, and forward
;;; nodes nothing needs, are removed by the segment's DCE pass.
;;;
;;; Double where: the gradient of where(c, a, b) towards A is where(c, g, 0).
;;; That is zero where A was not selected, but an elementwise op that produced
;;; A may have a non-finite local derivative exactly there (1/x at x = 0), and
;;; 0 * inf = NaN. So the mask (c, side) travels with the gradient through
;;; elementwise ops, and each local gradient computed under a mask is wrapped
;;; in where(c, local, 0) again before it is summed into the input. An
;;; elementwise mask (a condition with many elements) only lines up with
;;; positions through elementwise ops of unchanged shape; a scalar mask (the
;;; rank-0 condition of a rewritten `if`) holds for every position, so it is
;;; carried through every op (matmul, reduce_sum, reshape, ...) as well.
;;;
;;; Graph breaks: the backward walk covers the current segment only. If the
;;; loss depends on a value computed in an earlier segment (before a graph
;;; break) that could depend on the differentiated inputs, grad refuses.

(define-module (sonicboom trace autodiff)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace ops)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace tracer)
  #:export (grad value-and-grad))

(define (sym-of ctx name)
  (let ((t (table-ref (trace-context-types ctx) name)))
    (unless t (error "grad: untyped value" name))
    (make-sym name (car t) (cdr t))))

(define (emit op args attrs) (apply-op op args attrs))

(define (zeros-like s)
  (make-ctensor (sym-dtype s) (sym-shape s)
                (make-list (shape-numel (sym-shape s)) 0)))

(define (scalar-mask? m) (= 1 (shape-numel (sym-shape (car m)))))

(define (apply-masks g masks)
  (fold (lambda (m acc)
          (if (cdr m)
              (emit 'where (list (car m) acc 0) '())
              (emit 'where (list (car m) 0 acc) '())))
        g masks))

(define (grad loss wrt)
  "The gradients of a rank-0 LOSS with respect to WRT.

Called inside a traced function on symbolic values, it walks the nodes
recorded so far backwards from LOSS and appends the backward nodes to the
same graph, using each operator's gradient rule. A tensor that does not
affect LOSS gets a zero gradient.

Args:
  loss: the rank-0 symbolic loss.
  wrt: a symbolic tensor, or a list of symbolic tensors.
Returns: one gradient tensor, or a list of them in WRT's order.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic)
               (sonicboom trace tracer) (sonicboom trace autodiff))
  (define f (trace-function (lambda (x) (grad (sum (mul x x)) x))))
  (ctensor->list (f (list->ctensor '(1.0 2.0))))
  => (2.0 4.0)
PyTorch: torch.autograd.grad
Note: torch.autograd.grad runs over a graph a forward pass has already
built; this call happens while the forward graph is being traced. The loss
must have one element, and a value computed before a graph break cannot be
differentiated across."
  (let ((rec (current-tracer))
        (single? (sym? wrt))
        (wrt (if (sym? wrt) (list wrt) wrt)))
    (unless rec
      (error "grad: needs tracing (call it inside a traced function, or use value-and-grad)"))
    (unless (every sym? wrt)
      (error "grad: can only differentiate with respect to symbolic tensors" wrt))
    (let* ((ctx (recorder-context rec))
           (nodes (reverse (recorder-nodes rec)))
           (wrt-names (map sym-name wrt))
           ;; forward reachability from WRT
           (fwd (let ((h (make-table)))
                  (for-each (lambda (n) (table-set! h n #t)) wrt-names)
                  (for-each (lambda (g)
                              (when (any (lambda (i) (table-ref h i #f)) (gnode-inputs g))
                                (for-each (lambda (o) (table-set! h o #t))
                                          (gnode-output-names g))))
                            nodes)
                  h))
           (defined (let ((h (make-table)))
                      (for-each (lambda (g) (for-each (lambda (o) (table-set! h o #t))
                                                      (gnode-output-names g)))
                                nodes)
                      (for-each (lambda (p) (table-set! h (gparam-name p) #t))
                                (recorder-params rec))
                      h))
           (grads (make-table)))       ; name -> (gradient . masks)
      (define (needs-grad? name) (table-ref fwd name #f))
      (define (accumulate! name g masks)
        (let ((old (table-ref grads name #f)))
          (table-set! grads name
                      (if old
                          (cons (emit 'add (list (car old) g) '())
                                (filter (lambda (m) (member m (cdr old))) masks))
                          (cons g masks)))))
      (when (sym? loss)
        (unless (= 1 (shape-numel (sym-shape loss)))
          (error "grad: loss must have one element" (sym-shape loss)))
        (when (needs-grad? (sym-name loss))
          (accumulate! (sym-name loss)
                       (as-value rec (make-ctensor (sym-dtype loss) (sym-shape loss) '(1))
                                 #f)
                       '())))
      ;; reverse walk
      (for-each
       (lambda (g)
         (let* ((out-name (car (gnode-output-names g)))
                (entry (table-ref grads out-name #f)))
           (when entry
             (let ((rule (op-gradient (gnode-op g))))
               (unless rule
                 (error "grad: no gradient rule for op" (gnode-op g)))
               (let* ((ins (map (lambda (n) (sym-of ctx n)) (gnode-inputs g)))
                      (out (sym-of ctx out-name))
                      (elementwise? (op-elementwise? (gnode-op g)))
                      (masks (if elementwise? (cdr entry) (filter scalar-mask? (cdr entry))))
                      (locals (rule emit shape-of ins out (car entry) (gnode-attrs g) dtype-of)))
                 (for-each
                  (lambda (in local i)
                    (when (and local (needs-grad? (sym-name in)))
                      (let* ((masked (apply-masks local masks))
                             (reduced (if elementwise?
                                          (sum-to-shape emit shape-of masked (sym-shape in))
                                          masked))
                             ;; masks carried on to the input: the incoming ones,
                             ;; plus where's own, while shapes still line up
                             (own (if (and (eq? (gnode-op g) 'where) (> i 0))
                                      (list (cons (car ins) (= i 1)))
                                      '()))
                             (carry (if (and elementwise?
                                             (equal? (sym-shape in) (sym-shape out)))
                                        (append masks own)
                                        (filter scalar-mask? (append masks own)))))
                        (accumulate! (sym-name in) reduced carry))))
                  ins locals (iota (length ins))))))))
       (reverse nodes))
      ;; A loss ancestor computed in an earlier segment (before a graph break)
      ;; may depend on WRT through nodes this walk cannot see: refuse.
      (when (sym? loss)
        (let ((anc (make-table)))
          (table-set! anc (sym-name loss) #t)
          (for-each (lambda (g)
                      (when (any (lambda (o) (table-ref anc o #f)) (gnode-output-names g))
                        (for-each (lambda (i) (table-set! anc i #t)) (gnode-inputs g))))
                    (reverse nodes))
          (for-each
           (lambda (name)
             (unless (or (table-ref defined name #f) (member name wrt-names)
                         (string-prefix? "arg" name))
               (error "grad: the loss depends on a value computed before a graph \
break; differentiating across graph breaks is not supported" name)))
           (map car (filter cdr (table->alist anc))))))
      (let ((result (map (lambda (w)
                           (let ((e (table-ref grads (sym-name w) #f)))
                             (if e (car e) (as-value rec (zeros-like w) #f))))
                         wrt)))
        (if single? (car result) result)))))

(define (value-and-grad f)
  "A traced function returning a loss together with its gradients.

Calling the result with tensor arguments runs F and returns
(loss gradient ...), one gradient per tensor argument; non-tensor arguments
are passed through to F and are static. Called inside another trace, the
function is inlined.

Args:
  f: a procedure from tensor arguments to a rank-0 loss.
Returns: a traced procedure.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic)
               (sonicboom trace tracer) (sonicboom trace autodiff))
  (let* ((f (value-and-grad (lambda (x) (sum (mul x x)))))
         (r (f (list->ctensor '(1.0 2.0)))))
    (map ctensor->list r))
  => ((5.0) (2.0 4.0))
PyTorch: torch.func.grad_and_value
Note: Returns the value before the gradients; grad_and_value returns the
gradients first."
  (trace-function
   (lambda args
     (let ((loss (apply f args)))
       (cons loss (grad loss (filter sym? args)))))))
