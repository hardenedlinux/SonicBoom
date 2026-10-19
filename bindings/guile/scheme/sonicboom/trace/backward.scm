;;; SonicBoom — tracing prototype: tape reverse pass with fused adjoint ops
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; `backward!` is the training counterpart of `grad` in (sonicboom trace
;;; autodiff). Called on a symbolic loss inside a traced function, it walks the
;;; nodes recorded so far backwards from the loss and appends the adjoint nodes
;;; to the same recording, so the graph handed to the lowering holds the forward
;;; and the backward pass together.
;;;
;;; The difference from `grad` is where the adjoints come from: the fused
;;; training ops (matmul, softmax, cross-entropy, RMSNorm, RoPE, SwiGLU,
;;; embedding) emit their Boom IR backward operators
;;;
;;;   matmul_bwd, softmax_bwd, cross_entropy_bwd, rmsnorm_bwd,
;;;   rope_bwd, silu_mul_bwd, embedding_bwd
;;;
;;; instead of being decomposed into primitive ops. Every other op reuses its
;;; rule from the op table, so the walk covers the same elementwise, reduction,
;;; shape and indexing vocabulary as `grad`.
;;;
;;; Results land in the `.grad` field of the caller's leaf tensors (parameters):
;;; a leaf passed as an argument is a leaf in the trace too, `backward!` stores
;;; the symbolic adjoint there, the tracer makes it a segment output, and after
;;; replay writes the concrete gradient back into the caller's leaf. `zero-grad!`
;;; clears the slot. Because the training graph is replayed, parameters should
;;; be leaves passed as arguments, not free variables computed inside the body.

(define-module (sonicboom trace backward)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace ops)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom trace tracer)
  #:export (backward! zero-grad! leaf-grads))

(define (sym-of ctx name)
  (let ((t (table-ref (trace-context-types ctx) name)))
    (unless t (error "backward!: untyped value" name))
    (make-sym name (car t) (cdr t))))

(define (emit op args attrs) (apply-op op args attrs))

(define (sym-type s) (cons (sym-dtype s) (sym-shape s)))

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

;;; --- adjoints ----------------------------------------------------------------------
;;
;;; (adjoint op emit multi shape ins out g attrs) -> list of input gradients
;;; (#f for inputs that carry none), in the forward op's input order. MULTI
;;; records a node with several results and returns their symbolic values; its
;;; result types are declared here because the op table describes single-result
;;; ops only.

(define (adjoint op emit multi shape ins out g attrs)
  (define (generic)
    (let ((rule (op-gradient op)))
      (unless rule (error "backward!: no adjoint rule for op" op))
      (rule emit shape ins out g attrs)))
  (case op
    ((matmul)
     (let ((a (car ins)) (b (cadr ins)))
       (multi 'matmul_bwd (list a b g) '()
              (list (sym-type a) (sym-type b)))))
    ((softmax)
     ;; The backward needs the forward softmax output, not the logits.
     (list (emit 'softmax_bwd (list out g) attrs)))
    ((cross_entropy)
     ;; Results are [loss, dlogits]; only dlogits continues backwards.
     (let ((ds (multi 'cross_entropy_bwd (list (car ins) (cadr ins)) attrs
                      (list (cons 'float32 '()) (sym-type (car ins))))))
       (list (emit 'mul (list (cadr ds) g) '()) #f)))
    ((rms_norm)
     (multi 'rmsnorm_bwd (list (car ins) (cadr ins) g) attrs
            (list (sym-type (car ins)) (sym-type (cadr ins)))))
    ((rope)
     ;; pos0/head_dim/theta_base/freq_scale ride on the forward node.
     (list (emit 'rope_bwd (list g) attrs)))
    ((silu_mul)
     (multi 'silu_mul_bwd (list (car ins) (cadr ins) g) attrs
            (list (sym-type (car ins)) (sym-type (cadr ins)))))
    ((index_select)
     ;; A row gather is an embedding: the table's gradient is embedding_bwd's
     ;; result (the ids carry no gradient).
     (if (= 0 (attr-ref attrs 'axis 0))
         (let* ((table (car ins))
                (ids (cadr ins))
                (vocab (car (sym-shape table))))
           (list (emit 'embedding_bwd (list ids g)
                       `((vocab (int ,vocab))))
                 #f))
         (generic)))
    (else (generic))))

;;; --- public API ----------------------------------------------------------------------

(define (backward! loss)
  "Reverse-mode differentiation of LOSS (a rank-0 symbolic tensor) on the tape
being traced, emitting fused adjoint ops. Stores each parameter leaf's gradient
in its .grad field and returns the list of gradients, in the order the leaves
were bound. Needs tracing; call it inside a traced function."
  (let ((rec (current-tracer)))
    (unless rec
      (error "backward!: needs tracing (call it inside a traced function)"))
    (let* ((loss (unwrap-leaf loss))
           (ctx (recorder-context rec))
           (nodes (reverse (recorder-nodes rec)))
           (leaves (trace-context-leaves ctx)))
      (unless (sym? loss)
        (error "backward!: loss must be a symbolic tensor"))
      (unless (= 1 (shape-numel (sym-shape loss)))
        (error "backward!: loss must have one element" (sym-shape loss)))
      (let* ((leaf-names (filter-map (lambda (p)
                                       (let ((v (leaf-value (cdr p))))
                                         (and (sym? v) (sym-name v))))
                                     leaves))
             ;; forward reachability from the leaves: nothing else needs an adjoint
             (fwd (let ((h (make-table)))
                    (for-each (lambda (n) (table-set! h n #t)) leaf-names)
                    (for-each (lambda (g)
                                (when (any (lambda (i) (table-ref h i #f))
                                           (gnode-inputs g))
                                  (for-each (lambda (o) (table-set! h o #t))
                                            (gnode-output-names g))))
                              nodes)
                    h))
             (grads (make-table)))
        (define (needs-grad? name) (table-ref fwd name #f))
        (define (accumulate! name g masks)
          (let ((old (table-ref grads name #f)))
            (table-set! grads name
                        (if old
                            (cons (emit 'add (list (car old) g) '())
                                  (filter (lambda (m) (member m (cdr old))) masks))
                            (cons g masks)))))
        (when (needs-grad? (sym-name loss))
          (accumulate! (sym-name loss)
                       (as-value rec
                                 (make-ctensor (sym-dtype loss) (sym-shape loss) '(1))
                                 #f)
                       '()))
        ;; reverse walk
        (for-each
         (lambda (g)
           (let* ((out-name (car (gnode-output-names g)))
                  (entry (table-ref grads out-name #f)))
             (when entry
               (let* ((ins (map (lambda (n) (sym-of ctx n)) (gnode-inputs g)))
                      (out (sym-of ctx out-name))
                      (elementwise? (op-elementwise? (gnode-op g)))
                      (masks (if elementwise?
                                 (cdr entry)
                                 (filter scalar-mask? (cdr entry))))
                      (multi (lambda (op args attrs out-types)
                               (apply-op-multi op args attrs out-types)))
                      (locals (adjoint (gnode-op g) emit multi shape-of ins out
                                       (car entry) (gnode-attrs g))))
                 (for-each
                  (lambda (in local i)
                    (when (and local (needs-grad? (sym-name in)))
                      (let* ((masked (apply-masks local masks))
                             (reduced (if elementwise?
                                          (sum-to-shape emit shape-of masked
                                                        (sym-shape in))
                                          masked))
                             (own (if (and (eq? (gnode-op g) 'where) (> i 0))
                                      (list (cons (car ins) (= i 1)))
                                      '()))
                             (carry (if (and elementwise?
                                             (equal? (sym-shape in) (sym-shape out)))
                                        (append masks own)
                                        (filter scalar-mask? (append masks own)))))
                        (accumulate! (sym-name in) reduced carry))))
                  ins locals (iota (length ins)))))))
         (reverse nodes))
        ;; write the gradients into the leaves
        (map (lambda (p)
               (let* ((v (leaf-value (cdr p)))
                      (entry (and (sym? v) (table-ref grads (sym-name v) #f)))
                      (g (if entry (car entry)
                             (as-value rec (zeros-like v) #f)))
                      (old (leaf-grad (cdr p)))
                      (g (if (and (sym? old) (not (eq? entry #f)))
                             (emit 'add (list old g) '())
                             g)))
                 (set-leaf-grad! (cdr p) g)
                 g))
             leaves)))))

(define* (zero-grad! leaves)
  "Clear the .grad slot of a leaf or of every leaf in a list."
  (for-each (lambda (l) (set-leaf-grad! l #f))
            (if (leaf? leaves) (list leaves) leaves))
  leaves)

(define* (leaf-grads leaves)
  "The current .grad of a leaf or of every leaf in a list."
  (if (leaf? leaves)
      (leaf-grad leaves)
      (map leaf-grad leaves)))
