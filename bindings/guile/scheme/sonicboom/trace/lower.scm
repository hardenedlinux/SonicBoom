;;; SonicBoom — tracing prototype: lower a traced forward graph to S-Expr v0.1
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The tracer records a graph over its own operator vocabulary (`matmul`,
;;; `relu`, `add`, ...). The S-Expr v0.1 runtime accepts a much smaller set —
;;; the operators the CPU Boom executor lowers (core/src/exec/cpu_executor.cpp;
;;; the frozen vocabulary is still the seven ops the parser and planner call
;;; "MLIR-routed", now executed directly):
;;;
;;;   relu add reshape conv max_pool reduce_mean gemm
;;;
;;; This module rewrites a traced document datum onto that set so it can be
;;; executed by libsonicboom.so instead of the pure-Scheme reference
;;; interpreter. Emission follows (sonicboom nn)'s conventions exactly: quoted
;;; value names, `(parameter ... (data :values ...))` for every constant, and a
;;; `linear` (matmul + bias add) lowered to a single `gemm` node with the bias
;;; as its third input.
;;;
;;; Mapped so far: `matmul` (+ rank-1 `add`) → `gemm`, `add`, `relu` (all via
;;; gemm), `reshape`, `reduce_mean`. The runtime has no axes attribute, so a
;;; `reduce_mean` gets a synthesised rank-1 int64 `:values` axes parameter, and
;;; a non-keepdims reduction (rank-lowering) becomes `reduce_mean` + `reshape`.
;;;
;;; Anything outside the mapping raises an error naming the offending op —
;;; this is a forward-graph lowering only (no backward, no control flow).
;;;
;;; The core's `add` is `arith.addf` with no broadcasting, which is why the
;;; bias add is fused into `gemm` rather than lowered on its own.

(define-module (sonicboom trace lower)
  #:use-module (srfi srfi-1)
  #:use-module (ice-9 hash-table)
  #:use-module (rnrs bytevectors)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace tracer)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom executable)
  #:export (trace-graph traced->s-expr run-traced))

;;; --- document datum accessors ------------------------------------------------

(define (graph-of doc)
  (unless (and (pair? doc) (eq? (car doc) 'sonicboom-s-expr))
    (error "traced->s-expr: not an S-Expr v0.1 document" doc))
  (caddr doc))                          ; the (graph ...) form

(define (field g key)
  (let ((f (assq key (cdr g))))
    (if f (cdr f) (error "traced->s-expr: graph is missing section" key))))

(define (node-op n) (cadr n))
(define (node-inputs n) (cdr (assq 'inputs (cddr n))))
(define (node-outputs n) (cdr (assq 'outputs (cddr n))))
(define (node-output-names n) (map car (node-outputs n)))
(define (node-attrs n)
  (let ((a (assq 'attrs (cddr n)))) (if a (cdr a) '())))

;; Attributes are stored as `(key (int 3))` / `(key (ints 1 2))`.
(define (attr-int attrs key default)
  (let ((a (assq key attrs))) (if a (cadr (cadr a)) default)))
(define (attr-ints attrs key)
  (let ((a (assq key attrs))) (if a (cdr (cadr a)) '())))

(define (decl-type type)
  "A `(tensor <dtype> (shape <dims>))` form -> (dtype <dims>), i.e. car is the
dtype and cadr is the shape list; `(length (cadr t))` is the rank."
  (list (cadr type) (cdr (caddr type))))

;; value name -> (dtype . shape) for the graph's inputs, parameters and every
;; node output — i.e. every value whose rank the lowering may need to know.
(define (value-types g)
  (let ((tbl (make-hash-table)))
    (for-each (lambda (i) (hash-set! tbl (cadr i) (decl-type (caddr i))))
              (field g 'inputs))
    (for-each (lambda (p) (hash-set! tbl (cadr p) (decl-type (caddr p))))
              (field g 'parameters))
    (for-each (lambda (n)
                (for-each (lambda (o)
                            (hash-set! tbl (car o) (decl-type (cadr o))))
                          (node-outputs n)))
              (field g 'nodes))
    tbl))

(define (type-of types name)
  (or (hash-ref types name #f)
      (error "traced->s-expr: no declared type for value" name)))

;;; --- op mapping ---------------------------------------------------------------

;; A node that has no S-Expr v0.1 lowering. `linear` is not a tracer op (it is
;; matmul + add, fused below), and softmax is built from exp/sub/amax/sum/div,
;; none of which the runtime lowers either.
(define (unsupported! op)
  (error (string-append "traced->s-expr: op has no S-Expr v0.1 lowering: "
                        (symbol->string op))))

;; Find the matmuls that complete a `linear`: a matmul whose sole consumer is
;; an `add` with a rank-1 bias. Returns a table MATMUL-OUTPUT -> MATMUL-NODE for
;; every such matmul; the matmul is dropped and the add becomes a `gemm`.
(define (find-linear-fusions nodes types)
  (let ((refs (make-hash-table)) (mms (make-hash-table)) (fuse (make-hash-table)))
    (for-each (lambda (n)
                (for-each (lambda (i) (hash-set! refs i (+ 1 (hash-ref refs i 0))))
                          (node-inputs n)))
              nodes)
    (for-each (lambda (n)
                (when (eq? (node-op n) 'matmul)
                  (hash-set! mms (car (node-output-names n)) n)))
              nodes)
    (for-each
     (lambda (n)
       (when (eq? (node-op n) 'add)
         (let* ((ins (node-inputs n))
                ;; the matmul may be either operand of the commutative add
                (acc (find (lambda (i) (hash-ref mms i #f)) ins))
                (bias (find (lambda (i) (not (eq? i acc))) ins))
                ;; the rank-1 check applies to the bias operand (a parameter),
                ;; not to the matmul output (no declared type)
                (ty (and bias (hash-ref types bias #f))))
           (when (and acc
                      (= 1 (hash-ref refs acc 0))       ; consumed only here
                      ty
                      (= 1 (length (cadr ty))))          ; rank-1 bias
             (hash-set! fuse acc (hash-ref mms acc))))))
     nodes)
    fuse))

;;; --- reduce_mean support ------------------------------------------------------
;;; The runtime takes its axes as an int64 `:values` parameter (there is no
;;; axes attribute) and always keeps the rank, so an axis list is synthesised
;;; per reduction and a rank-lowering reduction gets a trailing `reshape`.

;; The axis list a `reduce_mean` node actually reduces over: an empty `axes`
;; attr means "every axis" (see trace/ops.scm `reduce-shape`), so it is
;; expanded against the operand's rank here.
(define (reduce-axes n types)
  (let* ((in-name (car (node-inputs n)))
         (rank (length (cadr (type-of types in-name))))
         (axes (attr-ints (node-attrs n) 'axes)))
    (map (lambda (a) (if (< a 0) (+ a rank) a))
         (if (null? axes) (iota rank) axes))))

;; The name of the (synthesised) axes parameter for reduction node N. Value
;; names are unique per graph, so this cannot collide with a real parameter.
(define (axes-param-name n)
  (string-append (car (node-output-names n)) ".axes"))

;; Every `reduce_mean` node contributes one parameter of the form
;;   (parameter "<out>.axes" (tensor int64 (shape k)) (data :values ...))
(define (reduce-axes-params nodes types)
  (filter-map
   (lambda (n)
     (if (eq? (node-op n) 'reduce_mean)
         (let ((axes (reduce-axes n types)))
           (list 'parameter (axes-param-name n)
                 `(tensor int64 (shape ,(length axes)))
                 `(data :values ,@axes)))
         #f))
   nodes))

;; SHAPE with the reduced AXES set to 1 — the runtime's reduction output shape.
(define (keepdims-shape shape axes)
  (map (lambda (d i) (if (memv i axes) 1 d)) shape (iota (length shape))))

;; `matmul(x, w)` -> `gemm(x, w)`; the tracer stores w as [k, n], which is the
;; runtime's un-transposed B, so transB is 0. Every branch returns a LIST of
;; node forms (a fused matmul returns none, a rank-lowering reduction returns
;; two), so the caller can plain-append them.
(define (lower-node n fuse types)
  (let ((op (node-op n)))
    (case op
      ((matmul)
       (let ((out (car (node-output-names n))))
         (if (hash-ref fuse out #f)
             '()                             ; folded into the following add
             (list
              (list 'node 'gemm
                    (cons 'inputs (node-inputs n))
                    (cons 'outputs (node-outputs n))
                    (list 'attrs '(transB (int 0))))))))
      ((add)
       ;; a fused add becomes gemm(mm-a, mm-b, bias) producing the add's value
       (let* ((ins (node-inputs n))
              (acc (find (lambda (i) (hash-ref fuse i #f)) ins)))
         (if acc
             (let* ((mm (hash-ref fuse acc))
                    (bias (find (lambda (i) (not (eq? i acc))) ins)))
               (list
                (list 'node 'gemm
                      (cons 'inputs (append (node-inputs mm) (list bias)))
                      (cons 'outputs (node-outputs n))
                      (list 'attrs '(transB (int 0))))))
             (list
              (list 'node 'add
                    (cons 'inputs ins)
                    (cons 'outputs (node-outputs n))
                    (list 'attrs))))))
      ((relu)
       (list
        (list 'node 'relu
              (cons 'inputs (node-inputs n))
              (cons 'outputs (node-outputs n))
              (list 'attrs))))
      ((reshape)
       ;; the runtime reads the target shape from the output's declared type;
       ;; the tracer's `(shape (ints ...))` attr is redundant with it.
       (list
        (list 'node 'reshape
              (cons 'inputs (node-inputs n))
              (cons 'outputs (node-outputs n))
              (list 'attrs))))
      ((reduce_mean)
       (let* ((out-name (car (node-output-names n)))
              (out-ty (decl-type (cadr (assoc out-name (node-outputs n)))))
              (keep (attr-int (node-attrs n) 'keepdims 1))
              (axes (reduce-axes n types))
              (in-name (car (node-inputs n)))
              ;; the runtime keeps the rank, so a rank-lowering reduction emits
              ;; an intermediate keepdims value and reshapes it away
              (mid-name (if (= keep 1) out-name (string-append out-name ".keep")))
              (mid-shape (if (= keep 1)
                             (cadr out-ty)
                             (keepdims-shape (cadr (type-of types in-name)) axes)))
              (reduce
               (list 'node 'reduce_mean
                     (list 'inputs in-name (axes-param-name n))
                     (list 'outputs
                           (list mid-name (list 'tensor (car out-ty)
                                                (cons 'shape mid-shape))))
                     (list 'attrs))))
         (if (= keep 1)
             (list reduce)
             (list reduce
                   (list 'node 'reshape
                         (list 'inputs mid-name)
                         (cons 'outputs (node-outputs n))
                         (list 'attrs))))))
      (else (unsupported! op)))))

;;; --- public API ---------------------------------------------------------------

(define (trace-graph proc . args)
  "Trace PROC on ARGS (which must not graph-break) and return the parsed
S-Expr v0.1 document datum — the graph `traced->s-expr` and `run-traced` take."
  (call-with-input-string (apply trace->sexpr proc args) read))

(define (traced->s-expr graph)
  "Lower the traced document datum GRAPH onto the S-Expr v0.1 runtime
vocabulary and return the document text. Raises an error naming the op for any
node with no lowering."
  (let* ((g (graph-of graph))
         (types (value-types g))
         (nodes (field g 'nodes))
         (fuse (find-linear-fusions nodes types))
         (extra (reduce-axes-params nodes types))
         (lowered (append-map (lambda (n) (lower-node n fuse types)) nodes)))
    (sexpr->text
     `(sonicboom-s-expr
       (version 0 1)
       (graph
        (name ,(cadr (assq 'name (cdr g))))
        (inputs ,@(field g 'inputs))
        (outputs ,@(field g 'outputs))
        (parameters ,@(field g 'parameters) ,@extra)
        (nodes ,@lowered))))))

;;; --- execution ----------------------------------------------------------------

(define (ctensor->f32-bytes t)
  (let* ((n (ctensor-numel t))
         (bv (make-bytevector (* n 4) 0)))
    (let loop ((i 0))
      (when (< i n)
        (bytevector-ieee-single-native-set! bv (* i 4) (* 1.0 (ctensor-ref t i)))
        (loop (+ i 1))))
    bv))

(define (f32-bytes->ctensor bv shape)
  (make-ctensor 'float32 shape
                (let loop ((i 0) (acc '()))
                  (if (< i (shape-numel shape))
                      (loop (+ i 1)
                            (cons (bytevector-ieee-single-native-ref bv (* i 4)) acc))
                      (reverse acc)))))

(define (graph-output-type g name)
  (let loop ((ns (field g 'nodes)))
    (if (null? ns)
        (error "run-traced: no node produces the output" name)
        (let ((hit (assoc name (node-outputs (car ns)))))
          (if hit (decl-type (cadr hit)) (loop (cdr ns)))))))

(define (run-traced graph inputs)
  "Lower GRAPH, run it through libsonicboom.so (`run-s-expr`) on INPUTS — an
alist (value-name . ctensor) as `interpret-sexpr` takes — and return the output
ctensor. The S-Expr C ABI binds exactly one input and returns one output."
  (let* ((g (graph-of graph))
         (in-names (map cadr (field g 'inputs)))
         (out-names (map cadr (field g 'outputs))))
    (unless (= 1 (length in-names))
      (error "run-traced: the S-Expr C ABI binds exactly one input" in-names))
    (unless (= 1 (length out-names))
      (error "run-traced: the S-Expr C ABI returns exactly one output" out-names))
    (let ((x (assoc-ref inputs (car in-names))))
      (unless x (error "run-traced: missing input" (car in-names)))
      (let ((ty (graph-output-type g (car out-names))))
        (call-with-values (lambda () (run-s-expr/cached (traced->s-expr graph)
                                                        (ctensor->f32-bytes x)))
          (lambda (bv err)
            (if err
                (error "run-traced:" err)
                (f32-bytes->ctensor bv (cadr ty)))))))))
