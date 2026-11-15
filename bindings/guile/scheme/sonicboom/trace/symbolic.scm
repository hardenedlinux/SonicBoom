;;; SonicBoom — tracing prototype: symbolic tensors and the op front end
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A symbolic tensor carries only a value name, a dtype and a static shape.
;;; The user-facing ops (add, mul, matmul, relu, sum, ...) behave in two modes,
;;; selected by the `current-tracer` parameter (switched with `parameterize`):
;;;
;;;   eager   (current-tracer = #f)   compute on concrete ctensors right away;
;;;   tracing (current-tracer = rec)  if any operand is symbolic, append a node
;;;                                   to recorder REC and return a new symbolic
;;;                                   tensor; nothing is computed.
;;;
;;; Operations whose operands are all concrete are computed even while tracing
;;; (constant folding); concrete operands of a traced op become inline S-Expr
;;; parameters. `bool` is the only place user code can observe a value: on a
;;; symbolic tensor it aborts to %break-tag, which the tracer turns into a graph
;;; break or a multi-resume (see (sonicboom trace tracer)).

(define-module (sonicboom trace symbolic)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace ops)
  #:use-module (sonicboom trace graph)
  #:export (;; symbolic tensors
            make-sym sym? sym-name sym-dtype sym-shape
            shape-of dtype-of
            ;; mode
            current-tracer tracing? %break-tag %branch-assumptions
            strict-tracing covered-procedure! covered-procedure? trace-pure
            ;; tracing context / recorder (used by the tracer)
            make-trace-context trace-context? trace-context-types
            trace-context-counters context-count!
            trace-context-strict trace-context-notes add-context-note!
            trace-context-max-unroll trace-context-leaves
            set-trace-context-leaves! register-leaf!
            fresh-name! register-type!
            make-recorder recorder? recorder-context recorder-nodes
            recorder-params recorder-param-index
            recorder-count recorder-assumptions
            recorder-depth recorder-spec-base
            set-recorder-nodes! set-recorder-params! set-recorder-param-index!
            set-recorder-count!
            as-value record-node! apply-op-multi unwrap-leaf ref-dtype
            ;; ops
            add sub mul matmul relu neg sum gt lt where bool trace-print
            equal-to logical-and logical-or logical-not
            div transpose permute reshape expand apply-op
            texp tlog ttanh tsqrt gelu amax mean bmm
            index-select index-add gather scatter-add concat slice))

;;; --- symbolic tensor ------------------------------------------------------------

;; A symbolic tensor: a value name, a dtype and a static shape.
;;
;; A symbolic tensor carries no data; it stands for a value of the segment being
;; traced.
;;
;; PyTorch: none
(define-record-type <sym>
  (make-sym name dtype shape)
  sym?
  (name  sym-name)
  (dtype sym-dtype)
  (shape sym-shape))

(define (unwrap-leaf x)
  "X with a leaf tensor replaced by its value; anything else unchanged.

Args:
  x: any value.
Returns: X or the leaf's value.
PyTorch: none"
  (if (leaf? x) (leaf-value x) x))

(define (shape-of x)
  "The shape of X.

Args:
  x: a symbolic tensor, a ctensor, a leaf or a Scheme number or boolean.
Returns: a list of dimensions; a Scheme scalar has the empty shape.
PyTorch: none"
  (cond ((sym? x) (sym-shape x))
        ((ctensor? x) (ctensor-shape x))
        ((leaf? x) (shape-of (leaf-value x)))
        ((or (number? x) (boolean? x)) '())
        (else (error "shape-of: not a tensor" x))))

(define (dtype-of x)
  "The dtype of X, or #f for a value without one.

Args:
  x: a symbolic tensor, a ctensor or a leaf.
Returns: a dtype symbol.
PyTorch: none"
  (cond ((sym? x) (sym-dtype x))
        ((ctensor? x) (ctensor-dtype x))
        ((leaf? x) (dtype-of (leaf-value x)))
        (else #f)))

;;; --- mode -------------------------------------------------------------------------

;; The parameter holding the active recorder, or #f in eager mode.
;;
;; `parameterize` sets it to a recorder while a traced function runs.
;;
;; PyTorch: none
(define current-tracer (make-parameter #f))
(define (tracing?)
  "True when a recorder is active.

Returns: a boolean.
PyTorch: none"
  (and (current-tracer) #t))

;; The prompt tag `bool` aborts to when it needs a concrete condition value.
;;
;; The tracer catches the abort and resumes tracing on one or both sides.
;;
;; PyTorch: none
(define %break-tag (%make-prompt-tag 'sonicboom-trace-break))

;; The default for traced functions created without #:strict?.
;;
;; When true, passing a symbolic tensor to a procedure the macro walker did not
;; cover is reported.
;;
;; PyTorch: none
(define strict-tracing (make-parameter #f))

;; Procedures known not to hide a core `if` on a symbolic tensor: the tensor
;; ops below, procedures created inside a define-traced body, traced
;; functions (checked separately by the tracer).
(define (covered-procedure! proc)
  "Mark PROC as covered by the macro walker for strict mode.

Args:
  proc: a procedure.
Returns: PROC.
PyTorch: none"
  (procedure-tag! proc 'sonicboom-trace-covered #t)
  proc)
(define (covered-procedure? proc)
  "True when PROC was marked with `covered-procedure!`.

Args:
  proc: a procedure.
Returns: a boolean.
PyTorch: none"
  (and (procedure-tag proc 'sonicboom-trace-covered) #t))

(define (trace-pure proc)
  "Declare PROC safe to receive symbolic tensors in strict mode.

PROC must not branch on a tensor value.

Args:
  proc: a procedure.
Returns: PROC.
PyTorch: none"
  (covered-procedure! proc))

;; The condition values fixed by an enclosing macro-rewritten `if`, as an alist.
;;
;; Kept as a parameter rather than recorder state, so that a continuation captured
;; inside an arm restores it.
;;
;; PyTorch: none
(define %branch-assumptions (make-parameter '()))

;;; --- tracing context ------------------------------------------------------------

;; The shared state of one traced function.
;;
;; TYPES maps every value name introduced so far to a (dtype . shape) pair;
;; COUNTERS counts events; LEAVES is an alist input-name -> leaf tensor of the
;; current trace. STRICT is #t, #f or 'default, NOTES collects strict-mode
;; reports, and MAX-UNROLL bounds loop unrolling.
;;
;; PyTorch: none
(define-record-type <trace-context>
  (%make-trace-context next-id types max-side-nodes counters strict notes
                       max-unroll leaves)
  trace-context?
  (next-id trace-context-next-id set-trace-context-next-id!)
  (types trace-context-types)               ; table: name -> (dtype . shape)
  (max-side-nodes trace-context-max-side-nodes)
  (counters trace-context-counters)         ; table: event -> count
  (strict trace-context-strict)
  (notes trace-context-notes set-trace-context-notes!)
  (max-unroll trace-context-max-unroll)
  ;; Alist input-name -> leaf tensor (parameters of the current fresh trace).
  ;; backward! reads their .grad, and the tracer turns those gradient values
  ;; into segment outputs and writes them back to the caller's leaves.
  (leaves trace-context-leaves set-trace-context-leaves!))

(define* (make-trace-context max-side-nodes #:optional (strict 'default)
                             (max-unroll 64))
  "A fresh trace context.

Args:
  max-side-nodes: the node budget of one speculative side.
  strict: optional; #t, #f or 'default (default 'default).
  max-unroll: optional; the loop-unrolling bound (default 64).
Returns: a trace context.
PyTorch: none"
  (%make-trace-context 0 (make-table) max-side-nodes (make-table) strict '()
                       max-unroll '()))

(define (register-leaf! ctx name leaf)
  "Record LEAF as the leaf tensor bound to input NAME for the current trace.

A leaf registered here receives the gradient `backward!` records and any value
the traced body writes to it.

Args:
  ctx: a trace context.
  name: the graph input name.
  leaf: a leaf tensor.
Returns: unspecified.
PyTorch: none"
  (set-trace-context-leaves!
   ctx (acons name leaf (filter (lambda (p) (not (equal? (car p) name)))
                                (trace-context-leaves ctx)))))

(define (context-count! ctx key)
  "Increment the counter KEY in CTX.

Args:
  ctx: a trace context.
  key: a symbol.
Returns: unspecified.
PyTorch: none"
  (let ((h (trace-context-counters ctx)))
    (table-set! h key (+ 1 (table-ref h key 0)))))

(define (add-context-note! ctx note)
  "Append NOTE to the strict-mode notes of CTX.

Args:
  ctx: a trace context.
  note: a string.
Returns: unspecified.
PyTorch: none"
  (set-trace-context-notes! ctx (append (trace-context-notes ctx) (list note))))

(define (fresh-name! ctx prefix)
  "A new value name with PREFIX, unique within CTX.

Args:
  ctx: a trace context.
  prefix: a string.
Returns: a string.
PyTorch: none"
  (let ((n (trace-context-next-id ctx)))
    (set-trace-context-next-id! ctx (+ n 1))
    (string-append prefix (number->string n))))

(define (register-type! ctx name dtype shape)
  "Record that value NAME in CTX has DTYPE and SHAPE.

Args:
  ctx: a trace context.
  name: the value name.
  dtype: a dtype symbol.
  shape: a list of dimensions.
Returns: unspecified.
PyTorch: none"
  (table-set! (trace-context-types ctx) name (cons dtype shape)))

;; The state of one segment being traced.
;;
;; NODES and PARAMS are kept in reverse order as shared-tail lists. PARAM-INDEX
;; maps concrete objects to parameter names by identity. ASSUMPTIONS maps
;; condition value names to the boolean the current path took. SPEC-BASE is #f
;; outside speculation.
;;
;; PyTorch: none
(define-record-type <recorder>
  (make-recorder context nodes params param-index count assumptions depth spec-base)
  recorder?
  (context     recorder-context)
  (nodes       recorder-nodes set-recorder-nodes!)
  (params      recorder-params set-recorder-params!)
  (param-index recorder-param-index set-recorder-param-index!)
  (count       recorder-count set-recorder-count!)
  (assumptions recorder-assumptions)
  (depth       recorder-depth)
  (spec-base   recorder-spec-base))

(define (lift x dtype)
  "Scheme number / boolean / ctensor / leaf -> ctensor."
  (let ((x (unwrap-leaf x)))
    (cond ((ctensor? x) x)
          ((boolean? x) (scalar->ctensor x 'bool))
          ((number? x) (scalar->ctensor x (or dtype 'float32)))
          (else (error "not a tensor operand" x)))))

(define (as-value rec x dtype)
  "The symbolic value of X in recorder REC.

A symbolic tensor passes through; a concrete operand becomes an inline
parameter of the current segment.

Args:
  rec: a recorder.
  x: a symbolic tensor, a ctensor or a Scheme number or boolean.
  dtype: the dtype used to lift a bare number, or #f.
Returns: a symbolic tensor.
PyTorch: none"
  (let ((x (unwrap-leaf x)))
    (if (sym? x)
        x
        (as-value-1 rec x dtype))))

(define (as-value-1 rec x dtype)
  "Concrete operand X captured by REC: a parameter of the current segment.
A ctensor captured again by identity in the same segment reuses the parameter
already emitted for it, so its value list is not materialised twice. Bare
numbers and booleans are lifted to a fresh scalar ctensor on every capture and
are therefore not deduplicated (identity, not value, is the key)."
  (let ((hit (and (ctensor? x) (assq x (recorder-param-index rec)))))
    (if hit
        (make-sym (cdr hit) (ctensor-dtype x) (ctensor-shape x))
        (let* ((t (lift x dtype))
               (ctx (recorder-context rec))
               (name (fresh-name! ctx "c")))
          (register-type! ctx name (ctensor-dtype t) (ctensor-shape t))
          (set-recorder-params!
           rec (cons (make-gparam name (ctensor-dtype t) (ctensor-shape t)
                                  (ctensor->list t))
                     (recorder-params rec)))
          (when (ctensor? x)
            (set-recorder-param-index!
             rec (cons (cons x name) (recorder-param-index rec))))
          (make-sym name (ctensor-dtype t) (ctensor-shape t))))))

(define (record-node! rec op ins attrs)
  "Append the node OP(INS) to REC and return its symbolic output.

Args:
  rec: a recorder.
  op: an operator symbol.
  ins: the input symbolic values.
  attrs: the S-Expr attribute list.
Returns: the symbolic output.
PyTorch: none"
  (let* ((type (op-type op (map (lambda (s) (cons (sym-dtype s) (sym-shape s))) ins)
                        attrs))
         (ctx (recorder-context rec))
         (name (fresh-name! ctx "v")))
    (register-type! ctx name (car type) (cdr type))
    (set-recorder-nodes!
     rec (cons (make-gnode op (map sym-name ins)
                           (list (list name (car type) (cdr type)))
                           attrs)
               (recorder-nodes rec)))
    (set-recorder-count! rec (+ 1 (recorder-count rec)))
    (let ((base (recorder-spec-base rec)))
      (when (and base (> (- (recorder-count rec) base)
                         (trace-context-max-side-nodes ctx)))
        (%raise-signal 'sonicboom-trace-speculation-overflow)))
    (make-sym name (car type) (cdr type))))

(define (record-node-multi! rec op ins attrs out-types)
  "Append a node OP(INS) with several outputs to REC and return the symbolic
outputs, one per element of OUT-TYPES (a list of (dtype . shape)). The type
rules of the op table describe single-output ops only; a multi-output op (the
fused backward operators) declares its result types here instead."
  (let* ((ctx (recorder-context rec))
         (names (map (lambda (_) (fresh-name! ctx "v")) out-types)))
    (for-each (lambda (n t) (register-type! ctx n (car t) (cdr t)))
              names out-types)
    (set-recorder-nodes!
     rec (cons (make-gnode op (map sym-name ins)
                           (map (lambda (n t) (list n (car t) (cdr t)))
                                names out-types)
                           attrs)
               (recorder-nodes rec)))
    (set-recorder-count! rec (+ 1 (recorder-count rec)))
    (let ((base (recorder-spec-base rec)))
      (when (and base (> (- (recorder-count rec) base)
                         (trace-context-max-side-nodes ctx)))
        (%raise-signal 'sonicboom-trace-speculation-overflow)))
    (map (lambda (n t) (make-sym n (car t) (cdr t))) names out-types)))

;;; --- op front end -------------------------------------------------------------

;; dtype used to lift bare Scheme numbers: that of the first non-bool tensor
;; operand (so `(add x 1)` on an int64 tensor stays int64), else float32.
(define (ref-dtype args)
  "The dtype used to lift bare Scheme numbers among ARGS.

The dtype of the first non-boolean tensor operand, or float32 when there is
none.

Args:
  args: a list of operands.
Returns: a dtype symbol.
PyTorch: none"
  (or (find (lambda (d) (and d (not (eq? d 'bool)))) (map dtype-of args))
      'float32))

(define (apply-op op args attrs)
  "Apply OP to ARGS with ATTRS in the current mode.

When no argument is symbolic the kernel runs and the concrete result is
returned; while tracing, a node is recorded and a symbolic tensor returned.

Args:
  op: an operator symbol.
  args: the operands.
  attrs: the S-Expr attribute list.
Returns: a ctensor in eager mode, a symbolic tensor while tracing.
PyTorch: none"
  (let* ((args (map unwrap-leaf args))
         (rec (current-tracer))
         (dt (ref-dtype args)))
    (cond ((not (or (any sym? args)
                    ;; an effect must happen at run time, not at trace time
                    (and rec (op-effectful? op))))
           (op-kernel op (map (lambda (a) (lift a dt)) args) attrs))
          ((not rec)
           (error "symbolic tensor used outside of tracing" op))
          (else
           (record-node! rec op (map (lambda (a) (as-value rec a dt)) args)
                         attrs)))))

(define (apply-op-multi op args attrs out-types)
  "Apply an operator with several results.

Like `apply-op`, with the result types declared by OUT-TYPES.

Args:
  op: an operator symbol.
  args: the operands.
  attrs: the S-Expr attribute list.
  out-types: a list of (dtype . shape), one per result.
Returns: a list of ctensors in eager mode, or of symbolic tensors while
tracing.
PyTorch: none"
  (let* ((args (map unwrap-leaf args))
         (rec (current-tracer))
         (dt (ref-dtype args)))
    (cond ((not (or (any sym? args) (and rec (op-effectful? op))))
           (op-kernel op (map (lambda (a) (lift a dt)) args) attrs))
          ((not rec)
           (error "symbolic tensor used outside of tracing" op))
          (else
           (record-node-multi! rec op
                               (map (lambda (a) (as-value rec a dt)) args)
                               attrs out-types)))))

(define (add a b)
  "The elementwise sum of A and B.

Args:
  a: a tensor.
  b: a tensor, broadcast against A.
Returns: a tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (add (list->ctensor '(1.0 2.0)) (list->ctensor '(3.0 4.0))))
  => (4.0 6.0)
PyTorch: torch.add"
  (apply-op 'add (list a b) '()))
(define (sub a b)
  "The elementwise difference A - B.

Args:
  a: a tensor.
  b: a tensor, broadcast against A.
Returns: a tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (sub (list->ctensor '(1.0 2.0)) (list->ctensor '(3.0 4.0))))
  => (-2.0 -2.0)
PyTorch: torch.sub"
  (apply-op 'sub (list a b) '()))
(define (mul a b)
  "The elementwise product of A and B.

Args:
  a: a tensor.
  b: a tensor, broadcast against A.
Returns: a tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (mul (list->ctensor '(1.0 2.0)) (list->ctensor '(3.0 4.0))))
  => (3.0 8.0)
PyTorch: torch.mul"
  (apply-op 'mul (list a b) '()))
(define (matmul a b)
  "The matrix product of A and B.

Args:
  a: a tensor [.., m, k].
  b: a tensor [.., k, n].
Returns: a tensor [.., m, n].
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (matmul (list->ctensor '((1.0 2.0)))
                         (list->ctensor '((3.0) (4.0)))))
  => (11.0)
PyTorch: torch.matmul"
  (apply-op 'matmul (list a b) '()))
(define (relu x)
  "The elementwise rectified linear unit max(X, 0).

Args:
  x: a tensor.
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (relu (list->ctensor '(-1.0 2.0))))
  => (0.0 2.0)
PyTorch: torch.relu"
  (apply-op 'relu (list x) '()))
(define (neg x)
  "The elementwise negation of X.

Args:
  x: a tensor.
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (neg (list->ctensor '(1.0 2.0))))
  => (-1.0 -2.0)
PyTorch: torch.neg"
  (apply-op 'neg (list x) '()))
(define (gt a b)
  "The elementwise comparison A > B.

Args:
  a: a tensor.
  b: a tensor, broadcast against A.
Returns: a bool tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (gt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0))))
  => (#f #f)
PyTorch: torch.gt"
  (apply-op 'greater (list a b) '()))
(define (lt a b)
  "The elementwise comparison A < B.

Args:
  a: a tensor.
  b: a tensor, broadcast against A.
Returns: a bool tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (lt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0))))
  => (#t #f)
PyTorch: torch.lt"
  (apply-op 'less (list a b) '()))
(define (where c x y)
  "The elementwise selection C ? X : Y.

Args:
  c: a bool tensor.
  x: the values where C is true.
  y: the values where C is false.
Returns: a tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list
   (where (gt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0)))
          (list->ctensor '(1.0 2.0)) (list->ctensor '(3.0 4.0))))
  => (3.0 4.0)
PyTorch: torch.where"
  (apply-op 'where (list c x y) '()))
(define (equal-to a b)
  "The elementwise equality A == B.

Args:
  a: a tensor.
  b: a tensor, broadcast against A.
Returns: a bool tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list
   (equal-to (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0))))
  => (#f #t)
PyTorch: torch.eq"
  (apply-op 'equal (list a b) '()))
(define (logical-and a b)
  "The elementwise logical conjunction of two bool tensors.

Args:
  a: a bool tensor.
  b: a bool tensor, broadcast against A.
Returns: a bool tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list
   (logical-and (gt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0)))
                (lt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0)))))
  => (#f #f)
PyTorch: torch.logical_and"
  (apply-op 'and (list a b) '()))
(define (logical-or a b)
  "The elementwise logical disjunction of two bool tensors.

Args:
  a: a bool tensor.
  b: a bool tensor, broadcast against A.
Returns: a bool tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list
   (logical-or (gt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0)))
               (lt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0)))))
  => (#t #f)
PyTorch: torch.logical_or"
  (apply-op 'or (list a b) '()))
(define (logical-not x)
  "The elementwise logical negation of a bool tensor.

Args:
  x: a bool tensor.
Returns: a bool tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list
   (logical-not (gt (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0)))))
  => (#t #t)
PyTorch: torch.logical_not"
  (apply-op 'not (list x) '()))
(define (div a b)
  "The elementwise quotient A / B.

Args:
  a: a tensor.
  b: a tensor, broadcast against A.
Returns: a tensor of the broadcast shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (div (list->ctensor '(1.0 2.0)) (list->ctensor '(2.0 2.0))))
  => (0.5 1.0)
PyTorch: torch.div"
  (apply-op 'div (list a b) '()))
(define (permute x dims)
  "X with its axes in the order DIMS.

Args:
  x: a tensor.
  dims: the new order of the axes.
Returns: a tensor.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-shape
   (permute (make-ctensor 'float32 '(2 3 4) (iota 24)) '(2 0 1)))
  => (4 2 3)
PyTorch: torch.permute"
  (apply-op 'permute (list x) `((dims (ints ,@dims)))))
(define (transpose x)
  "X with its last two axes swapped.

Args:
  x: a tensor of rank two or more.
Returns: a tensor.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-shape (transpose (make-ctensor 'float32 '(2 3) (iota 6))))
  => (3 2)
PyTorch: torch.transpose"
  (let ((r (length (shape-of x))))
    (permute x (append (iota (- r 2)) (list (- r 1) (- r 2))))))
;; t-prefixed so that R7RS exp / log / sqrt on numbers stay unshadowed
(define (texp x)
  "The elementwise exponential of X.

Named with a t prefix so that the R7RS `exp` on numbers is not shadowed.

Args:
  x: a tensor.
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (texp (list->ctensor '(0.0))))
  => (1.0)
PyTorch: torch.exp"
  (apply-op 'exp (list x) '()))
(define (tlog x)
  "The elementwise natural logarithm of X.

Named with a t prefix so that the R7RS `log` on numbers is not shadowed.

Args:
  x: a tensor.
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (tlog (list->ctensor '(1.0))))
  => (0.0)
PyTorch: torch.log"
  (apply-op 'log (list x) '()))
(define (ttanh x)
  "The elementwise hyperbolic tangent of X.

Named with a t prefix so that the R7RS name is not shadowed.

Args:
  x: a tensor.
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (ttanh (list->ctensor '(0.0))))
  => (0.0)
PyTorch: torch.tanh"
  (apply-op 'tanh (list x) '()))
(define (tsqrt x)
  "The elementwise square root of X.

Named with a t prefix so that the R7RS `sqrt` on numbers is not shadowed.

Args:
  x: a tensor.
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (tsqrt (list->ctensor '(4.0))))
  => (2.0)
PyTorch: torch.sqrt"
  (apply-op 'sqrt (list x) '()))
(define (gelu x)
  "The elementwise Gaussian error linear unit of X.

Args:
  x: a tensor.
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (gelu (list->ctensor '(0.0))))
  => (0.0)
PyTorch: torch.nn.functional.gelu"
  (apply-op 'gelu (list x) '()))
(define* (amax x #:key (axes '()) (keepdims #f))
  "The maximum of X over AXES.

An empty AXES reduces every axis.

Args:
  x: a tensor.
  axes: keyword; the axes to reduce (default: every axis).
  keepdims: keyword; keep the reduced axes with size 1 (default #f).
Returns: a tensor with X's shape, or the reduced shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-item (amax (list->ctensor '((1.0 3.0) (2.0 0.0)))))
  => 3.0
PyTorch: torch.amax"
  (apply-op 'reduce_max (list x)
            `((axes (ints ,@axes)) (keepdims (int ,(if keepdims 1 0))))))
(define* (mean x #:key (axes '()) (keepdims #f))
  "The mean of X over AXES.

An empty AXES averages every axis.

Args:
  x: a tensor.
  axes: keyword; the axes to reduce (default: every axis).
  keepdims: keyword; keep the reduced axes with size 1 (default #f).
Returns: a tensor with X's shape, or the reduced shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-item (mean (list->ctensor '(1.0 2.0 3.0))))
  => 2.0
PyTorch: torch.mean"
  (apply-op 'reduce_mean (list x)
            `((axes (ints ,@axes)) (keepdims (int ,(if keepdims 1 0))))))
(define (bmm a b)
  "The batched matrix product of A and B.

Args:
  a: a tensor [batch, m, k].
  b: a tensor [batch, k, n].
Returns: a tensor [batch, m, n].
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-shape (bmm (make-ctensor 'float32 '(2 2 3) (iota 12))
                      (make-ctensor 'float32 '(2 3 2) (iota 12))))
  => (2 2 2)
PyTorch: torch.bmm"
  (apply-op 'bmm (list a b) '()))
(define (index-select x axis index)
  "The slices of X along AXIS at the positions INDEX.

Args:
  x: a tensor.
  axis: the axis to select along.
  index: an integer tensor of positions.
Returns: a tensor; the selected axis has the length of INDEX.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->nested
   (index-select (list->ctensor '((1.0 2.0) (3.0 4.0))) 0
                 (make-ctensor 'int64 '(2) '(1 0))))
  => ((3.0 4.0) (1.0 2.0))
PyTorch: torch.index_select"
  (apply-op 'index_select (list x index) `((axis (int ,axis)))))
(define (index-add base axis index src)
  "BASE with SRC added at the positions INDEX along AXIS.

Args:
  base: a tensor.
  axis: the axis to add along.
  index: an integer tensor of positions.
  src: the values to add; the selected axis has the length of INDEX.
Returns: a new tensor; BASE is not modified.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->nested
   (index-add (make-ctensor 'float32 '(2 2) '(0.0 0.0 0.0 0.0)) 0
              (make-ctensor 'int64 '(1) '(0))
              (list->ctensor '((1.0 2.0)))))
  => ((1.0 2.0) (0.0 0.0))
PyTorch: torch.Tensor.index_add_
Note: torch.Tensor.index_add_ updates the base tensor in place."
  (apply-op 'index_add (list base index src) `((axis (int ,axis)))))
(define (gather x axis index)
  "The elements of X selected by INDEX along AXIS.

Args:
  x: a tensor.
  axis: the axis to gather along.
  index: an integer tensor; it has the shape of the result.
Returns: a tensor with INDEX's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->nested
   (gather (list->ctensor '((1.0 2.0) (3.0 4.0))) 1
           (make-ctensor 'int64 '(2 1) '(1 0))))
  => ((2.0) (3.0))
PyTorch: torch.gather"
  (apply-op 'gather (list x index) `((axis (int ,axis)))))
(define (scatter-add base axis index src)
  "BASE with SRC added at the positions INDEX along AXIS.

Positions that repeat in INDEX receive every corresponding value of SRC.

Args:
  base: a tensor.
  axis: the axis to scatter along.
  index: an integer tensor giving, for each element of SRC, its position.
  src: the values to add.
Returns: a new tensor; BASE is not modified.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->nested
   (scatter-add (make-ctensor 'float32 '(2 2) '(0.0 0.0 0.0 0.0)) 0
                (make-ctensor 'int64 '(1 2) '(0 1))
                (list->ctensor '((1.0 2.0)))))
  => ((1.0 0.0) (0.0 2.0))
PyTorch: torch.Tensor.scatter_add_
Note: torch.Tensor.scatter_add_ updates the base tensor in place."
  (apply-op 'scatter_add (list base index src) `((axis (int ,axis)))))
(define (concat xs axis)
  "The tensors XS concatenated along AXIS.

Args:
  xs: a list of tensors.
  axis: the axis to concatenate along; every other axis must match.
Returns: a tensor.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (concat (list (list->ctensor '(1.0 2.0))
                               (list->ctensor '(3.0 4.0)))
                         0))
  => (1.0 2.0 3.0 4.0)
PyTorch: torch.concat"
  (apply-op 'concat xs `((axis (int ,axis)))))
(define (slice x axis start end)
  "The elements of X along AXIS from START to END.

Args:
  x: a tensor.
  axis: the axis to slice.
  start: the first position (inclusive).
  end: the end position (exclusive).
Returns: a tensor; the sliced axis has length END - START.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor->list (slice (list->ctensor '(1.0 2.0 3.0 4.0)) 0 1 3))
  => (2.0 3.0)
PyTorch: none
Note: For torch.narrow, pass start and start + length."
  (apply-op 'slice (list x) `((axis (int ,axis)) (start (int ,start)) (end (int ,end)))))
(define (reshape x shape)
  "X with a new shape.

Args:
  x: a tensor.
  shape: the target shape; it must have the same number of elements.
Returns: a tensor.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-shape (reshape (make-ctensor 'float32 '(2 3) (iota 6)) '(3 2)))
  => (3 2)
PyTorch: torch.reshape"
  (apply-op 'reshape (list x) `((shape (ints ,@shape)))))
(define (expand x shape)
  "X broadcast to SHAPE.

Args:
  x: a tensor.
  shape: the target shape; every axis must match or be 1 in X.
Returns: a tensor.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-shape (expand (make-ctensor 'float32 '(1 2) '(1.0 2.0)) '(3 2)))
  => (3 2)
PyTorch: torch.Tensor.expand"
  (apply-op 'expand (list x) `((shape (ints ,@shape)))))

(define* (trace-print x #:optional (label "trace-print"))
  "Print X and return it.

In eager mode the value is printed immediately; while tracing the print is
recorded as an effectful node, so the value is printed every time the graph
runs.

Args:
  x: a tensor.
  label: optional; the text written before the values (default
    \"trace-print\").
Returns: X.
PyTorch: none"
  (apply-op 'print (list x) `((label (string ,label)))))

(define* (sum x #:key (axes '()) (keepdims #f))
  "The sum of X over AXES.

An empty AXES sums every axis.

Args:
  x: a tensor.
  axes: keyword; the axes to sum (default: every axis).
  keepdims: keyword; keep the reduced axes with size 1 (default #f).
Returns: a tensor with X's shape, or the reduced shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (ctensor-item (sum (list->ctensor '((1.0 2.0) (3.0 4.0)))))
  => 10.0
PyTorch: torch.sum"
  (apply-op 'reduce_sum (list x)
            `((axes (ints ,@axes)) (keepdims (int ,(if keepdims 1 0))))))

(define (bool x)
  "The truth value of a one-element tensor.

On a symbolic tensor the value is needed at trace time: unless the current path
already assumed it, tracing suspends and the tracer resumes with #t or #f.

Args:
  x: a one-element tensor, a Scheme number or a boolean.
Returns: a boolean.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace symbolic))
  (bool (list->ctensor '(0.0)))
  => #f
PyTorch: none"
  (cond ((boolean? x) x)
        ((number? x) (not (zero? x)))
        ((ctensor? x)
         (let ((v (ctensor-item x))) (if (boolean? v) v (not (zero? v)))))
        ((sym? x)
         (unless (= 1 (shape-numel (sym-shape x)))
           (error "bool: tensor has more than one element" (sym-shape x)))
         (let ((rec (current-tracer)))
           (unless rec (error "bool: symbolic tensor outside of tracing"))
           (let ((a (or (assoc (sym-name x) (%branch-assumptions))
                        (assoc (sym-name x) (recorder-assumptions rec)))))
             (if a (cdr a) (%abort-to-prompt %break-tag x)))))
        (else (error "bool: not a tensor" x))))

(for-each covered-procedure!
          (list add sub mul matmul relu neg sum gt lt where bool trace-print
                equal-to logical-and logical-or logical-not
                div transpose permute reshape expand texp tlog ttanh tsqrt gelu
                amax mean bmm index-select index-add gather scatter-add concat slice))
