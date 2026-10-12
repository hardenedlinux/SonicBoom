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
            strict-tracing covered-procedure! covered-procedure?
            ;; tracing context / recorder (used by the tracer)
            make-trace-context trace-context? trace-context-types
            trace-context-counters context-count!
            trace-context-strict trace-context-notes add-context-note!
            fresh-name! register-type!
            make-recorder recorder? recorder-context recorder-nodes
            recorder-params recorder-count recorder-assumptions
            recorder-depth recorder-spec-base
            set-recorder-nodes! set-recorder-params! set-recorder-count!
            as-value record-node!
            ;; ops
            add sub mul matmul relu neg sum gt lt where bool trace-print
            equal-to logical-and logical-or logical-not))

;;; --- symbolic tensor ------------------------------------------------------------

(define-record-type <sym>
  (make-sym name dtype shape)
  sym?
  (name  sym-name)
  (dtype sym-dtype)
  (shape sym-shape))

(define (shape-of x)
  (cond ((sym? x) (sym-shape x))
        ((ctensor? x) (ctensor-shape x))
        ((or (number? x) (boolean? x)) '())
        (else (error "shape-of: not a tensor" x))))

(define (dtype-of x)
  (cond ((sym? x) (sym-dtype x))
        ((ctensor? x) (ctensor-dtype x))
        (else #f)))

;;; --- mode -------------------------------------------------------------------------

;; #f in eager mode; the active recorder while tracing.
(define current-tracer (make-parameter #f))
(define (tracing?) (and (current-tracer) #t))

;; Prompt tag used by `bool` to suspend tracing.
(define %break-tag (%make-prompt-tag 'sonicboom-trace-break))

;; Default for traced functions created without #:strict?: when true, passing
;; a symbolic tensor to a procedure the macro walker did not cover is
;; reported (see %call in (sonicboom trace macro)).
(define strict-tracing (make-parameter #f))

;; Procedures known not to hide a core `if` on a symbolic tensor: the tensor
;; ops below, procedures created inside a define-traced body, traced
;; functions (checked separately by the tracer).
(define (covered-procedure! proc)
  (procedure-tag! proc 'sonicboom-trace-covered #t)
  proc)
(define (covered-procedure? proc)
  (and (procedure-tag proc 'sonicboom-trace-covered) #t))

;; Condition values fixed by an enclosing macro-rewritten `if` (%branch) while
;; it traces one of its arms: alist name -> boolean. A parameter rather than
;; recorder state, so that a continuation captured inside an arm restores it.
(define %branch-assumptions (make-parameter '()))

;;; --- tracing context ------------------------------------------------------------

;; Shared by every recorder of one traced function: a name counter (so values
;; recorded on different resumptions of a continuation never collide), the
;; type of every value name ever introduced, and the per-side node budget used
;; by speculative (multi-resume) tracing.
;; STRICT is #t, #f or 'default (follow the strict-tracing parameter); NOTES
;; collects strict-mode reports.
(define-record-type <trace-context>
  (%make-trace-context next-id types max-side-nodes counters strict notes)
  trace-context?
  (next-id trace-context-next-id set-trace-context-next-id!)
  (types trace-context-types)               ; table: name -> (dtype . shape)
  (max-side-nodes trace-context-max-side-nodes)
  (counters trace-context-counters)         ; table: event -> count
  (strict trace-context-strict)
  (notes trace-context-notes set-trace-context-notes!))

(define* (make-trace-context max-side-nodes #:optional (strict 'default))
  (%make-trace-context 0 (make-table) max-side-nodes (make-table) strict '()))

(define (context-count! ctx key)
  (let ((h (trace-context-counters ctx)))
    (table-set! h key (+ 1 (table-ref h key 0)))))

(define (add-context-note! ctx note)
  (set-trace-context-notes! ctx (append (trace-context-notes ctx) (list note))))

(define (fresh-name! ctx prefix)
  (let ((n (trace-context-next-id ctx)))
    (set-trace-context-next-id! ctx (+ n 1))
    (string-append prefix (number->string n))))

(define (register-type! ctx name dtype shape)
  (table-set! (trace-context-types ctx) name (cons dtype shape)))

;; A recorder is the state of one segment being traced. NODES and PARAMS are
;; kept in reverse order as shared-tail lists, so copying a recorder for a
;; speculative resumption is O(1). ASSUMPTIONS maps condition value names to
;; the boolean the current path took. SPEC-BASE is #f outside speculation,
;; otherwise the node count when the speculative side started.
(define-record-type <recorder>
  (make-recorder context nodes params count assumptions depth spec-base)
  recorder?
  (context     recorder-context)
  (nodes       recorder-nodes set-recorder-nodes!)
  (params      recorder-params set-recorder-params!)
  (count       recorder-count set-recorder-count!)
  (assumptions recorder-assumptions)
  (depth       recorder-depth)
  (spec-base   recorder-spec-base))

(define (lift x dtype)
  "Scheme number / boolean / ctensor -> ctensor."
  (cond ((ctensor? x) x)
        ((boolean? x) (scalar->ctensor x 'bool))
        ((number? x) (scalar->ctensor x (or dtype 'float32)))
        (else (error "not a tensor operand" x))))

(define (as-value rec x dtype)
  "Symbolic value for X in recorder REC: symbolic tensors pass through;
concrete operands become an inline parameter of the current segment."
  (if (sym? x)
      x
      (let* ((t (lift x dtype))
             (ctx (recorder-context rec))
             (name (fresh-name! ctx "c")))
        (register-type! ctx name (ctensor-dtype t) (ctensor-shape t))
        (set-recorder-params!
         rec (cons (make-gparam name (ctensor-dtype t) (ctensor-shape t)
                                (ctensor->list t))
                   (recorder-params rec)))
        (make-sym name (ctensor-dtype t) (ctensor-shape t)))))

(define (record-node! rec op ins attrs)
  "Append node OP(INS) to REC and return its symbolic output."
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

;;; --- op front end -------------------------------------------------------------

;; dtype used to lift bare Scheme numbers: that of the first non-bool tensor
;; operand (so `(add x 1)` on an int64 tensor stays int64), else float32.
(define (ref-dtype args)
  (or (find (lambda (d) (and d (not (eq? d 'bool)))) (map dtype-of args))
      'float32))

(define (apply-op op args attrs)
  (let ((rec (current-tracer))
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

(define (add a b) (apply-op 'add (list a b) '()))
(define (sub a b) (apply-op 'sub (list a b) '()))
(define (mul a b) (apply-op 'mul (list a b) '()))
(define (matmul a b) (apply-op 'matmul (list a b) '()))
(define (relu x) (apply-op 'relu (list x) '()))
(define (neg x) (apply-op 'neg (list x) '()))
(define (gt a b) (apply-op 'greater (list a b) '()))
(define (lt a b) (apply-op 'less (list a b) '()))
(define (where c x y) (apply-op 'where (list c x y) '()))
(define (equal-to a b) (apply-op 'equal (list a b) '()))
(define (logical-and a b) (apply-op 'and (list a b) '()))
(define (logical-or a b) (apply-op 'or (list a b) '()))
(define (logical-not x) (apply-op 'not (list x) '()))

(define* (trace-print x #:optional (label "trace-print"))
  "Print X (as LABEL: nested values) to current-output-port and return it. In
eager mode this prints immediately; while tracing it records an effectful
`print` node, so the value is printed every time the graph runs, not when it
is traced. DCE never removes it."
  (apply-op 'print (list x) `((label (string ,label)))))

(define* (sum x #:key (axes '()) (keepdims #f))
  "Sum over AXES (all axes when empty). Recorded as reduce_sum."
  (apply-op 'reduce_sum (list x)
            `((axes (ints ,@axes)) (keepdims (int ,(if keepdims 1 0))))))

(define (bool x)
  "Truth value of a one-element tensor. On a symbolic tensor this needs a
concrete value: unless the current path already assumed one, suspend tracing
by aborting to %break-tag with X; the tracer resumes with #t or #f."
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
                equal-to logical-and logical-or logical-not))
