;;; SonicBoom — tracing prototype: graph breaks, trace-tree cache, multi-resume
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; `trace-function` wraps a Scheme procedure written against the ops of
;;; (sonicboom trace symbolic). Calling the wrapper with concrete tensors:
;;;
;;;  1. traces the procedure on symbolic tensors inside a prompt;
;;;  2. when the procedure calls `bool` on a symbolic tensor it aborts to the
;;;     prompt with the condition; the handler receives the continuation and
;;;     the partial graph recorded so far and either
;;;       - multi-resumes: re-enters the continuation once with #t and once with
;;;         #f under speculative recorders and joins both results with a
;;;         `where` node (no graph break), or
;;;       - graph-breaks: runs the partial graph on the pluggable executor to
;;;         get the concrete condition, then re-enters the continuation with it
;;;         under a fresh prompt and keeps tracing the next segment;
;;;  3. stores the result as a trace tree (segments at the nodes, branch values
;;;     on the edges, the suspended continuation kept at each branch) keyed by
;;;     the call's input signature and the current values of its guards;
;;;  4. executes the tree segment by segment with the executor. A later call
;;;     replays cached segments and, on reaching a branch value that has not
;;;     been traced yet, either re-enters the stored continuation or (with
;;;     #:retrace-on-miss?) traces the procedure again from the start.
;;;
;;; Macro-rewritten `if`s ((sonicboom trace macro)) never reach step 2: they
;;; are joined in the graph directly. The continuation path is the fallback for
;;; `bool` calls the macro cannot see.
;;;
;;; Every segment emitted (partial graphs at breaks and the cached segments) is
;;; an S-Expr v0.1 document pruned by one dead-node-elimination pass from its
;;; outputs. While user code is being traced, current-output-port is a
;;; recording port; anything written there is reported as a trace-time side
;;; effect.
;;;
;;; Prompts, output capture, non-local exits, tables and procedure tags come
;;; from (sonicboom trace compat).

(define-module (sonicboom trace tracer)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom trace dce)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace interp)
  #:export (current-executor trace-function traced-function?
            trace-stats trace-segments trace-tree trace-warnings
            trace-strict-warnings
            trace->sexpr))

;;; --- executor -----------------------------------------------------------------

;; An executor is (text inputs) -> outputs, where TEXT is an S-Expr v0.1
;; document, INPUTS an alist (name . ctensor) and the result an alist
;; (output-name . ctensor). The default is the pure-Scheme reference
;; interpreter; a native backend would plug in here.
(define current-executor (make-parameter interpret-sexpr-text))

;;; --- records --------------------------------------------------------------------

;; A trace-tree node. NODES/PARAMS are the segment as recorded (topological,
;; before DCE). KEY is what the segment ends on: for a branch the condition's
;; value name; for a leaf the result template (the returned structure with a
;; symbolic tensor at each tensor position). K is the continuation suspended at the
;; branch (#f in retrace-on-miss mode); CHILDREN maps the condition's boolean
;; to the next segment.
(define-record-type <tnode>
  (make-tnode kind nodes params key k children compiled)
  tnode?
  (kind     tnode-kind)                       ; 'leaf | 'branch
  (nodes    tnode-nodes)
  (params   tnode-params)
  (key      tnode-key)
  (k        tnode-k)
  (children tnode-children set-tnode-children!)
  (compiled tnode-compiled set-tnode-compiled!))   ; <cseg>, set by analyze!

;; A compiled (DCE'd, emitted) segment.
(define-record-type <cseg>
  (make-cseg label inputs outputs text)
  cseg?
  (label   cseg-label)
  (inputs  cseg-inputs)
  (outputs cseg-outputs)
  (text    cseg-text))

;; One cached trace: the guard values it was traced under and its tree.
(define-record-type <entry>
  (make-entry guard-values root)
  entry?
  (guard-values entry-guard-values)            ; alist name -> snapshot
  (root         entry-root set-entry-root!))

;; Per traced function.
(define-record-type <tstate>
  (make-tstate proc ctx multi? max-depth executor retrace? guards
               trees env stats warnings)
  tstate?
  (proc      tstate-proc)
  (ctx       tstate-ctx)
  (multi?    tstate-multi?)
  (max-depth tstate-max-depth)
  (executor  tstate-executor)
  (retrace?  tstate-retrace?)
  (guards    tstate-guards)                   ; alist name -> thunk
  (trees     tstate-trees)                    ; table: signature -> (entry ...)
  (env       tstate-env set-tstate-env!)      ; table: name -> ctensor (per call)
  (stats     tstate-stats)                    ; table: counter -> integer
  (warnings  tstate-warnings set-tstate-warnings!))

(define (stat! st key)
  (table-set! (tstate-stats st) key (+ 1 (table-ref (tstate-stats st) key 0))))

;;; --- segments --------------------------------------------------------------------

(define (segment-defs nodes params)
  (append (append-map gnode-output-names nodes) (map gparam-name params)))

(define (segment-leaf-outputs st nodes params outputs)
  "OUTPUTS plus the value and .grad of every registered leaf that this segment
defines. A traced call reads those back: the gradient is written into the
caller's leaf, and a value the body replaced (an optimizer step's new
parameter/moment) replaces the caller's leaf value (see sync-leaf-state!)."
  (let* (;; Only values a node produces can be graph outputs: a leaf whose
         ;; gradient is a constant (a leaf the loss does not use) needs no
         ;; output, and marking the parameter as one would be rejected by the
         ;; verifier (a graph output must be produced by a node).
         (defs (append-map gnode-output-names nodes))
         (leaves (filter-map
                  (lambda (p)
                    (let ((v (leaf-value (cdr p))) (g (leaf-grad (cdr p))))
                      (append
                       ;; A leaf whose value is still its own input symbol was
                       ;; not updated by this segment.
                       (if (and (sym? v) (not (equal? (sym-name v) (car p)))
                                (member (sym-name v) defs))
                           (list (sym-name v)) '())
                       (if (and (sym? g) (member (sym-name g) defs))
                           (list (sym-name g)) '()))))
                  (trace-context-leaves (tstate-ctx st)))))
    (delete-duplicates (append outputs (apply append leaves)))))

(define (compile-segment st label nodes params outputs)
  "DCE NODES/PARAMS from OUTPUTS and emit the S-Expr v0.1 text. Inputs are the
values referenced but not defined by what survives. Leaf state this segment
defines is always an output, so a traced call observes it."
  (let ((outputs (segment-leaf-outputs st nodes params outputs)))
    (compile-segment-1 st label nodes params outputs)))

(define (compile-segment-1 st label nodes params outputs)
  (call-with-values (lambda () (eliminate-dead-nodes nodes params outputs))
    (lambda (nodes* params*)
      (let* ((defs (segment-defs nodes* params*))
             (refs (delete-duplicates
                    (append (append-map gnode-inputs nodes*) outputs)))
             (ins (remove (lambda (n) (member n defs)) refs))
             (types (trace-context-types (tstate-ctx st)))
             (typed (map (lambda (n)
                           (let ((t (or (table-ref types n #f)
                                        (error "trace: untyped value" n))))
                             (list n (car t) (cdr t))))
                         ins)))
        (make-cseg label ins outputs
                   (sexpr->text
                    (graph->sexpr label typed outputs params* nodes*)))))))

(define (truth t)
  (let ((v (ctensor-item t))) (if (boolean? v) v (not (zero? v)))))

(define (run-segment! st cs)
  "Execute compiled segment CS on the current call's environment and bind its
outputs there."
  (stat! st 'executor-calls)
  (let* ((env (tstate-env st))
         (exec (or (tstate-executor st) (current-executor)))
         (ins (map (lambda (n)
                     (cons n (or (table-ref env n #f)
                                 (error "trace: no concrete value for" n))))
                   (cseg-inputs cs))))
    (for-each (lambda (r) (table-set! env (car r) (cdr r)))
              (exec (cseg-text cs) ins))))

(define (run-segment-silently! st cs)
  ;; Executions that only rebuild concrete values (at a graph break, or after
  ;; a cache miss) must not repeat the effects (`print`) of the real run.
  (call-with-output-suppressed (lambda () (run-segment! st cs))))

(define (full-segment st n)
  "N's segment with every value it defines as an output."
  (compile-segment st "replay" (tnode-nodes n) (tnode-params n)
                   (delete-duplicates
                    (append (key-names n)
                            (append-map gnode-output-names (tnode-nodes n))))))

;;; --- recorders and prompts ----------------------------------------------------------

(define (segment-recorder st assumptions)
  (make-recorder (tstate-ctx st) '() '() 0 assumptions 0 #f))

(define (side-recorder rec cond-name value)
  "Speculative copy of REC continuing on the path where COND-NAME = VALUE."
  (make-recorder (recorder-context rec) (recorder-nodes rec)
                 (recorder-params rec) (recorder-count rec)
                 (acons cond-name value (recorder-assumptions rec))
                 (+ 1 (recorder-depth rec)) (recorder-count rec)))

(define (warn! st text)
  (stat! st 'output-warnings)
  (set-tstate-warnings! st (append (tstate-warnings st) (list text)))
  (trace-warn (string-append
         "sonicboom-trace: warning: output while tracing is a trace-time side "
         "effect and is not replayed by the cached graph (use trace-print): "
         (let ((p (open-output-string))) (write text p) (get-output-string p)))))

;; BODY returns (done <value>) when the traced procedure returns. The tracer
;; and output-port bindings are established outside the prompt, so a
;; continuation re-entered under another run-prompt records into that run's
;; recorder and port.
(define (run-prompt st rec body)
  (call-with-values
      (lambda ()
        (call-with-output-captured
         (lambda ()
           (parameterize ((current-tracer rec))
             (%call-with-prompt %break-tag
               body
               (lambda (k c) (list 'break k c)))))))
    (lambda (r out)
      (unless (string-null? out) (warn! st out))
      r)))

(define (leaf rec template)
  (make-tnode 'leaf (reverse (recorder-nodes rec)) (reverse (recorder-params rec))
              template #f '() #f))

;;; Results. A traced procedure returns a tensor or a structure (pairs,
;;; vectors) of tensors and other data; the leaf keeps it as a template.

(define (lift-result rec x)
  "Template for the returned value X: tensors and Scheme scalars at tensor
positions become symbolic values of the segment; pairs and vectors are kept;
other data (symbols, strings, '()) is static."
  (cond ((sym? x) x)
        ((or (ctensor? x) (number? x) (boolean? x)) (as-value rec x #f))
        ((pair? x) (cons (lift-result rec (car x)) (lift-result rec (cdr x))))
        ((vector? x) (list->vector (map (lambda (y) (lift-result rec y))
                                        (vector->list x))))
        (else x)))

(define (template-map f t)
  (cond ((sym? t) (f t))
        ((pair? t) (cons (template-map f (car t)) (template-map f (cdr t))))
        ((vector? t) (list->vector (map (lambda (y) (template-map f y))
                                        (vector->list t))))
        (else t)))

(define (template-names t)
  (cond ((sym? t) (list (sym-name t)))
        ((pair? t) (append (template-names (car t)) (template-names (cdr t))))
        ((vector? t) (append-map template-names (vector->list t)))
        (else '())))

(define (key-names n)
  "Value names segment N must output for its own sake."
  (if (eq? (tnode-kind n) 'leaf)
      (delete-duplicates (template-names (tnode-key n)))
      (list (tnode-key n))))

;;; --- multi-resume -------------------------------------------------------------------

(define (speculate st rec body)
  "Trace BODY to completion without graph breaks. Returns (recorder . output)
or #f if the side needs a break that cannot be merged or exceeds the node
budget."
  (%catch-signal 'sonicboom-trace-speculation-overflow
    (lambda ()
      (match (run-prompt st rec body)
        ;; only single-tensor results can be joined with where
        (('done out) (and (or (sym? out) (ctensor? out) (number? out) (boolean? out))
                          (cons rec (as-value rec out #f))))
        (('break k c) (try-merge st rec k c))))
    (lambda _ #f)))

(define (try-merge st rec k c)
  "Re-enter K with #t and with #f and join both outcomes with where(C, ...).
Returns (recorder . output) or #f."
  (and (eq? (sym-dtype c) 'bool)
       (< (recorder-depth rec) (tstate-max-depth st))
       (let* ((t (speculate st (side-recorder rec (sym-name c) #t)
                            (lambda () (k #t))))
              (f (and t (speculate st (side-recorder rec (sym-name c) #f)
                                   (lambda () (k #f))))))
         (and t f
              (let ((ot (cdr t)) (of (cdr f)))
                (and (eq? (sym-dtype ot) (sym-dtype of))
                     (equal? (sym-shape ot) (sym-shape of))
                     (%catch-signal 'sonicboom-trace-speculation-overflow
                       (lambda ()
                         (let ((m (join-sides rec (car t) (car f) c ot of)))
                           (stat! st 'merges)
                           m))
                       (lambda _ #f))))))))

(define (join-sides rec rt rf c ot of)
  ;; Both side recorders extend REC's lists, so their new entries are the
  ;; leading (count difference) elements. Order: prefix, #t side, #f side.
  (let* ((n0 (recorder-count rec))
         (new-t (take (recorder-nodes rt) (- (recorder-count rt) n0)))
         (new-f (take (recorder-nodes rf) (- (recorder-count rf) n0)))
         (p0 (length (recorder-params rec)))
         (par-t (take (recorder-params rt) (- (length (recorder-params rt)) p0)))
         (par-f (take (recorder-params rf) (- (length (recorder-params rf)) p0)))
         (m (make-recorder (recorder-context rec)
                           (append new-f new-t (recorder-nodes rec))
                           (append par-f par-t (recorder-params rec))
                           (+ n0 (length new-t) (length new-f))
                           (recorder-assumptions rec) (recorder-depth rec)
                           (recorder-spec-base rec))))
    (cons m (if (equal? (sym-name ot) (sym-name of))
                ot
                (record-node! m 'where (list c ot of) '())))))

;;; --- graph break ------------------------------------------------------------------------

(define (trace-segment st rec body)
  "Trace from BODY into recorder REC; returns the trace (sub)tree."
  (stat! st 'segments-traced)
  (match (run-prompt st rec body)
    (('done out) (leaf rec (lift-result rec out)))
    (('break k c)
     (let ((m (and (tstate-multi? st) (try-merge st rec k c))))
       (if m
           (leaf (car m) (cdr m))
           (graph-break st rec k c))))))

(define (graph-break st rec k c)
  ;; The partial graph's outputs are the condition plus every value it
  ;; defines: which of them the continuation will use is not known yet.
  (stat! st 'graph-breaks)
  (let* ((nodes (reverse (recorder-nodes rec)))
         (params (reverse (recorder-params rec)))
         (name (sym-name c))
         (cs (compile-segment
              st "partial" nodes params
              (delete-duplicates (cons name (append-map gnode-output-names nodes))))))
    (run-segment-silently! st cs)
    (let* ((v (truth (table-ref (tstate-env st) name)))
           ;; retrace-on-miss never re-enters K after this call, so does not
           ;; keep it (nor the stack segment it holds) alive in the cache.
           (node (make-tnode 'branch nodes params name
                             (and (not (tstate-retrace? st)) k) '() #f))
           (rec* (segment-recorder st (acons name v (recorder-assumptions rec)))))
      (set-tnode-children! node
                           (list (cons v (trace-segment st rec* (lambda () (k v))))))
      node)))

;;; --- trace tree analysis ----------------------------------------------------------------

(define (analyze! st node label)
  "Recompute every segment's outputs (its own key plus whatever later segments
read from it), DCE and emit it. Returns the names NODE's subtree needs from
earlier segments or the call's arguments."
  (let* ((child-needs
          (delete-duplicates
           (append-map (lambda (c)
                         (analyze! st (cdr c)
                                   (string-append label (if (car c) ".t" ".f"))))
                       (tnode-children node))))
         (defs (segment-defs (tnode-nodes node) (tnode-params node)))
         (req (delete-duplicates
               (append (key-names node)
                       (filter (lambda (n) (member n defs)) child-needs))))
         (cs (compile-segment st label (tnode-nodes node) (tnode-params node) req)))
    (set-tnode-compiled! node cs)
    (lset-union equal? (cseg-inputs cs) (lset-difference equal? child-needs defs))))

;;; --- calls ----------------------------------------------------------------------------------

;;; Arguments. A tensor argument becomes the graph input "arg<i>"; a list or
;;; vector argument is walked and each tensor in it becomes "arg<i>.<k>" (k in
;;; depth-first order), so parameter lists and optimizer state can be passed
;;; as structures. Everything else in an argument is static: part of the
;;; cache key and passed to the procedure as is.

(define (tensor-arg? x) (or (ctensor? x) (leaf? x)))

(define (arg-items pred args)
  "Alist (input-name . item) of every element of ARGS satisfying PRED, named
\"arg<i>\" at the top level and \"arg<i>.<k>\" inside list/vector arguments.
Tensors and leaves share one numbering, so a leaf's name matches the graph
input its value becomes."
  (append-map
   (lambda (a i)
     (let ((base (string-append "arg" (number->string i))))
       (if (tensor-arg? a)
           (if (pred a) (list (cons base a)) '())
           (let ((k 0) (acc '()))
             (let walk ((x a))
               (cond ((tensor-arg? x)
                      (when (pred x)
                        (set! acc (cons (cons (string-append base "." (number->string k)) x)
                                        acc)))
                      (set! k (+ k 1)))
                     ((pair? x) (walk (car x)) (walk (cdr x)))
                     ((vector? x) (for-each walk (vector->list x)))))
             (reverse acc)))))
   args (iota (length args))))

(define (arg-tensors args)
  "Alist (input-name . ctensor) of every tensor value in ARGS. A leaf
contributes its value: that is the graph input the trace substitutes."
  (map (lambda (p) (cons (car p) (unwrap-leaf (cdr p))))
       (arg-items tensor-arg? args)))

(define (arg-leaves args)
  "Alist (input-name . leaf) of every leaf tensor in ARGS, names as in
arg-tensors."
  (arg-items leaf? args))

(define* (map-tensors f args #:optional (keep-leaves? #f) (ctx #f))
  "ARGS with each tensor (or leaf value) replaced by (F name tensor), names as
in arg-tensors. With KEEP-LEAVES? a leaf argument stays a leaf wrapping the
result (the trace-time leaf), and is registered in CTX so backward! can store
gradients on it."
  (define (bind name x)
    (let ((r (f name (unwrap-leaf x))))
      (if (and keep-leaves? (leaf? x))
          (let ((l (make-leaf r)))
            (when ctx (register-leaf! ctx name l))
            l)
          r)))
  (map (lambda (a i)
         (let ((base (string-append "arg" (number->string i))))
           (if (tensor-arg? a)
               (bind base a)
               (let ((k 0))
                 (let walk ((x a))
                   (cond ((tensor-arg? x)
                          (let ((r (bind (string-append base "." (number->string k)) x)))
                            (set! k (+ k 1))
                            r))
                         ((pair? x) (let* ((h (walk (car x))) (t (walk (cdr x)))) (cons h t)))
                         ((vector? x) (list->vector (map walk (vector->list x))))
                         (else x)))))))
       args (iota (length args))))

(define (signature args)
  (map-tensors (lambda (name t) (list 'tensor (ctensor-dtype t) (ctensor-shape t))) args))

(define (reset-env! st args)
  (let ((env (make-table)))
    (for-each (lambda (p) (table-set! env (car p) (cdr p))) (arg-tensors args))
    (set-tstate-env! st env)))

(define (rebuild-env! st path args)
  "Recompute every concrete value defined along PATH (tree nodes, root first)."
  (reset-env! st args)
  (for-each (lambda (n) (run-segment-silently! st (full-segment st n))) path))

(define (path-assumptions st walk node v)
  (map (lambda (n)
         (cons (tnode-key n)
               (if (eq? n node) v (truth (table-ref (tstate-env st) (tnode-key n))))))
       walk))

(define (extend! st entry node v path args)
  "Cache miss at branch NODE (reached through PATH) for value V: rebuild every
concrete value on the path, then re-enter NODE's stored continuation with V.
Returns the new child."
  (stat! st 'traces)
  (let ((walk (append path (list node))))
    (rebuild-env! st walk args)
    (let ((child (trace-segment st (segment-recorder st (path-assumptions st walk node v))
                                (lambda () ((tnode-k node) v)))))
      (set-tnode-children! node (append (tnode-children node)
                                        (list (cons v child))))
      (analyze! st (entry-root entry) "trace")
      child)))

;;; Retrace-on-miss: trace the procedure again from the start on this call's
;;; inputs (no stored continuation is re-entered), then graft the new suffix
;;; into the cached tree. The fresh trace has fresh value names; its prefix is
;;; matched node-by-node against the cached path to build a renaming, so the
;;; grafted suffix refers to the cached prefix values.

(define (match-segments! renames old new)
  "Extend RENAMES (table new-name -> old-name) by matching tnode NEW against OLD
position by position. Returns #f on any structural difference."
  (define (rn x) (table-ref renames x x))
  (and (eq? (tnode-kind old) (tnode-kind new))
       (= (length (tnode-nodes old)) (length (tnode-nodes new)))
       (= (length (tnode-params old)) (length (tnode-params new)))
       (every (lambda (po pn)
                (and (eq? (gparam-dtype po) (gparam-dtype pn))
                     (equal? (gparam-shape po) (gparam-shape pn))
                     (equal? (gparam-values po) (gparam-values pn))
                     (begin (table-set! renames (gparam-name pn) (gparam-name po))
                            #t)))
              (tnode-params old) (tnode-params new))
       (every (lambda (no nn)
                (and (eq? (gnode-op no) (gnode-op nn))
                     (equal? (gnode-attrs no) (gnode-attrs nn))
                     (equal? (gnode-inputs no) (map rn (gnode-inputs nn)))
                     (= (length (gnode-outputs no)) (length (gnode-outputs nn)))
                     (begin
                       (for-each (lambda (o n) (table-set! renames (car n) (car o)))
                                 (gnode-outputs no) (gnode-outputs nn))
                       #t)))
              (tnode-nodes old) (tnode-nodes new))
       (equal? (key-names old) (map rn (key-names new)))))

(define (rename-tree renames n)
  (define (rn x) (table-ref renames x x))
  (make-tnode (tnode-kind n)
              (map (lambda (g)
                     (make-gnode (gnode-op g) (map rn (gnode-inputs g))
                                 (map (lambda (o) (cons (rn (car o)) (cdr o)))
                                      (gnode-outputs g))
                                 (gnode-attrs g)))
                   (tnode-nodes n))
              (map (lambda (p)
                     (make-gparam (rn (gparam-name p)) (gparam-dtype p)
                                  (gparam-shape p) (gparam-values p)))
                   (tnode-params n))
              (if (eq? (tnode-kind n) 'leaf)
                  (template-map (lambda (s) (make-sym (rn (sym-name s)) (sym-dtype s)
                                                      (sym-shape s)))
                                (tnode-key n))
                  (rn (tnode-key n)))
              #f
              (map (lambda (c) (cons (car c) (rename-tree renames (cdr c))))
                   (tnode-children n))
              #f))

(define (retrace! st entry node v path args)
  "Cache miss in retrace-on-miss mode. Returns the new child of NODE, or #f if
the fresh trace did not reproduce the cached prefix (the entry's tree is then
replaced by the fresh trace and the caller restarts from the root)."
  (stat! st 'traces)
  (stat! st 'retraces)
  (reset-env! st args)
  (let* ((fresh (fresh-trace st args))
         (walk (append path (list node)))
         (renames (make-table))
         (graft
          ;; Walk the cached path and the fresh chain together.
          (let loop ((olds walk) (new fresh))
            (let ((old (car olds)))
              (and (match-segments! renames old new)
                   (let* ((val (if (eq? old node) v
                                   (car (find (lambda (c) (memq (cdr c) walk))
                                              (tnode-children old)))))
                          (next (assq-ref (tnode-children new) val)))
                     (and next
                          (if (null? (cdr olds))
                              (rename-tree renames next)
                              (loop (cdr olds) next)))))))))
    (cond (graft
           (set-tnode-children! node (append (tnode-children node)
                                             (list (cons v graft))))
           (analyze! st (entry-root entry) "trace")
           (rebuild-env! st walk args)
           graft)
          (else
           (stat! st 'retrace-divergences)
           (trace-warn (string-append
                  "sonicboom-trace: warning: retrace did not reproduce the "
                  "cached prefix (unguarded state?); replacing the cached tree"))
           (set-entry-root! entry fresh)
           (analyze! st fresh "trace")
           #f))))

(define (replay st entry args)
  "Run the entry's tree on the call's arguments. Returns (values result
extended?)."
  (let restart ((extended? #f))
    (reset-env! st args)
    (let loop ((node (entry-root entry)) (path '()) (extended? extended?))
      (run-segment! st (tnode-compiled node))
      (if (eq? (tnode-kind node) 'leaf)
          (values (template-map (lambda (s) (table-ref (tstate-env st) (sym-name s)))
                                (tnode-key node))
                  extended?)
          (let ((v (table-ref (tstate-env st) (tnode-key node))))
            (let* ((b (truth v))
                   (child (assq-ref (tnode-children node) b)))
              (cond
               (child (loop child (append path (list node)) extended?))
               ;; The miss handlers rebuild the environment along the path
               ;; (silently), so the new child can run directly.
               ((if (tstate-retrace? st)
                    (retrace! st entry node b path args)
                    (extend! st entry node b path args))
                => (lambda (child) (loop child (append path (list node)) #t)))
               (else (restart #t)))))))))

(define (fresh-trace st args)
  (let ((ctx (tstate-ctx st)))
    ;; Leaves of a previous trace carry gradient names of that trace.
    (set-trace-context-leaves! ctx '())
    (let ((sym-args (map-tensors (lambda (name t)
                                   (make-sym name (ctensor-dtype t) (ctensor-shape t)))
                                 args #t ctx)))
      (trace-segment st (segment-recorder st '())
                     (lambda () (list 'done (apply (tstate-proc st) sym-args)))))))

;;; Guards: the current values of the procedure's free variables are part of
;;; the cache key. Values are snapshotted (pairs and vectors copied) when a
;;; trace is stored and compared structurally; procedures, records and other
;;; objects compare by identity.

(define (snapshot x)
  (cond ((pair? x) (cons (snapshot (car x)) (snapshot (cdr x))))
        ((vector? x) (list->vector (map snapshot (vector->list x))))
        ((string? x) (string-copy x))
        (else x)))

(define (guard-same? a b)
  (cond ((and (pair? a) (pair? b))
         (and (guard-same? (car a) (car b)) (guard-same? (cdr a) (cdr b))))
        ((and (vector? a) (vector? b))
         (and (= (vector-length a) (vector-length b))
              (every guard-same? (vector->list a) (vector->list b))))
        ((and (string? a) (string? b)) (string=? a b))
        ((and (number? a) (number? b)) (eqv? a b))
        (else (eq? a b))))

(define (guard-values st)
  (map (lambda (g) (cons (car g) ((cdr g)))) (tstate-guards st)))

(define (find-entry entries values)
  (find (lambda (e)
          (every (lambda (old new) (guard-same? (cdr old) (cdr new)))
                 (entry-guard-values e) values))
        entries))

(define (traced-call st args)
  (stat! st 'calls)
  (let* ((ctx (tstate-ctx st))
         (sig (signature args))
         (entries (table-ref (tstate-trees st) sig '()))
         (gvals (guard-values st))
         (entry (find-entry entries gvals)))
    (for-each (lambda (p)
                (register-type! ctx (car p) (ctensor-dtype (cdr p)) (ctensor-shape (cdr p))))
              (arg-tensors args))
    (let ((fresh? (not entry)))
      (when fresh?
        (stat! st 'traces)
        (unless (null? entries) (stat! st 'guard-misses))
        (reset-env! st args)
        (let ((root (fresh-trace st args)))
          (set! entry (make-entry (map (lambda (g) (cons (car g) (snapshot (cdr g))))
                                       gvals)
                                  root))
          (table-set! (tstate-trees st) sig (append entries (list entry)))
          (analyze! st root "trace")))
      (call-with-values (lambda () (replay st entry args))
        (lambda (result extended?)
          (unless (or fresh? extended?) (stat! st 'cache-hits))
          (sync-leaf-state! st args)
          result)))))

(define (sync-leaf-state! st args)
  "Copy each caller leaf's concrete state out of the environment the executed
graph left behind. The trace-time leaf holds the symbolic adjoint backward!
recorded and, when the body updated it (an optimizer step), the symbolic new
value; compile-segment made both graph outputs, so replay bound them here. A
gradient is written to the caller's .grad; an updated value replaces the
caller's .value, unless the trace-time value is still the leaf's own input."
  (let ((ctx (tstate-ctx st))
        (env (tstate-env st)))
    (for-each
     (lambda (p)
       (let ((trace-leaf (assoc-ref (trace-context-leaves ctx) (car p))))
         (when trace-leaf
           (let ((v (leaf-value trace-leaf))
                 (g (leaf-grad trace-leaf)))
             (when (and (sym? v) (not (equal? (sym-name v) (car p))))
               (let ((new (table-ref env (sym-name v) #f)))
                 (when new (set-leaf-value! (cdr p) new))))
             (when (sym? g)
               (let ((new (table-ref env (sym-name g) #f)))
                 (when new (set-leaf-grad! (cdr p) new))))))))
     (arg-leaves args))))

;;; --- public API ------------------------------------------------------------------------------

(define* (trace-function proc #:key (multi-resume? #f) (max-side-nodes 16)
                         (max-depth 3) (executor #f) (retrace-on-miss? #f)
                         (guards '()) (strict? 'default) (max-unroll 64))
  "Return a procedure that runs PROC through the tracer. Tensor arguments are
ctensors (they become symbolic inputs \"arg<i>\"); other arguments are static
and part of the cache key. GUARDS is an alist (name . thunk): the thunks'
values at call time are also part of the cache key (define-traced /
traced-lambda fill this with the body's free variables). MULTI-RESUME?
asserts that PROC has no side effects after its `bool` points, allowing a
branch to be resolved inside the graph when each side records at most
MAX-SIDE-NODES nodes and branches nest at most MAX-DEPTH deep.
RETRACE-ON-MISS? traces PROC again from the start on a cache miss instead of
re-entering a stored continuation (and stores none). STRICT? reports
symbolic tensors passed to procedures the macro walker did not cover
('default follows the strict-tracing parameter). MAX-UNROLL bounds how many
iterations of a do / named-let loop are unrolled into the graph; a longer loop
becomes a host loop over one cached body graph (#f: always unroll). EXECUTOR
overrides `current-executor`."
  (let* ((st (make-tstate proc (make-trace-context max-side-nodes strict? max-unroll)
                          multi-resume? max-depth executor retrace-on-miss? guards
                          (make-table) #f (make-table) '()))
         (wrapper (lambda args
                    (if (or (any sym? args) (tracing?))
                        (apply proc args)    ; called while tracing: inline
                        (traced-call st args)))))
    (procedure-tag! wrapper 'sonicboom-trace st)
    wrapper))

(define (traced-function? f)
  (tstate? (procedure-tag f 'sonicboom-trace)))

(define (state-of f)
  (if (traced-function? f)
      (procedure-tag f 'sonicboom-trace)
      (error "not a traced function" f)))

(define (trace-stats f)
  "Counters of traced function F: calls, cache-hits (calls that traced
nothing), traces (fresh traces + cache-miss re-entries or retraces),
guard-misses (fresh traces caused by a changed guard value), retraces,
retrace-divergences, segments-traced, graph-breaks, merges (continuation
multi-resume joins), branch-merges (macro-rewritten ifs joined with where),
logic-merges (and/or on symbolic conditions), loop-breaks (symbolic branches
re-entered recursively, i.e. data-dependent loops, resolved by a graph
break), output-warnings, strict-warnings, executor-calls."
  (let* ((st (state-of f))
         (h (tstate-stats st))
         (c (trace-context-counters (tstate-ctx st))))
    (map (lambda (k)
           (cons k (if (eq? k 'strict-warnings)
                       (length (trace-context-notes (tstate-ctx st)))
                       (table-ref h k (table-ref c k 0)))))
         '(calls cache-hits traces guard-misses retraces retrace-divergences
           segments-traced graph-breaks merges branch-merges logic-merges
           loop-breaks host-loops loop-body-traces loop-body-hits
           output-warnings strict-warnings executor-calls))))

(define (trace-strict-warnings f)
  "Strict-mode reports of F: one string per procedure that received a
symbolic tensor without being covered by the macro walker."
  (trace-context-notes (tstate-ctx (state-of f))))

(define (trace-warnings f)
  "Text written to current-output-port while F was being traced, one string
per tracing run that wrote something."
  (tstate-warnings (state-of f)))

(define (all-trees f)
  (append-map (lambda (p)
                (map (lambda (e) (cons (cons (car p) (entry-guard-values e))
                                       (entry-root e)))
                     (cdr p)))
              (table->alist (tstate-trees (state-of f)))))

(define (trace-segments f)
  "List of (label . S-Expr text) for every cached segment of F, depth first.
Labels are paths: \"trace\", \"trace.t\", \"trace.t.f\", ..."
  (append-map
   (lambda (entry)
     (let walk ((n (cdr entry)))
       (cons (cons (cseg-label (tnode-compiled n)) (cseg-text (tnode-compiled n)))
             (append-map (lambda (c) (walk (cdr c))) (tnode-children n)))))
   (all-trees f)))

(define (data-guards guard-values)
  ;; Procedure-valued guards (compared by identity) and unbound names are part
  ;; of the key but would only clutter the view.
  (filter (lambda (g) (not (or (procedure? (cdr g))
                               (eq? (cdr g) %unbound-guard))))
          guard-values))

(define (trace-tree f)
  "The shape of F's trace trees: ((signature . data-guards) . tree), where
data-guards are the guard values other than procedures and unbound names, and
tree is (leaf <output>) or (branch <condition> (#t . tree) (#f . tree))."
  (map (lambda (entry)
         (cons (cons (caar entry) (data-guards (cdar entry)))
               (let walk ((n (cdr entry)))
                 (if (eq? (tnode-kind n) 'leaf)
                     (list 'leaf (template-map sym-name (tnode-key n)))
                     (cons* 'branch (tnode-key n)
                            (map (lambda (c) (cons (car c) (walk (cdr c))))
                                 (tnode-children n)))))))
       (all-trees f)))

(define (trace->sexpr proc . args)
  "Trace PROC once on ARGS (with multi-resume enabled) and return the S-Expr
v0.1 text of the single resulting graph. Errors if a graph break remained."
  (let ((f (trace-function proc #:multi-resume? #t)))
    (apply f args)
    (match (trace-segments f)
      (((label . text)) text)
      (_ (error "trace->sexpr: PROC needs a graph break")))))
