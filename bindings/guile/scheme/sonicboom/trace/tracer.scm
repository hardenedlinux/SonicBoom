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
;;;     the call's input signature;
;;;  4. executes the tree segment by segment with the executor. A later call
;;;     replays cached segments and only re-enters a stored continuation when
;;;     it reaches a branch value that has not been traced yet.
;;;
;;; Every segment emitted (partial graphs at breaks and the cached segments) is
;;; an S-Expr v0.1 document pruned by one dead-node-elimination pass from its
;;; outputs.

(define-module (sonicboom trace tracer)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom trace dce)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace interp)
  #:export (current-executor trace-function traced-function?
            trace-stats trace-segments trace-tree trace->sexpr))

;;; --- executor -----------------------------------------------------------------

;; An executor is (text inputs) -> outputs, where TEXT is an S-Expr v0.1
;; document, INPUTS an alist (name . ctensor) and the result an alist
;; (output-name . ctensor). The default is the pure-Scheme reference
;; interpreter; a native backend would plug in here.
(define current-executor (make-parameter interpret-sexpr-text))

;;; --- records --------------------------------------------------------------------

;; A trace-tree node. NODES/PARAMS are the segment as recorded (topological,
;; before DCE). KEY is the value the segment ends on: the returned value for a
;; leaf, the condition for a branch. K is the continuation suspended at the
;; branch; CHILDREN maps the condition's boolean to the next segment.
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

;; Per traced function.
(define-record-type <tstate>
  (make-tstate proc ctx multi? max-depth executor trees env stats)
  tstate?
  (proc      tstate-proc)
  (ctx       tstate-ctx)
  (multi?    tstate-multi?)
  (max-depth tstate-max-depth)
  (executor  tstate-executor)
  (trees     tstate-trees)                    ; hash: signature -> root tnode
  (env       tstate-env set-tstate-env!)      ; hash: name -> ctensor (per call)
  (stats     tstate-stats))                   ; hash: counter -> integer

(define (stat! st key)
  (hash-set! (tstate-stats st) key (+ 1 (hash-ref (tstate-stats st) key 0))))

;;; --- segments --------------------------------------------------------------------

(define (segment-defs nodes params)
  (append (append-map gnode-output-names nodes) (map gparam-name params)))

(define (compile-segment st label nodes params outputs)
  "DCE NODES/PARAMS from OUTPUTS and emit the S-Expr v0.1 text. Inputs are the
values referenced but not defined by what survives."
  (call-with-values (lambda () (eliminate-dead-nodes nodes params outputs))
    (lambda (nodes* params*)
      (let* ((defs (segment-defs nodes* params*))
             (refs (delete-duplicates
                    (append (append-map gnode-inputs nodes*) outputs)))
             (ins (remove (lambda (n) (member n defs)) refs))
             (types (trace-context-types (tstate-ctx st)))
             (typed (map (lambda (n)
                           (let ((t (or (hash-ref types n #f)
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
                     (cons n (or (hash-ref env n #f)
                                 (error "trace: no concrete value for" n))))
                   (cseg-inputs cs))))
    (for-each (lambda (r) (hash-set! env (car r) (cdr r)))
              (exec (cseg-text cs) ins))))

;;; --- recorders and prompts ----------------------------------------------------------

(define (segment-recorder st assumptions)
  (make-recorder (tstate-ctx st) '() '() 0 assumptions 0 #f))

(define (side-recorder rec cond-name value)
  "Speculative copy of REC continuing on the path where COND-NAME = VALUE."
  (make-recorder (recorder-context rec) (recorder-nodes rec)
                 (recorder-params rec) (recorder-count rec)
                 (acons cond-name value (recorder-assumptions rec))
                 (+ 1 (recorder-depth rec)) (recorder-count rec)))

;; BODY returns (done <value>) when the traced procedure returns. The tracer
;; binding is established outside the prompt, so a continuation re-entered
;; under another run-prompt records into that run's recorder.
(define (run-prompt rec body)
  (parameterize ((current-tracer rec))
    (call-with-prompt %break-tag
      body
      (lambda (k c) (list 'break k c)))))

(define (leaf rec out)
  (make-tnode 'leaf (reverse (recorder-nodes rec)) (reverse (recorder-params rec))
              (sym-name out) #f '() #f))

;;; --- multi-resume -------------------------------------------------------------------

(define (speculate st rec body)
  "Trace BODY to completion without graph breaks. Returns (recorder . output)
or #f if the side needs a break that cannot be merged or exceeds the node
budget."
  (catch 'sonicboom-trace-speculation-overflow
    (lambda ()
      (match (run-prompt rec body)
        (('done out) (cons rec (as-value rec out #f)))
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
                     (catch 'sonicboom-trace-speculation-overflow
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
  (match (run-prompt rec body)
    (('done out) (leaf rec (as-value rec out #f)))
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
    (run-segment! st cs)
    (let* ((v (truth (hash-ref (tstate-env st) name)))
           (node (make-tnode 'branch nodes params name k '() #f))
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
               (cons (tnode-key node)
                     (filter (lambda (n) (member n defs)) child-needs))))
         (cs (compile-segment st label (tnode-nodes node) (tnode-params node) req)))
    (set-tnode-compiled! node cs)
    (lset-union equal? (cseg-inputs cs) (lset-difference equal? child-needs defs))))

;;; --- calls ----------------------------------------------------------------------------------

(define (arg-name i) (string-append "arg" (number->string i)))

(define (signature args)
  (map (lambda (a)
         (if (ctensor? a)
             (list 'tensor (ctensor-dtype a) (ctensor-shape a))
             (list 'static a)))
       args))

(define (reset-env! st args)
  (let ((env (make-hash-table)))
    (for-each (lambda (a i) (when (ctensor? a) (hash-set! env (arg-name i) a)))
              args (iota (length args)))
    (set-tstate-env! st env)))

(define (extend! st root node v path args)
  "Cache miss at branch NODE (reached through PATH) for value V: rebuild every
concrete value on the path, then re-enter NODE's stored continuation with V."
  (stat! st 'traces)
  (reset-env! st args)
  (let ((walk (append path (list node))))
    (for-each (lambda (n)
                (run-segment!
                 st (compile-segment
                     st "replay" (tnode-nodes n) (tnode-params n)
                     (delete-duplicates
                      (cons (tnode-key n)
                            (append-map gnode-output-names (tnode-nodes n)))))))
              walk)
    (let* ((assumptions (map (lambda (n)
                               (cons (tnode-key n)
                                     (if (eq? n node) v
                                         (truth (hash-ref (tstate-env st)
                                                          (tnode-key n))))))
                             walk))
           (child (trace-segment st (segment-recorder st assumptions)
                                 (lambda () ((tnode-k node) v)))))
      (set-tnode-children! node (append (tnode-children node)
                                        (list (cons v child))))
      (analyze! st root "trace")
      child)))

(define (replay st root args)
  "Run the tree on the call's arguments. Returns (values result extended?)."
  (let loop ((node root) (path '()) (extended? #f))
    (run-segment! st (tnode-compiled node))
    (let ((v (hash-ref (tstate-env st) (tnode-key node))))
      (if (eq? (tnode-kind node) 'leaf)
          (values v extended?)
          (let* ((b (truth v))
                 (child (assq-ref (tnode-children node) b)))
            (if child
                (loop child (append path (list node)) extended?)
                (let ((child (extend! st root node b path args)))
                  ;; extend! rebuilt the environment along the path, so the
                  ;; new child can run directly.
                  (loop child (append path (list node)) #t))))))))

(define (traced-call st args)
  (stat! st 'calls)
  (let* ((ctx (tstate-ctx st))
         (sig (signature args))
         (root (hash-ref (tstate-trees st) sig)))
    (for-each (lambda (a i)
                (when (ctensor? a)
                  (register-type! ctx (arg-name i) (ctensor-dtype a)
                                  (ctensor-shape a))))
              args (iota (length args)))
    (reset-env! st args)
    (let ((fresh? (not root)))
      (when fresh?
        (stat! st 'traces)
        (let ((sym-args (map (lambda (a i)
                               (if (ctensor? a)
                                   (make-sym (arg-name i) (ctensor-dtype a)
                                             (ctensor-shape a))
                                   a))
                             args (iota (length args)))))
          (set! root (trace-segment
                      st (segment-recorder st '())
                      (lambda () (list 'done (apply (tstate-proc st) sym-args))))))
        (hash-set! (tstate-trees st) sig root)
        (analyze! st root "trace")
        (reset-env! st args))
      (call-with-values (lambda () (replay st root args))
        (lambda (result extended?)
          (unless (or fresh? extended?) (stat! st 'cache-hits))
          result)))))

;;; --- public API ------------------------------------------------------------------------------

(define* (trace-function proc #:key (multi-resume? #f) (max-side-nodes 16)
                         (max-depth 3) (executor #f))
  "Return a procedure that runs PROC through the tracer. Tensor arguments are
ctensors (they become symbolic inputs \"arg<i>\"); other arguments are static
and part of the cache key. MULTI-RESUME? asserts that PROC has no side effects
after its `bool` points, allowing a branch to be resolved inside the graph
when each side records at most MAX-SIDE-NODES nodes and branches nest at most
MAX-DEPTH deep. EXECUTOR overrides `current-executor`."
  (let* ((st (make-tstate proc (make-trace-context max-side-nodes) multi-resume?
                          max-depth executor (make-hash-table) #f
                          (make-hash-table)))
         (wrapper (lambda args
                    (if (any sym? args)
                        (apply proc args)    ; called while tracing: inline
                        (traced-call st args)))))
    (set-procedure-property! wrapper 'sonicboom-trace st)
    wrapper))

(define (traced-function? f)
  (and (procedure? f) (tstate? (procedure-property f 'sonicboom-trace))))

(define (state-of f)
  (if (traced-function? f)
      (procedure-property f 'sonicboom-trace)
      (error "not a traced function" f)))

(define (trace-stats f)
  "Counters of traced function F: calls, cache-hits (calls that traced
nothing), traces (fresh traces + continuation re-entries on a miss),
segments-traced, graph-breaks, merges, executor-calls."
  (let ((h (tstate-stats (state-of f))))
    (map (lambda (k) (cons k (hash-ref h k 0)))
         '(calls cache-hits traces segments-traced graph-breaks merges
           executor-calls))))

(define (all-trees f)
  (hash-map->list (lambda (sig root) (cons sig root)) (tstate-trees (state-of f))))

(define (trace-segments f)
  "List of (label . S-Expr text) for every cached segment of F, depth first.
Labels are paths: \"trace\", \"trace.t\", \"trace.t.f\", ..."
  (append-map
   (lambda (entry)
     (let walk ((n (cdr entry)))
       (cons (cons (cseg-label (tnode-compiled n)) (cseg-text (tnode-compiled n)))
             (append-map (lambda (c) (walk (cdr c))) (tnode-children n)))))
   (all-trees f)))

(define (trace-tree f)
  "The shape of F's trace trees: (signature . tree), where tree is
(leaf <output>) or (branch <condition> (#t . tree) (#f . tree))."
  (map (lambda (entry)
         (cons (car entry)
               (let walk ((n (cdr entry)))
                 (if (eq? (tnode-kind n) 'leaf)
                     (list 'leaf (tnode-key n))
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
