;;; SonicBoom — tracing prototype: loops (unroll, or host loop over one graph)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; `do` and named `let` in a traced body expand (via the macro walker) into
;;; (%loop site (lambda (self v ...) body) init ...): BODY runs one iteration
;;; and continues by calling SELF.
;;;
;;; - Eager mode: plain named-let recursion.
;;; - Tracing: the loop is unrolled into the graph, one iteration after the
;;;   other. If an instance runs more than #:max-unroll iterations (default 64)
;;;   while its variables carry tensors, the unrolled part is dropped and the
;;;   loop becomes a HOST LOOP: each iteration's body is traced once more on
;;;   fresh symbolic loop variables into its own small graph; consecutive
;;;   iterations with the same graph are run-length encoded; each run becomes
;;;   one `loop` node whose `body` attribute is the S-Expr text of that graph,
;;;   executed `trip_count` times by the executor. The exit iteration is then
;;;   traced into the main graph on the loop node's outputs.
;;;
;;;   Host loops need the loop to continue only by a tail call to SELF, the
;;;   tensor loop variables to keep their dtype and shape, and the body not to
;;;   take a graph break (`bool` on a symbolic tensor).
;;;
;;; A loop whose exit test is symbolic never gets here: %branch detects it as
;;; a data-dependent loop and resolves the test with a graph break per
;;; iteration (see (sonicboom trace macro)).

(define-module (sonicboom trace loop)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom trace dce)
  #:use-module (sonicboom trace symbolic)
  #:export (%loop))

(define (tensor? x) (or (sym? x) (ctensor? x)))
(define (type-of x) (cons (dtype-of x) (shape-of x)))

;;; --- eager -------------------------------------------------------------------------

(define (plain-loop body inits)
  (letrec ((self (covered-procedure! (lambda args (apply body self args)))))
    (apply self inits)))

;;; --- unrolled, with a limit ------------------------------------------------------

(define (%loop site body . inits)
  (let ((rec (current-tracer)))
    (if (not rec)
        (plain-loop body inits)
        (unrolled-loop site rec body inits))))

(define (unrolled-loop site rec body inits)
  (let ((limit (trace-context-max-unroll (recorder-context rec)))
        (count 0)
        (instance (list 'loop))
        (nodes (recorder-nodes rec))
        (params (recorder-params rec))
        (n (recorder-count rec)))
    (%catch-signal 'sonicboom-trace-unroll-limit
      (lambda ()
        (letrec ((self (covered-procedure!
                        (lambda args
                          (set! count (+ count 1))
                          (when (and limit (> count limit)
                                     (eq? (current-tracer) rec)
                                     (any tensor? args))
                            (%raise-signal 'sonicboom-trace-unroll-limit instance))
                          (apply body self args)))))
          (apply self inits)))
      (lambda (which)
        (cond ((eq? which instance)
               ;; drop the unrolled iterations, run it as a host loop
               (set-recorder-nodes! rec nodes)
               (set-recorder-params! rec params)
               (set-recorder-count! rec n)
               (host-loop site rec body inits))
              (else (%raise-signal 'sonicboom-trace-unroll-limit which)))))))

;;; --- host loop ---------------------------------------------------------------------

(define-record-type <continue>
  (make-continue args)
  continue?
  (args continue-args))

(define (loop-error site fmt . args)
  (error (apply format #f (string-append "traced loop at ~s: " fmt)
                (cons site args))))

;; One traced iteration: (values kind data) where kind is 'continue with
;; data = (text externals next-static next-types) or 'exit.
(define (trace-iteration site rec body args)
  (let* ((ctx (recorder-context rec))
         (scratch (make-recorder ctx '() '() 0 (recorder-assumptions rec) 0 #f))
         (holders (map (lambda (a)
                         (and (tensor? a)
                              (let ((name (fresh-name! ctx "v")))
                                (register-type! ctx name (dtype-of a) (shape-of a))
                                (make-sym name (dtype-of a) (shape-of a)))))
                       args))
         (call-args (map (lambda (a h) (or h a)) args holders))
         (token #f)
         (self (covered-procedure!
                (lambda next (set! token (make-continue next)) token)))
         (r (parameterize ((current-tracer scratch))
              (%call-with-prompt %break-tag
                (lambda () (apply body self call-args))
                (lambda (k c)
                  (loop-error site "graph break (bool on a symbolic tensor) inside a host loop body is not supported"))))))
    (cond
     ((and token (eq? r token))
      (let ((next (continue-args token)))
        (unless (= (length next) (length args))
          (loop-error site "loop called with ~a arguments, expected ~a"
                      (length next) (length args)))
        (for-each (lambda (h a)
                    (cond ((and h (not (tensor? a)) (not (number? a)))
                           (loop-error site "a tensor loop variable became ~s" a))
                          ((and (not h) (tensor? a))
                           (loop-error site "a static loop variable became a tensor"))))
                  holders next)
        (let* ((outs (filter-map (lambda (h a)
                                   (and h (let ((v (as-value scratch a (sym-dtype h))))
                                            (unless (equal? (type-of v) (type-of h))
                                              (loop-error site "loop variable changes type: ~s -> ~s"
                                                          (type-of h) (type-of v)))
                                            v)))
                                 holders next))
               (static (map (lambda (h a) (if h #f a)) holders next)))
          (call-with-values
              (lambda () (body-graph ctx scratch (filter identity holders) outs))
            (lambda (text externals)
              (values 'continue (list text externals static)))))))
     (token
      (loop-error site "host loops need the loop to continue by a tail call"))
     (else (values 'exit #f)))))

(define (body-graph ctx scratch holders outs)
  "Emit one iteration as a graph with canonical names: inputs s0.. (loop
variables) then x0.. (values from outside the loop), outputs in loop-variable
order. Returns (values text external-names)."
  (call-with-values
      (lambda () (eliminate-dead-nodes (reverse (recorder-nodes scratch))
                                       (reverse (recorder-params scratch))
                                       (map sym-name outs)))
    (lambda (nodes params)
      (let* ((types (trace-context-types ctx))
             (renames (make-table))
             (counter 0)
             (local! (lambda (prefix name)
                       (unless (table-ref renames name #f)
                         (table-set! renames name
                                     (string-append prefix (number->string counter)))
                         (set! counter (+ counter 1)))))
             (holder-names (map sym-name holders))
             (defined (append (append-map gnode-output-names nodes)
                              (map gparam-name params)))
             (externals
              (delete-duplicates
               (filter (lambda (n) (not (or (member n defined) (member n holder-names))))
                       (append (append-map gnode-inputs nodes) (map sym-name outs))))))
        (for-each (lambda (n i) (table-set! renames n (string-append "s" (number->string i))))
                  holder-names (iota (length holder-names)))
        (for-each (lambda (n i) (table-set! renames n (string-append "x" (number->string i))))
                  externals (iota (length externals)))
        (for-each (lambda (p) (local! "p" (gparam-name p))) params)
        (for-each (lambda (g) (for-each (lambda (o) (local! "t" o)) (gnode-output-names g)))
                  nodes)
        (let* ((rn (lambda (n) (table-ref renames n n)))
               (typed (lambda (n) (let ((t (table-ref types n)))
                                    (list (rn n) (car t) (cdr t)))))
               (doc (graph->sexpr
                     "loop-body"
                     (map typed (append holder-names externals))
                     (map (lambda (o) (rn (sym-name o))) outs)
                     (map (lambda (p) (make-gparam (rn (gparam-name p)) (gparam-dtype p)
                                                   (gparam-shape p) (gparam-values p)))
                          params)
                     (map (lambda (g)
                            (make-gnode (gnode-op g) (map rn (gnode-inputs g))
                                        (map (lambda (o) (cons (rn (car o)) (cdr o)))
                                             (gnode-outputs g))
                                        (gnode-attrs g)))
                          nodes))))
          (values (sexpr->text doc) externals))))))

(define (emit-loop-node! rec state text externals trips)
  "Append a `loop` node running TEXT TRIPS times on STATE (symbolic values);
returns the new state."
  (let* ((ctx (recorder-context rec))
         (outs (map (lambda (s)
                      (let ((name (fresh-name! ctx "v")))
                        (register-type! ctx name (sym-dtype s) (sym-shape s))
                        (make-sym name (sym-dtype s) (sym-shape s))))
                    state)))
    (set-recorder-nodes!
     rec (cons (make-gnode 'loop
                           (append (map sym-name state) externals)
                           (map (lambda (o) (list (sym-name o) (sym-dtype o) (sym-shape o)))
                                outs)
                           `((body (string ,text))
                             (trip_count (int ,trips))
                             (state_count (int ,(length state)))))
               (recorder-nodes rec)))
    (set-recorder-count! rec (+ 1 (recorder-count rec)))
    (context-count! ctx 'host-loops)
    outs))

(define (host-loop site rec body inits)
  (let ((ctx (recorder-context rec)))
    ;; Probe the iterations: trace each body once, run-length encode.
    (let probe ((args inits) (runs '()))
      (call-with-values (lambda () (trace-iteration site rec body args))
        (lambda (kind data)
          (if (eq? kind 'continue)
              (let ((text (car data)) (externals (cadr data)) (static (caddr data)))
                (context-count! ctx 'loop-body-traces)
                (probe (map (lambda (a s) (if (tensor? a) a s)) args static)
                       ;; consecutive identical bodies share a run
                       (if (and (pair? runs)
                                (equal? (caar runs) text)
                                (equal? (cadar runs) externals))
                           (begin (context-count! ctx 'loop-body-hits)
                                  (cons (list text externals (+ 1 (caddar runs)))
                                        (cdr runs)))
                           (cons (list text externals 1) runs))))
              ;; exit: emit the runs, then trace the exit iteration for real
              (let* ((state0 (filter-map (lambda (a) (and (tensor? a) (as-value rec a #f)))
                                         inits))
                     (state (fold (lambda (run st)
                                    (emit-loop-node! rec st (car run) (cadr run) (caddr run)))
                                  state0 (reverse runs)))
                     (final-args (let loop ((as args) (st state))
                                   (cond ((null? as) '())
                                         ((tensor? (car as)) (cons (car st) (loop (cdr as) (cdr st))))
                                         (else (cons (car as) (loop (cdr as) st))))))
                     (self (covered-procedure!
                            (lambda _ (loop-error site "exit iteration continued the loop")))))
                (apply body self final-args))))))))
