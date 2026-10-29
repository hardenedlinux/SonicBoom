;;; SonicBoom — tracing prototype: syntactic branches (define-traced / traced-lambda)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; `traced-lambda` / `define-traced` build a traced function (trace-function)
;;; whose body is first rewritten by a code walker written only in
;;; `syntax-rules` (continuation-passing style, R7RS). The walker matches the
;;; standard control forms literally and rewrites them so that a symbolic
;;; condition is resolved inside the graph instead of being taken as a true
;;; Scheme value:
;;;
;;;   if, when, unless, cond (incl. =>)  -> %branch: trace both arms, join with
;;;                                         `where` at the form
;;;   and, or                            -> %and / %or: `and` / `or` tensor ops
;;;   case                               -> %case: `equal` tests, %branch chain
;;;   do, named let                      -> %loop (see (sonicboom trace loop)):
;;;                                         unrolled up to #:max-unroll iterations,
;;;                                         beyond that a host loop over one graph;
;;;                                         do's test is a %branch
;;;   lambda, define                     -> body walked; the procedure is marked
;;;                                         as covered (see strict mode)
;;;
;;; It also walks the standard binding forms (let, let*, letrec, letrec*,
;;; let-values, let*-values, case-lambda, set!, begin, parameterize, delay) so
;;; that the forms above are found anywhere in the body. Anything else with an
;;; identifier head is handed to the %maybe-call hook of (sonicboom trace
;;; compat): a procedure call becomes (%call f arg ...), with the arguments
;;; walked; a macro use is left exactly as written (macros defined elsewhere
;;; are not rewritten, and their arguments are not touched). quote,
;;; quasiquote and (trace-opaque form) are never walked.
;;;
;;; A symbolic branch whose own arms re-enter it (a data-dependent loop written
;;; with do / named let / recursion) cannot be unrolled; the re-entry is
;;; detected and that branch falls back to `bool`, i.e. a graph break through
;;; the continuation mechanism of (sonicboom trace tracer).
;;;
;;; The only non-syntax-rules pieces are the three hooks %site, %maybe-call and
;;; %free-variable-guards, which live in (sonicboom trace compat).

(define-module (sonicboom trace macro)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace tracer)
  #:use-module (sonicboom trace loop)
  #:export (traced-lambda traced-lambda* define-traced define-traced*
            trace-opaque
            %branch %and %or %case %call %covered))

;;; =====================================================================
;;; Run time
;;; =====================================================================

;;; --- sites and errors ------------------------------------------------------------

;; A site identifies one rewritten form: the constant produced by %site (a
;; location alist, or the quoted form), or #(site clause-marker) for the
;; clauses of one cond / case / and / or.
(define (site-location site)
  (if (vector? site) (site-location (vector-ref site 0)) site))

(define (location->string site)
  (let ((loc (site-location site)))
    (if (and (pair? loc) (pair? (car loc)) (assq 'line loc))
        (string-append (let ((f (assq 'filename loc)))
                         (if (and f (cdr f)) (cdr f) "<unknown file>"))
                       ":" (number->string (+ 1 (cdr (assq 'line loc))))
                       ":" (number->string (cdr (assq 'column loc))))
        (let ((p (open-output-string))) (write loc p) (get-output-string p)))))

(define (branch-error site fmt . args)
  (error (string-append "traced form at " (location->string site) ": "
                        (apply format #f fmt args))))

(define (tensor? x) (or (sym? x) (ctensor? x)))
(define (scalar? x) (or (number? x) (boolean? x)))
(define (describe x)
  (if (tensor? x) (list (dtype-of x) (shape-of x)) x))

;;; --- conditions -------------------------------------------------------------------------

(define (require-condition site c)
  "C is symbolic: check it can steer a branch and return the active recorder."
  (let ((rec (current-tracer)))
    (unless rec (branch-error site "symbolic condition outside of tracing"))
    (unless (= 1 (shape-numel (sym-shape c)))
      (branch-error site "condition has more than one element: ~s" (sym-shape c)))
    (unless (eq? (sym-dtype c) 'bool)
      (branch-error site "condition must be a bool tensor, got ~s" (sym-dtype c)))
    rec))

(define (known-value c rec)
  "The boolean C is already fixed to on this path, as (name . value), or #f."
  (let ((name (sym-name c)))
    (or (assoc name (%branch-assumptions))
        (assoc name (recorder-assumptions rec)))))

(define (with-assumption c value thunk)
  (parameterize ((%branch-assumptions
                  (cons (cons (sym-name c) value) (%branch-assumptions))))
    (thunk)))

;;; --- data-dependent loops ------------------------------------------------------------

;; Sites whose symbolic join is in progress (a parameter, so a continuation
;; captured inside an arm restores it).
(define %active-sites (make-parameter '()))

(define (symbolic-site site rec join fallback)
  "Run JOIN, which traces both outcomes of the symbolic condition at SITE. If
SITE is re-entered from inside its own join (a loop whose iteration count
depends on the data), unwind to here, drop what the join recorded, and run
FALLBACK, which resolves the condition with `bool` (a graph break)."
  (if (member site (%active-sites))
      (%raise-signal 'sonicboom-trace-site-reentry site)
      (let ((nodes (recorder-nodes rec))
            (params (recorder-params rec))
            (index (recorder-param-index rec))
            (count (recorder-count rec)))
        (%catch-signal 'sonicboom-trace-site-reentry
          (lambda ()
            (parameterize ((%active-sites (cons site (%active-sites))))
              (join)))
          (lambda (s)
            (cond ((equal? s site)
                   (when (eq? (current-tracer) rec)
                     (set-recorder-nodes! rec nodes)
                     (set-recorder-params! rec params)
                     (set-recorder-param-index! rec index)
                     (set-recorder-count! rec count))
                   (context-count! (recorder-context (current-tracer)) 'loop-breaks)
                   (fallback))
                  (else (%raise-signal 'sonicboom-trace-site-reentry s))))))))

;;; --- joining values -------------------------------------------------------------------

(define (merge site rec c a b)
  "Join the values A (then arm) and B (else arm) of a symbolic branch on C."
  (cond
   ((eq? a b) a)
   ((and (scalar? a) (scalar? b) (equal? a b)) a)
   ((and (or (tensor? a) (scalar? a)) (or (tensor? b) (scalar? b)))
    ;; Two tensors must agree exactly; a bare Scheme number/boolean is a
    ;; rank-0 constant of the other side's dtype and broadcasts.
    (when (and (tensor? a) (tensor? b)
               (not (and (eq? (dtype-of a) (dtype-of b))
                         (equal? (shape-of a) (shape-of b)))))
      (branch-error site "arms differ in dtype/shape: then ~s, else ~s"
                    (describe a) (describe b)))
    (let* ((dt (or (dtype-of a) (dtype-of b)
                   (if (boolean? a) 'bool 'float32)))
           (a* (as-value rec a dt))
           (b* (as-value rec b dt)))
      (unless (eq? (sym-dtype a*) (sym-dtype b*))
        (branch-error site "arms differ in dtype: then ~s, else ~s"
                      (describe a) (describe b)))
      (context-count! (recorder-context rec) 'branch-merges)
      (record-node! rec 'where (list c a* b*) '())))
   ((and (pair? a) (pair? b))
    (cons (merge site rec c (car a) (car b)) (merge site rec c (cdr a) (cdr b))))
   ((and (vector? a) (vector? b) (= (vector-length a) (vector-length b)))
    (list->vector (map (lambda (x y) (merge site rec c x y))
                       (vector->list a) (vector->list b))))
   ((equal? a b) a)
   (else
    (branch-error site "arms return values that cannot be joined: then ~s, else ~s"
                  (describe a) (describe b)))))

;;; --- %branch: if / when / unless / cond / do ------------------------------------------

(define (%branch site test then-thunk else-thunk . statement)
  "Run time of a rewritten conditional. A non-empty STATEMENT marks forms used
for effect (one-armed if, when, unless): on a symbolic test both arms are
traced and the result is unspecified."
  (cond
   ((sym? test)
    (let* ((rec (require-condition site test))
           (known (known-value test rec)))
      (if known
          (if (cdr known) (then-thunk) (else-thunk))
          (symbolic-site
           site rec
           (lambda ()
             (let* ((a (with-assumption test #t then-thunk))
                    (b (with-assumption test #f else-thunk)))
               (if (pair? statement)
                   (if #f #f)
                   ;; the arms may have crossed a graph break: record into
                   ;; whatever recorder is current now
                   (merge site (current-tracer) test a b))))
           (lambda () (if (bool test) (then-thunk) (else-thunk)))))))
   ((ctensor? test) (if (bool test) (then-thunk) (else-thunk)))
   (test (then-thunk))
   (else (else-thunk))))

;;; --- %and / %or -------------------------------------------------------------------------

(define (symbolic-logic site op first rest-thunk)
  ;; FIRST is a symbolic condition whose truth cannot be decided: evaluate the
  ;; rest under the assumption that it does not short-circuit, then combine
  ;; with the `and` / `or` tensor op.
  (let* ((rec (require-condition site first))
         (known (known-value first rec))
         (short (lambda ()
                  (if (eq? op 'and)
                      (if (bool first) (rest-thunk) first)
                      (if (bool first) first (rest-thunk))))))
    (if known
        (if (eq? (cdr known) (eq? op 'and)) (rest-thunk) first)
        (symbolic-site
         site rec
         (lambda ()
           (let ((r (with-assumption first (eq? op 'and) rest-thunk))
                 (rec (current-tracer)))
             (unless (or (boolean? r) (and (tensor? r) (eq? (dtype-of r) 'bool)))
               (branch-error site "~a on a symbolic condition needs bool operands, got ~s"
                             op (describe r)))
             (context-count! (recorder-context rec) 'logic-merges)
             (record-node! rec op (list first (as-value rec r 'bool)) '())))
         short))))

(define (%and site first rest-thunk)
  (cond ((sym? first) (symbolic-logic site 'and first rest-thunk))
        ((ctensor? first) (if (bool first) (rest-thunk) first))
        (first (rest-thunk))
        (else first)))

(define (%or site first rest-thunk)
  (cond ((sym? first) (symbolic-logic site 'or first rest-thunk))
        ((ctensor? first) (if (bool first) first (rest-thunk)))
        (first first)
        (else (rest-thunk))))

;;; --- %case ----------------------------------------------------------------------------------

(define (%case site key clauses else-proc)
  "CLAUSES is a list of (datums . proc), ELSE-PROC a proc or #f; procs take
the key (for =>). A ctensor key is compared numerically by its single
element; a symbolic key becomes a chain of %branch on `equal` tests."
  (define (match? k datums)
    (if (ctensor? key)
        (any (lambda (d) (if (and (number? k) (number? d)) (= k d) (eqv? k d))) datums)
        (memv k datums)))
  (if (sym? key)
      (let loop ((cs clauses) (i 0))
        (if (null? cs)
            (if else-proc (else-proc key) (if #f #f))
            (let ((ds (filter (if (eq? (sym-dtype key) 'bool) boolean? number?)
                              (caar cs))))
              (if (null? ds)
                  (loop (cdr cs) (+ i 1))
                  (%branch (vector site i)
                           (reduce (lambda (t acc) (logical-or acc t)) #f
                                   (map (lambda (d) (equal-to key d)) ds))
                           (lambda () ((cdar cs) key))
                           (lambda () (loop (cdr cs) (+ i 1))))))))
      (let ((k (if (ctensor? key) (ctensor-item key) key)))
        (let loop ((cs clauses))
          (cond ((null? cs) (if else-proc (else-proc key) (if #f #f)))
                ((match? k (caar cs)) ((cdar cs) key))
                (else (loop (cdr cs))))))))

;;; --- strict mode: %call / %covered -------------------------------------------------------

(define (%covered proc) (covered-procedure! proc))

(for-each covered-procedure! (list shape-of dtype-of))

(define (strict? rec)
  (let ((s (trace-context-strict (recorder-context rec))))
    (if (eq? s 'default) (strict-tracing) s)))

(define (carries-symbolic? x depth)
  (cond ((sym? x) #t)
        ((zero? depth) #f)
        ((pair? x) (or (carries-symbolic? (car x) (- depth 1))
                       (carries-symbolic? (cdr x) depth)))
        ((vector? x) (any (lambda (y) (carries-symbolic? y (- depth 1)))
                          (vector->list x)))
        (else #f)))

(define (covered? f)
  (or (covered-procedure? f) (traced-function? f)))

(define (strict-report! rec f)
  (let* ((ctx (recorder-context rec))
         (label (let ((p (open-output-string)))
                  (display (procedure-label f) p)
                  (get-output-string p)))
         (note (string-append
                "procedure " label " received a symbolic tensor but is not "
                "defined with define-traced/traced-lambda: a core `if` inside "
                "it takes the tensor as true (define it with define-traced, "
                "or decide with bool)")))
    (unless (member note (trace-context-notes ctx))
      (add-context-note! ctx note)
      (trace-warn (string-append "sonicboom-trace: strict: " note)))))

(define (%call f . args)
  "A procedure call from a traced body. In strict mode, report a symbolic
tensor reaching a procedure the walker did not cover."
  (let ((rec (current-tracer)))
    (when (and rec (strict? rec)
               (any (lambda (a) (carries-symbolic? a 2)) args))
      (cond ((covered? f) #t)
            ((memq f %structural-procedures)
             (when (memq f %higher-order-procedures)
               (for-each (lambda (p)
                           (when (and (procedure? p) (not (covered? p))
                                      (not (memq p %structural-procedures)))
                             (strict-report! rec p)))
                         args)))
            (else (strict-report! rec f)))))
  (apply f args))

;;; =====================================================================
;;; Expansion time: the syntax-rules walker
;;; =====================================================================
;;;
;;; Protocol: (%walk form k) rewrites FORM and delivers the result to the
;;; continuation K = (k-macro extra ...) as (k-macro result extra ...), via
;;; %wret. Continuation macros are named %k-....

(define-syntax %wret
  (syntax-rules ()
    ((_ (kname extra ...) v) (kname v extra ...))))

(define-syntax %walk
  (syntax-rules ()
    ((_ form k) (%walk-form form form k))))

;; Walk every element of a list as an expression (an improper tail is kept).
(define-syntax %walk-list
  (syntax-rules ()
    ((_ () k) (%wret k ()))
    ((_ (x . rest) k) (%walk x (%k-list-head rest k)))
    ((_ tail k) (%wret k tail))))
(define-syntax %k-list-head
  (syntax-rules ()
    ((_ x* rest k) (%walk-list rest (%k-list-tail x* k)))))
(define-syntax %k-list-tail
  (syntax-rules ()
    ((_ rest* x* k) (%wret k (x* . rest*)))))

;; ((head . exprs) ...): keep each head, walk its exprs (let bindings, case
;; clauses, do specs, case-lambda clauses).
(define-syntax %walk-tails
  (syntax-rules ()
    ((_ () k) (%wret k ()))
    ((_ ((h . t) . rest) k) (%walk-list t (%k-tails-head h rest k)))
    ((_ (x . rest) k) (%walk-tails rest (%k-list-tail x k)))))
(define-syntax %k-tails-head
  (syntax-rules ()
    ((_ t* h rest k) (%walk-tails rest (%k-list-tail (h . t*) k)))))

;; (exprs ...): walk each element as a list of expressions (cond clauses,
;; parameterize bindings).
(define-syntax %walk-each
  (syntax-rules ()
    ((_ () k) (%wret k ()))
    ((_ (x . rest) k) (%walk-list x (%k-each-head rest k)))))
(define-syntax %k-each-head
  (syntax-rules ()
    ((_ x* rest k) (%walk-each rest (%k-list-tail x* k)))))

(define-syntax %walk-form
  (syntax-rules (quote quasiquote trace-opaque if when unless cond case and or
                 do let let* letrec letrec* let-values let*-values lambda
                 case-lambda define set! begin parameterize delay delay-force)
    ((_ (quote x) w k) (%wret k (quote x)))
    ((_ (quasiquote x) w k) (%wret k (quasiquote x)))
    ((_ (trace-opaque x) w k) (%wret k x))
    ;; conditionals
    ((_ (if c a b) w k) (%walk-list (c a b) (%k-if w k)))
    ((_ (if c a) w k) (%walk-list (c a) (%k-if1 w k)))
    ((_ (when c e ...) w k) (%walk-list (c e ...) (%k-when w k)))
    ((_ (unless c e ...) w k) (%walk-list (c e ...) (%k-unless w k)))
    ((_ (cond clause ...) w k) (%walk-each (clause ...) (%k-cond w k)))
    ((_ (case key clause ...) w k) (%walk key (%k-case1 (clause ...) w k)))
    ((_ (and e ...) w k) (%walk-list (e ...) (%k-and w k)))
    ((_ (or e ...) w k) (%walk-list (e ...) (%k-or w k)))
    ((_ (do (spec ...) (test r ...) body ...) w k)
     (%walk-tails (spec ...) (%k-do1 (test r ...) (body ...) w k)))
    ;; binding forms
    ((_ (let (b ...) body ...) w k) (%walk-let let (b ...) (body ...) k))
    ((_ (let name (b ...) body ...) w k)
     (%walk-tails (b ...) (%k-named-let1 name (body ...) w k)))
    ((_ (let* (b ...) body ...) w k) (%walk-let let* (b ...) (body ...) k))
    ((_ (letrec (b ...) body ...) w k) (%walk-let letrec (b ...) (body ...) k))
    ((_ (letrec* (b ...) body ...) w k) (%walk-let letrec* (b ...) (body ...) k))
    ((_ (let-values (b ...) body ...) w k) (%walk-let let-values (b ...) (body ...) k))
    ((_ (let*-values (b ...) body ...) w k) (%walk-let let*-values (b ...) (body ...) k))
    ((_ (lambda formals body ...) w k) (%walk-list (body ...) (%k-lambda formals k)))
    ((_ (case-lambda clause ...) w k) (%walk-tails (clause ...) (%k-case-lambda k)))
    ((_ (define (name . formals) body ...) w k)
     (%walk-list (body ...) (%k-define-proc name formals k)))
    ((_ (define name e) w k) (%walk e (%k-wrap2 define name k)))
    ((_ (set! v e) w k) (%walk e (%k-wrap2 set! v k)))
    ((_ (begin e ...) w k) (%walk-list (e ...) (%k-wrap begin k)))
    ((_ (parameterize (b ...) body ...) w k)
     (%walk-each (b ...) (%k-param1 (body ...) k)))
    ((_ (delay e) w k) (%walk e (%k-wrap1 delay k)))
    ((_ (delay-force e) w k) (%walk e (%k-wrap1 delay-force k)))
    ;; a call, or a macro use the walker does not know
    ((_ (h . t) w k) (%walk-list (h . t) (%k-app w k)))
    ((_ atom w k) (%wret k atom))))

;;; --- continuations: conditionals ------------------------------------------------------

(define-syntax %k-if
  (syntax-rules ()
    ((_ (c a b) w k)
     (%wret k (%branch (%site w) c (lambda () a) (lambda () b))))))
(define-syntax %k-if1
  (syntax-rules ()
    ((_ (c a) w k)
     (%wret k (%branch (%site w) c (lambda () a) (lambda () (if #f #f)) 'statement)))))
(define-syntax %k-when
  (syntax-rules ()
    ((_ (c e ...) w k)
     (%wret k (%branch (%site w) c (lambda () e ...) (lambda () (if #f #f)) 'statement)))))
(define-syntax %k-unless
  (syntax-rules ()
    ((_ (c e ...) w k)
     (%wret k (%branch (%site w) c (lambda () (if #f #f)) (lambda () e ...) 'statement)))))
(define-syntax %k-cond
  (syntax-rules ()
    ((_ (clause ...) w k) (%wret k (%cond-branch (%site w) () clause ...)))))
(define-syntax %k-and
  (syntax-rules ()
    ((_ (e ...) w k) (%wret k (%build-logic %and #t (%site w) () e ...)))))
(define-syntax %k-or
  (syntax-rules ()
    ((_ (e ...) w k) (%wret k (%build-logic %or #f (%site w) () e ...)))))
(define-syntax %k-case1
  (syntax-rules ()
    ((_ key* clauses w k) (%walk-tails clauses (%k-case2 key* w k)))))
(define-syntax %k-case2
  (syntax-rules ()
    ((_ (clause ...) key* w k) (%wret k (%case-clauses (%site w) key* () clause ...)))))
(define-syntax %k-do1
  (syntax-rules ()
    ((_ specs* (test r ...) body w k) (%walk-list (test r ...) (%k-do2 specs* body w k)))))
(define-syntax %k-do2
  (syntax-rules ()
    ((_ tests* specs* (body ...) w k) (%walk-list (body ...) (%k-do3 specs* tests* w k)))))
(define-syntax %k-do3
  (syntax-rules ()
    ((_ (body ...) specs* tests* w k) (%wret k (%do (%site w) specs* tests* body ...)))))

;;; --- continuations: binding forms ---------------------------------------------------

(define-syntax %walk-let
  (syntax-rules ()
    ((_ kw bindings body k) (%walk-tails bindings (%k-let1 kw body k)))))
(define-syntax %k-let1
  (syntax-rules ()
    ((_ bindings* kw (body ...) k) (%walk-list (body ...) (%k-let2 kw bindings* k)))))
(define-syntax %k-let2
  (syntax-rules ()
    ((_ (body ...) kw bindings* k) (%wret k (kw bindings* body ...)))))
(define-syntax %k-named-let1
  (syntax-rules ()
    ((_ bindings* name (body ...) w k) (%walk-list (body ...) (%k-named-let2 name bindings* w k)))))
(define-syntax %k-named-let2
  (syntax-rules ()
    ((_ (body ...) name ((v e) ...) w k)
     (%wret k (%loop (%site w) (lambda (name v ...) body ...) e ...)))))
(define-syntax %k-lambda
  (syntax-rules ()
    ((_ (body ...) formals k) (%wret k (%covered (lambda formals body ...))))))
(define-syntax %k-case-lambda
  (syntax-rules ()
    ((_ (clause ...) k) (%wret k (%covered (case-lambda clause ...))))))
(define-syntax %k-define-proc
  (syntax-rules ()
    ((_ (body ...) name formals k)
     (%wret k (define name (%covered (lambda formals body ...)))))))
(define-syntax %k-param1
  (syntax-rules ()
    ((_ bindings* (body ...) k) (%walk-list (body ...) (%k-let2 parameterize bindings* k)))))
(define-syntax %k-wrap
  (syntax-rules ()
    ((_ (e ...) kw k) (%wret k (kw e ...)))))
(define-syntax %k-wrap1
  (syntax-rules ()
    ((_ e kw k) (%wret k (kw e)))))
(define-syntax %k-wrap2
  (syntax-rules ()
    ((_ e kw x k) (%wret k (kw x e)))))
(define-syntax %k-app
  (syntax-rules ()
    ((_ walked w k) (%wret k (%maybe-call %call w walked)))))

;;; --- output forms -------------------------------------------------------------------------

;; Clauses get distinct sites, #(site (marker ...)), so that a later clause,
;; which runs inside the else arm of an earlier one, is not mistaken for a
;; re-entry of the same branch.
(define-syntax %cond-branch
  (syntax-rules (else =>)
    ((_ site (i ...)) (if #f #f))
    ((_ site (i ...) (else e ...)) (let () e ...))
    ((_ site (i ...) (test => f) clause ...)
     (let ((t test))
       (%branch (vector site '(i ...)) t
                (lambda () (f t))
                (lambda () (%cond-branch site (i ... x) clause ...)))))
    ((_ site (i ...) (test) clause ...)
     (let ((t test))
       (%branch (vector site '(i ...)) t
                (lambda () t)
                (lambda () (%cond-branch site (i ... x) clause ...)))))
    ((_ site (i ...) (test e ...) clause ...)
     (%branch (vector site '(i ...)) test
              (lambda () e ...)
              (lambda () (%cond-branch site (i ... x) clause ...))))))

(define-syntax %build-logic
  (syntax-rules ()
    ((_ op unit site (i ...)) unit)
    ((_ op unit site (i ...) e) e)
    ((_ op unit site (i ...) e1 e2 ...)
     (op (vector site '(i ...)) e1
         (lambda () (%build-logic op unit site (i ... x) e2 ...))))))

(define-syntax %case-clauses
  (syntax-rules (else =>)
    ((_ site key (acc ...))
     (%case site key (list acc ...) #f))
    ((_ site key (acc ...) (else => f))
     (%case site key (list acc ...) (lambda (k) (f k))))
    ((_ site key (acc ...) (else e ...))
     (%case site key (list acc ...) (lambda (k) e ...)))
    ((_ site key (acc ...) ((d ...) => f) clause ...)
     (%case-clauses site key (acc ... (cons '(d ...) (lambda (k) (f k)))) clause ...))
    ((_ site key (acc ...) ((d ...) e ...) clause ...)
     (%case-clauses site key (acc ... (cons '(d ...) (lambda (k) e ...))) clause ...))))

(define-syntax %do-step
  (syntax-rules ()
    ((_ v) v)
    ((_ v s) s)))

(define-syntax %do
  (syntax-rules ()
    ((_ site ((v i s ...) ...) (test r ...) body ...)
     (%loop site
            (lambda (loop v ...)
              (%branch site test
                       (lambda () (if #f #f) r ...)
                       (lambda () body ... (loop (%do-step v s ...) ...))))
            i ...))))

;;; --- entry points ----------------------------------------------------------------------------

(define-syntax trace-opaque
  ;; (trace-opaque form): FORM is left exactly as written by the walker.
  (syntax-rules ()
    ((_ form) form)))

(define-syntax %k-traced
  (syntax-rules ()
    ((_ (body* ...) formals (opt ...) (body ...))
     (trace-function (lambda formals body* ...)
                     #:guards (%free-variable-guards formals body ...)
                     opt ...))))

(define-syntax traced-lambda*
  ;; (traced-lambda* (option ...) formals body ...), options as trace-function
  (syntax-rules ()
    ((_ (opt ...) formals body ...)
     (%walk-list (body ...) (%k-traced formals (opt ...) (body ...))))))

(define-syntax traced-lambda
  (syntax-rules ()
    ((_ formals body ...) (traced-lambda* () formals body ...))))

(define-syntax define-traced*
  (syntax-rules ()
    ((_ (opt ...) (name . formals) body ...)
     (define name (traced-lambda* (opt ...) formals body ...)))))

(define-syntax define-traced
  (syntax-rules ()
    ((_ (name . formals) body ...)
     (define name (traced-lambda formals body ...)))))
