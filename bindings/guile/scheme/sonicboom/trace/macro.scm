;;; SonicBoom — tracing prototype: syntactic branches (define-traced / traced-lambda)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; `traced-lambda` / `define-traced` build a traced function (trace-function)
;;; whose body is rewritten at expansion time:
;;;
;;;   - every `if`, `cond`, `when` and `unless` written in the body becomes
;;;     (%branch <source-location> test then-thunk else-thunk);
;;;   - the body's free variables become guards: their values at call time
;;;     are part of the cache key. Outer lexicals are found at expansion time
;;;     (syntax-local-binding); module-level names are looked up in the module
;;;     at call time, because when a file is compiled its later top-level
;;;     definitions are not bound yet while it is being expanded.
;;;
;;; %branch on a concrete test behaves like `if` (a ctensor test uses its
;;; truth value). On a symbolic test while tracing, it traces both arms and
;;; joins their results with `where` right at the `if`, so the code after the
;;; `if` is traced once and the graph grows linearly in the number of `if`s.
;;; No continuation is involved. `if`s the rewrite cannot see (inside macros
;;; defined elsewhere, in called procedures) still need `bool`, which falls
;;; back to the continuation mechanism of (sonicboom trace tracer).
;;;
;;; The rewrite shadows the four keywords with `let-syntax` bound in the
;;; user's own syntactic context; other macros, including `and`, `or`, `case`
;;; and `do`, keep expanding to the core `if`.

(define-module (sonicboom trace macro)
  #:use-module (srfi srfi-1)
  #:use-module (system syntax)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace tracer)
  #:export (traced-lambda define-traced %branch))

;;; --- run time --------------------------------------------------------------------

(define (location->string loc)
  (if (and (pair? loc) (assq 'line loc))
      (format #f "~a:~a:~a"
              (or (assq-ref loc 'filename) "<unknown file>")
              (+ 1 (assq-ref loc 'line))        ; lines are 0-based
              (assq-ref loc 'column))
      "<unknown location>"))

(define (branch-error loc fmt . args)
  (error (string-append "traced if at " (location->string loc) ": "
                        (apply format #f fmt args))))

(define (tensor? x) (or (sym? x) (ctensor? x)))
(define (scalar? x) (or (number? x) (boolean? x)))
(define (describe x)
  (cond ((tensor? x) (list (dtype-of x) (shape-of x)))
        (else x)))

(define (merge loc rec c a b)
  "Join the values A (then arm) and B (else arm) of a symbolic `if` on C."
  (cond
   ((eq? a b) a)
   ((and (scalar? a) (scalar? b) (equal? a b)) a)
   ((and (or (tensor? a) (scalar? a)) (or (tensor? b) (scalar? b)))
    ;; Two tensors must agree exactly; a bare Scheme number/boolean is a
    ;; rank-0 constant of the other side's dtype and broadcasts.
    (when (and (tensor? a) (tensor? b)
               (not (and (eq? (dtype-of a) (dtype-of b))
                         (equal? (shape-of a) (shape-of b)))))
      (branch-error loc "arms differ in dtype/shape: then ~s, else ~s"
                    (describe a) (describe b)))
    (let* ((dt (or (dtype-of a) (dtype-of b)
                   (if (boolean? a) 'bool 'float32)))
           (a* (as-value rec a dt))
           (b* (as-value rec b dt)))
      (unless (eq? (sym-dtype a*) (sym-dtype b*))
        (branch-error loc "arms differ in dtype: then ~s, else ~s"
                      (describe a) (describe b)))
      (context-count! (recorder-context rec) 'branch-merges)
      (record-node! rec 'where (list c a* b*) '())))
   ((and (pair? a) (pair? b))
    (cons (merge loc rec c (car a) (car b)) (merge loc rec c (cdr a) (cdr b))))
   ((and (vector? a) (vector? b) (= (vector-length a) (vector-length b)))
    (list->vector (map (lambda (x y) (merge loc rec c x y))
                       (vector->list a) (vector->list b))))
   ((equal? a b) a)
   (else
    (branch-error loc "arms return values that cannot be joined: then ~s, else ~s"
                  (describe a) (describe b)))))

;; Guard value of a module-level name at call time. Unbound names (typically
;; names bound inside the body that the expander could not rule out) read as
;; %unbound-guard, so they never cause a miss.
(define (%global-guard module name)
  (let ((v (module-variable module name)))
    (if (and v (variable-bound? v)) (variable-ref v) %unbound-guard)))

(define* (%branch loc test then-thunk else-thunk #:optional (statement? #f))
  "Run time of a rewritten `if`. LOC is the source location of the form.
STATEMENT? marks forms used for effect (one-armed `if`, `when`, `unless`): on
a symbolic test both arms are traced and the result is unspecified."
  (cond
   ((sym? test)
    (let ((rec (current-tracer)))
      (unless rec
        (branch-error loc "symbolic condition outside of tracing"))
      (unless (= 1 (shape-numel (sym-shape test)))
        (branch-error loc "condition has more than one element: ~s"
                      (sym-shape test)))
      (unless (eq? (sym-dtype test) 'bool)
        (branch-error loc "condition must be a bool tensor, got ~s"
                      (sym-dtype test)))
      (let* ((name (sym-name test))
             (known (or (assoc name (%branch-assumptions))
                        (assoc name (recorder-assumptions rec)))))
        (if known
            (if (cdr known) (then-thunk) (else-thunk))
            (let* ((a (parameterize ((%branch-assumptions
                                      (acons name #t (%branch-assumptions))))
                        (then-thunk)))
                   (b (parameterize ((%branch-assumptions
                                      (acons name #f (%branch-assumptions))))
                        (else-thunk))))
              (if statement?
                  (if #f #f)
                  (merge loc rec test a b)))))))
   ((ctensor? test)
    (if (bool test) (then-thunk) (else-thunk)))
   (test (then-thunk))
   (else (else-thunk))))

;;; --- expansion time ---------------------------------------------------------------

(define-syntax %cond-branch
  (syntax-rules (else =>)
    ((_ loc) (if #f #f))
    ((_ loc (else e ...)) (let () e ...))
    ((_ loc (test => f) clause ...)
     (let ((t test))
       (%branch 'loc t (lambda () (f t)) (lambda () (%cond-branch loc clause ...)))))
    ((_ loc (test) clause ...)
     (let ((t test))
       (%branch 'loc t (lambda () t) (lambda () (%cond-branch loc clause ...)))))
    ((_ loc (test e ...) clause ...)
     (%branch 'loc test (lambda () e ...)
              (lambda () (%cond-branch loc clause ...))))))

(eval-when (expand load eval)
  (define (source-of x)
    (let ((s (syntax-source x)))
      (and s (list (assq 'filename s) (assq 'line s) (assq 'column s)))))

  (define (rewrite kind x)
    ;; X is a use of the shadowed keyword KIND in the traced body.
    (with-syntax ((loc (datum->syntax x (source-of x))))
      (syntax-case x ()
        ((_ test then alt) (eq? kind 'if)
         #'(%branch 'loc test (lambda () then) (lambda () alt)))
        ((_ test then) (eq? kind 'if)
         #'(%branch 'loc test (lambda () then) (lambda () (if #f #f)) #t))
        ((_ clause ...) (eq? kind 'cond)
         #'(%cond-branch loc clause ...))
        ((_ test e ...) (eq? kind 'when)
         #'(%branch 'loc test (lambda () e ...) (lambda () (if #f #f)) #t))
        ((_ test e ...) (eq? kind 'unless)
         #'(%branch 'loc test (lambda () (if #f #f)) (lambda () e ...) #t)))))

  (define (identifiers stx)
    (syntax-case stx ()
      ((a . b) (append (identifiers #'a) (identifiers #'b)))
      (#(x ...) (identifiers #'(x ...)))
      (id (identifier? #'id) (list #'id))
      (_ '())))

  (define (formal-names formals)
    (let loop ((f (syntax->datum formals)))
      (cond ((pair? f) (cons (car f) (loop (cdr f))))
            ((symbol? f) (list f))
            (else '()))))

  (define (binding-type id)
    ;; 'lexical (outer local), 'global (module-level or unbound), or other
    ;; (macros, core syntax, ...), which are never guarded. Names bound inside
    ;; the body itself are not bound yet here: they look global (and read as
    ;; unbound at call time) or, if they shadow an outer local, lexical (an
    ;; extra guard on the outer one). Either way a guard is never missing.
    (call-with-values (lambda () (syntax-local-binding id))
      (lambda (type value) type)))

  (define (free-variables formals body)
    "Two lists of identifiers: outer lexicals, module-level names."
    (let ((skip (formal-names formals)))
      (let loop ((ids (identifiers body)) (lex '()) (glob '()))
        (if (null? ids)
            (values (reverse lex) (reverse glob))
            (let ((id (car ids)))
              (if (or (memq (syntax->datum id) skip)
                      (any (lambda (seen) (free-identifier=? seen id))
                           (append lex glob)))
                  (loop (cdr ids) lex glob)
                  (case (binding-type id)
                    ((lexical) (loop (cdr ids) (cons id lex) glob))
                    ((global) (loop (cdr ids) lex (cons id glob)))
                    (else (loop (cdr ids) lex glob))))))))))

(define-syntax %traced-lambda
  (lambda (stx)
    (syntax-case stx ()
      ((_ ctx formals (opt ...) body ...)
       (call-with-values (lambda () (free-variables #'formals #'(body ...)))
        (lambda (lexicals globals)
       (with-syntax (((v ...) lexicals)
                     ((g ...) globals)
                     (mod     (datum->syntax #'ctx (module-name (current-module))))
                     (if*     (datum->syntax #'ctx 'if))
                     (cond*   (datum->syntax #'ctx 'cond))
                     (when*   (datum->syntax #'ctx 'when))
                     (unless* (datum->syntax #'ctx 'unless)))
         #'(trace-function
            (lambda formals
              (let-syntax ((if*     (lambda (x) (rewrite 'if x)))
                           (cond*   (lambda (x) (rewrite 'cond x)))
                           (when*   (lambda (x) (rewrite 'when x)))
                           (unless* (lambda (x) (rewrite 'unless x))))
                body ...))
            #:guards (let ((m (resolve-module 'mod)))
                       (list (cons 'v (lambda () v)) ...
                             (cons 'g (lambda () (%global-guard m 'g))) ...))
            opt ...))))))))

(eval-when (expand load eval)
  (define (split-options stx)
    ;; Leading #:keyword value pairs before the body.
    (let loop ((xs stx) (opts '()))
      (syntax-case xs ()
        ((k v . rest) (keyword? (syntax->datum #'k))
         (loop #'rest (cons* #'v #'k opts)))
        (_ (values (reverse opts) xs))))))

(define-syntax traced-lambda
  (lambda (stx)
    "(traced-lambda formals [#:option value ...] body ...)"
    (syntax-case stx ()
      ((kw formals . rest)
       (call-with-values (lambda () (split-options #'rest))
         (lambda (opts body)
           (with-syntax (((opt ...) opts) ((b ...) body))
             #'(%traced-lambda kw formals (opt ...) b ...))))))))

(define-syntax define-traced
  (lambda (stx)
    "(define-traced (name . formals) [#:option value ...] body ...)"
    (syntax-case stx ()
      ((kw (name . formals) . rest)
       (call-with-values (lambda () (split-options #'rest))
         (lambda (opts body)
           (with-syntax (((opt ...) opts) ((b ...) body))
             #'(define name (%traced-lambda kw formals (opt ...) b ...)))))))))
