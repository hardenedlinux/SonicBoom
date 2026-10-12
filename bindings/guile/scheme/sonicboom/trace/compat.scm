;;; SonicBoom — tracing prototype: implementation-specific mechanisms (Guile)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Every mechanism of the tracing prototype that is not portable R7RS-small
;;; lives here; the other trace modules reach them only through this
;;; interface. Porting the prototype to another Scheme means re-implementing
;;; this file (see "Porting guide" in design/guile-tracing-prototype.md), plus
;;; the mechanical library translations listed there.
;;;
;;; Run time
;;;   %make-prompt-tag name                    delimited control (one-shot or
;;;   %call-with-prompt tag thunk handler      multi-shot); handler receives
;;;   %abort-to-prompt tag value ...           (k value ...), k re-enterable
;;;   %raise-signal tag datum ...              non-local exit to the nearest
;;;   %catch-signal tag thunk handler          %catch-signal for TAG
;;;   call-with-output-captured thunk          -> (values result text)
;;;   call-with-output-suppressed thunk        -> result, output discarded
;;;   trace-warn message                       one line to current-error-port
;;;   procedure-tag! proc key value            associate data with a procedure
;;;   procedure-tag proc key                   ... or #f
;;;   procedure-label proc                     a name for messages
;;;   make-table table-ref table-set!          equal?-keyed mutable tables;
;;;                                            (table-ref t key [default])
;;;   table-count table->alist
;;;   %structural-procedures                   procedures that only move values
;;;   %higher-order-procedures                 ... of which these call a procedure
;;;   %unbound-guard                           marker for unbound guard names
;;;   (read-located port)                      read one datum, recording positions
;;;   (datum-position datum)                   (line . column), 1-based, or #f
;;;   (call-with-error-message thunk handler)  run THUNK; on any error call
;;;                                            (HANDLER message-string)
;;;
;;; Expansion time (syntax hooks used by the portable syntax-rules walker)
;;;   (%site form)                             source location of FORM, as a
;;;                                            constant (or the quoted form)
;;;   (%maybe-call call original walked)       an application or a macro use?
;;;   (%free-variable-guards formals body ...) guards for the body's free names

(define-module (sonicboom trace compat)
  #:use-module (srfi srfi-1)
  #:use-module (system syntax)
  #:export (%make-prompt-tag %call-with-prompt %abort-to-prompt
            %raise-signal %catch-signal
            call-with-output-captured call-with-output-suppressed trace-warn
            procedure-tag! procedure-tag procedure-label
            make-table table-ref table-set! table-count table->alist
            %structural-procedures %higher-order-procedures
            %unbound-guard %global-guard
            read-located datum-position call-with-error-message
            %site %maybe-call %free-variable-guards))

;;; --- delimited control -----------------------------------------------------------
;;; Guile prompts. Captures only the frames between the prompt and the abort;
;;; dynamic bindings made outside the prompt come from the resumption site.

(define (%make-prompt-tag name) (make-prompt-tag name))
(define (%call-with-prompt tag thunk handler) (call-with-prompt tag thunk handler))
(define (%abort-to-prompt tag . vals) (apply abort-to-prompt tag vals))

;;; --- non-local exits -----------------------------------------------------------------

(define (%raise-signal tag . data) (apply throw tag data))
(define (%catch-signal tag thunk handler)
  (catch tag thunk (lambda (key . data) (apply handler data))))

;;; --- output ports ------------------------------------------------------------------------

(define (call-with-output-captured thunk)
  (let* ((port (open-output-string))
         (result (parameterize ((current-output-port port)) (thunk))))
    (values result (get-output-string port))))

(define (call-with-output-suppressed thunk)
  (parameterize ((current-output-port (%make-void-port "w"))) (thunk)))

(define (trace-warn message)
  (let ((port (current-error-port)))
    (display message port)
    (newline port)))

;;; --- procedure tags ----------------------------------------------------------------------

(define (procedure-tag! proc key value) (set-procedure-property! proc key value))
(define (procedure-tag proc key)
  (and (procedure? proc) (procedure-property proc key)))
(define (procedure-label proc)
  (or (and (procedure? proc) (procedure-name proc)) proc))

;;; --- tables ----------------------------------------------------------------------------------

(define (make-table) (make-hash-table))
(define (table-ref table key . default)
  (hash-ref table key (if (pair? default) (car default) #f)))
(define (table-set! table key value) (hash-set! table key value))
(define (table-count table) (hash-count (const #t) table))
(define (table->alist table) (hash-map->list cons table))

;;; --- procedures that never branch on a value -------------------------------------------
;;; Strict mode does not report symbolic tensors passed to these. Listed here
;;; because the procedure objects are implementation-specific (Guile has both
;;; core and SRFI-1 versions of map / for-each / reduce-style procedures).

(define %higher-order-procedures
  (list apply map for-each (@ (srfi srfi-1) map) (@ (srfi srfi-1) for-each)))

(define %structural-procedures
  (append %higher-order-procedures
          (list list cons car cdr caar cadr cdar cddr caddr cdddr list-ref
                list-tail vector vector-ref append reverse length values)))

;;; --- source positions and error messages (validator) ------------------------------
;;; Guile's reader records line/column for every pair it reads. Portable
;;; fallback: plain `read`, and datum-position always #f (locations are then
;;; given only as document paths, e.g. "nodes[3]").

(define (read-located port) (read port))

(define (datum-position d)
  (and (pair? d)
       (let ((props (source-properties d)))
         (and (assq 'line props)
              (cons (+ 1 (cdr (assq 'line props))) (+ 1 (cdr (assq 'column props))))))))

(define (call-with-error-message thunk handler)
  (catch #t
    thunk
    (lambda (key . args)
      (handler
       (let ((p (open-output-string)))
         (if (and (= 4 (length args)) (string? (cadr args)) (list? (caddr args)))
             (display (apply format #f (cadr args) (caddr args)) p)
             (begin (display key p) (display ": " p) (write args p)))
         (get-output-string p))))))

;;; --- guards ----------------------------------------------------------------------------------

;; Guard value of a free name that is not bound when the traced procedure is
;; called (typically a name bound inside the body itself).
(define %unbound-guard (list 'unbound))

(define (%global-guard module name)
  (let ((v (module-variable module name)))
    (if (and v (variable-bound? v)) (variable-ref v) %unbound-guard)))

;;; --- syntax hooks -------------------------------------------------------------------------
;;; These three are the only non-syntax-rules macros of the prototype. Each has
;;; a portable fallback, given in the porting guide.

(define-syntax %site
  ;; Source location of FORM as a quoted alist (filename line column), or the
  ;; quoted form itself when the reader recorded no location.
  (lambda (x)
    (syntax-case x ()
      ((_ form)
       (let ((s (syntax-source #'form)))
         (if (and s (assq 'line s))
             (with-syntax ((loc (datum->syntax
                                 x (list (assq 'filename s) (assq 'line s)
                                         (assq 'column s)))))
               #''loc)
             #''form))))))

(define-syntax %maybe-call
  ;; ORIGINAL is a user form (head . args) the walker did not recognise and
  ;; WALKED its rewritten (head* . args*). If the head names a variable, it is
  ;; a procedure call: (call head* args* ...). If it names a macro or core
  ;; syntax, ORIGINAL is returned untouched (its arguments are not
  ;; expressions the walker may rewrite).
  (lambda (x)
    (syntax-case x ()
      ((_ call (h . args) walked)
       (if (identifier? #'h)
           (call-with-values (lambda () (syntax-local-binding #'h))
             (lambda (type value)
               (if (memq type '(lexical global))
                   #'(call . walked)
                   #'(h . args))))
           #'(call . walked))))))

(eval-when (expand load eval)
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
    (call-with-values (lambda () (syntax-local-binding id))
      (lambda (type value) type)))

  (define (free-variables formals body)
    ;; Two lists: outer lexicals, module-level names. Names bound inside the
    ;; body are not bound yet here: they look global (and read as unbound at
    ;; call time) or, if they shadow an outer local, lexical (an extra guard
    ;; on the outer one). Either way a guard is never missing.
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

(define-syntax %free-variable-guards
  ;; An expression evaluating to the guard alist (name . thunk) for the free
  ;; variables of BODY: outer lexicals (found now, read through a closure) and
  ;; module-level names (looked up in the module at call time, because a
  ;; compiled file is expanded before its later definitions are bound).
  (lambda (x)
    (syntax-case x ()
      ((_ formals body ...)
       (call-with-values (lambda () (free-variables #'formals #'(body ...)))
         (lambda (lexicals globals)
           (with-syntax (((v ...) lexicals)
                         ((g ...) globals)
                         (mod (datum->syntax x (module-name (current-module)))))
             #'(let ((m (resolve-module 'mod)))
                 (list (cons 'v (lambda () v)) ...
                       (cons 'g (lambda () (%global-guard m 'g))) ...)))))))))
