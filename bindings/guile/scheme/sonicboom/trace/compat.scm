;;; SonicBoom — tracing prototype: implementation-specific mechanisms (Guile)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Every mechanism of the tracing prototype that is not portable R7RS-small
;;; lives here; the other trace modules reach them only through this
;;; interface. Porting the prototype to another Scheme means re-implementing
;;; this file (see the tracing prototype's porting guide), plus
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

(define (%make-prompt-tag name)
  "Create a fresh prompt tag named NAME.

Args:
  name: a symbol naming the tag.
Returns: a prompt tag for `%call-with-prompt` and `%abort-to-prompt`.
PyTorch: none"
  (make-prompt-tag name))
(define (%call-with-prompt tag thunk handler)
  "Call THUNK with a prompt tagged TAG.

The handler is called with the continuation and the values of an abort to TAG.
Captured frames are those between the prompt and the abort; dynamic bindings
made outside the prompt are restored from the resumption site.

Args:
  tag: a prompt tag.
  thunk: the body.
  handler: a procedure of (continuation . values).
Returns: what THUNK or the handler returns.
Example:
  (use-modules (sonicboom trace compat))
  (let ((tag (%make-prompt-tag 'example)))
    (%call-with-prompt tag
      (lambda () (%abort-to-prompt tag 7))
      (lambda (k value) (k value))))
  => 7
PyTorch: none"
  (call-with-prompt tag thunk handler))
(define (%abort-to-prompt tag . vals)
  "Abort to the prompt tagged TAG, passing VALS to its handler.

Args:
  tag: a prompt tag.
  vals: the values the handler receives after the continuation.
Returns: the value the handler passes to the continuation.
Example:
  (use-modules (sonicboom trace compat))
  (let ((tag (%make-prompt-tag 'example)))
    (%call-with-prompt tag
      (lambda () (%abort-to-prompt tag 7))
      (lambda (k value) (k (+ value 1)))))
  => 8
PyTorch: none"
  (apply abort-to-prompt tag vals))

;;; --- non-local exits -----------------------------------------------------------------

(define (%raise-signal tag . data)
  "Raise a non-local exit tagged TAG with DATA.

Args:
  tag: the signal tag.
  data: the values the handler receives.
Returns: never returns normally.
Example:
  (use-modules (sonicboom trace compat))
  (%catch-signal 'my-signal
                 (lambda () (%raise-signal 'my-signal 1 2))
                 (lambda (a b) (list a b)))
  => (1 2)
PyTorch: none"
  (apply throw tag data))
(define (%catch-signal tag thunk handler)
  "Run THUNK, calling HANDLER with the data of a `%raise-signal` for TAG.

Args:
  tag: the signal tag to catch.
  thunk: the body.
  handler: a procedure of the raised data.
Returns: what THUNK or HANDLER returns.
Example:
  (use-modules (sonicboom trace compat))
  (%catch-signal 'my-signal
                 (lambda () (%raise-signal 'my-signal 1 2))
                 (lambda (a b) (list a b)))
  => (1 2)
PyTorch: none"
  (catch tag thunk (lambda (key . data) (apply handler data))))

;;; --- output ports ------------------------------------------------------------------------

(define (call-with-output-captured thunk)
  "Run THUNK with current-output-port bound to a string port.

Args:
  thunk: the body.
Returns: two values, THUNK's result and the text written to the port.
Example:
  (use-modules (sonicboom trace compat))
  (call-with-values
      (lambda () (call-with-output-captured (lambda () (display \"hi\") 42)))
    list)
  => (42 \"hi\")
PyTorch: none"
  (let* ((port (open-output-string))
         (result (parameterize ((current-output-port port)) (thunk))))
    (values result (get-output-string port))))

(define (call-with-output-suppressed thunk)
  "Run THUNK with its output discarded.

Args:
  thunk: the body.
Returns: THUNK's result.
Example:
  (use-modules (sonicboom trace compat))
  (call-with-output-suppressed (lambda () 42))
  => 42
PyTorch: none"
  (parameterize ((current-output-port (%make-void-port "w"))) (thunk)))

(define (trace-warn message)
  "Write MESSAGE as one line to current-error-port.

Args:
  message: a string.
Returns: unspecified.
PyTorch: none"
  (let ((port (current-error-port)))
    (display message port)
    (newline port)))

;;; --- procedure tags ----------------------------------------------------------------------

(define (procedure-tag! proc key value)
  "Associate VALUE with KEY on PROC.

Args:
  proc: a procedure.
  key: the property key.
  value: the property value.
Returns: unspecified.
Example:
  (use-modules (sonicboom trace compat))
  (let ((proc (lambda () 1)))
    (procedure-tag! proc 'kind 'demo)
    (procedure-tag proc 'kind))
  => demo
PyTorch: none"
  (set-procedure-property! proc key value))
(define (procedure-tag proc key)
  "The value associated with KEY on PROC, or #f.

Args:
  proc: a procedure.
  key: the property key.
Returns: the value, or #f.
Example:
  (use-modules (sonicboom trace compat))
  (procedure-tag (lambda () 1) 'kind)
  => #f
PyTorch: none"
  (and (procedure? proc) (procedure-property proc key)))
(define (procedure-label proc)
  "A name for PROC in messages.

Args:
  proc: a procedure.
Returns: the procedure's name when it has one, otherwise PROC itself.
Example:
  (use-modules (sonicboom trace compat))
  (procedure-label car)
  => car
PyTorch: none"
  (or (and (procedure? proc) (procedure-name proc)) proc))

;;; --- tables ----------------------------------------------------------------------------------

(define (make-table)
  "Create an empty mutable table.

Keys are compared with `equal?`.

Returns: a table.
Example:
  (use-modules (sonicboom trace compat))
  (let ((table (make-table)))
    (table-set! table 'a 1)
    (table-ref table 'a))
  => 1
PyTorch: none"
  (make-hash-table))
(define (table-ref table key . default)
  "The value of KEY in TABLE, or DEFAULT.

Args:
  table: a table.
  key: the key.
  default: optional; the value when KEY is absent (default #f).
Returns: the value.
Example:
  (use-modules (sonicboom trace compat))
  (table-ref (make-table) 'missing 'dflt)
  => dflt
PyTorch: none"
  (hash-ref table key (if (pair? default) (car default) #f)))
(define (table-set! table key value)
  "Bind KEY to VALUE in TABLE.

Args:
  table: a table.
  key: the key.
  value: the value.
Returns: unspecified.
Example:
  (use-modules (sonicboom trace compat))
  (let ((table (make-table)))
    (table-set! table 'a 1)
    (table-ref table 'a))
  => 1
PyTorch: none"
  (hash-set! table key value))
(define (table-count table)
  "The number of entries in TABLE.

Args:
  table: a table.
Returns: a non-negative integer.
Example:
  (use-modules (sonicboom trace compat))
  (let ((table (make-table)))
    (table-set! table 'a 1)
    (table-set! table 'b 2)
    (table-count table))
  => 2
PyTorch: none"
  (hash-count (const #t) table))
(define (table->alist table)
  "The entries of TABLE as an alist.

Args:
  table: a table.
Returns: an alist of (key . value) pairs, in the table's order.
Example:
  (use-modules (sonicboom trace compat))
  (let ((table (make-table)))
    (table-set! table 'a 1)
    (table->alist table))
  => ((a . 1))
PyTorch: none"
  (hash-map->list cons table))

;;; --- procedures that never branch on a value -------------------------------------------
;;; Strict mode does not report symbolic tensors passed to these. Listed here
;;; because the procedure objects are implementation-specific (Guile has both
;;; core and SRFI-1 versions of map / for-each / reduce-style procedures).

;; The procedures in `%structural-procedures` that call a procedure.
;;
;; These are the map and for-each variants of the core and of SRFI-1.
;;
;; PyTorch: none
(define %higher-order-procedures
  (list apply map for-each (@ (srfi srfi-1) map) (@ (srfi srfi-1) for-each)))

;; The procedures the tracer treats as structure rather than computation.
;;
;; A symbolic tensor passed to one of these is not reported in strict mode,
;; because the procedure only moves or re-arranges values.
;;
;; PyTorch: none
(define %structural-procedures
  (append %higher-order-procedures
          (list list cons car cdr caar cadr cdar cddr caddr cdddr list-ref
                list-tail vector vector-ref append reverse length values)))

;;; --- source positions and error messages (validator) ------------------------------
;;; Guile's reader records line/column for every pair it reads. Portable
;;; fallback: plain `read`, and datum-position always #f (locations are then
;;; given only as document paths, e.g. "nodes[3]").

(define (read-located port)
  "Read one datum from PORT, recording the source position of its pairs.

Args:
  port: an input port.
Returns: the datum.
Example:
  (use-modules (sonicboom trace compat))
  (read-located (open-input-string \"(a b)\"))
  => (a b)
PyTorch: none"
  (read port))

(define (datum-position d)
  "The source position of DATUM.

Args:
  datum: a datum read with `read-located`.
Returns: a pair (line . column), both 1-based, or #f when no position was
recorded.
Example:
  (use-modules (sonicboom trace compat))
  (datum-position (read-located (open-input-string \"(a b)\")))
  => (1 . 1)
PyTorch: none"
  (and (pair? d)
       (let ((props (source-properties d)))
         (and (assq 'line props)
              (cons (+ 1 (cdr (assq 'line props))) (+ 1 (cdr (assq 'column props))))))))

(define (call-with-error-message thunk handler)
  "Run THUNK; on any error call HANDLER with the message text.

Args:
  thunk: the body.
  handler: a procedure of one string.
Returns: what THUNK or HANDLER returns.
Example:
  (use-modules (sonicboom trace compat))
  (call-with-error-message (lambda () (error \"boom\"))
                           (lambda (message) message))
  => \"boom\"
PyTorch: none"
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

;; A marker object standing for a guard name that is unbound at call time.
;;
;; `%free-variable-guards` uses it for a free name that is not bound when the
;; traced procedure is called, typically a name bound inside the body itself.
;;
;; PyTorch: none
(define %unbound-guard (list 'unbound))

(define (%global-guard module name)
  "The value of NAME in MODULE, or `%unbound-guard` when it is unbound.

Args:
  module: a module object.
  name: a symbol.
Returns: the variable's value, or the marker.
Example:
  (use-modules (sonicboom trace compat))
  (eq? (%global-guard (current-module) 'car) car)
  => #t
PyTorch: none"
  (let ((v (module-variable module name)))
    (if (and v (variable-bound? v)) (variable-ref v) %unbound-guard)))

;;; --- syntax hooks -------------------------------------------------------------------------
;;; These three are the only non-syntax-rules macros of the prototype. Each has
;;; a portable fallback, given in the porting guide.

;; The source location of FORM as a quoted alist.
;;
;; Expands to the alist (filename line column) of FORM, or to the quoted form
;; itself when the reader recorded no location.
;;
;; PyTorch: none
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

;; Turn a walked application back into a call, or leave a macro use untouched.
;;
;; The macro walker rewrites the arguments of a form it does not recognise and
;; hands both the original form and the rewritten one here. When the head names a
;; variable the result is a call of the rewritten arguments; when it names a macro
;; or core syntax the original form is returned untouched.
;;
;; PyTorch: none
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

;; An expression evaluating to the guard alist for the free names of BODY.
;;
;; The alist maps a name to a thunk. Outer lexical variables are captured when
;; the traced procedure is defined; module-level names are looked up in the module
;; when it is called.
;;
;; PyTorch: none
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
