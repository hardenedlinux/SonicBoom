;;; SonicBoom — tracing prototype: reference S-Expr interpreter (pure Scheme)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Evaluates an S-Expr v0.1 document *text* on concrete tensors. It reads the
;;; text back with `read`, so every evaluation also checks that the emitted text
;;; round-trips. It enforces the parts of the spec the tracer relies on: version
;;; (0 1), single definition, topological order, declared input types, and that
;;; every node output type matches what the kernel computed. It is the default
;;; graph-break executor; any procedure with the same signature can replace it
;;; (see `current-executor` in (sonicboom trace tracer)).

(define-module (sonicboom trace interp)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace compat)
  #:use-module (ice-9 match)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace ops)
  #:export (interpret-sexpr interpret-sexpr-text interp-body-cache-stats))

;;; `loop` nodes (host loops, see (sonicboom trace loop)) carry their body as
;;; S-Expr text in a string attribute; it is parsed once per distinct text and
;;; then run trip_count times.
(define %body-cache (make-table))
(define %body-parses 0)
(define %body-hits 0)
(define (interp-body-cache-stats)
  "The parses and hits of the loop-body parse cache since start.

Returns: a pair (parses . hits).
Example:
  (use-modules (sonicboom trace interp))
  (pair? (interp-body-cache-stats))
  => #t
PyTorch: none"
  (cons %body-parses %body-hits))
(define (body-doc text)
  (let ((doc (table-ref %body-cache text #f)))
    (if doc
        (begin (set! %body-hits (+ 1 %body-hits)) doc)
        (let ((doc (call-with-input-string text read)))
          (set! %body-parses (+ 1 %body-parses))
          (table-set! %body-cache text doc)
          doc))))

(define (run-loop-node ins attrs)
  (let* ((k (attr-ref attrs 'state_count 0))
         (n (attr-ref attrs 'trip_count 0))
         (text (attr-ref attrs 'body #f))
         (invariants (map (lambda (x i) (cons (string-append "x" (number->string i)) x))
                          (drop ins k) (iota (- (length ins) k)))))
    (let iterate ((i 0) (state (take ins k)))
      (if (= i n)
          state
          (iterate (+ i 1)
                   (map cdr (interpret-sexpr
                             (body-doc text)
                             (append (map (lambda (x j) (cons (string-append "s" (number->string j)) x))
                                          state (iota k))
                                     invariants))))))))

(define (field graph key)
  (let ((f (assq key (cdr graph))))
    (if f (cdr f) (error "interp: graph is missing section" key))))

(define (parse-type t)
  (match t
    (('tensor dtype ('shape dims ...))
     (unless (valid-dtype? dtype) (error "interp: unknown dtype" dtype))
     (unless (every (lambda (d) (and (integer? d) (exact? d) (>= d 0))) dims)
       (error "interp: non-static dim" dims))
     (cons dtype dims))
    (_ (error "interp: bad type" t))))

(define (check-type who name t type)
  (unless (and (eq? (ctensor-dtype t) (car type))
               (equal? (ctensor-shape t) (cdr type)))
    (error "interp: type mismatch" who name
           (cons (ctensor-dtype t) (ctensor-shape t)) type)))

(define (interpret-sexpr doc inputs)
  "Evaluate a parsed S-Expr v0.1 document on concrete tensors.

It enforces the parts of the format the tracer relies on: version (0 1), a
single definition per value name, topological order, the declared input types,
and that every node output matches the type the kernel produced. `loop` nodes
run their body document TRIP-COUNT times.

Args:
  doc: the document datum.
  inputs: an alist (name . ctensor).
Returns: an alist (output-name . ctensor) in the graph's output order.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace interp))
  (ctensor->list
   (assoc-ref
    (interpret-sexpr
     '(sonicboom-s-expr
       (version 0 1)
       (graph
        (name \"g\")
        (inputs (input \"x\" (tensor float32 (shape 2))))
        (outputs (output \"y\"))
        (parameters)
        (nodes (node relu (inputs \"x\")
                     (outputs (\"y\" (tensor float32 (shape 2))))
                     (attrs)))))
     (list (cons \"x\" (list->ctensor '(-1.0 2.0)))))
    \"y\"))
  => (0.0 2.0)
PyTorch: none
Note: this is the reference interpreter for the S-Expr format; it reads the
text back and re-derives every declared type."
  (match doc
    (('sonicboom-s-expr ('version 0 1) ('graph . _)) #t)
    (_ (error "interp: not a sonicboom-s-expr v0.1 document")))
  (let ((graph (caddr doc))
        (env (make-table)))
    (define (define! name t)
      (when (table-ref env name #f) (error "interp: value defined twice" name))
      (table-set! env name t))
    (define (lookup name)
      (or (table-ref env name #f)
          (error "interp: value used before definition" name)))
    (for-each
     (match-lambda
       (('input name type)
        (let ((t (or (assoc-ref inputs name)
                     (error "interp: missing input" name))))
          (check-type 'input name t (parse-type type))
          (define! name t)))
       (x (error "interp: bad input form" x)))
     (field graph 'inputs))
    (for-each
     (match-lambda
       (('parameter name type ('data tag vals ...))
        (unless (eq? tag (string->symbol ":values"))
          (error "interp: only inline :values data is supported" tag))
        (let ((ty (parse-type type)))
          (define! name (make-ctensor (car ty) (cdr ty) vals))))
       (x (error "interp: bad parameter form" x)))
     (field graph 'parameters))
    (for-each
     (match-lambda
       (('node 'loop ('inputs ins ...) ('outputs (names types) ...) ('attrs attrs ...))
        (for-each (lambda (name type r)
                    (check-type 'loop name r (parse-type type))
                    (define! name r))
                  names types (run-loop-node (map lookup ins) attrs)))
       (('node op ('inputs ins ...) ('outputs (names types) ...) ('attrs attrs ...))
        (unless (op-known? op) (error "interp: unsupported op" op))
        (let ((r (op-kernel op (map lookup ins) attrs)))
          (if (= 1 (length names))
              (begin
                (check-type op (car names) r (parse-type (car types)))
                (define! (car names) r))
              ;; A multi-output node (the fused backward ops): the kernel
              ;; returns one ctensor per declared output, in order.
              (begin
                (unless (and (list? r) (= (length r) (length names)))
                  (error "interp: multi-output kernel returned the wrong arity" op
                         (if (list? r) (length r) 1) (length names)))
                (for-each (lambda (name type value)
                            (check-type op name value (parse-type type))
                            (define! name value))
                          names types r)))))
       (x (error "interp: bad node form" x)))
     (field graph 'nodes))
    (map (match-lambda
           (('output name) (cons name (lookup name)))
           (x (error "interp: bad output form" x)))
         (field graph 'outputs))))

;; Parsed documents of the last few texts, keyed by the string object (the
;; tracer hands the executor the same emitted string on every cached call).
(define %doc-cache '())
(define (parse-text text)
  (let ((hit (assq text %doc-cache)))
    (if hit
        (cdr hit)
        (let ((doc (call-with-input-string text read)))
          (set! %doc-cache (cons (cons text doc)
                                 (if (> (length %doc-cache) 31)
                                     (take %doc-cache 31)
                                     %doc-cache)))
          doc))))

(define (interpret-sexpr-text text inputs)
  "Parse TEXT and evaluate it.

TEXT is one S-Expr v0.1 document. Parsed documents are cached by the text
string, so a repeated call on the same text does not parse it again.

Args:
  text: the document text.
  inputs: an alist (name . ctensor).
Returns: an alist (output-name . ctensor); see `interpret-sexpr`.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace graph)
               (sonicboom trace interp))
  (let ((text (sexpr->text (graph->sexpr \"g\" '((\"x\" float32 (2)))
                                         '(\"x\") '() '()))))
    (ctensor->list
     (assoc-ref (interpret-sexpr-text text
                                     (list (cons \"x\" (list->ctensor '(1.0 2.0)))))
                \"x\")))
  => (1.0 2.0)
PyTorch: none"
  (interpret-sexpr (parse-text text) inputs))
