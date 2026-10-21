;;; SonicBoom — tracing prototype: S-Expr v0.2 draft validator
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Checks an S-Expr document against docs/s-expr-v0.2-draft.md:
;;;
;;;   - structure: sole top-level form, version (0 1) or (0 2), one graph,
;;;     known sections only, well-formed inputs / outputs / parameters / nodes;
;;;   - types: dtype enum, static non-negative dims, (tensor ...) everywhere a
;;;     value is defined; inline :values count and kinds match the type;
;;;   - names: strings, single definition, every reference defined before use
;;;     (topological order), outputs defined and not repeated;
;;;   - nodes: known op (core vocabulary, plus the optional `loop` and
;;;     `print`), arity, attribute tags, and the output type re-derived from
;;;     the input types and attributes by the op's type rule must equal the
;;;     declared one;
;;;   - aten: a node whose op has a Core ATen counterpart carries
;;;     (aten (string "<name>")) equal to the mapping table entry; ops
;;;     without one carry none;
;;;   - loop: the body text is itself a valid document whose inputs are s0..
;;;     (the state, typed as the node's first state_count inputs) and x.. (the
;;;     remaining inputs) and whose outputs have the state types.
;;;
;;; Every error carries a location: line:column in the text when the reader
;;; recorded one (compat datum-position), and a document path such as
;;; nodes[12] (node add).
;;;
;;; The validator only reads ops.scm (type rules, arities, ATen names); it
;;; does not execute anything.

(define-module (sonicboom trace validate)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace ops)
  #:export (validate-sexpr-text validate-sexpr valid-sexpr-text?))

;;; --- error collection ---------------------------------------------------------------

(define (where-string datum path)
  (let ((pos (datum-position datum)))
    (string-append (if pos
                       (string-append "line " (number->string (car pos))
                                      ", column " (number->string (cdr pos)) ": ")
                       "")
                   path)))

(define (w x) (let ((p (open-output-string))) (write x p) (get-output-string p)))

(define (fmt . args)
  (let ((p (open-output-string)))
    (for-each (lambda (a) (if (string? a) (display a p) (write a p))) args)
    (get-output-string p)))

;;; --- small predicates ----------------------------------------------------------------

(define (dims? x) (and (list? x) (every (lambda (d) (and (integer? d) (exact? d) (>= d 0))) x)))

(define (tagged? x tag) (and (pair? x) (eq? (car x) tag)))

(define %core-ops
  '(add sub mul div neg relu gelu exp log tanh sqrt matmul bmm permute reshape expand
    reduce_sum reduce_mean reduce_max greater less equal and or not where
    index_select index_add gather scatter_add concat slice))
(define %optional-ops '(loop print))

(define %attr-tags '(int float string bool ints floats strings bools))

;; Attributes each op may carry, with their tags (draft section 8); `aten`
;; is checked separately.
(define %attr-table
  '((reduce_sum (axes . ints) (keepdims . int))
    (reduce_mean (axes . ints) (keepdims . int))
    (reduce_max (axes . ints) (keepdims . int))
    (reshape (shape . ints)) (expand (shape . ints)) (permute (dims . ints))
    (index_select (axis . int)) (index_add (axis . int)) (gather (axis . int))
    (scatter_add (axis . int)) (concat (axis . int))
    (slice (axis . int) (start . int) (end . int))
    (print (label . string))
    (loop (body . string) (trip_count . int) (state_count . int))))

(define (attr-value-ok? tv)
  (and (pair? tv) (memq (car tv) %attr-tags) (list? tv)
       (case (car tv)
         ((int) (and (= 2 (length tv)) (integer? (cadr tv)) (exact? (cadr tv))))
         ((float) (and (= 2 (length tv)) (real? (cadr tv))))
         ((string) (and (= 2 (length tv)) (string? (cadr tv))))
         ((bool) (and (= 2 (length tv)) (boolean? (cadr tv))))
         ((ints) (every (lambda (x) (and (integer? x) (exact? x))) (cdr tv)))
         ((floats) (every real? (cdr tv)))
         ((strings) (every string? (cdr tv)))
         ((bools) (every boolean? (cdr tv))))))

(define (value-fits? dtype x)
  (cond ((eq? dtype 'bool) (boolean? x))
        ((float-dtype? dtype) (real? x))
        ((int-dtype? dtype) (and (integer? x) (exact? x)))
        (else #f)))

;;; --- the validator ---------------------------------------------------------------------

(define* (validate-sexpr doc #:key (require-aten? #t) (path-prefix ""))
  "Errors (list of strings) of document datum DOC; '() if it is valid."
  (let ((errors '())
        (types (make-table)))           ; value name -> (dtype . shape)
    (define (err! datum path . msg)
      (set! errors (cons (string-append (where-string datum (string-append path-prefix path))
                                        ": " (apply fmt msg))
                         errors)))
    (define (parse-type t datum path)
      ;; (dtype . shape) or #f after reporting
      (if (and (list? t) (= 3 (length t)) (eq? (car t) 'tensor)
               (tagged? (caddr t) 'shape))
          (let ((dtype (cadr t)) (dims (cdr (caddr t))))
            (cond ((not (valid-dtype? dtype))
                   (err! datum path "unknown dtype " dtype) #f)
                  ((not (dims? dims))
                   (err! datum path "dims must be static non-negative integers: " dims) #f)
                  (else (cons dtype dims))))
          (begin (err! datum path "malformed type " t) #f)))
    (define (define-name! name type datum path)
      (cond ((not (string? name)) (err! datum path "value names must be strings: " (w name)))
            ((table-ref types name #f) (err! datum path "value " (w name) " is defined twice"))
            (else (table-set! types name type))))

    (define (validate-top)
    (cond
     ((not (and (list? doc) (= 3 (length doc)) (eq? (car doc) 'sonicboom-s-expr)))
      (err! doc "document" "expected (sonicboom-s-expr (version ...) (graph ...))"))
     ((not (member (cadr doc) '((version 0 1) (version 0 2))))
      (err! (cadr doc) "version" "unsupported version " (cadr doc)
            " (this reader accepts (version 0 1) and (version 0 2))"))
     ((not (and (list? (caddr doc)) (eq? (car (caddr doc)) 'graph)))
      (err! (caddr doc) "graph" "expected exactly one (graph ...)"))
     (else
      (let* ((graph (caddr doc))
             (sections (cdr graph))
             (section (lambda (key) (let ((s (assq key sections))) (and s (cdr s))))))
        ;; known sections, each at most once
        (for-each (lambda (s)
                    (cond ((not (and (pair? s) (memq (car s) '(name opset inputs outputs
                                                               parameters nodes))))
                           (err! s "graph" "unknown graph section " (if (pair? s) (car s) s)))
                          ((and (not (eq? (car s) 'opset))
                                (> (count (lambda (t) (and (pair? t) (eq? (car t) (car s))))
                                          sections)
                                   1))
                           (err! s "graph" "section " (car s) " appears more than once"))))
                  sections)
        (for-each (lambda (key)
                    (unless (assq key sections)
                      (err! graph "graph" "missing section " key)))
                  '(inputs outputs parameters nodes))
        ;; inputs
        (for-each
         (lambda (in i)
           (let ((path (fmt "inputs[" i "]")))
             (if (and (list? in) (= 3 (length in)) (eq? (car in) 'input))
                 (let ((t (parse-type (caddr in) in path)))
                   (when t (define-name! (cadr in) t in path)))
                 (err! in path "expected (input <name> <type>)"))))
         (or (section 'inputs) '()) (iota (length (or (section 'inputs) '()))))
        ;; parameters
        (for-each
         (lambda (p i)
           (let ((path (fmt "parameters[" i "]")))
             (if (and (list? p) (= 4 (length p)) (eq? (car p) 'parameter)
                      (tagged? (cadddr p) 'data))
                 (let ((t (parse-type (caddr p) p path))
                       (data (cadddr p)))
                   (when t
                     (define-name! (cadr p) t p path)
                     (let ((layout (and (pair? (cdr data)) (cadr data))))
                       (cond
                        ((eq? layout (string->symbol ":values"))
                         (let ((vals (cddr data)) (n (shape-numel (cdr t))))
                           (unless (= n (length vals))
                             (err! p path "parameter " (w (cadr p)) " has " (length vals)
                                   " values for " n " elements"))
                           (unless (every (lambda (v) (value-fits? (car t) v)) vals)
                             (err! p path "parameter " (w (cadr p)) " values do not fit dtype "
                                   (car t)))))
                        ((eq? layout (string->symbol ":external"))
                         (unless (and (= 5 (length data)) (string? (caddr data))
                                      (nat? (cadddr data)) (nat? (list-ref data 4)))
                           (err! p path "expected (data :external <file> <offset> <length>)")))
                        (else (err! p path "unknown data layout " layout))))))
                 (err! p path "expected (parameter <name> <type> (data ...))"))))
         (or (section 'parameters) '()) (iota (length (or (section 'parameters) '()))))
        ;; nodes, in order
        (for-each
         (lambda (n i)
           (validate-node n i))
         (or (section 'nodes) '()) (iota (length (or (section 'nodes) '()))))
        ;; outputs
        (let loop ((outs (or (section 'outputs) '())) (i 0) (seen '()))
          (unless (null? outs)
            (let* ((o (car outs)) (path (fmt "outputs[" i "]")))
              (cond ((not (and (list? o) (= 2 (length o)) (eq? (car o) 'output)))
                     (err! o path "expected (output <name>)"))
                    ((not (table-ref types (cadr o) #f))
                     (err! o path "output " (w (cadr o)) " is not defined"))
                    ((member (cadr o) seen)
                     (err! o path "output " (w (cadr o)) " is listed twice")))
              (loop (cdr outs) (+ i 1) (if (pair? o) (cons (cadr o) seen) seen)))))))))

    (define (validate-node n i)
      (let ((path (fmt "nodes[" i "]")))
        (if (not (and (list? n) (>= (length n) 5) (eq? (car n) 'node) (symbol? (cadr n))
                      (tagged? (list-ref n 2) 'inputs) (tagged? (list-ref n 3) 'outputs)
                      (tagged? (list-ref n 4) 'attrs)))
            (err! n path "expected (node <op> (inputs ...) (outputs ...) (attrs ...))")
            (let* ((op (cadr n))
                   (path (fmt path " (node " op ")"))
                   (ins (cdr (list-ref n 2)))
                   (outs (cdr (list-ref n 3)))
                   (attrs (cdr (list-ref n 4)))
                   (in-types (map (lambda (name)
                                    (or (table-ref types name #f)
                                        (begin (err! n path "input " (w name)
                                                     " is used before it is defined"
                                                     " (topological order) or never defined")
                                               #f)))
                                  ins))
                   (out-types (map (lambda (o)
                                     (if (and (list? o) (= 2 (length o)))
                                         (parse-type (cadr o) n path)
                                         (begin (err! n path "malformed output " o) #f)))
                                   outs)))
              ;; extra node fields (domain / version) are v0.1-legal
              (for-each (lambda (f)
                          (unless (and (pair? f) (memq (car f) '(domain version)))
                            (err! n path "unknown node field " f)))
                        (drop n 5))
              ;; attributes: well-formed, allowed for the op, with the right tag
              (for-each (lambda (a)
                          (cond
                           ((not (and (list? a) (= 2 (length a)) (symbol? (car a))
                                      (attr-value-ok? (cadr a))))
                            (err! n path "malformed attribute " a))
                           ((eq? (car a) 'aten) #t)
                           (else
                            (let ((spec (assq (car a) (or (assq-ref %attr-table op) '()))))
                              (cond ((not spec)
                                     (err! n path "attribute " (car a) " is not defined for op " op))
                                    ((not (eq? (cdr spec) (car (cadr a))))
                                     (err! n path "attribute " (car a) " must be tagged "
                                           (cdr spec) ", got " (car (cadr a)))))))))
                        attrs)
              (cond
               ((not (or (memq op %core-ops) (memq op %optional-ops)))
                (err! n path "unknown op " op))
               ((not (every identity in-types)) #f)          ; already reported
               ((not (every identity out-types)) #f)
               ((eq? op 'loop) (validate-loop n path in-types out-types attrs))
               (else
                (unless (= 1 (length outs))
                  (err! n path "op " op " has exactly one output, got " (length outs)))
                (let ((expected #f))
                  (call-with-error-message
                   (lambda () (set! expected (op-type op in-types (strip-aten attrs))))
                   (lambda (msg) (err! n path "type rule rejects the inputs: " msg)))
                  (when (and expected (= 1 (length out-types))
                             (not (equal? expected (car out-types))))
                    (err! n path "declared output type " (type->datum (car out-types))
                          " but the inputs give " (type->datum expected))))))
              (check-aten n path op attrs)
              ;; define outputs after checking inputs (no self reference)
              (for-each (lambda (o t) (when (and t (pair? o)) (define-name! (car o) t n path)))
                        outs out-types)))))

    (define (check-aten n path op attrs)
      (let ((a (assq 'aten attrs))
            (expected (and (op-known? op) (op-aten-name op))))
        (cond
         ((and expected (not a))
          (when require-aten?
            (err! n path "missing (aten (string " (w expected) ")) attribute")))
         ((and a (not expected))
          (err! n path "op " op " has no ATen counterpart in the mapping table, "
                "but carries " a))
         ((and a (not (equal? (cadr a) (list 'string expected))))
          (err! n path "aten attribute " (cadr a) " does not match the mapping table entry "
                (w expected))))))

    (define (validate-loop n path in-types out-types attrs)
      (let* ((k (attr-int attrs 'state_count))
             (trips (attr-int attrs 'trip_count))
             (body (let ((b (assq 'body attrs))) (and b (tagged? (cadr b) 'string)
                                                      (cadr (cadr b))))))
        (cond
         ((not (and k trips body))
          (err! n path "loop needs (body (string ...)), (trip_count (int ...)) and "
                "(state_count (int ...))"))
         ((or (< trips 0) (< k 0) (> k (length in-types)))
          (err! n path "bad trip_count / state_count " trips " / " k))
         ((not (equal? (take in-types k) out-types))
          (err! n path "loop outputs must have the types of the state inputs"))
         (else
          (let* ((doc #f)
                 (_ (call-with-error-message
                     (lambda () (set! doc (read-located (open-input-string body))))
                     (lambda (msg) (err! n path "loop body text does not parse: " msg)))))
            (when doc
              (for-each (lambda (e) (set! errors (cons e errors)))
                        (validate-sexpr doc #:require-aten? require-aten?
                                        #:path-prefix (string-append path-prefix path
                                                                     " body > ")))
              (let ((sec (lambda (key) (let ((g (and (pair? doc) (pair? (cddr doc))
                                                     (caddr doc))))
                                        (and (pair? g) (assq key (cdr g)))))))
                (let ((body-ins (and (sec 'inputs) (cdr (sec 'inputs))))
                      (body-outs (and (sec 'outputs) (cdr (sec 'outputs)))))
                  (when body-ins
                    (let ((want (append
                                 (map (lambda (t j) (list (string-append "s" (number->string j)) t))
                                      (take in-types k) (iota k))
                                 (map (lambda (t j) (list (string-append "x" (number->string j)) t))
                                      (drop in-types k) (iota (- (length in-types) k)))))
                          (have (filter-map (lambda (b)
                                              (and (list? b) (= 3 (length b))
                                                   (list (cadr b)
                                                         (let ((t (caddr b)))
                                                           (and (list? t) (= 3 (length t))
                                                                (cons (cadr t) (cdr (caddr t))))))))
                                            body-ins)))
                      (unless (equal? want have)
                        (err! n path "loop body inputs must be s0..s" (- k 1)
                              " (state) then x0.. (invariants), typed as the node inputs"))))
                  (when (and body-outs (not (= k (length body-outs))))
                    (err! n path "loop body must output the " k " state values"))))))))))

    (validate-top)
    (reverse errors)))

(define (nat? x) (and (integer? x) (exact? x) (>= x 0)))

(define (strip-aten attrs) (filter (lambda (a) (not (and (pair? a) (eq? (car a) 'aten)))) attrs))

(define (attr-int attrs key)
  (let ((a (assq key attrs)))
    (and a (tagged? (cadr a) 'int) (cadr (cadr a)))))

(define (type->datum t) `(tensor ,(car t) (shape ,@(cdr t))))

(define* (validate-sexpr-text text #:key (require-aten? #t))
  "Parse TEXT and validate it. Returns a list of error strings ('() if valid)."
  (let ((doc #f) (errs '()))
    (call-with-error-message
     (lambda ()
       (let ((port (open-input-string text)))
         (set! doc (read-located port))
         (unless (eof-object? (read-located port))
           (set! errs (list "document: more than one top-level form")))))
     (lambda (msg) (set! errs (list (string-append "document: does not parse: " msg)))))
    (if (and doc (null? errs))
        (validate-sexpr doc #:require-aten? require-aten?)
        errs)))

(define* (valid-sexpr-text? text #:key (require-aten? #t))
  (null? (validate-sexpr-text text #:require-aten? require-aten?)))
