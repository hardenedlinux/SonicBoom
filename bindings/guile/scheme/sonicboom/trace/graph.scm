;;; SonicBoom — tracing prototype: recorded graph segments and S-Expr v0.1 output
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A traced segment is a topologically ordered list of nodes plus the inline
;;; constant parameters they use. This module defines those records and turns a
;;; segment into an S-Expr v0.1 document (docs/s-expr-v0-spec.md): every value
;;; typed at its definition site, nodes in producer-before-consumer order, all
;;; user names as strings, constants as `(data :values ...)`.

(define-module (sonicboom trace graph)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace ops)
  #:export (make-gnode gnode? gnode-op gnode-inputs gnode-outputs gnode-attrs
            gnode-output-names
            make-gparam gparam? gparam-name gparam-dtype gparam-shape
            gparam-values
            type->sx graph->sexpr sexpr->text))

;; A recorded graph node: an operator, its inputs, its outputs and its attributes.
;;
;; OUTPUTS is a list of (name dtype shape) entries; ATTRS is the S-Expr attribute
;; list. A segment is a list of nodes in topological order.
;;
;; PyTorch: none
(define-record-type <gnode>
  (make-gnode op inputs outputs attrs)
  gnode?
  (op      gnode-op)
  (inputs  gnode-inputs)
  (outputs gnode-outputs)
  (attrs   gnode-attrs))

(define (gnode-output-names n)
  "The output value names of node N.

Args:
  n: a graph node.
Returns: a list of strings, one per output.
Example:
  (use-modules (sonicboom trace graph))
  (gnode-output-names (make-gnode 'relu '(\"x\") '((\"y\" float32 (2))) '()))
  => (\"y\")
PyTorch: none"
  (map car (gnode-outputs n)))

;; An inline graph constant: its name, dtype, shape and flat values.
;;
;; The value list is stored already coerced to DTYPE. The S-Expr emission writes
;; it as (data :values ...).
;;
;; PyTorch: none
(define-record-type <gparam>
  (make-gparam name dtype shape values)
  gparam?
  (name   gparam-name)
  (dtype  gparam-dtype)
  (shape  gparam-shape)
  (values gparam-values))   ; flat list, already coerced to dtype

(define (type->sx dtype shape)
  "The S-Expr type form of DTYPE and SHAPE.

Args:
  dtype: a dtype symbol.
  shape: a list of dimensions.
Returns: (tensor <dtype> (shape <dims> ...)).
Example:
  (use-modules (sonicboom trace graph))
  (type->sx 'float32 '(2 3))
  => (tensor float32 (shape 2 3))
PyTorch: none"
  `(tensor ,dtype (shape ,@shape)))

(define (node-attrs n)
  ;; Emitted attrs: the node's own, plus the PyTorch Core ATen operator it
  ;; corresponds to, (aten (string "...")), when there is one. Kernels never
  ;; read this attribute.
  (let ((aten (op-aten-name (gnode-op n))))
    (if (and aten (not (assq 'aten (gnode-attrs n))))
        (append (gnode-attrs n) `((aten (string ,aten))))
        (gnode-attrs n))))

(define (graph->sexpr name inputs outputs params nodes)
  "Build the S-Expr v0.1 document datum of one segment.

INPUTS is a list of (name dtype shape), OUTPUTS a list of value names, PARAMS
a list of gparams and NODES a list of gnodes in topological order. Every node
whose operator has an ATen counterpart gains an (aten (string ...)) attribute.

Args:
  name: the graph name.
  inputs: a list of (name dtype shape).
  outputs: a list of output value names.
  params: a list of gparams.
  nodes: a list of gnodes in topological order.
Returns: the document datum, ready for `sexpr->text`.
Example:
  (use-modules (sonicboom trace graph))
  (car (graph->sexpr \"g\" '((\"x\" float32 (2))) '(\"y\") '()
                     (list (make-gnode 'relu '(\"x\") '((\"y\" float32 (2))) '()))))
  => sonicboom-s-expr
PyTorch: none"
  `(sonicboom-s-expr
    (version 0 1)
    (graph
     (name ,name)
     (inputs ,@(map (lambda (i) `(input ,(car i) ,(type->sx (cadr i) (caddr i))))
                    inputs))
     (outputs ,@(map (lambda (o) `(output ,o)) outputs))
     (parameters
      ,@(map (lambda (p)
               `(parameter ,(gparam-name p)
                           ,(type->sx (gparam-dtype p) (gparam-shape p))
                           (data ,(string->symbol ":values") ,@(gparam-values p))))
             params))
     (nodes
      ,@(map (lambda (n)
               `(node ,(gnode-op n)
                      (inputs ,@(gnode-inputs n))
                      (outputs ,@(map (lambda (o)
                                        (list (car o) (type->sx (cadr o) (caddr o))))
                                      (gnode-outputs n)))
                      (attrs ,@(node-attrs n))))
             nodes)))))

(define (sexpr->text doc)
  "Render a document datum as text.

One graph section per line group, one input, output, parameter or node per
line. Strings are written as strings, vocabulary as symbols and floats with a
decimal point or an exponent.

Args:
  doc: an S-Expr document datum.
Returns: the document text.
Example:
  (use-modules (sonicboom trace graph))
  (string? (sexpr->text
            (graph->sexpr \"g\" '((\"x\" float32 (2))) '(\"y\") '()
                          (list (make-gnode 'relu '(\"x\")
                                            '((\"y\" float32 (2))) '())))))
  => #t
PyTorch: none"
  (define (w x) (let ((p (open-output-string))) (write x p) (get-output-string p)))
  (define (section head items)
    (if (null? items)
        (string-append "    (" (symbol->string head) ")")
        (string-append "    (" (symbol->string head) "\n"
                       (string-join (map (lambda (i) (string-append "      " (w i)))
                                         items)
                                    "\n")
                       ")")))
  (let* ((graph (caddr doc))
         (fields (cdr graph)))
    (string-append
     "(sonicboom-s-expr\n  " (w (cadr doc)) "\n  (graph\n"
     (string-join
      (map (lambda (f)
             (if (eq? (car f) 'name)
                 (string-append "    " (w f))
                 (section (car f) (cdr f))))
           fields)
      "\n")
     "))\n")))
