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

;; OUTPUTS is a list of (name dtype shape); ATTRS is the S-Expr attr list.
(define-record-type <gnode>
  (make-gnode op inputs outputs attrs)
  gnode?
  (op      gnode-op)
  (inputs  gnode-inputs)
  (outputs gnode-outputs)
  (attrs   gnode-attrs))

(define (gnode-output-names n) (map car (gnode-outputs n)))

(define-record-type <gparam>
  (make-gparam name dtype shape values)
  gparam?
  (name   gparam-name)
  (dtype  gparam-dtype)
  (shape  gparam-shape)
  (values gparam-values))   ; flat list, already coerced to dtype

(define (type->sx dtype shape) `(tensor ,dtype (shape ,@shape)))

(define (node-attrs n)
  ;; Emitted attrs: the node's own, plus the PyTorch Core ATen operator it
  ;; corresponds to, (aten (string "...")), when there is one. Kernels never
  ;; read this attribute.
  (let ((aten (op-aten-name (gnode-op n))))
    (if (and aten (not (assq 'aten (gnode-attrs n))))
        (append (gnode-attrs n) `((aten (string ,aten))))
        (gnode-attrs n))))

(define (graph->sexpr name inputs outputs params nodes)
  "Build the S-Expr v0.1 document datum. INPUTS is a list of (name dtype shape),
OUTPUTS a list of value names, PARAMS a list of gparams, NODES a list of gnodes
in topological order."
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
  "Render a document datum as text: one graph section per line group, one
input/output/parameter/node per line. `write` gives strings for names, symbols
for vocabulary, and floats with a decimal point or exponent."
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
