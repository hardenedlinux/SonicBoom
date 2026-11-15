;;; SonicBoom — tracing prototype: dead-node elimination
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; One backward pass from the segment's outputs. A node is live iff it is
;;; effectful (e.g. `print`) or one of its outputs is live; a parameter is live
;;; iff a live node or a graph output references it.

(define-module (sonicboom trace dce)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom trace ops)
  #:export (eliminate-dead-nodes))

(define (eliminate-dead-nodes nodes params outputs)
  "Remove the nodes and parameters that do not reach OUTPUTS.

A node is live when it is effectful or when one of its outputs is live; a
parameter is live when a live node or a graph output references it.

Args:
  nodes: the segment's nodes in topological order.
  params: a list of gparams.
  outputs: the value names that must survive.
Returns: two values, the live nodes and the live parameters, both in their
original order.
Example:
  (use-modules (sonicboom trace graph) (sonicboom trace dce))
  (let ((a (make-gnode 'relu '(\"x\") '((\"a\" float32 (2))) '()))
        (b (make-gnode 'relu '(\"x\") '((\"b\" float32 (2))) '())))
    (call-with-values
        (lambda () (eliminate-dead-nodes (list a b) '() '(\"b\")))
      (lambda (nodes params)
        (map gnode-output-names nodes))))
  => ((\"b\"))
PyTorch: none"
  (let ((live (make-table)))
    (for-each (lambda (o) (table-set! live o #t)) outputs)
    (let ((kept
           (fold (lambda (n acc)
                   (if (or (op-effectful? (gnode-op n))
                           (any (lambda (o) (table-ref live o #f))
                                (gnode-output-names n)))
                       (begin
                         (for-each (lambda (i) (table-set! live i #t))
                                   (gnode-inputs n))
                         (cons n acc))
                       acc))
                 '()
                 (reverse nodes))))
      (values kept
              (filter (lambda (p) (table-ref live (gparam-name p) #f)) params)))))
