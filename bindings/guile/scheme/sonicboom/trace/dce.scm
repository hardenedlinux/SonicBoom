;;; SonicBoom — tracing prototype: dead-node elimination
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; One backward pass from the segment's outputs. Every op in the prototype is
;;; pure, so a node is live iff one of its outputs is live; a parameter is live
;;; iff a live node or a graph output references it.

(define-module (sonicboom trace dce)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace graph)
  #:export (eliminate-dead-nodes))

(define (eliminate-dead-nodes nodes params outputs)
  "NODES in topological order, PARAMS a list of gparams, OUTPUTS the value
names that must survive. Returns (values live-nodes live-params), both in
their original order."
  (let ((live (make-hash-table)))
    (for-each (lambda (o) (hash-set! live o #t)) outputs)
    (let ((kept
           (fold (lambda (n acc)
                   (if (any (lambda (o) (hash-ref live o #f))
                            (gnode-output-names n))
                       (begin
                         (for-each (lambda (i) (hash-set! live i #t))
                                   (gnode-inputs n))
                         (cons n acc))
                       acc))
                 '()
                 (reverse nodes))))
      (values kept
              (filter (lambda (p) (hash-ref live (gparam-name p) #f)) params)))))
