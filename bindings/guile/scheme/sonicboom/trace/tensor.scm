;;; SonicBoom — tracing prototype: concrete reference tensors
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Research prototype (see design/guile-tracing-prototype.md). Pure Scheme; no
;;; libsonicboom. A concrete tensor ("ctensor") is a dtype, a static shape and a
;;; flat row-major vector of Scheme values: inexact reals for float dtypes, exact
;;; integers for integer dtypes, booleans for `bool`. It exists only so that the
;;; reference interpreter and eager mode have something to compute on; it is not
;;; a SonicBoom runtime tensor.

(define-module (sonicboom trace tensor)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:export (make-ctensor ctensor? ctensor-dtype ctensor-shape ctensor-data
            ctensor-numel ctensor-ref ctensor->list ctensor->nested
            list->ctensor scalar->ctensor ctensor-item
            make-leaf leaf? leaf-value set-leaf-value! leaf-grad set-leaf-grad!
            make-lr-tensor
            shape-numel float-dtype? int-dtype? valid-dtype?
            coerce-value))

(define %dtypes
  '(float32 float16 bfloat16 float64 int8 uint8 int16 int32 int64 bool))

(define (valid-dtype? d) (and (memq d %dtypes) #t))
(define (float-dtype? d) (and (memq d '(float32 float16 bfloat16 float64)) #t))
(define (int-dtype? d) (and (memq d '(int8 uint8 int16 int32 int64)) #t))

(define (vmap f v) (list->vector (map f (vector->list v))))

(define (shape-numel shape) (fold * 1 shape))

(define (coerce-value dtype x)
  "Convert Scheme value X to the representation used for DTYPE."
  (cond ((float-dtype? dtype)
         (if (real? x) (exact->inexact x)
             (error "coerce-value: not a real number" dtype x)))
        ((int-dtype? dtype)
         (if (and (real? x) (integer? x)) (inexact->exact x)
             (error "coerce-value: not an integer" dtype x)))
        ((eq? dtype 'bool)
         (cond ((boolean? x) x)
               ((real? x) (not (zero? x)))
               (else (error "coerce-value: not a boolean" x))))
        (else (error "coerce-value: unknown dtype" dtype))))

(define-record-type <ctensor>
  (%make-ctensor dtype shape data)
  ctensor?
  (dtype ctensor-dtype)
  (shape ctensor-shape)
  (data  ctensor-data))     ; vector, row-major

(define (make-ctensor dtype shape data)
  "DATA is a flat list or vector of length (product SHAPE)."
  (unless (valid-dtype? dtype) (error "make-ctensor: unknown dtype" dtype))
  (unless (every (lambda (d) (and (integer? d) (exact? d) (>= d 0))) shape)
    (error "make-ctensor: shape must be static non-negative ints" shape))
  (let ((v (if (vector? data) data (list->vector data))))
    (unless (= (vector-length v) (shape-numel shape))
      (error "make-ctensor: data length does not match shape" shape
             (vector-length v)))
    (%make-ctensor dtype shape
                   (vmap (lambda (x) (coerce-value dtype x)) v))))

(define (ctensor-numel t) (vector-length (ctensor-data t)))
(define (ctensor-ref t i) (vector-ref (ctensor-data t) i))
(define (ctensor->list t) (vector->list (ctensor-data t)))

(define (ctensor->nested t)
  "Flat data regrouped as nested lists following the shape (a rank-0 tensor
gives its single element)."
  (let loop ((shape (ctensor-shape t)) (xs (ctensor->list t)))
    (if (null? shape)
        (car xs)
        (let ((inner (shape-numel (cdr shape))))
          (map (lambda (i) (loop (cdr shape) (take (drop xs (* i inner)) inner)))
               (iota (car shape)))))))

(define* (list->ctensor nested #:optional (dtype #f))
  "Build a ctensor from nested lists (rectangular). A bare number or boolean
gives a rank-0 tensor. The dtype defaults to bool for booleans, float32
otherwise."
  (define (shape-of x)
    (if (pair? x) (cons (length x) (shape-of (car x))) '()))
  (define (flatten x) (if (pair? x) (append-map flatten x) (list x)))
  (let* ((flat (flatten nested))
         (dt (or dtype (if (and (pair? flat) (boolean? (car flat))) 'bool
                           'float32))))
    (make-ctensor dt (shape-of nested) flat)))

(define* (scalar->ctensor x #:optional (dtype #f))
  (make-ctensor (or dtype (if (boolean? x) 'bool 'float32)) '() (list x)))

(define (ctensor-item t)
  "The single element of a one-element tensor."
  (unless (= (ctensor-numel t) 1)
    (error "ctensor-item: tensor has more than one element" (ctensor-shape t)))
  (ctensor-ref t 0))

;;; --- leaf tensors (parameters) ------------------------------------------------------
;;
;;; A leaf is what training differentiates: a tensor value with a gradient slot.
;;; It is deliberately not the value itself, so that the tracer can replace the
;;; value by a symbolic input while the object the caller holds stays the same
;;; identity across calls. During tracing a leaf contains a symbolic tensor and
;;; its GRAD is the symbolic adjoint recorded by `backward!`; after a traced
;;; call the tracer writes the concrete gradient of the executed graph back into
;;; the caller's leaf (see (sonicboom trace tracer)). Ops accept a leaf anywhere
;;; they accept a tensor and use its value.

(define-record-type <leaf>
  (%make-leaf value grad)
  leaf?
  (value leaf-value set-leaf-value!)
  (grad  leaf-grad  set-leaf-grad!))

(define* (make-leaf value #:optional (grad #f))
  "A leaf tensor: VALUE (a ctensor, or the symbolic tensor the tracer substitutes
for it) plus a GRAD slot."
  (%make-leaf value grad))

(define* (make-lr-tensor #:optional (device 'cpu) #:key (value 1e-3))
  "A one-element float32 learning-rate tensor for a traced training step.

Passed to `adamw-step!` as #:lr, it becomes a graph input instead of an
attribute, so a schedule can change the rate every step without retracing.
Returns a leaf: replace the rate between steps with set-leaf-value!.

DEVICE names where the tensor lives ('cpu or 'cuda). The tracing executor
transfers the tensor's current value on every run, so the tensor is held on
the host today; the argument is accepted for call-site symmetry with device
tensors and for a future device-resident path."
  (unless (memq device '(cpu cuda))
    (error "make-lr-tensor: unknown device" device))
  (make-leaf (scalar->ctensor value 'float32)))
