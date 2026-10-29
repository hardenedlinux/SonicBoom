;;; SonicBoom — tracing prototype: concrete reference tensors
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Research prototype. Pure Scheme; no
;;; libsonicboom. A concrete tensor ("ctensor") is a dtype, a static shape and a
;;; flat row-major bytevector holding the elements in the dtype's native
;;; representation (native-endian, like libsonicboom's tensors): IEEE binary32
;;; for float32, IEEE binary16 for float16, bfloat16 for bfloat16, IEEE binary64
;;; for float64, two's-complement native integers, and one byte 0/1 for bool.
;;; Element access converts to and from Scheme numbers (inexact reals for float
;;; dtypes, exact integers for integer dtypes, booleans for bool).
;;;
;;; A bytevector is the storage, not a serialised view: `ctensor-ref` and
;;; `ctensor-set!` read and write it in place, and `bytevector->ctensor` shares
;;; it (reshape/view aliasing). Materialising a whole tensor into Scheme values
;;; is explicit: `ctensor->list` / `ctensor->nested` / `ctensor-write` are the
;;; only such interfaces, and printing streams from the storage.
;;;
;;; It exists only so that the reference interpreter and eager mode have
;;; something to compute on; it is not a SonicBoom runtime tensor.

(define-module (sonicboom trace tensor)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (rnrs bytevectors)
  #:export (make-ctensor ctensor? ctensor-dtype ctensor-shape ctensor-data
            bytevector->ctensor ctensor-copy ctensor-nbytes dtype-size
            ctensor-numel ctensor-ref ctensor-set! ctensor->list ctensor->nested
            ctensor-write
            list->ctensor scalar->ctensor ctensor-item
            make-leaf leaf? leaf-value set-leaf-value! leaf-grad set-leaf-grad!
            make-lr-tensor
            shape-numel float-dtype? int-dtype? valid-dtype?
            coerce-value))

(define %dtypes
  '(float32 float16 bfloat16 float64 int8 uint8 int16 int32 int64 bool))

(define %dtype-sizes
  '((float32 . 4) (float16 . 2) (bfloat16 . 2) (float64 . 8)
    (int8 . 1) (uint8 . 1) (int16 . 2) (int32 . 4) (int64 . 8) (bool . 1)))

(define (valid-dtype? d) (and (memq d %dtypes) #t))
(define (float-dtype? d) (and (memq d '(float32 float16 bfloat16 float64)) #t))
(define (int-dtype? d) (and (memq d '(int8 uint8 int16 int32 int64)) #t))

(define (dtype-size d)
  "Bytes per element of DTYPE's native representation."
  (or (assq-ref %dtype-sizes d)
      (error "dtype-size: unknown dtype" d)))

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

;;; --- IEEE-754 binary16 / bfloat16 ------------------------------------------------
;;;
;;; float32 and float64 are read and written by the bytevector primitives.
;;; binary16 and bfloat16 are packed by hand. Both round to nearest, ties to
;;; even. A float16 overflow becomes an infinity; a NaN keeps the sign and
;;; becomes a quiet NaN (payload bits are not preserved).

(define (f32-bits->f16-bits u)
  (let* ((sign (logand (ash u -16) #x8000))
         (exponent (logand (ash u -23) #xff))
         (mantissa (logand u #x7fffff))
         (exp (- exponent 127 -15)))
    (cond
     ((= exponent #xff)                       ; inf / NaN
      (logior sign #x7c00 (if (zero? mantissa) 0 #x200)))
     ((>= exp 31)                             ; overflow to infinity
      (logior sign #x7c00))
     ((<= exp 0)
      (if (< exp -10)
          sign
          (let* ((m (logior mantissa #x800000))
                 (shift (- 14 exp))
                 (r (ash m (- shift)))
                 (rem (logand m (1- (ash 1 shift))))
                 (half (ash 1 (1- shift))))
            (when (or (> rem half) (and (= rem half) (odd? r)))
              (set! r (1+ r)))
            (logior sign r))))
     (else
      (let* ((r (ash mantissa -13))
             (rem (logand mantissa #x1fff)))
        (when (or (> rem #x1000)
                  (and (= rem #x1000) (odd? r)))
          (set! r (1+ r))
          (when (= r #x400)
            (set! r 0)
            (set! exp (1+ exp))))
        (if (>= exp 31)
            (logior sign #x7c00)
            (logior sign (ash exp 10) r)))))))

(define (f16-bits->value h)
  (let* ((sign (if (zero? (logand h #x8000)) 1.0 -1.0))
         (exponent (logand (ash h -10) #x1f))
         (mantissa (logand h #x3ff)))
    (cond
     ((= exponent 0)
      (if (zero? mantissa)
          (* sign 0.0)
          (* sign mantissa (expt 2.0 -24))))
     ((= exponent #x1f)
      (if (zero? mantissa) (* sign +inf.0) +nan.0))
     (else
      (* sign (+ 1.0 (/ mantissa 1024.0)) (expt 2.0 (- exponent 15)))))))

(define (value->f32-bits x)
  (let ((bv (make-bytevector 4 0)))
    (bytevector-ieee-single-native-set! bv 0 (exact->inexact x))
    (bytevector-u32-native-ref bv 0)))

(define (value->f16-bits x) (f32-bits->f16-bits (value->f32-bits x)))

(define (f32-bits->bf16-bits u)
  (if (= #xff (logand (ash u -23) #xff))
      (ash u -16)             ; inf / NaN: keep sign+exp+top mantissa
      (let* ((lsb (logand (ash u -16) 1))
             (bias (+ #x7fff lsb)))
        (logand (ash (+ u bias) -16) #xffff))))

(define (bf16-bits->value h)
  (let ((bv (make-bytevector 4 0)))
    (bytevector-u32-native-set! bv 0 (ash h 16))
    (bytevector-ieee-single-native-ref bv 0)))

;;; --- storage ---------------------------------------------------------------------

(define-record-type <ctensor>
  (%make-ctensor dtype shape data)
  ctensor?
  (dtype ctensor-dtype)
  (shape ctensor-shape)
  (data  ctensor-data))     ; bytevector, row-major, native-endian

(define (check-shape shape)
  (unless (every (lambda (d) (and (integer? d) (exact? d) (>= d 0))) shape)
    (error "ctensor: shape must be static non-negative ints" shape)))

(define (check-index who t i)
  (unless (and (integer? i) (exact? i) (>= i 0) (< i (ctensor-numel t)))
    (error who "index out of range" i (ctensor-shape t)))
  i)

(define %int-ranges
  '((int8 . (-128 127)) (uint8 . (0 255)) (int16 . (-32768 32767))
    (int32 . (-2147483648 2147483647))
    (int64 . (-9223372036854775808 9223372036854775807))))

(define (check-int-range d x)
  (let ((range (assq-ref %int-ranges d)))
    (unless (and (>= x (car range)) (<= x (cadr range)))
      (error "ctensor-set!: integer out of range for dtype" d x)))
  x)

(define (pack-value! bv d i v)
  "Write V (a Scheme value for DTYPE D) at element I of BV. Bounds and integer
range are the caller's responsibility."
  (let ((v (coerce-value d v)))
    (case d
      ((float32) (bytevector-ieee-single-native-set! bv (* i 4) v))
      ((float64) (bytevector-ieee-double-native-set! bv (* i 8) v))
      ((float16) (bytevector-u16-native-set! bv (* i 2) (value->f16-bits v)))
      ((bfloat16) (bytevector-u16-native-set! bv (* i 2)
                                             (f32-bits->bf16-bits (value->f32-bits v))))
      ((int8) (bytevector-s8-set! bv i (check-int-range d v)))
      ((uint8) (bytevector-u8-set! bv i (check-int-range d v)))
      ((int16) (bytevector-s16-native-set! bv (* i 2) (check-int-range d v)))
      ((int32) (bytevector-s32-native-set! bv (* i 4) (check-int-range d v)))
      ((int64) (bytevector-s64-native-set! bv (* i 8) (check-int-range d v)))
      ((bool) (bytevector-u8-set! bv i (if v 1 0)))
      (else (error "pack-value!: unknown dtype" d)))))

(define (unpack-value bv d i)
  "Read element I of BV as a Scheme value for DTYPE D."
  (case d
    ((float32) (bytevector-ieee-single-native-ref bv (* i 4)))
    ((float64) (bytevector-ieee-double-native-ref bv (* i 8)))
    ((float16) (f16-bits->value (bytevector-u16-native-ref bv (* i 2))))
    ((bfloat16) (bf16-bits->value (bytevector-u16-native-ref bv (* i 2))))
    ((int8) (bytevector-s8-ref bv i))
    ((uint8) (bytevector-u8-ref bv i))
    ((int16) (bytevector-s16-native-ref bv (* i 2)))
    ((int32) (bytevector-s32-native-ref bv (* i 4)))
    ((int64) (bytevector-s64-native-ref bv (* i 8)))
    ((bool) (not (zero? (bytevector-u8-ref bv i))))
    (else (error "unpack-value: unknown dtype" d))))

(define (make-ctensor dtype shape data)
  "DATA is a flat list or vector of Scheme values, or a bytevector of packed
DTYPE values. Either way the new ctensor owns a fresh bytevector: a bytevector
argument is copied, not wrapped (use `bytevector->ctensor` to share storage)."
  (unless (valid-dtype? dtype) (error "make-ctensor: unknown dtype" dtype))
  (check-shape shape)
  (let* ((n (shape-numel shape))
         (sz (dtype-size dtype))
         (bv (cond ((bytevector? data)
                    (unless (= (bytevector-length data) (* n sz))
                      (error "make-ctensor: bytevector length does not match shape"
                             shape (bytevector-length data)))
                    (bytevector-copy data))
                   (else
                    (let ((xs (if (vector? data) (vector->list data) data)))
                      (unless (= (length xs) n)
                        (error "make-ctensor: data length does not match shape" shape
                               (length xs)))
                      (let ((bv (make-bytevector (* n sz) 0)))
                        (let loop ((i 0) (xs xs))
                          (unless (null? xs)
                            (pack-value! bv dtype i (car xs))
                            (loop (+ i 1) (cdr xs))))
                        bv))))))
    (%make-ctensor dtype shape bv)))

(define (bytevector->ctensor dtype shape bv)
  "Wrap BV, a bytevector of packed DTYPE values (no copy), as a ctensor. The
ctensor and BV share storage: writes through `ctensor-set!` are visible in BV
and vice versa, which is what reshape/view aliasing relies on."
  (unless (valid-dtype? dtype)
    (error "bytevector->ctensor: unknown dtype" dtype))
  (check-shape shape)
  (unless (bytevector? bv)
    (error "bytevector->ctensor: expected a bytevector" bv))
  (unless (= (bytevector-length bv) (* (shape-numel shape) (dtype-size dtype)))
    (error "bytevector->ctensor: bytevector length does not match shape"
           shape (bytevector-length bv)))
  (%make-ctensor dtype shape bv))

(define (ctensor-copy t)
  "An independent ctensor with the same dtype, shape and bytes."
  (%make-ctensor (ctensor-dtype t) (ctensor-shape t)
                 (bytevector-copy (ctensor-data t))))

(define (ctensor-nbytes t) (bytevector-length (ctensor-data t)))

(define (ctensor-numel t) (/ (ctensor-nbytes t) (dtype-size (ctensor-dtype t))))

(define (ctensor-ref t i)
  "Element I of T as a Scheme value (inexact real, exact integer or boolean)."
  (check-index "ctensor-ref" t i)
  (unpack-value (ctensor-data t) (ctensor-dtype t) i))

(define (ctensor-set! t i v)
  "Store V as element I of T, converting to T's dtype. Floats round to the
dtype's precision (nearest, ties to even); integers must be in range."
  (check-index "ctensor-set!" t i)
  (pack-value! (ctensor-data t) (ctensor-dtype t) i v))

(define (ctensor->list t)
  "All elements of T as a flat Scheme list. This materialises the whole tensor;
it is an explicit boundary (parameter emission, display, tests), never part of
a shape/dtype query or of printing."
  (let ((n (ctensor-numel t)))
    (let loop ((i (- n 1)) (acc '()))
      (if (< i 0) acc (loop (- i 1) (cons (ctensor-ref t i) acc))))))

(define (ctensor-write t port)
  "Write T's values as nested lists in shape order, streaming from storage (no
whole-tensor list is built)."
  (define (emit shape i)
    ;; Writes the group of SHAPE starting at flat index I; returns the next
    ;; flat index.
    (if (null? shape)
        (begin (write (ctensor-ref t i) port) (+ i 1))
        (let ((dim (car shape)))
          (display "(" port)
          (let loop ((k 0) (i i))
            (if (= k dim)
                (begin (display ")" port) i)
                (begin
                  (when (> k 0) (display " " port))
                  (loop (+ k 1) (emit (cdr shape) i))))))))
  (emit (ctensor-shape t) 0))

(define (ctensor->nested t)
  "Flat data regrouped as nested lists following the shape (a rank-0 tensor
gives its single element). Explicit whole-tensor materialisation."
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
