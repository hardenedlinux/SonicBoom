;;; SonicBoom — tracing prototype: concrete reference tensors
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Research prototype. Pure Scheme; no
;;; libsonicboom. A concrete tensor ("ctensor") is a dtype, a static shape and an
;;; owned <storage> over a flat row-major bytevector holding the elements in the
;;; dtype's native representation (native-endian, like libsonicboom's tensors):
;;; IEEE binary32 for float32, IEEE binary16 for float16, bfloat16 for bfloat16,
;;; IEEE binary64 for float64, two's-complement native integers, and one byte 0/1
;;; for bool. Element access converts to and from Scheme numbers (inexact reals
;;; for float dtypes, exact integers for integer dtypes, booleans for bool).
;;;
;;; The bytevector is the storage, not a serialised view: `ctensor-ref` and
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
  #:use-module (sonicboom storage)
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

(define (valid-dtype? d)
  "Whether DTYPE is one of the dtypes a ctensor accepts.

Args:
  dtype: a symbol.
Returns: #t for float32, float16, bfloat16, float64, int8, uint8, int16,
  int32, int64 and bool; #f otherwise.
Example:
  (use-modules (sonicboom trace tensor))
  (valid-dtype? 'bfloat16)
  => #t
PyTorch: none"
  (and (memq d %dtypes) #t))
(define (float-dtype? d)
  "Whether DTYPE is a floating-point dtype.

Args:
  dtype: a symbol.
Returns: #t for float32, float16, bfloat16 and float64; #f otherwise.
Example:
  (use-modules (sonicboom trace tensor))
  (float-dtype? 'float16)
  => #t
PyTorch: none"
  (and (memq d '(float32 float16 bfloat16 float64)) #t))
(define (int-dtype? d)
  "Whether DTYPE is an integer dtype.

Args:
  dtype: a symbol.
Returns: #t for int8, uint8, int16, int32 and int64; #f otherwise.
Example:
  (use-modules (sonicboom trace tensor))
  (int-dtype? 'int64)
  => #t
PyTorch: none"
  (and (memq d '(int8 uint8 int16 int32 int64)) #t))

(define (dtype-size d)
  "The number of bytes per element of DTYPE.

Args:
  dtype: a valid dtype.
Returns: 1, 2, 4 or 8.
Example:
  (use-modules (sonicboom trace tensor))
  (dtype-size 'float64)
  => 8
PyTorch: none"
  (or (assq-ref %dtype-sizes d)
      (error "dtype-size: unknown dtype" d)))

(define (shape-numel shape)
  "The number of elements of SHAPE.

Args:
  shape: a list of non-negative exact integers.
Returns: the product of the dimensions; 1 for the empty shape.
Example:
  (use-modules (sonicboom trace tensor))
  (shape-numel '(2 3 4))
  => 24
PyTorch: none"
  (fold * 1 shape))

(define (coerce-value dtype x)
  "Convert X to the Scheme representation used for DTYPE.

A float dtype accepts any real and returns it inexact; an integer dtype
accepts an integer value and returns it exact; bool accepts a boolean or a
real and returns a boolean.

Args:
  dtype: a valid dtype.
  x: the value to convert.
Returns: the converted Scheme value.
Example:
  (use-modules (sonicboom trace tensor))
  (coerce-value 'int32 2.0)
  => 2
PyTorch: none"
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

;; A concrete reference tensor: a dtype, a static shape and owned storage.
;;
;; The bytes are row-major and native-endian in the dtype's native
;; representation. ctensor? tests the type; ctensor-dtype and ctensor-shape read
;; the dtype and the shape; ctensor-data exposes the backing bytevector.
;;
;; PyTorch: torch.Tensor
;; Note: This is the tracing prototype's reference tensor, written in pure
;; Scheme. It models a tensor for the tracer at compile time; it is not the
;; runtime tensor of (sonicboom tensor).
(define-record-type <ctensor>
  (%make-ctensor dtype shape storage)
  ctensor?
  (dtype   ctensor-dtype)
  (shape   ctensor-shape)
  (storage ctensor-storage))   ; owned storage: row-major, native-endian bytes

(define (ctensor-data t)
  "The bytevector backing T.

The bytevector is the tensor's storage, not a copy: writing to it changes the
tensor.

Args:
  t: a ctensor.
Returns: a bytevector of (ctensor-nbytes T) bytes.
Example:
  (use-modules (sonicboom trace tensor))
  (bytevector-length (ctensor-data (make-ctensor 'float32 '(2) '(1.0 2.0))))
  => 8
PyTorch: none
Note: The result is the raw bytevector, not a NumPy array."
  (storage-data (ctensor-storage t)))

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
  "A tensor built from a dtype, a shape and flat data.

DATA is a flat list or vector of Scheme values, or a bytevector of packed
dtype values. A bytevector is copied; to share storage, use
bytevector->ctensor.

Args:
  dtype: the element dtype; see valid-dtype?.
  shape: a list of static non-negative exact integers.
  data: a flat list or vector with one value per element, or a bytevector of
    the packed dtype's size.
Returns: a new ctensor.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor->list (make-ctensor 'float32 '(2 2) '(1.0 2.0 3.0 4.0)))
  => (1.0 2.0 3.0 4.0)
PyTorch: torch.tensor
Note: The shape is given separately from the flat data; PyTorch infers it
from the nesting of its input."
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
    (%make-ctensor dtype shape (owned-storage bv))))

(define (bytevector->ctensor dtype shape bv)
  "A tensor that shares BV's storage.

BV must hold exactly one packed element per element of SHAPE in DTYPE. No
copy is made: writes through ctensor-set! are visible in BV, and writes to BV
are visible through the tensor.

Args:
  dtype: the element dtype.
  shape: a list of static non-negative exact integers.
  bv: a bytevector of (shape-numel SHAPE) times (dtype-size DTYPE) bytes.
Returns: a new ctensor sharing BV.
Example:
  (use-modules (sonicboom trace tensor) (rnrs bytevectors))
  (let* ((bv (make-bytevector 8 0))
         (t (bytevector->ctensor 'int32 '(2) bv)))
    (ctensor-set! t 0 7)
    (bytevector-s32-native-ref bv 0))
  => 7
PyTorch: torch.frombuffer
Note: torch.frombuffer also views a buffer without copying."
  (unless (valid-dtype? dtype)
    (error "bytevector->ctensor: unknown dtype" dtype))
  (check-shape shape)
  (unless (bytevector? bv)
    (error "bytevector->ctensor: expected a bytevector" bv))
  (unless (= (bytevector-length bv) (* (shape-numel shape) (dtype-size dtype)))
    (error "bytevector->ctensor: bytevector length does not match shape"
           shape (bytevector-length bv)))
  (%make-ctensor dtype shape (owned-storage bv)))

(define (ctensor-copy t)
  "An independent tensor with the same dtype, shape and bytes.

Args:
  t: a ctensor.
Returns: a new ctensor; changing one does not affect the other.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor->list (ctensor-copy (make-ctensor 'float32 '(2) '(1.0 2.0))))
  => (1.0 2.0)
PyTorch: torch.Tensor.clone"
  (%make-ctensor (ctensor-dtype t) (ctensor-shape t)
                 (owned-storage (bytevector-copy (ctensor-data t)))))

(define (ctensor-nbytes t)
  "The size of T's storage in bytes.

Args:
  t: a ctensor.
Returns: an exact integer.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor-nbytes (make-ctensor 'float32 '(2 3) '(0.0 0.0 0.0 0.0 0.0 0.0)))
  => 24
PyTorch: torch.Tensor.nbytes"
  (bytevector-length (ctensor-data t)))

(define (ctensor-numel t)
  "The number of elements in T.

Args:
  t: a ctensor.
Returns: an exact integer.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor-numel (make-ctensor 'float32 '(2 3) '(0.0 0.0 0.0 0.0 0.0 0.0)))
  => 6
PyTorch: torch.Tensor.numel"
  (/ (ctensor-nbytes t) (dtype-size (ctensor-dtype t))))

(define (ctensor-ref t i)
  "Element I of T as a Scheme value.

The index is flat and row-major. The value is an inexact real for a float
dtype, an exact integer for an integer dtype, and a boolean for bool.

Args:
  t: a ctensor.
  i: the flat element index, from 0 to (ctensor-numel T) - 1.
Returns: the element.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor-ref (make-ctensor 'float32 '(2) '(1.5 2.5)) 1)
  => 2.5
PyTorch: none
Note: PyTorch indexes with one subscript per axis; this index is flat."
  (check-index "ctensor-ref" t i)
  ;; Direct field read: ctensor-data is an exported procedure, and calling it
  ;; per element measurably slows ctensor->list and the interpreter's kernels.
  (unpack-value (storage-data (ctensor-storage t)) (ctensor-dtype t) i))

(define (ctensor-set! t i v)
  "Store V as element I of T.

The value is converted to T's dtype: floats round to the dtype's precision
(nearest, ties to even) and integers must be within the dtype's range.

Args:
  t: a ctensor.
  i: the flat element index.
  v: the new value.
Returns: an unspecified value.
Example:
  (use-modules (sonicboom trace tensor))
  (let ((t (make-ctensor 'float32 '(2) '(1.0 2.0))))
    (ctensor-set! t 0 5.0)
    (ctensor->list t))
  => (5.0 2.0)
PyTorch: none
Note: The write is direct; PyTorch item assignment is subscript-based and
records autograd history."
  (check-index "ctensor-set!" t i)
  (pack-value! (storage-data (ctensor-storage t)) (ctensor-dtype t) i v))

(define (ctensor->list t)
  "All elements of T as a flat Scheme list in row-major order.

Args:
  t: a ctensor.
Returns: a list with (ctensor-numel T) elements.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor->list (make-ctensor 'float32 '(2) '(1.0 2.0)))
  => (1.0 2.0)
PyTorch: torch.Tensor.tolist"
  (let ((n (ctensor-numel t)))
    (let loop ((i (- n 1)) (acc '()))
      (if (< i 0) acc (loop (- i 1) (cons (ctensor-ref t i) acc))))))

(define (ctensor-write t port)
  "Write T's values to PORT as nested lists in shape order.

The values are streamed from storage; no whole-tensor list is built.

Args:
  t: a ctensor.
  port: an output port.
Returns: an unspecified value.
Example:
  (use-modules (sonicboom trace tensor))
  (call-with-output-string
    (lambda (p)
      (ctensor-write (make-ctensor 'float32 '(2 2) '(1.0 2.0 3.0 4.0)) p)))
  => \"((1.0 2.0) (3.0 4.0))\"
PyTorch: none"
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
  "The elements of T regrouped into nested lists following its shape.

A rank-0 tensor gives its single element rather than an empty list.

Args:
  t: a ctensor.
Returns: a nested list, or the single element of a rank-0 tensor.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor->nested (make-ctensor 'float32 '(2 2) '(1.0 2.0 3.0 4.0)))
  => ((1.0 2.0) (3.0 4.0))
PyTorch: torch.Tensor.tolist"
  (let loop ((shape (ctensor-shape t)) (xs (ctensor->list t)))
    (if (null? shape)
        (car xs)
        (let ((inner (shape-numel (cdr shape))))
          (map (lambda (i) (loop (cdr shape) (take (drop xs (* i inner)) inner)))
               (iota (car shape)))))))

(define* (list->ctensor nested #:optional (dtype #f))
  "A tensor built from nested lists.

The nesting gives the shape; a bare number or boolean gives a rank-0 tensor.

Args:
  nested: a rectangular nested list of numbers or booleans.
  dtype: optional; the default is bool when the elements are booleans and
    float32 otherwise.
Returns: a new ctensor.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor->nested (list->ctensor '((1 2) (3 4))))
  => ((1.0 2.0) (3.0 4.0))
PyTorch: torch.tensor"
  (define (shape-of x)
    (if (pair? x) (cons (length x) (shape-of (car x))) '()))
  (define (flatten x) (if (pair? x) (append-map flatten x) (list x)))
  (let* ((flat (flatten nested))
         (dt (or dtype (if (and (pair? flat) (boolean? (car flat))) 'bool
                           'float32))))
    (make-ctensor dt (shape-of nested) flat)))

(define* (scalar->ctensor x #:optional (dtype #f))
  "A rank-0 tensor holding X.

Args:
  x: a number or a boolean.
  dtype: optional; the default is bool for a boolean and float32 otherwise.
Returns: a rank-0 ctensor.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor-item (scalar->ctensor 3.0))
  => 3.0
PyTorch: torch.tensor"
  (make-ctensor (or dtype (if (boolean? x) 'bool 'float32)) '() (list x)))

(define (ctensor-item t)
  "The single element of a one-element tensor.

T may have any shape as long as it has exactly one element.

Args:
  t: a ctensor with one element.
Returns: the element.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor-item (make-ctensor 'float32 '(1 1) '(7.0)))
  => 7.0
PyTorch: torch.Tensor.item"
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

;; A tensor value and its gradient slot.
;;
;; A leaf is what a traced training step differentiates. leaf-value holds the
;; tensor (a ctensor, or the symbolic tensor the tracer substitutes for it) and
;; leaf-grad holds its gradient, if any. leaf? tests the type; leaf-value and
;; leaf-grad read the slots and set-leaf-value! and set-leaf-grad! replace them.
;; The object keeps its identity across calls while the slots change.
;;
;; PyTorch: torch.nn.Parameter
;; Note: The nearest PyTorch notion is a leaf tensor with requires_grad; here
;; the value and the gradient are two slots of an explicit object.
(define-record-type <leaf>
  (%make-leaf value grad)
  leaf?
  (value leaf-value set-leaf-value!)
  (grad  leaf-grad  set-leaf-grad!))

(define* (make-leaf value #:optional (grad #f))
  "A leaf with VALUE and an optional gradient.

The value is a ctensor, or a symbolic tensor while the tracer runs.

Args:
  value: the tensor value.
  grad: optional; the gradient, initially absent.
Returns: a new leaf.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor->list (leaf-value (make-leaf (make-ctensor 'float32 '(2) '(1.0 2.0)))))
  => (1.0 2.0)
PyTorch: none
Note: A PyTorch parameter is a tensor that owns its gradient; a leaf keeps
the value and the gradient in separate slots."
  (%make-leaf value grad))

(define* (make-lr-tensor #:optional (device 'cpu) #:key (value 1e-3))
  "A one-element float32 learning-rate tensor for a traced training step.

Passed to a training step as #:lr, the tensor becomes a graph input rather
than an attribute, so a schedule can change the rate without retracing. Read
the rate with leaf-value and replace it with set-leaf-value!.

Args:
  device: optional; 'cpu (the default) or 'cuda, accepted for call-site
    symmetry with device tensors.
  value: keyword; the initial rate (default 1e-3).
Returns: a leaf holding a rank-0 float32 tensor.
Example:
  (use-modules (sonicboom trace tensor))
  (ctensor-item (leaf-value (make-lr-tensor 'cpu #:value 0.5)))
  => 0.5
PyTorch: none
Note: PyTorch passes the learning rate as a number to the optimizer."
  (unless (memq device '(cpu cuda))
    (error "make-lr-tensor: unknown device" device))
  (make-leaf (scalar->ctensor value 'float32)))
