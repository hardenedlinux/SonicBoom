;;; SonicBoom — Guile binding: tensor & parameter types
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The Guile-side foundation of the SonicBoom AI framework: a <tensor> is a
;;; frozen dtype + a static shape + owned storage over a raw little-endian
;;; bytevector; a <parameter> is a tensor with `requires-grad` set (and an
;;; optional `.grad` slot) — mirroring the PyTorch mental model. Tensors are pure
;;; Guile data here: no native C++ object backs them, and no arithmetic is
;;; performed in Scheme loops. When a model runs, its tensors are marshalled
;;; across the SonicBoom C ABI into the native backends (see the execution steps).
;;;
;;; Representation matches the frozen S-Expr v0.1 type system
;;; (docs/s-expr-v0-spec.md): ten dtype names, static non-negative shape dims.

(define-module (sonicboom tensor)
  #:use-module (rnrs bytevectors)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-27)
  #:use-module (ice-9 optargs)
  #:use-module (ice-9 match)
  #:use-module (sonicboom storage)
  #:export (<tensor> tensor?
            make-tensor %make-tensor bytevector->tensor tensor->bytevector
            tensor-dtype tensor-shape tensor-data tensor-numel
            tensor-ref tensor-set! tensor->list list->tensor tensor-copy
            tensor-release!
            zeros ones rand randn
            parameter parameter? tensor-requires-grad?
            tensor-grad set-tensor-grad! parameter-grad parameter-grad-set!
            dtype-size valid-dtype? valid-shape?))

;;; --- dtype vocabulary -------------------------------------------------------

(define *dtype-sizes*
  '((float32 . 4)  (float16 . 2)  (bfloat16 . 2) (float64 . 8)
    (int8 . 1)     (uint8 . 1)    (int16 . 2)    (int32 . 4)
    (int64 . 8)    (bool . 1)))

(define (valid-dtype? d)
  "Whether a value names a supported dtype.

Args:
  d: any value.
Returns: #t for one of the ten v0 dtype names, #f otherwise.
Example:
  (use-modules (sonicboom tensor))
  (list (valid-dtype? 'float32) (valid-dtype? 'complex64))
  => (#t #f)
PyTorch: none"
  (and (assq d *dtype-sizes*) #t))

(define (dtype-size d)
  "The size in bytes of one element of a dtype.

Args:
  d: a dtype name.
Returns: the element size in bytes.
Example:
  (use-modules (sonicboom tensor))
  (dtype-size 'float32)
  => 4
PyTorch: none
Note: In PyTorch the size comes from torch.finfo or torch.iinfo for a dtype."
  (or (assq-ref *dtype-sizes* d)
      (error "dtype-size: unknown dtype" d)))

(define (valid-shape? s)
  "Whether a value is a well-formed shape.

Args:
  s: any value.
Returns: #t for a proper list of non-negative integers.
Example:
  (use-modules (sonicboom tensor))
  (list (valid-shape? '(2 3)) (valid-shape? '(2 -1)))
  => (#t #f)
PyTorch: none"
  (and (list? s)
       (every (lambda (dim) (and (integer? dim) (>= dim 0))) s)))

(define (numel s)
  (fold * 1 s))

;; Seed the default random source once at load time so `rand`/`randn` are
;; non-deterministic across runs.
(random-source-randomize! default-random-source)

;;; --- record -----------------------------------------------------------------

;; The tensor record: a dtype, a static shape, storage and an optional gradient.
;;
;; A tensor is a frozen dtype symbol, a list of non-negative shape dimensions, and
;; little-endian storage. `requires-grad` marks a trainable tensor (see
;; `parameter`); `grad` holds its gradient, or #f when none has been set. Tensors
;; are pure Scheme data: constructing one allocates a bytevector, and arithmetic
;; happens only when a graph is executed by the runtime.
;; PyTorch: torch.Tensor
;; Note: There is no automatic dtype promotion and no device placement on the
;; Scheme side; both are decided by the executing graph.
(define-record-type <tensor>
  (%make-tensor-record dtype shape storage requires-grad grad)
  tensor?
  (dtype        tensor-dtype)
  (shape        tensor-shape)
  (storage      tensor-storage)
  (requires-grad tensor-requires-grad? set-tensor-requires-grad!)
  (grad         tensor-grad set-tensor-grad!))

;; %make-tensor is exported, and took a bytevector before storage existed. Keep
;; that contract: a bytevector argument is wrapped as owned storage (not copied);
;; a <storage> argument is used as given.
(define (%make-tensor dtype shape data requires-grad grad)
  "Build a tensor from storage that is already allocated.

This is the low-level constructor under the others. A bytevector is wrapped as
owned storage without copying; a storage object is used as given.
Args:
  dtype: a dtype name.
  shape: a list of non-negative integers.
  data: a bytevector or a storage object backing the tensor.
  requires-grad: mark the tensor as a parameter.
  grad: the gradient tensor or #f.
Returns: a new tensor.
Example:
  (use-modules (sonicboom tensor) (rnrs bytevectors))
  (tensor->list (%make-tensor 'float32 '(2) (make-bytevector 8 0) #f #f))
  => (0.0 0.0)
PyTorch: none
Note: Prefer make-tensor or the initializers; this constructor performs no
validation of the byte length."
  (%make-tensor-record dtype shape
                       (if (storage? data) data (owned-storage data))
                       requires-grad grad))

(define (tensor-data t)
  "The bytevector backing T: a field read, never a copy.

For an owned tensor it is T's storage; for a read-only borrowed view (an
executable parameter) it aliases native memory. Read-only is enforced by
tensor-set!, not by the bytevector: Guile bytevectors are always mutable, so
writing through this bytevector would modify the parameter bytes and must not
be done.
Args:
  t: a tensor.
Returns: the tensor's bytevector.
Example:
  (use-modules (sonicboom tensor))
  (bytevector-length (tensor-data (make-tensor '(1.0 2.0) '(2))))
  => 8
PyTorch: none
Note: In PyTorch the nearest operation is .numpy().tobytes(), which copies;
tensor-data never copies."
  (storage-data (tensor-storage t)))

;;; --- element access ---------------------------------------------------------

(define (tensor-byte-offset t i)
  (* i (dtype-size (tensor-dtype t))))

(define (tensor-ref t i)
  "Read one element by its flat index.

Args:
  t: a tensor whose dtype has element access: float32, float64, int8, uint8,
     int16, int32, int64 or bool.
  i: a zero-based flat index in row-major order.
Returns: the element as a Scheme number or boolean.
Example:
  (use-modules (sonicboom tensor))
  (tensor-ref (make-tensor '(10 20 30) '(3)) 1)
  => 20.0
PyTorch: none
Note: PyTorch reaches elements through indexing and item; float16 and bfloat16
element access are not available in this binding."
  ;; The element accessors read the field directly: tensor-data is an exported
  ;; procedure, and calling it per element measurably slows tensor->list.
  (let ((d (tensor-dtype t))
        (bv (storage-data (tensor-storage t))))
    (case d
      ((float32) (bytevector-ieee-single-native-ref bv (* i 4)))
      ((float64) (bytevector-ieee-double-native-ref bv (* i 8)))
      ((int8)    (bytevector-s8-ref bv i))
      ((uint8)   (bytevector-u8-ref bv i))
      ((int16)   (bytevector-s16-native-ref bv (* i 2)))
      ((int32)   (bytevector-s32-native-ref bv (* i 4)))
      ((int64)   (bytevector-s64-native-ref bv (* i 8)))
      ((bool)    (not (zero? (bytevector-u8-ref bv i))))
      ((float16 bfloat16)
       (error "tensor-ref: element access for" d "is not supported in v0"))
      (else (error "tensor-ref: unknown dtype" d)))))

(define (tensor-set! t i v)
  "Write one element by its flat index, in place.

Args:
  t: a writable tensor whose dtype has element access.
  i: a zero-based flat index in row-major order.
  v: the new element.
Returns: unspecified.
Example:
  (use-modules (sonicboom tensor))
  (let ((t (make-tensor '(1 2 3) '(3))))
    (tensor-set! t 1 99)
    (tensor->list t))
  => (1.0 99.0 3.0)
PyTorch: none
Note: Writing to a read-only borrowed view raises an error instead of touching
native read-only memory."
  ;; A borrowed view may be native read-only memory (an executable parameter);
  ;; reject the write before touching it.
  (unless (storage-writable? (tensor-storage t))
    (error "tensor-set!: storage is read-only" (tensor-shape t)))
  (let ((d (tensor-dtype t))
        (bv (storage-data (tensor-storage t))))
    (case d
      ((float32) (bytevector-ieee-single-native-set! bv (* i 4) v))
      ((float64) (bytevector-ieee-double-native-set! bv (* i 8) v))
      ((int8)    (bytevector-s8-set! bv i v))
      ((uint8)   (bytevector-u8-set! bv i v))
      ((int16)   (bytevector-s16-native-set! bv (* i 2) v))
      ((int32)   (bytevector-s32-native-set! bv (* i 4) v))
      ((int64)   (bytevector-s64-native-set! bv (* i 8) v))
      ((bool)    (bytevector-u8-set! bv i (if v 1 0)))
      ((float16 bfloat16)
       (error "tensor-set!: element access for" d "is not supported in v0"))
      (else (error "tensor-set!: unknown dtype" d)))))

;;; --- constructors -----------------------------------------------------------

(define* (make-tensor data shape #:key (dtype 'float32) (requires-grad #f))
  "Build a tensor from a flat list of numbers.

Args:
  data: the elements in row-major order; its length must be the product of shape.
  shape: a list of non-negative integers.
  dtype: a dtype name such as float32 or int64; float32 by default.
  requires-grad: mark the tensor as a parameter; #f by default.
Returns: a new tensor owning a fresh bytevector.
Example:
  (use-modules (sonicboom tensor))
  (tensor->list (make-tensor '(1.0 2.0 3.0 4.0) '(2 2)))
  => (1.0 2.0 3.0 4.0)
PyTorch: torch.tensor
Note: The shape is given separately instead of being inferred from the data."
  (unless (valid-dtype? dtype)
    (error "make-tensor: invalid dtype" dtype))
  (unless (valid-shape? shape)
    (error "make-tensor: invalid shape" shape))
  (unless (= (length data) (numel shape))
    (error "make-tensor: data length" (length data)
           "does not match shape" shape))
  (let ((bv (make-bytevector (* (numel shape) (dtype-size dtype)) 0)))
    (let loop ((i 0) (xs data))
      (unless (null? xs)
        (tensor-set!-raw bv dtype i (car xs))
        (loop (+ i 1) (cdr xs))))
    (%make-tensor dtype shape (owned-storage bv) requires-grad #f)))

;;; Raw element write into an owned bytevector (no <tensor> allocation), used by
;;; the constructor before the record exists.
(define (tensor-set!-raw bv d i v)
  (case d
    ((float32) (bytevector-ieee-single-native-set! bv (* i 4) v))
    ((float64) (bytevector-ieee-double-native-set! bv (* i 8) v))
    ((int8)    (bytevector-s8-set! bv i v))
    ((uint8)   (bytevector-u8-set! bv i v))
    ((int16)   (bytevector-s16-native-set! bv (* i 2) v))
    ((int32)   (bytevector-s32-native-set! bv (* i 4) v))
    ((int64)   (bytevector-s64-native-set! bv (* i 8) v))
    ((bool)    (bytevector-u8-set! bv i (if v 1 0)))
    (else (error "tensor-set!-raw: unknown dtype" d))))

(define* (bytevector->tensor bv shape #:key (dtype 'float32) (requires-grad #f))
  "Wrap an existing bytevector as a tensor without copying it.

Args:
  bv: little-endian bytes; its length must equal the element count times the dtype size.
  shape: a list of non-negative integers.
  dtype: a dtype name; float32 by default.
  requires-grad: mark the tensor as a parameter; #f by default.
Returns: a tensor sharing the caller's bytevector.
Example:
  (use-modules (sonicboom tensor) (rnrs bytevectors))
  (define bv (make-bytevector 12 0))
  (eq? (tensor->bytevector (bytevector->tensor bv '(3))) bv)
  => #t
PyTorch: torch.frombuffer
Note: There is no dtype conversion: the bytes must already have the dtype's
element size."
  (unless (valid-dtype? dtype)
    (error "bytevector->tensor: invalid dtype" dtype))
  (unless (valid-shape? shape)
    (error "bytevector->tensor: invalid shape" shape))
  (unless (= (bytevector-length bv) (* (numel shape) (dtype-size dtype)))
    (error "bytevector->tensor: bytevector length mismatch"))
  (%make-tensor dtype shape (owned-storage bv) requires-grad #f))

(define (tensor->bytevector t)
  "The bytevector backing a tensor; a field read, never a copy.

Args:
  t: a tensor.
Returns: the tensor's bytevector.
Example:
  (use-modules (sonicboom tensor))
  (bytevector-length (tensor->bytevector (make-tensor '(1.0 2.0) '(2))))
  => 8
PyTorch: none
Note: PyTorch exposes no equivalent accessor; tensor->list is the portable
conversion."
  (tensor-data t))

;;; --- initializers -----------------------------------------------------------

(define* (zeros shape #:key (dtype 'float32))
  "A tensor filled with zeros.

Args:
  shape: a list of non-negative integers.
  dtype: a dtype name; float32 by default.
Returns: a new tensor.
Example:
  (use-modules (sonicboom tensor))
  (tensor->list (zeros '(2 2)))
  => (0.0 0.0 0.0 0.0)
PyTorch: torch.zeros"
  (unless (valid-shape? shape)
    (error "zeros: invalid shape" shape))
  (%make-tensor dtype shape
                (owned-storage
                 (make-bytevector (* (numel shape) (dtype-size dtype)) 0))
                #f #f))

(define* (ones shape #:key (dtype 'float32))
  "A tensor filled with ones.

Args:
  shape: a list of non-negative integers.
  dtype: a dtype name; float32 by default.
Returns: a new tensor.
Example:
  (use-modules (sonicboom tensor))
  (tensor->list (ones '(2 2)))
  => (1.0 1.0 1.0 1.0)
PyTorch: torch.ones"
  (let ((t (zeros shape #:dtype dtype)))
    (let ((n (numel shape)))
      (let loop ((i 0))
        (when (< i n)
          (tensor-set! t i 1)
          (loop (+ i 1)))))
    t))

(define (rand)
  "A uniform real in [0, 1) from the default random source.

Returns: a Scheme real.
Example:
  (use-modules (sonicboom tensor))
  (let ((x (rand))) (and (>= x 0.0) (< x 1.0)))
  => #t
PyTorch: none
Note: This returns a number, not a tensor; torch.rand(()) would return a 0-d
tensor."
  (random-real))

(define* (randn shape #:key (dtype 'float32))
  "A tensor of independent standard-normal samples.

The samples are drawn with the Box-Muller transform from the default random
source.
Args:
  shape: a list of non-negative integers.
  dtype: a dtype name; float32 by default.
Returns: a new tensor.
Example:
  (use-modules (sonicboom tensor))
  (tensor-numel (randn '(4)))
  => 4
PyTorch: torch.randn"
  (unless (valid-shape? shape)
    (error "randn: invalid shape" shape))
  (let* ((n (numel shape))
         (t (zeros shape #:dtype dtype)))
    (let loop ((i 0))
      (when (< i n)
        (let* ((u1 (max 1e-12 (random-real)))
               (u2 (random-real))
               (r (sqrt (* -2.0 (log u1))))
               (theta (* 2.0 (acos -1.0) u2)))
          (tensor-set! t i (* r (cos theta))))
        (loop (+ i 1))))
    t))

;;; --- introspection & conversion --------------------------------------------

(define (tensor-numel t)
  "The number of elements in a tensor.

Args:
  t: a tensor.
Returns: the product of the shape dimensions.
Example:
  (use-modules (sonicboom tensor))
  (tensor-numel (make-tensor '(1 2 3 4 5 6) '(2 3)))
  => 6
PyTorch: torch.Tensor.numel"
  (numel (tensor-shape t)))

(define (tensor->list t)
  "Read every element as a flat Scheme list.

Args:
  t: a tensor.
Returns: the elements in row-major order.
Example:
  (use-modules (sonicboom tensor))
  (tensor->list (make-tensor '(1.5 -2.0 0.25) '(3)))
  => (1.5 -2.0 0.25)
PyTorch: torch.Tensor.tolist"
  (let ((n (tensor-numel t)))
    (let loop ((i 0) (acc '()))
      (if (= i n)
          (reverse acc)
          (loop (+ i 1) (cons (tensor-ref t i) acc))))))

(define* (list->tensor data shape #:key (dtype 'float32))
  "Build a tensor from a flat list of numbers.

Args:
  data: the elements in row-major order.
  shape: a list of non-negative integers.
  dtype: a dtype name; float32 by default.
Returns: a new tensor.
Example:
  (use-modules (sonicboom tensor))
  (tensor->list (list->tensor '(1 2) '(2) #:dtype 'int64))
  => (1 2)
PyTorch: torch.tensor"
  (make-tensor data shape #:dtype dtype))

(define (tensor-copy t)
  "An independent tensor with the same dtype, shape and bytes.

Always owned and writable, even when the source is a read-only borrowed view.
Args:
  t: a tensor.
Returns: a new tensor with copied bytes.
Example:
  (use-modules (sonicboom tensor))
  (let* ((a (make-tensor '(1 2) '(2)))
         (b (tensor-copy a)))
    (tensor-set! b 0 9)
    (tensor->list a))
  => (1.0 2.0)
PyTorch: torch.Tensor.clone
Note: The gradient slot is carried over as-is; no autograd history is built."
  (%make-tensor (tensor-dtype t) (tensor-shape t)
                (owned-storage (bytevector-copy (tensor-data t)))
                (tensor-requires-grad? t) (tensor-grad t)))

(define (tensor-release! t)
  "Release a tensor's borrowed native view, if it has one.

Idempotent; a no-op for owned storage. The bytes stay valid until the last view
and executable are gone, but the view handle is dropped: do not read the tensor
after releasing it.
Args:
  t: a tensor.
Returns: unspecified.
PyTorch: none
Note: Requires a model file or an executable and therefore has no example here."
  (storage-release! (tensor-storage t))
  *unspecified*)

;;; --- parameters -------------------------------------------------------------

(define* (parameter data #:key (shape #f) (dtype 'float32))
  "A trainable tensor.

Args:
  data: an existing tensor, or a flat list of numbers.
  shape: the shape to use when data is a list; defaults to a rank-1 shape.
  dtype: a dtype name for a list; float32 by default.
Returns: a tensor with requires-grad set.
Example:
  (use-modules (sonicboom tensor))
  (tensor->list (parameter '(1.0 2.0)))
  => (1.0 2.0)
PyTorch: torch.nn.Parameter
Note: The tensor is marked in place; passing an existing tensor with
requires-grad already set returns it unchanged apart from the flag."
  (let ((t (if (tensor? data)
               data
               (make-tensor data (or shape (list (length data))) #:dtype dtype))))
    (set-tensor-requires-grad! t #t)
    t))

(define (parameter? x)
  "Whether a value is a parameter.

Args:
  x: any value.
Returns: #t when x is a tensor with requires-grad set.
Example:
  (use-modules (sonicboom tensor))
  (list (parameter? (parameter '(1.0))) (parameter? (make-tensor '(1.0) '(1))))
  => (#t #f)
PyTorch: none
Note: In PyTorch this is an instance test against torch.nn.Parameter; here it is
the requires-grad flag."
  (and (tensor? x) (tensor-requires-grad? x)))

(define (parameter-grad p)
  "The gradient stored on a parameter, or #f.

Args:
  p: a parameter.
Returns: the gradient tensor, or #f if none has been set.
Example:
  (use-modules (sonicboom tensor))
  (let ((p (parameter '(1.0 2.0))) (g (make-tensor '(0.5 -1.0) '(2))))
    (parameter-grad-set! p g)
    (tensor->list (parameter-grad p)))
  => (0.5 -1.0)
PyTorch: torch.Tensor.grad"
  (tensor-grad p))

(define (parameter-grad-set! p g)
  "Attach a gradient tensor to a parameter.

Args:
  p: a parameter.
  g: the gradient tensor.
Returns: unspecified.
Example:
  (use-modules (sonicboom tensor))
  (let ((p (parameter '(1.0))) (g (make-tensor '(0.25) '(1))))
    (parameter-grad-set! p g)
    (tensor->list (parameter-grad p)))
  => (0.25)
PyTorch: none
Note: In PyTorch gradients are produced by backward() and stored in .grad;
here the slot is set explicitly."
  (unless (parameter? p)
    (error "parameter-grad-set!: not a parameter" p))
  (set-tensor-grad! p g))
