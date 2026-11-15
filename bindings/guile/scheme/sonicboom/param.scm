;;; SonicBoom — Guile binding: trainable parameters with fused AdamW
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A <param> is a native float32 parameter (capi.h boom_param_*): its values, its
;;; gradient and AdamW's two moment buffers. When a CUDA device is present the
;;; four buffers stay in device memory across steps and the update is one fused
;;; kernel launch; otherwise they are host buffers and the host reference runs.
;;; The handle hides which — (param-on-device?) tells.
;;;
;;; Values cross the boundary only as little-endian float32 bytevectors.
;;;
;;;   (make-param dims #:init bv)            → a <param>
;;;   (param-step! p lr b1 b2 eps wd)        → AdamW update, new step count
;;;   (param-step! p lr-tensor b1 b2 eps wd) → the lr-input form
;;;   (param-read p)                         → the current values
;;;
;;; The native handle is owned by the record. If the record is collected
;;; without an explicit `free-param!`, a GC guardian releases the handle when
;;; `reap-params!` is next called; `free-param!` is idempotent, so the two never
;;; double-free. (This Guile has no `set-finalizer!`.)

(define-module (sonicboom param)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 optargs)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom error)
  #:export (make-param param? param-freed?
            param-numel param-on-device? param-step-count
            param-zero-grad! param-set-grad! param-step! param-read
            free-param! reap-params!))

;;; --- the param record --------------------------------------------------------

;; The record of a native trainable parameter.
;;
;; A parameter holds its values, its gradient and AdamW's two moment buffers in
;; one boom_param handle (see (sonicboom param)). The handle is owned by this
;; record: `free-param!` releases it explicitly, and the guardian used by
;; `reap-params!` releases it if the record is collected first.
;; PyTorch: none
;; Note: The nearest PyTorch concept is a parameter together with its optimizer
;; state, which PyTorch keeps in separate objects.
(define-record-type <param>
  (%make-param dims numel ptr freed)
  param?
  (dims param-dims)
  (numel param-numel)
  (ptr param-ptr)
  ;; Flipped to #t by whoever releases the handle first (explicit free or the
  ;; guardian), so the other one becomes a no-op instead of a double free.
  (freed param-freed? set-param-freed!))

(define (free-param! p)
  "Release a parameter's native handle.

Idempotent: freeing an already-freed (or guardian-reaped) parameter is a no-op,
never an error.
Args:
  p: a parameter.
Returns: unspecified.
Example:
  (use-modules (sonicboom param))
  (let ((p (make-param '(2)))) (free-param! p) (param-freed? p))
  => #t
PyTorch: none
Note: PyTorch parameters are released by garbage collection; there is no
explicit free."
  (unless (param-freed? p)
    (set-param-freed! p #t)             ; claim it before touching the pointer
    (let ((ptr (param-ptr p)))
      (unless (null-pointer? ptr)
        ((proc 'boom_param_free) ptr))))
  *unspecified*)

;;; --- GC fallback -------------------------------------------------------------

(define %param-guardian (make-guardian))

(define (reap-params!)
  "Release every parameter collected without an explicit free-param!.

Returns: the number of handles released.
Example:
  (use-modules (sonicboom param))
  (reap-params!)
PyTorch: none
Note: The count is zero in a fresh process; the example shows the call, and a
collection is what makes it return a positive number."
  (let loop ((n 0))
    (let ((dead (%param-guardian)))
      (if (or (not dead) (null? dead))
          n
          (begin (free-param! dead) (loop (+ 1 n)))))))

;;; --- helpers -----------------------------------------------------------------

;; The native pointer of a live parameter; a released handle is a binding error,
;; never a dangling call into the C ABI.
(define (live-ptr p what)
  (if (param-freed? p)
      (raise-sonicboom-error
       BOOM_ERR_BINDING 'binding
       (string-append what ": parameter has already been freed"))
      (param-ptr p)))

(define (dims->pointer dims)
  (let ((bv (make-bytevector (* 8 (length dims)) 0)))
    (let loop ((i 0) (ls dims))
      (unless (null? ls)
        (bytevector-s64-native-set! bv (* i 8) (car ls))
        (loop (+ i 1) (cdr ls))))
    bv))

;;; --- construction ------------------------------------------------------------

(define* (make-param dims #:key init)
  "Create a native float32 parameter of a given shape.

The gradient and both AdamW moment buffers start at zero. The handle is owned by
the returned record; release it with free-param!.
Args:
  dims: a non-empty list of positive integers, row-major.
  init: the initial values as a little-endian float32 bytevector of numel*4
        bytes, or #f for zeros.
Returns: a <param>.
Example:
  (use-modules (sonicboom param))
  (list (param? (make-param '(2 3))) (param-numel (make-param '(2 3))))
  => (#t 6)
PyTorch: torch.nn.Parameter
Note: The parameter lives in native memory and its optimizer state is part of
the same handle; it is not a Scheme tensor."
  (unless (and (pair? dims)
               (every (lambda (d) (and (exact-integer? d) (>= d 1))) dims))
    (raise-sonicboom-error
     BOOM_ERR_BINDING 'binding
     "make-param: dims must be a non-empty list of positive integers"))
  (let* ((n (apply * dims))
         (bytes (* n 4))
         (init-bv
          (cond ((not init) (make-bytevector bytes 0))
                ((not (bytevector? init))
                 (raise-sonicboom-error
                  BOOM_ERR_BINDING 'binding
                  "make-param: init must be a bytevector of numel*4 bytes"))
                ((not (= (bytevector-length init) bytes))
                 (raise-sonicboom-error
                  BOOM_ERR_BINDING 'binding
                  (format #f "make-param: init is ~a bytes, the shape needs ~a"
                          (bytevector-length init) bytes)))
                (else init)))
         (dims-bv (dims->pointer dims))
         (out-slot (make-slot))
         (err-slot (make-slot)))
    (sb-check! ((proc 'boom_param_create)
                (bytevector->pointer dims-bv) (length dims)
                (bytevector->pointer init-bv) bytes
                (slot-ptr out-slot) (slot-ptr err-slot))
               err-slot "make-param")
    (let ((p (%make-param (list-copy dims) n (slot-ref out-slot) #f)))
      (%param-guardian p)
      p)))

;;; --- inspection --------------------------------------------------------------

(define (param-on-device? p)
  "Whether a parameter's buffers are device-resident.

A CUDA device present at creation keeps the four buffers in device memory, so
each update is one fused kernel launch; otherwise the host reference runs.
Args:
  p: a parameter.
Returns: #t for device-resident buffers.
Example:
  (use-modules (sonicboom param))
  (param-on-device? (make-param '(2)))
PyTorch: torch.Tensor.is_cuda
Note: The expected value depends on the build and the machine, so the example
does not compare it."
  (live-ptr p "param-on-device?")
  (not (zero? ((proc 'boom_param_on_device) (param-ptr p)))))

(define (param-step-count p)
  "The number of AdamW updates applied to a parameter so far.

Args:
  p: a parameter.
Returns: the step count; 0 before the first update.
Example:
  (use-modules (sonicboom param))
  (param-step-count (make-param '(2)))
  => 0
PyTorch: none
Note: PyTorch keeps the step count inside each optimizer's per-parameter state
rather than on the parameter."
  (live-ptr p "param-step-count")
  ((proc 'boom_param_step_count) (param-ptr p)))

(define (param-read p)
  "Copy a parameter's current values out.

Args:
  p: a parameter.
Returns: a little-endian float32 bytevector of numel*4 bytes.
Example:
  (use-modules (sonicboom param))
  (bytevector-length (param-read (make-param '(2 3))))
  => 24
PyTorch: none
Note: In PyTorch the values are read from the parameter tensor itself."
  (let* ((ptr (live-ptr p "param-read"))
         (bytes (* 4 (param-numel p)))
         (out (make-bytevector bytes 0))
         (written (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot)))
    (sb-check! ((proc 'boom_param_read) ptr (bytevector->pointer out) bytes
                (bytevector->pointer written) (slot-ptr err-slot))
               err-slot "param-read")
    out))

;;; --- updates -----------------------------------------------------------------

(define (param-zero-grad! p)
  "Zero a parameter's gradient.

Args:
  p: a parameter.
Returns: unspecified.
Example:
  (use-modules (sonicboom param))
  (let ((p (make-param '(2))))
    (param-zero-grad! p)
    (param-step-count p))
  => 0
PyTorch: torch.optim.Optimizer.zero_grad
Note: It zeros one parameter, whereas the PyTorch method walks the optimizer's
parameter groups."
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'boom_param_zero_grad) (live-ptr p "param-zero-grad!")
                (slot-ptr err-slot))
               err-slot "param-zero-grad!")
    *unspecified*))

(define (param-set-grad! p grad)
  "Replace a parameter's gradient with a bytevector.

Args:
  p: a parameter.
  grad: a little-endian float32 bytevector of numel*4 bytes.
Returns: unspecified.
Example:
  (use-modules (sonicboom param) (rnrs bytevectors))
  (let ((p (make-param '(2))))
    (param-set-grad! p (make-bytevector 8 0))
    (param-step-count p))
  => 0
PyTorch: none
Note: In PyTorch gradients are produced by backward() and stored in .grad."
  (let ((ptr (live-ptr p "param-set-grad!")))
    (unless (bytevector? grad)
      (raise-sonicboom-error BOOM_ERR_BINDING 'binding
                             "param-set-grad!: grad must be a bytevector"))
    (unless (= (bytevector-length grad) (* 4 (param-numel p)))
      (raise-sonicboom-error
       BOOM_ERR_BINDING 'binding
       (format #f "param-set-grad!: grad is ~a bytes, the parameter needs ~a"
               (bytevector-length grad) (* 4 (param-numel p)))))
    (let ((err-slot (make-slot)))
      (sb-check! ((proc 'boom_param_set_grad) ptr (bytevector->pointer grad)
                  (bytevector-length grad) (slot-ptr err-slot))
                 err-slot "param-set-grad!")
      *unspecified*)))

(define (param-step! p lr beta1 beta2 eps weight-decay)
  "Apply one AdamW update to a parameter.

Args:
  p: a parameter.
  lr: a real learning rate, or a one-element float32 tensor handle read on the host.
  beta1: the first-moment decay.
  beta2: the second-moment decay.
  eps: the denominator epsilon.
  weight-decay: the decoupled weight decay.
Returns: the new step count.
Example:
  (use-modules (sonicboom param))
  (let ((p (make-param '(2))))
    (param-step! p 0.1 0.9 0.999 1e-8 0.01))
  => 1
PyTorch: torch.optim.AdamW.step
Note: One call is one update of one parameter, not of a parameter group. The
tensor-handle form of lr lets a traced learning-rate schedule carry the rate
while the parameter's own buffers stay device-resident."
  (let ((ptr (live-ptr p "param-step!"))
        (err-slot (make-slot)))
    (let ((st
           (cond
            ((pointer? lr)
             ((proc 'boom_param_step_lr) ptr lr
              (exact->inexact beta1) (exact->inexact beta2)
              (exact->inexact eps) (exact->inexact weight-decay)
              (slot-ptr err-slot)))
            ((real? lr)
             ((proc 'boom_param_step) ptr
              (exact->inexact lr) (exact->inexact beta1)
              (exact->inexact beta2) (exact->inexact eps)
              (exact->inexact weight-decay)
              (slot-ptr err-slot)))
            (else
             (raise-sonicboom-error
              BOOM_ERR_BINDING 'binding
              "param-step!: lr must be a real number or a one-element float32 tensor handle")))))
      (sb-check! st err-slot "param-step!")
      (param-step-count p))))
