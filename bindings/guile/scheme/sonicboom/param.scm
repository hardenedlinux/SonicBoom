;;; SonicBoom — Guile binding: trainable parameters with fused AdamW
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A <param> is a native float32 parameter (capi.h sb_param_*): its values, its
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
  "Release P's native handle. Idempotent: freeing an already-freed (or
guardian-reaped) parameter is a no-op, never an error."
  (unless (param-freed? p)
    (set-param-freed! p #t)             ; claim it before touching the pointer
    (let ((ptr (param-ptr p)))
      (unless (null-pointer? ptr)
        ((proc 'sb_param_free) ptr))))
  *unspecified*)

;;; --- GC fallback -------------------------------------------------------------

(define %param-guardian (make-guardian))

(define (reap-params!)
  "Release every parameter whose record has been collected without an explicit
`free-param!`. Returns the number of handles released."
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
       SB_ERR_BINDING 'binding
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
  "Create a float32 parameter of shape DIMS (a non-empty list of positive
integers, row-major) holding INIT's little-endian bytes, or zeros when INIT is
#f. The gradient and both AdamW moment buffers start at zero. Returns a
<param>; release it with `free-param!`."
  (unless (and (pair? dims)
               (every (lambda (d) (and (exact-integer? d) (>= d 1))) dims))
    (raise-sonicboom-error
     SB_ERR_BINDING 'binding
     "make-param: dims must be a non-empty list of positive integers"))
  (let* ((n (apply * dims))
         (bytes (* n 4))
         (init-bv
          (cond ((not init) (make-bytevector bytes 0))
                ((not (bytevector? init))
                 (raise-sonicboom-error
                  SB_ERR_BINDING 'binding
                  "make-param: init must be a bytevector of numel*4 bytes"))
                ((not (= (bytevector-length init) bytes))
                 (raise-sonicboom-error
                  SB_ERR_BINDING 'binding
                  (format #f "make-param: init is ~a bytes, the shape needs ~a"
                          (bytevector-length init) bytes)))
                (else init)))
         (dims-bv (dims->pointer dims))
         (out-slot (make-slot))
         (err-slot (make-slot)))
    (sb-check! ((proc 'sb_param_create)
                (bytevector->pointer dims-bv) (length dims)
                (bytevector->pointer init-bv) bytes
                (slot-ptr out-slot) (slot-ptr err-slot))
               err-slot "make-param")
    (let ((p (%make-param (list-copy dims) n (slot-ref out-slot) #f)))
      (%param-guardian p)
      p)))

;;; --- inspection --------------------------------------------------------------

(define (param-on-device? p)
  "Whether P's buffers are device-resident (a CUDA device was present at
creation), so each update is one fused kernel launch."
  (live-ptr p "param-on-device?")
  (not (zero? ((proc 'sb_param_on_device) (param-ptr p)))))

(define (param-step-count p)
  "The number of AdamW updates applied so far (0 before the first step)."
  (live-ptr p "param-step-count")
  ((proc 'sb_param_step_count) (param-ptr p)))

(define (param-read p)
  "Copy P's current values out as a little-endian float32 bytevector."
  (let* ((ptr (live-ptr p "param-read"))
         (bytes (* 4 (param-numel p)))
         (out (make-bytevector bytes 0))
         (written (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot)))
    (sb-check! ((proc 'sb_param_read) ptr (bytevector->pointer out) bytes
                (bytevector->pointer written) (slot-ptr err-slot))
               err-slot "param-read")
    out))

;;; --- updates -----------------------------------------------------------------

(define (param-zero-grad! p)
  "Zero P's gradient."
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'sb_param_zero_grad) (live-ptr p "param-zero-grad!")
                (slot-ptr err-slot))
               err-slot "param-zero-grad!")
    *unspecified*))

(define (param-set-grad! p grad)
  "Replace P's gradient with GRAD, a little-endian float32 bytevector of
numel*4 bytes."
  (let ((ptr (live-ptr p "param-set-grad!")))
    (unless (bytevector? grad)
      (raise-sonicboom-error SB_ERR_BINDING 'binding
                             "param-set-grad!: grad must be a bytevector"))
    (unless (= (bytevector-length grad) (* 4 (param-numel p)))
      (raise-sonicboom-error
       SB_ERR_BINDING 'binding
       (format #f "param-set-grad!: grad is ~a bytes, the parameter needs ~a"
               (bytevector-length grad) (* 4 (param-numel p)))))
    (let ((err-slot (make-slot)))
      (sb-check! ((proc 'sb_param_set_grad) ptr (bytevector->pointer grad)
                  (bytevector-length grad) (slot-ptr err-slot))
                 err-slot "param-set-grad!")
      *unspecified*)))

(define (param-step! p lr beta1 beta2 eps weight-decay)
  "Apply one AdamW update to P and return the new step count. LR is either a
real number or a one-element float32 tensor handle (as returned by
`tensor-from-list`); the handle form reads the rate on the host while P's own
buffers stay device-resident, which is how a traced schedule carries the rate."
  (let ((ptr (live-ptr p "param-step!"))
        (err-slot (make-slot)))
    (let ((st
           (cond
            ((pointer? lr)
             ((proc 'sb_param_step_lr) ptr lr
              (exact->inexact beta1) (exact->inexact beta2)
              (exact->inexact eps) (exact->inexact weight-decay)
              (slot-ptr err-slot)))
            ((real? lr)
             ((proc 'sb_param_step) ptr
              (exact->inexact lr) (exact->inexact beta1)
              (exact->inexact beta2) (exact->inexact eps)
              (exact->inexact weight-decay)
              (slot-ptr err-slot)))
            (else
             (raise-sonicboom-error
              SB_ERR_BINDING 'binding
              "param-step!: lr must be a real number or a one-element float32 tensor handle")))))
      (sb-check! st err-slot "param-step!")
      (param-step-count p))))
