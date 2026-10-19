;;; SonicBoom — Guile binding: parameter checkpoints
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Named float32 tensors persisted through the C ABI (sb_save_params /
;;; sb_load_params). A checkpoint is one self-describing little-endian binary
;;; file; entries are matched by name, so save and load order need not agree.
;;; Tensors cross the boundary as sb_tensor handles built from the ctensor's
;;; float32 bytes, exactly like the rest of (sonicboom train): Guile never does
;;; arithmetic here, it only marshals.
;;;
;;;   (save-params! path '((embedding . tensor) (wq . tensor) ...))
;;;   (load-params! path '((embedding . template) (wq . template) ...))
;;;     → '((embedding . tensor) (wq . tensor) ...)
;;;
;;; The template tensors supply each entry's shape (and must be float32; the
;;; C ABI's checkpoint surface is float32-only). Loading is all-or-nothing: a
;;; missing name or a shape mismatch signals an error before any value is
;;; produced.

(define-module (sonicboom checkpoint)
  #:use-module (srfi srfi-1)
  #:use-module (rnrs bytevectors)
  #:use-module (system foreign)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom trace tensor)
  #:export (save-params! load-params!))

;;; --- marshalling -------------------------------------------------------------

(define (entry-name x)
  "Checkpoint entry names are Scheme symbols or strings; the file stores them
as UTF-8 strings."
  (cond ((string? x) x)
        ((symbol? x) (symbol->string x))
        (else (error "checkpoint: entry name must be a symbol or string" x))))

(define (normalize entries)
  "Validate ENTRIES and return it unchanged: the caller's keys (symbols or
strings) identify the returned entries, while `entry-name` renders them for
the file."
  (unless (and (list? entries) (every pair? entries))
    (error "checkpoint: expected an alist of name → ctensor" entries))
  (for-each (lambda (e) (entry-name (car e))) entries)
  entries)

(define (dims->pointer dims)
  (let ((bv (make-bytevector (* 8 (max 1 (length dims))) 0)))
    (let loop ((i 0) (ds dims))
      (unless (null? ds)
        (bytevector-s64-native-set! bv (* i 8) (car ds))
        (loop (+ i 1) (cdr ds))))
    bv))

(define (pointer-array ptrs)
  ;; A zero-length bytevector has no usable pointer, so keep one spare slot.
  (let ((bv (make-bytevector (* 8 (max 1 (length ptrs))) 0)))
    (let loop ((i 0) (ps ptrs))
      (unless (null? ps)
        (bytevector-u64-native-set! bv (* i 8) (pointer-address (car ps)))
        (loop (+ i 1) (cdr ps))))
    bv))

(define (check-float32 who t)
  (unless (ctensor? t)
    (error who "expected a ctensor" t))
  (unless (eq? (ctensor-dtype t) 'float32)
    (error who "checkpoints only store float32 tensors" (ctensor-dtype t))))

(define (ctensor->bytevector t)
  (let* ((n (ctensor-numel t))
         (bv (make-bytevector (* 4 n))))
    (let loop ((i 0) (xs (ctensor->list t)))
      (unless (null? xs)
        (bytevector-ieee-single-native-set! bv (* i 4)
                                            (exact->inexact (car xs)))
        (loop (+ i 1) (cdr xs))))
    bv))

(define (bytevector->ctensor bv shape)
  (let ((n (/ (bytevector-length bv) 4)))
    (make-ctensor 'float32 shape
                  (let loop ((i 0) (acc '()))
                    (if (= i n)
                        (reverse acc)
                        (loop (+ i 1)
                              (cons (bytevector-ieee-single-native-ref bv (* i 4))
                                    acc)))))))

(define (ctensor->handle who t)
  (check-float32 who t)
  (let* ((bv (ctensor->bytevector t))
         (dims (ctensor-shape t))
         (out (make-slot))
         (err (make-slot))
         (st ((proc 'sb_tensor_from_f32) (bytevector->pointer (dims->pointer dims))
              (length dims) (bytevector->pointer bv) (bytevector-length bv)
              (slot-ptr out) (slot-ptr err))))
    (unless (sb-status-ok? st)
      (error who (capture-error err)))
    (slot-ref out)))

(define (handle->ctensor who h shape)
  (let ((n (shape-numel shape)))
    (if (zero? n)
        (make-ctensor 'float32 shape '())
        (let* ((bv (make-bytevector (* 4 n)))
               (written (make-bytevector 8 0))
               (err (make-slot))
               (st ((proc 'sb_tensor_bytes) h (bytevector->pointer bv)
                    (bytevector-length bv) (bytevector->pointer written)
                    (slot-ptr err))))
          (unless (sb-status-ok? st)
            (error who (capture-error err)))
          (bytevector->ctensor bv shape)))))

;;; --- save / load -------------------------------------------------------------

(define (save-params! path entries)
  "Write ENTRIES, an alist name → float32 ctensor, to checkpoint PATH,
overwriting it. Names must be unique. Returns PATH."
  (let* ((pairs (normalize entries))
         (handles (map (lambda (p) (ctensor->handle 'save-params! (cdr p))) pairs))
         (name-ptrs (map (lambda (p) (string->pointer (entry-name (car p))))
                         pairs))
         ;; Bound so the bytevectors and string pointers stay referenced across
         ;; the call; Guile's bytevectors do not move, but they must stay live.
         (name-arr (pointer-array name-ptrs))
         (handle-arr (pointer-array handles))
         (path-ptr (string->pointer path))
         (err (make-slot)))
    (dynamic-wind
      (lambda () #f)
      (lambda ()
        (let ((st ((proc 'sb_save_params) path-ptr
                   (bytevector->pointer name-arr)
                   (bytevector->pointer handle-arr)
                   (length handles) (slot-ptr err))))
          (unless (sb-status-ok? st)
            (error "save-params!" (capture-error err))))
        path)
      (lambda ()
        (for-each (lambda (h) ((proc 'sb_tensor_free) h)) handles)))))

(define (load-params! path entries)
  "Load ENTRIES, an alist name → template float32 ctensor whose shapes describe
the checkpoint entries to restore, from checkpoint PATH. Returns an alist
name → new ctensor in the same order. All-or-nothing: on a missing name or a
shape mismatch, an error is signalled and no tensor is returned."
  (let* ((pairs (normalize entries))
         (handles (map (lambda (p) (ctensor->handle 'load-params! (cdr p))) pairs))
         (name-ptrs (map (lambda (p) (string->pointer (entry-name (car p))))
                         pairs))
         (name-arr (pointer-array name-ptrs))
         (handle-arr (pointer-array handles))
         (path-ptr (string->pointer path))
         (err (make-slot)))
    (dynamic-wind
      (lambda () #f)
      (lambda ()
        (let ((st ((proc 'sb_load_params) path-ptr
                   (bytevector->pointer name-arr)
                   (bytevector->pointer handle-arr)
                   (length handles) (slot-ptr err))))
          (unless (sb-status-ok? st)
            (error "load-params!" (capture-error err))))
        (map (lambda (p h)
               (cons (car p)
                     (handle->ctensor 'load-params! h (ctensor-shape (cdr p)))))
             pairs handles))
      (lambda ()
        (for-each (lambda (h) ((proc 'sb_tensor_free) h)) handles)))))
