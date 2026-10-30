;;; SonicBoom — Guile binding: internal tensor storage
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The one place where a tensor's bytes live. <tensor> and <ctensor> stay
;;; separate semantic types with separate dtype and access rules, but each keeps
;;; its data in a <storage> record, so a second storage kind is added once
;;; rather than twice.
;;;
;;; INTERNAL. This module is not part of the Scheme programming interface and no
;;; public module re-exports it. Two kinds exist:
;;;
;;;   'owned          a Guile bytevector the tensor owns (writable)
;;;   'borrowed-host  a Guile bytevector viewing native executable parameter
;;;                   memory, read-only through the tensor API (tensor-set!
;;;                   rejects it). Guile bytevectors cannot be read-only, so
;;;                   the aliasing bytevector itself stays mutable and must
;;;                   not be written. OWNER is the <executable> record that
;;;                   keeps the handle alive and HANDLE owns the native
;;;                   sb_storage view that pins the memory itself.
;;;
;;; A borrowed view does not depend on the executable handle: releasing the view
;;; (storage-release!) drops its native ownership reference, and if a tensor is
;;; collected without an explicit release the guardian below releases it later.

(define-module (sonicboom storage)
  #:use-module (srfi srfi-9)
  #:use-module (rnrs bytevectors)
  #:use-module (system foreign)
  #:use-module (sonicboom ffi)
  #:export (storage? owned-storage borrowed-host-storage storage-data
            storage-writable? storage-release! reap-storage-views!))

;; DATA is deliberately the first field: storage-data is inlined as a direct
;; struct slot so that element accessors keep one indirection, not two.
(define-record-type <storage>
  (%make-storage data kind owner handle writable? generation)
  storage?
  (data        %storage-data)          ; bytevector, or #f for future device storage
  (kind        storage-kind)           ; 'owned | 'borrowed-host
  (owner       %storage-owner set-storage-owner!)   ; #f | Scheme object pinning memory
  (handle      %storage-handle set-storage-handle!) ; #f | guarded native view box
  (writable?   storage-writable?)      ; element writes permitted?
  (generation  storage-generation))    ; bumped when the owner's memory changes

;; A record's struct slots are its declared fields, in order; DATA is slot 0.
;; Inlined on purpose (it is on the element-access path), so it is a call form,
;; not a first-class procedure.
(define-inlinable (storage-data s) (struct-ref s 0))

(define (owned-storage bv)
  "A freshly owned storage over BV. It does not copy BV: the caller decides
whether to hand over a private bytevector (constructors, *-copy) or to share its
own (bytevector->tensor, bytevector->ctensor)."
  (%make-storage bv 'owned #f #f #t 0))

;;; --- borrowed native parameter views -----------------------------------------
;;;
;;; HANDLE is a box #(native-view released?) shared with the guardian entry. Both
;;; explicit release and the guardian flip `released?` before releasing, so the
;;; native sb_storage is released exactly once whichever path runs first. The
;;; guardian only holds a weak reference: a tensor that becomes unreachable is
;;; handed back by the next reap-storage-views! call.

(define %view-guardian (make-guardian))

(define (borrowed-host-storage data owner native-view)
  "A read-only-through-the-tensor-API storage viewing native memory through DATA
(a bytevector that aliases it, no copy). OWNER is kept alive for the caller's
convenience; NATIVE-VIEW is the sb_storage handle that pins the memory itself.
The aliasing bytevector is still mutable: tensor-set! is what enforces the
read-only rule."
  (let ((box (vector native-view #f)))
    (%view-guardian box)
    (%make-storage data 'borrowed-host owner box #f 0)))

(define (release-box! box)
  "Release the native view in BOX once; later calls are no-ops. Returns #t when
this call released it."
  (if (vector-ref box 1)
      #f
      (begin
        (vector-set! box 1 #t)
        (let ((view (vector-ref box 0)))
          (vector-set! box 0 %null-pointer)
          (unless (null-pointer? view)
            ((proc 'sb_storage_release) view)
            #t)))))

(define (storage-release! s)
  "Drop S's borrowed native view, if any. Idempotent; a no-op for owned storage.
The bytes are not freed while any other view or the executable still pins them.
After release S must not be read."
  (let ((box (%storage-handle s)))
    (when box
      (set-storage-handle! s #f)
      (set-storage-owner! s #f)
      (release-box! box)))
  *unspecified*)

(define (reap-storage-views!)
  "Release every borrowed view whose tensor was collected without an explicit
storage-release!. Returns the number of views released."
  (let loop ((n 0))
    (let ((box (%view-guardian)))
      (if (or (not box) (null? box))
          n
          (loop (+ n (if (release-box! box) 1 0)))))))

;; Guard the hand-written slot above: a reordered record must fail here, at load
;; time, instead of silently reading the wrong field.
(let ((probe (make-bytevector 0)))
  (unless (eq? probe (storage-data (owned-storage probe)))
    (error "<storage> layout changed: storage-data's struct slot is stale")))
