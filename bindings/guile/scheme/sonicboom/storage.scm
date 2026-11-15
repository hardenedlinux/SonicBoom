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
;;; This is a low-level module: no other module re-exports it, and application
;;; code normally builds tensors through (sonicboom tensor) instead. Two kinds:
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

;; The record holding a tensor's bytes.
;;
;; Both <tensor> and <ctensor> keep their data in a storage record, so a new
;; storage kind is added once rather than twice. There are two kinds: an owned
;; Guile bytevector, and a borrowed-host view of native executable parameter
;; memory that the tensor API keeps read-only.
;; PyTorch: none
;; Note: PyTorch's storage is an untyped device buffer reached through tensor
;; internals; this record is visible to Scheme code and is not part of the
;; user-level API.
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

;; The bytevector stored in a storage record.
;;
;; Args:
;;   s: a storage record.
;; Returns: the backing bytevector.
;; Example:
;;   (use-modules (sonicboom storage) (rnrs bytevectors))
;;   (let ((bv (make-bytevector 4 7)))
;;     (eq? bv (storage-data (owned-storage bv))))
;;   => #t
;; PyTorch: none
;; Note: The call never copies. A borrowed view's bytevector must not be written,
;; even though Guile keeps it mutable.
(define-inlinable (storage-data s) (struct-ref s 0))

(define (owned-storage bv)
  "Storage over a bytevector, without copying it.

The caller decides whether to hand over a private bytevector (constructors,
copies) or to share its own (bytevector->tensor).
Args:
  bv: the bytevector to own.
Returns: writable storage over bv.
Example:
  (use-modules (sonicboom storage) (rnrs bytevectors))
  (bytevector-length (storage-data (owned-storage (make-bytevector 4))))
  => 4
PyTorch: none"
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
  "Storage viewing native parameter memory through a bytevector.

The bytevector aliases native memory without copying. The owner is kept alive
for the caller's convenience; the native view is the sb_storage handle that
pins the memory itself. The aliasing bytevector stays mutable because Guile
bytevectors cannot be read-only; tensor-set! enforces the read-only rule.
Args:
  data: a bytevector aliasing the native memory.
  owner: the object that must outlive the view, usually an executable.
  native-view: the sb_storage handle pinning the memory.
Returns: read-only-through-the-tensor-API storage.
PyTorch: none
Note: This requires an executable with native parameters and therefore has no
example here."
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
  "Drop a storage's borrowed native view, if any.

Idempotent; a no-op for owned storage. The bytes are not freed while another
view or the executable still pins them, and the storage must not be read after
it is released.
Args:
  s: a storage record.
Returns: unspecified.
PyTorch: none"
  (let ((box (%storage-handle s)))
    (when box
      (set-storage-handle! s #f)
      (set-storage-owner! s #f)
      (release-box! box)))
  *unspecified*)

(define (reap-storage-views!)
  "Release every borrowed view whose tensor was collected.

Returns: the number of views released.
Example:
  (use-modules (sonicboom storage))
  (reap-storage-views!)
PyTorch: none
Note: The count is zero in a fresh process; the example shows the call."
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
