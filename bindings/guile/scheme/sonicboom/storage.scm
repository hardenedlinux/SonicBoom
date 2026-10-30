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
;;; public module re-exports it. Only the 'owned kind exists: a Guile bytevector
;;; the tensor owns. The owner, handle and generation slots are placeholders for
;;; native storage.

(define-module (sonicboom storage)
  #:use-module (srfi srfi-9)
  #:use-module (rnrs bytevectors)
  #:export (storage? owned-storage storage-data))

;; DATA is deliberately the first field: storage-data is inlined as a direct
;; struct slot so that element accessors keep one indirection, not two.
(define-record-type <storage>
  (%make-storage data kind owner handle writable? generation)
  storage?
  (data        %storage-data)       ; bytevector, or #f for device storage
  (kind        storage-kind)        ; 'owned (later: 'borrowed-host, 'device)
  (owner       storage-owner)       ; #f | Scheme object pinning native memory
  (handle      storage-handle)      ; #f | opaque native view handle
  (writable?   storage-writable?)   ; element writes permitted?
  (generation  storage-generation)) ; bumped when the owner's memory changes

;; A record's struct slots are its declared fields, in order; DATA is slot 0.
;; Inlined on purpose (it is on the element-access path), so it is a call form,
;; not a first-class procedure.
(define-inlinable (storage-data s) (struct-ref s 0))

(define (owned-storage bv)
  "A freshly owned storage over BV. It does not copy BV: the caller decides
whether to hand over a private bytevector (constructors, *-copy) or to share its
own (bytevector->tensor, bytevector->ctensor)."
  (%make-storage bv 'owned #f #f #t 0))

;; Guard the hand-written slot above: a reordered record must fail here, at load
;; time, instead of silently reading the wrong field.
(let ((probe (make-bytevector 0)))
  (unless (eq? probe (storage-data (owned-storage probe)))
    (error "<storage> layout changed: storage-data's struct slot is stale")))
