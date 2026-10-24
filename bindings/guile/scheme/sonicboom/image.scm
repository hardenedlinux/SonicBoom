;;; SonicBoom — Guile binding: image generation
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The (sonicboom image) module binds the MOSS-Image C ABI (capi.h
;;; sb_image_*) through (sonicboom ffi), the only path Guile uses to reach
;;; libsonicboom.so:
;;;
;;;   Guile → (sonicboom image) → (sonicboom ffi) → libsonicboom.so
;;;
;;; One opaque model handle (load-image-model / free-image-model!), one
;;; generation call that returns a list of (width height rgb-bytevector), and
;;; the library's own minimal PNG writer. The model directory is the
;;; MOSS-Image root (text_encoder/, transformer/, vae/, tokenizer/).

(define-module (sonicboom image)
  #:use-module (sonicboom ffi)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (ice-9 optargs)
  #:export (load-image-model free-image-model! image-model-arch
            generate-images write-png))

;;; --- errors -----------------------------------------------------------------

(define (sb-check! st err what)
  (unless (sb-status-ok? st)
    (let ((msg (capture-error err)))
      (throw 'sb-image
             (string-append what ": "
                            (if (string-null? msg)
                                (sb-status->message st)
                                msg))))))

;; Free (and NULL) whatever error an expected non-OK status left in ERR.
(define (discard-error! err) (capture-error err))

;; A call the compiler cannot drop, used to keep foreign pointer objects
;; alive: the request bytevector holds only raw addresses, so the pointer
;; objects themselves must still be referenced after the C call returns.
(define (keep! . objects) (if (null? objects) #f #t))

;;; --- model ------------------------------------------------------------------

(define (load-image-model dir)
  "Load a MOSS-Image model directory and return an opaque image-model handle.
Signals 'sb-image if the directory is not a model the pipeline accepts."
  (let* ((model-slot (make-slot))
         (err-slot (make-slot))
         (st ((proc 'sb_image_model_load) (string->pointer dir)
              (slot-ptr model-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "load-image-model")
    (slot-ref model-slot)))

(define (free-image-model! model)
  ((proc 'sb_image_model_free) model))

;; The architecture name, two-call sizing like the LLM surface's model-arch:
;; the sizing call fills the length (no NUL) and reports SB_ERR_BINDING.
(define (image-model-arch model)
  "The model's architecture name (\"moss-image\")."
  (let* ((len-slot (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot)))
    (let ((st ((proc 'sb_image_model_arch) model %null-pointer 0
               (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st SB_ERR_BINDING))
        (sb-check! st err-slot "image-model-arch"))
      (discard-error! err-slot))
    (let* ((m (bytevector-u64-native-ref len-slot 0))
           (out (make-bytevector (+ m 1) 0))    ; +1 for the NUL terminator
           (st ((proc 'sb_image_model_arch) model (bytevector->pointer out)
                (+ m 1) (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (sb-check! st err-slot "image-model-arch")
      (pointer->string (bytevector->pointer out)))))

;;; --- generation -------------------------------------------------------------

;; sb_image_request as a native-layout bytevector (x86-64; see capi.h):
;;   struct_size size_t@0, prompt ptr@8, negative_prompt ptr@16,
;;   width u32@24, height u32@28, steps u32@32, guidance_scale f32@36,
;;   seed u64@40, batch u32@48  → 56 bytes.
(define (make-image-request-bv prompt-ptr negative-ptr width height steps
                               guidance seed batch)
  (let ((bv (make-bytevector 56 0)))
    (bytevector-u64-native-set! bv 0 56)
    (bytevector-u64-native-set! bv 8 (pointer-address prompt-ptr))
    (bytevector-u64-native-set! bv 16
                                (if negative-ptr
                                    (pointer-address negative-ptr)
                                    0))
    (bytevector-u32-native-set! bv 24 width)
    (bytevector-u32-native-set! bv 28 height)
    (bytevector-u32-native-set! bv 32 steps)
    (bytevector-ieee-single-native-set! bv 36 (exact->inexact guidance))
    (bytevector-u64-native-set! bv 40 seed)
    (bytevector-u32-native-set! bv 48 batch)
    bv))

;; The C progress callback returns non-zero to cancel; the Scheme procedure
;; returns #f to cancel. Keep the returned pointer referenced for the duration
;; of the call — (procedure->pointer) does not keep the closure alive by itself.
(define (make-progress-callback progress)
  (and progress
       (procedure->pointer int
         (lambda (user image step total)
           (if (eq? #f (progress image step total)) 1 0))
         (list '* uint32 uint32 uint32))))

(define* (generate-images model
                          #:key
                          (prompt "")
                          (negative #f)
                          (width 512)
                          (height 512)
                          (steps 50)
                          (guidance 3.0)
                          (seed 0)
                          (batch 1)
                          (progress #f))
  "Generate BATCH images for PROMPT from MODEL and return a list of
(width height rgb-bytevector), one per image, RGB HWC, 8 bits per channel.

WIDTH and HEIGHT are multiples of 16. NEGATIVE is the negative prompt used by
classifier-free guidance (#f is the empty prompt). PROGRESS, when a procedure,
is called as (image-index step total-steps) after each denoising step; returning
#f cancels the generation, which then signals 'sb-image (the C status is
SB_ERR_CANCELLED). Image i starts from the same seeded noise as sb-moss's
`sample --seed`, so the same request is reproducible."
  (let* ((prompt-ptr (string->pointer prompt))
         (negative-ptr (and negative (string->pointer negative)))
         (req (make-image-request-bv prompt-ptr negative-ptr width height steps
                                     guidance seed batch))
         (cb (make-progress-callback progress))
         (bytes-slot (make-bytevector 8 0))
         (err-slot (make-slot)))
    ;; Two-call sizing: the byte count depends only on the request.
    (let ((st ((proc 'sb_image_generate) model (bytevector->pointer req)
               %null-pointer %null-pointer %null-pointer 0
               (bytevector->pointer bytes-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st SB_ERR_BINDING))
        (sb-check! st err-slot "generate-images"))
      (discard-error! err-slot))
    (let* ((n (bytevector-u64-native-ref bytes-slot 0))
           (rgb (make-bytevector n 0))
           (st ((proc 'sb_image_generate) model (bytevector->pointer req)
                (if cb cb %null-pointer) %null-pointer
                (bytevector->pointer rgb) n
                (bytevector->pointer bytes-slot) (slot-ptr err-slot))))
      (unless (sb-status-ok? st)
        (let ((msg (capture-error err-slot)))
          (throw 'sb-image
                 (if (= st SB_ERR_CANCELLED)
                     "generate-images: cancelled"
                     (string-append "generate-images: "
                                    (if (string-null? msg)
                                        (sb-status->message st)
                                        msg))))))
      (let* ((stride (* width height 3))
             (images
              (let loop ((i (- batch 1)) (acc '()))
                (if (< i 0)
                    acc
                    (let ((copy (make-bytevector stride 0)))
                      (bytevector-copy! rgb (* i stride) copy 0 stride)
                      (loop (- i 1)
                            (cons (list width height copy) acc)))))))
        (keep! prompt-ptr negative-ptr cb)
        images))))

;;; --- output -----------------------------------------------------------------

(define (write-png path rgb width height)
  "Write RGB (a width*height*3 bytevector, RGB HWC) to PATH as a PNG
(8-bit RGB, stored deflate blocks) and return #t."
  (unless (and (bytevector? rgb) (>= (bytevector-length rgb) (* width height 3)))
    (throw 'sb-image "write-png: rgb is shorter than width*height*3"))
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'sb_image_write_png) (string->pointer path)
                (bytevector->pointer rgb) width height (slot-ptr err-slot))
               err-slot "write-png")
    #t))
