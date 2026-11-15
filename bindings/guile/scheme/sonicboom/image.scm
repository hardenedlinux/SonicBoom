;;; SonicBoom — Guile binding: image generation
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The (sonicboom image) module binds the MOSS-Image C ABI (capi.h
;;; boom_image_*) through (sonicboom ffi), the only path Guile uses to reach
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
  #:use-module (sonicboom error)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (ice-9 optargs)
  #:export (load-image-model free-image-model! image-model-arch
            generate-images write-png))

;;; --- errors -----------------------------------------------------------------

;; sb-check! comes from (sonicboom ffi): a C ABI failure is a &sonicboom-error
;; with the status, the boom_error_kind and the message. A cancelled generation
;; arrives here with kind 'cancelled.

;; Free (and NULL) whatever error an expected non-OK status left in ERR.
(define (discard-error! err) (capture-error err))

;; A call the compiler cannot drop, used to keep foreign pointer objects
;; alive: the request bytevector holds only raw addresses, so the pointer
;; objects themselves must still be referenced after the C call returns.
(define (keep! . objects) (if (null? objects) #f #t))

;;; --- model ------------------------------------------------------------------

(define (load-image-model dir)
  "Load an image-generation model directory.

The directory is the model root, holding text_encoder/, transformer/, vae/
and tokenizer/.
Args:
  dir: the model directory.
Returns: an opaque image-model handle.
PyTorch: none
Note: Diffusion-model inference is not part of PyTorch's core library."
  (let* ((model-slot (make-slot))
         (err-slot (make-slot))
         (st ((proc 'boom_image_model_load) (string->pointer dir)
              (slot-ptr model-slot) (slot-ptr err-slot))))
    (sb-check! st err-slot "load-image-model")
    (slot-ref model-slot)))

(define (free-image-model! model)
  "Release an image-model handle.

Args:
  model: an image-model handle from load-image-model.
Returns: unspecified.
PyTorch: none"
  ((proc 'boom_image_model_free) model))

;; The architecture name, two-call sizing like the LLM surface's model-arch:
;; the sizing call fills the length (no NUL) and reports BOOM_ERR_BINDING.
(define (image-model-arch model)
  "The image model's architecture name.

Args:
  model: an image-model handle.
Returns: \"moss-image\" for the model family this binding supports.
PyTorch: none"
  (let* ((len-slot (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot)))
    (let ((st ((proc 'boom_image_model_arch) model %null-pointer 0
               (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st BOOM_ERR_BINDING))
        (sb-check! st err-slot "image-model-arch"))
      (discard-error! err-slot))
    (let* ((m (bytevector-u64-native-ref len-slot 0))
           (out (make-bytevector (+ m 1) 0))    ; +1 for the NUL terminator
           (st ((proc 'boom_image_model_arch) model (bytevector->pointer out)
                (+ m 1) (bytevector->pointer len-slot) (slot-ptr err-slot))))
      (sb-check! st err-slot "image-model-arch")
      (pointer->string (bytevector->pointer out)))))

;;; --- generation -------------------------------------------------------------

;; boom_image_request as a native-layout bytevector (x86-64; see capi.h):
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
  "Generate images from a prompt.

Args:
  model: an image-model handle.
  prompt: the text prompt.
  negative: the negative prompt used by classifier-free guidance; #f uses
            the empty prompt.
  width, height: the image size, both multiples of 16; 512 by default.
  steps: the number of denoising steps; 50 by default.
  guidance: the classifier-free guidance scale; 3.0 by default.
  seed: the noise seed; 0 by default. The same request is reproducible, and
        image i of a batch starts from the same noise as a single image with
        that seed would.
  batch: the number of images to generate; 1 by default.
  progress: a procedure (image-index step total-steps) called after each
            denoising step; returning #f cancels the generation, which
            signals a &sonicboom-error of kind 'cancelled.
Returns: a list of (width height rgb-bytevector), one per image, RGB
channels with 8 bits each.
PyTorch: none"
  (let* ((prompt-ptr (string->pointer prompt))
         (negative-ptr (and negative (string->pointer negative)))
         (req (make-image-request-bv prompt-ptr negative-ptr width height steps
                                     guidance seed batch))
         (cb (make-progress-callback progress))
         (bytes-slot (make-bytevector 8 0))
         (err-slot (make-slot)))
    ;; Two-call sizing: the byte count depends only on the request.
    (let ((st ((proc 'boom_image_generate) model (bytevector->pointer req)
               %null-pointer %null-pointer %null-pointer 0
               (bytevector->pointer bytes-slot) (slot-ptr err-slot))))
      (unless (or (sb-status-ok? st) (= st BOOM_ERR_BINDING))
        (sb-check! st err-slot "generate-images"))
      (discard-error! err-slot))
    (let* ((n (bytevector-u64-native-ref bytes-slot 0))
           (rgb (make-bytevector n 0))
           (st ((proc 'boom_image_generate) model (bytevector->pointer req)
                (if cb cb %null-pointer) %null-pointer
                (bytevector->pointer rgb) n
                (bytevector->pointer bytes-slot) (slot-ptr err-slot))))
      (sb-check! st err-slot "generate-images")
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
  "Write an RGB image to a PNG file.

The PNG is 8-bit RGB and uses stored deflate blocks; it needs no external
library.
Args:
  path: the file to write, overwritten if it exists.
  rgb: the pixels as a bytevector of width*height*3 bytes, RGB order.
  width: the image width in pixels.
  height: the image height in pixels.
Returns: #t.
Example:
  (use-modules (sonicboom image) (rnrs bytevectors))
  (write-png \"/tmp/sonicboom-docs-1x1.png\" (make-bytevector 3 255) 1 1)
  => #t
PyTorch: none
Note: An rgb bytevector shorter than width*height*3 signals a
&sonicboom-error."
  (unless (and (bytevector? rgb) (>= (bytevector-length rgb) (* width height 3)))
    (raise-sonicboom-error BOOM_ERR_BINDING 'binding
                           "write-png: rgb is shorter than width*height*3"))
  (let ((err-slot (make-slot)))
    (sb-check! ((proc 'boom_image_write_png) (string->pointer path)
                (bytevector->pointer rgb) width height (slot-ptr err-slot))
               err-slot "write-png")
    #t))
