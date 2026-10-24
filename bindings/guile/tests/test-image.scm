;;; SonicBoom — Guile image-generation binding test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Loads the MOSS-Image model through (sonicboom image) and checks:
;;;
;;;   1. the architecture name through the two-call sizing path;
;;;   2. generate-images returns (width height rgb) images of the requested
;;;      size, and the same request twice is byte-identical;
;;;   3. image 0 of a batch of 2 equals the single-image result;
;;;   4. #:progress sees (image step total) after each step, and returning #f
;;;      cancels with an 'sb-image exception;
;;;   5. write-png writes the 2x2 known-answer PNG byte-for-byte.
;;;
;;; Skips cleanly without the model or a CUDA device (the f32 CPU DiT is far
;;; too slow for a unit test).

(use-modules (srfi srfi-64)
             (rnrs bytevectors)
             (ice-9 binary-ports)
             (sonicboom ffi)
             (sonicboom image))

(define (find-model)
  (let ((cands
         (filter (lambda (p) (and p (file-exists? p)
                                  (file-exists? (string-append p "/transformer"))))
                 (list
                  (getenv "SONICBOOM_MOSS_IMAGE_DIR")
                  (and=> (getenv "SONICBOOM_SRC_DIR")
                         (lambda (d)
                           (string-append d "/models/MOSS-Image-SFT")))
                  "models/MOSS-Image-SFT"
                  "../../models/MOSS-Image-SFT"))))
    (and (pair? cands) (car cands))))

(define (read-file path)
  (call-with-input-file path get-bytevector-all))

(define (u32be bv off)
  (+ (* (bytevector-u8-ref bv off) 16777216)
     (* (bytevector-u8-ref bv (+ off 1)) 65536)
     (* (bytevector-u8-ref bv (+ off 2)) 256)
     (bytevector-u8-ref bv (+ off 3))))

(test-begin "sonicboom-image")

(let ((dir (find-model)))
  (if (or (not dir)
          (not (file-exists? "/dev/nvidia0"))
          (not (libsonicboom-available?)))
      (test-skip 10)
      (let* ((model (load-image-model dir))
             (w 64) (h 64))
        (test-equal "the architecture is moss-image"
          "moss-image" (image-model-arch model))

        (let* ((once (generate-images model #:prompt "a red fox"
                                      #:width w #:height h #:steps 2
                                      #:seed 5))
               (image (car once)))
          (test-equal "one image in the single-image request" 1 (length once))
          (test-equal "the image has the requested width" w (car image))
          (test-equal "the image has the requested height" h (cadr image))
          (test-equal "the image is RGB HWC"
            (* w h 3) (bytevector-length (caddr image)))

          (let ((again (generate-images model #:prompt "a red fox"
                                        #:width w #:height h #:steps 2
                                        #:seed 5)))
            (test-assert "the same request is deterministic"
              (equal? (caddr image) (caddr (car again)))))

          (let* ((pair (generate-images model #:prompt "a red fox"
                                        #:width w #:height h #:steps 2
                                        #:seed 5 #:batch 2)))
            (test-equal "two images in the batch request" 2 (length pair))
            (test-assert "batch image 0 equals the single image"
              (equal? (caddr image) (caddr (car pair)))))

          ;; Progress: two steps, steps counted 1-based.
          (let ((seen '()))
            (generate-images model #:prompt "a red fox" #:width w #:height h
                             #:steps 2 #:seed 5
                             #:progress (lambda (i step total)
                                          (set! seen
                                                (cons (list i step total) seen))
                                          #t))
            (test-equal "progress saw both steps, 1-based"
              '((0 2 2) (0 1 2)) seen))

          ;; Cancelling from the callback surfaces as an 'sb-image exception.
          (test-assert "a #f from progress cancels"
            (catch 'sb-image
              (lambda ()
                (generate-images model #:prompt "a red fox" #:width w
                                 #:height h #:steps 2 #:seed 5
                                 #:progress (lambda (i step total) #f))
                #f)
              (lambda args #t))))

        (free-image-model! model))))

;;; write-png: the 2x2 known-answer image (red, green / blue, white).
(let ((path "/tmp/sb-guile-test-image.png"))
  (when (file-exists? path) (delete-file path))
  (write-png path #vu8(255 0 0  0 255 0  0 0 255  255 255 255) 2 2)
  (let ((png (read-file path)))
    (delete-file path)
    (test-equal "the 2x2 PNG is 82 bytes" 82 (bytevector-length png))
    (test-equal "the PNG signature"
      #vu8(137 80 78 71 13 10 26 10)
      (let ((sig (make-bytevector 8 0)))
        (bytevector-copy! png 0 sig 0 8)
        sig))
    (test-equal "the IHDR width is 2" 2 (u32be png 16))
    (test-equal "the IHDR height is 2" 2 (u32be png 20))))

(test-end "sonicboom-image")
