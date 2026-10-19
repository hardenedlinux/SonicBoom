;;; SonicBoom — R7: parameter checkpoint save/load through the Guile binding
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; (sonicboom checkpoint) marshals ctensors through sb_save_params /
;;; sb_load_params. The test writes a checkpoint of several named tensors
;;; (model parameters and AdamW moment buffers), loads it back in a different
;;; order into freshly initialised templates, and requires every value to be
;;; bit-identical. It also checks the error paths: a non-float32 tensor is
;;; rejected without overwriting an existing checkpoint, a missing name and a
;;; shape mismatch are rejected at load time.
;;;
;;; Run from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-checkpoint.scm

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 format)
             (sonicboom ffi)
             (sonicboom checkpoint)
             (sonicboom trace tensor))

(define path (format #f "/tmp/sonicboom-checkpoint-~a.ckpt" (getpid)))

(define (cleanup!) (when (file-exists? path) (delete-file path)))

(unless (libsonicboom-available?)
  (format #t "SKIP: libsonicboom.so could not be loaded~%")
  (exit 0))

(cleanup!)
(test-begin "sonicboom-checkpoint")

;;; Values are multiples of 1/32 plus -0.5, exactly representable in float32,
;;; so a round-trip can be compared for exact equality.
(define (tensor shape)
  (make-ctensor 'float32 shape
                (map (lambda (i) (- (* 0.03125 i) 0.5))
                     (iota (shape-numel shape)))))

(define emb (tensor '(6 4)))
(define norm-w (tensor '(4)))
(define bias (tensor '(1)))
(define empty (make-ctensor 'float32 '(0 2) '()))
(define m1 (tensor '(6 4)))
(define step (make-ctensor 'float32 '(1) '(3.0)))

(define entries
  (list (cons 'embedding emb)
        (cons "norm" norm-w)
        (cons 'bias bias)
        (cons 'empty empty)
        (cons 'adamw-m1 m1)
        (cons 'step step)))

(test-assert "save-params! writes the checkpoint"
  (begin (save-params! path entries) (file-exists? path)))

;; Load in a different order, into tensors initialised to junk: the values must
;; come from the file, not from the templates.
(define (junk shape)
  (make-ctensor 'float32 shape (make-list (shape-numel shape) 99.0)))

(define templates
  (list (cons 'step (junk '(1)))
        (cons 'adamw-m1 (junk '(6 4)))
        (cons 'empty (junk '(0 2)))
        (cons 'bias (junk '(1)))
        (cons 'norm (junk '(4)))
        (cons 'embedding (junk '(6 4)))))

(define loaded (load-params! path templates))
(define (loaded-ref name) (assoc-ref loaded name))

(test-assert "entry order follows the request"
  (equal? (map car loaded) (map car templates)))

(test-equal "embedding round-trips bit-exactly"
  (ctensor->list emb) (ctensor->list (loaded-ref 'embedding)))
(test-equal "norm round-trips bit-exactly"
  (ctensor->list norm-w) (ctensor->list (loaded-ref 'norm)))
(test-equal "bias round-trips bit-exactly"
  (ctensor->list bias) (ctensor->list (loaded-ref 'bias)))
(test-equal "zero-element tensor round-trips"
  (ctensor->list empty) (ctensor->list (loaded-ref 'empty)))
(test-equal "AdamW moment round-trips bit-exactly"
  (ctensor->list m1) (ctensor->list (loaded-ref 'adamw-m1)))
(test-equal "step counter round-trips"
  (ctensor->list step) (ctensor->list (loaded-ref 'step)))

(test-equal "loading twice is idempotent"
  (ctensor->list emb)
  (ctensor->list (assoc-ref (load-params! path templates) 'embedding)))

;; A subset load: the requested entry comes from the file and extra entries in
;; the file are ignored.
(test-equal "a subset of entries can be loaded"
  (ctensor->list norm-w)
  (ctensor->list (assoc-ref (load-params! path (list (cons 'norm (junk '(4)))))
                            'norm)))

;;; --- error paths -------------------------------------------------------------

(define (signals? thunk)
  (catch #t thunk (lambda _ #t)))

(test-assert "a non-float32 tensor is rejected on save"
  (signals? (lambda ()
              (save-params! path (list (cons 'bad (make-ctensor 'int32 '(2) '(1 2))))))))

(test-assert "the rejected save left the previous checkpoint intact"
  (equal? (ctensor->list emb)
          (ctensor->list (assoc-ref (load-params! path templates) 'embedding))))

(test-assert "a name missing from the checkpoint is rejected"
  (signals? (lambda ()
              (load-params! path (list (cons 'embedding (junk '(6 4)))
                                       (cons 'absent (junk '(2))))))))

(test-assert "a shape mismatch is rejected"
  (signals? (lambda ()
              (load-params! path (list (cons 'norm (junk '(5))))))))

(test-assert "a missing checkpoint file is rejected"
  (signals? (lambda ()
              (load-params! "/nonexistent/sonicboom.ckpt"
                            (list (cons 'norm (junk '(4))))))))

(cleanup!)
(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-checkpoint")
(exit (if (zero? failures) 0 1))
