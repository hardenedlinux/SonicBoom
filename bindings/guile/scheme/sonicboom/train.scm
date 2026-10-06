;;; SonicBoom — Guile binding: training (minimal autograd)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A thin training driver over the C ABI autograd surface (capi.h). Guile only
;;; expresses the model (by composing linear/relu/mse) and runs the training
;;; loop (zero → forward → backward → SGD step); every numeric operation is
;;; dispatched to native C++ in libsonicboom.so. Tensors cross the boundary as
;;; little-endian float32 bytevectors, never as Scheme-level arithmetic.

(define-module (sonicboom train)
  #:use-module (sonicboom ffi)
  #:use-module (rnrs bytevectors)
  #:use-module (system foreign)
  #:export (tensor-from-list
            tensor->list
            list->f32-bv
            f32-bv->list
            tensor-free!
            make-tape
            tape-free!
            tape-zero-grad!
            autograd-linear
            autograd-relu
            autograd-mse
            autograd-backward
            autograd-sgd-step!
            autograd-grad
            train-xor))

;;; --- float32 bytevector <-> Scheme list --------------------------------------

(define (list->f32-bv lst)
  (let ((bv (make-bytevector (* 4 (length lst)))))
    (let loop ((i 0) (ls lst))
      (unless (null? ls)
        (bytevector-ieee-single-native-set! bv (* i 4) (car ls))
        (loop (+ i 1) (cdr ls))))
    bv))

(define (f32-bv->list bv)
  (let ((n (/ (bytevector-length bv) 4)))
    (let loop ((i 0))
      (if (= i n)
          '()
          (cons (bytevector-ieee-single-native-ref bv (* i 4))
                (loop (+ i 1)))))))

;;; --- tensor / tape handles ---------------------------------------------------

(define (dims->pointer dims)
  ;; dims is a list of non-negative ints → pointer to an int64[] array.
  (let ((bv (make-bytevector (* 8 (length dims)) 0)))
    (let loop ((i 0) (ls dims))
      (unless (null? ls)
        (bytevector-s64-native-set! bv (* i 8) (car ls))
        (loop (+ i 1) (cdr ls))))
    (bytevector->pointer bv)))

(define (tensor-from-list dims lst)
  "Create a native float32 tensor of shape DIMS from the Scheme number list Lst.
Returns an opaque handle (pointer)."
  (let* ((bv (list->f32-bv lst))
         (out-slot (make-slot))
         (err-slot (make-slot)))
    (let ((st ((proc 'sb_tensor_from_f32) (dims->pointer dims) (length dims)
               (bytevector->pointer bv) (bytevector-length bv)
               (slot-ptr out-slot) (slot-ptr err-slot))))
      (unless (sb-status-ok? st)
        (error "tensor-from-list:" (capture-error err-slot)))
      (slot-ref out-slot))))

(define (tensor->list handle numel)
  "Read a native float32 tensor's elements back as a Scheme list."
  (let* ((bv (make-bytevector (* 4 numel)))
         (wr (make-bytevector 8 0))       ; size_t out slot (64-bit target)
         (err-slot (make-slot)))
    (let ((st ((proc 'sb_tensor_bytes) handle (bytevector->pointer bv)
               (* 4 numel) (bytevector->pointer wr) (slot-ptr err-slot))))
      (unless (sb-status-ok? st)
        (error "tensor->list:" (capture-error err-slot)))
      (f32-bv->list bv))))

(define (tensor-free! handle) ((proc 'sb_tensor_free) handle))

(define (make-tape)
  (let ((s (make-slot)) (e (make-slot)))
    (let ((st ((proc 'sb_tape_new) (slot-ptr s) (slot-ptr e))))
      (unless (sb-status-ok? st)
        (error "make-tape:" (capture-error e)))
      (slot-ref s))))

(define (tape-free! tape) ((proc 'sb_tape_free) tape))
(define (tape-zero-grad! tape) ((proc 'sb_tape_zero_grad) tape))

;;; --- differentiable ops (each records on the tape) ---------------------------

(define (call-tensor-op name args)
  (let ((out-slot (make-slot)) (err-slot (make-slot)))
    (let ((st (apply (proc name)
                     (append args (list (slot-ptr out-slot) (slot-ptr err-slot))))))
      (unless (sb-status-ok? st)
        (error (symbol->string name) ":" (capture-error err-slot)))
      (slot-ref out-slot))))

(define (call-status name args)
  (let ((err-slot (make-slot)))
    (let ((st (apply (proc name) (append args (list (slot-ptr err-slot))))))
      (unless (sb-status-ok? st)
        (error (symbol->string name) ":" (capture-error err-slot)))
      st)))

(define (autograd-linear x w b tape)
  (call-tensor-op 'sb_linear (list x w b tape)))

(define (autograd-relu x tape)
  (call-tensor-op 'sb_relu (list x tape)))

(define (autograd-mse pred target tape)
  (call-tensor-op 'sb_mse_loss (list pred target tape)))

(define (autograd-backward loss tape)
  (call-status 'sb_backward (list loss tape)))

(define (autograd-sgd-step! tape param lr)
  (call-status 'sb_sgd_step (list tape param lr)))

(define (autograd-grad tape param)
  (call-tensor-op 'sb_grad (list tape param)))

;;; --- demo: a deterministic 2-layer MLP (relu hidden) trained on XOR ----------

(define* (train-xor #:key (epochs 3000) (lr 0.5))
  "Train a 2→4→1 MLP (relu hidden) on XOR by plain SGD and return
(values initial final), the mean-squared-error before and after training.
Deterministic init; floats never leave the native side except as the reported
loss scalars."
  (define x   (tensor-from-list (list 4 2) '(0 0 0 1 1 0 1 1)))
  (define tgt (tensor-from-list (list 4 1) '(0 1 1 0)))
  (define w1  (tensor-from-list (list 4 2)
                                '(0.6 -0.3 0.2 0.5 -0.5 0.4 0.3 -0.7)))
  (define b1  (tensor-from-list (list 4) '(0.1 -0.1 0.2 -0.2)))
  (define w2  (tensor-from-list (list 1 4) '(0.5 -0.6 0.3 -0.4)))
  (define b2  (tensor-from-list (list 1) '(0.0)))
  (define tape (make-tape))

  (define (forward)
    (let* ((l1 (autograd-linear x w1 b1 tape))
           (h  (autograd-relu l1 tape))
           (l2 (autograd-linear h w2 b2 tape))
           (loss (autograd-mse l2 tgt tape)))
      (tensor-free! l1) (tensor-free! h) (tensor-free! l2)
      loss))

  (define (cleanup!)
    (tensor-free! x) (tensor-free! tgt)
    (tensor-free! w1) (tensor-free! b1) (tensor-free! w2) (tensor-free! b2)
    (tape-free! tape))

  (let loop ((e 0) (final #f) (initial #f))
    (if (>= e epochs)
        (begin (cleanup!) (values initial final))
        (begin
          (tape-zero-grad! tape)
          (let ((loss (forward)))
            (let ((v (car (tensor->list loss 1))))
              (autograd-backward loss tape)
              (autograd-sgd-step! tape w1 lr)
              (autograd-sgd-step! tape b1 lr)
              (autograd-sgd-step! tape w2 lr)
              (autograd-sgd-step! tape b2 lr)
              (tensor-free! loss)
              (loop (+ e 1) v (or initial v))))))))
