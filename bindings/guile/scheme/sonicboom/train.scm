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
            tensor-from-bytes
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
            clip-grad-norm!
            train-xor))

;;; --- float32 bytevector <-> Scheme list --------------------------------------

(define (list->f32-bv lst)
  "Pack a list of Scheme numbers into little-endian IEEE-754 float32 bytes.

Args:
  lst: a list of real numbers.
Returns: a bytevector of four bytes per number.
Example:
  (use-modules (sonicboom train))
  (bytevector-length (list->f32-bv '(1.5 2.5)))
  => 8
PyTorch: none
Note: This is a conversion helper for the C ABI, which moves float32 tensors as
bytevectors."
  (let ((bv (make-bytevector (* 4 (length lst)))))
    (let loop ((i 0) (ls lst))
      (unless (null? ls)
        (bytevector-ieee-single-native-set! bv (* i 4) (car ls))
        (loop (+ i 1) (cdr ls))))
    bv))

(define (f32-bv->list bv)
  "Read little-endian IEEE-754 float32 bytes back as Scheme numbers.

Args:
  bv: a bytevector whose length is a multiple of four.
Returns: a list of reals, one per four bytes.
Example:
  (use-modules (sonicboom train))
  (f32-bv->list (list->f32-bv '(1.5 2.5)))
  => (1.5 2.5)
PyTorch: none
Note: The inverse of list->f32-bv."
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

(define (tensor-from-bytes dims bv)
  "Create a native float32 tensor from little-endian float32 bytes.

The C ABI copies the bytes, so the caller may reuse the bytevector.
Args:
  dims: the tensor shape as a list of non-negative integers.
  bv: the float32 element bytes, in row-major order.
Returns: an opaque native tensor handle.
Example:
  (use-modules (sonicboom train))
  (let ((t (tensor-from-bytes '(2) (list->f32-bv '(1.0 2.0)))))
    (let ((v (tensor->list t 2))) (tensor-free! t) v))
  => (1.0 2.0)
PyTorch: torch.frombuffer
Note: This path does no per-element Scheme conversion; use it when the source
already holds float32 bytes."
  (let* ((out-slot (make-slot))
         (err-slot (make-slot)))
    (let ((st ((proc 'boom_tensor_from_f32) (dims->pointer dims) (length dims)
               (bytevector->pointer bv) (bytevector-length bv)
               (slot-ptr out-slot) (slot-ptr err-slot))))
      (unless (sb-status-ok? st)
        (error "tensor-from-bytes:" (capture-error err-slot)))
      (slot-ref out-slot))))

(define (tensor-from-list dims lst)
  "Create a native float32 tensor from a list of numbers.

Args:
  dims: the tensor shape as a list of non-negative integers.
  lst: the elements in row-major order.
Returns: an opaque native tensor handle.
Example:
  (use-modules (sonicboom train))
  (let ((t (tensor-from-list '(2) '(1.0 2.0))))
    (let ((v (tensor->list t 2))) (tensor-free! t) v))
  => (1.0 2.0)
PyTorch: torch.tensor"
  (tensor-from-bytes dims (list->f32-bv lst)))

(define (tensor->list handle numel)
  "Read a native float32 tensor's elements as a Scheme list.

Args:
  handle: a native tensor handle.
  numel: the number of elements to read, normally the product of the shape.
Returns: the elements in row-major order.
Example:
  (use-modules (sonicboom train))
  (let ((t (tensor-from-list '(2) '(1.0 2.0))))
    (let ((v (tensor->list t 2))) (tensor-free! t) v))
  => (1.0 2.0)
PyTorch: torch.Tensor.tolist"
  (let* ((bv (make-bytevector (* 4 numel)))
         (wr (make-bytevector 8 0))       ; size_t out slot (64-bit target)
         (err-slot (make-slot)))
    (let ((st ((proc 'boom_tensor_bytes) handle (bytevector->pointer bv)
               (* 4 numel) (bytevector->pointer wr) (slot-ptr err-slot))))
      (unless (sb-status-ok? st)
        (error "tensor->list:" (capture-error err-slot)))
      (f32-bv->list bv))))

(define (tensor-free! handle)
  "Release a native tensor handle.

Args:
  handle: a native tensor handle.
Returns: unspecified.
PyTorch: none
Note: The caller owns every tensor obtained from this module and must release
it; PyTorch tensors are released by the garbage collector."
  ((proc 'boom_tensor_free) handle))

(define (make-tape)
  "Create an autograd tape that records differentiable operations.

Args:
Returns: an opaque tape handle.
Example:
  (use-modules (sonicboom train))
  (let ((tape (make-tape))) (tape-free! tape) 'ok)
  => ok
PyTorch: none
Note: PyTorch records on an implicit graph inside each tensor; here the tape
is an explicit object passed to every differentiable operation."
  (let ((s (make-slot)) (e (make-slot)))
    (let ((st ((proc 'boom_tape_new) (slot-ptr s) (slot-ptr e))))
      (unless (sb-status-ok? st)
        (error "make-tape:" (capture-error e)))
      (slot-ref s))))

(define (tape-free! tape)
  "Release a tape handle.

Args:
  tape: a tape handle from make-tape.
Returns: unspecified.
PyTorch: none"
  ((proc 'boom_tape_free) tape))
(define (tape-zero-grad! tape)
  "Clear the gradients the tape has accumulated.

Call this before a forward pass when starting a new step; the zero is applied
to the tape's gradient store, not to the parameter tensors.
Args:
  tape: a tape handle.
Returns: unspecified.
PyTorch: torch.optim.Optimizer.zero_grad"
  ((proc 'boom_tape_zero_grad) tape))

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
  "A differentiable affine transform, y = x W-transpose + b.

The operation is appended to the tape.
Args:
  x: the native input tensor, shape [batch, in].
  w: the native weight tensor, shape [out, in].
  b: the native bias tensor, shape [out].
  tape: the tape to record on.
Returns: the native output tensor, shape [batch, out].
Example:
  (use-modules (sonicboom train))
  (let* ((x (tensor-from-list '(1 1) '(2.0)))
         (w (tensor-from-list '(1 1) '(3.0)))
         (b (tensor-from-list '(1) '(0.5)))
         (tape (make-tape)))
    (tensor->list (autograd-linear x w b tape) 1))
  => (6.5)
PyTorch: torch.nn.functional.linear
Note: The caller owns the returned tensor."
  (call-tensor-op 'boom_linear (list x w b tape)))

(define (autograd-relu x tape)
  "A differentiable rectified linear unit.

The operation is appended to the tape.
Args:
  x: the native input tensor.
  tape: the tape to record on.
Returns: the native output tensor, the same shape as x.
Example:
  (use-modules (sonicboom train))
  (let* ((x (tensor-from-list '(2) '(1.0 -2.0)))
         (tape (make-tape)))
    (tensor->list (autograd-relu x tape) 2))
  => (1.0 0.0)
PyTorch: torch.nn.functional.relu
Note: The caller owns the returned tensor."
  (call-tensor-op 'boom_relu (list x tape)))

(define (autograd-mse pred target tape)
  "A differentiable mean squared error.

The operation is appended to the tape.
Args:
  pred: the native prediction tensor.
  target: the native target tensor, the same shape as pred.
  tape: the tape to record on.
Returns: a native tensor holding the mean squared error.
Example:
  (use-modules (sonicboom train))
  (let* ((p (tensor-from-list '(1) '(2.0)))
         (t (tensor-from-list '(1) '(0.0)))
         (tape (make-tape)))
    (tensor->list (autograd-mse p t tape) 1))
  => (4.0)
PyTorch: torch.nn.functional.mse_loss
Note: The reduction is always a mean over all elements."
  (call-tensor-op 'boom_mse_loss (list pred target tape)))

(define (autograd-backward loss tape)
  "Backpropagate from a loss through the recorded operations.

After the call, autograd-grad returns each parameter's gradient and
autograd-sgd-step! can apply an update.
Args:
  loss: a native scalar tensor produced by a recorded operation.
  tape: the tape that recorded the forward pass.
Returns: unspecified.
PyTorch: torch.Tensor.backward"
  (call-status 'boom_backward (list loss tape)))

(define (autograd-sgd-step! tape param lr)
  "Apply one plain SGD update to a parameter, in place.

Updates the parameter as p - lr * grad(p), using the gradient that
autograd-backward recorded on the tape.
Args:
  tape: a tape holding gradients.
  param: a native parameter tensor that took part in the forward pass.
  lr: the learning rate.
Returns: unspecified.
Example:
  (use-modules (sonicboom train))
  (let* ((x (tensor-from-list '(1 1) '(2.0)))
         (w (tensor-from-list '(1 1) '(3.0)))
         (b (tensor-from-list '(1) '(0.0)))
         (tgt (tensor-from-list '(1 1) '(8.0)))
         (tape (make-tape))
         (y (autograd-linear x w b tape))
         (loss (autograd-mse y tgt tape)))
    (autograd-backward loss tape)
    (autograd-sgd-step! tape w 0.1)
    (tensor->list w 1))
  => (3.799999952316284)
PyTorch: torch.optim.SGD.step"
  (call-status 'boom_sgd_step (list tape param lr)))

(define (autograd-grad tape param)
  "The gradient autograd-backward accumulated for a parameter.

Args:
  tape: a tape holding gradients.
  param: a native parameter tensor that took part in the forward pass.
Returns: a new native tensor holding the gradient.
Example:
  (use-modules (sonicboom train))
  (let* ((x (tensor-from-list '(1 1) '(2.0)))
         (w (tensor-from-list '(1 1) '(3.0)))
         (b (tensor-from-list '(1) '(0.0)))
         (tgt (tensor-from-list '(1 1) '(8.0)))
         (tape (make-tape))
         (y (autograd-linear x w b tape))
         (loss (autograd-mse y tgt tape)))
    (autograd-backward loss tape)
    (tensor->list (autograd-grad tape w) 1))
  => (-8.0)
PyTorch: torch.Tensor.grad
Note: The caller owns the returned tensor."
  (call-tensor-op 'boom_grad (list tape param)))

;;; --- global-norm gradient clipping ------------------------------------------

(define (clip-grad-norm! tape params max-norm)
  "Clip accumulated gradients in place by their global L2 norm.

The norm is accumulated in double over every element of every gradient. When
it exceeds max-norm, each gradient is scaled by max-norm / (norm + 1e-6).
Args:
  tape: a tape holding gradients from autograd-backward.
  params: a list of native parameter tensor handles.
  max-norm: the clipping threshold.
Returns: the pre-clip global norm as a real.
Example:
  (use-modules (sonicboom train))
  (let* ((x (tensor-from-list '(1 1) '(1.5)))
         (w (tensor-from-list '(1 1) '(1.0)))
         (b (tensor-from-list '(1) '(0.0)))
         (tgt (tensor-from-list '(1 1) '(2.5)))
         (tape (make-tape))
         (y (autograd-linear x w b tape))
         (loss (autograd-mse y tgt tape)))
    (autograd-backward loss tape)
    (clip-grad-norm! tape (list w) 1.0))
  => 3.0
PyTorch: torch.nn.utils.clip_grad_norm_"
  (let* ((grads (map (lambda (p) (autograd-grad tape p)) params))
         (count (length grads))
         (ptr-bv (make-bytevector (* 8 count) 0))
         (norm-bv (make-bytevector 4 0))
         (err-slot (make-slot)))
    (let loop ((i 0) (gs grads))
      (unless (null? gs)
        (bytevector-u64-native-set! ptr-bv (* 8 i)
                                    (pointer-address (car gs)))
        (loop (+ i 1) (cdr gs))))
    (let ((st ((proc 'boom_clip_grad_norm)
               (bytevector->pointer ptr-bv) count (exact->inexact max-norm)
               (bytevector->pointer norm-bv) (slot-ptr err-slot))))
      (for-each tensor-free! grads)
      (unless (sb-status-ok? st)
        (error "clip-grad-norm!:" (capture-error err-slot)))
      (bytevector-ieee-single-native-ref norm-bv 0))))

;;; --- demo: a deterministic 2-layer MLP (relu hidden) trained on XOR ----------

(define* (train-xor #:key (epochs 3000) (lr 0.5))
  "Train a small multilayer perceptron on XOR and report the loss.

A 2 to 4 to 1 network with a relu hidden layer, deterministic initial
parameters, plain SGD, and mean squared error. The data and every gradient
stay native; only the reported losses cross back into Scheme.
Args:
  epochs: the number of training steps.
  lr: the learning rate.
Returns: two values, the loss before and after training.
Example:
  (use-modules (sonicboom train))
  (call-with-values (lambda () (train-xor #:epochs 3000))
    (lambda (initial final) (< final initial)))
  => #t
PyTorch: none
Note: This is the module's self-check; it is not a general training API."
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
