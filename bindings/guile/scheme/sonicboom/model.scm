;;; SonicBoom — Guile binding: unified trainable model
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Composes the (sonicboom nn) layer structure with the (sonicboom train)
;;; autograd surface into one trainable model object. A model holds:
;;;
;;;   - the layer structure (the S-Expr deployment graph's source of truth), and
;;;   - one native sb_tensor per parameter — the single authoritative store.
;;;
;;; Training mutates those native tensors in place through the C ABI autograd;
;;; export writes them out through sb_export_model. Deployment (S-Expr → MLIR)
;;; and training (native-torch autograd) are two distinct engines that share
;;; structure, parameter naming, shapes and math semantics — never a graph
;;; object. No per-element arithmetic happens in Scheme.

(define-module (sonicboom model)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 optargs)
  #:use-module (system foreign)
  #:use-module ((sonicboom tensor)
                #:select (tensor-shape tensor-numel tensor-dtype
                          tensor->list tensor->bytevector))
  #:use-module ((sonicboom nn)
                #:select (model-layers model-parameters model-graph-text
                          layer-kind layer-name layer-parameters))
  #:use-module ((sonicboom train) #:prefix train:)
  #:use-module ((sonicboom ffi)
                #:select (proc make-slot slot-ptr capture-error sb-status-ok?))
  #:export (nn-model? nn-model-structure nn-model-params nn-model-input-dims
            make-model model-param model-forward-native model-train!
            model-export!))

;;; --- the model record -------------------------------------------------------

;; A trainable model: an (sonicboom nn) structure and native parameters.
;;
;; Each parameter of the structure becomes one native sb_tensor, which is the
;; authoritative store that training updates in place and export writes out. The
;; input dimensions are part of the record because they are fixed per model.
;; PyTorch: torch.nn.Module
;; Note: The structure decides the shape and naming of the parameters, while the
;; values live in native tensors rather than in Scheme.
(define-record-type <nn-model>
  (%make-nn-model structure params input-dims)
  nn-model?
  (structure   nn-model-structure)   ; (sonicboom nn) <model>
  (params      nn-model-params)      ; alist of (name-string . sb_tensor)
  (input-dims  nn-model-input-dims)) ; list of ints; the input tensor's shape

(define (model-param model name)
  "Look up a native parameter by its \"<layer>.<key>\" name.

Args:
  model: a trainable model.
  name: the parameter name as a string.
Returns: the native sb_tensor handle.
Example:
  (use-modules (sonicboom model) (sonicboom nn) (system foreign))
  (let ((m (make-model '(1 2) (sequential (linear 2 1 #:name 'l)))))
    (pointer? (model-param m \"l.weight\")))
  => #t
PyTorch: none
Note: PyTorch reaches the same tensor as a named parameter of the module."
  (or (assoc-ref (nn-model-params model) name)
      (error "model-param: no such parameter" name)))

;;; --- construction -----------------------------------------------------------

(define* (make-model input-dims structure #:key init-values)
  "Build a trainable model from an (sonicboom nn) structure.

Each parameter of the structure becomes a native sb_tensor, seeded from the
layer's Guile tensor. The input dimensions are recorded on the model.
Args:
  input-dims: the input shape as a list of integers.
  structure: a model made with sequential.
  init-values: an alist of \"<layer>.<key>\" to flat number list overriding the
               seed, for deterministic training.
Returns: a trainable model.
Example:
  (use-modules (sonicboom model) (sonicboom nn))
  (nn-model-input-dims (make-model '(1 2) (sequential (linear 2 3))))
  => (1 2)
PyTorch: torch.nn.Module
Note: A PyTorch module is built by its constructor and finds its parameters in
the module tree; here they are copied into native tensors at construction."
  (let ((params
         (map (lambda (pair)
                (let* ((sym  (car pair))
                       (name (symbol->string sym))
                       (gt   (cdr pair))               ; Guile <tensor>
                       (dims (tensor-shape gt))
                       (override (and init-values (assoc-ref init-values name))))
                  (cons name
                        (if override
                            (begin
                              (unless (= (length override) (tensor-numel gt))
                                (error "make-model: init-values length mismatch for"
                                       name))
                              (train:tensor-from-list dims override))
                            ;; sb_tensor_from_f32 takes float32 bytes, so a
                            ;; float32 parameter crosses as its own bytes — no
                            ;; per-element Scheme list. Other dtypes keep the
                            ;; numeric path, which converts their values to
                            ;; float32 exactly as before.
                            (if (eq? (tensor-dtype gt) 'float32)
                                (train:tensor-from-bytes
                                 dims (tensor->bytevector gt))
                                (train:tensor-from-list
                                 dims (tensor->list gt)))))))
              (model-parameters structure))))
    (%make-nn-model structure params input-dims)))

;;; --- forward (native-torch autograd) ----------------------------------------

(define (apply-layer-forward model layer x tape)
  "One layer's autograd step over native tensor X, recording on TAPE."
  (case (layer-kind layer)
    ((linear)
     (let* ((nm (symbol->string (layer-name layer)))
            (params (nn-model-params model))
            (w (assoc-ref params (string-append nm ".weight")))
            (b (assoc-ref params (string-append nm ".bias"))))
       (unless (and w b)
         (error "apply-layer-forward: linear layer" nm
                "requires weight and bias in v0"))
       (train:autograd-linear x w b tape)))
    ((relu) (train:autograd-relu x tape))
    (else (error "apply-layer-forward: unknown layer kind" (layer-kind layer)))))

(define (model-forward-native model x tape)
  "Run a model's layers forward over a native tensor.

Every intermediate tensor allocated during the pass is freed; the input is
left untouched. The pass records on the tape so a backward pass can follow.
Args:
  model: a trainable model.
  x: a native input tensor.
  tape: the autograd tape that records the pass.
Returns: the native output tensor.
Example:
  (use-modules (sonicboom model) (sonicboom nn) (sonicboom tensor)
               ((sonicboom train) #:prefix t:))
  (let* ((m (make-model '(1 2) (sequential (linear 2 1 #:init 'ones))))
         (tape (t:make-tape))
         (x (t:tensor-from-list '(1 2) '(1.0 2.0)))
         (y (model-forward-native m x tape))
         (out (t:tensor->list y 1)))
    (t:tensor-free! y) (t:tensor-free! x) (t:tape-free! tape)
    out)
  => (3.0)
PyTorch: torch.nn.Module.forward
Note: The caller owns the returned tensor and must free it. A linear layer
must carry both a weight and a bias in v0, so a bias-less linear is rejected."
  (let loop ((layers (model-layers (nn-model-structure model)))
             (cur x)
             (own-cur? #f))
    (if (null? layers)
        cur
        (let ((next (apply-layer-forward model (car layers) cur tape)))
          (when own-cur? (train:tensor-free! cur))
          (loop (cdr layers) next #t)))))

;;; --- training (plain SGD) ---------------------------------------------------

(define* (model-train! model x tgt #:key (epochs 1000) (lr 0.5)
                       (max-grad-norm #f))
  "Train a model by plain SGD on a full dataset.

Parameters are updated in place for the given number of epochs; the loss is
the mean squared error.
Args:
  model: a trainable model.
  x: the native input dataset, shape [N, in].
  tgt: the native target dataset, shape [N, out].
  epochs: the number of passes over the dataset.
  lr: the SGD learning rate.
  max-grad-norm: when a number, the global gradient norm is clipped to it after
                 every backward pass.
Returns: two values, the initial and the final loss.
Example:
  (use-modules (sonicboom model) (sonicboom nn)
               ((sonicboom train) #:prefix t:))
  (let* ((m (make-model '(1 1) (sequential (linear 1 1 #:init 'ones))))
         (x (t:tensor-from-list '(1 1) '(1.0)))
         (tgt (t:tensor-from-list '(1 1) '(2.0))))
    (call-with-values (lambda () (model-train! m x tgt #:epochs 2 #:lr 0.1))
                      list))
PyTorch: none
Note: In PyTorch a training loop combines an optimizer's step with backward();
here one call performs a fixed number of full-batch SGD steps."
  (let ((tape (train:make-tape)))
    (define (step)
      (train:tape-zero-grad! tape)
      (let* ((pred (model-forward-native model x tape))
             (loss (train:autograd-mse pred tgt tape)))
        (let ((v (car (train:tensor->list loss 1))))
          (train:autograd-backward loss tape)
          (when max-grad-norm
            (train:clip-grad-norm! tape (map cdr (nn-model-params model))
                                   max-grad-norm))
          (for-each (lambda (p) (train:autograd-sgd-step! tape (cdr p) lr))
                    (nn-model-params model))
          (train:tensor-free! pred)
          (train:tensor-free! loss)
          v)))
    (let loop ((e 0) (final #f) (initial #f))
      (if (>= e epochs)
          (begin (train:tape-free! tape) (values initial final))
          (let ((v (step)))
            (loop (+ e 1) v (or initial v)))))))

;;; --- export -----------------------------------------------------------------

(define (model-export! model dir)
  "Write a model's graph text and trained parameters to a directory.

The output is model.sx plus weights.bin, written through sb_export_model. The
artifact loads independently in C++ with no Guile or Python.
Args:
  model: a trainable model.
  dir: the target directory.
Returns: unspecified.
PyTorch: none
Note: The nearest PyTorch operation is torch.onnx.export, which belongs to a
part of PyTorch that SonicBoom does not mirror."
  (let* ((text (model-graph-text (nn-model-structure model)
                                 (nn-model-input-dims model)))
         (params (nn-model-params model))
         (names (map car params))
         (tensors (map cdr params))
         (name-ptrs (map string->pointer names))  ; keep alive across the call
         (names-arr (make-c-struct (map (lambda (_) '*) names) name-ptrs))
         (params-arr (make-c-struct (map (lambda (_) '*) tensors) tensors))
         (err-slot (make-slot))
         (st ((proc 'sb_export_model) (string->pointer text) (string-length text)
              names-arr params-arr (length params) (string->pointer dir)
              (slot-ptr err-slot))))
    (unless (sb-status-ok? st)
      (error "model-export!:" (capture-error err-slot)))))
