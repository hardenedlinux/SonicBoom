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
                #:select (tensor-shape tensor-numel tensor->list))
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

(define-record-type <nn-model>
  (%make-nn-model structure params input-dims)
  nn-model?
  (structure   nn-model-structure)   ; (sonicboom nn) <model>
  (params      nn-model-params)      ; alist of (name-string . sb_tensor)
  (input-dims  nn-model-input-dims)) ; list of ints; the input tensor's shape

(define (model-param model name)
  "Look up a native parameter sb_tensor by its \"<layer>.<key>\" name."
  (or (assoc-ref (nn-model-params model) name)
      (error "model-param: no such parameter" name)))

;;; --- construction -----------------------------------------------------------

(define* (make-model input-dims structure #:key init-values)
  "Build a trainable model from an (sonicboom nn) STRUCTURE. Each parameter of
STRUCTURE becomes a native sb_tensor (the authoritative store), seeded from the
layer's Guile tensor. INIT-VALUES, when given, is an alist of
(\"<layer>.<key>\" . flat-number-list) overriding the seed, for deterministic
training."
  (let ((params
         (map (lambda (pair)
                (let* ((sym  (car pair))
                       (name (symbol->string sym))
                       (gt   (cdr pair))               ; Guile <tensor>
                       (dims (tensor-shape gt))
                       (override (and init-values (assoc-ref init-values name)))
                       (lst  (or override (tensor->list gt))))
                  (unless (= (length lst) (tensor-numel gt))
                    (error "make-model: init-values length mismatch for" name))
                  (cons name (train:tensor-from-list dims lst))))
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
  "Run MODEL's layers forward over native tensor X on TAPE. Returns the output
sb_tensor; every intermediate tensor it allocates is freed, and X (caller-owned)
is left untouched."
  (let loop ((layers (model-layers (nn-model-structure model)))
             (cur x)
             (own-cur? #f))
    (if (null? layers)
        cur
        (let ((next (apply-layer-forward model (car layers) cur tape)))
          (when own-cur? (train:tensor-free! cur))
          (loop (cdr layers) next #t)))))

;;; --- training (plain SGD) ---------------------------------------------------

(define* (model-train! model x tgt #:key (epochs 1000) (lr 0.5))
  "Train MODEL by plain SGD on the full dataset X (native [N,in]) and TGT
(native [N,out]) for EPOCHS iterations, updating parameters in place. Returns
(values initial-loss final-loss), the mean-squared error before/after."
  (let ((tape (train:make-tape)))
    (define (step)
      (train:tape-zero-grad! tape)
      (let* ((pred (model-forward-native model x tape))
             (loss (train:autograd-mse pred tgt tape)))
        (let ((v (car (train:tensor->list loss 1))))
          (train:autograd-backward loss tape)
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
  "Write MODEL's graph text plus its trained native parameters to DIR
(model.sx + weights.bin) via sb_export_model. The artifact loads independently
in C++ with no Guile or Python."
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
