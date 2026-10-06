;;; SonicBoom — Guile binding: layers & model composition
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A minimal PyTorch/Keras-like layer API on top of the (sonicboom tensor) and
;;; (sonicboom ffi) modules. A layer is a named bundle of parameters plus a
;;; `forward` step that emits S-Expr v0.1 node text; a model composes layers into
;;; a single topologically-ordered graph and runs it through the native C++
;;; backends via the C ABI. No tensor arithmetic happens in Scheme: forward is
;;; always lowered to native `gemm` / `relu` nodes and executed by libsonicboom.so.

(define-module (sonicboom nn)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (ice-9 optargs)
  #:use-module (sonicboom tensor)
  #:use-module (sonicboom ffi)
  #:export (layer? layer-kind layer-name layer-parameters layer-parameter-ref
            model? model-layers model-parameters model-parameter-ref
            model-graph-text linear relu sequential model-forward))

;;; --- text formatting (S-Expr v0.1) -----------------------------------------

(define (string->sx s)
  ;; Escape the characters the S-Expr lexer treats specially inside strings.
  (list->string
   (append-map
    (lambda (c)
      (case c
        ((#\\) '(#\\ #\\))
        ((#\") '(#\\ #\"))
        ((#\newline) '(#\\ #\n))
        ((#\return) '(#\\ #\r))
        ((#\tab) '(#\\ #\t))
        (else (list c))))
    (string->list s))))

(define (float->sx x)
  ;; Guarantee a decimal point / exponent so the lexer reads a Float, never an Int.
  (let ((s (number->string x)))
    (if (or (string-index s #\.) (string-index s #\e) (string-index s #\E))
        s
        (string-append s ".0"))))

(define (shape->sx dims)
  (string-append "(shape " (string-join (map number->string dims) " ") ")"))

(define (type->sx dtype dims)
  (string-append "(tensor " (symbol->string dtype) " " (shape->sx dims) ")"))

(define (values->sx t)
  (string-join (map float->sx (tensor->list t)) " "))

(define (quoted name) (string-append "\"" (string->sx name) "\""))

;;; --- layer naming -----------------------------------------------------------

(define %layer-id (let ((n 0)) (lambda () (set! n (+ n 1)) n)))

(define (default-layer-name kind)
  (string->symbol (string-append (symbol->string kind)
                                 (number->string (%layer-id)))))

;;; --- layers -----------------------------------------------------------------

(define-record-type <layer>
  (%make-layer kind name parameters apply)
  layer?
  (kind       layer-kind)         ; 'linear | 'relu (drives native-torch forward)
  (name       layer-name)
  (parameters layer-parameters)   ; alist of (key . tensor)
  (apply      layer-apply))       ; (x-name x-dims mk reg) -> (node-strs out-name out-dims)

(define (layer-parameter-ref layer key)
  (assq-ref (layer-parameters layer) key))

(define (initializer dims init)
  (case init
    ((zeros) (zeros dims))
    ((ones)  (ones dims))
    ((randn) (randn dims))
    (else (error "linear: unknown initializer" init))))

(define* (linear in-features out-features #:key (bias #t) (init 'randn)
                 (name #f))
  "A dense (fully-connected) layer. Input [batch, in], output [batch, out].
Lowered to native gemm(x, Wᵀ, b) with the bias broadcast along the trailing axis."
  (let* ((lname (or name (default-layer-name 'linear)))
         (w (initializer (list out-features in-features) init))
         (b (and bias (zeros (list out-features))))
         (params (if b `((weight . ,w) (bias . ,b)) `((weight . ,w)))))
    (%make-layer
     'linear lname params
     (lambda (x-name x-dims mk reg)
       (let* ((w-name (reg 'weight w))
              (b-name (and b (reg 'bias b)))
              (out-name (mk))
              (out-dims (list (car x-dims) out-features))
              (inputs (if b-name
                          (list x-name w-name b-name)
                          (list x-name w-name))))
         (values
          (list (string-append
                 "(node gemm (inputs " (string-join (map quoted inputs) " ") ")"
                 " (outputs (" (quoted out-name) " "
                 (type->sx 'float32 out-dims) "))"
                 " (attrs (transB (int 1))))"))
          out-name out-dims))))))

(define* (relu #:key (name #f))
  "Rectified linear unit. Elementwise; preserves the input shape. Lowered to
native relu."
  (let ((lname (or name (default-layer-name 'relu))))
    (%make-layer
     'relu lname '()
     (lambda (x-name x-dims mk reg)
       (let ((out-name (mk)))
         (values
          (list (string-append
                 "(node relu (inputs " (quoted x-name) ")"
                 " (outputs (" (quoted out-name) " "
                 (type->sx 'float32 x-dims) ")))"))
          out-name x-dims))))))

;;; --- model ------------------------------------------------------------------

(define-record-type <model>
  (%make-model layers parameters)
  model?
  (layers     model-layers)
  (parameters model-parameters))   ; merged alist of (key . tensor)

(define (sequential . layers)
  "Compose LAYERS into a model. Parameter names are made unique per layer."
  (let ((params (append-map
                 (lambda (layer)
                   (map (lambda (pair)
                          (cons (string->symbol
                                 (string-append (symbol->string (layer-name layer))
                                                "." (symbol->string (car pair))))
                                (cdr pair)))
                        (layer-parameters layer)))
                 layers)))
    (%make-model layers params)))

(define (model-parameter-ref model key)
  (assq-ref (model-parameters model) key))

;;; Build the S-Expr v0.1 text for a forward pass, threading the input value
;;; through each layer in order. Returns (values text out-dims).
(define (build-graph input-name input-dtype input-dims layers)
  (let ((params '())    ; reversed: (ssa-name dtype dims tensor)
        (nodes '())     ; in-order: formatted node strings
        (counter 0))
    (define (mk)
      (set! counter (+ counter 1))
      (string-append "v" (number->string counter)))
    (let loop ((ls layers) (x-name input-name) (x-dims input-dims))
      (if (null? ls)
          (let ((param-strs
                 (map (lambda (p)
                        (string-append
                         "(parameter " (quoted (car p)) " "
                         (type->sx (cadr p) (caddr p))
                         " (data :values " (values->sx (cadddr p)) "))"))
                      (reverse params))))
            (values
             (string-append
              "(sonicboom-s-expr (version 0 1) (graph (name \"guile-model\")"
              " (inputs (input " (quoted input-name) " "
              (type->sx input-dtype input-dims) "))"
              " (outputs (output " (quoted x-name) "))"
              " (parameters " (string-join param-strs " ") ")"
              " (nodes " (string-join nodes " ") ")))")
             x-dims))
          (let* ((layer (car ls))
                 ;; Parameter SSA names match sequential's model-parameters keys
                 ;; ("<layer-name>.<key>") so export_model can match trained
                 ;; tensors to graph parameters by name.
                 (prefix (string-append (symbol->string (layer-name layer)) "."))
                 (seen '()))
            (define (reg key tensor)
              (let ((existing (assq key seen)))
                (if existing
                    (cdr existing)
                    (let ((ssa (string-append prefix (symbol->string key))))
                      (set! seen (cons (cons key ssa) seen))
                      (set! params (cons (list ssa (tensor-dtype tensor)
                                               (tensor-shape tensor) tensor)
                                         params))
                      ssa))))
            (call-with-values
                (lambda () ((layer-apply layer) x-name x-dims mk reg))
              (lambda (new-nodes out-name out-dims)
                (set! nodes (append nodes new-nodes))
                (loop (cdr ls) out-name out-dims))))))))

(define (model-graph-text model input-dims)
  "Return the S-Expr v0.1 graph text for MODEL's forward pass (input named
\"x\", float32, shape INPUT-DIMS). The parameter names in the text match
model-parameters (\"<layer-name>.<key>\"), so the trained tensors can be
matched to graph parameters by name at export time."
  (call-with-values
      (lambda () (build-graph "x" 'float32 input-dims (model-layers model)))
    (lambda (text out-dims) text)))

(define (model-forward model x)
  "Run a forward pass of MODEL on input tensor X through the native backend.
Returns the output tensor. Only float32 is supported in v0."
  (unless (eq? (tensor-dtype x) 'float32)
    (error "model-forward: only float32 inputs are supported in v0"
           (tensor-dtype x)))
  (call-with-values
      (lambda () (build-graph "x" 'float32 (tensor-shape x) (model-layers model)))
    (lambda (text out-dims)
      (call-with-values
          (lambda () (run-s-expr text (tensor->bytevector x)))
        (lambda (out-bv err)
          (if err
              (error "model-forward:" err)
              (bytevector->tensor out-bv out-dims #:dtype 'float32)))))))
