;;; SonicBoom — Guile binding: layers & model composition
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A minimal PyTorch/Keras-like layer API on top of the (sonicboom tensor) and
;;; (sonicboom ffi) modules. A layer is a named bundle of parameters plus two
;;; ways to say what it computes over a value:
;;;
;;;   - `apply`, which emits S-Expr v0.1 node text (the deployment route); and
;;;   - `trace`, which applies the primitive ops of (sonicboom trace symbolic)
;;;     so the whole model can be traced and then lowered back onto the same
;;;     S-Expr vocabulary by (sonicboom trace lower) (the traced route).
;;;
;;; Both routes are driven by the same layer structure and the same parameter
;;; tensors, and both end up executing the same `gemm` / `relu` nodes through
;;; libsonicboom.so, so a traced model and `model-forward` agree. No tensor
;;; arithmetic happens in Scheme.

(define-module (sonicboom nn)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (ice-9 optargs)
  #:use-module (rnrs bytevectors)
  #:use-module (sonicboom tensor)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom executable)
  #:use-module ((sonicboom trace tensor) #:prefix ct:)
  #:use-module ((sonicboom trace symbolic) #:prefix sx:)
  #:use-module ((sonicboom trace layers) #:prefix tl:)
  #:use-module ((sonicboom trace lower) #:select (trace-graph run-traced))
  #:export (layer? layer-kind layer-name layer-parameters layer-parameter-ref
            model? model-layers model-parameters model-parameter-ref
            model-graph-text linear relu sequential model-forward
            model-trace model-trace-forward))

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
  (%make-layer kind name parameters apply trace)
  layer?
  (kind       layer-kind)         ; 'linear | 'relu (drives native-torch forward)
  (name       layer-name)
  (parameters layer-parameters)   ; alist of (key . tensor)
  (apply      layer-apply)        ; (x-name x-dims mk reg) -> (node-strs out-name out-dims)
  (trace      layer-trace))       ; (x params) -> symbolic value

(define (layer-parameter-ref layer key)
  (assq-ref (layer-parameters layer) key))

(define (initializer dims init)
  (case init
    ((zeros) (zeros dims))
    ((ones)  (ones dims))
    ((randn) (randn dims))
    (else (error "linear: unknown initializer" init))))

;;; --- tracing a layer ---------------------------------------------------------
;;; The traced route closes over concrete ctensors, which the tracer records as
;;; graph parameters. A layer looks its own parameters up by the same
;;; "<layer-name>.<key>" naming `sequential` uses for model-parameters.

(define (layer-param-ct params lname key)
  (let ((hit (assoc-ref params (string->symbol
                                (string-append (symbol->string lname) "."
                                               (symbol->string key))))))
    (and hit (tensor->ctensor hit))))

(define (tensor->ctensor t)
  ;; make-ctensor copies the packed bytes, so the ctensor does not alias the
  ;; parameter tensor.
  (ct:make-ctensor (tensor-dtype t) (tensor-shape t) (tensor->bytevector t)))

(define (transpose-2d ct)
  "Transpose a rank-2 ctensor. Used once, when a linear's [out, in] weight is
handed to the tracer: the symbolic `linear` contracts over the weight's trailing
axis ([in, out]). This is a constant layout change, not arithmetic in the graph."
  (let* ((sh (ct:ctensor-shape ct))
         (rows (car sh)) (cols (cadr sh))
         (out (make-vector (* rows cols) 0)))
    (let loop ((i 0))
      (when (< i rows)
        (let inner ((j 0))
          (when (< j cols)
            (vector-set! out (+ (* j rows) i) (ct:ctensor-ref ct (+ (* i cols) j)))
            (inner (+ j 1))))
        (loop (+ i 1))))
    (ct:make-ctensor (ct:ctensor-dtype ct) (list cols rows) out)))

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
          out-name out-dims)))
     (lambda (x params)
       (let ((w (layer-param-ct params lname 'weight))
             (bb (and b (layer-param-ct params lname 'bias))))
         (unless w (error "linear: no traced weight for" lname))
         (let ((wt (transpose-2d w)))
           (if bb (tl:linear x wt bb) (sx:matmul x wt))))))))

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
          out-name x-dims)))
     (lambda (x params) (sx:relu x)))))

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

;;; --- the traced route --------------------------------------------------------
;;; model-forward emits S-Expr text from the layer structure; model-trace runs
;;; the same layers as symbolic ops through the tracer and gets an equivalent
;;; document back, which (sonicboom trace lower) then lowers onto the runtime's
;;; op set. Both are driven by the one layer structure and the one set of
;;; parameter tensors.

(define (trace-layers layers x params)
  (let loop ((ls layers) (v x))
    (if (null? ls) v (loop (cdr ls) ((layer-trace (car ls)) v params)))))

(define (model-trace model input-dims)
  "Trace MODEL's forward pass over an input of shape INPUT-DIMS — a list of
ints — and return the traced S-Expr v0.1 document datum. Lower it with
`traced->s-expr` / run it with `run-traced` (both in (sonicboom trace lower));
`linear` layers come out as the same `gemm` nodes model-forward emits."
  (let* ((params (map (lambda (p) (cons (car p) (cdr p)))
                      (model-parameters model)))
         (x (ct:make-ctensor 'float32 input-dims
                             (make-list (ct:shape-numel input-dims) 0.0))))
    (trace-graph (lambda (x) (trace-layers (model-layers model) x params)) x)))

(define (doc-input-name graph)
  (cadr (car (cdr (assq 'inputs (cdr (caddr graph)))))))

(define (ctensor->tensor ct)
  (let* ((n (ct:ctensor-numel ct))
         (bv (make-bytevector (* n 4) 0)))
    (let loop ((i 0))
      (when (< i n)
        (bytevector-ieee-single-native-set! bv (* i 4)
                                            (* 1.0 (ct:ctensor-ref ct i)))
        (loop (+ i 1))))
    (bytevector->tensor bv (ct:ctensor-shape ct) #:dtype 'float32)))

(define (model-trace-forward model x)
  "Run MODEL's forward pass on tensor X through the traced route: trace the
layers, lower the graph onto the runtime vocabulary and execute it with
libsonicboom.so. Returns the output tensor — the same value model-forward
computes, reached through the tracer instead of emitted text."
  (unless (eq? (tensor-dtype x) 'float32)
    (error "model-trace-forward: only float32 inputs are supported in v0"
           (tensor-dtype x)))
  (let ((graph (model-trace model (tensor-shape x))))
    (ctensor->tensor
     (run-traced graph (list (cons (doc-input-name graph) (tensor->ctensor x)))))))

(define (model-forward model x)
  "Run a forward pass of MODEL on input tensor X through the native backend.
Returns the output tensor. Only float32 is supported in v0."
  (unless (eq? (tensor-dtype x) 'float32)
    (error "model-forward: only float32 inputs are supported in v0"
           (tensor-dtype x)))
  (call-with-values
      (lambda () (build-graph "x" 'float32 (tensor-shape x) (model-layers model)))
    (lambda (text out-dims)
      ;; The graph text is a pure function of the model's layer structure and
      ;; the input shape, so the compiled executable is reused across forward
      ;; passes (see (sonicboom executable)); disable with
      ;; `executable-cache-enabled` to compile per call.
      (call-with-values
          (lambda () (run-s-expr/cached text (tensor->bytevector x)))
        (lambda (out-bv err)
          (if err
              (error "model-forward:" err)
              (bytevector->tensor out-bv out-dims #:dtype 'float32)))))))
