;;; SonicBoom — Guile binding: TTS forward-mode compilation
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Compile a Scheme forward function into one device graph whose weights live
;;; in a weight store instead of in the graph text:
;;;
;;;   (define model (load-model "model.safetensors"))
;;;   (define fwd (compile-forward model '((x float32 (2 8)))
;;;                                (lambda (x w) (relu (matmul x (w "fc.weight" 'float32 '(8 4)))))))
;;;   (fwd (list->ctensor '((1.0 ...))))
;;;
;;; COMPILE-FORWARD traces PROC (a function written against the ops of
;;; (sonicboom trace symbolic)) into an S-Expr document in which each weight is a
;;; `(data :bound "NAME")` parameter, compiles it as a device inference graph
;;; (Boom IR Forward pass phase: dead-node elimination, buffer reuse, and
;;; CUDA-graph capture on by default), and binds every weight from MODEL. The
;;; weight store uploads each weight to the device on first use and shares its
;;; device buffers with every executable bound to the store, so compiling
;;; several shapes against one model uploads the weights once.
;;;
;;; PROC receives one argument per input specification, in order, plus a weight
;;; accessor W last: (w NAME DTYPE SHAPE) returns the symbolic parameter for the
;;; weight NAME (a name in the store) with the given dtype and shape. Only
;;; float32 weights are supplied; a store opened with #:convert-to-f32? supplies
;;; f16/bf16/f64 entries as float32. A weight whose name the store lacks, or
;;; whose stored shape differs, is reported at compile with both shapes named.
;;;
;;; A build without CUDA (or #:backend 'cpu) compiles the same document for the
;;; CPU interpreter instead, binding the same weights from the store's host
;;; bytes. Both paths produce the same arithmetic from the same kernels; the
;;; public test exercises the CPU path.
;;;
;;; The returned procedure owns its native executable: a GC guardian releases
;;; it when the procedure is collected, or free-compiled-forward! releases it
;;; explicitly. The weight store may be freed immediately after compile — the
;;; executable keeps the store (and its device buffers) alive itself.

(define-module (sonicboom compile)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (rnrs bytevectors)
  #:use-module (system foreign)
  #:use-module (ice-9 threads)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom error)
  #:use-module (sonicboom trace compat)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace graph)
  #:use-module (sonicboom trace symbolic)
  #:use-module (sonicboom trace device)
  #:export (load-model weights? weights-path free-weights! reap-weights!
            compile-forward compiled-forward?
            compiled-forward-memory-stats compiled-forward-backend
            free-compiled-forward! reap-compiled-forwards!))

(define %null %null-pointer)

;;; --- weight store ------------------------------------------------------------

;; A weight store: one mapped safetensors or GGUF file.
;;
;; Its named tensors are uploaded to the device as float32 on first use (see
;; boom_weights_open). The record owns the native handle; FREED? makes the
;; explicit free and the guardian's reap idempotent.
;;
;; PyTorch: none
;; Note: the nearest PyTorch concept is a state_dict or a safetensors file
;; loaded into tensors.
(define-record-type <weights>
  (%make-weights ptr path format convert freed)
  weights?
  (ptr weights-ptr)
  (path weights-path)
  (format weights-format)
  (convert weights-convert?)
  (freed weights-freed? set-weights-freed!))

(define %weights-guardian (make-guardian))

(define (free-weights! model)
  "Release a weight store handle.

Idempotent; an executable bound to the store keeps the store and the device
buffers it bound alive regardless of free order.
Args:
  model: a weight store from load-model.
Returns: unspecified.
PyTorch: none
Note: a state_dict is plain tensors with no handle to free. The executable
keeps the store alive; freeing first is safe."
  (unless (weights? model)
    (error "free-weights!: not a weight store" model))
  (unless (weights-freed? model)
    (set-weights-freed! model #t)
    (let ((p (weights-ptr model)))
      (unless (null-pointer? p)
        ((proc 'boom_weights_free) p))))
  *unspecified*)

(define (reap-weights!)
  "Release weight stores collected without an explicit free.

The module keeps a weak reference to every store it opened; this releases the
handles of those whose record has been collected. Executables bound to such a
store keep working — they own a reference to the store themselves.
Args:
Returns: the number of store handles released.
PyTorch: none"
  (let loop ((n 0))
    (let ((dead (%weights-guardian)))
      (if (or (not dead) (null? dead))
          n
          (begin (free-weights! dead) (loop (+ 1 n)))))))

(define %weight-formats
  '((auto . 0) (safetensors . 1) (gguf . 2)))

(define* (load-model path #:key (format 'auto) (convert-to-f32? #f))
  "Map a safetensors or GGUF weight file for use by compiled forward graphs.

The file is parsed once. Named tensors are uploaded to the device as float32
when a compiled graph binds them; f32 entries are used as stored, and with
CONVERT-TO-F32? f16, bf16 and f64 entries are converted on the host. A
quantized GGUF entry cannot be supplied to a float32 graph and is reported by
name and type when requested.
Args:
  path: the file path.
  format: keyword; 'auto (default) detects GGUF from its magic and reads
    anything else as safetensors; 'safetensors or 'gguf forces the format and
    rejects a file that is not it.
  convert-to-f32?: keyword; convert f16, bf16 and f64 entries to float32 at
    upload (default #f).
Returns: a weight store.
PyTorch: none
Note: no example is given because one needs a weight file on disk. The same
store can be bound to several compiled graphs; each weight is uploaded once and
shared. The nearest PyTorch concept is safetensors.safe_open, with the tensors
kept on the device."
  (unless (string? path)
    (error "load-model: path must be a string" path))
  (let ((code (assq-ref %weight-formats format)))
    (unless code
      (error "load-model: format must be auto, safetensors or gguf" format))
    (let ((out (make-slot))
          (err (make-slot)))
      (sb-check! ((proc 'boom_weights_open) (string->pointer path) code
                  (if convert-to-f32? 1 0) (slot-ptr out) (slot-ptr err))
                 err "load-model")
      (let ((model (%make-weights (slot-ref out) path format convert-to-f32? #f)))
        (%weights-guardian model)
        model))))

;;; --- input specifications ----------------------------------------------------

;; dtype symbols a forward graph input may declare. The device and CPU
;; interpreters both take these; a weight is always float32.
(define %input-dtypes
  '(float32 float16 bfloat16 float64 int8 uint8 int16 int32 int64 bool))

(define (check-spec spec)
  "Validate one (name dtype shape) input specification.

NAME is a non-empty symbol or string; the S-Expr document declares an input
identifier as a string, so both forms are normalized to one."
  (unless (and (list? spec) (= 3 (length spec)))
    (error "compile-forward: each input must be (name dtype shape)" spec))
  (let* ((raw (car spec))
         (name (cond ((symbol? raw) (symbol->string raw))
                     ((and (string? raw) (not (string-null? raw))) raw)
                     (else #f)))
         (dtype (cadr spec)) (shape (caddr spec)))
    (unless name
      (error "compile-forward: input name must be a non-empty symbol or string"
             spec))
    (unless (memq dtype %input-dtypes)
      (error "compile-forward: unsupported input dtype" name dtype))
    (unless (and (list? shape)
                 (every (lambda (d) (and (exact-integer? d) (>= d 0))) shape))
      (error "compile-forward: input shape must be a list of non-negative integers"
             name shape))
    (list name dtype shape)))

;;; --- tracing -----------------------------------------------------------------

(define (make-weight-ref rec)
  "The weight accessor PROC receives: (w NAME DTYPE SHAPE) -> symbolic value.

Each distinct store NAME becomes one bound parameter of the document. Asking
for the same name again with the same dtype and shape returns the parameter
already recorded; asking with a different dtype or shape is an error, because
one name denotes one stored tensor."
  (lambda (name dtype shape)
    (unless (and (string? name) (not (string-null? name)))
      (error "compile-forward: a weight name must be a non-empty string" name))
    (unless (eq? dtype 'float32)
      (error "compile-forward: weights are supplied as float32; declare weight"
             name "as float32, got" dtype))
    (unless (and (list? shape)
                 (every (lambda (d) (and (exact-integer? d) (>= d 0))) shape))
      (error "compile-forward: weight shape must be a list of non-negative integers"
             name shape))
    (let ((hit (find (lambda (p) (equal? (gparam-bound p) name))
                     (recorder-params rec))))
      (cond
       (hit
        (unless (and (eq? (gparam-dtype hit) dtype)
                     (equal? (gparam-shape hit) shape))
          (error "compile-forward: weight used with two shapes or dtypes" name
                 (list (gparam-dtype hit) (gparam-shape hit)) (list dtype shape)))
        (make-sym (gparam-name hit) dtype shape))
       (else
        (let* ((ctx (recorder-context rec))
               (vname (fresh-name! ctx "w")))
          (register-type! ctx vname dtype shape)
          (set-recorder-params!
           rec (cons (make-gparam vname dtype shape #f name)
                     (recorder-params rec)))
          (make-sym vname dtype shape)))))))

(define (forward-outputs result)
  "The symbolic outputs of a traced forward function.

A single symbolic tensor or a list of them; anything else cannot become a
graph output."
  (cond ((sym? result) (list result))
        ((and (pair? result) (every sym? result)) result)
        (else
         (error "compile-forward: the forward function must return a symbolic \
tensor or a list of them" result))))

(define (trace-forward inputs proc)
  "Trace PROC on symbolic inputs, returning (values nodes params outputs single?).

NODES and PARAMS are in topological order and PARAMS are bound weights. SINGLE?
is #t when PROC returned one tensor rather than a list. A data-dependent branch
(`bool` on a symbolic tensor) cannot be compiled as one forward graph and is
reported."
  (let* ((ctx (make-trace-context 16 'default 64))
         (rec (make-recorder ctx '() '() '() 0 '() 0 #f))
         (syms (map (lambda (spec)
                      (let ((name (car spec)) (dtype (cadr spec))
                            (shape (caddr spec)))
                        (register-type! ctx name dtype shape)
                        (make-sym name dtype shape)))
                    inputs))
         (result
          (%call-with-prompt
           %break-tag
           (lambda ()
             (parameterize ((current-tracer rec))
               (apply proc (append syms (list (make-weight-ref rec))))))
           (lambda (k c)
             (error "compile-forward: the forward function branches on a tensor \
value through bool; a data-dependent branch cannot be one forward graph"))))
         (outputs (forward-outputs result))
         (nodes (reverse (recorder-nodes rec)))
         (params (reverse (recorder-params rec)))
         (defs (append (append-map gnode-output-names nodes)
                       (map gparam-name params)))
         (clash (lset-intersection equal? defs (map car inputs)))
         (names (map sym-name outputs)))
    (unless (null? clash)
      (error "compile-forward: an input name is reused by a generated value; \
rename the input" clash))
    (unless (equal? (delete-duplicates names) names)
      (error "compile-forward: an output is returned more than once" names))
    (values nodes params outputs (sym? result))))

(define (forward-text inputs outputs params nodes)
  "Emit the S-Expr document text of a traced forward graph.

Every declared input is a graph input (in declaration order); every value a
node reads but no node or parameter defines must be one of them."
  (let* ((declared (map car inputs))
         (defs (append (append-map gnode-output-names nodes)
                       (map gparam-name params)))
         (refs (delete-duplicates
                (append (append-map gnode-inputs nodes)
                        (map sym-name outputs))))
         (undeclared (lset-difference equal?
                                      (remove (lambda (n) (member n defs)) refs)
                                      declared)))
    (unless (null? undeclared)
      (error "compile-forward: the graph reads values it does not define" undeclared))
    (sexpr->text
     (graph->sexpr "forward"
                   (map (lambda (s) (list (car s) (cadr s) (caddr s))) inputs)
                   (map sym-name outputs)
                   params nodes))))

;;; --- executables -------------------------------------------------------------

(define (query-count exe direction)
  (let ((bv (make-bytevector 8 0)))
    (let ((st ((proc (if (eq? direction 'in) 'boom_input_count 'boom_output_count))
               exe (bytevector->pointer bv))))
      (unless (sb-status-ok? st)
        (error "compile-forward: introspection failed" st)))
    (bytevector-u64-native-ref bv 0)))

(define (query-desc exe index direction)
  "Descriptor (name dtype shape bytes) of input/output INDEX; NAME is #f when
the compiled form does not retain names (the CPU interpreter)."
  (let* ((info (if (eq? direction 'in) 'boom_input_info_at 'boom_output_info_at))
         (named (if (eq? direction 'in) 'boom_input_name_at 'boom_output_name_at))
         (dtype-bv (make-bytevector 4 0))
         (rank-bv (make-bytevector 8 0))
         (dims-slot (make-slot))
         (bytes-bv (make-bytevector 8 0))
         (st ((proc info) exe index (bytevector->pointer dtype-bv)
              (bytevector->pointer rank-bv) (slot-ptr dims-slot)
              (bytevector->pointer bytes-bv))))
    (unless (sb-status-ok? st)
      (error "compile-forward: introspection failed" st))
    (let* ((dtype (vector-ref (vector 'float32 'float16 'bfloat16 'float64
                                      'int8 'uint8 'int16 'int32 'int64 'bool)
                              (bytevector-s32-native-ref dtype-bv 0)))
           (rank (bytevector-s64-native-ref rank-bv 0))
           (dims (slot-ref dims-slot))
           (shape (if (<= rank 0) '()
                      (let ((bv (pointer->bytevector dims (* 8 rank))))
                        (let loop ((i (- rank 1)) (acc '()))
                          (if (< i 0)
                              acc
                              (loop (- i 1)
                                    (cons (bytevector-s64-native-ref bv (* 8 i))
                                          acc)))))))
           (name-slot (make-slot))
           (st ((proc named) exe index (slot-ptr name-slot))))
      (unless (sb-status-ok? st)
        (error "compile-forward: name introspection failed" st))
      (let ((p (slot-ref name-slot)))
        (list (and (not (null-pointer? p)) (pointer->string p))
              dtype shape (bytevector-u64-native-ref bytes-bv 0))))))

(define (query-descs exe direction)
  (map (lambda (i) (query-desc exe i direction))
       (iota (query-count exe direction))))

;; A compiled forward pass. EXE is the native executable and BACKEND is 'device
;; or 'cpu; IN-DESCS and OUT-DESCS describe its input and output slots in
;; executable order and IN-POS / OUT-POS map each slot to the caller's argument
;; position / the symbolic output position. SINGLE? is #t when the traced
;; function returned one tensor. LOCK serializes runs (one executable owns one
;; set of buffers); FREED? makes the explicit free and the guardian's reap
;; idempotent.
;;
;; PyTorch: none
;; Note: the nearest PyTorch concept is a module made callable by
;; torch.compile; here the executable is built by `compile-forward` from a
;; traced Scheme function.
(define-record-type <compiled-forward>
  (%make-compiled-forward exe backend in-descs out-descs in-pos out-pos single?
                          lock freed)
  %compiled-forward?
  (exe          compiled-forward-exe)
  (backend      %compiled-forward-backend)
  (in-descs     compiled-forward-in-descs)
  (out-descs    compiled-forward-out-descs)
  (in-pos       compiled-forward-in-pos)
  (out-pos      compiled-forward-out-pos)
  (single?      compiled-forward-single?)
  (lock         compiled-forward-lock)
  (freed        compiled-forward-freed? set-compiled-forward-freed!))

(define (%compiled-forward-of who fwd)
  "The compiled-forward record behind FWD, which must be a compiled callable."
  (let ((cf (and (procedure? fwd)
                 (procedure-tag fwd 'sonicboom-compiled-forward))))
    (unless cf
      (error (string-append who ": not a compiled forward procedure") fwd))
    cf))

(define (compiled-forward? fwd)
  "Whether FWD is the callable compile-forward returned.

Args:
  fwd: any value.
Returns: #t for a compiled forward procedure, #f otherwise.
Example:
  (use-modules (sonicboom compile))
  (compiled-forward? car)
  => #f
PyTorch: none
Note: an example that compiles a forward needs a weight file, so it is left
out of the docstring; the public test compiles one against a generated store.
Note: the nearest PyTorch concept is isinstance(module, torch.nn.Module) after
torch.compile; this recognizes the procedure compile-forward returned."
  (and (procedure? fwd)
       (procedure-tag fwd 'sonicboom-compiled-forward)
       #t))

(define (compiled-forward-backend fwd)
  "The backend a compiled forward runs on: the symbol `device' or `cpu'.

Args:
  fwd: a compiled forward procedure from compile-forward.
Returns: the symbol `device' or `cpu'.
PyTorch: none"
  (%compiled-forward-backend (%compiled-forward-of "compiled-forward-backend"
                                                  fwd)))

(define %forward-guardian (make-guardian))

(define (free-compiled-forward! fwd)
  "Release a compiled forward pass's native executable.

Idempotent. A run that is in flight finishes first; later runs raise.
Args:
  fwd: a compiled forward procedure from compile-forward.
Returns: unspecified.
PyTorch: none"
  (let ((cf (and (procedure? fwd)
                 (procedure-tag fwd 'sonicboom-compiled-forward))))
    (unless cf
      (error "free-compiled-forward!: not a compiled forward procedure" fwd))
    (let ((lock (compiled-forward-lock cf)))
      (lock-mutex lock)
      (unless (compiled-forward-freed? cf)
        (set-compiled-forward-freed! cf #t)
        (let ((p (compiled-forward-exe cf)))
          (unless (null-pointer? p)
            ((proc 'boom_executable_free) p))))
      (unlock-mutex lock)))
  *unspecified*)

(define (reap-compiled-forwards!)
  "Release compiled forward passes collected without an explicit free.

The module keeps a weak reference to every compiled pass; this releases those
whose record has been collected.
Args:
Returns: the number of executables released.
PyTorch: none"
  (let loop ((n 0))
    (let ((dead (%forward-guardian)))
      (if (or (not dead) (null? dead))
          n
          (begin
            (let ((lock (compiled-forward-lock dead)))
              (lock-mutex lock)
              (unless (compiled-forward-freed? dead)
                (set-compiled-forward-freed! dead #t)
                ((proc 'boom_executable_free) (compiled-forward-exe dead)))
              (unlock-mutex lock))
            (loop (+ 1 n)))))))

;;; --- compiling ---------------------------------------------------------------

(define (parse-or-raise text)
  (let ((doc (make-slot))
        (err (make-slot)))
    (sb-check! ((proc 'boom_parse) (string->pointer text) (string-length text)
                (slot-ptr doc) (slot-ptr err))
               err "compile-forward")
    (slot-ref doc)))

(define (compile-device-exe text model capture?)
  "Compile TEXT as a device inference graph and bind MODEL's weights."
  (let* ((doc (parse-or-raise text))
         (exe-slot (make-slot))
         (err (make-slot))
         (st ((proc 'boom_compile_device_ex) doc 1 (slot-ptr exe-slot)
              (slot-ptr err))))
    ((proc 'boom_document_free) doc)
    (sb-check! st err "compile-forward")
    (let ((exe (slot-ref exe-slot)))
      (catch #t
        (lambda ()
          (let ((err (make-slot)))
            (sb-check! ((proc 'boom_executable_bind_weights) exe
                        (weights-ptr model) (slot-ptr err))
                       err "compile-forward")
            (sb-check! ((proc 'boom_executable_set_capture) exe
                        (if capture? 1 0) (slot-ptr err))
                       err "compile-forward"))
          exe)
        (lambda (key . args)
          ((proc 'boom_executable_free) exe)
          (apply throw key args))))))

;; A shape as the store prints it: (3, 2).
(define (shape->string shape)
  (let loop ((rest (cdr shape))
             (acc (if (null? shape) "" (number->string (car shape)))))
    (if (null? rest)
        (string-append "(" acc ")")
        (loop (cdr rest) (string-append acc ", " (number->string (car rest)))))))

(define (weight-entry-info model name)
  "The store's (stored type name, shape, byte count) for weight NAME.

The CPU fallback checks each graph weight against this before binding it, so a
shape or type mismatch is reported in the store's own words rather than as a
byte-count difference."
  (let* ((dtype-slot (make-slot))
         (rank-bv (make-bytevector 8 0))
         (dims-slot (make-slot))
         (bytes-bv (make-bytevector 8 0))
         (err (make-slot)))
    (sb-check! ((proc 'boom_weights_tensor_info) (weights-ptr model)
                (string->pointer name) (slot-ptr dtype-slot)
                (bytevector->pointer rank-bv) (slot-ptr dims-slot)
                (bytevector->pointer bytes-bv) (slot-ptr err))
               err "compile-forward")
    (let* ((rank (bytevector-s64-native-ref rank-bv 0))
           (dims (slot-ref dims-slot))
           (bv (if (<= rank 0) #f (pointer->bytevector dims (* 8 rank))))
           (shape (if bv
                      (let loop ((i (- rank 1)) (acc '()))
                        (if (< i 0)
                            acc
                            (loop (- i 1)
                                  (cons (bytevector-s64-native-ref bv (* 8 i))
                                        acc))))
                      '())))
      (list (pointer->string (slot-ref dtype-slot)) shape
            (bytevector-u64-native-ref bytes-bv 0)))))

(define (compile-cpu-exe text model params)
  "Compile TEXT for the CPU interpreter and bind MODEL's bytes as PARAMS'
(data :bound ...) values. PARAMS is a list of (bound-name dtype shape)."
  (let* ((doc (parse-or-raise text))
         (exe-slot (make-slot))
         (err (make-slot))
         (st ((proc 'boom_compile) doc (string->pointer "") (slot-ptr exe-slot)
              (slot-ptr err))))
    ((proc 'boom_document_free) doc)
    (sb-check! st err "compile-forward")
    (let ((exe (slot-ref exe-slot)))
      (catch #t
        (lambda ()
          (for-each
           (lambda (p)
             (let* ((name (car p))
                    (want-shape (caddr p))
                    (entry (weight-entry-info model name))
                    (stored-type (car entry))
                    (stored-shape (cadr entry))
                    (want (* (dtype-size 'float32) (shape-numel want-shape)))
                    (data-slot (make-slot))
                    (bytes-bv (make-bytevector 8 0))
                    (err (make-slot)))
               (unless (equal? stored-shape want-shape)
                 (error "compile-forward: weight" name "is"
                        (shape->string stored-shape) "in" (weights-path model)
                        "but the graph expects" (shape->string want-shape)))
               ;; safetensors names the stored type "F32" and GGUF "f32"; the
               ;; same float32 entry must pass whichever file holds it.
               (unless (or (string-ci=? stored-type "f32")
                           (weights-convert? model))
                 (error "compile-forward: weight" name "is stored as" stored-type
                        "; open the model with #:convert-to-f32? #t to convert it"))
               (sb-check! ((proc 'boom_weights_data_f32) (weights-ptr model)
                           (string->pointer name) (slot-ptr data-slot)
                           (bytevector->pointer bytes-bv) (slot-ptr err))
                          err "compile-forward")
               (let ((got (bytevector-u64-native-ref bytes-bv 0)))
                 (unless (= got want)
                   (error "compile-forward: weight" name "holds" got
                          "float32 bytes in the store but the graph expects" want))
                 (let ((err (make-slot)))
                   (sb-check! ((proc 'boom_bind_parameter) exe
                               (string->pointer name) (slot-ref data-slot) got
                               (slot-ptr err))
                              err "compile-forward")))))
           params)
          (let ((complete (make-bytevector 4 0))
                (err (make-slot)))
            (sb-check! ((proc 'boom_bindings_complete) exe
                        (bytevector->pointer complete) (slot-ptr err))
                       err "compile-forward")
            (unless (not (zero? (bytevector-s32-native-ref complete 0)))
              (error "compile-forward: not every bound weight was supplied")))
          exe)
        (lambda (key . args)
          ((proc 'boom_executable_free) exe)
          (apply throw key args))))))

;; Executable input slot -> caller argument position. The device graph retains
;; input names, so its binding order may differ from the declaration order; the
;; CPU interpreter does not retain names and binds in document order, which is
;; the declaration order.
(define (input-plan descs inputs backend)
  (if (eq? backend 'device)
      ;; The executable names a tensor with the document's input identifier.
      (let ((names (map car inputs)))
        (unless (every car descs)
          (error "compile-forward: the device graph does not name its inputs"))
        (map (lambda (desc)
               (let ((name (car desc)))
                 (or (list-index (lambda (n) (equal? n name)) names)
                     (error "compile-forward: the executable input" name
                            "is not a declared input" names))))
             descs))
      (begin
        (unless (= (length descs) (length inputs))
          (error "compile-forward: the CPU executable takes" (length descs)
                 "inputs but the forward function declares" (length inputs)))
        (iota (length inputs)))))

(define (output-plan descs outputs backend)
  (let ((names (map sym-name outputs)))
    (if (eq? backend 'device)
        (begin
          (unless (every car descs)
            (error "compile-forward: the device graph does not name its outputs"))
          (unless (= (length descs) (length names))
            (error "compile-forward: the executable produces" (length descs)
                   "outputs but the forward function returns" (length names)))
          (map (lambda (desc)
                 (let ((name (car desc)))
                   (or (list-index (lambda (n) (equal? n name)) names)
                       (error "compile-forward: the executable output" name
                              "is not a returned value" names))))
               descs))
        (begin
          (unless (= (length descs) (length names))
            (error "compile-forward: the executable produces" (length descs)
                   "outputs but the forward function returns" (length names)))
          (iota (length names))))))

;;; --- running -----------------------------------------------------------------

(define (u64-array xs)
  (let ((bv (make-bytevector (* 8 (length xs)) 0)))
    (for-each (lambda (x i) (bytevector-u64-native-set! bv (* 8 i) x))
              xs (iota (length xs)))
    bv))

(define (pointer-array bvs)
  (let* ((n (length bvs))
         (bv (make-bytevector (* 8 n) 0)))
    (for-each (lambda (b i)
                (bytevector-u64-native-set! bv (* 8 i)
                                            (pointer-address (bytevector->pointer b))))
              bvs (iota n))
    bv))

(define (encode-forward-input desc arg pos)
  "ARG (the caller's POS-th argument) as the bytes of DESCs slot."
  (let ((dtype (cadr desc)) (shape (caddr desc)))
    (unless (ctensor? arg)
      (error "compile-forward: argument" (+ pos 1) "must be a ctensor" arg))
    (unless (eq? (ctensor-dtype arg) dtype)
      (error "compile-forward: argument" (+ pos 1) "dtype mismatch"
             (ctensor-dtype arg) dtype))
    (unless (equal? (ctensor-shape arg) shape)
      (error "compile-forward: argument" (+ pos 1) "shape mismatch"
             (ctensor-shape arg) shape))
    (ctensor-data arg)))

(define (run-compiled-forward cf args)
  (let ((in-descs (compiled-forward-in-descs cf))
        (out-descs (compiled-forward-out-descs cf))
        (in-pos (compiled-forward-in-pos cf))
        (out-pos (compiled-forward-out-pos cf)))
    (unless (= (length args) (length in-pos))
      (error "compile-forward: the compiled forward takes" (length in-pos)
             "arguments, got" (length args)))
    (let* ((ins (map (lambda (desc pos) (encode-forward-input desc (list-ref args pos) pos))
                     in-descs in-pos))
           (out-bvs (map (lambda (desc) (make-bytevector (list-ref desc 3) 0))
                         out-descs))
           (in-ptr (pointer-array ins))
           (in-lens (u64-array (map bytevector-length ins)))
           (out-ptr (pointer-array out-bvs))
           (out-caps (u64-array (map (lambda (desc) (list-ref desc 3)) out-descs)))
           (err (make-slot))
           (st ((proc 'boom_run_multi) (compiled-forward-exe cf)
                (bytevector->pointer in-ptr) (bytevector->pointer in-lens)
                (length ins)
                (bytevector->pointer out-ptr) (bytevector->pointer out-caps)
                (length out-bvs)
                %null (slot-ptr err))))
      (sb-check! st err "compile-forward")
      (let ((results (map (lambda (desc bv pos)
                            (cons pos (bytevector->ctensor (cadr desc)
                                                           (caddr desc) bv)))
                          out-descs out-bvs out-pos)))
        (let ((ordered (map (lambda (i) (cdr (assv i results)))
                            (iota (length results)))))
          (if (compiled-forward-single? cf)
              (car ordered)
              ordered))))))

(define (call-compiled-forward cf args)
  (let ((lock (compiled-forward-lock cf)))
    (dynamic-wind
      (lambda () (lock-mutex lock))
      (lambda ()
        (when (compiled-forward-freed? cf)
          (error "compile-forward: the compiled forward has already been freed"))
        (run-compiled-forward cf args))
      (lambda () (unlock-mutex lock)))))

;;; --- public entry ------------------------------------------------------------

(define (resolve-backend backend)
  (case backend
    ((auto #f)
     (if (device-available?) 'device 'cpu))
    ((device cuda gpu)
     (unless (device-available?)
       (error "compile-forward: the loaded libsonicboom.so has no CUDA device; \
use #:backend 'cpu or a CUDA build" backend))
     'device)
    ((cpu) 'cpu)
    (else (error "compile-forward: backend must be auto, device or cpu" backend))))

(define* (compile-forward model inputs proc #:key (backend 'auto) (capture #t))
  "Compile PROC into a device inference graph whose weights come from MODEL.

PROC is a Scheme forward function written against the ops of (sonicboom trace
symbolic). It is called once, during compilation, with one symbolic tensor per
INPUTS entry in order, plus a weight accessor last: (w NAME DTYPE SHAPE)
returns the symbolic parameter for weight NAME. It must return a symbolic
tensor or a list of them with no data-dependent branch (`bool` on a tensor).

The resulting procedure takes one ctensor per input and returns the output
tensor (or the list of outputs, in return order). Runs of one compiled forward
are serialized; different compiled forwards may run in parallel.

Args:
  model: a weight store from load-model.
  inputs: a list of (name dtype shape), the graph's inputs in argument order.
  proc: the forward function.
  backend: keyword; 'auto (default) uses the device when the loaded library
    has CUDA and the CPU interpreter otherwise; 'device requires CUDA;
    'cpu forces the CPU interpreter.
  capture: keyword; CUDA-graph capture of the node launches for a device graph
    (default #t). Ignored by the CPU interpreter.
Returns: a procedure taking one ctensor per input.
PyTorch: torch.compile
Note: no example is given because one needs a weight file on disk. Unlike
torch.compile, this builds one traced forward graph (no graph breaks) whose
weights are bound by name from the store rather than captured from a module.
Weights must be float32 in the store (open it with #:convert-to-f32? for
f16/bf16/f64 files); the store may be freed after compiling."
  (unless (weights? model)
    (error "compile-forward: not a weight store" model))
  (when (weights-freed? model)
    (error "compile-forward: the weight store has already been freed"))
  (unless (procedure? proc)
    (error "compile-forward: proc must be a procedure" proc))
  (let ((inputs (map check-spec inputs)))
    (unless (equal? (delete-duplicates (map car inputs))
                    (map car inputs))
      (error "compile-forward: input names must be distinct" (map car inputs)))
    (let ((backend (resolve-backend backend)))
      (call-with-values
          (lambda () (trace-forward inputs proc))
        (lambda (nodes params outputs single?)
          (let* ((text (forward-text inputs outputs params nodes))
                 (exe (if (eq? backend 'device)
                          (compile-device-exe text model capture)
                          (compile-cpu-exe
                           text model
                           ;; Only a bound weight is bound from the store;
                           ;; an inline constant is already in the document.
                           (map (lambda (p)
                                  (list (gparam-bound p) (gparam-dtype p)
                                        (gparam-shape p)))
                                (filter gparam-bound params)))))
                 (cf (catch #t
                       (lambda ()
                         (let ((in-descs (query-descs exe 'in))
                               (out-descs (query-descs exe 'out)))
                           (%make-compiled-forward
                            exe backend in-descs out-descs
                            (input-plan in-descs inputs backend)
                            (output-plan out-descs outputs backend)
                            single? (make-mutex) #f)))
                       (lambda (key . args)
                         ;; No record owns the executable yet.
                         ((proc 'boom_executable_free) exe)
                         (apply throw key args))))
                 (callable (lambda args (call-compiled-forward cf args))))
            (%forward-guardian cf)
            (procedure-tag! callable 'sonicboom-compiled-forward cf)
            callable))))))

(define (compiled-forward-memory-stats fwd)
  "Device memory a compiled forward holds, in bytes.

A CPU-interpreter forward reports zero for every field.
Args:
  fwd: a compiled forward procedure.
Returns: an alist ((activation . bytes) (weights . bytes) (scratch . bytes)
  (total . bytes)).
PyTorch: none
Note: weights are owned by the weight store the graph was bound from, so the
same weight appears in the statistics of every graph bound to that store. The
nearest PyTorch concept is torch.cuda.memory_allocated, which reports the
allocator's total rather than a per-module breakdown."
  (let ((cf (and (procedure? fwd)
                 (procedure-tag fwd 'sonicboom-compiled-forward))))
    (unless cf
      (error "compiled-forward-memory-stats: not a compiled forward procedure" fwd))
    (let ((stats (make-bytevector 32 0))
          (err (make-slot)))
      (sb-check! ((proc 'boom_executable_memory_stats) (compiled-forward-exe cf)
                  (bytevector->pointer stats) (slot-ptr err))
                 err "compiled-forward-memory-stats")
      (list (cons 'activation (bytevector-u64-native-ref stats 0))
            (cons 'weights (bytevector-u64-native-ref stats 8))
            (cons 'scratch (bytevector-u64-native-ref stats 16))
            (cons 'total (bytevector-u64-native-ref stats 24))))))
