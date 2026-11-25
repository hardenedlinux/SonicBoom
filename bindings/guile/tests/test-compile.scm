;;; SonicBoom — Guile TTS forward-mode compilation: (sonicboom compile)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-compile.scm
;;;
;;; Writes its own tiny safetensors file (no model is needed) and checks:
;;;
;;;   1. load-model detects the format and reports an explicit-format mismatch;
;;;   2. compile-forward traces a forward function with each weight declared as
;;;      (data :bound ...), compiles it for the CPU interpreter, and computes
;;;      the expected values with the store's weights;
;;;   3. several inputs and list results keep their caller/return order;
;;;   4. the diagnostics name a missing weight, a shape mismatch with both
;;;      shapes, a non-f32 weight, a weight used with two shapes, and a
;;;      data-dependent branch;
;;;   5. f16 entries need #:convert-to-f32? and convert exactly;
;;;   6. the returned procedure owns its executable: freeing the store first is
;;;      safe, freeing the procedure is idempotent and refuses later runs;
;;;   7. two graphs share one store; repeat runs are bit-identical.
;;;
;;; The CPU interpreter is forced (#:backend 'cpu) so the test needs no GPU.

(use-modules (ice-9 binary-ports)
             (ice-9 exceptions)
             (ice-9 format)
             (srfi srfi-13)
             (srfi srfi-34)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom compile)
             (sonicboom error)
             (sonicboom ffi)
             (sonicboom trace tensor)
             ;; The traced forward functions compose these ops by name while
             ;; compile-forward runs them under the tracer.
             (sonicboom trace symbolic))

;;; --- a generated safetensors fixture -----------------------------------------

;; Safetensors: an 8-byte little-endian JSON header length, the JSON header
;; (padded with spaces to a multiple of 8), then the data section. Offsets are
;; relative to the start of the data section.
(define (le-u64 n)
  (let ((bv (make-bytevector 8 0)))
    ;; x86-64 is little-endian, which is the format's byte order.
    (bytevector-u64-native-set! bv 0 n)
    bv))

(define (float-bytes xs)
  (let ((bv (make-bytevector (* 4 (length xs)) 0)))
    (let loop ((i 0) (rest xs))
      (unless (null? rest)
        (bytevector-ieee-single-native-set! bv (* 4 i) (exact->inexact (car rest)))
        (loop (+ i 1) (cdr rest))))
    bv))

;; f32 -> f16 bits. The test converts only small integers, which are exact in
;; f16 (their mantissa's low 13 bits are zero), so rounding never comes up.
(define (f32->f16 v)
  (let* ((bv (make-bytevector 4 0)))
    (bytevector-ieee-single-native-set! bv 0 (exact->inexact v))
    (let* ((bits (bytevector-u32-native-ref bv 0))
           (sign (quotient bits #x80000000))
           (exp (modulo (quotient bits #x800000) #x100))
           (mant (modulo bits #x800000)))
      (+ (* sign #x8000)
         (cond ((= exp 0) 0)                       ; zero
               ((>= (+ exp -127 15) 31) #x7c00)    ; overflow -> inf
               (else (+ (* (+ exp -127 15) 1024) (quotient mant 8192))))))))

(define (f16-bytes xs)
  (let ((bv (make-bytevector (* 2 (length xs)) 0)))
    (let loop ((i 0) (rest xs))
      (unless (null? rest)
        (bytevector-u16-native-set! bv (* 2 i) (f32->f16 (car rest)))
        (loop (+ i 1) (cdr rest))))
    bv))

(define (write-safetensors path entries)
  "Write ENTRIES, a list of (name F32-or-F16 shape values), as safetensors."
  (let* ((packed (map (lambda (e)
                        (let* ((name (car e))
                               (dtype (cadr e))
                               (shape (caddr e))
                               (bv (if (eq? dtype 'F32)
                                       (float-bytes (cadddr e))
                                       (f16-bytes (cadddr e)))))
                          (list name dtype shape bv (bytevector-length bv))))
                      entries))
         (header-json
          (string-append
           "{"
           (string-join
            (let loop ((rest packed) (off 0) (acc '()))
              (if (null? rest)
                  (reverse acc)
                  (let* ((t (car rest))
                         (entry (format #f "~s:{\"dtype\":\"~a\",\"shape\":[~a],\"data_offsets\":[~a,~a]}"
                                        (symbol->string (car t)) (cadr t)
                                        (string-join (map number->string (caddr t)) ",")
                                        off (+ off (list-ref t 4)))))
                    (loop (cdr rest) (+ off (list-ref t 4))
                          (cons entry acc)))))
            ",")
           "}"))
         (pad (modulo (- 8 (modulo (string-length header-json) 8)) 8))
         (header (string->utf8 (string-append header-json (make-string pad #\space)))))
    (call-with-output-file path
      (lambda (port)
        (put-bytevector port (le-u64 (bytevector-length header)))
        (put-bytevector port header)
        (for-each (lambda (t) (put-bytevector port (list-ref t 3))) packed)))))

(define fixture-dir (mkdtemp "/tmp/sb-compile-XXXXXX"))
(define f32-path (string-append fixture-dir "/w.safetensors"))
(define f16-path (string-append fixture-dir "/w16.safetensors"))

;; w = [[1,-2],[3,-4],[5,-6]] (3x2), b = [[10,-10],[10,-10]] (2x2).
;; x = [[1,0,1],[0,1,1]]: x.w = [[6,-8],[8,-10]]; relu(x.w + b) = [[16,0],[18,0]].
(define w-values '(1.0 -2.0 3.0 -4.0 5.0 -6.0))
(define b-values '(10.0 -10.0 10.0 -10.0))

(unless (libsonicboom-available?)
  (error "test-compile: libsonicboom.so is not on the dynamic-link search path"))

(write-safetensors f32-path
                   (list (list 'w 'F32 '(3 2) w-values)
                         (list 'b 'F32 '(2 2) b-values)))
(write-safetensors f16-path
                   (list (list 'w 'F16 '(3 2) w-values)
                         (list 'b 'F16 '(2 2) b-values)))

;;; --- helpers -----------------------------------------------------------------

(define x-input (make-ctensor 'float32 '(2 3) '(1.0 0.0 1.0 0.0 1.0 1.0)))

(define (linear-xw x w) (matmul x (w "w" 'float32 '(3 2))))
(define (with-bias x w) (add (linear-xw x w) (w "b" 'float32 '(2 2))))
(define (forward x w) (relu (with-bias x w)))

;; The text of any condition, rendered: a &sonicboom-error's diagnostic, or
;; Guile's formatted message for a plain (error ...).
(define (condition-text c)
  (cond
   ((sonicboom-error? c) (sonicboom-error-message c))
   (else
    (let ((m (false-if-exception (exception-message c)))
          (i (false-if-exception (exception-irritants c))))
      (cond ((not (string? m)) (format #f "~s" c))
            ((and (list? i) (not (null? i)))
             (catch #t (lambda () (apply format #f m i)) (lambda _ m)))
            (else m))))))

(define (guard-error thunk)
  "The text of the exception THUNK raises, or #f when it returns."
  (call/cc
   (lambda (escape)
     (guard (c (#t (escape (condition-text c))))
       (thunk)
       #f))))

;; Comparisons tolerate the sign of zero, which a clamp may produce either way.
(define (scrub-zero v) (if (and (number? v) (zero? v)) 0.0 v))
(define (scrub nested)
  (if (pair? nested) (map scrub nested) (scrub-zero nested)))

(test-begin "guile-compile")

;;; --- 1. the weight store -----------------------------------------------------

(let ((store (load-model f32-path)))
  (test-assert "load-model detects safetensors" (weights? store))
  (test-equal "load-model keeps the path" f32-path (weights-path store))
  (test-assert "an explicit wrong format is an error"
               (guard-error (lambda () (load-model f32-path #:format 'gguf))))
  (free-weights! store))

;;; --- 2. a compiled forward ---------------------------------------------------

(let* ((store (load-model f32-path))
       (fwd (compile-forward store '((x float32 (2 3))) forward #:backend 'cpu)))
  (test-assert "compile-forward returns a procedure" (procedure? fwd))
  (test-assert "compiled-forward? recognizes it" (compiled-forward? fwd))
  (test-equal "the backend is cpu" 'cpu (compiled-forward-backend fwd))
  (test-equal "the forward computes relu(x.w + b)"
    '((16.0 0.0) (18.0 0.0)) (scrub (ctensor->nested (fwd x-input))))
  (test-equal "a repeat run is bit-identical"
    (ctensor-data (fwd x-input)) (ctensor-data (fwd x-input)))
  (test-equal "memory statistics are zero for the CPU interpreter"
    '((activation . 0) (weights . 0) (scratch . 0) (total . 0))
    (compiled-forward-memory-stats fwd))
  (test-assert "a wrong argument dtype is an error"
    (guard-error (lambda () (fwd (make-ctensor 'float16 '(2 3)
                                               (make-list 6 0.0))))))
  (test-assert "a wrong argument shape is an error"
    (guard-error (lambda () (fwd (make-ctensor 'float32 '(3 2)
                                               (make-list 6 0.0))))))
  (test-assert "a wrong argument count is an error"
    (guard-error (lambda () (fwd))))
  (free-compiled-forward! fwd)
  (test-assert "a freed compiled forward refuses to run"
    (guard-error (lambda () (fwd x-input))))
  (free-compiled-forward! fwd)          ; idempotent
  (free-weights! store))

;;; --- 3. several inputs, list results -----------------------------------------

(let* ((store (load-model f32-path))
       (two (lambda (x y w)
              (list (relu (with-bias x w))
                    (relu (with-bias y w)))))
       (fwd (compile-forward store
                             '((x float32 (2 3)) (y float32 (2 3)))
                             two #:backend 'cpu))
       (y-input (make-ctensor 'float32 '(2 3) '(1.0 1.0 0.0 1.0 0.0 1.0)))
       ;; y.w = [[4,-6],[6,-8]]; + b = [[14,-16],[16,-18]]; relu = [[14,0],[16,0]]
       (outs (fwd x-input y-input)))
  (test-equal "a list result keeps its order and arity" 2 (length outs))
  (test-equal "the first output" '((16.0 0.0) (18.0 0.0))
    (scrub (ctensor->nested (car outs))))
  (test-equal "the second output" '((14.0 0.0) (16.0 0.0))
    (scrub (ctensor->nested (cadr outs))))
  (free-compiled-forward! fwd)
  (free-weights! store))

;;; --- 4. diagnostics ----------------------------------------------------------

(let* ((store (load-model f32-path))
       (msg (guard-error
             (lambda ()
               (compile-forward store '((x float32 (2 3)))
                                (lambda (x w) (matmul x (w "nope" 'float32 '(3 2))))
                                #:backend 'cpu)))))
  (test-assert "a missing weight is reported by name"
               (and msg (string-contains msg "nope")))
  (free-weights! store))

(let* ((store (load-model f32-path))
       (msg (guard-error
             (lambda ()
               (compile-forward store '((x float32 (2 3)))
                                (lambda (x w) (matmul x (w "w" 'float32 '(3 3))))
                                #:backend 'cpu)))))
  (test-assert "a shape mismatch names both shapes"
               (and msg (string-contains msg "(3, 2)")
                    (string-contains msg "(3, 3)")))
  (free-weights! store))

(let* ((store (load-model f32-path))
       (msg (guard-error
             (lambda ()
               (compile-forward store '((x float32 (2 3)))
                                (lambda (x w) (matmul x (w "w" 'float16 '(3 2))))
                                #:backend 'cpu)))))
  (test-assert "a non-f32 weight is rejected at compile"
               (and msg (string-contains msg "float32")))
  (free-weights! store))

(let* ((store (load-model f32-path))
       (msg (guard-error
             (lambda ()
               (compile-forward
                store '((x float32 (2 3)))
                (lambda (x w)
                  (let ((t (linear-xw x w)))
                    (if (bool (amax t)) t (neg t))))
                #:backend 'cpu)))))
  (test-assert "a data-dependent branch is reported"
               (and msg (string-contains msg "data-dependent branch")))
  (free-weights! store))

(let* ((store (load-model f32-path))
       (msg (guard-error
             (lambda ()
               (compile-forward store '((x float32 (2 3)) (x float32 (2 3)))
                                forward #:backend 'cpu)))))
  (test-assert "duplicate input names are rejected"
               (and msg (string-contains msg "distinct")))
  (free-weights! store))

;; One name is one stored tensor: asking for it again with the same shape is
;; the same parameter, with a different shape is an error.
(let* ((store (load-model f32-path))
       (fwd (compile-forward store '((x float32 (2 3)))
                             (lambda (x w)
                               (let ((y (linear-xw x w)))
                                 (add y (relu y))))
                             #:backend 'cpu)))
  (test-equal "a weight used twice is one parameter"
    '((12.0 -8.0) (16.0 -10.0)) (scrub (ctensor->nested (fwd x-input))))
  (free-compiled-forward! fwd)
  (free-weights! store))

(let* ((store (load-model f32-path))
       (msg (guard-error
             (lambda ()
               (compile-forward store '((x float32 (2 3)))
                                (lambda (x w)
                                  (add (matmul x (w "w" 'float32 '(3 2)))
                                       (matmul x (w "w" 'float32 '(2 3)))))
                                #:backend 'cpu)))))
  (test-assert "one weight with two shapes is an error"
               (and msg (string-contains msg "two shapes")))
  (free-weights! store))

;;; --- 5. f16 conversion -------------------------------------------------------

(let* ((store (load-model f16-path #:convert-to-f32? #t))
       (fwd (compile-forward store '((x float32 (2 3))) forward #:backend 'cpu)))
  (test-equal "an f16 store converts exactly"
    '((16.0 0.0) (18.0 0.0)) (scrub (ctensor->nested (fwd x-input))))
  (free-compiled-forward! fwd)
  (free-weights! store))

(let* ((store (load-model f16-path))
       (msg (guard-error
             (lambda ()
               (compile-forward store '((x float32 (2 3))) forward #:backend 'cpu)))))
  (test-assert "an f16 store without conversion names #:convert-to-f32?"
               (and msg (string-contains msg "convert-to-f32")))
  (free-weights! store))

;;; --- 6. ownership and sharing ------------------------------------------------

(let* ((store (load-model f32-path))
       (fwd (compile-forward store '((x float32 (2 3))) forward #:backend 'cpu))
       (other (compile-forward store '((x float32 (2 3))) linear-xw #:backend 'cpu)))
  (free-weights! store)                 ; the executables keep the store alive
  (test-equal "a graph runs after its store handle is freed"
    '((16.0 0.0) (18.0 0.0)) (scrub (ctensor->nested (fwd x-input))))
  (test-equal "two graphs share one store"
    '((6.0 -8.0) (8.0 -10.0)) (scrub (ctensor->nested (other x-input))))
  (free-compiled-forward! fwd)
  (free-compiled-forward! other))

;;; --- 7. the backend choice ---------------------------------------------------

(let* ((store (load-model f32-path))
       (msg (guard-error
             (lambda ()
               (compile-forward store '((x float32 (2 3))) forward
                                #:backend 'tpu)))))
  (test-assert "an unknown backend is rejected"
               (and msg (string-contains msg "backend")))
  (free-weights! store))

;;; --- cleanup -----------------------------------------------------------------

(define %runner (test-runner-current))
(delete-file f32-path)
(delete-file f16-path)
(rmdir fixture-dir)
(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
