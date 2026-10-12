;;; SonicBoom — tracing prototype: operator table (type rules + reference kernels)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; One entry per canonical S-Expr op-name. Each entry carries
;;;   - a type rule:  (input-types attrs) -> output type   used while tracing,
;;;   - a kernel:     (input-ctensors attrs) -> ctensor     used by eager mode and
;;;                                                         the reference interpreter,
;;;   - the corresponding PyTorch Core ATen operator name (#f if none),
;;;   - a gradient rule (#f: not differentiable), see "gradients" below.
;;; A type is a pair (dtype . shape). Keeping both in one table guarantees that
;;; the shapes recorded at trace time are the shapes the interpreter computes.
;;;
;;; This table is the prototype's own; it is not SonicBoom's operator registry.

(define-module (sonicboom trace ops)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace tensor)
  #:export (op-known? op-arity op-type op-kernel op-effectful? attr-ref
            op-aten-name op-gradient op-elementwise? sum-to-shape
            broadcast-shapes))

;;; --- attributes ---------------------------------------------------------------

;; ATTRS is the S-Expr attr list: ((name (tag v ...)) ...).
(define (attr-ref attrs name default)
  (let ((a (assq name attrs)))
    (if (not a)
        default
        (let ((tv (cadr a)))
          (case (car tv)
            ((int float string bool) (cadr tv))
            ((ints floats strings bools) (cdr tv))
            (else (error "attr-ref: unknown attribute tag" tv)))))))

;;; --- shapes -------------------------------------------------------------------

(define (broadcast-shapes . shapes)
  "NumPy-style broadcast of SHAPES; error if incompatible."
  (let* ((rank (apply max 0 (map length shapes)))
         (padded (map (lambda (s) (append (make-list (- rank (length s)) 1) s))
                      shapes)))
    (apply map
           (lambda dims
             (let ((big (filter (lambda (d) (not (= d 1))) dims)))
               (cond ((null? big) 1)
                     ((every (lambda (d) (= d (car big))) big) (car big))
                     (else (error "broadcast: incompatible shapes" shapes)))))
           padded)))

(define (broadcast-shapes* shapes)
  (if (every null? shapes) '() (apply broadcast-shapes shapes)))

;; Row-major strides of SHAPE, with stride 0 on broadcast dims, aligned to
;; OUT-SHAPE.
(define (broadcast-strides shape out-shape)
  (let* ((pad (- (length out-shape) (length shape)))
         (full (append (make-list pad 1) shape))
         (strides (let loop ((ds (reverse full)) (acc 1) (out '()))
                    (if (null? ds) out
                        (loop (cdr ds) (* acc (car ds)) (cons acc out))))))
    (map (lambda (d s) (if (= d 1) 0 s)) full strides)))

;; Call (PROC out-index operand-indices) for every element of OUT-SHAPE.
(define (for-each-broadcast out-shape operand-shapes proc)
  (let ((n (shape-numel out-shape))
        (strides (map (lambda (s) (broadcast-strides s out-shape))
                      operand-shapes)))
    (do ((i 0 (+ i 1))) ((= i n))
      ;; decompose i into a multi-index over OUT-SHAPE (last dim fastest)
      (let ((idx (let loop ((ds (reverse out-shape)) (r i) (acc '()))
                   (if (null? ds) acc
                       (loop (cdr ds) (quotient r (car ds))
                             (cons (remainder r (car ds)) acc))))))
        (proc i (map (lambda (st) (fold + 0 (map * idx st))) strides))))))

;;; --- type rules ----------------------------------------------------------------

(define (vmap f v) (list->vector (map f (vector->list v))))

(define (numeric? dtype) (not (eq? dtype 'bool)))

(define (same-dtype op types)
  (let ((d (car (car types))))
    (unless (every (lambda (t) (eq? (car t) d)) types)
      (error "op: operand dtypes differ" op (map car types)))
    d))

(define (binary-type op result-dtype)
  (lambda (types attrs)
    (let ((d (same-dtype op types)))
      (unless (numeric? d) (error "op: numeric operands required" op d))
      (cons (or result-dtype d) (broadcast-shapes* (map cdr types))))))

(define (compare-type op)
  (lambda (types attrs)
    (same-dtype op types)
    (cons 'bool (broadcast-shapes* (map cdr types)))))

(define (logical-type op)
  (lambda (types attrs)
    (unless (eq? 'bool (same-dtype op types))
      (error "op: bool operands required" op (map car types)))
    (cons 'bool (broadcast-shapes* (map cdr types)))))

(define (unary-type op)
  (lambda (types attrs)
    (let ((d (car (car types))))
      (unless (numeric? d) (error "op: numeric operand required" op d))
      (car types))))

(define (normalize-axes axes rank)
  (map (lambda (a)
         (let ((a* (if (< a 0) (+ a rank) a)))
           (unless (and (>= a* 0) (< a* rank))
             (error "reduce_sum: axis out of range" a rank))
           a*))
       axes))

(define (reduce-shape shape attrs)
  (let* ((rank (length shape))
         (axes (normalize-axes (attr-ref attrs 'axes '()) rank))
         (axes (if (null? axes) (iota rank) axes))
         (keep (= 1 (attr-ref attrs 'keepdims 1))))
    (values axes
            (append-map (lambda (d i)
                          (cond ((not (memv i axes)) (list d))
                                (keep (list 1))
                                (else '())))
                        shape (iota rank)))))

(define (matmul-type types attrs)
  (let ((d (same-dtype 'matmul types))
        (a (cdr (car types))) (b (cdr (cadr types))))
    (unless (and (= 2 (length a)) (= 2 (length b)) (= (cadr a) (car b)))
      (error "matmul: expects [m k] x [k n]" a b))
    (cons d (list (car a) (cadr b)))))

(define (reduce-type types attrs)
  (let ((t (car types)))
    (unless (numeric? (car t)) (error "reduce_sum: numeric operand required"))
    (call-with-values (lambda () (reduce-shape (cdr t) attrs))
      (lambda (axes shape) (cons (car t) shape)))))

(define (where-type types attrs)
  (let ((c (car types)) (x (cadr types)) (y (caddr types)))
    (unless (eq? (car c) 'bool) (error "where: condition must be bool" (car c)))
    (unless (eq? (car x) (car y)) (error "where: branch dtypes differ"))
    (cons (car x) (broadcast-shapes* (map cdr types)))))

(define (transpose-type types attrs)
  (let ((t (car types)))
    (unless (= 2 (length (cdr t))) (error "transpose: expects a 2-D tensor" (cdr t)))
    (cons (car t) (reverse (cdr t)))))

(define (reshape-type types attrs)
  (let ((t (car types)) (shape (attr-ref attrs 'shape #f)))
    (unless (and shape (= (shape-numel shape) (shape-numel (cdr t))))
      (error "reshape: element count differs" (cdr t) shape))
    (cons (car t) shape)))

(define (expand-type types attrs)
  (let ((t (car types)) (shape (attr-ref attrs 'shape #f)))
    (unless (and shape (equal? shape (broadcast-shapes* (list (cdr t) shape))))
      (error "expand: cannot broadcast" (cdr t) shape))
    (cons (car t) shape)))

;;; --- kernels ---------------------------------------------------------------------

(define (binary-kernel op f result-dtype)
  (lambda (ts attrs)
    (let* ((type ((binary-type op result-dtype)
                  (map (lambda (t) (cons (ctensor-dtype t) (ctensor-shape t))) ts)
                  attrs))
           (out (make-vector (shape-numel (cdr type)))))
      (for-each-broadcast (cdr type) (map ctensor-shape ts)
        (lambda (i js)
          (vector-set! out i (apply f (map ctensor-ref ts js)))))
      (make-ctensor (car type) (cdr type) out))))

(define (typed-binary-kernel type-rule f)
  (lambda (ts attrs)
    (let* ((type (type-rule (map (lambda (t) (cons (ctensor-dtype t) (ctensor-shape t)))
                                 ts)
                            attrs))
           (out (make-vector (shape-numel (cdr type)))))
      (for-each-broadcast (cdr type) (map ctensor-shape ts)
        (lambda (i js)
          (vector-set! out i (apply f (map ctensor-ref ts js)))))
      (make-ctensor (car type) (cdr type) out))))

(define (same-value? a b) (if (boolean? a) (eq? a b) (= a b)))

(define (unary-kernel f)
  (lambda (ts attrs)
    (let ((t (car ts)))
      (make-ctensor (ctensor-dtype t) (ctensor-shape t)
                    (vmap f (ctensor-data t))))))

(define (relu x) (if (> x 0) x (- x x)))

(define (matmul-kernel ts attrs)
  (let* ((a (car ts)) (b (cadr ts))
         (type (matmul-type (list (cons (ctensor-dtype a) (ctensor-shape a))
                                  (cons (ctensor-dtype b) (ctensor-shape b)))
                            attrs))
         (m (car (ctensor-shape a))) (k (cadr (ctensor-shape a)))
         (n (cadr (ctensor-shape b)))
         (out (make-vector (* m n))))
    (do ((i 0 (+ i 1))) ((= i m))
      (do ((j 0 (+ j 1))) ((= j n))
        (vector-set! out (+ (* i n) j)
                     (let loop ((p 0) (acc 0))
                       (if (= p k) acc
                           (loop (+ p 1)
                                 (+ acc (* (ctensor-ref a (+ (* i k) p))
                                           (ctensor-ref b (+ (* p n) j))))))))))
    (make-ctensor (car type) (cdr type) out)))

(define (reduce-kernel ts attrs)
  (let* ((t (car ts)) (shape (ctensor-shape t)))
    (call-with-values (lambda () (reduce-shape shape attrs))
      (lambda (axes out-shape)
        ;; Output index = input multi-index with the reduced axes zeroed, mapped
        ;; through the keepdims-shaped strides.
        (let* ((kept-shape (map (lambda (d i) (if (memv i axes) 1 d))
                                shape (iota (length shape))))
               (out (make-vector (shape-numel out-shape) 0)))
          (for-each-broadcast shape (list kept-shape)
            (lambda (i js)
              (vector-set! out (car js)
                           (+ (vector-ref out (car js)) (ctensor-ref t i)))))
          (make-ctensor (ctensor-dtype t) out-shape out))))))

(define (where-kernel ts attrs)
  (let* ((type (where-type (map (lambda (t) (cons (ctensor-dtype t)
                                                  (ctensor-shape t)))
                                ts)
                           attrs))
         (out (make-vector (shape-numel (cdr type)))))
    (for-each-broadcast (cdr type) (map ctensor-shape ts)
      (lambda (i js)
        (vector-set! out i (if (ctensor-ref (car ts) (car js))
                               (ctensor-ref (cadr ts) (cadr js))
                               (ctensor-ref (caddr ts) (caddr js))))))
    (make-ctensor (car type) (cdr type) out)))

;; print: identity on the value, writes "label: nested-values" as an effect.
(define (print-kernel ts attrs)
  (let ((t (car ts)) (port (current-output-port)))
    (display (attr-ref attrs 'label "print") port)
    (display ": " port)
    (write (ctensor->nested t) port)
    (newline port)
    t))

(define (transpose-kernel ts attrs)
  (let* ((t (car ts)) (m (car (ctensor-shape t))) (n (cadr (ctensor-shape t)))
         (out (make-vector (* m n))))
    (do ((i 0 (+ i 1))) ((= i m))
      (do ((j 0 (+ j 1))) ((= j n))
        (vector-set! out (+ (* j m) i) (ctensor-ref t (+ (* i n) j)))))
    (make-ctensor (ctensor-dtype t) (list n m) out)))

(define (reshape-kernel ts attrs)
  (let ((t (car ts)))
    (make-ctensor (ctensor-dtype t) (attr-ref attrs 'shape #f) (ctensor-data t))))

(define (expand-kernel ts attrs)
  (let* ((t (car ts)) (shape (attr-ref attrs 'shape #f))
         (out (make-vector (shape-numel shape))))
    (for-each-broadcast shape (list (ctensor-shape t))
      (lambda (i js) (vector-set! out i (ctensor-ref t (car js)))))
    (make-ctensor (ctensor-dtype t) shape out)))

;;; --- gradients ----------------------------------------------------------------------
;;; A gradient rule is a pure function
;;;   (rule emit shape ins out g attrs) -> list of input gradients (#f = none)
;;; INS / OUT are the forward input and output values, G the gradient of OUT.
;;; EMIT is (emit op args attrs) -> value: it creates the backward node (args
;;; may be Scheme numbers, lifted to constants). SHAPE gives a value's shape.
;;; Rules of elementwise ops return input gradients at the OUTPUT shape; the
;;; autodiff engine masks them (double where) and then reduces them to the
;;; input shape with sum-to-shape. Other rules return input-shaped gradients.

(define (ints . xs) `(ints ,@xs))

(define (sum-to-shape emit shape g target)
  "Reduce G (broadcast result) to shape TARGET: sum the leading axes and the
axes TARGET has as 1, then reshape."
  (let* ((gs (shape g))
         (lead (- (length gs) (length target))))
    (if (equal? gs target)
        g
        (let ((axes (filter (lambda (i)
                              (or (< i lead)
                                  (and (= 1 (list-ref target (- i lead)))
                                       (not (= 1 (list-ref gs i))))))
                            (iota (length gs)))))
          (emit 'reshape
                (list (emit 'reduce_sum (list g)
                            `((axes ,(apply ints axes)) (keepdims (int 1)))))
                `((shape ,(apply ints target))))))))

(define (grad-add e shape ins out g attrs) (list g g))
(define (grad-sub e shape ins out g attrs) (list g (e 'neg (list g) '())))
(define (grad-mul e shape ins out g attrs)
  (list (e 'mul (list g (cadr ins)) '()) (e 'mul (list g (car ins)) '())))
(define (grad-div e shape ins out g attrs)
  (let ((a (car ins)) (b (cadr ins)))
    (list (e 'div (list g b) '())
          (e 'neg (list (e 'div (list (e 'mul (list g a) '())
                                      (e 'mul (list b b) '()))
                           '()))
             '()))))
(define (grad-neg e shape ins out g attrs) (list (e 'neg (list g) '())))
(define (grad-relu e shape ins out g attrs)
  (list (e 'where (list (e 'greater (list (car ins) 0) '()) g 0) '())))
(define (grad-where e shape ins out g attrs)
  (let ((c (car ins)))
    (list #f (e 'where (list c g 0) '()) (e 'where (list c 0 g) '()))))
(define (grad-identity e shape ins out g attrs) (list g))
(define (grad-matmul e shape ins out g attrs)
  (let ((a (car ins)) (b (cadr ins)))
    (list (e 'matmul (list g (e 'transpose (list b) '())) '())
          (e 'matmul (list (e 'transpose (list a) '()) g) '()))))
(define (grad-transpose e shape ins out g attrs) (list (e 'transpose (list g) '())))
(define (grad-reshape e shape ins out g attrs)
  (list (e 'reshape (list g) `((shape ,(apply ints (shape (car ins))))))))
(define (grad-expand e shape ins out g attrs)
  (list (sum-to-shape e shape g (shape (car ins)))))
(define (grad-reduce-sum e shape ins out g attrs)
  (let* ((in-shape (shape (car ins)))
         (rank (length in-shape))
         (axes (normalize-axes (attr-ref attrs 'axes '()) rank))
         (axes (if (null? axes) (iota rank) axes))
         (kept (map (lambda (d i) (if (memv i axes) 1 d)) in-shape (iota rank))))
    (list (e 'expand (list (e 'reshape (list g) `((shape ,(apply ints kept)))))
             `((shape ,(apply ints in-shape)))))))
(define (no-grad e shape ins out g attrs) (map (lambda (x) #f) ins))

;;; --- table ------------------------------------------------------------------------

;; name -> (arity type-rule kernel aten-name gradient-rule elementwise?)
;; ATen names are PyTorch Core ATen IR operators (design doc, op table).
(define %ops
  `((add       2 ,(binary-type 'add #f) ,(binary-kernel 'add + #f)
               "aten::add.Tensor" ,grad-add #t)
    (sub       2 ,(binary-type 'sub #f) ,(binary-kernel 'sub - #f)
               "aten::sub.Tensor" ,grad-sub #t)
    (mul       2 ,(binary-type 'mul #f) ,(binary-kernel 'mul * #f)
               "aten::mul.Tensor" ,grad-mul #t)
    (div       2 ,(binary-type 'div #f) ,(binary-kernel 'div / #f)
               "aten::div.Tensor" ,grad-div #t)
    (greater   2 ,(binary-type 'greater 'bool) ,(binary-kernel 'greater > 'bool)
               "aten::gt.Tensor" ,no-grad #f)
    (less      2 ,(binary-type 'less 'bool) ,(binary-kernel 'less < 'bool)
               "aten::lt.Tensor" ,no-grad #f)
    (equal     2 ,(compare-type 'equal)
               ,(typed-binary-kernel (compare-type 'equal) same-value?)
               "aten::eq.Tensor" ,no-grad #f)
    (and       2 ,(logical-type 'and)
               ,(typed-binary-kernel (logical-type 'and) (lambda (a b) (and a b)))
               "aten::logical_and" ,no-grad #f)
    (or        2 ,(logical-type 'or)
               ,(typed-binary-kernel (logical-type 'or) (lambda (a b) (or a b)))
               "aten::logical_or" ,no-grad #f)
    (not       1 ,(logical-type 'not) ,(typed-binary-kernel (logical-type 'not) not)
               "aten::logical_not" ,no-grad #f)
    (matmul    2 ,matmul-type ,matmul-kernel "aten::mm" ,grad-matmul #f)
    (transpose 1 ,transpose-type ,transpose-kernel "aten::permute" ,grad-transpose #f)
    (reshape   1 ,reshape-type ,reshape-kernel "aten::view" ,grad-reshape #f)
    (expand    1 ,expand-type ,expand-kernel "aten::expand" ,grad-expand #f)
    (relu      1 ,(unary-type 'relu) ,(unary-kernel relu) "aten::relu" ,grad-relu #t)
    (neg       1 ,(unary-type 'neg) ,(unary-kernel -) "aten::neg" ,grad-neg #t)
    (reduce_sum 1 ,reduce-type ,reduce-kernel "aten::sum.dim_IntList"
               ,grad-reduce-sum #f)
    (where     3 ,where-type ,where-kernel "aten::where.self" ,grad-where #t)
    (print     1 ,(lambda (types attrs) (car types)) ,print-kernel
               #f ,grad-identity #t)))

;; Ops whose kernel has an effect: always recorded while tracing and never
;; removed by DCE.
(define %effectful-ops '(print))
(define (op-effectful? op) (and (memq op %effectful-ops) #t))

(define (entry op)
  (or (assq-ref %ops op) (error "unsupported op" op)))

(define (op-known? op) (and (assq op %ops) #t))
(define (op-arity op) (car (entry op)))

(define (check-arity op n)
  (unless (= n (op-arity op))
    (error "op: wrong number of inputs" op n)))

(define (op-type op types attrs)
  (check-arity op (length types))
  ((cadr (entry op)) types attrs))

(define (op-kernel op tensors attrs)
  (check-arity op (length tensors))
  ((caddr (entry op)) tensors attrs))

(define (op-aten-name op) (and (op-known? op) (list-ref (entry op) 3)))
(define (op-gradient op) (and (op-known? op) (list-ref (entry op) 4)))
(define (op-elementwise? op) (and (op-known? op) (list-ref (entry op) 5)))
