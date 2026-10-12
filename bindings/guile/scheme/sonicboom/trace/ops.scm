;;; SonicBoom — tracing prototype: operator table (type rules + reference kernels)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; One entry per canonical S-Expr op-name. Each entry carries
;;;   - a type rule:  (input-types attrs) -> output type   used while tracing,
;;;   - a kernel:     (input-ctensors attrs) -> ctensor     used by eager mode and
;;;                                                         the reference interpreter.
;;; A type is a pair (dtype . shape). Keeping both in one table guarantees that
;;; the shapes recorded at trace time are the shapes the interpreter computes.
;;;
;;; This table is the prototype's own; it is not SonicBoom's operator registry.

(define-module (sonicboom trace ops)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace tensor)
  #:export (op-known? op-arity op-type op-kernel attr-ref
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

;;; --- table ------------------------------------------------------------------------

;; name -> (arity type-rule kernel)
(define %ops
  `((add       2 ,(binary-type 'add #f)       ,(binary-kernel 'add + #f))
    (sub       2 ,(binary-type 'sub #f)       ,(binary-kernel 'sub - #f))
    (mul       2 ,(binary-type 'mul #f)       ,(binary-kernel 'mul * #f))
    (greater   2 ,(binary-type 'greater 'bool) ,(binary-kernel 'greater > 'bool))
    (less      2 ,(binary-type 'less 'bool)    ,(binary-kernel 'less < 'bool))
    (matmul    2 ,matmul-type                 ,matmul-kernel)
    (relu      1 ,(unary-type 'relu)          ,(unary-kernel relu))
    (neg       1 ,(unary-type 'neg)           ,(unary-kernel -))
    (reduce_sum 1 ,reduce-type                ,reduce-kernel)
    (where     3 ,where-type                  ,where-kernel)))

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
