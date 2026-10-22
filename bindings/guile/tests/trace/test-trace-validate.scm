;;; SonicBoom — tracing prototype tests: S-Expr v0.2 draft validator
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root (about two minutes: it re-runs the other
;;; trace test files under a validating executor):
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-trace-validate.scm
;;;
;;; 1. Every graph the other trace test files execute validates against
;;;    docs/s-expr-v0.2-draft.md (each file runs in a subprocess through
;;;    validate-graphs.scm).
;;; 2. The examples of the draft validate.
;;; 3. Deliberately broken documents are rejected with the expected message
;;;    and location.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 format)
             (ice-9 rdelim)
             (sonicboom trace validate))

(define here (dirname (current-filename)))
(define scheme-dir
  (let ((p (search-path %load-path "sonicboom/trace/validate.scm")))
    (dirname (dirname (dirname p)))))

(define (read-report file)
  (call-with-input-file file
    (lambda (port)
      (let loop ((acc '()))
        (let ((x (read port)))
          (if (eof-object? x) (reverse acc) (loop (cons x acc))))))))

(define (has-error? errors needle) (any (lambda (e) (string-contains e needle)) errors))

(test-begin "sonicboom-trace-validate")

;;; --- 1. every graph the prototype tests execute ----------------------------------

(define total-graphs 0)

(for-each
 (lambda (name)
   (let* ((test (string-append here "/" name ".scm"))
          (report (string-append (or (getenv "TMPDIR") "/tmp") "/sonicboom-validate-"
                                 name "-" (number->string (getpid)) ".txt"))
          (status (system* "guile" "--no-auto-compile" "-L" scheme-dir
                           (string-append here "/validate-graphs.scm") test report))
          (lines (if (file-exists? report) (read-report report) '()))
          (ok (count (lambda (l) (eq? (car l) 'ok)) lines))
          (bad (filter (lambda (l) (eq? (car l) 'fail)) lines)))
     (when (file-exists? report) (delete-file report))
     (set! total-graphs (+ total-graphs ok))
     (format #t "~a: ~a distinct graphs validated, ~a invalid, test exit ~a~%"
             name ok (length bad) (status:exit-val status))
     (test-assert (string-append name ": every executed graph validates")
       (and (> ok 0) (null? bad)))
     (test-equal (string-append name ": still passes under the validating executor")
       0 (status:exit-val status))))
 '("test-trace" "test-trace-macro" "test-trace-forms" "test-trace-grad"
   "test-trace-transformer"))

(format #t "total: ~a distinct graphs validated~%" total-graphs)

;;; --- 2. the draft's examples ------------------------------------------------------

(define (doc . sections)
  (string-append "(sonicboom-s-expr (version 0 2) (graph (name \"ex\") "
                 (apply string-append sections) "))"))

(define ex-multi-output    ; draft section 1 / 2
  "(sonicboom-s-expr
  (version 0 2)
  (graph
    (name \"two-outputs\")
    (inputs
      (input \"arg0.0\" (tensor float32 (shape 2)))
      (input \"arg1\" (tensor float32 (shape 2)))
      (input \"arg0.1\" (tensor float32 (shape 2))))
    (outputs
      (output \"v0\")
      (output \"arg0.1\"))
    (parameters)
    (nodes
      (node add (inputs \"arg0.0\" \"arg1\") (outputs (\"v0\" (tensor float32 (shape 2))))
            (attrs (aten (string \"aten::add.Tensor\")))))))")

(define ex-constants       ; draft section 4
  (doc "(inputs (input \"arg0\" (tensor float32 (shape 2 2))))
        (outputs (output \"v4\"))
        (parameters
          (parameter \"c0\" (tensor int64 (shape 2)) (data :values 1 0))
          (parameter \"c2\" (tensor bool (shape 2 2)) (data :values #t #f #t #t))
          (parameter \"c3\" (tensor float32 (shape)) (data :values 0.0)))
        (nodes
          (node index_select (inputs \"arg0\" \"c0\") (outputs (\"v1\" (tensor float32 (shape 2 2))))
                (attrs (axis (int 0)) (aten (string \"aten::index_select\"))))
          (node where (inputs \"c2\" \"v1\" \"c3\") (outputs (\"v4\" (tensor float32 (shape 2 2))))
                (attrs (aten (string \"aten::where.self\")))))"))

(define ex-loop            ; draft section 5
  "(sonicboom-s-expr
  (version 0 2)
  (graph
    (name \"halve3\")
    (inputs
      (input \"arg0\" (tensor float32 (shape 2))))
    (outputs
      (output \"v14\"))
    (parameters)
    (nodes
      (node loop (inputs \"arg0\") (outputs (\"v14\" (tensor float32 (shape 2))))
            (attrs (body (string \"(sonicboom-s-expr (version 0 2) (graph (name \\\"loop-body\\\") (inputs (input \\\"s0\\\" (tensor float32 (shape 2)))) (outputs (output \\\"t1\\\")) (parameters (parameter \\\"p0\\\" (tensor float32 (shape)) (data :values 0.5))) (nodes (node mul (inputs \\\"s0\\\" \\\"p0\\\") (outputs (\\\"t1\\\" (tensor float32 (shape 2)))) (attrs (aten (string \\\"aten::mul.Tensor\\\")))))))\"))
                   (trip_count (int 3))
                   (state_count (int 1)))))))")

(define ex-print           ; draft section 6
  (doc "(inputs (input \"arg0\" (tensor float32 (shape 2))))
        (outputs (output \"v4\"))
        (parameters (parameter \"c0\" (tensor float32 (shape)) (data :values 2.0))
                    (parameter \"c3\" (tensor float32 (shape)) (data :values 1.0)))
        (nodes
          (node mul (inputs \"arg0\" \"c0\") (outputs (\"v1\" (tensor float32 (shape 2))))
                (attrs (aten (string \"aten::mul.Tensor\"))))
          (node print (inputs \"v1\") (outputs (\"v2\" (tensor float32 (shape 2))))
                (attrs (label (string \"dbl\"))))
          (node add (inputs \"arg0\" \"c3\") (outputs (\"v4\" (tensor float32 (shape 2))))
                (attrs (aten (string \"aten::add.Tensor\")))))"))

(define ex-attrs           ; draft section 8
  (doc "(inputs (input \"v1\" (tensor float32 (shape 2 2))))
        (outputs (output \"v2\"))
        (parameters)
        (nodes
          (node reduce_sum (inputs \"v1\") (outputs (\"v2\" (tensor float32 (shape 2 1))))
                (attrs (axes (ints 1)) (keepdims (int 1)) (aten (string \"aten::sum.dim_IntList\")))))"))

(for-each (lambda (name text)
            (test-equal (string-append "draft example validates: " name) '()
              (validate-sexpr-text text)))
          '("multi-output / list inputs" "constants" "loop" "print" "attributes")
          (list ex-multi-output ex-constants ex-loop ex-print ex-attrs))

(test-equal "a v0.1-style document (no aten) validates when aten is not required" '()
  (validate-sexpr-text
   (doc "(inputs (input \"a\" (tensor float32 (shape 2))))
         (outputs (output \"v\")) (parameters)
         (nodes (node relu (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2)))) (attrs)))")
   #:require-aten? #f))

;;; --- 3. counterexamples ------------------------------------------------------------

(define (rejects? label text needle . line)
  ;; TEXT must be rejected with an error containing NEEDLE (and LINE, if given)
  (let ((errors (validate-sexpr-text text)))
    (test-assert (string-append "rejects: " label)
      (and (has-error? errors needle)
           (or (null? line)
               (has-error? errors (string-append "line " (number->string (car line)) ",")))))))

(define (node-doc node-text . params)
  (doc "(inputs (input \"a\" (tensor float32 (shape 2 3))) (input \"b\" (tensor float32 (shape 3))))\n"
       "(outputs (output \"v\"))\n"
       "(parameters " (apply string-append params) ")\n"
       "(nodes\n" node-text ")"))

(rejects? "declared shape disagrees with the type rule"
  (node-doc "(node add (inputs \"a\" \"b\") (outputs (\"v\" (tensor float32 (shape 3 3)))) (attrs (aten (string \"aten::add.Tensor\"))))")
  "but the inputs give (tensor float32 (shape 2 3))" 5)
(rejects? "declared dtype disagrees (greater gives bool)"
  (node-doc "(node greater (inputs \"a\" \"b\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::gt.Tensor\"))))")
  "but the inputs give (tensor bool")
(rejects? "input used before its definition (topological order)"
  (node-doc "(node relu (inputs \"w\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::relu\"))))\n(node neg (inputs \"a\") (outputs (\"w\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::neg\"))))")
  "input \"w\" is used before it is defined" 5)
(rejects? "value defined twice"
  (node-doc "(node relu (inputs \"a\") (outputs (\"a\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::relu\"))))")
  "value \"a\" is defined twice")
(rejects? "missing aten attribute"
  (node-doc "(node relu (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs))")
  "missing (aten (string \"aten::relu\"))" 5)
(rejects? "aten attribute not matching the mapping table"
  (node-doc "(node relu (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::gelu\"))))")
  "does not match the mapping table entry \"aten::relu\"")
(rejects? "aten attribute on an op without a mapping (print)"
  (node-doc "(node print (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (label (string \"x\")) (aten (string \"aten::print\"))))")
  "has no ATen counterpart")
(rejects? "unknown op"
  (node-doc "(node conv (inputs \"a\" \"b\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs))")
  "unknown op conv")
(rejects? "attribute not defined for the op"
  (node-doc "(node relu (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (axis (int 0)) (aten (string \"aten::relu\"))))")
  "attribute axis is not defined for op relu")
(rejects? "attribute with the wrong tag"
  (node-doc "(node reduce_sum (inputs \"a\") (outputs (\"v\" (tensor float32 (shape)))) (attrs (axes (int 0)) (keepdims (int 0)) (aten (string \"aten::sum.dim_IntList\"))))")
  "attribute axes must be tagged ints")
(rejects? "constant with the wrong number of values"
  (node-doc "(node add (inputs \"a\" \"c\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::add.Tensor\"))))"
            "(parameter \"c\" (tensor float32 (shape 3)) (data :values 1.0 2.0))")
  "has 2 values for 3 elements" 3)
(rejects? "constant values that do not fit the dtype"
  (node-doc "(node relu (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::relu\"))))"
            "(parameter \"c\" (tensor int64 (shape 2)) (data :values 1.5 2))")
  "values do not fit dtype int64")
(rejects? "unknown dtype"
  (doc "(inputs (input \"a\" (tensor float8 (shape 2)))) (outputs) (parameters) (nodes)")
  "unknown dtype float8")
(rejects? "negative dim"
  (doc "(inputs (input \"a\" (tensor float32 (shape 2 -1)))) (outputs) (parameters) (nodes)")
  "dims must be static non-negative integers")
(rejects? "output not defined"
  (doc "(inputs (input \"a\" (tensor float32 (shape 2)))) (outputs (output \"zz\")) (parameters) (nodes)")
  "output \"zz\" is not defined")
(rejects? "output listed twice"
  (doc "(inputs (input \"a\" (tensor float32 (shape 2)))) (outputs (output \"a\") (output \"a\")) (parameters) (nodes)")
  "output \"a\" is listed twice")
(rejects? "unsupported version"
  "(sonicboom-s-expr (version 1 0) (graph (name \"g\") (inputs) (outputs) (parameters) (nodes)))"
  "unsupported version")
(rejects? "unknown graph section"
  (doc "(inputs) (outputs) (parameters) (nodes) (meta (author \"x\"))")
  "unknown graph section meta")
(rejects? "type rule failure (matmul inner dims)"
  (node-doc "(node matmul (inputs \"a\" \"a\") (outputs (\"v\" (tensor float32 (shape 2 3)))) (attrs (aten (string \"aten::mm\"))))")
  "type rule rejects the inputs: matmul")
(rejects? "loop body that does not validate (located inside the body)"
  (doc "(inputs (input \"a\" (tensor float32 (shape 2)))) (outputs (output \"v\")) (parameters)
        (nodes (node loop (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2))))
          (attrs (body (string \"(sonicboom-s-expr (version 0 2) (graph (name \\\"b\\\") (inputs (input \\\"s0\\\" (tensor float32 (shape 2)))) (outputs (output \\\"t\\\")) (parameters) (nodes (node relu (inputs \\\"s0\\\") (outputs (\\\"t\\\" (tensor float32 (shape 3)))) (attrs (aten (string \\\"aten::relu\\\")))))))\"))
                 (trip_count (int 2)) (state_count (int 1)))))")
  "(node loop) body > nodes[0] (node relu): declared output type")
(rejects? "loop body inputs not matching the node"
  (doc "(inputs (input \"a\" (tensor float32 (shape 2)))) (outputs (output \"v\")) (parameters)
        (nodes (node loop (inputs \"a\") (outputs (\"v\" (tensor float32 (shape 2))))
          (attrs (body (string \"(sonicboom-s-expr (version 0 2) (graph (name \\\"b\\\") (inputs (input \\\"q\\\" (tensor float32 (shape 2)))) (outputs (output \\\"q\\\")) (parameters) (nodes)))\"))
                 (trip_count (int 2)) (state_count (int 1)))))")
  "loop body inputs must be s0..s0")
(rejects? "text that does not parse"
  "(sonicboom-s-expr (version 0 2) (graph"
  "does not parse")

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-trace-validate")
(exit (if (zero? failures) 0 1))
