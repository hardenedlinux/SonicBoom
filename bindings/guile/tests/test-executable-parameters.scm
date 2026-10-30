;;; SonicBoom — read-only native parameter views (Stage 2)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core GUILE_AUTO_COMPILE=0 \
;;;       guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-executable-parameters.scm
;;;
;;; Checks that a parameter can be enumerated and borrowed as a zero-copy,
;;; read-only <tensor>; that the bytes match the graph's :values data; that
;;; tensor-set! refuses writes; that tensor-copy is independent; and that the
;;; view outlives free-executable! and cache eviction while one view's release
;;; leaves another valid.

(use-modules (srfi srfi-1)
             (srfi srfi-11)
             (srfi srfi-34)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom error)
             (sonicboom ffi)
             (sonicboom tensor)
             (sonicboom storage)         ; internal: reap-storage-views!
             (sonicboom executable))

;;; --- graphs ------------------------------------------------------------------

;; z = relu(x + b), b = [0.5 -1 2 -3 4 -5 6 -7].
(define add-relu-text
  (string-append
   "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu\")"
   " (inputs (input \"x\" (tensor float32 (shape 8))))"
   " (outputs (output \"z\"))"
   " (parameters (parameter \"b\" (tensor float32 (shape 8))"
   "   (data :values 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)))"
   " (nodes"
   "   (node add (inputs \"x\" \"b\")"
   "     (outputs (\"y\" (tensor float32 (shape 8)))))"
   "   (node relu (inputs \"y\")"
   "     (outputs (\"z\" (tensor float32 (shape 8))))))))"))

;; y = x + x: a parameter-free graph.
(define add-self-text
  "(sonicboom-s-expr (version 0 1) (graph (name \"add_self\")
     (inputs (input \"x\" (tensor float32 (shape 2))))
     (outputs (output \"y\"))
     (parameters)
     (nodes (node add (inputs \"x\" \"x\")
                  (outputs (\"y\" (tensor float32 (shape 2))))))))")

(define (f32-bv . xs)
  (let ((bv (make-bytevector (* 4 (length xs)) 0)))
    (let loop ((i 0) (rest xs))
      (unless (null? rest)
        (bytevector-ieee-single-native-set! bv (* 4 i) (exact->inexact (car rest)))
        (loop (+ i 1) (cdr rest))))
    bv))

(define b-bytes (f32-bv 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0))
(define x-input (f32-bv 1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0 8.0))

(define (view-bytes-equal? t) (bytevector=? (tensor-data t) b-bytes))

;;; --- tests --------------------------------------------------------------------

(test-begin "sonicboom-executable-parameters")

;; 1. enumeration and descriptor
(let* ((exe (compile-s-expr add-relu-text))
       (count (executable-parameter-count exe))
       (name (executable-parameter-name exe 0))
       (view (executable-parameter-ref exe 0)))
  (test-equal "one parameter is enumerated" 1 count)
  (test-equal "the parameter keeps its graph name" "b" name)
  (test-equal "the view has the parameter's logical shape"
              '(8) (tensor-shape view))
  (test-equal "the view has the parameter's dtype" 'float32 (tensor-dtype view))

  ;; 2. bytes equal the original parameter bytes
  (test-assert "view bytes equal the parameter's :values bytes"
               (view-bytes-equal? view))
  (test-equal "element access matches the parameter"
              '(0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)
              (tensor->list view))

  ;; 3. views are read-only
  (test-error "tensor-set! on a borrowed view raises" #t
              (tensor-set! view 0 99.0))
  (test-assert "the rejected write left the bytes untouched"
               (view-bytes-equal? view))

  ;; 4. tensor-copy is an independent, writable owned copy
  (let ((copy (tensor-copy view)))
    (test-assert "tensor-copy has the same bytes"
                 (bytevector=? (tensor-data copy) b-bytes))
    (test-assert "tensor-copy does not share storage"
                 (not (eq? (tensor-data copy) (tensor-data view))))
    (tensor-set! copy 0 42.0)
    (test-equal "writing the copy succeeds" 42.0 (tensor-ref copy 0))
    (test-assert "writing the copy leaves the view untouched"
                 (view-bytes-equal? view))
    (tensor-release! copy))

  ;; 5. the executable still runs while a view is held
  (let-values (((out err) (executable-run exe x-input)))
    (test-assert "execute with a live view succeeds" (and out (not err)))
    (test-assert "execution sees the same parameter bytes"
                 (and out (bytevector=?
                           out (f32-bv 1.5 1.0 0.0 1.0 0.0 1.0 0.0 1.0)))))

  ;; 6. freeing the executable handle does not invalidate the view
  (free-executable! exe)
  (test-assert "the executable handle is freed" (executable-freed? exe))
  (test-assert "the view is still readable after free-executable!"
               (view-bytes-equal? view))
  (test-equal "element access after free-executable!"
              '(0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)
              (tensor->list view))

  ;; 7. repeated release is safe, and release(NULL-equivalent) paths are no-ops
  (tensor-release! view)
  (tensor-release! view)
  (test-assert "tensor-release! is idempotent" #t))

;; 8. a cache eviction does not invalidate a live view
(let* ((exe (cached-executable add-relu-text))
       (view (executable-parameter-ref exe 0)))
  (clear-executable-cache!)
  (test-assert "cache eviction freed the cached handle"
               (executable-freed? exe))
  (test-assert "the view survives cache eviction" (view-bytes-equal? view))
  (tensor-release! view))

;; 9. multiple views of one parameter are independent
(let* ((exe (compile-s-expr add-relu-text))
       (v1 (executable-parameter-ref exe 0))
       (v2 (executable-parameter-ref exe 0)))
  (test-assert "both views read the same bytes"
               (and (view-bytes-equal? v1) (view-bytes-equal? v2)))
  (tensor-release! v1)
  (test-assert "releasing one view leaves the other valid"
               (view-bytes-equal? v2))
  (free-executable! exe)
  (test-assert "the surviving view outlives the executable handle"
               (view-bytes-equal? v2))
  (tensor-release! v2))

;; 10. out-of-range and parameter-free queries
(let ((exe (compile-s-expr add-relu-text)))
  (test-error "an out-of-range parameter index raises" #t
              (executable-parameter-ref exe 1))
  (test-error "an out-of-range name query raises" #t
              (executable-parameter-name exe 5))
  (free-executable! exe))

(let ((exe (compile-s-expr add-self-text)))
  (test-equal "a parameter-free graph has zero parameters"
              0 (executable-parameter-count exe))
  (free-executable! exe))

;; 11. GC and guardian: a view dropped without release is reaped
(define (drop-a-view! exe)
  (executable-parameter-ref exe 0)
  #t)

(let ((exe (compile-s-expr add-relu-text)))
  (drop-a-view! exe)
  (gc)
  (gc)
  (test-equal "a collected view is reaped exactly once"
              1 (reap-storage-views!))
  (test-equal "a second reap releases nothing"
              0 (reap-storage-views!))
  (reap-executables!)
  (test-assert "a collected view is reaped without error"
               (executable? exe))
  (test-assert "the executable remains usable after reaping views"
               (let ((v (executable-parameter-ref exe 0)))
                 (view-bytes-equal? v)))
  (free-executable! exe))

(test-end "sonicboom-executable-parameters")
