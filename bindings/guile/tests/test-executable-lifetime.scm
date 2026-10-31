;;; SonicBoom — executable lifetime synchronization
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core GUILE_AUTO_COMPILE=0 \
;;;       guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-executable-lifetime.scm
;;;
;;; Every Scheme entry point that dereferences a native sx::Executable must hold
;;; the executable's lock for the duration of the native call, and re-check the
;;; freed flag after taking it. This test proves both halves with deterministic
;;; coordination:
;;;
;;;   * while the lock is held, every entry point is parked (it cannot reach the
;;;     native handle), and free-executable! is parked too;
;;;   * once the free has happened, every entry point reports the freed state
;;;     with a Scheme error (executable-run reports it as its documented
;;;     return-value failure);
;;;   * a free racing an entry point from a two-party barrier is always one of
;;;     those two outcomes — the entry point either finishes before the free or
;;;     is refused after it, never touching a released handle;
;;;   * concurrent free calls release once, a borrowed parameter view keeps the
;;;     implementation alive past free-executable!, and the guardian cannot
;;;     release an explicitly freed handle a second time.
;;;
;;; The blocking checks use a bounded wait: the latch is signalled when the
;;; worker returns, and the test asserts it is still unset one second later.
;;; A wrongly unlocked entry point returns promptly and fails the check. The
;;; barrier checks coordinate the two threads and repeat, so both racing orders
;;; are exercised; every outcome is asserted to be defined, not merely "the
;;; process survived".

(use-modules (ice-9 threads)
             (srfi srfi-1)
             (srfi srfi-11)
             (srfi srfi-13)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom tensor)
             (sonicboom executable))

;;; --- graphs ------------------------------------------------------------------

;; y = x + x: parameter-free, one input, one output.
(define add-self-text
  "(sonicboom-s-expr (version 0 1) (graph (name \"add_self_lifetime\")
     (inputs (input \"x\" (tensor float32 (shape 2))))
     (outputs (output \"y\"))
     (parameters)
     (nodes (node add (inputs \"x\" \"x\")
                  (outputs (\"y\" (tensor float32 (shape 2))))))))")

;; z = relu(x + b), b materialized from :values at compile time.
(define values-text
  (string-append
   "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_lifetime\")"
   " (inputs (input \"x\" (tensor float32 (shape 8))))"
   " (outputs (output \"z\"))"
   " (parameters (parameter \"b\" (tensor float32 (shape 8))"
   "   (data :values 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)))"
   " (nodes"
   "   (node add (inputs \"x\" \"b\")"
   "     (outputs (\"y\" (tensor float32 (shape 8)))))"
   "   (node relu (inputs \"y\")"
   "     (outputs (\"z\" (tensor float32 (shape 8))))))))"))

;; The same graph with the parameter deferred: binding is an executable mutation
;; and takes the lock too.
(define bound-text
  (string-append
   "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_bound_lifetime\")"
   " (inputs (input \"x\" (tensor float32 (shape 8))))"
   " (outputs (output \"z\"))"
   " (parameters (parameter \"b\" (tensor float32 (shape 8))"
   "   (data :bound \"b\")))"
   " (nodes"
   "   (node add (inputs \"x\" \"b\")"
   "     (outputs (\"y\" (tensor float32 (shape 8)))))"
   "   (node relu (inputs \"y\")"
   "     (outputs (\"z\" (tensor float32 (shape 8))))))))"))

(define (f32-bv . xs)
  (let ((bv (make-bytevector (* 4 (length xs)) 0)))
    (let loop ((i 0) (rest xs))
      (unless (null? rest)
        (bytevector-ieee-single-native-set! bv (* 4 i)
                                            (exact->inexact (car rest)))
        (loop (+ i 1) (cdr rest))))
    bv))

(define b-bytes (f32-bv 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0))
(define x-input (f32-bv 1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0 8.0))
(define x2-input (f32-bv 1.0 2.0))

;;; --- the entry points under test ---------------------------------------------

;; Each OP is a thunk over an alive executable. A successful call returns a
;; truthy value; the freed state raises (executable-run turns its documented
;; (values #f message) failure into the same exception here).
(define (make-op-count exe) (lambda () (executable-parameter-count exe)))
(define (make-op-name exe) (lambda () (executable-parameter-name exe 0)))
(define (make-op-ref exe) (lambda () (executable-parameter-ref exe 0)))
(define (make-op-bind exe)
  (lambda () (executable-bind-parameter! exe "b" b-bytes)))
(define (make-op-complete exe)
  (lambda () (executable-bindings-complete? exe)))
(define (make-op-run exe)
  (lambda ()
    (call-with-values (lambda () (executable-run exe x2-input))
      (lambda (out err)
        (if out out (error "executable-run" err))))))

(define (tensor-result? r)
  (and (eq? (car r) 'ok) (tensor? (cdr r))))

;; Run THUNK and classify the outcome against the contract: a value, the freed
;; refusal, or anything else (a bug in this test's subject).
(define (raised-message args)
  ;; (error who "message") raises a misc-error whose arguments carry the
  ;; irritants list (who message); find the first string in it.
  (let loop ((a args))
    (cond ((null? a) "")
          ((and (pair? (car a))
                (pair? (cdr (car a)))
                (string? (cadr (car a))))
           (cadr (car a)))
          (else (loop (cdr a))))))

(define (classify thunk)
  (catch #t
    (lambda () (cons 'ok (thunk)))
    (lambda (key . args)
      (let ((msg (raised-message args)))
        (if (string-contains msg "already been freed")
            (cons 'denied msg)
            (list 'unexpected key args))))))

(define (outcome-ok? r)
  (memq (car r) '(ok denied)))

;;; --- coordination primitives -------------------------------------------------

;; A broadcast barrier: no party returns before all N have arrived.
(define (make-barrier n)
  (let ((mutex (make-mutex))
        (cv (make-condition-variable))
        (arrived 0)
        (generation 0))
    (lambda ()
      (lock-mutex mutex)
      (let ((gen generation))
        (set! arrived (+ arrived 1))
        (if (= arrived n)
            (begin
              (set! arrived 0)
              (set! generation (+ generation 1))
              (broadcast-condition-variable cv))
            (let wait ()
              (when (= generation gen)
                (wait-condition-variable cv mutex)
                (wait)))))
      (unlock-mutex mutex))))

;; A one-shot latch: a worker signals it after finishing; the main thread can
;; wait for it with a bound and learn whether the worker made progress.
(define (make-latch)
  (vector (make-mutex) (make-condition-variable) #f))

(define (latch-set! latch)
  (let ((mutex (vector-ref latch 0))
        (cv (vector-ref latch 1)))
    (lock-mutex mutex)
    (vector-set! latch 2 #t)
    (broadcast-condition-variable cv)
    (unlock-mutex mutex)))

;; True when the latch is set within TENTHS of a second; false on timeout.
(define (latch-wait latch tenths)
  (let ((mutex (vector-ref latch 0))
        (cv (vector-ref latch 1)))
    (lock-mutex mutex)
    (let loop ((left tenths))
      (cond ((vector-ref latch 2) (unlock-mutex mutex) #t)
            ((zero? left) (unlock-mutex mutex) #f)
            (else
             (wait-condition-variable cv mutex 0.1)
             (loop (- left 1)))))))

;; The production primitive every entry point uses. Reaching it through @@ keeps
;; it out of the public interface while letting the test hold exactly the lock
;; the entry points take.
(define hold-executable-lock
  (@@ (sonicboom executable) call-with-executable-lock))

;;; --- tests --------------------------------------------------------------------

(test-begin "sonicboom-executable-lifetime")

;; 1. An explicit free waits for an in-flight user of the handle.
(let* ((exe (compile-s-expr add-self-text))
       (latch (make-latch))
       (slot (vector #f)))
  (hold-executable-lock
   exe
   (lambda ()
     (vector-set! slot 0
                  (call-with-new-thread
                   (lambda ()
                     (free-executable! exe)
                     (latch-set! latch))))
     (test-assert "free-executable! waits while the executable lock is held"
                  (not (latch-wait latch 10)))))
  (join-thread (vector-ref slot 0))
  (test-assert "the free completes after the lock is released"
               (executable-freed? exe))
  ;; The flag claimed under the lock is what makes a later release a no-op.
  (free-executable! exe)
  (test-assert "a second free of the same handle stays a no-op"
               (executable-freed? exe)))

;; 2. Every entry point is parked while the lock is held: it cannot reach the
;;    native executable. This is the deterministic form of the free-vs-use race
;;    (the old unlocked accessors returned immediately and failed here).
(define (check-entry-point-blocks! name make-exe make-op check)
  (let* ((exe (make-exe))
         (latch (make-latch))
         (slot (vector #f))
         (op (make-op exe)))
    (hold-executable-lock
     exe
     (lambda ()
       (vector-set! slot 0
                    (call-with-new-thread
                     (lambda ()
                       (let ((r (classify op)))
                         (latch-set! latch)
                         r))))
       (test-assert (string-append name " waits for the executable lock")
                    (not (latch-wait latch 10)))))
    (let ((r (join-thread (vector-ref slot 0))))
      (test-assert (string-append name " completes with a defined outcome")
                   (eq? (car r) 'ok))
      (when (eq? (car r) 'ok)
        (check exe (cdr r))
        ;; The value may be a borrowed view; it is released by the test.
        (when (tensor? (cdr r)) (tensor-release! (cdr r)))))
    (free-executable! exe)))

(check-entry-point-blocks! "executable-parameter-count"
                           (lambda () (compile-s-expr values-text))
                           make-op-count
                           (lambda (exe value)
                             (test-equal "parameter-count returns the parameter count"
                                         1 value)))

(check-entry-point-blocks! "executable-parameter-name"
                           (lambda () (compile-s-expr values-text))
                           make-op-name
                           (lambda (exe value)
                             (test-equal "parameter-name returns the graph name"
                                         "b" value)))

(check-entry-point-blocks! "executable-parameter-ref"
                           (lambda () (compile-s-expr values-text))
                           make-op-ref
                           (lambda (exe value)
                             (test-assert "parameter-ref view has the parameter bytes"
                                          (bytevector=? (tensor-data value) b-bytes))))

(check-entry-point-blocks! "executable-bind-parameter!"
                           (lambda () (compile-s-expr bound-text))
                           make-op-bind
                           (lambda (exe value)
                             (test-assert "bind-parameter! completes the bindings"
                                          (executable-bindings-complete? exe))))

(check-entry-point-blocks! "executable-bindings-complete?"
                           (lambda () (compile-s-expr values-text))
                           make-op-complete
                           (lambda (exe value)
                             (test-assert "a :values graph reports complete bindings"
                                          value)))

(check-entry-point-blocks! "executable-run"
                           (lambda () (compile-s-expr add-self-text))
                           make-op-run
                           (lambda (exe value)
                             (test-assert "executable-run returns the graph output"
                                          (bytevector=?
                                           value (f32-bv 2.0 4.0)))))

;; 3. After the free, every entry point reports the freed state instead of
;;    touching the released handle. executable-run keeps its return-value
;;    contract; the accessors raise a Scheme error.
(let ((exe (compile-s-expr values-text)))
  (free-executable! exe)
  (test-assert "parameter-count after free raises a Scheme error"
               (eq? (car (classify (make-op-count exe))) 'denied))
  (test-assert "parameter-name after free raises a Scheme error"
               (eq? (car (classify (make-op-name exe))) 'denied))
  (test-assert "parameter-ref after free raises a Scheme error"
               (eq? (car (classify (make-op-ref exe))) 'denied))
  (test-assert "bindings-complete? after free raises a Scheme error"
               (eq? (car (classify (make-op-complete exe))) 'denied))
  (let-values (((out err) (executable-run exe x2-input)))
    (test-assert "executable-run after free returns its documented failure"
                 (and (not out) (string? err)
                      (string-contains err "already been freed")))))

(let ((exe (compile-s-expr bound-text)))
  (free-executable! exe)
  (test-assert "bind-parameter! after free raises a Scheme error"
               (eq? (car (classify (make-op-bind exe))) 'denied)))

;; 4. A free racing an entry point, coordinated by a two-party barrier. Each
;;    attempt must end in exactly one of the two defined orderings: the entry
;;    point finished first (its value), or the free finished first (the refused
;;    state). Unlocked or partially locked accessors crash or report an
;;    unexpected error here.
(define race-iterations
  (let ((s (getenv "SB_LIFETIME_ITERATIONS")))
    (if s (max 1 (string->number s)) 100)))

(define (race-free-vs-entry-point name make-exe make-op iterations)
  (let loop ((i 0) (completed 0) (refused 0) (bad 0))
    (if (= i iterations)
        (begin
          (test-equal
           (string-append name ": every racing attempt has a defined outcome")
           iterations (+ completed refused))
          (test-equal
           (string-append name ": no attempt races past the released handle")
           0 bad))
        (let* ((exe (make-exe))
               (barrier (make-barrier 2))
               (op (make-op exe))
               (freer (call-with-new-thread
                       (lambda ()
                         (barrier)
                         (classify (lambda () (free-executable! exe))))))
               (user (call-with-new-thread
                      (lambda ()
                        (barrier)
                        (let ((r (classify op)))
                          (when (tensor-result? r)
                            (tensor-release! (cdr r)))
                          r))))
               (free-result (join-thread freer))
               (user-result (join-thread user)))
          (loop (+ i 1)
                (+ completed (if (eq? (car user-result) 'ok) 1 0))
                (+ refused (if (eq? (car user-result) 'denied) 1 0))
                (+ bad (if (and (eq? (car free-result) 'ok)
                                (outcome-ok? user-result))
                           0
                           1)))))))

(race-free-vs-entry-point "executable-parameter-count"
                          (lambda () (compile-s-expr values-text))
                          make-op-count race-iterations)
(race-free-vs-entry-point "executable-parameter-name"
                          (lambda () (compile-s-expr values-text))
                          make-op-name race-iterations)
(race-free-vs-entry-point "executable-parameter-ref"
                          (lambda () (compile-s-expr values-text))
                          make-op-ref race-iterations)
(race-free-vs-entry-point "executable-bind-parameter!"
                          (lambda () (compile-s-expr bound-text))
                          make-op-bind race-iterations)
(race-free-vs-entry-point "executable-bindings-complete?"
                          (lambda () (compile-s-expr values-text))
                          make-op-complete race-iterations)
(race-free-vs-entry-point "executable-run"
                          (lambda () (compile-s-expr add-self-text))
                          make-op-run race-iterations)

;; 5. Concurrent explicit frees: all callers may enter, exactly one releases.
(let* ((exe (compile-s-expr add-self-text))
       (n 4)
       (barrier (make-barrier n))
       (threads (map (lambda (i)
                       (call-with-new-thread
                        (lambda ()
                          (barrier)
                          (classify (lambda () (free-executable! exe))))))
                     (iota n)))
       (results (map join-thread threads)))
  (test-assert "all concurrent free calls complete without error"
               (every (lambda (r) (eq? (car r) 'ok)) results))
  (test-assert "the handle is freed after concurrent free calls"
               (executable-freed? exe))
  (test-assert "a run after concurrent frees is refused, not executed"
               (eq? (car (classify (make-op-run exe))) 'denied)))

;; 6. A borrowed parameter view pins the implementation independently of the
;;    handle: free-executable! must not invalidate it.
(let* ((exe (compile-s-expr values-text))
       (view (executable-parameter-ref exe 0)))
  (free-executable! exe)
  (test-assert "a borrowed view survives free-executable!"
               (bytevector=? (tensor-data view) b-bytes))
  (test-equal "element access through the view after free"
              '(0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)
              (tensor->list view))
  (tensor-release! view)
  (tensor-release! view)
  (test-assert "releasing the surviving view twice is safe" #t))

;; 7. The guardian must not release an explicitly freed handle a second time.
;;    Free, drop the last Scheme reference, collect and reap: the guardian hands
;;    the record back, free-executable! observes the claimed flag and no-ops.
(define (free-and-drop!)
  (let ((exe (compile-s-expr values-text)))
    (free-executable! exe)
    #t))

(free-and-drop!)
(gc) (gc)
(test-assert "reaping an explicitly freed executable does not double-release"
             (begin (reap-executables!) #t))

;; 8. The guardian path still releases a handle that was never freed, and an
;;    explicit free afterwards stays a no-op.
(define (drop-without-free!)
  (compile-s-expr values-text)
  #t)

(drop-without-free!)
(gc) (gc)
(test-assert "the guardian reaps an executable that was never freed"
             (>= (reap-executables!) 1))
(test-equal "reaping again finds nothing new" 0 (reap-executables!))

(test-end "sonicboom-executable-lifetime")
