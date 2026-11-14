;;; SonicBoom — every documented example runs
;;;
;;; The examples in the docstrings are source code, so they can be checked
;;; mechanically. This test evaluates:
;;;
;;;   1. every Example: block of every exported symbol, in a fresh module that
;;;      imports the symbol's own module, with the documented `=> value`
;;;      compared against the last value the code produced;
;;;   2. every recipe of docs/site/pytorch-map.scm, after its own setup code,
;;;      in a fresh module;
;;;   3. every Scheme code block of docs/site/pages/getting-started.md.
;;;
;;; One example that runs longer than ten seconds is a failure. None of this
;;; loads a model file or needs a GPU; the examples that would are written
;;; without an Example: block, and the docstring says why.
;;;
;;; Needs libsonicboom.so on the dynamic-link search path (CTest sets
;;; LD_LIBRARY_PATH): the arithmetic of most modules goes through the C ABI.

(use-modules (ice-9 format)
             (ice-9 textual-ports)
             (srfi srfi-1)
             (srfi srfi-13)
             (srfi srfi-64))

;; bindings/guile/tests/docs/<this file> -> the repository root.
(define (repo-root)
  (let ((file (current-filename)))
    (canonicalize-path
     (dirname (dirname (dirname (dirname (dirname file))))))))

(load (string-append (repo-root) "/docs/site/extract.scm"))
(load (string-append (repo-root) "/docs/site/pytorch-map.scm"))

(define timeout-seconds 10)

(define (read-file path)
  (call-with-input-file path get-string-all))

;;; ------------------------------------------------------------ evaluation ---

(define (fresh-module module-name)
  (let ((module (make-fresh-user-module)))
    (module-use! module (resolve-module module-name))
    module))

;; Evaluate every datum of CODE-LINES in MODULE; the value of the last one is
;; the example's value.
(define (eval-code module code-lines)
  (let ((port (open-input-string (string-join code-lines "\n"))))
    (let loop ((value #f))
      (let ((datum (read port)))
        (if (eof-object? datum)
            value
            (loop (eval datum module)))))))

(define (single-value vals)
  (if (= (length vals) 1)
      (car vals)
      (cons 'values vals)))

;; (ok . value), (error . (KEY ARGS ...)) when the code raised, or
;; (timeout . #f) when it ran past the limit.
(define (call-with-timeout thunk)
  (catch 'docs-example-timeout
    (lambda ()
      (alarm timeout-seconds)
      (let ((result
             (catch #t
               (lambda () (call-with-values thunk list))
               (lambda (key . args) (cons 'error (cons key args))))))
        (alarm 0)
        (if (eq? (car result) 'error)
            result
            (cons 'ok (single-value result)))))
    (lambda (key . args)
      (alarm 0)
      (cons 'timeout #f))))

(define (written value)
  (string-trim (with-output-to-string (lambda () (write value)))))

;;; ----------------------------------------------------------------- report ---

(define failures '())
(define run-count 0)

(define (record-failure! kind name detail)
  (set! failures (cons (list kind name detail) failures))
  (format (current-error-port) "  FAIL ~a ~a: ~a~%" kind name detail))

;; Run one piece of code and, when there is an expected value, compare it.
(define (run-case kind name module-name code expected)
  (set! run-count (+ run-count 1))
  (let* ((module (fresh-module module-name))
         (start (get-internal-real-time))
         (outcome (call-with-timeout (lambda () (eval-code module code))))
         (elapsed (/ (- (get-internal-real-time) start)
                     (exact->inexact internal-time-units-per-second))))
    (cond
     ((eq? (car outcome) 'timeout)
      (record-failure! kind name
                       (format #f "ran longer than ~a s" timeout-seconds)))
     ((eq? (car outcome) 'error)
      (record-failure! kind name
                       (format #f "raised ~s" (cdr outcome))))
     ((> elapsed timeout-seconds)
      (record-failure! kind name
                       (format #f "took ~,1f s" elapsed)))
     (expected
      (let ((actual (written (cdr outcome))))
        (unless (string=? actual (string-trim expected))
          (record-failure! kind name
                           (format #f "expected ~s, got ~s"
                                   (string-trim expected) actual)))))
     (else #t))
    (eq? (car outcome) 'ok)))

;;; --------------------------------------------------------------- examples ---

(define modules (site-modules))

(define example-count 0)

(define (run-module-examples module)
  (let ((module-name (assq-ref module 'module)))
    (for-each
     (lambda (export)
       (let* ((doc (assq-ref export 'doc))
              (example (and doc (assq-ref doc 'example))))
         (when (and example (pair? (assq-ref example 'code)))
           (set! example-count (+ example-count 1))
           (run-case 'example
                     (string-append (assq-ref module 'name) " "
                                    (assq-ref export 'name))
                     module-name
                     (assq-ref example 'code)
                     (assq-ref example 'expected)))))
     (assq-ref module 'exports))))

(for-each run-module-examples modules)

;;; ---------------------------------------------------------------- recipes ---

(define (recipe-entries)
  (filter (lambda (entry)
            (let ((status (assq-ref (cdr entry) 'status)))
              (and (eq? (car entry) 'torch) status
                   (eq? (car status) 'recipe))))
          pytorch-map))

(define (entry-field entry key)
  (let ((field (assq-ref (cdr entry) key)))
    (and field (car field))))

(define recipe-count 0)

(for-each
 (lambda (entry)
   (set! recipe-count (+ recipe-count 1))
   (let* ((name (cadr entry))
          (setup (entry-field entry 'setup))
          (recipe (entry-field entry 'recipe))
          (module (fresh-module '(guile)))
          (outcome (call-with-timeout
                    (lambda ()
                      (when setup
                        (eval-code module (string-split setup #\newline)))
                      (eval-code module (string-split recipe #\newline))))))
     (cond
      ((eq? (car outcome) 'timeout)
       (record-failure! 'recipe name
                        (format #f "ran longer than ~a s" timeout-seconds)))
      ((eq? (car outcome) 'error)
       (record-failure! 'recipe name (format #f "raised ~s" (cdr outcome))))
      (else #t))))
 (recipe-entries))

;;; --------------------------------------------------------- getting started --

;; The Scheme code blocks of the getting-started page: a line "=> VALUE" inside
;; a block is that block's expected result.
(define (scheme-blocks path)
  (let loop ((lines (string-split (read-file path) #\newline))
             (fence #f)
             (block '())
             (blocks '()))
    (cond
     ((null? lines)
      (reverse (if fence (cons (reverse block) blocks) blocks)))
     ((and (not fence) (string-prefix? "```scheme" (string-trim (car lines))))
      (loop (cdr lines) #t '() blocks))
     ((and fence (string-prefix? "```" (string-trim (car lines))))
      (loop (cdr lines) #f '() (cons (reverse block) blocks)))
     (fence
      (loop (cdr lines) #t (cons (car lines) block) blocks))
     (else
      (loop (cdr lines) #f block blocks)))))

(define getting-started-count 0)

(let ((blocks (scheme-blocks
               (string-append (repo-root) "/docs/site/pages/getting-started.md"))))
  (for-each
   (lambda (block)
     (let* ((expected-line
             (find (lambda (line) (string-prefix? "=> " (string-trim line)))
                   block))
            (code (remove (lambda (line) (eq? line expected-line)) block))
            (expected (and expected-line
                           (string-trim (substring (string-trim expected-line) 3
                                                   (string-length
                                                    (string-trim expected-line)))))))
       (set! getting-started-count (+ getting-started-count 1))
       (run-case 'getting-started
                 (format #f "block ~a" getting-started-count)
                 '(guile) code expected)))
   blocks))

;;; ------------------------------------------------------------------ tests ---

(test-begin "docs-examples")
(define %runner (test-runner-current))

(format (current-error-port)
        "examples: ~a, recipes: ~a, getting-started blocks: ~a, failures: ~a~%"
        example-count recipe-count getting-started-count (length failures))

(test-assert "the documentation has examples and recipes"
  (and (> example-count 0) (> recipe-count 0) (> getting-started-count 0)))
(test-equal "every documented example, recipe and page block runs"
  0 (length failures))

(test-end "docs-examples")

(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
