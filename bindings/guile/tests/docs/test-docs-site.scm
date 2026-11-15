;;; SonicBoom — the generated site is self-contained and internally sound
;;;
;;; The site is generated into a temporary directory with docs/site/build.scm,
;;; then every page is read back. The checks are the ones the site promises:
;;;
;;;   1. every href and src is relative (it does not start with "/") and not
;;;      absolute, and the only external links are to the PyTorch
;;;      documentation; nothing mentions hardenedvault;
;;;   2. every relative target is a file in the site, and every "#anchor" names
;;;      an id on its page;
;;;   3. search-index.json and pytorch-map.json parse as JSON;
;;;   4. index.html carries the redirect script that makes /sonicboom reach the
;;;      home page;
;;;   5. every page's <head> holds, directly, only meta, title, link and script
;;;      elements and whitespace, and no page holds a <style> element
;;;      (docs/site/check-html.py).
;;;
;;; Needs pandoc and python3; no native library and no network.

(use-modules (ice-9 format)
             (ice-9 popen)
             (ice-9 rdelim)
             (ice-9 textual-ports)
             (ice-9 ftw)
             (ice-9 regex)
             (srfi srfi-1)
             (srfi srfi-13)
             (srfi srfi-64))

;; bindings/guile/tests/docs/<this file> -> the repository root.
(define (repo-root)
  (let ((file (current-filename)))
    (canonicalize-path
     (dirname (dirname (dirname (dirname (dirname file))))))))

(define (read-file path)
  (call-with-input-file path get-string-all))

;;; ------------------------------------------------------------ temp output ---

(define site-output
  (string-append (or (getenv "TMPDIR") "/tmp")
                 "/sb-docs-site-test-" (number->string (getpid))))

(define (delete-tree path)
  (when (file-exists? path)
    (if (eq? (stat:type (stat path)) 'directory)
        (begin
          (for-each (lambda (name)
                      (unless (member name '("." ".."))
                        (delete-tree (string-append path "/" name))))
                    (scandir path))
          (rmdir path))
        (delete-file path))))

(delete-tree site-output)

(define (generate-site)
  (let ((status (system* "guile" "--no-auto-compile"
                         (string-append (repo-root) "/docs/site/build.scm")
                         "--out" site-output)))
    (and (status:exit-val status) (zero? (status:exit-val status)))))

(define generated? (generate-site))
(define site (string-append site-output "/sonicboom"))

;;; ------------------------------------------------------------- page walk ---

(define (html-files directory)
  (append-map
   (lambda (name)
     (if (member name '("." ".."))
         '()
         (let ((path (string-append directory "/" name)))
           (if (eq? (stat:type (stat path)) 'directory)
               (html-files path)
               (if (string-suffix? ".html" name) (list path) '())))))
   (scandir directory)))

(define (relative-url path)
  (substring path (+ (string-length site) 1) (string-length path)))

;;; ------------------------------------------------------------------ json ---

;; A recursive-descent JSON syntax check; throws (json-error POSITION) on a
;; malformed document.
(define (json-check text)
  (define text-length (string-length text))
  (define position 0)

  (define (peek) (and (< position text-length) (string-ref text position)))
  (define (fail) (throw 'json-error position))
  (define (skip-space)
    (let loop ()
      (let ((c (peek)))
        (when (and c (memv c '(#\space #\tab #\newline #\return)))
          (set! position (+ position 1))
          (loop)))))
  (define (expect c)
    (skip-space)
    (if (eqv? (peek) c)
        (set! position (+ position 1))
        (fail)))
  (define (has-prefix? prefix)
    (and (<= (+ position (string-length prefix)) text-length)
         (string=? (substring text position
                              (+ position (string-length prefix)))
                   prefix)))
  (define (literal word)
    (if (has-prefix? word)
        (set! position (+ position (string-length word)))
        (fail)))

  (define (string-value)
    (set! position (+ position 1))
    (let loop ()
      (let ((c (peek)))
        (cond ((not c) (fail))
              ((char=? c #\") (set! position (+ position 1)))
              ((char=? c #\\) (set! position (+ position 2)) (loop))
              (else (set! position (+ position 1)) (loop))))))

  (define (number-value)
    (let loop ((digits 0))
      (let ((c (peek)))
        (if (and c (or (char-numeric? c)
                       (memv c '(#\- #\+ #\. #\e #\E))))
            (begin
              (when (char-numeric? c) (set! digits (+ digits 1)))
              (set! position (+ position 1))
              (loop digits))
            (when (zero? digits) (fail))))))

  (define (array-value)
    (set! position (+ position 1))
    (skip-space)
    (if (eqv? (peek) #\])
        (set! position (+ position 1))
        (let loop ()
          (value)
          (skip-space)
          (let ((c (peek)))
            (cond ((eqv? c #\,) (set! position (+ position 1)) (loop))
                  ((eqv? c #\]) (set! position (+ position 1)))
                  (else (fail)))))))

  (define (object-value)
    (set! position (+ position 1))
    (skip-space)
    (if (eqv? (peek) #\})
        (set! position (+ position 1))
        (let loop ()
          (skip-space)
          (if (eqv? (peek) #\") (string-value) (fail))
          (expect #\:)
          (value)
          (skip-space)
          (let ((c (peek)))
            (cond ((eqv? c #\,) (set! position (+ position 1)) (loop))
                  ((eqv? c #\}) (set! position (+ position 1)))
                  (else (fail)))))))

  (define (value)
    (skip-space)
    (cond ((not (peek)) (fail))
          ((eqv? (peek) #\{) (object-value))
          ((eqv? (peek) #\[) (array-value))
          ((eqv? (peek) #\") (string-value))
          ((or (eqv? (peek) #\-) (char-numeric? (peek))) (number-value))
          ((eqv? (peek) #\t) (literal "true"))
          ((eqv? (peek) #\f) (literal "false"))
          ((eqv? (peek) #\n) (literal "null"))
          (else (fail))))

  (value)
  (skip-space)
  (when (< position text-length)
    (fail))
  #t)

;;; ----------------------------------------------------------- link checks ---

(define link-pattern (make-regexp "(href|src)=\"([^\"]*)\""))
(define allowed-external "https://pytorch.org/docs/stable/")

(define problems '())

(define (problem! text)
  (set! problems (cons text problems)))

(define (normalize-path path)
  (let loop ((parts (string-split path #\/)) (out '()))
    (cond ((null? parts) (string-join (reverse out) "/"))
          ((or (string=? (car parts) "") (string=? (car parts) "."))
           (loop (cdr parts) out))
          ((string=? (car parts) "..")
           (loop (cdr parts) (if (pair? out) (cdr out) out)))
          (else (loop (cdr parts) (cons (car parts) out))))))

(define (anchor-exists? file fragment)
  (and (string? fragment)
       (not (string=? fragment ""))
       (let ((text (read-file file)))
         (or (string-contains text (string-append "id=\"" fragment "\""))
             (string-contains text (string-append "name=\"" fragment "\""))))))

(define (check-link page target)
  (let ((page-url (relative-url page))
        (page-dir (dirname (relative-url page))))
    (cond
     ((string=? target "")
      (problem! (format #f "~a: empty link" page-url)))
     ((string-contains target "hardenedvault")
      (problem! (format #f "~a: link mentions hardenedvault: ~a" page-url target)))
     ((string-prefix? "/" target)
      (problem! (format #f "~a: absolute link: ~a" page-url target)))
     ((string-contains target "://")
      (unless (string-prefix? allowed-external target)
        (problem! (format #f "~a: external link: ~a" page-url target))))
     ((string-prefix? "#" target)
      (unless (anchor-exists? page (substring target 1 (string-length target)))
        (problem! (format #f "~a: no anchor ~a on the page" page-url target))))
     (else
      (let* ((parts (string-split target #\#))
             ;; A query (pytorch.html?q=...) is not part of the file name.
             (path (car (string-split (car parts) #\?)))
             (fragment (and (pair? (cdr parts)) (cadr parts)))
             (file (normalize-path
                    (string-append (if (string=? page-dir ".") ""
                                       (string-append page-dir "/"))
                                   path))))
        (if (not (file-exists? (string-append site "/" file)))
            (problem! (format #f "~a: missing target ~a" page-url target))
            (when fragment
              (unless (anchor-exists?
                       (string-append site "/" file) fragment)
                (problem! (format #f "~a: no anchor ~a in ~a"
                                  page-url target file))))))))))

(define (check-page page)
  (let ((text (read-file page)))
    (let loop ((start 0))
      (let ((match (regexp-exec link-pattern text start)))
        (when match
          (check-link page (match:substring match 2))
          (when (= (match:start match 0) (match:end match 0))
            (error "link pattern matched the empty string"))
          (loop (match:end match 0)))))))

;;; -------------------------------------------------------- html structure ---

;; One line per problem; docs/site/check-html.py holds the rules.
(define (check-page-structure page)
  (let* ((port (open-input-pipe
                (string-append "python3 "
                               (string-append (repo-root) "/docs/site/check-html.py")
                               " " page)))
         (lines (let loop ((line (read-line port)) (acc '()))
                  (if (eof-object? line)
                      (reverse acc)
                      (loop (read-line port) (cons line acc)))))
         (status (close-pipe port)))
    (if (and (not (zero? status)) (null? lines))
        (list (format #f "~a: the HTML structure check failed" page))
        lines)))

;;; ------------------------------------------------------------------ tests ---

(test-begin "docs-site")
(define %runner (test-runner-current))

(test-assert "the site generator ran" generated?)

(when generated?
  (for-each check-page (html-files site))

  (format (current-error-port) "pages: ~a, problems: ~a~%"
          (length (html-files site)) (length problems))
  (for-each (lambda (text)
              (format (current-error-port) "  ~a~%" text))
            (reverse problems))

  (test-equal "every link is relative, internal and sound"
    0 (length problems))

  (let ((structure-problems
         (append-map check-page-structure (html-files site))))
    (format (current-error-port) "html structure problems: ~a~%"
            (length structure-problems))
    (for-each (lambda (text)
                (format (current-error-port) "  ~a~%" text))
              structure-problems)
    (test-equal "every page carries only linked styles"
      0 (length structure-problems)))

  (for-each
   (lambda (name)
     (let ((path (string-append site "/" name)))
       (test-assert (format #f "~a parses as JSON" name)
         (and (file-exists? path)
              (catch 'json-error
                (lambda () (json-check (read-file path)))
                (lambda (key position)
                  (format (current-error-port)
                          "  ~a: JSON error at byte ~a~%" name position)
                  #f))))))
   '("search-index.json" "pytorch-map.json"))

  (test-assert "index.html carries the redirect script"
    (let ((path (string-append site "/index.html")))
      (and (file-exists? path)
           (string-contains (read-file path)
                            "location.replace(location.pathname + '/' +")))))

(delete-tree site-output)

(test-end "docs-site")

(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
