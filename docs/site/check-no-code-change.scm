;;; Check that documenting the Guile binding changed no code.
;;;
;;;   guile docs/site/check-no-code-change.scm [--base REV]
;;;
;;; Reads every Scheme source in the current tree and the same file at REV
;;; (the start of the documentation work by default), removes the docstring of
;;; each definition from both, and requires the remaining forms to be equal one
;;; by one. Comments are not part of a read form, so changed or added
;;; documentation blocks never enter the comparison; a changed body, argument
;;; list, export list or module dependency does.
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>

(use-modules (ice-9 textual-ports)
             (ice-9 popen)
             (srfi srfi-1))

(load (string-append (dirname (current-filename)) "/extract.scm"))

(define default-base "b5c0220a55dcfdb5a85c6a9c01d473eb160fe4f2")

(define (argument name fallback)
  (let ((args (cdr (command-line))))
    (let loop ((ls args))
      (cond ((null? ls) fallback)
            ((string=? (car ls) name)
             (and (pair? (cdr ls)) (cadr ls)))
            (else (loop (cdr ls)))))))

;; --------------------------------------------------------------- reading ---

(define (shell-quote text)
  (string-append "'" (string-replace-substring text "'" "'\\''") "'"))

(define (string-replace-substring text from to)
  (with-output-to-string
    (lambda ()
      (let loop ((start 0))
        (let ((at (string-contains text from start)))
          (if at
              (begin (display (substring text start at)) (display to)
                     (loop (+ at (string-length from))))
              (display (substring text start (string-length text)))))))))

(define (git-show revision path)
  (let* ((command (string-append
                   "git -C " (shell-quote (repo-root))
                   " show " (shell-quote (string-append revision ":" path))))
         (port (open-input-pipe command))
         (text (get-string-all port)))
    (unless (zero? (close-pipe port))
      (error "cannot read" path "at" revision))
    text))

(define (forms-of-text text)
  (let ((port (open-input-string text)))
    (let loop ((forms '()))
      (let ((form (read port)))
        (if (eof-object? form)
            (reverse forms)
            (loop (cons form forms)))))))

;; ---------------------------------------------------------- doc removal ---

(define (strip-body body)
  (if (and (pair? body) (string? (car body))) (cdr body) body))

(define (strip-lambda form)
  (append (take form 2) (strip-body (drop form 2))))

(define (strip-doc form)
  (if (not (pair? form))
      form
      (case (car form)
        ((define define*)
         (let ((rest (cdr form)))
           (cond ((and (pair? rest) (pair? (car rest)))
                  (cons (car form)
                        (cons (car rest) (strip-body (cdr rest)))))
                 ((and (pair? rest) (pair? (cdr rest))
                       (lambda-form? (cadr rest)))
                  (list (car form) (car rest) (strip-lambda (cadr rest))))
                 (else form))))
        ((lambda lambda*) (strip-lambda form))
        (else form))))

(define (form-name form)
  (and (pair? form)
       (memq (car form) '(define define* define-inlinable define-syntax
                          define-syntax-rule define-record-type
                          define-condition-type))
       (let ((rest (cadr form)))
         (cond ((symbol? rest) rest)
               ((pair? rest) (car rest))
               (else #f)))))

;; ------------------------------------------------------------------ main ---

(define base (or (argument "--base" #f) default-base))
(define scheme-directory (string-append (repo-root) "/bindings/guile/scheme"))
(define prefix-length (1+ (string-length (repo-root))))

(define total 0)
(define differences 0)

(for-each
 (lambda (path)
   (let* ((relative (substring path prefix-length (string-length path)))
          (current (forms-of-text (call-with-input-file path get-string-all)))
          (base-forms (forms-of-text (git-show base relative)))
          (label (string-append relative ": " (number->string (length current))
                                " forms")))
     (set! total (+ total (length current)))
     (cond ((not (= (length current) (length base-forms)))
            (set! differences (+ differences 1))
            (format #t "DIFF ~a: form count ~a -> ~a\n"
                    relative (length base-forms) (length current)))
           (else
            (let loop ((ours current) (theirs base-forms) (index 0))
              (unless (null? ours)
                (let ((name (form-name (car ours))))
                  (unless (equal? (strip-doc (car ours))
                                  (strip-doc (car theirs)))
                    (set! differences (+ differences 1))
                    (format #t "DIFF ~a form ~a ~a\n"
                            relative index (or name 'top-level))))
                (loop (cdr ours) (cdr theirs) (+ index 1))))
            (format #t "ok   ~a\n" label)))))
 (scm-files scheme-directory))

(format #t "~a forms compared, ~a differ\n" total differences)
(exit (if (zero? differences) 0 1))
