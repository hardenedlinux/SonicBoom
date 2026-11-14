;;; The documentation extractor for the Guile binding.
;;;
;;; This program reads the sources as text (it never loads the modules) and
;;; turns every exported symbol of every module under bindings/guile/scheme/
;;; into structured data: its kind, its signature, its documentation (a
;;; docstring for procedures, a comment block above the definition for macros,
;;; record types and variables), and its source location.
;;;
;;; The documentation format is described in CONVENTIONS.md next to this file.
;;;
;;; Exits:
;;;   (module-docs)                    all modules, as an association list
;;;   (site-modules)                   the modules not excluded by public-modules.scm
;;;   (excluded-modules)               the module names excluded from the site
;;;   guile extract.scm --tsv          one line per exported symbol:
;;;                                    module symbol kind doc pytorch file:line
;;;   guile extract.scm --scheme-dir DIR   read another source tree
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>

(use-modules (ice-9 textual-ports)
             (ice-9 format)
             (ice-9 ftw)
             ((rnrs io ports) #:select (port-position open-bytevector-input-port
                                       get-bytevector-n))
             (rnrs bytevectors)
             (srfi srfi-1)
             (srfi srfi-11)
             (srfi srfi-13))

;;; ---------------------------------------------------------------- paths ----

(define (script-directory)
  (let ((file (current-filename)))
    (if file
        (dirname (canonicalize-path file))
        (getcwd))))

(define (repo-root)
  (canonicalize-path (string-append (script-directory) "/../..")))

(define (default-scheme-directory)
  (string-append (repo-root) "/bindings/guile/scheme"))

(define *scheme-directory* (default-scheme-directory))

(define *exclusion-file* (string-append (script-directory) "/public-modules.scm"))

(define (set-scheme-directory! dir)
  (set! *scheme-directory* (canonicalize-path dir)))

;;; --------------------------------------------------------------- reader ----

(define (offset->line text offset)
  (1+ (string-count (substring text 0 offset) #\newline)))

;; port-position counts bytes; strings are indexed by characters.
(define (byte->character-offset text byte-offset)
  (let* ((bytes (string->utf8 text))
         (limit (min byte-offset (bytevector-length bytes)))
         (port (open-bytevector-input-port bytes)))
    (string-length (utf8->string (get-bytevector-n port limit)))))

(define (skip-line text i)
  (let loop ((i i))
    (cond ((>= i (string-length text)) i)
          ((char=? (string-ref text i) #\newline) (+ i 1))
          (else (loop (+ i 1))))))

;; The position of the next datum. The reader skips the same whitespace and
;; line comments, so the first significant character is where the datum begins.
(define (skip-space-and-comments text i)
  (let loop ((i i))
    (cond ((>= i (string-length text)) i)
          ((char-whitespace? (string-ref text i)) (loop (+ i 1)))
          ((char=? (string-ref text i) #\;) (loop (skip-line text i)))
          (else i))))

;; Returns three values: the text, its lines, and a list of
;; (start-line form), in source order.
(define (read-source path)
  (let* ((text (call-with-input-file path get-string-all))
         (lines (string-split text #\newline))
         (port (open-input-file path)))
    (let loop ((previous 0) (forms '()))
      (let* ((start (skip-space-and-comments text previous))
             (line (offset->line text start))
             (form (read port))
             (end (byte->character-offset text (port-position port))))
        (if (eof-object? form)
            (values text lines (reverse forms))
            (loop end (cons (list line form) forms)))))))

(define (scm-files directory)
  (let ((out '()))
    (define (walk dir)
      (for-each
       (lambda (entry)
         (let ((path (string-append dir "/" entry)))
           (cond ((member entry '("." "..") string=?) #t)
                 ((eq? (stat:type (stat path)) 'directory) (walk path))
                 ((string-suffix? ".scm" entry) (set! out (cons path out))))))
       (scandir dir)))
    (walk directory)
    (sort out string<?)))

;;; ---------------------------------------------------------- comment text ----

;; The documentation of a non-procedure is the block of ';' comment lines
;; immediately above its definition.
(define (comment-block-above lines start-line)
  (let loop ((i (- start-line 2)) (acc '()))
    (if (< i 0)
        (and (pair? acc) (string-join acc "\n"))
        (let ((line (list-ref lines i)))
          (if (string-prefix? ";" (trim-left line))
              (loop (- i 1) (cons (strip-comment-prefix line) acc))
              (and (pair? acc) (string-join acc "\n")))))))

(define (trim-left string)
  (let loop ((i 0))
    (if (and (< i (string-length string))
             (char-whitespace? (string-ref string i)))
        (loop (+ i 1))
        (substring string i (string-length string)))))

(define (strip-comment-prefix line)
  (let loop ((i 0))
    (cond ((>= i (string-length line)) "")
          ((char=? (string-ref line i) #\;) (loop (+ i 1)))
          (else
           (let ((rest (substring line i (string-length line))))
             (if (and (> (string-length rest) 0)
                      (char=? (string-ref rest 0) #\space))
                 (substring rest 1 (string-length rest))
                 rest))))))

;; Module descriptions: the leading ';;;' block, without SPDX and copyright.
(define (header-block lines)
  (let loop ((ls lines) (acc '()))
    (cond ((null? ls) (reverse acc))
          ((string-prefix? ";;;" (car ls))
           (loop (cdr ls) (cons (substring (car ls) 3 (string-length (car ls))) acc)))
          (else (reverse acc)))))

(define (module-header lines)
  (let* ((block (header-block lines))
         (kept (remove (lambda (line)
                         (or (string-contains line "SPDX-License-Identifier")
                             (string-contains line "Copyright")))
                       block)))
    (drop-while (lambda (line) (string=? (string-trim line) "")) kept)))

(define (module-summary block)
  (let ((first (find (lambda (line) (not (string=? (string-trim line) ""))) block)))
    (and first (string-trim first))))

(define (module-description block)
  (let* ((rest (drop-while (lambda (line) (string=? (string-trim line) "")) block))
         (rest (if (pair? rest) (cdr rest) '())))
    (let ((text (string-trim (string-join (map string-trim-right rest) "\n"))))
      (and (not (string=? text "")) text))))

;;; ------------------------------------------------------------ doc parsing --

(define label-prefixes '("Args:" "Returns:" "Example:" "PyTorch:" "Note:"))

(define (label-line? line)
  (and (> (string-length line) 0)
       (not (char=? (string-ref line 0) #\space))
       (any (lambda (label) (string-prefix? label line)) label-prefixes)))

(define (lines-after label lines)
  ;; The label's value may start on the label line itself ("Returns: a tensor")
  ;; and continue on the lines that follow it.
  (let loop ((ls lines) (started #f) (acc '()))
    (cond ((null? ls) (reverse acc))
          ((not started)
           (if (and (label-line? (car ls)) (string-prefix? label (car ls)))
               (let ((inline (string-trim
                              (substring (car ls) (string-length label)
                                         (string-length (car ls))))))
                 (loop (cdr ls) #t (if (string=? inline "") acc (list inline))))
               (loop (cdr ls) #f acc)))
          ((label-line? (car ls)) (reverse acc))
          (else (loop (cdr ls) #t (cons (car ls) acc))))))

(define (parse-text-after label lines)
  (let ((text (string-trim (string-join (map string-trim (lines-after label lines)) "\n"))))
    (and (not (string=? text "")) text)))

(define (parse-args lines)
  (let loop ((ls (lines-after "Args:" lines)) (acc '()))
    (cond ((null? ls) (reverse acc))
          ((string=? (string-trim (car ls)) "") (loop (cdr ls) acc))
          (else
           (let* ((line (string-trim (car ls)))
                  (colon (string-index line #\:)))
             (if colon
                 (loop (cdr ls)
                       (cons (cons (string-trim (substring line 0 colon))
                                   (string-trim (substring line (+ colon 1)
                                                           (string-length line))))
                             acc))
                 (loop (cdr ls) acc)))))))

(define (parse-example lines)
  (let loop ((ls lines) (started #f) (code '()) (expected #f))
    (cond ((null? ls)
           (and started (list (cons 'code (reverse code)) (cons 'expected expected))))
          ((not started)
           (if (and (label-line? (car ls)) (string-prefix? "Example:" (car ls)))
               (loop (cdr ls) #t code expected)
               (loop (cdr ls) #f code expected)))
          ((label-line? (car ls))
           (list (cons 'code (reverse code)) (cons 'expected expected)))
          ((string=? (string-trim (car ls)) "")
           (list (cons 'code (reverse code)) (cons 'expected expected)))
          (else
           (let ((line (car ls)))
             (cond ((string-prefix? "  " line)
                    (let ((stripped (substring line 2 (string-length line))))
                      (if (string-prefix? "=> " stripped)
                          (loop (cdr ls) #t code
                                (substring stripped 3 (string-length stripped)))
                          (loop (cdr ls) #t (cons stripped code) expected))))
                   (else
                    (loop (cdr ls) #t (cons line code) expected))))))))

(define (parse-pytorch lines)
  (let ((text (parse-text-after "PyTorch:" lines)))
    (and text
         (let ((names (map string-trim (string-split text #\,))))
           (if (and (= (length names) 1) (string=? (car names) "none"))
               'none
               names)))))

(define (parse-note lines)
  (let ((text (string-trim (string-join (map string-trim-right (lines-after "Note:" lines))
                                        "\n"))))
    (and (not (string=? text "")) text)))

(define (parse-doc raw)
  (if (or (not raw) (string=? (string-trim raw) ""))
      #f
      (let* ((lines (map string-trim-right (string-split raw #\newline)))
             (before (take-while (lambda (line) (not (label-line? line))) lines))
             (first (find (lambda (line) (not (string=? (string-trim line) ""))) before))
             (after-summary
              (if first
                  (cdr (drop-while (lambda (line) (not (eq? line first))) before))
                  '()))
             (description (string-trim (string-join after-summary "\n"))))
        (list (cons 'summary (and first (string-trim first)))
              (cons 'description (and (not (string=? description "")) description))
              (cons 'args (parse-args lines))
              (cons 'returns (parse-text-after "Returns:" lines))
              (cons 'example (parse-example lines))
              (cons 'pytorch (parse-pytorch lines))
              (cons 'note (parse-note lines))
              (cons 'raw raw)))))

;;; ------------------------------------------------------------- definitions --

(define (lambda-form? form)
  (and (pair? form) (memq (car form) '(lambda lambda*))))

(define (record-definition? form)
  (and (pair? form) (memq (car form) '(define-record-type define-condition-type))))

(define (record-type-name form)
  (cadr form))

(define (record-generated-names form)
  (cond
   ((eq? (car form) 'define-record-type)
    (let* ((type-name (cadr form))
           (constructor (caddr form))
           (predicate (cadddr form))
           (fields (cddddr form))
           (ctor-name (and (pair? constructor) (car constructor)))
           (accessors (append-map (lambda (field)
                                    (if (pair? field) (cdr field) '()))
                                  fields)))
      (filter symbol? (cons type-name (cons ctor-name (cons predicate accessors))))))
   ((eq? (car form) 'define-condition-type)
    ;; (define-condition-type type parent predicate (field accessor) ...)
    (let* ((type-name (cadr form))
           (predicate (cadddr form))
           (fields (cddddr form))
           (accessors (map cadr fields)))
      (filter symbol? (cons type-name (cons predicate accessors)))))
   (else '())))

;; Returns (name kind args body) for a definition, or #f.
(define (parse-definition form)
  (and (pair? form)
       (case (car form)
         ((define define*)
          (let ((target (cadr form)))
            (cond ((pair? target)
                   (list (car target) 'procedure (cdr target) (cddr form)))
                  (else
                   (let ((value (and (pair? (cddr form)) (caddr form))))
                     (if (lambda-form? value)
                         (list target 'procedure (cadr value) (cddr value))
                         (list target 'variable #f (cddr form))))))))
         ((define-syntax)
          (list (cadr form) 'macro #f (cddr form)))
         ((define-inlinable)
          ;; (define-inlinable (name args) body): a call form, documented by the
          ;; comment block above it (it cannot carry a docstring).
          (list (car (cadr form)) 'inlinable (cdr (cadr form)) (cddr form)))
         (else #f))))

(define (docstring-of body)
  (and (pair? body) (string? (car body)) (car body)))

(define (render-signature name args)
  (format #f "(~a~a)" name
          (if (null? args) "" (string-append " " (join-args args)))))

;; Render an argument list, including a dotted rest argument ("a . rest").
(define (join-args args)
  (cond ((null? args) "")
        ((pair? args)
         (string-append (render (car args))
                        (let ((rest (cdr args)))
                          (cond ((null? rest) "")
                                ((pair? rest) (string-append " " (join-args rest)))
                                (else (string-append " . " (render rest)))))))
        (else (string-append ". " (render args)))))

(define (join-datums datums separator)
  (if (null? datums)
      ""
      (fold (lambda (datum acc) (string-append acc separator (render datum)))
            (render (car datums)) (cdr datums))))

(define (render datum)
  (call-with-output-string (lambda (port) (write datum port))))

;;; ---------------------------------------------------------------- modules ---

(define (keyword-value form key)
  (let loop ((xs (cddr form)))
    (cond ((null? xs) '())
          ((eq? (car xs) key)
           (let ((value (and (pair? (cdr xs)) (cadr xs))))
             (cond ((symbol? value) (list value))
                   ((pair? value) (filter symbol? value))
                   (else '()))))
          ((keyword? (car xs)) (loop (cddr xs)))
          (else (loop (cdr xs))))))

(define (module-name-string name)
  (string-join (map symbol->string name) "/"))

(define (build-export sym kind signature doc line record target)
  (list (cons 'name (symbol->string sym))
        (cons 'kind kind)
        (cons 'signature signature)
        (cons 'doc doc)
        (cons 'line line)
        (cons 'record record)
        (cons 'target target)))

(define (load-module path)
  (let-values (((text lines forms) (read-source path)))
    (let ((module-form (and (pair? forms) (cadr (car forms)))))
      (and (pair? module-form)
           (eq? (car module-form) 'define-module)
           (let* ((module-name (cadr module-form))
                  (name-string (module-name-string module-name))
                  (header (module-header lines))
                  (exports (delete-duplicates
                            (append (keyword-value module-form #:export)
                                    (keyword-value module-form #:replace))
                            eq?))
                  (reexports (delete-duplicates
                              (keyword-value module-form #:re-export) eq?))
                  (definitions (make-hash-table))
                  (records (make-hash-table)))
             (for-each
              (lambda (entry)
                (let ((line (car entry)) (form (cadr entry)))
                  (cond
                   ((record-definition? form)
                    (for-each (lambda (sym)
                                (hashq-set! records sym (cons line form)))
                              (record-generated-names form)))
                   ((parse-definition form)
                    (hashq-set! definitions (car (parse-definition form))
                                (cons line (parse-definition form)))))))
              forms)
             (let ((export-records
                    (append
                     (map (lambda (sym)
                            (let ((definition (hashq-ref definitions sym))
                                  (record (hashq-ref records sym)))
                              (cond
                               (definition
                                (let* ((line (car definition))
                                       (parsed (cdr definition))
                                       (kind (cadr parsed))
                                       (args (caddr parsed))
                                       (body (cadddr parsed))
                                       (doc (if (eq? kind 'procedure)
                                                (docstring-of body)
                                                (comment-block-above lines line))))
                                  (build-export sym kind
                                                (and (memq kind '(procedure inlinable))
                                                     (render-signature sym args))
                                                (parse-doc doc) line #f #f)))
                               (record
                                (let* ((line (car record))
                                       (type-name (record-type-name (cdr record))))
                                  (build-export sym 'record-type #f
                                                (parse-doc (comment-block-above lines line))
                                                line type-name #f)))
                               (else
                                (build-export sym 'missing #f #f #f #f #f)))))
                          exports)
                     (map (lambda (sym)
                            (build-export sym 'reexport #f #f #f #f 'unknown))
                          reexports))))
               (list (cons 'name name-string)
                     (cons 'module module-name)
                     (cons 'path path)
                     (cons 'header (string-trim (string-join header "\n")))
                     (cons 'summary (module-summary header))
                     (cons 'description (module-description header))
                     (cons 'exports export-records))))))))

;; Re-exported symbols get the documentation of the module that defines them.
(define (resolve-reexports! modules)
  (for-each
   (lambda (module)
     (for-each
      (lambda (export)
        (when (eq? (assq-ref export 'kind) 'reexport)
          (let ((origin
                 (find (lambda (other)
                         (and (not (eq? other module))
                              (find (lambda (candidate)
                                      (and (string=? (assq-ref candidate 'name)
                                                     (assq-ref export 'name))
                                           (not (memq (assq-ref candidate 'kind)
                                                      '(reexport missing)))))
                                    (assq-ref other 'exports))))
                       modules)))
            (when origin
              (let ((source (find (lambda (candidate)
                                    (string=? (assq-ref candidate 'name)
                                              (assq-ref export 'name)))
                                  (assq-ref origin 'exports))))
                (set-cdr! (assq 'doc export) (assq-ref source 'doc))
                (set-cdr! (assq 'signature export) (assq-ref source 'signature))
                (set-cdr! (assq 'target export)
                          (list (assq-ref origin 'name) (assq-ref export 'name))))))))
      (assq-ref module 'exports)))
   modules))

(define (module-docs)
  (let ((modules (filter identity (map load-module (scm-files *scheme-directory*)))))
    (resolve-reexports! modules)
    modules))

;;; ------------------------------------------------------------- exclusions ---

(define (excluded-modules)
  (if (not (file-exists? *exclusion-file*))
      '()
      (let-values (((text lines forms) (read-source *exclusion-file*)))
        (let ((form (find (lambda (entry)
                            (and (pair? (cadr entry))
                                 (eq? (car (cadr entry)) 'define)
                                 (eq? (cadr (cadr entry)) 'public-modules)))
                          forms)))
          (if form
              (map (lambda (entry)
                     (cond ((string? entry) entry)
                           ((list? entry) (module-name-string entry))
                           ((symbol? entry) (symbol->string entry))
                           (else (render entry))))
                   (caddr (cadr form)))
              '())))))

(define (site-modules)
  (let ((excluded (excluded-modules)))
    (filter (lambda (module) (not (member (assq-ref module 'name) excluded)))
            (module-docs))))

;;; -------------------------------------------------------------------- cli ---

(define (has-arg? arg)
  (member arg (command-line) string=?))

(define (arg-value name)
  (let loop ((args (command-line)))
    (cond ((null? args) #f)
          ((string=? (car args) name) (and (pair? (cdr args)) (cadr args)))
          (else (loop (cdr args))))))

(define (print-tsv modules)
  (for-each
   (lambda (module)
     (for-each
      (lambda (export)
        (let* ((doc (assq-ref export 'doc))
               (pytorch (and doc (assq-ref doc 'pytorch)))
               (line (assq-ref export 'line)))
          (format #t "~a\t~a\t~a\t~a\t~a\t~a:~a\n"
                  (assq-ref module 'name)
                  (assq-ref export 'name)
                  (assq-ref export 'kind)
                  (if doc "yes" "no")
                  (if pytorch "yes" "no")
                  (assq-ref module 'path)
                  (or line 0))))
      (assq-ref module 'exports)))
   modules))

(when (has-arg? "--tsv")
  (let ((dir (arg-value "--scheme-dir")))
    (when dir (set-scheme-directory! dir))
    (print-tsv (module-docs)))
  (exit 0))
