;;; SonicBoom — the documentation covers every public symbol
;;;
;;; The extractor in docs/site/extract.scm reads the Scheme sources as text and
;;; reports, for every module and every exported symbol, whether there is a
;;; docstring or a comment block and what it says. This test holds that output
;;; to the conventions of docs/site/CONVENTIONS.md:
;;;
;;;   1. every module not excluded by docs/site/public-modules.scm has a header
;;;      description after its title line;
;;;   2. every export has documentation, and its first line is one sentence;
;;;   3. every procedure has a PyTorch: tag (none counts as a tag);
;;;   4. no export is of kind "missing", i.e. every exported name is defined
;;;      where the module says it is.
;;;
;;; A known-missing list exists so that a symbol that was already missing at
;;; the start of the work does not fail the suite; it is empty now.
;;;
;;; Pure Scheme: no native library, no files beyond the sources.

(use-modules (ice-9 format)
             (srfi srfi-1)
             (srfi srfi-13)
             (srfi srfi-64))

;; bindings/guile/tests/docs/<this file> -> the repository root.
(define (repo-root)
  (let ((file (current-filename)))
    (canonicalize-path
     (dirname (dirname (dirname (dirname (dirname file))))))))

(load (string-append (repo-root) "/docs/site/extract.scm"))

;; Exports that were already missing when the coverage work began. There are
;; none; a name is added here only with an entry in the report.
(define known-missing '())

(define modules (site-modules))
(define exports (append-map (lambda (module) (assq-ref module 'exports))
                            modules))

(define (module-name module) (assq-ref module 'name))
(define (export-name export) (assq-ref export 'name))
(define (export-kind export) (assq-ref export 'kind))
(define (export-doc export) (assq-ref export 'doc))

(define (module-of export)
  (find (lambda (module)
          (memq export (assq-ref module 'exports)))
        modules))

(test-begin "docs-coverage")
(define %runner (test-runner-current))

(test-assert "the binding has modules and exports"
  (and (pair? modules) (pair? exports)))

;;; 1. Module headers.

(define modules-without-header
  (filter-map (lambda (module)
                (let ((description (assq-ref module 'description)))
                  (and (or (not description)
                           (string=? (string-trim description) ""))
                       (module-name module))))
              modules))

(format (current-error-port) "modules: ~a, without a header description: ~a~%"
        (length modules) (length modules-without-header))
(for-each (lambda (name)
            (format (current-error-port) "  no header description: ~a~%" name))
          modules-without-header)

(test-equal "every module has a header description"
  0 (length modules-without-header))

;;; 2. Every export is documented, and its first line is one sentence.

(define exports-without-doc
  (filter (lambda (export) (not (export-doc export))) exports))

(define summaries-that-are-not-sentences
  (filter-map (lambda (export)
                (let* ((doc (export-doc export))
                       (summary (and doc (assq-ref doc 'summary))))
                  (and (not (and summary (string-suffix? "." summary)))
                       (list (module-name (module-of export)) (export-name export)
                              summary))))
              exports))

(format (current-error-port)
        "exports: ~a, without documentation: ~a, first line not a sentence: ~a~%"
        (length exports) (length exports-without-doc)
        (length summaries-that-are-not-sentences))
(for-each (lambda (export)
            (format (current-error-port) "  no documentation: ~a ~a~%"
                    (module-name (module-of export)) (export-name export)))
          exports-without-doc)
(for-each (lambda (row)
            (format (current-error-port) "  first line not a sentence: ~a ~a: ~s~%"
                    (car row) (cadr row) (caddr row)))
          summaries-that-are-not-sentences)

(test-equal "every export has documentation" 0 (length exports-without-doc))
(test-equal "every export's first line is one sentence"
  0 (length summaries-that-are-not-sentences))

;;; 3. Every procedure has a PyTorch: tag.

(define procedures-without-tag
  (filter-map (lambda (export)
                (let ((doc (export-doc export)))
                  (and (memq (export-kind export) '(procedure inlinable))
                       (not (and doc (assq-ref doc 'pytorch)))
                       (list (module-name (module-of export))
                             (export-name export)))))
              exports))

(format (current-error-port) "procedures without a PyTorch: tag: ~a~%"
        (length procedures-without-tag))
(for-each (lambda (row)
            (format (current-error-port) "  no PyTorch: tag: ~a ~a~%"
                    (car row) (cadr row)))
          procedures-without-tag)

(test-equal "every procedure has a PyTorch: tag"
  0 (length procedures-without-tag))

;;; 4. Every exported name is defined.

(define missing
  (filter-map (lambda (export)
                (and (eq? (export-kind export) 'missing)
                     (list (module-name (module-of export))
                           (export-name export))))
              exports))

(define unexpected-missing
  (remove (lambda (row) (member (cadr row) known-missing))
          missing))

(format (current-error-port)
        "exports of kind missing: ~a (known: ~a)~%" (length missing)
        (length known-missing))
(for-each (lambda (row)
            (format (current-error-port) "  missing: ~a ~a~%" (car row) (cadr row)))
          unexpected-missing)

(test-equal "no export is of kind missing" 0 (length unexpected-missing))

(test-end "docs-coverage")

(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
