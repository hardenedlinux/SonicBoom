;;; SonicBoom — the PyTorch cross-reference is complete and consistent
;;;
;;; docs/site/pytorch-map.scm says, for PyTorch names, whether the binding has
;;; a counterpart, a recipe or nothing, and which SonicBoom function a
;;; supported name means. This test checks that the file and the docstrings
;;; agree:
;;;
;;;   1. every name of docs/site/pytorch-names.tsv resolves, by its own entry,
;;;      a same-as entry or a prefix entry;
;;;   2. every use of a supported entry names a module that exists and a symbol
;;;      that module exports;
;;;   3. the two directions of the tag match hold: a supported entry's function
;;;      carries the entry's name (or an alias) in its PyTorch: tag, and every
;;;      name in a function's PyTorch: tag has a supported entry;
;;;   4. same-as has no cycle and every target exists;
;;;   5. out-of-scope appears only on the prefixes the specification names, with
;;;      the specified note.
;;;
;;; Pure Scheme: no native library. The recipes themselves are run by
;;; test-docs-examples.

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

(define (read-file path)
  (call-with-input-file path get-string-all))

;;; ---------------------------------------------------------------- helpers ---

(define (entry-ref entry key)
  (assq-ref (cdr entry) key))

(define (entry-field entry key)
  (let ((field (entry-ref entry key)))
    (and field (car field))))

(define (entry-name entry) (cadr entry))

(define (entry-status entry)
  (let ((status (entry-ref entry 'status)))
    (and status (car status))))

(define (name->string name)
  (if (symbol? name) (symbol->string name) name))

(define torch-entries
  (filter (lambda (entry) (eq? (car entry) 'torch)) pytorch-map))

(define prefix-entries
  (filter (lambda (entry) (eq? (car entry) 'prefix)) pytorch-map))

(define (exact-entry name)
  (find (lambda (entry) (string=? (entry-name entry) name)) torch-entries))

(define (longest-prefix name)
  (let loop ((rest prefix-entries) (best #f))
    (cond ((null? rest) best)
          ((string-prefix? (entry-name (car rest)) name)
           (if (and best (>= (string-length (entry-name best))
                             (string-length (entry-name (car rest)))))
               (loop (cdr rest) best)
               (loop (cdr rest) (car rest))))
          (else (loop (cdr rest) best)))))

;; The entry a name resolves to, with same-as followed once (the map's
;; same-as entries point at real entries).
(define (resolve-name name)
  (let* ((exact (exact-entry name))
         (target (and exact (entry-field exact 'same-as))))
    (cond (target (exact-entry target))
          (exact exact)
          (else (longest-prefix name)))))

(define (tsv-names)
  (filter (lambda (line)
            (and (not (string-null? line))
                 (not (string-prefix? "#" line))))
          (string-split
           (read-file (string-append (repo-root) "/docs/site/pytorch-names.tsv"))
           #\newline)))

(define (tsv-name line)
  (car (string-split line #\tab)))

(define modules (site-modules))

(define (module-named name)
  (find (lambda (module) (string=? (assq-ref module 'name) name)) modules))

(define (module-exports module)
  (and module
       (map (lambda (export) (name->string (assq-ref export 'name)))
            (assq-ref module 'exports))))

;; (module symbol) -> the PyTorch: tag list of the exporting module's symbol.
(define tag-table (make-hash-table))

(define (fill-tag-table module)
  (for-each
   (lambda (export)
     (let* ((doc (assq-ref export 'doc))
            (pytorch (and doc (assq-ref doc 'pytorch))))
       (when (list? pytorch)
         (hash-set! tag-table
                    (cons (assq-ref module 'name)
                          (name->string (assq-ref export 'name)))
                    pytorch))))
   (assq-ref module 'exports)))

(for-each fill-tag-table modules)

(define supported
  (filter (lambda (entry) (eq? (entry-status entry) 'supported)) torch-entries))

;;; --------------------------------------------------------------- the checks --

(define problems '())

(define (problem! text)
  (set! problems (cons text problems)))

;; 1. Every name in the list resolves.
(let ((unresolved
       (filter (lambda (name) (not (resolve-name name)))
               (map tsv-name (tsv-names)))))
  (for-each (lambda (name)
              (problem! (format #f "unresolved name ~a" name)))
            unresolved))

;; 2. Every use exists and is exported.
(for-each
 (lambda (entry)
   (for-each
    (lambda (use)
      (let ((module-name (string-join (map symbol->string (car use)) "/"))
            (symbol (name->string (cadr use))))
        (let ((exports (module-exports (module-named module-name))))
          (cond ((not exports)
                 (problem! (format #f "~a uses the unknown module ~a"
                                   (entry-name entry) module-name)))
                ((not (member symbol exports))
                 (problem! (format #f "~a uses ~a, which ~a does not export"
                                   (entry-name entry) symbol module-name)))))))
    (or (entry-ref entry 'use) '())))
 torch-entries)

;; 3a. A supported entry's function carries the entry's name in its tag.
(for-each
 (lambda (entry)
   (let ((aliases (or (entry-ref entry 'aliases) '())))
     (for-each
      (lambda (use)
        (let* ((module-name (string-join (map symbol->string (car use)) "/"))
               (symbol (name->string (cadr use)))
               (tags (hash-ref tag-table (cons module-name symbol))))
          (unless (and tags
                       (or (member (entry-name entry) tags string=?)
                           (any (lambda (alias) (member alias tags string=?))
                                aliases)))
            (problem! (format #f "~a is supported by ~a ~a, whose tag does not name it"
                              (entry-name entry) module-name symbol)))))
      (or (entry-ref entry 'use) '()))))
 supported)

;; 3b. Every name in a function's tag has a supported entry.
(hash-for-each
 (lambda (where tags)
   (for-each
    (lambda (tag)
      (unless (any (lambda (entry)
                     (or (string=? (entry-name entry) tag)
                         (let ((aliases (entry-ref entry 'aliases)))
                           (and aliases (member tag aliases string=?)))))
                   supported)
        (problem! (format #f "the tag ~a of ~a ~a has no supported entry"
                          tag (car where) (cdr where)))))
    tags))
 tag-table)

;; 3c. A function with the tag none is not the function of a supported entry.
;; (The tag table holds only name lists; a supported entry must have one, which
;; 3a already reports when the tag is missing or none.)

;; 4. same-as: target exists, no cycle.
(let ((same-as (filter (lambda (entry) (entry-field entry 'same-as))
                       torch-entries)))
  (for-each
   (lambda (entry)
     (let ((target (entry-field entry 'same-as)))
       (unless (exact-entry target)
         (problem! (format #f "~a is the same as the unknown ~a"
                           (entry-name entry) target)))
       (let loop ((name (entry-name entry)) (seen '()))
         (let ((next (let ((entry (exact-entry name)))
                       (and entry (entry-field entry 'same-as)))))
           (cond ((not next) #t)
                 ((member next seen)
                  (problem! (format #f "same-as cycle through ~a" next)))
                 (else (loop next (cons name seen))))))))
   same-as))

;; 5. out-of-scope only on the specified prefixes, with the specified note.
(define out-of-scope-notes
  '(("torch.distributed." "SonicBoom targets single-node execution.")
    ("torch.jit." "SonicBoom has its own tracing and compilation path.")
    ("torch.fx." "SonicBoom has its own graph representation.")
    ("torch.onnx." "Import from ONNX is a separate tool; see the importer guide.")
    ("torch.sparse." "Sparse tensors are not part of the current scope.")
    ("torch.quantization." "Quantized weights use SonicBoom's own formats; see the weight format page.")
    ("torch.ao." "Quantized weights use SonicBoom's own formats; see the weight format page.")
    ("torch.distributions." "Probability distributions are not part of the current scope.")
    ("torch.package." "Not applicable.")
    ("torch.hub." "Not applicable.")
    ("torch.utils.cpp_extension." "Not applicable.")
    ("torch.utils.tensorboard." "Not applicable.")
    ("torch.utils.bottleneck." "Not applicable.")
    ("torch.utils.benchmark." "Not applicable.")
    ("torch.utils.mobile_optimizer." "Not applicable.")
    ("torch.backends." "Backend selection is handled by SonicBoom's runtime.")
    ("torch.mps." "Backend selection is handled by SonicBoom's runtime.")
    ("torch.xpu." "Backend selection is handled by SonicBoom's runtime.")
    ("torch.mtia." "Backend selection is handled by SonicBoom's runtime.")))

(define declared-out-of-scope
  (filter (lambda (entry) (eq? (entry-status entry) 'out-of-scope))
          pytorch-map))

(for-each
 (lambda (entry)
   (unless (eq? (car entry) 'prefix)
     (problem! (format #f "out-of-scope on the single name ~a"
                       (entry-name entry))))
   (let ((wanted (assoc (entry-name entry) out-of-scope-notes)))
     (cond ((not wanted)
            (problem! (format #f "out-of-scope prefix ~a is not in the list"
                              (entry-name entry))))
           ((not (string=? (string-trim (or (entry-field entry 'note) ""))
                           (cadr wanted)))
            (problem! (format #f "out-of-scope prefix ~a has a different note"
                              (entry-name entry)))))))
 declared-out-of-scope)

(for-each
 (lambda (wanted)
   (unless (find (lambda (entry)
                   (string=? (entry-name entry) (car wanted)))
                 declared-out-of-scope)
     (problem! (format #f "the specified prefix ~a is not marked out-of-scope"
                       (car wanted)))))
 out-of-scope-notes)

;; A recipe has its code and setup; a supported entry has a use. A same-as
;; entry carries no status of its own: it inherits its target's.
(for-each
 (lambda (entry)
   (unless (entry-field entry 'same-as)
     (case (entry-status entry)
       ((recipe missing out-of-scope supported) #t)
       (else (problem! (format #f "~a has the unknown status ~a"
                               (entry-name entry) (entry-status entry)))))))
 torch-entries)

(for-each
 (lambda (entry)
   (case (entry-status entry)
     ((recipe)
      (unless (entry-field entry 'recipe)
        (problem! (format #f "recipe ~a has no recipe" (entry-name entry))))
      (unless (entry-field entry 'setup)
        (problem! (format #f "recipe ~a has no setup" (entry-name entry)))))
     ((supported)
      (unless (entry-ref entry 'use)
        (problem! (format #f "supported ~a has no use" (entry-name entry)))))))
 torch-entries)

;;; ------------------------------------------------------------------ tests ---

(test-begin "docs-map")
(define %runner (test-runner-current))

(format (current-error-port)
        "names: ~a, entries: ~a, problems: ~a~%"
        (length (tsv-names)) (length pytorch-map) (length problems))
(for-each (lambda (text)
            (format (current-error-port) "  ~a~%" text))
          (reverse problems))

(test-equal "the map has no problems" 0 (length problems))

(test-end "docs-map")

(exit (if (zero? (test-runner-fail-count %runner)) 0 1))
