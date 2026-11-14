;;; The static site generator for the SonicBoom documentation.
;;;
;;;   guile docs/site/build.scm --out DIR
;;;
;;; writes the complete site to DIR/sonicboom/. The pages come from three
;;; places: the hand-written pages in docs/site/pages/, the articles in docs/
;;; (read in place, never modified), and the API pages generated from the Guile
;;; modules' documentation by extract.scm.
;;;
;;; Every page is converted by pandoc with docs/site/template.html and links
;;; only to files inside the site, with relative paths. No external resources.
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>

(use-modules (ice-9 textual-ports)
             (ice-9 format)
             (ice-9 regex)
             (ice-9 ftw)
             (srfi srfi-1)
             (srfi srfi-13))

(define tools-directory
  (let ((file (current-filename)))
    (if file
        (dirname (canonicalize-path file))
        (getcwd))))

(load (string-append tools-directory "/extract.scm"))
(load (string-append tools-directory "/nav.scm"))

(define (repository-root)
  (canonicalize-path (string-append tools-directory "/../..")))

(define (pages-source-directory)
  (string-append tools-directory "/pages"))

(define (docs-source-directory)
  (string-append (repository-root) "/docs"))

(define (template-file)
  (string-append tools-directory "/template.html"))

;;; ------------------------------------------------------------------ files ---

(define (ensure-directory path)
  (unless (file-exists? path)
    (ensure-directory (dirname path))
    (mkdir path)))

(define (read-file path)
  (call-with-input-file path get-string-all))

(define (write-file path text)
  (call-with-output-file path
    (lambda (port) (display text port))))

(define (copy-file source target)
  (write-file target (read-file source)))

(define temp-file-counter 0)

(define (temporary-file suffix)
  (set! temp-file-counter (+ temp-file-counter 1))
  (string-append (or (getenv "TMPDIR") "/tmp")
                 "/sb-docs-site-" (number->string (getpid))
                 "-" (number->string temp-file-counter) suffix))

;;; ------------------------------------------------------------------ html ----

(define (html-escape text)
  (with-output-to-string
    (lambda ()
      (string-for-each
       (lambda (ch)
         (case ch
           ((#\&) (display "&amp;"))
           ((#\<) (display "&lt;"))
           ((#\>) (display "&gt;"))
           ((#\") (display "&quot;"))
           (else (display ch))))
       text))))

(define (attribute-escape text)
  (html-escape text))

;; The anchor of an exported symbol on its module page. The same function is
;; used for the links the PyTorch lookup generates.
(define (symbol-anchor name)
  (string-append
   "sym-"
   (list->string
    (map (lambda (ch)
           (if (or (char-alphabetic? ch) (char-numeric? ch)
                   (memv ch '(#\- #\_ #\.)))
               ch
               #\-))
         (string->list name)))))

(define (root-prefix relative-path)
  (let ((depth (string-count relative-path #\/)))
    (if (= depth 0)
        ""
        (string-append (string-join (make-list depth "..") "/") "/"))))

;;; ---------------------------------------------------------------- pandoc ----

(define (run-pandoc input output title root nav pytorch)
  (let ((arguments
         (append
          (list "pandoc"
                "--from" "markdown"
                "--to" "html5"
                "--standalone"
                "--template" (template-file)
                "--toc"
                "--toc-depth=3"
                "--highlight-style=tango"
                "--wrap=none"
                "--metadata" (string-append "title=" title)
                "--variable" (string-append "page-title=" title)
                "--variable" (string-append "root=" root)
                "--variable" (string-append "nav=" nav)
                "--output" output)
          (if pytorch (list "--variable" "pytorch=true") '())
          (list input))))
    (let ((status (apply system* arguments)))
      (unless (zero? status)
        (error "pandoc failed on" input)))))

;;; ------------------------------------------------------------ navigation ----

(define (page-path slug)
  (if (string=? slug "api/index")
      "api/index.html"
      (string-append slug ".html")))

;; Every page the site contains, as (SECTION TITLE SLUG).
(define (nav-entries)
  (fold-right
   (lambda (section acc)
     (append (map (lambda (page)
                    (list (car section) (cadr page) (car page)))
                  (cdr section))
             acc))
   '()
   site-nav))

(define (documented-articles)
  (let ((known (map (lambda (entry) (caddr entry)) (nav-entries))))
    (sort (filter (lambda (name)
                    (not (member name known)))
                  (map (lambda (name)
                         (string-drop-right name 3))
                       (filter (lambda (name)
                                 (string-suffix? ".md" name))
                               (scandir (docs-source-directory)))))
          string<?)))

(define (concepts-tail-entries)
  (map (lambda (slug) (list "Concepts" (default-title slug) slug))
       (documented-articles)))

;; A title for an article that the navigation does not name: its first heading.
(define (default-title slug)
  (let* ((source (page-source slug))
         (heading (and source (first-heading source))))
    (if heading heading (string-capitalize slug))))

(define (first-heading path)
  (let ((text (read-file path)))
    (let loop ((lines (string-split text #\newline)))
      (cond ((null? lines) #f)
            ((string-prefix? "# " (car lines))
             (string-trim (substring (car lines) 2 (string-length (car lines)))))
            (else (loop (cdr lines)))))))

(define (page-source slug)
  (let ((own (string-append (pages-source-directory) "/" slug ".md"))
        (article (string-append (docs-source-directory) "/" slug ".md")))
    (cond ((file-exists? own) own)
          ((file-exists? article) article)
          (else #f))))

;; Group consecutive entries of the same section, keeping their order.
(define (group-by-section entries)
  (let loop ((rest entries) (groups '()))
    (if (null? rest)
        (map (lambda (group) (cons (car group) (reverse (cdr group))))
             (reverse groups))
        (let ((entry (car rest)))
          (if (and (pair? groups) (equal? (caar groups) (car entry)))
              (loop (cdr rest)
                    (cons (cons (caar groups) (cons entry (cdar groups)))
                          (cdr groups)))
              (loop (cdr rest) (cons (list (car entry) entry) groups)))))))

(define (nav-html current-slug root)
  (let ((sections (group-by-section
                   (append (nav-entries) (concepts-tail-entries)))))
    (with-output-to-string
      (lambda ()
        (for-each
         (lambda (section)
           (format #t "<div class=\"nav-section\">~a</div>~%<ul>~%"
                   (html-escape (car section)))
           (for-each
            (lambda (entry)
              (let ((title (cadr entry)) (slug (caddr entry)))
                (format #t "<li><a~a href=\"~a~a\">~a</a></li>~%"
                        (if (string=? slug current-slug) " class=\"current\"" "")
                        root (page-path slug) (html-escape title))))
            (cdr section))
           (display "</ul>\n"))
         sections)))))

;;; ------------------------------------------------------- article rewriting --

(define (known-page-slugs)
  (append (map caddr (nav-entries))
          (documented-articles)
          (map (lambda (module)
                 (string-append "api/" (assq-ref module 'name)))
               (site-modules))))

;; Links to the repository README files have no site page of their own; point
;; them at the pages that carry the same material.
(define (rewritten-link path fragment)
  (let ((base (basename path)))
    (cond ((string=? base "README") "index.html")
          ((member base (known-page-slugs))
           (string-append base ".html" (or fragment "")))
          (else #f))))

;; The site is self-contained: a link either points at another site page or it
;; is not a link. Article links become .html links; links to repository files
;; and to other hosts keep their text and lose the target, so that no page
;; refers to anything outside the site.
(define (kept-link? target)
  (or (string-prefix? "#" target)
      (and (not (string-contains target "://"))
           (string-suffix? ".html" target))
      (and (not (string-contains target "://"))
           (string-suffix? ".md" (car (string-split target #\#))))))

(define (link-replacement text target)
  (cond ((not (kept-link? target)) text)
        ((string-contains target "://")
         (format #f "[~a](~a)" text target))
        ((string-prefix? "#" target)
         (format #f "[~a](~a)" text target))
        ((string-suffix? ".html" target)
         (format #f "[~a](~a)" text target))
        (else
         (let* ((parts (string-split target #\#))
                (rewritten
                 (rewritten-link
                  (car parts)
                  (if (pair? (cdr parts))
                      (string-append "#" (cadr parts))
                      ""))))
           (if rewritten
               (format #f "[~a](~a)" text rewritten)
               text)))))

(define (rewrite-line-links line)
  (let ((pattern (make-regexp
                  "\\[([^][]+)\\]\\(([^() \t]+)([ \t]+\"[^\"]*\")?\\)")))
    (let loop ((start 0) (acc ""))
      (let ((match (regexp-exec pattern line start)))
        (if (not match)
            (string-append acc (substring line start))
            (let* ((begin (match:start match 0))
                   (end (match:end match 0))
                   (text (match:substring match 1))
                   (target (match:substring match 2))
                   (replacement (link-replacement text target)))
              (loop end
                    (string-append acc
                                   (substring line start begin)
                                   replacement))))))))

;; Fenced code blocks are copied verbatim; links only occur in prose.
(define (rewrite-markdown-links text)
  (let loop ((lines (string-split text #\newline))
             (fenced #f)
             (acc '()))
    (if (null? lines)
        (string-join (reverse acc) "\n")
        (let ((line (car lines)))
          (if (string-prefix? "```" (string-trim line))
              (loop (cdr lines) (not fenced) (cons line acc))
              (loop (cdr lines) fenced
                    (cons (if fenced line (rewrite-line-links line)) acc)))))))

;;; --------------------------------------------------------------- api pages --

(define (kind-label kind)
  (case kind
    ((procedure) "procedure")
    ((inlinable) "inline procedure")
    ((macro) "macro")
    ((record-type) "record type")
    ((variable) "variable")
    ((reexport) "re-export")
    (else (symbol->string kind))))

(define (doc-field doc key)
  (and doc (assq-ref doc key)))

(define (text-or-nothing text)
  (or text ""))

(define (pytorch-link root name)
  (format #f "[~a](~apytorch.html?q=~a)" name root name))

(define (render-args args)
  (if (null? args)
      ""
      (string-append
       "**Arguments:**\n\n"
       (apply string-append
              (map (lambda (argument)
                     (format #f "- `~a`: ~a\n"
                             (car argument)
                             (text-or-nothing (cdr argument))))
                   args))
       "\n")))

(define (render-example example)
  (if (not example)
      ""
      (let* ((code (string-join (assq-ref example 'code) "\n"))
             (expected (assq-ref example 'expected)))
        (string-append
         "**Example:**\n\n```scheme\n"
         code
         "\n```\n\n"
         (if expected
             (format #f "`=> ~a`\n\n" expected)
             "")))))

(define (render-export root export anchors)
  (let* ((name (assq-ref export 'name))
         (kind (assq-ref export 'kind))
         (signature (assq-ref export 'signature))
         (doc (assq-ref export 'doc))
         (summary (doc-field doc 'summary))
         (description (doc-field doc 'description))
         (args (doc-field doc 'args))
         (returns (doc-field doc 'returns))
         (example (doc-field doc 'example))
         (pytorch (doc-field doc 'pytorch))
         (note (doc-field doc 'note))
         (anchor (or (assq-ref anchors name) (symbol-anchor name))))
    (string-append
     (format #f "## ~a {#~a}\n\n" (html-escape name) anchor)
     (format #f "<span class=\"kind\">~a</span>\n\n" (kind-label kind))
     (if signature
         (format #f "<div class=\"signature\">~a</div>\n\n"
                 (html-escape signature))
         "")
     (if summary (format #f "~a\n\n" summary) "")
     (if description (format #f "~a\n\n" description) "")
     (render-args (or args '()))
     (if returns (format #f "**Returns:** ~a\n\n" returns) "")
     (render-example example)
     (format #f "**PyTorch:** ~a\n\n"
             (cond ((not pytorch) "none")
                   ((eq? pytorch 'none) "none")
                   (else (string-join (map (lambda (one)
                                             (pytorch-link root one))
                                           pytorch)
                                      ", "))))
     (if note (format #f "**Note:** ~a\n\n" note) ""))))

(define (unique-anchors exports)
  (let loop ((rest exports) (seen '()) (acc '()))
    (if (null? rest)
        (reverse acc)
        (let* ((name (assq-ref (car rest) 'name))
               (base (symbol-anchor name))
               (anchor (let try ((candidate base) (n 1))
                         (if (member candidate seen)
                             (try (format #f "~a-~a" base n) (+ n 1))
                             candidate))))
          (loop (cdr rest)
                (cons anchor seen)
                (cons (cons name anchor) acc))))))

(define (module-page-markdown module root)
  (let* ((name (assq-ref module 'name))
         (description (or (assq-ref module 'description)
                          (assq-ref module 'summary)
                          ""))
         (exports (assq-ref module 'exports))
         (anchors (unique-anchors exports)))
    (string-append
     (format #f "# ~a\n\n" name)
     (if (string=? description "")
         ""
         (string-append description "\n\n"))
     (apply string-append
            (map (lambda (export) (render-export root export anchors))
                 exports)))))

(define (module-group module-name)
  (let ((parts (string-split module-name #\/)))
    (if (<= (length parts) 2)
        "Core"
        (let ((group (cadr parts)))
          (cond ((string=? group "data") "Data")
                ((string=? group "serve") "Serve")
                ((string=? group "tokenizer") "Tokenizer")
                ((string=? group "trace") "Trace")
                ((string=? group "train") "Train")
                (else "Other"))))))

(define (module-summary-line module)
  (let ((summary (assq-ref module 'summary))
        (name (assq-ref module 'name)))
    (if (and summary (not (string=? summary name)))
        summary
        (let ((description (assq-ref module 'description)))
          (and description
               (car (string-split description #\newline)))))))

(define (api-index-markdown modules)
  (let* ((groups '("Core" "Data" "Serve" "Tokenizer" "Trace" "Train" "Other"))
         (by-group
          (map (lambda (group)
                 (cons group
                       (filter (lambda (module)
                                 (string=? (module-group (assq-ref module 'name))
                                           group))
                               modules)))
               groups)))
    (string-append
     "# Guile binding\n\n"
     "The Guile binding of SonicBoom. Every module is public; each section below\n"
     "documents the procedures, macros, record types and variables a module\n"
     "exports.\n\n"
     (apply string-append
            (map (lambda (entry)
                   (let ((group (car entry)) (members (cdr entry)))
                     (if (null? members)
                         ""
                         (string-append
                          (format #f "## ~a\n\n" group)
                          (apply string-append
                                 (map (lambda (module)
                                        (format #f "- [~a](~a.html) — ~a\n"
                                                (assq-ref module 'name)
                                                (assq-ref module 'name)
                                                (or (module-summary-line module) "")))
                                      (sort members
                                            (lambda (a b)
                                              (string<? (assq-ref a 'name)
                                                        (assq-ref b 'name))))))
                          "\n"))))
                 by-group)))))

;;; ------------------------------------------------------------- generation ---

(define (render-page site slug source title pytorch)
  (let* ((relative (page-path slug))
         (output (string-append site "/" relative))
         (root (root-prefix relative))
         (nav (nav-html slug root)))
    (if source
        (let* ((text (rewrite-markdown-links (read-file source)))
               (temporary (temporary-file ".md")))
          (write-file temporary text)
          (run-pandoc temporary output title root nav pytorch)
          (delete-file temporary))
        (error "no source for page" slug))))

(define (render-api-page site module)
  (let* ((name (assq-ref module 'name))
         (relative (string-append "api/" name ".html"))
         (output (string-append site "/" relative))
         (root (root-prefix relative))
         (slug (string-append "api/" name))
         (nav (nav-html slug root))
         (temporary (temporary-file ".md")))
    (write-file temporary (module-page-markdown module root))
    (run-pandoc temporary output (module-title module) root nav #f)
    (delete-file temporary)))

(define (module-title module)
  (assq-ref module 'name))

(define (copy-assets site)
  (copy-file (string-append tools-directory "/site.css")
             (string-append site "/site.css"))
  (for-each (lambda (name)
              (let ((source (string-append tools-directory "/" name)))
                (when (file-exists? source)
                  (copy-file source (string-append site "/" name)))))
            '("search.js" "pytorch.js")))

(define (build-site out)
  (ensure-directory out)
  (let* ((base (canonicalize-path out))
         (site (string-append base "/sonicboom"))
         (modules (site-modules))
         (entries (append (nav-entries) (concepts-tail-entries))))
    (ensure-directory (string-append site "/api"))
    (copy-assets site)
    (for-each
     (lambda (entry)
       (let ((slug (caddr entry)) (nav-title (cadr entry)))
         (cond
          ((string=? slug "api/index")
           (let* ((relative "api/index.html")
                  (root (root-prefix relative))
                  (temporary (temporary-file ".md")))
             (write-file temporary (api-index-markdown modules))
             (run-pandoc temporary (string-append site "/" relative)
                         nav-title root (nav-html slug root) #f)
             (delete-file temporary)))
          (else
           (let* ((source (page-source slug))
                  (title (or (and source (first-heading source)) nav-title)))
             (render-page site slug source title (string=? slug "pytorch")))))))
     entries)
    (for-each
     (lambda (module)
       (ensure-directory
        (dirname (string-append site "/api/" (assq-ref module 'name) ".html")))
       (render-api-page site module))
     modules)
    (site-index-redirect-script site)
    (let ((indexed (write-search-index site modules)))
      (format (current-error-port)
              "site generated: ~a (~a pages, ~a api pages, ~a search entries)\n"
              site (length entries) (length modules) indexed))))

;; /sonicboom without the trailing slash redirects to /sonicboom/, so that the
;; relative links of index.html resolve.
(define (site-index-redirect-script site)
  (let ((path (string-append site "/index.html")))
    (when (file-exists? path)
      (let* ((text (read-file path))
             (script (string-append
                      "<script>if (!location.pathname.endsWith('/') &&"
                      " !location.pathname.endsWith('.html'))"
                      " location.replace(location.pathname + '/' +"
                      " location.search + location.hash);</script>\n"))
             (head (string-contains text "</head>")))
        (when head
          (write-file path
                      (string-append (substring text 0 head)
                                     script
                                     (substring text head
                                                (string-length text)))))))))

;;; ----------------------------------------------------------------- search ---

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

(define (strip-tags text)
  (regexp-substitute/global
   #f "<[^>]*>" text 'pre "" 'post))

(define entity-pattern
  (make-regexp "&(amp|lt|gt|quot|#39|nbsp);"))

(define (unescape-html text)
  (regexp-substitute/global
   #f entity-pattern text 'pre
   (lambda (match)
     (case (string->symbol (match:substring match 1))
       ((amp) "&") ((lt) "<") ((gt) ">") ((quot) "\"")
       ((nbsp) " ") (else "'")))
   'post))

(define (text-of-html html)
  (string-trim (unescape-html (strip-tags html))))

(define (collapse-space text)
  (string-join (delete "" (string-split text #\space)) " "))

(define (summarize text limit)
  (let ((flat (collapse-space text)))
    (if (> (string-length flat) limit)
        (string-append (substring flat 0 limit) "…")
        flat)))

;; All matches of a pattern anchored at the opening of a tag.
(define (tag-start-matches text pattern)
  (let loop ((start 0) (acc '()))
    (let ((match (regexp-exec pattern text start)))
      (if match
          (loop (match:end match 0) (cons match acc))
          (reverse acc)))))

(define (inner-html text start end-tag)
  (let ((close (string-contains text end-tag start)))
    (and close (substring text start close))))

(define h1-start-pattern (make-regexp "<h1[^>]*>"))
(define h2-start-pattern (make-regexp "<h2 id=\"([^\"]+)\"[^>]*>"))
(define paragraph-start-pattern (make-regexp "<p>"))

(define (first-heading-text text)
  (let ((match (regexp-exec h1-start-pattern text)))
    (and match
         (let ((inner (inner-html text (match:end match 0) "</h1>")))
           (and inner (text-of-html inner))))))

(define (first-paragraph-text text)
  (let ((match (regexp-exec paragraph-start-pattern text)))
    (and match
         (let ((inner (inner-html text (match:end match 0) "</p>")))
           (and inner (text-of-html inner))))))

(define (h2-blocks text)
  (map (lambda (match)
         (list (match:substring match 1)
               (let ((inner (inner-html text (match:end match 0) "</h2>")))
                 (if inner (text-of-html inner) ""))))
       (tag-start-matches text h2-start-pattern)))

(define (relative-url site file)
  (substring file (+ (string-length site) 1) (string-length file)))

(define (nav-entry-for-path url)
  (find (lambda (one) (string=? (page-path (caddr one)) url))
        (append (nav-entries) (concepts-tail-entries))))

(define (section-of-page url)
  (let ((entry (nav-entry-for-path url)))
    (cond (entry (car entry))
          ((string-prefix? "api/" url) "API reference")
          (else "SonicBoom"))))

(define (module-by-name modules name)
  (find (lambda (module) (string=? (assq-ref module 'name) name)) modules))

(define (export-summary module symbol)
  (and module
       (let ((export (find (lambda (one)
                             (string=? (assq-ref one 'name) symbol))
                           (assq-ref module 'exports))))
         (and export
              (doc-field (assq-ref export 'doc) 'summary)))))

(define (heading-entries site file page-title modules)
  (let* ((text (read-file file))
         (module (module-by-name modules page-title))
         (relative (relative-url site file)))
    (map (lambda (block)
           (let* ((anchor (car block))
                  (heading (cadr block))
                  (symbol? (string-prefix? "sym-" anchor))
                  (summary (and symbol? (export-summary module heading))))
             (list (cons 't heading)
                   (cons 'k (if symbol? "symbol" "heading"))
                   (cons 'm page-title)
                   (cons 's (or summary ""))
                   (cons 'u (string-append relative "#" anchor)))))
         (h2-blocks text))))

(define (page-title-of text relative)
  (or (first-heading-text text)
      (let ((entry (nav-entry-for-path relative)))
        (if entry (cadr entry) relative))))

(define (page-search-entries site modules)
  (append-map
   (lambda (file)
     (let* ((text (read-file file))
            (relative (relative-url site file))
            (title (page-title-of text relative))
            (paragraph (first-paragraph-text text))
            (page (list (cons 't title)
                        (cons 'k "page")
                        (cons 'm (section-of-page relative))
                        (cons 's (if paragraph (summarize paragraph 160) ""))
                        (cons 'u relative))))
       (append (list page)
               (heading-entries site file title modules))))
   (html-files site)))

(define (json-escape text)
  (with-output-to-string
    (lambda ()
      (string-for-each
       (lambda (ch)
         (case ch
           ((#\") (display "\\\""))
           ((#\\) (display "\\\\"))
           ((#\newline) (display "\\n"))
           ((#\return) (display "\\r"))
           ((#\tab) (display "\\t"))
           (else
            (if (< (char->integer ch) 32)
                (format #t "\\u~4,'0x" (char->integer ch))
                (display ch)))))
       text))))

(define (entry->json entry)
  (format #f "{\"t\":\"~a\",\"k\":\"~a\",\"m\":\"~a\",\"s\":\"~a\",\"u\":\"~a\"}"
          (json-escape (assq-ref entry 't))
          (json-escape (assq-ref entry 'k))
          (json-escape (assq-ref entry 'm))
          (json-escape (assq-ref entry 's))
          (json-escape (assq-ref entry 'u))))

(define (write-search-index site modules)
  (let ((entries (page-search-entries site modules)))
    (write-file (string-append site "/search-index.json")
                (string-append "[\n"
                               (string-join (map entry->json entries)
                                            ",\n")
                               "\n]\n"))
    (length entries)))

;;; -------------------------------------------------------------------- cli ---

(define (main)
  (let ((out (arg-value "--out")))
    (unless out
      (format (current-error-port) "usage: guile build.scm --out DIR\n")
      (exit 2))
    (build-site out)
    (exit 0)))

(main)
