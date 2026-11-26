;;; SonicBoom — inference service: style assets
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A style is a named, versioned writing style a chat request can ask for with
;;;   "sonicboom": {"style": {"id": ..., "version": ..., "alpha": ...}}
;;; It has two halves:
;;;
;;;   card    a system text plus up to K user/assistant exemplar turns, put in
;;;           front of the request's own messages. They are a fixed prefix, so
;;;           the session pool keeps reusing them across requests.
;;;   vector  a steering vector: the mean residual stream at one block over the
;;;           exemplars' replies, minus the same mean over a fixed set of
;;;           neutral reference sentences, added to the residual stream after
;;;           that block during prefill and every decode step. alpha scales it
;;;           per request; the registry keeps the unscaled vector.
;;;
;;; A style made without a layer has only a card. A vector needs a backend that
;;; can read residuals ((sonicboom serve backend): residual-capture and
;;; model-embedding); building one runs that backend's prefill on a throwaway
;;; cache, so it happens under the engine's queue, with the model resident.
;;;
;;; The registry is in memory, like the replay store; it does not survive the
;;; process. Creation is deterministic: the same exemplars, model and layer
;;; produce the same vector.

(define-module (sonicboom serve styles)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-13)
  #:use-module (ice-9 threads)
  #:use-module (rnrs bytevectors)
  #:use-module (sonicboom serve backend)
  #:export (make-style-registry style-registry?
            style-create! style-ref style-list style-delete!
            style-json style-id style-version style-model style-layer
            style-has-vector?
            parse-style-create
            resolve-style-choice))

;;; ---------------------------------------------------------------------------
;;; The records

(define-record-type <style>
  (%make-style id version model system exemplars layer vector created)
  style?
  (id style-id)
  (version style-version)
  (model style-model)          ; model id the vector was measured under
  (system style-system)        ; string or #f
  (exemplars style-exemplars)  ; list of ((prompt . string) (reply . string))
  (layer style-layer)          ; block index, or #f for a card-only style
  (vector style-vector)        ; unscaled f32 bytevector, or #f
  (created style-created))

(define (style-has-vector? style)
  "Whether STYLE carries a steering vector (it was created with a layer).

Args:
  style: a style.
Returns: a boolean.
Example:
  (use-modules (sonicboom serve styles) (sonicboom serve backend)
               (sonicboom serve mock-backend))
  (let* ((reg (make-style-registry))
         (model (load-model (make-mock-backend) '((id . \"m\"))))
         (style (style-create! reg
                  '((id . \"dry\") (version . 1) (model . \"m\")
                    (system . \"Write plainly.\") (exemplars . ()) (layer . #f))
                  model)))
    (style-has-vector? style))
  => #f
PyTorch: none"
  (and (style-layer style) #t))

(define-record-type <style-registry>
  (%make-style-registry table lock)
  style-registry?
  (table style-registry-table)
  (lock style-registry-lock))

(define (make-style-registry)
  "Create an empty style registry.

The registry maps a style id to its current version. It lives in memory: styles
are created with style-create! and served until the process ends or the id is
deleted.

Returns: a style registry.
Example:
  (use-modules (sonicboom serve styles))
  (style-registry? (make-style-registry))
  => #t
PyTorch: none"
  (%make-style-registry (make-hash-table) (make-mutex)))

;;; ---------------------------------------------------------------------------
;;; Parsing the creation body
;;;
;;; The body is the already-decoded JSON object (string keys, arrays as
;;; vectors). Errors are thrown as (style-invalid param message); the HTTP layer
;;; answers 400.

(define (json-object? v)
  (and (list? v) (every pair? v)))

(define (field obj key) (and (list? obj) (assoc-ref obj key)))

(define (present? v) (not (or (eq? v 'null) (eq? v #f))))

(define (invalid param message)
  (throw 'style-invalid param message))

(define (check-id v)
  (unless (and (string? v) (<= 1 (string-length v) 64))
    (invalid "id" "id must be a string of 1 to 64 characters"))
  (unless (string-every (lambda (c)
                          (or (char-alphabetic? c) (char-numeric? c)
                              (memv c '(#\- #\_ #\.))))
                        v)
    (invalid "id" "id may hold letters, digits, '-', '_' and '.' only"))
  v)

(define (check-text param v limit)
  (unless (and (string? v) (<= 1 (string-length v) limit))
    (invalid param (format #f "~a must be a non-empty string of at most ~a characters"
                           param limit)))
  v)

(define (check-exemplars v)
  (unless (or (not (present? v)) (vector? v))
    (invalid "exemplars" "exemplars must be an array"))
  (let ((l (if (vector? v) (vector->list v) '())))
    (unless (<= (length l) 8)
      (invalid "exemplars" "at most 8 exemplars"))
    (map (lambda (e)
           (unless (json-object? e)
             (invalid "exemplars" "each exemplar must be an object"))
           (list (cons 'prompt (check-text "exemplars" (field e "prompt") 2000))
                 (cons 'reply (check-text "exemplars" (field e "reply") 2000))))
         l)))

(define (check-layer v)
  (cond ((not (present? v)) #f)
        ((and (integer? v) (exact? v) (<= 0 v 1023)) v)
        (else (invalid "layer" "layer must be a block index"))))

;; A validated creation spec:
;;   ((id . string) (version . integer) (model . string)
;;    (system . string-or-#f) (exemplars . alist list) (layer . integer-or-#f))
(define (parse-style-create body)
  "Check a style-creation body and return the validated spec.

The body holds id, model, and optionally version (default 1), system,
exemplars (an array of {\"prompt\", \"reply\"} objects, at most 8), and layer
(absent for a card-only style). A style needs a system text or an exemplar.

Args:
  body: the decoded JSON request body.
Returns: a spec alist for style-create!.
Example:
  (use-modules (sonicboom serve styles))
  (assq-ref (parse-style-create
             '((\"id\" . \"dry\") (\"model\" . \"m\")
               (\"system\" . \"Write plainly.\")))
           'id)
  => \"dry\"
Note: a bad body raises (style-invalid param message).
PyTorch: none"
  (unless (json-object? body)
    (invalid 'null "the request body must be a JSON object"))
  (let* ((id (check-id (field body "id")))
         (version (let ((v (field body "version")))
                    (cond ((not (present? v)) 1)
                          ((and (integer? v) (exact? v) (>= v 1)) v)
                          (else (invalid "version" "version must be a positive integer")))))
         (model (check-text "model" (field body "model") 256))
         (system (let ((v (field body "system")))
                   (if (present? v) (check-text "system" v 8192) #f)))
         (exemplars (check-exemplars (field body "exemplars")))
         (layer (check-layer (field body "layer"))))
    (when (and (not system) (null? exemplars))
      (invalid "exemplars" "a style needs a system text or at least one exemplar"))
    (when (and layer (null? exemplars))
      (invalid "exemplars" "a steering vector is built from exemplar replies"))
    (list (cons 'id id) (cons 'version version) (cons 'model model)
          (cons 'system system) (cons 'exemplars exemplars)
          (cons 'layer layer))))

;;; ---------------------------------------------------------------------------
;;; The steering vector
;;;
;;; The neutral reference set is fixed in the source so the same exemplars
;;; always give the same vector. It is deliberately plain, factual prose: the
;;; difference from it is the style, not the topic.
(define neutral-references
  '("The sky is blue."
    "Water boils at one hundred degrees Celsius."
    "A week has seven days."
    "The library opens at nine in the morning."
    "Trains leave from the central station."
    "A rectangle has four sides."
    "Please close the window before you leave."
    "The meeting is scheduled for next Tuesday."))

(define (f32-ref bv i) (bytevector-ieee-single-native-ref bv (* 4 i)))
(define (f32-set! bv i v) (bytevector-ieee-single-native-set! bv (* 4 i) v))

;; acc[j] += every row's j-th value; returns the row count seen.
(define (accumulate-rows! acc bv width)
  (let ((n (quotient (bytevector-length bv) (* 4 width))))
    (do ((row 0 (+ row 1))) ((= row n) n)
      (do ((j 0 (+ j 1))) ((= j width))
        (vector-set! acc j
                     (+ (vector-ref acc j) (exact->inexact (f32-ref bv (+ (* row width) j)))))))))

;; The mean residual at LAYER over every position of every TEXT, as a vector of
;; doubles plus the position count. Each text is one prefill pass on the
;; backend's throwaway cache.
(define (mean-residual model texts layer width)
  (let ((acc (make-vector width 0.0))
        (count 0))
    (for-each (lambda (text)
                (let* ((tokens (tokenize model text))
                       (bv (residual-capture model tokens layer)))
                  (when (zero? (bytevector-length bv))
                    (throw 'style-invalid 'exemplars "residual capture returned nothing"))
                  (set! count (+ count (accumulate-rows! acc bv width)))))
              texts)
    (values acc count)))

(define (mean-difference-model->vector model exemplar-replies layer)
  (let* ((width (model-embedding model))
         (e (make-vector width 0.0))
         (n (make-vector width 0.0)))
    (call-with-values (lambda () (mean-residual model exemplar-replies layer width))
      (lambda (e-acc e-count)
        (when (zero? e-count)
          (throw 'style-invalid 'exemplars "the exemplars produced no tokens"))
        (call-with-values (lambda () (mean-residual model neutral-references layer width))
          (lambda (n-acc n-count)
            (do ((j 0 (+ j 1))) ((= j width))
              (vector-set! e j (/ (vector-ref e-acc j) e-count))
              (vector-set! n j (/ (vector-ref n-acc j) n-count)))))))
    (let ((bv (make-bytevector (* 4 width) 0)))
      (do ((j 0 (+ j 1))) ((= j width))
        (f32-set! bv j (- (vector-ref e j) (vector-ref n j))))
      bv)))

;;; ---------------------------------------------------------------------------
;;; The registry

(define (now-seconds) (car (gettimeofday)))

(define (style-create! registry spec model)
  "Create (or replace) the style SPEC names, measured under MODEL.

MODEL is the loaded model handle the spec's model id resolves to. A spec with
a layer has its steering vector built here, from the exemplar replies and the
module's fixed neutral references; that needs a backend with residual capture
and runs one prefill per text. A spec without a layer stores only the card.

Args:
  registry: a style registry.
  spec: a validated spec from parse-style-create.
  model: the loaded model the spec's model id names.
Returns: the stored style.
Example:
  (use-modules (sonicboom serve styles) (sonicboom serve backend)
               (sonicboom serve mock-backend))
  (let* ((reg (make-style-registry))
         (model (load-model (make-mock-backend) '((id . \"m\"))))
         (style (style-create! reg
                  '((id . \"dry\") (version . 1) (model . \"m\")
                    (system . \"Write plainly.\") (exemplars . ()) (layer . #f))
                  model)))
    (style-id style))
  => \"dry\"
Note: a layer on a backend without residual capture raises (style-invalid
layer message); the vector's block must exist in the model.
PyTorch: none"
  (let* ((id (assq-ref spec 'id))
         (layer (assq-ref spec 'layer))
         (exemplars (assq-ref spec 'exemplars))
         (vector (and layer
                      (begin
                        (unless (backend-residual? model)
                          (invalid "layer"
                                   "this backend cannot build a steering vector"))
                        (mean-difference-model->vector
                         model (map (lambda (e) (assq-ref e 'reply)) exemplars)
                         layer))))
         (style (%make-style id (assq-ref spec 'version) (assq-ref spec 'model)
                             (assq-ref spec 'system) exemplars layer vector
                             (now-seconds))))
    (with-mutex (style-registry-lock registry)
      (hash-set! (style-registry-table registry) id style))
    style))

(define (style-ref registry id)
  "The style named ID in REGISTRY, or #f.

Args:
  registry: a style registry.
  id: a style id string.
Returns: a style or #f.
Example:
  (use-modules (sonicboom serve styles))
  (style-ref (make-style-registry) \"dry\")
  => #f
PyTorch: none"
  (with-mutex (style-registry-lock registry)
    (hash-ref (style-registry-table registry) id #f)))

(define (style-list registry)
  "Every style in REGISTRY, ordered by id.

Args:
  registry: a style registry.
Returns: a list of styles.
Example:
  (use-modules (sonicboom serve styles))
  (style-list (make-style-registry))
  => ()
PyTorch: none"
  (with-mutex (style-registry-lock registry)
    (sort (hash-map->list (lambda (k v) v) (style-registry-table registry))
          (lambda (a b) (string<? (style-id a) (style-id b))))))

(define (style-delete! registry id)
  "Delete the style named ID, if any.

Args:
  registry: a style registry.
  id: a style id string.
Returns: #t when a style was deleted, #f when ID was not registered.
Example:
  (use-modules (sonicboom serve styles))
  (style-delete! (make-style-registry) \"dry\")
  => #f
PyTorch: none"
  (with-mutex (style-registry-lock registry)
    (if (hash-ref (style-registry-table registry) id #f)
        (begin (hash-remove! (style-registry-table registry) id) #t)
        #f)))

(define (style-json style)
  "STYLE as an alist for a JSON response.

The vector itself is not returned: only whether the style has one.

Args:
  style: a style.
Returns: an alist.
Example:
  (use-modules (sonicboom serve styles) (sonicboom serve backend)
               (sonicboom serve mock-backend))
  (let* ((reg (make-style-registry))
         (model (load-model (make-mock-backend) '((id . \"m\"))))
         (style (style-create! reg
                  '((id . \"dry\") (version . 1) (model . \"m\")
                    (system . \"Write plainly.\") (exemplars . ()) (layer . #f))
                  model)))
    (assq-ref (style-json style) \"id\"))
  => \"dry\"
PyTorch: none"
  `(("id" . ,(style-id style))
    ("version" . ,(style-version style))
    ("model" . ,(style-model style))
    ("layer" . ,(or (style-layer style) 'null))
    ("vector" . ,(if (style-has-vector? style) #t #f))
    ("system" . ,(if (style-system style) #t #f))
    ("exemplars" . ,(length (style-exemplars style)))
    ("created" . ,(style-created style))))

;;; ---------------------------------------------------------------------------
;;; Using a style at request time

;; The card's messages, in the same string-keyed shape as a request's: the
;; system text first, then one user/assistant pair per exemplar.
(define (card-messages style)
  (append
   (if (style-system style)
       (list (list (cons "role" "system") (cons "content" (style-system style))))
       '())
   (append-map (lambda (e)
                 (list (list (cons "role" "user")
                             (cons "content" (assq-ref e 'prompt)))
                       (list (cons "role" "assistant")
                             (cons "content" (assq-ref e 'reply)))))
               (style-exemplars style))))

(define (scale-vector bv alpha)
  (if (or (not alpha) (= alpha 1.0))
      bv
      (let* ((n (quotient (bytevector-length bv) 4))
             (out (make-bytevector (bytevector-length bv) 0)))
        (do ((i 0 (+ i 1))) ((= i n))
          (f32-set! out i (* alpha (f32-ref bv i))))
        out)))

;; A request's style spec is (("id" . s) ("version" . n) ("alpha" . x)) as
;; (sonicboom serve openai) validated it: an alist with those string keys.
(define (resolve-style-choice registry spec model-id alpha)
  "Resolve a request's style spec into the engine's style value.

The engine takes (card-messages layer vector) or #f: the card's messages to
prepend, the block the steering vector follows, and that vector scaled to
ALPHA. A card-only style has #f layer and vector.

Args:
  registry: a style registry.
  spec: the request's style spec alist (id, optional version, optional alpha).
  model-id: the id of the model the request names.
  alpha: the steering strength (1.0 for the vector as stored).
Returns: the engine's style value, or #f.
Example:
  (use-modules (sonicboom serve styles) (sonicboom serve backend)
               (sonicboom serve mock-backend))
  (let* ((reg (make-style-registry))
         (model (load-model (make-mock-backend) '((id . \"m\")))))
    (style-create! reg
     '((id . \"dry\") (version . 1) (model . \"m\") (system . \"Write plainly.\")
       (exemplars . ()) (layer . #f))
     model)
    (car (resolve-style-choice reg '((\"id\" . \"dry\")) \"m\" 1.0)))
  => ((\"role\" . \"system\") (\"content\" . \"Write plainly.\"))
Note: an unknown id raises (style-not-found id); a version other than the
stored one raises (style-version-mismatch id want got); a model other than the
one the vector was measured under raises (style-model-mismatch id model).
PyTorch: none"
  (let* ((id (assoc-ref spec "id"))
         (version (assoc-ref spec "version"))
         (style (or (style-ref registry id) (throw 'style-not-found id))))
    (when (and version (not (= version (style-version style))))
      (throw 'style-version-mismatch id version (style-version style)))
    (unless (string=? model-id (style-model style))
      (throw 'style-model-mismatch id model-id (style-model style)))
    (list (card-messages style)
          (style-layer style)
          (and (style-has-vector? style) (scale-vector (style-vector style) alpha)))))
