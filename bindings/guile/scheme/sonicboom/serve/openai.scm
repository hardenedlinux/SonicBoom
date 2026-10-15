;;; SonicBoom — inference service: the OpenAI chat-completions format
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Decoded JSON in, decoded JSON out; no HTTP here. JSON values are the
;;; guile-json ones: objects are alists with string keys, arrays are vectors,
;;; null is the symbol null.
;;;
;;; parse-chat-request checks a /v1/chat/completions body and turns it into
;;; an engine request; a bad body raises
;;;   (openai-error status type message param)
;;; which openai-error-object turns into the OpenAI error body.

(define-module (sonicboom serve openai)
  #:use-module (srfi srfi-1)
  #:use-module (ice-9 match)
  #:use-module (sonicboom serve engine)
  #:use-module (sonicboom serve chat-template)
  #:export (parse-chat-request
            check-greedy-body
            openai-error-object raise-openai-error
            completion-object chunk-object usage-chunk-object
            role-delta content-delta tool-call-deltas finish-delta
            usage-object model-object models-object
            new-completion-id))

(define (raise-openai-error status type message . param)
  (throw 'openai-error status type message (if (pair? param) (car param) 'null)))

(define (invalid message param)
  (raise-openai-error 400 "invalid_request_error" message param))

(define (openai-error-object type message param code)
  `(("error" . (("message" . ,message)
                ("type" . ,type)
                ("param" . ,param)
                ("code" . ,code)))))

(define (ref obj key)
  (match (assoc key obj)
    ((_ . 'null) #f)
    ((_ . v) v)
    (#f #f)))

(define (object? x) (and (list? x) (every pair? x)))

(define (number-in name v lo hi)
  (unless (and (real? v) (<= lo v hi))
    (invalid (format #f "~a must be a number in [~a, ~a]" name lo hi) name))
  v)

(define (positive-integer name v)
  (unless (and (integer? v) (exact? v) (> v 0))
    (invalid (format #f "~a must be a positive integer" name) name))
  v)

(define (check-messages v)
  (unless (and (vector? v) (> (vector-length v) 0))
    (invalid "messages must be a non-empty array" "messages"))
  (let ((ms (vector->list v)))
    (for-each
     (lambda (m i)
       (unless (and (object? m) (string? (ref m "role")))
         (invalid (format #f "messages[~a] must be an object with a role" i) "messages"))
       (unless (member (ref m "role") '("system" "developer" "user" "assistant" "tool"))
         (invalid (format #f "messages[~a]: unknown role ~s" i (ref m "role")) "messages")))
     ms (iota (length ms)))
    ms))

(define (check-tools v)
  (match v
    (#f '())
    ((? vector? v)
     (let ((tools (vector->list v)))
       (for-each
        (lambda (t)
          (unless (and (object? t) (equal? (ref t "type") "function")
                       (object? (ref t "function"))
                       (string? (ref (ref t "function") "name")))
            (invalid "each tool must be {\"type\": \"function\", \"function\": {\"name\": ...}}"
                     "tools")))
        tools)
       tools))
    (_ (invalid "tools must be an array" "tools"))))

(define (check-tool-choice v tools)
  (match v
    (#f (if (null? tools) "none" "auto"))
    ((? string? s)
     (unless (member s '("none" "auto" "required"))
       (invalid "tool_choice must be none, auto, required or a function" "tool_choice"))
     (when (and (null? tools) (member s '("required")))
       (invalid "tool_choice requires tools" "tool_choice"))
     s)
    ((? object? o)
     (let ((name (and (object? (ref o "function")) (ref (ref o "function") "name"))))
       (unless (and (string? name)
                    (any (lambda (t) (equal? (ref (ref t "function") "name") name)) tools))
         (invalid "tool_choice names a function that is not in tools" "tool_choice"))
       name))
    (_ (invalid "tool_choice must be a string or an object" "tool_choice"))))

(define (check-stop v)
  (match v
    (#f '())
    ((? string? s) (list s))
    ((? vector? v)
     (let ((l (vector->list v)))
       (unless (and (<= (length l) 4) (every string? l))
         (invalid "stop must be a string or an array of up to 4 strings" "stop"))
       l))
    (_ (invalid "stop must be a string or an array of up to 4 strings" "stop"))))

;; body: the decoded request object (key order restored by json-normalize).
;; -> (values chat-request stream? include-usage?); include-usage? is
;; stream_options.include_usage, which only streaming replies look at.
(define (parse-chat-request body)
  (unless (object? body)
    (invalid "the request body must be a JSON object" 'null))
  (let* ((model (ref body "model"))
         (checked-model (unless (string? model) (invalid "model is required" "model")))
         (messages (check-messages (ref body "messages")))
         (n (ref body "n"))
         (checked-n (when (and n (not (eqv? n 1)))
              (invalid "only n=1 is supported" "n")))
         (stream (ref body "stream"))
         (include-usage (and (object? (ref body "stream_options"))
                             (eq? #t (ref (ref body "stream_options") "include_usage"))))
         (max-tokens (cond ((ref body "max_completion_tokens")
                            => (lambda (v) (positive-integer "max_completion_tokens" v)))
                           ((ref body "max_tokens")
                            => (lambda (v) (positive-integer "max_tokens" v)))
                           (else #f)))
         (temperature (match (ref body "temperature")
                        (#f 1.0)
                        (v (number-in "temperature" v 0 2))))
         (top-p (match (ref body "top_p")
                  (#f 1.0)
                  (v (number-in "top_p" v 0 1))))
         (top-k (match (ref body "top_k")              ; not OpenAI, common extension
                  (#f 0)
                  (v (unless (and (integer? v) (exact? v) (>= v 0))
                       (invalid "top_k must be a non-negative integer" "top_k"))
                     v)))
         (seed (match (ref body "seed")
                 (#f #f)
                 (v (unless (and (integer? v) (exact? v))
                      (invalid "seed must be an integer" "seed"))
                    (abs v))))
         (tools (check-tools (ref body "tools")))
         (tool-choice (check-tool-choice (ref body "tool_choice") tools)))
    (values (make-chat-request #:model model #:messages messages
                               #:tools tools #:tool-choice tool-choice
                               #:max-tokens max-tokens #:stop (check-stop (ref body "stop"))
                               #:temperature temperature #:top-p top-p #:top-k top-k
                               #:seed seed)
            (eq? stream #t)
            include-usage)))

;; A greedy-only backend cannot honor a sampling request, so an explicit one is
;; rejected instead of silently decoding greedily (the core's decode wants
;; temperature 0, top-p 1, top-k 0 and returns SB_ERR_UNSUPPORTED otherwise).
;; A parameter that is absent means "the backend's default", which for a
;; greedy-only backend is greedy — so the common OpenAI client that leaves
;; temperature at its default still works, while one that asks for sampling is
;; told so. BODY is the raw request object, before parse-chat-request's
;; defaults, so "absent" is still visible here. Call it after
;; parse-chat-request: that is what vets the parameter types.
(define (check-greedy-body body)
  (let ((t (ref body "temperature"))
        (p (ref body "top_p"))
        (k (ref body "top_k")))
    (when (and (number? t) (not (zero? t)))
      (invalid "this backend decodes greedily; temperature must be 0" "temperature"))
    (when (and (number? p) (not (= p 1)))
      (invalid "this backend decodes greedily; top_p must be 1" "top_p"))
    (when (and (number? k) (not (zero? k)))
      (invalid "this backend decodes greedily; top_k must be 0" "top_k"))))

;;; ---------------------------------------------------------------------------
;;; Responses

(define (new-completion-id)
  (string-append "chatcmpl-"
                 (number->string (random (expt 2 64) (random-state-from-platform)) 36)))

(define (usage-object result)
  `(("prompt_tokens" . ,(result-prompt-tokens result))
    ("completion_tokens" . ,(result-completion-tokens result))
    ("total_tokens" . ,(+ (result-prompt-tokens result) (result-completion-tokens result)))
    ("prompt_tokens_details" . (("cached_tokens" . ,(result-cached-tokens result))))))

(define (tool-call-object call)
  (match call
    ((id name args)
     `(("id" . ,id)
       ("type" . "function")
       ("function" . (("name" . ,name) ("arguments" . ,args)))))))

(define (completion-object id created model result)
  (let ((calls (result-tool-calls result))
        (content (result-content result)))
    `(("id" . ,id)
      ("object" . "chat.completion")
      ("created" . ,created)
      ("model" . ,model)
      ("choices"
       . #((("index" . 0)
            ("message"
             . (("role" . "assistant")
                ("content" . ,(if (and (string-null? content) (pair? calls)) 'null content))
                ,@(if (pair? calls)
                      `(("tool_calls" . ,(list->vector (map tool-call-object calls))))
                      '())))
            ("logprobs" . null)
            ("finish_reason" . ,(result-finish-reason result)))))
      ("usage" . ,(usage-object result)))))

(define* (chunk-object id created model delta #:optional (finish-reason 'null))
  `(("id" . ,id)
    ("object" . "chat.completion.chunk")
    ("created" . ,created)
    ("model" . ,model)
    ("choices" . #((("index" . 0)
                    ("delta" . ,delta)
                    ("logprobs" . null)
                    ("finish_reason" . ,finish-reason))))))

;; The last chunk when stream_options.include_usage is set: no choices.
(define (usage-chunk-object id created model result)
  `(("id" . ,id)
    ("object" . "chat.completion.chunk")
    ("created" . ,created)
    ("model" . ,model)
    ("choices" . #())
    ("usage" . ,(usage-object result))))

(define (role-delta) '(("role" . "assistant") ("content" . "")))
(define (content-delta text) `(("content" . ,text)))
(define (finish-delta) '())

;; One tool call as stream deltas: the header (id, type, name, empty
;; arguments), then the arguments in pieces of at most `piece' characters.
(define* (tool-call-deltas index id name args #:key (piece 8))
  (cons
   `(("tool_calls" . #((("index" . ,index)
                         ("id" . ,id)
                         ("type" . "function")
                         ("function" . (("name" . ,name) ("arguments" . "")))))))
   (let lp ((i 0) (out '()))
     (if (>= i (string-length args))
         (reverse out)
         (let ((j (min (string-length args) (+ i piece))))
           (lp j (cons `(("tool_calls" . #((("index" . ,index)
                                             ("function" . (("arguments" . ,(substring args i j))))))))
                       out)))))))

;; `family` is the chat-template family the loaded model selected (the serving
;; backend reads the GGUF architecture and maps it — see real-backend's
;; family-for-arch). It is not part of the OpenAI schema, but it tells a client
;; which prompt format and which stop tokens this model expects instead of
;; leaving it to infer the family from the file name.
(define (model-object id family created)
  `(("id" . ,id) ("object" . "model") ("created" . ,created)
    ("owned_by" . "sonicboom") ("family" . ,(symbol->string family))))

;; `models` is an (id . family) alist, in serving order.
(define (models-object models created)
  `(("object" . "list")
    ("data" . ,(list->vector
                (map (lambda (m) (model-object (car m) (cdr m) created)) models)))))
