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
  #:use-module (json)
  #:use-module (rnrs bytevectors)
  #:use-module (sonicboom serve backend)
  #:use-module (sonicboom serve engine)
  #:use-module (sonicboom serve chat-template)
  #:export (parse-chat-request
            check-greedy-body
            parse-image-request images-object base64-encode-bytevector
            openai-error-object raise-openai-error
            completion-object chunk-object usage-chunk-object
            role-delta content-delta tool-call-deltas finish-delta
            usage-object model-object models-object
            new-completion-id))

(define (raise-openai-error status type message . param)
  "Raise an OpenAI error.

The route catches (openai-error status type message param) and answers with
openai-error-object and the status.

Args:
  status: the HTTP status.
  type: the error type string.
  message: the message string.
  param: optional; the offending parameter name (default null).
Returns: never returns normally.
Example:
  (use-modules (sonicboom serve openai))
  (catch 'openai-error
    (lambda () (raise-openai-error 400 \"invalid_request_error\" \"bad\"))
    (lambda (key status type message param) (list status type)))
  => (400 \"invalid_request_error\")
PyTorch: none"
  (throw 'openai-error status type message (if (pair? param) (car param) 'null)))

(define (invalid message param)
  (raise-openai-error 400 "invalid_request_error" message param))

(define (openai-error-object type message param code)
  "The OpenAI error body.

Args:
  type: the error type string.
  message: the message string.
  param: the offending parameter name, or null.
  code: the error code, or #f.
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve openai))
  (assoc-ref (assoc-ref (openai-error-object \"t\" \"m\" 'null #f) \"error\")
             \"message\")
  => \"m\"
PyTorch: none"
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

;; response_format: {"type": "text"} (the default), {"type": "json_object"}, or
;; {"type": "json_schema", "json_schema": {"name": ..., "schema": <schema>}}.
;; The schema is passed on as JSON text; the backend compiles it.
(define (check-response-format v)
  (match v
    (#f #f)
    ((? object? o)
     (let ((type (ref o "type")))
       (cond
        ((equal? type "text") #f)
        ((equal? type "json_object")
         (make-json-schema-constraint "{\"type\":\"object\"}"))
        ((equal? type "json_schema")
         (let* ((js (ref o "json_schema"))
                (schema (and (object? js) (ref js "schema"))))
           (unless (object? schema)
             (invalid "response_format.json_schema.schema must be an object"
                      "response_format"))
           (make-json-schema-constraint (scm->json-string schema))))
        (else
         (invalid "response_format.type must be text, json_object or json_schema"
                  "response_format")))))
    (_ (invalid "response_format must be an object" "response_format"))))

;; One tool's call object: the name restricted to that tool, the arguments to
;; its parameters schema.
(define (tool-call-schema tool)
  (let* ((fn (ref tool "function"))
         (name (ref fn "name"))
         (params (or (ref fn "parameters") '(("type" . "object")))))
    `(("type" . "object")
      ("properties" . (("name" . (("const" . ,name)))
                       ("arguments" . ,params)))
      ("required" . #("name" "arguments"))
      ("additionalProperties" . #f))))

;; A tool-call constraint only for a forced call: "required" (any of the tools,
;; as an anyOf alternation) or a named function. "auto" leaves the model free
;; to answer in text, so there is no grammar that could apply to the reply.
(define (tool-choice-constraint tools tool-choice)
  (cond
   ((or (null? tools) (member tool-choice '("none" "auto"))) #f)
   ((string=? tool-choice "required")
    (make-tool-call-constraint
     (scm->json-string `(("anyOf" . ,(list->vector (map tool-call-schema tools)))))))
   (else  ; a function name, already checked against tools
    (make-tool-call-constraint
     (scm->json-string
      (tool-call-schema
       (find (lambda (t) (equal? (ref (ref t "function") "name") tool-choice)) tools)))))))

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

;; A seed for a sampling request that did not ask for one. It is reported back
;; in the reply (sonicboom_seed), so a client that keeps it replays the reply.
(define (pick-seed)
  (random (expt 2 32) (random-state-from-platform)))

;; body: the decoded request object (key order restored by json-normalize).
;; -> (values chat-request stream? include-usage?); include-usage? is
;; stream_options.include_usage, which only streaming replies look at.
(define (parse-chat-request body)
  "Check a chat-completions body and build the engine request.

JSON values are the guile-json ones: objects are alists with string keys,
arrays are vectors and null is the symbol null. An omitted temperature is 0
(greedy). An omitted seed is 0 for a greedy request and a fresh random seed
for a sampling one (the reply's sonicboom_seed tells the client which). Only
n=1 is supported.

Args:
  body: the decoded request object.
Returns: three values: the chat request, whether the reply is streamed, and
whether stream_options.include_usage is set.
Example:
  (use-modules (sonicboom serve engine) (sonicboom serve openai))
  (call-with-values
      (lambda ()
        (parse-chat-request
         '((\"model\" . \"m\")
           (\"messages\" . #(((\"role\" . \"user\") (\"content\" . \"hi\")))))))
    (lambda (request stream usage)
      (list (chat-request-model request) stream usage)))
  => (\"m\" #f #f)
PyTorch: none
Note: a bad body raises (openai-error status type message param)."
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
         ;; An omitted temperature is 0: greedy. That is not the OpenAI
         ;; service's 1.0 default — a client that wants sampling asks for it.
         (temperature (match (ref body "temperature")
                        (#f 0.0)
                        (v (number-in "temperature" v 0 2))))
         (top-p (match (ref body "top_p")
                  (#f 1.0)
                  (v (number-in "top_p" v 0 1))))
         (top-k (match (ref body "top_k")              ; not OpenAI, common extension
                  (#f 0)
                  (v (unless (and (integer? v) (exact? v) (>= v 0))
                       (invalid "top_k must be a non-negative integer" "top_k"))
                     v)))
         (min-p (match (ref body "min_p")             ; not OpenAI, common extension
                  (#f 0.0)
                  (v (number-in "min_p" v 0 1))))
         ;; A seed is only meaningful when sampling. A request that omits it
         ;; gets a fresh one, returned as sonicboom_seed, so the client can
         ;; replay the reply; a greedy request keeps 0.
         (seed (match (ref body "seed")
                 (#f (if (> temperature 0) (pick-seed) 0))
                 (v (unless (and (integer? v) (exact? v))
                      (invalid "seed must be an integer" "seed"))
                    (abs v))))
         (repeat-penalty (match (ref body "repeat_penalty") ; not OpenAI, common extension
                           (#f 1.0)
                           (v (if (and (number? v) (>= v 0))
                                  (exact->inexact v)
                                  (invalid "repeat_penalty must be a non-negative number"
                                           "repeat_penalty")))))
         (presence-penalty (match (ref body "presence_penalty")
                             (#f 0.0)
                             (v (number-in "presence_penalty" v -2 2))))
         (frequency-penalty (match (ref body "frequency_penalty")
                              (#f 0.0)
                              (v (number-in "frequency_penalty" v -2 2))))
         (tools (check-tools (ref body "tools")))
         (tool-choice (check-tool-choice (ref body "tool_choice") tools))
         ;; An explicit response_format is the reply's shape and wins; a forced
         ;; tool call shapes it only when no response_format is asked for.
         (constraint (or (check-response-format (ref body "response_format"))
                         (tool-choice-constraint tools tool-choice))))
    (values (make-chat-request #:model model #:messages messages
                               #:tools tools #:tool-choice tool-choice
                               #:max-tokens max-tokens #:stop (check-stop (ref body "stop"))
                               #:temperature temperature #:top-p top-p #:top-k top-k
                               #:min-p min-p #:seed seed
                               #:repeat-penalty repeat-penalty
                               #:presence-penalty presence-penalty
                               #:frequency-penalty frequency-penalty
                               #:constraint constraint)
            (eq? stream #t)
            include-usage)))

;; A backend that cannot sample (model-sampling? #f) cannot honor a sampling
;; request, so an explicit one is rejected instead of silently decoding
;; greedily. The app calls this only for such a backend.
;; A parameter that is absent means "the backend's default", which for a
;; non-sampling backend is greedy — so the common OpenAI client that leaves
;; temperature at its default still works, while one that asks for sampling is
;; told so. BODY is the raw request object, before parse-chat-request's
;; defaults, so "absent" is still visible here. Call it after
;; parse-chat-request: that is what vets the parameter types.
(define (check-greedy-body body)
  "Reject an explicit sampling request for a backend that decodes greedily.

A parameter that is absent means the backend's default, so a client that leaves
the sampling parameters alone is accepted; one that asks for sampling is told
why it cannot have it. Call this after parse-chat-request, which checks the
parameter types.

Args:
  body: the raw decoded request object, before the parsing defaults.
Returns: unspecified.
Example:
  (use-modules (sonicboom serve openai))
  (begin (check-greedy-body '((\"temperature\" . 0))) #t)
  => #t
PyTorch: none"
  (let ((t (ref body "temperature"))
        (p (ref body "top_p"))
        (k (ref body "top_k"))
        (m (ref body "min_p"))
        (r (ref body "repeat_penalty")))
    (when (and (number? t) (not (zero? t)))
      (invalid "this backend decodes greedily; temperature must be 0" "temperature"))
    (when (and (number? p) (not (= p 1)))
      (invalid "this backend decodes greedily; top_p must be 1" "top_p"))
    (when (and (number? k) (not (zero? k)))
      (invalid "this backend decodes greedily; top_k must be 0" "top_k"))
    (when (and (number? m) (not (zero? m)))
      (invalid "this backend decodes greedily; min_p must be 0" "min_p"))
    (when (and (number? r) (not (= r 1)))
      (invalid "this backend decodes greedily; repeat_penalty must be 1" "repeat_penalty"))))

;;; ---------------------------------------------------------------------------
;;; Images

;; "WIDTHxHEIGHT", both multiples of 16 (the C ABI's limit), -> (w . h).
(define (check-image-size v)
  (unless (string? v)
    (invalid "size must be a string \"WIDTHxHEIGHT\"" "size"))
  (let ((parts (string-split v #\x)))
    (unless (= 2 (length parts))
      (invalid "size must be a string \"WIDTHxHEIGHT\"" "size"))
    (let ((w (string->number (car parts)))
          (h (string->number (cadr parts))))
      (unless (and (integer? w) (exact? w) (> w 0)
                   (integer? h) (exact? h) (> h 0))
        (invalid "size must be \"WIDTHxHEIGHT\" with positive integers" "size"))
      (unless (and (zero? (remainder w 16)) (zero? (remainder h 16)))
        (invalid "size must name multiples of 16" "size"))
      (cons w h))))

;; A /v1/images/generations body -> an alist with string keys:
;;   prompt, negative_prompt, n, width, height, steps, guidance_scale, seed
;; prompt is required; the rest default like the Guile binding's keywords.
;; steps, guidance_scale and seed are extensions (the OpenAI image schema has
;; no such fields, but a reproducible request needs them). response_format is
;; b64_json only: the reference server has no storage to hand out URLs from.
(define (parse-image-request body)
  "Turn an images-generations body into an alist.

The alist has string keys prompt, negative_prompt, n, width, height, steps,
guidance_scale and seed. prompt is required; size defaults to 512x512 and both
sides must be multiples of 16. steps, guidance_scale, seed and
negative_prompt are extensions of the OpenAI schema.

Args:
  body: the decoded request object.
Returns: an alist with string keys.
Example:
  (use-modules (sonicboom serve openai))
  (assoc-ref (parse-image-request '((\"prompt\" . \"cat\"))) \"width\")
  => 512
PyTorch: none"
  (unless (object? body)
    (invalid "the request body must be a JSON object" 'null))
  (let ((prompt (ref body "prompt")))
    (unless (and (string? prompt) (not (string-null? prompt)))
      (invalid "prompt is required" "prompt"))
    (let* ((negative (ref body "negative_prompt"))
           (n (match (ref body "n")
                (#f 1)
                (v (let ((n (positive-integer "n" v)))
                     (when (> n 10)
                       (invalid "n must be at most 10" "n"))
                     n))))
           (wh (check-image-size (or (ref body "size") "512x512")))
           (steps (match (ref body "steps")
                    (#f 50)
                    (v (positive-integer "steps" v))))
           (guidance (match (ref body "guidance_scale")
                       (#f 3.0)
                       (v (number-in "guidance_scale" v 0 20))))
           (seed (match (ref body "seed")
                   (#f 0)
                   (v (unless (and (integer? v) (exact? v))
                        (invalid "seed must be an integer" "seed"))
                      (abs v))))
           (format (ref body "response_format")))
      (unless (or (not format) (equal? format "b64_json"))
        (invalid "response_format must be b64_json (the only supported format)"
                 "response_format"))
      ;; negative_prompt is an extension too: the string reaches the model's
      ;; classifier-free guidance pass.
      (unless (or (not negative) (string? negative))
        (invalid "negative_prompt must be a string" "negative_prompt"))
      `(("prompt" . ,prompt)
        ("negative_prompt" . ,negative)
        ("n" . ,n)
        ("width" . ,(car wh))
        ("height" . ,(cdr wh))
        ("steps" . ,steps)
        ("guidance_scale" . ,guidance)
        ("seed" . ,seed)))))

;; {"created": ..., "data": [{"b64_json": ...}, ...]}
(define (images-object created b64s)
  "The images response object.

The data entries hold one base64 PNG each.

Args:
  created: the creation time.
  b64s: the base64 image strings.
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve openai))
  (vector-length (assoc-ref (images-object 1 '(\"abc\")) \"data\"))
  => 1
PyTorch: none"
  `(("created" . ,created)
    ("data" . ,(list->vector
                (map (lambda (b64) `(("b64_json" . ,b64))) b64s)))))

;; Base64 (RFC 4648, padded) of a bytevector. Guile 3.0.9 ships no (ice-9
;; base64), and the response body needs the standard alphabet.
(define %base64-alphabet
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")

(define (base64-encode-bytevector bv)
  "The base64 text of BV, with padding (RFC 4648).

Args:
  bv: a bytevector.
Returns: a string.
Example:
  (use-modules (sonicboom serve openai) (rnrs bytevectors))
  (base64-encode-bytevector (string->utf8 \"hi\"))
  => \"aGk=\"
PyTorch: none"
  (let* ((n (bytevector-length bv))
         (out (make-string (* 4 (quotient (+ n 2) 3)) #\=)))
    (let loop ((i 0) (o 0))
      (if (>= i n)
          out
          (let* ((b0 (bytevector-u8-ref bv i))
                 (b1 (if (< (+ i 1) n) (bytevector-u8-ref bv (+ i 1)) 0))
                 (b2 (if (< (+ i 2) n) (bytevector-u8-ref bv (+ i 2)) 0))
                 (v (+ (* b0 65536) (* b1 256) b2)))
            (string-set! out o
                         (string-ref %base64-alphabet (quotient v 262144)))
            (string-set! out (+ o 1)
                         (string-ref %base64-alphabet
                                     (quotient (remainder v 262144) 4096)))
            (when (< (+ i 1) n)
              (string-set! out (+ o 2)
                           (string-ref %base64-alphabet
                                       (quotient (remainder v 4096) 64))))
            (when (< (+ i 2) n)
              (string-set! out (+ o 3)
                           (string-ref %base64-alphabet (remainder v 64))))
            (loop (+ i 3) (+ o 4)))))))

;;; ---------------------------------------------------------------------------
;;; Responses

(define (new-completion-id)
  "A fresh completion id, starting with chatcmpl-.

Returns: a string.
Example:
  (use-modules (sonicboom serve openai) (srfi srfi-13))
  (string-prefix? \"chatcmpl-\" (new-completion-id))
  => #t
PyTorch: none"
  (string-append "chatcmpl-"
                 (number->string (random (expt 2 64) (random-state-from-platform)) 36)))

(define (usage-object result)
  "The usage counts of RESULT.

It holds prompt_tokens, completion_tokens, total_tokens and the cached prompt
tokens.

Args:
  result: a result.
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve openai))
  (let* ((engine (make-engine
                  (list (load-model (make-mock-backend) '((id . \"m\"))))))
         (result
          (engine-generate
           engine
           (make-chat-request #:model \"m\"
                              #:messages (list (list (cons \"role\" \"user\")
                                                     (cons \"content\" \"hi\"))))
           (lambda (kind . args) #t))))
    (positive? (assoc-ref (usage-object result) \"total_tokens\")))
  => #t
PyTorch: none"
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

(define* (completion-object id created model result #:key (seed 0))
  "The chat-completion object for RESULT.

A reply that consists only of tool calls has a null content, as the OpenAI
schema expects.

Besides the OpenAI fields it carries sonicboom_seed, the sampling seed the
reply was drawn with (0 for greedy), so a sampling client can replay it.

Args:
  id: the completion id.
  created: the creation time.
  model: the model id.
  result: a result.
  seed: keyword; the request's sampling seed (default 0).
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve openai))
  (let* ((engine (make-engine
                  (list (load-model (make-mock-backend) '((id . \"m\"))))))
         (result
          (engine-generate
           engine
           (make-chat-request #:model \"m\"
                              #:messages (list (list (cons \"role\" \"user\")
                                                     (cons \"content\" \"hi\"))))
           (lambda (kind . args) #t)))
         (choice (car (vector->list
                       (assoc-ref (completion-object \"id\" 0 \"m\" result)
                                  \"choices\")))))
    (assoc-ref choice \"finish_reason\"))
  => \"stop\"
PyTorch: none"
  (let ((calls (result-tool-calls result))
        (content (result-content result)))
    `(("id" . ,id)
      ("object" . "chat.completion")
      ("created" . ,created)
      ("model" . ,model)
      ("sonicboom_seed" . ,seed)
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

(define* (chunk-object id created model delta #:optional (finish-reason 'null)
                      #:key seed)
  "One streaming chat-completion chunk.

Args:
  id: the completion id.
  created: the creation time.
  model: the model id.
  delta: the delta alist.
  finish-reason: optional; the finish reason, null while streaming is under
    way (default null).
  seed: keyword; the request's sampling seed, carried on the first chunk
    (default: no sonicboom_seed field).
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve openai))
  (assoc-ref (chunk-object \"id\" 0 \"m\" (content-delta \"hi\")) \"object\")
  => \"chat.completion.chunk\"
PyTorch: none"
  `(("id" . ,id)
    ("object" . "chat.completion.chunk")
    ("created" . ,created)
    ("model" . ,model)
    ,@(if seed `(("sonicboom_seed" . ,seed)) '())
    ("choices" . #((("index" . 0)
                    ("delta" . ,delta)
                    ("logprobs" . null)
                    ("finish_reason" . ,finish-reason))))))

;; The last chunk when stream_options.include_usage is set: no choices.
(define (usage-chunk-object id created model result)
  "The last streaming chunk, carrying the usage counts.

It has no choices, as the OpenAI schema specifies for this chunk.

Args:
  id: the completion id.
  created: the creation time.
  model: the model id.
  result: a result.
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve openai))
  (let* ((engine (make-engine
                  (list (load-model (make-mock-backend) '((id . \"m\"))))))
         (result
          (engine-generate
           engine
           (make-chat-request #:model \"m\"
                              #:messages (list (list (cons \"role\" \"user\")
                                                     (cons \"content\" \"hi\"))))
           (lambda (kind . args) #t))))
    (vector-length (assoc-ref (usage-chunk-object \"id\" 0 \"m\" result)
                              \"choices\")))
  => 0
PyTorch: none"
  `(("id" . ,id)
    ("object" . "chat.completion.chunk")
    ("created" . ,created)
    ("model" . ,model)
    ("choices" . #())
    ("usage" . ,(usage-object result))))

(define (role-delta)
  "The first streaming delta: the assistant role.

Returns: a delta alist.
Example:
  (use-modules (sonicboom serve openai))
  (role-delta)
  => ((\"role\" . \"assistant\") (\"content\" . \"\"))
PyTorch: none"
  '(("role" . "assistant") ("content" . "")))
(define (content-delta text)
  "A streaming delta that carries reply text.

Args:
  text: the text.
Returns: a delta alist.
Example:
  (use-modules (sonicboom serve openai))
  (content-delta \"x\")
  => ((\"content\" . \"x\"))
PyTorch: none"
  `(("content" . ,text)))
(define (finish-delta)
  "The final streaming delta: it carries nothing.

Returns: the empty alist.
Example:
  (use-modules (sonicboom serve openai))
  (finish-delta)
  => ()
PyTorch: none"
  '())

;; One tool call as stream deltas: the header (id, type, name, empty
;; arguments), then the arguments in pieces of at most `piece' characters.
(define* (tool-call-deltas index id name args #:key (piece 8))
  "The streaming deltas of one tool call.

The first delta carries the call's header (index, id, type and name); then the
arguments JSON follows in pieces of at most PIECE characters.

Args:
  index: the call's index in the reply.
  id: the call id.
  name: the function name.
  args: the arguments, as JSON text.
  piece: keyword; the maximum piece length (default 8).
Returns: a list of delta alists.
Example:
  (use-modules (sonicboom serve openai))
  (length (tool-call-deltas 0 \"id\" \"f\" \"{}\" #:piece 3))
  => 2
PyTorch: none"
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
(define* (model-object id family created #:key (loaded #t))
  "One entry of the model list.

Besides the OpenAI fields it carries family, the chat-template family the
loaded model selected, so a client knows which prompt format and stop tokens
the model expects, and loaded, whether the model is on the device right now.
A model that has not been loaded yet has no family to report: family is null
until its first load.

Args:
  id: the model id.
  family: the model's family symbol, or #f when not known yet.
  created: the creation time.
  loaded: keyword; whether the model is resident on the device (default #t).
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve openai))
  (assoc-ref (model-object \"m\" 'gemma 0) \"family\")
  => \"gemma\"
PyTorch: none"
  `(("id" . ,id) ("object" . "model") ("created" . ,created)
    ("owned_by" . "sonicboom")
    ("family" . ,(if family (symbol->string family) 'null))
    ("loaded" . ,loaded)))

;; `models` is an (id . family) alist, in serving order.
(define (models-object models created)
  "The model-list response.

Args:
  models: an alist (id . family), in serving order.
  created: the creation time.
Returns: a JSON object alist.
Example:
  (use-modules (sonicboom serve openai))
  (vector-length (assoc-ref (models-object '((\"m\" . gemma)) 0) \"data\"))
  => 1
PyTorch: none"
  `(("object" . "list")
    ("data" . ,(list->vector
                (map (lambda (m) (model-object (car m) (cdr m) created)) models)))))
