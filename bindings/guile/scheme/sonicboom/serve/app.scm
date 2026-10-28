#! /usr/bin/env guile
!#
;;; SonicBoom — inference service: the OpenAI-compatible HTTP app (Artanis)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A single-file GNU Artanis app, written like examples/test.scm and
;;; examples/ENTRY.websocket of Artanis: (init-server), routes with get/post,
;;; #:mime 'json with :mime for JSON, response-emit for other statuses,
;;; #:websocket 'raw with ws-send for WebSocket, call-with-runner for the
;;; blocking work, then (run). It is a script, not a module:
;;;
;;;   guile -L bindings/guile/scheme \
;;;         bindings/guile/scheme/sonicboom/serve/app.scm --port 8000
;;;
;;; Only the HTTP layer is here. The backend protocol, the mock backend, the
;;; session pool, the queue and the OpenAI format are the modules next to
;;; this file; they don't depend on Artanis, except (sonicboom serve sse).
;;;
;;; Endpoints
;;;   GET  /v1/models, /v1/models/:id
;;;   POST /v1/chat/completions       stream=false: JSON; stream=true: SSE
;;;   POST /v1/images/generations     MOSS-Image, PNG as base64 (b64_json)
;;;   GET  /v1/ws/chat                WebSocket: a request per text message,
;;;                                   answered by chunk JSON messages, "[DONE]"
;;;   GET  /sonicboom/stats           queue, sessions and backend counters
;;;   GET  /health
;;;
;;; Options
;;;   --host ADDR --port N            listen address (127.0.0.1:8000)
;;;   --backend real|mock             libsonicboom C ABI (real) or the
;;;                                   deterministic in-process mock (real)
;;;   --image-model DIR               MOSS-Image root, loaded on the first
;;;                                   image request (models/MOSS-Image-SFT)
;;;   --queue N                       requests allowed to wait (8)
;;;   --max-sessions N --token-budget N   session pool, per model (4, 32768)
;;;   --context-length N              of the mock models (8192)
;;;   --step-delay S --prefill-delay S    simulated device time, seconds
;;;   --runtime-dir DIR               Artanis' working directory
;;;                                   ($TMPDIR/sonicboom-serve-<pid>)

(use-modules (artanis artanis)
             ((artanis config) #:select (conf-set!))
             ((artanis env) #:select (%current-toplevel))
             (rnrs bytevectors)
             (srfi srfi-1)
             (ice-9 getopt-long)
             (ice-9 threads)
             (ice-9 binary-ports)
             (ice-9 match)
             (sonicboom error)
             (sonicboom image)
             (sonicboom serve backend)
             (sonicboom serve real-backend)
             (sonicboom serve mock-backend)
             (sonicboom serve engine)
             (sonicboom serve sessions)
             (sonicboom serve chat-template)
             (sonicboom serve openai)
             (sonicboom serve sse))

;;; ---------------------------------------------------------------------------
;;; Options

(define options
  (getopt-long (command-line)
               '((host (value #t)) (port (value #t)) (backend (value #t))
                 (model (value #t)) (model-id (value #t)) (queue (value #t))
                 (max-sessions (value #t)) (token-budget (value #t))
                 (context-length (value #t)) (step-delay (value #t))
                 (prefill-delay (value #t)) (image-model (value #t))
                 (runtime-dir (value #t)))))

(define (option key default)
  (match (option-ref options key #f)
    (#f default)
    (s (if (number? default) (string->number s) s))))

(define queue-limit (option 'queue 8))

;; Artanis looks for its working directory (sys/i18n, ...) from the toplevel
;; of an `art create' project; a single-file app has none, so give it one.
(define runtime-dir
  (option 'runtime-dir
          (format #f "~a/sonicboom-serve-~a" (or (getenv "TMPDIR") "/tmp") (getpid))))
(system* "mkdir" "-p" (string-append runtime-dir "/sys/i18n"))
(%current-toplevel runtime-dir)

(init-server #:statics '())
(conf-set! '(server websocket) #t)
;; Generation runs in runner threads; waiting requests hold one each.
(conf-set! '(server runners) (+ queue-limit 2))
(sse-install!)

;;; ---------------------------------------------------------------------------
;;; The engine: one real Gemma 4 model loaded through libsonicboom's C ABI, or
;;; (--backend mock) the deterministic in-process mock — one model per chat
;;; template family, no GGUF needed. Both go through the same backend protocol,
;;; so nothing below this point knows which one is in use.

(define backend-name (option 'backend "real"))

(define backend
  (case (string->symbol backend-name)
    ((real) (make-real-backend))
    ((mock) (make-mock-backend))
    (else (error "unknown backend (expected real or mock):" backend-name))))

(define (mock-model id family)
  (load-model backend `((id . ,id) (family . ,family)
                        (context-length . ,(option 'context-length 8192))
                        (step-delay . ,(option 'step-delay 0))
                        (prefill-delay . ,(option 'prefill-delay 0)))))

(define model-path (option 'model "models/gemma-4-E4B-it-Q3_K_M.gguf"))
(define model-id (option 'model-id "gemma-4-E4B-it-Q3_K_M"))

(define models
  (case (string->symbol backend-name)
    ((mock) (list (mock-model "sonicboom-mock-gemma" 'gemma)
                  (mock-model "sonicboom-mock-qwen" 'qwen)))
    (else
     (list (load-model backend
                       `((id . ,model-id) (path . ,model-path)
                         (context-length . ,(option 'context-length 512))))))))

(define engine
  (make-engine models
               #:queue-limit queue-limit
               #:max-sessions (option 'max-sessions 4)
               #:token-budget (option 'token-budget 32768)))

(define start-time (car (gettimeofday)))
(define (now) (car (gettimeofday)))

;;; ---------------------------------------------------------------------------
;;; The image model: loaded on the first image request, so a chat-only server
;;; never pays for the MOSS-Image weights or their device memory. There is no
;;; session pool for it: one generation at a time, serialized on this mutex.

(define image-model-path (option 'image-model "models/MOSS-Image-SFT"))
(define image-model-lock (make-mutex))
(define image-model #f)

(define (image-pipeline)
  (lock-mutex image-model-lock)
  (dynamic-wind
    (lambda () #t)
    (lambda ()
      (unless image-model
        (set! image-model (load-image-model image-model-path)))
      image-model)
    (lambda () (unlock-mutex image-model-lock))))

;; The C ABI's PNG writer takes a path, so encode through a temporary file and
;; read the bytes back. Each request has its own file name; it is removed even
;; when the write fails.
(define (png-bytes width height rgb)
  (let ((path (format #f "~a/image-~a-~a.png" runtime-dir (getpid)
                      (random 1000000000))))
    (dynamic-wind
      (lambda () #t)
      (lambda ()
        (write-png path rgb width height)
        (call-with-input-file path get-bytevector-all))
      (lambda ()
        (catch #t (lambda () (delete-file path)) (lambda _ #f))))))

;; One generation at a time: the pipeline's device buffers are reused in place
;; from request to request, so concurrent generations on it are not safe.
(define image-run-lock (make-mutex))

;; REQUEST is the alist parse-image-request returns; the result is the list of
;; base64 PNGs for the response's data array.
(define (render-images request)
  (lock-mutex image-run-lock)
  (dynamic-wind
    (lambda () #t)
    (lambda ()
      (let ((images (generate-images
                     (image-pipeline)
                     #:prompt (assoc-ref request "prompt")
                     #:negative (assoc-ref request "negative_prompt")
                     #:width (assoc-ref request "width")
                     #:height (assoc-ref request "height")
                     #:steps (assoc-ref request "steps")
                     #:guidance (assoc-ref request "guidance_scale")
                     #:seed (assoc-ref request "seed")
                     #:batch (assoc-ref request "n"))))
        (map (lambda (image)
               (base64-encode-bytevector
                (png-bytes (car image) (cadr image) (caddr image))))
             images)))
    (lambda () (unlock-mutex image-run-lock))))

;;; ---------------------------------------------------------------------------
;;; Helpers

(define (json-emit status obj)
  (response-emit (scm->json-string obj) #:status status
                 #:headers '((content-type . (application/json)))))

(define (error-emit status type message param code)
  (json-emit status (openai-error-object type message param code)))

(define (read-json-body rc)
  (let ((body (rc-body rc)))
    (unless body
      (raise-openai-error 400 "invalid_request_error" "the request body is empty"))
    (catch #t
      (lambda () (json-normalize (json-string->scm (utf8->string body))))
      (lambda _
        (raise-openai-error 400 "invalid_request_error" "the request body is not valid JSON")))))

;; Exceptions from parsing, admission and generation -> (status type message
;; param code).
(define (classify key args)
  (match (cons key args)
    (('openai-error status type message param) (list status type message param 'null))
    (('queue-full waiting)
     (list 429 "rate_limit_error"
           (format #f "the request queue is full (~a waiting)" waiting) 'null "queue_full"))
    (('model-not-found id)
     (list 404 "invalid_request_error"
           (format #f "The model `~a' does not exist" id) "model" "model_not_found"))
    (('context-length-exceeded n ctx)
     (list 400 "invalid_request_error"
           (format #f "the prompt has ~a tokens, the context length is ~a" n ctx)
           "messages" "context_length_exceeded"))
    (('invalid-constraint param message)
     (list 400 "invalid_request_error" message param "invalid_constraint"))
    ;; A Guile 3 condition raised by a binding module. `catch #t` reports the
    ;; condition itself with the key `%exception`. A &sonicboom-error carries
    ;; the C ABI's message (a failed model load, a generation failure, a
    ;; cancelled generation; the server never cancels one).
    (('%exception c)
     (if (sonicboom-error? c)
         (let ((message (sonicboom-error-message c)))
           (list 500 "server_error"
                 (if (string-null? message) "SonicBoom error" message)
                 'null 'null))
         (list 500 "server_error" (format #f "internal error: ~a" c)
               'null 'null)))
    (_ (list 500 "server_error" (format #f "~a: ~s" key args) 'null 'null))))

(define (error-object-of key args)
  (match (classify key args)
    ((status type message param code) (openai-error-object type message param code))))

;; Parse a request body and take a place in the queue.
;; -> (values request stream? include-usage? ticket)
(define (admit body)
  (call-with-values (lambda () (parse-chat-request body))
    (lambda (request stream? include-usage?)
      (let ((model (engine-model engine (chat-request-model request))))
        (unless model
          (throw 'model-not-found (chat-request-model request)))
        ;; A backend that cannot sample gets an explicit sampling request
        ;; rejected as a 400, not silently decoded greedily. Both shipped
        ;; backends sample; this guards a future one that does not.
        (unless (model-sampling? model)
          (check-greedy-body body))
        (values request stream? include-usage? (engine-admit! engine))))))

;; Run an admitted request, sending its chunks through send-obj (returns #f
;; once the client is gone). Ends with "[DONE]" through send-text.
(define (stream-chunks ticket request include-usage? send-obj send-text)
  (let ((id (new-completion-id))
        (created (now))
        (model (chat-request-model request)))
    (define (chunk delta . finish)
      (send-obj (apply chunk-object id created model delta finish)))
    (if (not (chunk (role-delta)))
        (engine-drop! engine ticket)    ; gone before the first chunk
        (catch #t
          (lambda ()
            (let ((result
                   (engine-run engine ticket request
                               (lambda (kind . args)
                                 (match (cons kind args)
                                   (('content text) (chunk (content-delta text)))
                                   (('tool-call index call-id name arguments)
                                    (every chunk (tool-call-deltas index call-id name
                                                                   arguments))))))))
              (unless (equal? (result-finish-reason result) "cancelled")
                (and (chunk (finish-delta) (result-finish-reason result))
                     (or (not include-usage?)
                         (send-obj (usage-chunk-object id created model result)))
                     (send-text "[DONE]")))))
          (lambda (key . args)
            (engine-drop! engine ticket)
            (and (send-obj (error-object-of key args))
                 (send-text "[DONE]")))))))

;;; ---------------------------------------------------------------------------
;;; Routes

(get "/health" (lambda (rc) "ok"))

(get "/v1/models" #:mime 'json
  (lambda (rc)
    (:mime rc (models-object (engine-model-families engine) start-time))))

(get "/v1/models/:id"
  (lambda (rc)
    (let* ((id (params rc "id"))
           (model (engine-model engine id)))
      (if model
          (json-emit 200 (model-object id (model-family model) start-time))
          (apply error-emit (classify 'model-not-found (list id)))))))

(post "/v1/chat/completions"
  (lambda (rc)
    (catch #t
      (lambda ()
        (call-with-values (lambda () (admit (read-json-body rc)))
          (lambda (request stream? include-usage? ticket)
            (if stream?
                (sse-response
                 (lambda (send)
                   (stream-chunks ticket request include-usage?
                                  (lambda (obj) (send (scm->json-string obj)))
                                  send)))
                ;; NOTE: No dynamic-wind around call-with-runner: the task
                ;;       is suspended by aborting to the Ragnarok scheduler,
                ;;       which would run the after thunk while the runner
                ;;       still works. engine-run leaves the queue itself.
                (let ((result (catch #t
                                (lambda ()
                                  (call-with-runner
                                   (lambda () (engine-run engine ticket request
                                                          (lambda _ #t)))
                                   #:timeout 0))
                                (lambda (key . args)
                                  (engine-drop! engine ticket)
                                  (apply throw key args)))))
                  (json-emit 200 (completion-object (new-completion-id) (now)
                                                    (chat-request-model request)
                                                    result)))))))
      (lambda (key . args)
        (apply error-emit (classify key args))))))

;; Image generation is blocking and long, so it runs in a runner thread like a
;; chat completion (same NOTE as above: no dynamic-wind). It does not use the
;; engine's queue or session pool; render-images serializes the generations.
(post "/v1/images/generations"
  (lambda (rc)
    (catch #t
      (lambda ()
        (let ((request (parse-image-request (read-json-body rc))))
          (json-emit 200
                     (images-object (now)
                                    (call-with-runner
                                     (lambda () (render-images request))
                                     #:timeout 0)))))
      (lambda (key . args)
        (apply error-emit (classify key args))))))

;; Threads for WebSocket replies run in the dynamic state of the app at load
;; time, like Artanis' runners, not in the one of the Ragnarok task that
;; spawns them (whose port waiters would suspend the task).
(define clean-dynamic-state (current-dynamic-state))

(define (spawn thunk)
  (call-with-new-thread
   (lambda ()
     (with-dynamic-state
      clean-dynamic-state
      (lambda ()
        (catch #t thunk
          (lambda (key . args)
            (format (current-error-port) "ws: reply failed: ~a ~s~%" key args))))))))

;; One chat request per text message; the reply is the stream of chunk
;; objects as text messages, then "[DONE]". A bad request gets an error
;; object (and no "[DONE]").
;; NOTE: Not call-with-runner here: the task of a connection is the only
;;       writer of its socket, and it writes the messages queued by ws-send
;;       only while it waits for the peer. So on-message returns at once and
;;       the reply is generated in a thread; the task then streams the queued
;;       chunks, and sees the peer go away (ws-send returns 'closed).
(get "/v1/ws/chat" #:websocket 'raw #:timeout 0
  (lambda (rc)
    (lambda (conn msg)
      (define (send-text s) (eq? #t (ws-send conn s)))
      (define (send-obj o) (send-text (scm->json-string o)))
      (catch #t
        (lambda ()
          (call-with-values
              (lambda ()
                (admit (catch #t
                         (lambda ()
                           (json-normalize (json-string->scm (ws-message-text msg))))
                         (lambda _
                           (raise-openai-error 400 "invalid_request_error"
                                               "the message is not valid JSON")))))
            (lambda (request stream? include-usage? ticket)
              (spawn
               (lambda () (stream-chunks ticket request include-usage? send-obj send-text))))))
        (lambda (key . args)
          (send-obj (error-object-of key args)))))))

(define (alist->json alist)
  (map (match-lambda
         ((k . (? pair? v)) (cons (symbol->string k) (alist->json v)))
         ((k . v) (cons (symbol->string k) v)))
       alist))

(get "/sonicboom/stats" #:mime 'json
  (lambda (rc)
    (:mime rc
           `(("engine" . ,(alist->json (engine-stats engine)))
             ("models"
              . ,(map (lambda (id)
                        (let ((m (engine-model engine id)))
                          (cons id
                                `(("sessions" . ,(alist->json (pool-stats (engine-pool engine id))))
                                  ("backend" . ,(alist->json (backend-stats m)))))))
                      (engine-model-ids engine)))))))

(run #:host (option 'host "127.0.0.1") #:port (option 'port 8000))
