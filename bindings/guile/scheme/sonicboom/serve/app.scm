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
;;;   GET  /v1/ws/chat                WebSocket: a request per text message,
;;;                                   answered by chunk JSON messages, "[DONE]"
;;;   GET  /sonicboom/stats           queue, sessions and backend counters
;;;   GET  /health
;;;
;;; Options
;;;   --host ADDR --port N            listen address (127.0.0.1:8000)
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
             (ice-9 match)
             (sonicboom serve backend)
             (sonicboom serve real-backend)
             (sonicboom serve engine)
             (sonicboom serve sessions)
             (sonicboom serve chat-template)
             (sonicboom serve openai)
             (sonicboom serve sse))

;;; ---------------------------------------------------------------------------
;;; Options

(define options
  (getopt-long (command-line)
               '((host (value #t)) (port (value #t)) (model (value #t))
                 (model-id (value #t)) (queue (value #t))
                 (max-sessions (value #t)) (token-budget (value #t))
                 (context-length (value #t)) (step-delay (value #t))
                 (prefill-delay (value #t)) (runtime-dir (value #t)))))

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
;;; The engine: one real Gemma 4 model loaded through libsonicboom's C ABI.

(define backend (make-real-backend))
(define model-path (option 'model "models/gemma-4-E4B-it-Q3_K_M.gguf"))
(define model-id (option 'model-id "gemma-4-E4B-it-Q3_K_M"))
(define model
  (load-model backend `((id . ,model-id) (path . ,model-path)
                        (context-length . ,(option 'context-length 512)))))

(define engine
  (make-engine (list model)
               #:queue-limit queue-limit
               #:max-sessions (option 'max-sessions 4)
               #:token-budget (option 'token-budget 32768)))

(define start-time (car (gettimeofday)))
(define (now) (car (gettimeofday)))

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
        ;; The libsonicboom backend decodes greedily only; an explicit
        ;; sampling request is a 400, not a silent fallback.
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
