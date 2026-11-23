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
;;;   GET  /v1/models, /v1/models/:id  chat models; loaded says which one is
;;;                                   on the device
;;;   POST /v1/chat/completions       stream=false: JSON; stream=true: SSE
;;;   POST /v1/images/generations     MOSS-Image, PNG as base64 (b64_json);
;;;                                   stream=true: SSE progress; the response
;;;                                   header sonicboom-job-id names the job
;;;   DELETE /sonicboom/jobs/:id      cancel a generation by job id
;;;   GET  /v1/ws/chat                WebSocket: a request per text message,
;;;                                   answered by chunk JSON messages, "[DONE]"
;;;   GET  /sonicboom/models          every served model and its residency
;;;   POST /sonicboom/models/:id/load, .../unload   queued device switches
;;;   GET  /sonicboom/stats           queue, sessions and backend counters
;;;   GET  /health
;;;
;;; One model is on the device at a time. Chat requests, images, loads and
;;; unloads share one FIFO queue; the job at the front makes its model the
;;; resident one (unloading the previous model) before it runs.
;;;
;;; Options
;;;   --host ADDR --port N            listen address (127.0.0.1:8000)
;;;   --backend real|mock             libsonicboom C ABI (real) or the
;;;                                   deterministic in-process mock (real)
;;;   --model FILE --model-id ID      the real backend's GGUF and served id
;;;                                   (models/qwen2.5-7b-instruct-1m-q4_k_m.gguf,
;;;                                   qwen2.5-7b)
;;;   --image-model DIR               MOSS-Image root, served as the
;;;                                   "moss-image" model and switched to on
;;;                                   an image request (models/MOSS-Image-SFT)
;;;   --queue N                       requests allowed to wait (8)
;;;   --max-sessions N --token-budget N   session pool, per model (1 for the
;;;                                   real backend, 4 for the mock; 32768)
;;;   --context-length N              of the loaded model (16384 real, 8192
;;;                                   mock)
;;;   --step-delay S --prefill-delay S    simulated device time, seconds
;;;   --runtime-dir DIR               Artanis' working directory
;;;                                   ($TMPDIR/sonicboom-serve-<pid>)

(use-modules (artanis artanis)
             ((artanis config) #:select (conf-set!))
             ((artanis env) #:select (%current-toplevel))
             (rnrs bytevectors)
             (srfi srfi-1)
             (srfi srfi-9)
             (ice-9 getopt-long)
             (ice-9 threads)
             (ice-9 binary-ports)
             (ice-9 match)
             (sonicboom error)
             (sonicboom image)
             ((sonicboom inference) #:select (device-trim!))
             (sonicboom serve backend)
             (sonicboom serve real-backend)
             (sonicboom serve mock-backend)
             (sonicboom serve engine)
             (sonicboom serve residency)
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
;; The job-cancel route is DELETE; Artanis' default leaves that method out.
(conf-set! '(server allowedmethods) '(HEAD GET POST PUT DELETE))
(conf-set! '(server websocket) #t)
;; Generation runs in runner threads; waiting requests hold one each.
(conf-set! '(server runners) (+ queue-limit 2))
(sse-install!)

;;; ---------------------------------------------------------------------------
;;; The catalog and the engine: one real Gemma 4 model loaded through
;;; libsonicboom's C ABI, or (--backend mock) the deterministic in-process mock
;;; — one model per chat template family, no GGUF needed. Both go through the
;;; same backend protocol, so nothing below this point knows which one is in
;;; use.
;;;
;;; The device holds one model at a time. The catalog lists every model the
;;; server can serve; the residency manager loads a model when a request for it
;;; reaches the front of the engine's queue, unloading the resident one first.
;;; Only the first chat model is loaded at startup, exactly as a single-model
;;; server always loaded its model up front; the others are listed as unloaded
;;; and load on demand.

(define backend-name (option 'backend "real"))

(define backend
  (case (string->symbol backend-name)
    ((real) (make-real-backend))
    ((mock) (make-mock-backend))
    (else (error "unknown backend (expected real or mock):" backend-name))))

(define max-sessions
  (option 'max-sessions (if (equal? backend-name "real") 1 4)))
(define token-budget (option 'token-budget 32768))

(define (mock-config id family)
  `((id . ,id) (family . ,family)
    (context-length . ,(option 'context-length 8192))
    (step-delay . ,(option 'step-delay 0))
    (prefill-delay . ,(option 'prefill-delay 0))))

(define model-path (option 'model "models/qwen2.5-7b-instruct-1m-q4_k_m.gguf"))
(define model-id (option 'model-id "qwen2.5-7b"))

(define image-model-path (option 'image-model "models/MOSS-Image-SFT"))
(define image-model-id "moss-image")

(define catalog
  (append
   (case (string->symbol backend-name)
     ((mock) (list (make-residency-entry "sonicboom-mock-gemma" 'chat
                                         (mock-config "sonicboom-mock-gemma" 'gemma))
                   (make-residency-entry "sonicboom-mock-qwen" 'chat
                                         (mock-config "sonicboom-mock-qwen" 'qwen))))
     (else
      (list (make-residency-entry
             model-id 'chat
             `((id . ,model-id) (path . ,model-path)
               (context-length . ,(option 'context-length 16384)))))))
   (list (make-residency-entry image-model-id 'image image-model-path
                               #:arch "moss-image"))))

(define engine
  (make-engine '()
               #:queue-limit queue-limit
               #:ids (map residency-entry-id catalog)
               ;; The device is idle when this runs: make the request's model
               ;; the one on it.
               #:prepare (lambda (request)
                           (residency-ensure! residency (chat-request-model request)))))

(define residency
  (make-residency engine catalog
                  #:backend backend
                  #:max-sessions max-sessions
                  #:token-budget token-budget
                  #:image-loader load-image-model
                  #:image-unloader free-image-model!
                  #:image-arch image-model-arch
                  ;; Only a real device has buffers to return; the mock has no
                  ;; library to call.
                  #:trim (if (equal? backend-name "real")
                             device-trim!
                             (lambda () #f))))

;; Like a single-model server, the first chat model is loaded before the
;; server serves anything; a failed load fails startup as it always did.
(residency-load! residency
                 (or (find (lambda (id) (eq? 'chat (residency-kind residency id)))
                           (residency-ids residency))
                     (error "the catalog has no chat model")))

(define start-time (car (gettimeofday)))
(define (now) (car (gettimeofday)))

;;; ---------------------------------------------------------------------------
;;; Generation jobs
;;;
;;; An image generation can be long, and a client may leave (an SSE write then
;;; fails) or cancel it by id. A job is registered before it takes its place in
;;; the queue, so the id — returned in the response header sonicboom-job-id —
;;; is known from the start. Cancelling sets a flag that the generation's
;;; progress callback reads after every denoising step; the C ABI then stops
;;; the generation and releases the buffers it had allocated. A job is
;;; unregistered when its request ends, so a later DELETE is a 404.

(define jobs (make-hash-table))
(define jobs-lock (make-mutex))

(define-record-type <job>
  (make-job* id cancelled?)
  job?
  (id job-id)
  (cancelled? job-cancelled? set-job-cancelled?!))

(define (new-job-id)
  (format #f "job-~8,'0x~8,'0x"
          (random (expt 2 32)) (random (expt 2 32))))

(define (job-register!)
  (let ((job (make-job* (new-job-id) #f)))
    (with-mutex jobs-lock (hash-set! jobs (job-id job) job))
    job))

(define (job-unregister! job)
  (with-mutex jobs-lock (hash-remove! jobs (job-id job))))

;; The job with this id, marked cancelled, or #f when there is no such job.
(define (job-cancel! id)
  (with-mutex jobs-lock
    (match (hash-ref jobs id)
      (#f #f)
      (job (set-job-cancelled?! job #t) job))))

(define (job-cancelled-now? job)
  (with-mutex jobs-lock (job-cancelled? job)))

(define (cancelled-condition? key args)
  (and (eq? key '%exception)
       (let ((c (car args)))
         (and (sonicboom-error? c)
              (eq? 'cancelled (sonicboom-error-kind c))))))

;;; ---------------------------------------------------------------------------
;;; The image model joins the same catalog and queue: a generation switches the
;;; device to MOSS-Image (unloading the chat model first) and switches back on
;;; the next chat request.

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

;; REQUEST is the alist parse-image-request returns; the caller made the image
;; model resident. The result is the list of base64 PNGs for the response's
;; data array. PROGRESS (image step total) is the generation's progress
;; callback, called after each denoising step; #f means no callback.
(define* (render-images request #:key (progress #f))
  (let ((images (generate-images
                 (residency-model residency image-model-id)
                 #:prompt (assoc-ref request "prompt")
                 #:negative (assoc-ref request "negative_prompt")
                 #:width (assoc-ref request "width")
                 #:height (assoc-ref request "height")
                 #:steps (assoc-ref request "steps")
                 #:guidance (assoc-ref request "guidance_scale")
                 #:seed (assoc-ref request "seed")
                 #:batch (assoc-ref request "n")
                 #:progress progress)))
    (map (lambda (image)
           (base64-encode-bytevector
            (png-bytes (car image) (cadr image) (caddr image))))
         images)))

;; One SSE progress event: the step just completed, the total, and the image
;; of the batch it belongs to.
(define (progress-object image step total)
  `(("step" . ,step) ("total" . ,total) ("image" . ,image)))

;; The generation's progress callback. It returns #f — stopping the generation
;; after the current step — when the job was cancelled by DELETE, or when SEND
;; (the SSE writer) failed because the client is gone; otherwise a streaming
;; request sends one event per denoising step. SEND is #f for a non-streaming
;; request, whose cancellation is only by id.
(define (image-progress job send)
  (lambda (image step total)
    (and (not (job-cancelled-now? job))
         (or (not send)
             (send (json-ascii-string (progress-object image step total)))))))

;; The body of a streaming image request, run by the SSE producer. The final
;; event is the same images object the non-stream reply holds (the same PNGs);
;; a cancellation ends it with {"id", "status": "cancelled"} instead, and
;; "[DONE]" ends the stream either way.
(define (stream-images job request send)
  (catch #t
    (lambda ()
      (send (json-ascii-string
             (images-object
              (now)
              (run-job
               (lambda ()
                 (residency-ensure! residency image-model-id)
                 (render-images request
                                #:progress (image-progress job send))))))))
    (lambda (key . args)
      (send (json-ascii-string
             (if (cancelled-condition? key args)
                 `(("id" . ,(job-id job)) ("status" . "cancelled"))
                 (error-object-of key args))))))
  (send "[DONE]"))

;;; ---------------------------------------------------------------------------
;;; Helpers

(define* (json-emit status obj #:key (headers '()))
  (response-emit (scm->json-string obj) #:status status
                 #:headers (append headers '((content-type . (application/json))))))

;; An SSE event body must be one byte per character: the event writer counts
;; characters for its chunk headers and the socket port it writes through is
;; ISO-8859-1. scm->json-string with #:unicode escapes most non-ASCII
;; characters but leaves Latin-1 raw, so escape everything above ASCII.
(define (json-ascii-string obj)
  (call-with-output-string
   (lambda (out)
     (string-for-each
      (lambda (ch)
        (let ((n (char->integer ch)))
          (cond
           ((< n 128) (write-char ch out))
           ((< n #x10000) (format out "\\u~4,'0x" n))
           (else
            (let ((m (- n #x10000)))
              (format out "\\u~4,'0x\\u~4,'0x"
                      (+ #xd800 (quotient m #x400))
                      (+ #xdc00 (remainder m #x400))))))))
      (scm->json-string obj #:unicode #t)))))

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
    (('job-not-found id)
     (list 404 "invalid_request_error"
           (format #f "The job `~a' does not exist" id) 'null "job_not_found"))
    (('context-length-exceeded n ctx)
     (list 400 "invalid_request_error"
           (format #f "the prompt has ~a tokens, the context length is ~a" n ctx)
           "messages" "context_length_exceeded"))
    (('invalid-constraint param message)
     (list 400 "invalid_request_error" message param "invalid_constraint"))
    ;; A Guile 3 condition raised by a binding module. `catch #t` reports the
    ;; condition itself with the key `%exception`. A &sonicboom-error carries
    ;; the C ABI's message (a failed model load, a generation failure). A
    ;; generation the client cancelled is not a server error: it is a 409.
    (('%exception c)
     (cond
      ((not (sonicboom-error? c))
       (list 500 "server_error" (format #f "internal error: ~a" c) 'null 'null))
      ((eq? 'cancelled (sonicboom-error-kind c))
       (list 409 "invalid_request_error" "the generation was cancelled"
             'null "cancelled"))
      (else
       (let ((message (sonicboom-error-message c)))
         (list 500 "server_error"
               (if (string-null? message) "SonicBoom error" message)
               'null 'null)))))
    (_ (list 500 "server_error" (format #f "~a: ~s" key args) 'null 'null))))

(define (error-object-of key args)
  (match (classify key args)
    ((status type message param code) (openai-error-object type message param code))))

;; Parse a request body and take a place in the queue.
;; -> (values request stream? include-usage? ticket)
(define (admit body)
  (call-with-values (lambda () (parse-chat-request body))
    (lambda (request stream? include-usage?)
      ;; The catalog decides whether the id is served, not the resident model:
      ;; a model that is not on the device right now is loaded when the
      ;; request reaches the front of the queue.
      (unless (eq? 'chat (residency-kind residency (chat-request-model request)))
        (throw 'model-not-found (chat-request-model request)))
      ;; A backend that cannot sample gets an explicit sampling request
      ;; rejected as a 400, not silently decoded greedily. Both shipped
      ;; backends sample; this guards a future one that does not.
      (unless (backend-sampling? backend)
        (check-greedy-body body))
      (values request stream? include-usage? (engine-admit! engine)))))

;; Run an admitted request, sending its chunks through send-obj (returns #f
;; once the client is gone). Ends with "[DONE]" through send-text.
(define (stream-chunks ticket request include-usage? send-obj send-text)
  (let ((id (new-completion-id))
        (created (now))
        (model (chat-request-model request)))
    (define (chunk delta . finish)
      (send-obj (apply chunk-object id created model delta finish)))
    ;; The first chunk carries the request's seed (sonicboom_seed) so a
    ;; streaming client can replay the reply.
    (if (not (send-obj (chunk-object id created model (role-delta)
                                     #:seed (chat-request-seed request))))
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

;; Run an arbitrary job (an image generation, a load or an unload) behind the
;; admitted chat requests, waiting for its turn in a runner thread. NOTE: the
;; same as chat: no dynamic-wind around call-with-runner. A failure drops the
;; ticket here; engine-run-job drops it when the job returns.
(define (run-job thunk)
  (let ((ticket (engine-admit! engine)))
    (catch #t
      (lambda ()
        (call-with-runner
         (lambda () (engine-run-job engine ticket thunk))
         #:timeout 0))
      (lambda (key . args)
        (engine-drop! engine ticket)
        (apply throw key args)))))

(define (alist->json alist)
  (map (match-lambda
         ((k . (? pair? v)) (cons (symbol->string k) (alist->json v)))
         ((k . v) (cons (symbol->string k) v)))
       alist))

;;; ---------------------------------------------------------------------------
;;; Routes

(get "/health" (lambda (rc) "ok"))

(get "/v1/models" #:mime 'json
  (lambda (rc)
    (:mime rc
           `(("object" . "list")
             ("data"
              . ,(list->vector
                  (filter-map
                   (lambda (id)
                     (and (eq? 'chat (residency-kind residency id))
                          (model-object id (residency-family residency id)
                                        start-time
                                        #:loaded (residency-loaded? residency id))))
                   (residency-ids residency))))))))

(get "/v1/models/:id"
  (lambda (rc)
    (let ((id (params rc "id")))
      (if (eq? 'chat (residency-kind residency id))
          (json-emit 200 (model-object id (residency-family residency id)
                                       start-time
                                       #:loaded (residency-loaded? residency id)))
          (apply error-emit (classify 'model-not-found (list id)))))))

;; Every served model, chat and image, with its residency state. Only one of
;; them is loaded at a time (the one the last job left on the device).
(get "/sonicboom/models" #:mime 'json
  (lambda (rc)
    (:mime rc
           `(("object" . "list")
             ("data"
              . ,(list->vector
                  (map (lambda (id)
                         (alist->json (residency-info residency id)))
                       (residency-ids residency))))))))

;; Load and unload are ordinary jobs: they queue behind running work and
;; switch the device only when it is idle. A load switches away from the
;; resident model, exactly as a request for this model would.
(post "/sonicboom/models/:id/load"
  (lambda (rc)
    (let ((id (params rc "id")))
      (catch #t
        (lambda ()
          (unless (residency-known? residency id)
            (throw 'model-not-found id))
          (run-job (lambda () (residency-load! residency id)))
          (json-emit 200 (alist->json (residency-info residency id))))
        (lambda (key . args)
          (apply error-emit (classify key args)))))))

(post "/sonicboom/models/:id/unload"
  (lambda (rc)
    (let ((id (params rc "id")))
      (catch #t
        (lambda ()
          (unless (residency-known? residency id)
            (throw 'model-not-found id))
          (run-job (lambda () (residency-unload! residency id)))
          (json-emit 200 (alist->json (residency-info residency id))))
        (lambda (key . args)
          (apply error-emit (classify key args)))))))

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
                                  (lambda (obj) (send (json-ascii-string obj)))
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
                  (json-emit 200 (completion-object
                                  (new-completion-id) (now)
                                  (chat-request-model request) result
                                  #:seed (chat-request-seed request))))))))
      (lambda (key . args)
        (apply error-emit (classify key args))))))

;; Image generation is blocking and long, so it runs in a runner thread like a
;; chat completion (same NOTE as above: no dynamic-wind around the runner),
;; and it takes the same queue: the job switches the device to the image model
;; before it renders and leaves it resident until another job switches away.
;;
;; stream=false answers with the images object, as before, plus the
;; sonicboom-job-id header. stream=true answers with SSE: one
;; {"step","total","image"} event per denoising step, then the images object,
;; then "[DONE]". Either way the job can be cancelled by id; a cancellation
;; answers 409 (non-stream) or {"id","status":"cancelled"} (stream).
(post "/v1/images/generations"
  (lambda (rc)
    (catch #t
      (lambda ()
        (let* ((body (read-json-body rc))
               (stream? (eq? #t (assoc-ref body "stream")))
               (request (parse-image-request body))
               (job (job-register!))
               (headers `((sonicboom-job-id . ,(job-id job)))))
          (if stream?
              (sse-response
               (lambda (send)
                 ;; The producer always runs, even when the client is already
                 ;; gone, so the job is unregistered either way.
                 (dynamic-wind
                   (lambda () #t)
                   (lambda () (stream-images job request send))
                   (lambda () (job-unregister! job))))
               #:headers headers)
              (catch #t
                (lambda ()
                  (let ((b64s (run-job
                               (lambda ()
                                 (residency-ensure! residency image-model-id)
                                 (render-images request
                                                #:progress
                                                (image-progress job #f))))))
                    (job-unregister! job)
                    (json-emit 200 (images-object (now) b64s)
                               #:headers headers)))
                (lambda (key . args)
                  (job-unregister! job)
                  (apply throw key args))))))
      (lambda (key . args)
        (apply error-emit (classify key args))))))

;; Cancel a running or queued generation. The device work stops after the
;; current denoising step and its buffers are released; the response names the
;; job and its status. An unknown id (a finished job included) is a 404.
;; NOTE: Artanis exports the DELETE route as page-delete; its usual name,
;;       delete, is the (srfi srfi-1) procedure here.
(page-delete "/sonicboom/jobs/:id"
  (lambda (rc)
    (let ((id (params rc "id")))
      (if (job-cancel! id)
          (json-emit 200 `(("id" . ,id) ("status" . "cancelled")))
          (apply error-emit (classify 'job-not-found (list id)))))))

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

(get "/sonicboom/stats" #:mime 'json
  (lambda (rc)
    (:mime rc
           `(("engine" . ,(alist->json (engine-stats engine)))
             ("models"
              . ,(map (lambda (id)
                        (let ((m (engine-model engine id)))
                          (cons id
                                (if m
                                    `(("sessions" . ,(alist->json
                                                      (pool-stats (engine-pool engine id))))
                                      ("backend" . ,(alist->json (backend-stats m))))
                                    ;; The residency manager unloaded it; its
                                    ;; counters went with the model.
                                    '(("loaded" . #f))))))
                      (engine-model-ids engine)))))))

(run #:host (option 'host "127.0.0.1") #:port (option 'port 8000))
