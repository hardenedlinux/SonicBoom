;;; SonicBoom — inference service: the request queue and the generation loop
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Independent of HTTP: the Artanis app (and anything else) hands a parsed
;;; chat request to the engine and gets events back.
;;;
;;; Queue   one device, one request at a time. engine-admit! puts a request
;;;         in a FIFO queue, or raises 'queue-full when queue-limit requests
;;;         are already waiting (the running one doesn't count). engine-run
;;;         waits for its turn, generates, and leaves the queue however it
;;;         returns. A request admitted but never run must be dropped with
;;;         engine-drop!.
;;; Loop    render the chat template, tokenize, pick a session from the pool
;;;         (prefix reuse), prefill, then step until an end-of-turn token,
;;;         max-tokens, the context length, a stop string, or cancellation.
;;;         The text is filtered on the fly: stop strings end the reply and
;;;         are not emitted; <tool_call>...</tool_call> spans become tool
;;;         calls. Text that might be the start of a stop string or of
;;;         <tool_call> is held back until it is decided.
;;; Events  (on-event 'content text) and
;;;         (on-event 'tool-call index id name arguments-json);
;;;         a #f result cancels the generation (the client is gone).

(define-module (sonicboom serve engine)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (ice-9 threads)
  #:use-module (sonicboom serve backend)
  #:use-module (sonicboom serve sessions)
  #:use-module (sonicboom serve chat-template)
  #:export (make-engine engine? engine-models engine-model-ids engine-model
            engine-model-families engine-model-loaded? engine-model-install!
            engine-model-clear! engine-pool engine-queue-limit

            make-chat-request chat-request?
            chat-request-model chat-request-messages chat-request-tools
            chat-request-tool-choice chat-request-max-tokens chat-request-stop
            chat-request-temperature chat-request-top-p chat-request-top-k
            chat-request-min-p chat-request-seed chat-request-repeat-penalty
            chat-request-presence-penalty
            chat-request-frequency-penalty chat-request-constraint

            engine-admit! engine-run engine-run-job engine-drop! engine-generate
            engine-ticket-wait-ms engine-stats

            result? result-finish-reason result-content result-tool-calls
            result-prompt-tokens result-completion-tokens result-cached-tokens))

;;; ---------------------------------------------------------------------------
;;; Requests and results

;; A chat request: what to send to a model and how to sample the reply.
;;
;; The fields are model (a model id), messages, tools, tool-choice, max-tokens
;; (#f: up to the context length), stop (a list of strings), temperature, top-p,
;; top-k, min-p, seed, repeat-penalty, presence-penalty, frequency-penalty and
;; constraint, with an accessor named after each. An omitted sampling parameter
;; is neutral: temperature 0 (greedy), top-p 1, no top-k, no min-p, seed 0,
;; repeat-penalty 1 (off) and no penalties.
;;
;; Example:
;;   (use-modules (sonicboom serve engine))
;;   (chat-request? (make-chat-request #:model "m" #:messages '()))
;;   => #t
;; PyTorch: none
(define-record-type <chat-request>
  (%make-chat-request model messages tools tool-choice max-tokens stop
                      temperature top-p top-k min-p seed repeat-penalty
                      presence-penalty frequency-penalty constraint)
  chat-request?
  (model chat-request-model)                ; model id
  (messages chat-request-messages)          ; list of message alists
  (tools chat-request-tools)                ; list of tool alists
  (tool-choice chat-request-tool-choice)    ; "none" "auto" "required" or a name
  (max-tokens chat-request-max-tokens)      ; #f: up to the context length
  (stop chat-request-stop)                  ; list of strings
  (temperature chat-request-temperature)    ; 0: greedy (the default)
  (top-p chat-request-top-p)
  (top-k chat-request-top-k)
  (min-p chat-request-min-p)                ; 0: off
  (seed chat-request-seed)                  ; #f only for a direct caller; the
                                            ; HTTP layer always resolves one
  (repeat-penalty chat-request-repeat-penalty) ; 1: off
  (presence-penalty chat-request-presence-penalty)
  (frequency-penalty chat-request-frequency-penalty)
  (constraint chat-request-constraint))     ; #f or a token filter

;; Omitted sampling parameters are neutral: temperature 0 (greedy), top-p 1,
;; no top-k, no min-p, repeat-penalty 1 and no penalties. A request is thus
;; greedy unless it asks for sampling, and reproducible when it does.
(define* (make-chat-request #:key model messages (tools '()) (tool-choice "auto")
                            (max-tokens #f) (stop '()) (temperature 0.0)
                            (top-p 1.0) (top-k 0) (min-p 0.0) (seed 0)
                            (repeat-penalty 1.0)
                            (presence-penalty 0.0) (frequency-penalty 0.0)
                            (constraint #f))
  "Create a chat request.

Every keyword is optional except model and messages; the sampling parameters
default to a greedy, reproducible request.

Args:
  model: keyword; the model id.
  messages: keyword; the list of message alists with string keys.
  tools: keyword; the tools offered to the model (default '()).
  tool-choice: keyword; \"none\", \"auto\", \"required\" or a function name
    (default \"auto\").
  max-tokens: keyword; the reply limit, or #f for the context length
    (default #f).
  stop: keyword; strings that end the reply (default '()).
  temperature: keyword; the sampling temperature, 0 for greedy (default 0.0).
  top-p: keyword; the nucleus mass (default 1.0).
  top-k: keyword; the candidate count, 0 for no limit (default 0).
  min-p: keyword; minimum relative probability, 0 for off (default 0.0).
  seed: keyword; the sampling seed (default 0).
  repeat-penalty: keyword; the repetition penalty, 1 for off (default 1.0).
  presence-penalty: keyword; the presence penalty (default 0.0).
  frequency-penalty: keyword; the frequency penalty (default 0.0).
  constraint: keyword; a constrained-decoding constraint or #f (default #f).
Returns: a chat request.
Example:
  (use-modules (sonicboom serve engine))
  (chat-request-temperature (make-chat-request #:model \"m\" #:messages '()))
  => 0.0
PyTorch: none"
  (%make-chat-request model messages tools tool-choice max-tokens stop
                      temperature top-p top-k min-p seed repeat-penalty
                      presence-penalty frequency-penalty constraint))

;; The result of one request.
;;
;; finish-reason is "stop", "length", "tool_calls" or "cancelled"; content is the
;; reply text; tool-calls the tool calls as (id name arguments-json); and
;; prompt-tokens, completion-tokens and cached-tokens count the tokens of the
;; request.
;;
;; Example:
;;   (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
;;                (sonicboom serve engine))
;;   (let* ((engine (make-engine
;;                   (list (load-model (make-mock-backend) '((id . "m"))))))
;;          (request (make-chat-request
;;                    #:model "m"
;;                    #:messages (list (list (cons "role" "user")
;;                                           (cons "content" "hi"))))))
;;     (result-finish-reason
;;      (engine-run engine (engine-admit! engine) request
;;                  (lambda (kind . args) #t))))
;;   => "stop"
;; PyTorch: none
(define-record-type <result>
  (make-result finish-reason content tool-calls prompt-tokens
               completion-tokens cached-tokens)
  result?
  (finish-reason result-finish-reason)      ; "stop" "length" "tool_calls" "cancelled"
  (content result-content)
  (tool-calls result-tool-calls)            ; list of (id name arguments-json)
  (prompt-tokens result-prompt-tokens)
  (completion-tokens result-completion-tokens)
  (cached-tokens result-cached-tokens))

;;; ---------------------------------------------------------------------------
;;; The engine

;; One served model and its session pool. The model may be absent: the serve
;; layer's residency manager unloads the model of a request that is not running
;; and installs the next one between jobs, so /v1/models still lists it.
(define-record-type <model-slot>
  (make-model-slot model pool)
  model-slot?
  (model slot-model set-slot-model!)
  (pool slot-pool set-slot-pool!))

;; An engine serves a set of models and runs one request at a time.
;;
;; engine-models is the alist id -> slot; every loaded model has its own
;; session pool. engine-queue-limit is how many requests may wait for their
;; turn. The engine is independent of HTTP: a caller hands it a chat request
;; and gets events back.
;;
;; Example:
;;   (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
;;                (sonicboom serve engine))
;;   (engine-model-ids
;;    (make-engine (list (load-model (make-mock-backend) '((id . "m"))))))
;;   => ("m")
;; PyTorch: none
(define-record-type <engine>
  (%make-engine models queue-limit lock cv running queue stats counter prepare)
  engine?
  (models engine-models)               ; alist: id -> <model-slot>
  (queue-limit engine-queue-limit)
  (lock engine-lock)
  (cv engine-cv)
  (running engine-running set-engine-running!)
  (queue engine-queue set-engine-queue!)
  (stats engine-stats-table)
  (counter engine-counter set-engine-counter!)
  (prepare engine-prepare))            ; (request) or #f, called with the turn

;; models: list of models (from load-model). Each gets its own session pool.
;; ids: models to serve without loading them; their slot starts empty and the
;; caller installs a model with engine-model-install! (see (sonicboom serve
;; residency)).
(define* (make-engine models #:key (queue-limit 8) (max-sessions 4)
                      (token-budget 32768) (ids '()) (prepare #f))
  "Create an engine that serves MODELS.

Every model gets its own session pool with the given limits. A model listed in
IDS is served without a model yet: its slot is empty until engine-model-install!
fills it, which is how the serve layer's residency manager loads on demand.

Args:
  models: the list of models to serve, already loaded.
  queue-limit: keyword; how many requests may wait for their turn
    (default 8).
  max-sessions: keyword; the session pool size per model (default 4).
  token-budget: keyword; the tokens the session pool may hold per model
    (default 32768).
  ids: keyword; further model ids to serve, with an empty slot (default '()).
  prepare: keyword; a procedure (request) called after a chat request reaches
    the front of the queue and before it runs, or #f (default #f).
Returns: an engine.
Example:
  (use-modules (sonicboom serve engine))
  (engine? (make-engine '()))
  => #t
PyTorch: none"
  (%make-engine (append
                 (map (lambda (m)
                        (cons (model-id m)
                              (make-model-slot
                               m (make-session-pool m #:max-sessions max-sessions
                                                    #:token-budget token-budget))))
                      models)
                 (map (lambda (id) (cons id (make-model-slot #f #f))) ids))
                queue-limit (make-mutex) (make-condition-variable) #f '()
                (make-hash-table) 0 prepare))

(define (engine-model-ids engine)
  "The ids of the models ENGINE serves, in serving order.

Args:
  engine: an engine.
Returns: a list of strings.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine))
  (engine-model-ids
   (make-engine (list (load-model (make-mock-backend) '((id . \"m\"))))))
  => (\"m\")
PyTorch: none"
  (map car (engine-models engine)))

(define (engine-model engine id)
  "The model ENGINE serves as ID, or #f.

Args:
  engine: an engine.
  id: a model id string.
Returns: a model or #f.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine))
  (model-id (engine-model
             (make-engine (list (load-model (make-mock-backend) '((id . \"m\")))))
             \"m\"))
  => \"m\"
PyTorch: none"
  (match (assoc id (engine-models engine)) ((_ . s) (slot-model s)) (#f #f)))

;; Whether the model ENGINE serves as ID is installed (loaded) right now.
(define (engine-model-loaded? engine id)
  "True when the model ID currently has a loaded model in ENGINE.

A model the residency manager unloaded is still served (its id and place in
the list remain), but engine-model returns #f for it until it is installed
again.

Args:
  engine: an engine.
  id: a model id string.
Returns: a boolean.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine))
  (engine-model-loaded?
   (make-engine '() #:ids '(\"m\")) \"m\")
  => #f
PyTorch: none"
  (and (engine-model engine id) #t))

;; Install a loaded model and its pool for ID (residency manager).
(define (engine-model-install! engine id model pool)
  "Install MODEL and its session POOL as the model served as ID.

Args:
  engine: an engine.
  id: a model id string the engine serves.
  model: the loaded model.
  pool: its session pool.
Returns: unspecified.
PyTorch: none"
  (with-mutex (engine-lock engine)
    (match (assoc id (engine-models engine))
      ((_ . s) (set-slot-model! s model) (set-slot-pool! s pool))
      (#f (error "the engine does not serve" id)))))

;; Drop the model and pool of ID after they were freed (residency manager).
(define (engine-model-clear! engine id)
  "Forget the model and session pool installed as ID.

The caller frees them first (sessions, then the model); the engine keeps only
the id, so a later engine-model-install! can serve it again.

Args:
  engine: an engine.
  id: a model id string the engine serves.
Returns: unspecified.
PyTorch: none"
  (with-mutex (engine-lock engine)
    (match (assoc id (engine-models engine))
      ((_ . s) (set-slot-model! s #f) (set-slot-pool! s #f))
      (#f (error "the engine does not serve" id)))))

;; (id . family) per served model, in serving order: the family each model
;; selected from its architecture, for /v1/models. A model that is unloaded at
;; the moment has no family to report and is left out.
(define (engine-model-families engine)
  "The chat-template family of every loaded model ENGINE serves.

A model the residency manager unloaded is left out until it is loaded again.

Args:
  engine: an engine.
Returns: an alist (id . family), in serving order.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine))
  (engine-model-families
   (make-engine (list (load-model (make-mock-backend) '((id . \"m\"))))))
  => ((\"m\" . gemma))
PyTorch: none"
  (append-map (lambda (id)
                (let ((model (engine-model engine id)))
                  (if model (list (cons id (model-family model))) '())))
              (engine-model-ids engine)))

(define (engine-pool engine id)
  "The session pool of the model ENGINE serves as ID, or #f.

#f when the model is not loaded at the moment either.

Args:
  engine: an engine.
  id: a model id string.
Returns: a session pool or #f.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve sessions))
  (session-pool? (engine-pool
                  (make-engine
                   (list (load-model (make-mock-backend) '((id . \"m\")))))
                  \"m\"))
  => #t
PyTorch: none"
  (match (assoc id (engine-models engine)) ((_ . s) (slot-pool s)) (#f #f)))

(define (bump! engine key n)
  (with-mutex (engine-lock engine)
    (hashq-set! (engine-stats-table engine) key
                (+ n (hashq-ref (engine-stats-table engine) key 0)))))

(define (stat-set! engine key v)
  (with-mutex (engine-lock engine)
    (hashq-set! (engine-stats-table engine) key v)))

;;; The queue. A ticket is a fresh list; its identity is what the queue
;;; compares, and it also carries when it was admitted (index 1) and when it
;;; started running (index 2, unset until then), so a request can report how
;;; long it waited.

(define (ticket-now-ms)
  (quotient (* 1000 (get-internal-real-time)) internal-time-units-per-second))

(define (engine-admit! engine)
  "Put a request in ENGINE's queue and return its ticket.

A request admitted but never run must be dropped with engine-drop!.

Args:
  engine: an engine.
Returns: a ticket.
Example:
  (use-modules (sonicboom serve engine))
  (pair? (engine-admit! (make-engine '())))
  => #t
PyTorch: none
Note: when queue-limit requests are already waiting (the running one does not
count), it raises (queue-full n) with the number of requests over the limit."
  (with-mutex (engine-lock engine)
    ;; Admitted = the running one + the queue; one of them runs, the others
    ;; wait.
    (let ((admitted (+ (if (engine-running engine) 1 0)
                       (length (engine-queue engine)))))
      (when (> admitted (engine-queue-limit engine))
        (hashq-set! (engine-stats-table engine) 'rejected
                    (1+ (hashq-ref (engine-stats-table engine) 'rejected 0)))
        (throw 'queue-full (1- admitted)))
      (let ((ticket (list 'ticket (ticket-now-ms) #f)))
        (set-engine-queue! engine (append (engine-queue engine) (list ticket)))
        (hashq-set! (engine-stats-table engine) 'max-waiting
                    (max (hashq-ref (engine-stats-table engine) 'max-waiting 0)
                         admitted))
        ticket))))

(define (wait-turn! engine ticket)
  (with-mutex (engine-lock engine)
    (let lp ()
      (unless (and (not (engine-running engine))
                   (eq? (car (engine-queue engine)) ticket))
        (wait-condition-variable (engine-cv engine) (engine-lock engine))
        (lp)))
    (set-engine-queue! engine (cdr (engine-queue engine)))
    (set-engine-running! engine ticket)
    (list-set! ticket 2 (ticket-now-ms))))

;; Leave the queue: done running, or never run.
(define (engine-drop! engine ticket)
  "Leave the queue with TICKET, whether it ran or not.

Args:
  engine: an engine.
  ticket: a ticket from engine-admit!.
Returns: unspecified.
Example:
  (use-modules (sonicboom serve engine))
  (let ((engine (make-engine '())))
    (engine-drop! engine (engine-admit! engine))
    #t)
  => #t
PyTorch: none"
  (with-mutex (engine-lock engine)
    (if (eq? (engine-running engine) ticket)
        (set-engine-running! engine #f)
        (set-engine-queue! engine (delete ticket (engine-queue engine) eq?)))
    (broadcast-condition-variable (engine-cv engine))))

(define (engine-ticket-wait-ms ticket)
  "How long TICKET waited for its turn in milliseconds.

Args:
  ticket: a ticket from engine-admit!.
Returns: the wait in ms, or #f when the ticket never started (it was dropped
  before its turn, for example because its client went away).
Example:
  (use-modules (sonicboom serve engine))
  (let* ((engine (make-engine '()))
         (ticket (engine-admit! engine)))
    (engine-run-job engine ticket (lambda () #t))
    (>= (engine-ticket-wait-ms ticket) 0))
  => #t
PyTorch: none
Note: admission and the start of running are stamped on the ticket, so the
value is what the queue cost the request, not the time it spent generating."
  (and (pair? (cddr ticket))
       (list-ref ticket 2)
       (max 0 (- (list-ref ticket 2) (list-ref ticket 1)))))

(define (next-call-id! engine)
  (with-mutex (engine-lock engine)
    (set-engine-counter! engine (1+ (engine-counter engine)))
    (format #f "call_~a" (number->string (engine-counter engine) 16))))

;;; ---------------------------------------------------------------------------
;;; Text filter: stop strings and tool-call spans

;; The length of the longest suffix of text that is a proper prefix of one of
;; the markers: that much must be held back.
(define (held-back text markers)
  (fold (lambda (m best)
          (let lp ((k (min (1- (string-length m)) (string-length text))))
            (cond ((<= k best) best)
                  ((string-suffix? (substring m 0 k) text) k)
                  (else (lp (1- k))))))
        0 markers))

(define* (earliest text markers #:optional (start 0))
  ;; -> (index . marker) of the first marker found at or after START, or #f.
  ;; Text before START was scanned by an earlier call and held no match; only
  ;; the tail a marker can reach into has to be looked at again.
  (fold (lambda (m best)
          (match (string-contains text m (max 0 start))
            (#f best)
            (i (if (or (not best) (< i (car best))) (cons i m) best))))
        #f markers))

;;; ---------------------------------------------------------------------------
;;; Generation

;; Admit, run, and leave the queue.
(define (engine-generate engine request on-event)
  "Admit REQUEST, run it and return its result.

This is engine-admit! followed by engine-run, for a caller that does not need
the ticket.

Args:
  engine: an engine.
  request: a chat request.
  on-event: the event procedure (see engine-run).
Returns: a result.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine))
  (let ((engine (make-engine
                 (list (load-model (make-mock-backend) '((id . \"m\")))))))
    (result-content
     (engine-generate
      engine
      (make-chat-request #:model \"m\"
                         #:messages (list (list (cons \"role\" \"user\")
                                                (cons \"content\" \"hi\"))))
      (lambda (kind . args) #t))))
  => \"You said: hi\"
PyTorch: none"
  (engine-run engine (engine-admit! engine) request on-event))

(define (engine-run engine ticket request on-event)
  "Wait for TICKET's turn, run REQUEST and return its result.

The ticket is dropped from the queue however this returns.

Args:
  engine: an engine.
  ticket: a ticket from engine-admit!.
  request: a chat request.
  on-event: a procedure called with each event: ('content text) or
    ('tool-call index id name arguments-json); a #f result cancels the
    generation.
Returns: a result.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine))
  (let* ((engine (make-engine
                  (list (load-model (make-mock-backend) '((id . \"m\"))))))
         (request (make-chat-request
                   #:model \"m\"
                   #:messages (list (list (cons \"role\" \"user\")
                                          (cons \"content\" \"hi\"))))))
    (result-finish-reason
     (engine-run engine (engine-admit! engine) request
                 (lambda (kind . args) #t))))
  => \"stop\"
PyTorch: none"
  (dynamic-wind
    (lambda () #t)
    (lambda ()
      (wait-turn! engine ticket)
      ;; The device is idle now: the serve layer's residency manager makes the
      ;; request's model the one on it (unloading another model first).
      (let ((prepare (engine-prepare engine)))
        (when prepare (prepare request)))
      (run-request engine request on-event))
    (lambda () (engine-drop! engine ticket))))

;; Run an arbitrary job under the same queue: the image pipeline and the
;; serve layer's load/unload endpoints share the FIFO with chat.
(define (engine-run-job engine ticket thunk)
  "Wait for TICKET's turn, call THUNK, and leave the queue however it returns.

The job runs with the device to itself, exactly like a chat request: the
caller's THUNK does its own model switch and device work.

Args:
  engine: an engine.
  ticket: a ticket from engine-admit!.
  thunk: a procedure of no arguments.
Returns: whatever THUNK returns.
Example:
  (use-modules (sonicboom serve engine))
  (let ((engine (make-engine '())))
    (engine-run-job engine (engine-admit! engine) (lambda () 'done)))
  => done
PyTorch: none"
  (dynamic-wind
    (lambda () #t)
    (lambda () (wait-turn! engine ticket) (thunk))
    (lambda () (engine-drop! engine ticket))))

(define (run-request engine request on-event)
  (let* ((id (chat-request-model request))
         (model (or (engine-model engine id) (throw 'model-not-found id)))
         (pool (engine-pool engine id))
         (tools (chat-request-tools request))
         (prompt-segments
          (render-chat-segments model (chat-request-messages request)
                                #:tools tools
                                #:tool-choice (chat-request-tool-choice request)))
         (prompt-tokens (tokenize-chat model prompt-segments))
         (n-prompt (length prompt-tokens))
         (ctx (model-context-length model))
         (eos (model-eos-tokens model))
         (stops (filter (lambda (s) (not (string-null? s))) (chat-request-stop request)))
         (tool-markers (if (or (null? tools)
                               (equal? (chat-request-tool-choice request) "none"))
                           '()
                           (list tool-call-open)))
         (limit (let ((room (- ctx n-prompt)))
                  (match (chat-request-max-tokens request)
                    (#f room)
                    (n (min n room)))))
         (sampler (make-sampler #:temperature (chat-request-temperature request)
                                #:top-p (chat-request-top-p request)
                                #:top-k (chat-request-top-k request)
                                #:min-p (chat-request-min-p request)
                                #:seed (chat-request-seed request)
                                #:repeat-penalty (chat-request-repeat-penalty request)
                                #:presence-penalty (chat-request-presence-penalty request)
                                #:frequency-penalty (chat-request-frequency-penalty request))))
    (when (>= n-prompt ctx)
      (throw 'context-length-exceeded n-prompt ctx))
    (bump! engine 'requests 1)
    (let* ((entry (pool-acquire! pool prompt-tokens))
           (session (pool-entry-session entry))
           (tokens (make-vector 64 0))     ; sampled tokens, in order
           (n-tokens 0)
           (prefix-off 0)                  ; first token of the decode window
           (read-off 0)                    ; first token whose text is unread
           (content '())                   ; emitted content pieces, newest first
           (calls '()))                    ; (id name args), newest first
      ;; The token log grows by doubling. The detokenizer is handed only the
      ;; window since the last emitted text, so its per-call work stays small.
      (define (token-add! t)
        (when (= n-tokens (vector-length tokens))
          (let ((bigger (make-vector (* 2 (vector-length tokens)) 0)))
            (vector-move-left! tokens 0 n-tokens bigger 0)
            (set! tokens bigger)))
        (vector-set! tokens n-tokens t)
        (set! n-tokens (1+ n-tokens)))
      (define (token-range a b)           ; [a, b) as a list, in order
        (let lp ((i (1- b)) (acc '()))
          (if (< i a) acc (lp (1- i) (cons (vector-ref tokens i) acc)))))
      (dynamic-wind
        (lambda () #t)
        (lambda ()
          (let ((cached (prefill session prompt-tokens
                                 (chat-request-constraint request))))
            ;; State of the text filter.
            (define buffer "")             ; undecided text
            (define in-call? #f)
            (define markers (append stops tool-markers))
            (define max-marker (fold max 0 (map string-length markers)))
            (define cancelled? #f)
            (define stopped? #f)
            (define (emit-content! text)
              (unless (or (string-null? text) cancelled?)
                (set! content (cons text content))
                (unless (on-event 'content text) (set! cancelled? #t))))
            (define (emit-call! body)
              (match (parse-tool-call body)
                (#f (emit-content! (string-append tool-call-open body tool-call-close)))
                ((name . args)
                 (let ((call (list (next-call-id! engine) name args)))
                   (set! calls (cons call calls))
                   (unless (or cancelled?
                               (apply on-event 'tool-call (1- (length calls)) call))
                     (set! cancelled? #t))))))
            (define eos-seen? #f)
            ;; Incremental decoding. prefix-off..read-off is the window whose
            ;; text was already emitted; read-off..end is the tail that may
            ;; still be incomplete. Both texts start at prefix-off, so the new
            ;; text extends the old one by the characters to emit next.
            (define (decode-window a b) (detokenize model (token-range a b)))
            (define (stream-text!)
              (let* ((prefix (decode-window prefix-off read-off))
                     (text (decode-window prefix-off n-tokens)))
                ;; A trailing U+FFFD may be an incomplete multi-byte character
                ;; that the next token completes, so wait for it.
                (when (and (> (string-length text) (string-length prefix))
                           (not (string-suffix? "\ufffd" text)))
                  (feed! (substring text (string-length prefix)) #f)
                  (set! prefix-off read-off)
                  (set! read-off n-tokens))))
            ;; The reply is over: emit what is left. An incomplete character at
            ;; the very end is dropped, as it was before.
            (define (flush-text!)
              (let* ((prefix (decode-window prefix-off read-off))
                     (text (decode-window prefix-off n-tokens))
                     (ready (if (string-suffix? "\ufffd" text)
                                (1- (string-length text))
                                (string-length text))))
                (when (> ready (string-length prefix))
                  (feed! (substring text (string-length prefix) ready) #f))))
            (define (feed! text final?)
              ;; A match can only begin in the tail a marker reaches back into:
              ;; everything before that was scanned when the text arrived and
              ;; held none. After a match is consumed the remaining text has
              ;; not been scanned, so the rescan starts at its beginning.
              (let* ((old-len (string-length buffer))
                     (scan-from (max 0 (- old-len (- max-marker 1))))
                     (close-from (max 0 (- old-len (- (string-length tool-call-close) 1)))))
                (set! buffer (string-append buffer text))
                (let lp ((from scan-from) (close-from close-from))
                  (cond
                   (in-call?
                    (match (string-contains buffer tool-call-close close-from)
                      (#f (when final?
                            (emit-call! buffer)
                            (set! buffer "")))
                      (i (emit-call! (substring buffer 0 i))
                         (set! buffer (substring buffer (+ i (string-length tool-call-close))))
                         (set! in-call? #f)
                         (lp 0 0))))
                   (else
                    (match (earliest buffer markers from)
                      ((i . m)
                       (emit-content! (substring buffer 0 i))
                       (cond
                        ((member m stops)
                         (set! stopped? #t)
                         (set! buffer ""))
                        (else
                         (set! buffer (substring buffer (+ i (string-length m))))
                         (set! in-call? #t)
                         (lp 0 0))))
                      (#f
                       (let ((keep (if final? 0 (held-back buffer markers))))
                         (emit-content! (substring buffer 0 (- (string-length buffer) keep)))
                         (set! buffer (substring buffer (- (string-length buffer) keep)))))))))))
            ;; One token of the reply: bookkeeping, the end-of-turn check and
            ;; the text side. Returns #t to keep decoding and #f when the reply
            ;; is over (an end-of-turn token, a stop string, the client gone, or
            ;; the limit reached).
            (define (handle-token! t)
              (if (memv t eos)
                  (begin
                    (set! eos-seen? #t)
                    ;; The end-of-turn token has no reply text; flush what the
                    ;; text tokens before it still held back, then log it.
                    (flush-text!)
                    (token-add! t)
                    (feed! "" #t)
                    #f)
                  (begin
                    (token-add! t)
                    (stream-text!)
                    (cond
                     (cancelled? #f)
                     (stopped? #f)
                     ((>= n-tokens limit)
                      (flush-text!)
                      (feed! "" #t)
                      #f)
                     (else #t)))))
            (define (finish-reason)
              (cond
               (cancelled? "cancelled")
               ((or stopped? eos-seen?) "stop")
               ((>= n-tokens limit) "length")
               (else "stop")))
            (define finish
              (cond
               ((>= 0 limit) (flush-text!) (feed! "" #t) "length")
               ;; A backend that produces the whole reply in one call drives
               ;; the token loop through its callback. Ending short of the
               ;; limit without a stop means a grammar closed the reply: the
               ;; stop token was consumed by the decoder and is not text, the
               ;; same as the step loop's (not t).
               ((and (not (getenv "SONICBOOM_SERVE_STEP"))
                     (can-generate? model))
                (let ((produced (generate session sampler
                                          (chat-request-constraint request)
                                          limit handle-token!)))
                  (when (and (< produced limit)
                             (not cancelled?)
                             (not stopped?))
                    (flush-text!)
                    (feed! "" #t))
                  (finish-reason)))
               (else
                (let lp ()
                  (let ((t (step session sampler
                                 (chat-request-constraint request))))
                    (cond
                     ((not t) (flush-text!) (feed! "" #t) (finish-reason))
                     ((handle-token! t) (lp))
                     (else (finish-reason))))))))
            (let* ((reason (cond ((equal? finish "cancelled") "cancelled")
                                 ((and (pair? calls) (equal? finish "stop")) "tool_calls")
                                 (else finish)))
                   (result (make-result reason
                                        (string-concatenate-reverse content)
                                        (reverse calls)
                                        n-prompt
                                        n-tokens
                                        cached)))
              (bump! engine (if (equal? reason "cancelled") 'cancelled 'completed) 1)
              (bump! engine 'prompt-tokens n-prompt)
              (bump! engine 'completion-tokens n-tokens)
              (bump! engine 'cached-tokens cached)
              (when (equal? reason "cancelled")
                (stat-set! engine 'last-cancelled-completion-tokens n-tokens))
              result)))
        (lambda ()
          ;; The session now holds the prompt and every sampled token.
          (pool-release! pool entry (append prompt-tokens (token-range 0 n-tokens))))))))

(define (engine-stats engine)
  "The counters of ENGINE as an alist.

The alist holds queue-limit, waiting, running and the request and token
counters.

Args:
  engine: an engine.
Returns: an alist.
Example:
  (use-modules (sonicboom serve engine))
  (assq-ref (engine-stats (make-engine '())) 'queue-limit)
  => 8
PyTorch: none"
  (with-mutex (engine-lock engine)
    `((queue-limit . ,(engine-queue-limit engine))
      (waiting . ,(length (engine-queue engine)))
      (running . ,(if (engine-running engine) 1 0))
      ,@(map (lambda (k) (cons k (hashq-ref (engine-stats-table engine) k 0)))
             '(requests completed cancelled rejected max-waiting
               prompt-tokens completion-tokens cached-tokens
               last-cancelled-completion-tokens)))))
