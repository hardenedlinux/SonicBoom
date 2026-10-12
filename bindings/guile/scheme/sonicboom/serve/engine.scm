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
            engine-pool engine-queue-limit

            make-chat-request chat-request?
            chat-request-model chat-request-messages chat-request-tools
            chat-request-tool-choice chat-request-max-tokens chat-request-stop
            chat-request-temperature chat-request-top-p chat-request-top-k
            chat-request-seed chat-request-constraint

            engine-admit! engine-run engine-drop! engine-generate
            engine-stats

            result? result-finish-reason result-content result-tool-calls
            result-prompt-tokens result-completion-tokens result-cached-tokens))

;;; ---------------------------------------------------------------------------
;;; Requests and results

(define-record-type <chat-request>
  (%make-chat-request model messages tools tool-choice max-tokens stop
                      temperature top-p top-k seed constraint)
  chat-request?
  (model chat-request-model)                ; model id
  (messages chat-request-messages)          ; list of message alists
  (tools chat-request-tools)                ; list of tool alists
  (tool-choice chat-request-tool-choice)    ; "none" "auto" "required" or a name
  (max-tokens chat-request-max-tokens)      ; #f: up to the context length
  (stop chat-request-stop)                  ; list of strings
  (temperature chat-request-temperature)
  (top-p chat-request-top-p)
  (top-k chat-request-top-k)
  (seed chat-request-seed)
  (constraint chat-request-constraint))     ; #f or a token filter

(define* (make-chat-request #:key model messages (tools '()) (tool-choice "auto")
                            (max-tokens #f) (stop '()) (temperature 1.0)
                            (top-p 1.0) (top-k 0) (seed #f) (constraint #f))
  (%make-chat-request model messages tools tool-choice max-tokens stop
                      temperature top-p top-k seed constraint))

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

(define-record-type <engine>
  (%make-engine models queue-limit lock cv running queue stats counter)
  engine?
  (models engine-models)               ; alist: id -> (model . pool)
  (queue-limit engine-queue-limit)
  (lock engine-lock)
  (cv engine-cv)
  (running engine-running set-engine-running!)
  (queue engine-queue set-engine-queue!)
  (stats engine-stats-table)
  (counter engine-counter set-engine-counter!))

;; models: list of models (from load-model). Each gets its own session pool.
(define* (make-engine models #:key (queue-limit 8) (max-sessions 4)
                      (token-budget 32768))
  (%make-engine (map (lambda (m)
                       (cons (model-id m)
                             (cons m (make-session-pool m #:max-sessions max-sessions
                                                        #:token-budget token-budget))))
                     models)
                queue-limit (make-mutex) (make-condition-variable) #f '()
                (make-hash-table) 0))

(define (engine-model-ids engine) (map car (engine-models engine)))

(define (engine-model engine id)
  (match (assoc id (engine-models engine)) ((_ . (m . _)) m) (#f #f)))

(define (engine-pool engine id)
  (match (assoc id (engine-models engine)) ((_ . (_ . p)) p) (#f #f)))

(define (bump! engine key n)
  (with-mutex (engine-lock engine)
    (hashq-set! (engine-stats-table engine) key
                (+ n (hashq-ref (engine-stats-table engine) key 0)))))

(define (stat-set! engine key v)
  (with-mutex (engine-lock engine)
    (hashq-set! (engine-stats-table engine) key v)))

;;; The queue. A ticket is a fresh pair; only its identity matters.

(define (engine-admit! engine)
  (with-mutex (engine-lock engine)
    ;; Admitted = the running one + the queue; one of them runs, the others
    ;; wait.
    (let ((admitted (+ (if (engine-running engine) 1 0)
                       (length (engine-queue engine)))))
      (when (> admitted (engine-queue-limit engine))
        (hashq-set! (engine-stats-table engine) 'rejected
                    (1+ (hashq-ref (engine-stats-table engine) 'rejected 0)))
        (throw 'queue-full (1- admitted)))
      (let ((ticket (list 'ticket)))
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
    (set-engine-running! engine ticket)))

;; Leave the queue: done running, or never run.
(define (engine-drop! engine ticket)
  (with-mutex (engine-lock engine)
    (if (eq? (engine-running engine) ticket)
        (set-engine-running! engine #f)
        (set-engine-queue! engine (delete ticket (engine-queue engine) eq?)))
    (broadcast-condition-variable (engine-cv engine))))

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

(define (earliest text markers)
  ;; -> (index . marker) of the first marker found, or #f
  (fold (lambda (m best)
          (match (string-contains text m)
            (#f best)
            (i (if (or (not best) (< i (car best))) (cons i m) best))))
        #f markers))

;;; ---------------------------------------------------------------------------
;;; Generation

;; Admit, run, and leave the queue.
(define (engine-generate engine request on-event)
  (engine-run engine (engine-admit! engine) request on-event))

(define (engine-run engine ticket request on-event)
  (dynamic-wind
    (lambda () #t)
    (lambda ()
      (wait-turn! engine ticket)
      (generate engine request on-event))
    (lambda () (engine-drop! engine ticket))))

(define (generate engine request on-event)
  (let* ((id (chat-request-model request))
         (model (or (engine-model engine id) (throw 'model-not-found id)))
         (pool (engine-pool engine id))
         (tools (chat-request-tools request))
         (prompt (render-chat model (chat-request-messages request)
                              #:tools tools
                              #:tool-choice (chat-request-tool-choice request)))
         (prompt-tokens (tokenize model prompt))
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
                                #:seed (chat-request-seed request))))
    (when (>= n-prompt ctx)
      (throw 'context-length-exceeded n-prompt ctx))
    (bump! engine 'requests 1)
    (let* ((entry (pool-acquire! pool prompt-tokens))
           (session (pool-entry-session entry))
           (generated '())                 ; sampled tokens, newest first
           (content '())                   ; emitted content pieces, newest first
           (calls '()))                    ; (id name args), newest first
      (dynamic-wind
        (lambda () #t)
        (lambda ()
          (let ((cached (prefill session prompt-tokens)))
            ;; State of the text filter.
            (define decoded 0)             ; chars of the decoded text consumed
            (define buffer "")             ; undecided text
            (define in-call? #f)
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
            (define (feed! text final?)
              (set! buffer (string-append buffer text))
              (let lp ()
                (cond
                 (in-call?
                  (match (string-contains buffer tool-call-close)
                    (#f (when final?
                          (emit-call! buffer)
                          (set! buffer "")))
                    (i (emit-call! (substring buffer 0 i))
                       (set! buffer (substring buffer (+ i (string-length tool-call-close))))
                       (set! in-call? #f)
                       (lp))))
                 (else
                  (match (earliest buffer (append stops tool-markers))
                    ((i . m)
                     (emit-content! (substring buffer 0 i))
                     (cond
                      ((member m stops)
                       (set! stopped? #t)
                       (set! buffer ""))
                      (else
                       (set! buffer (substring buffer (+ i (string-length m))))
                       (set! in-call? #t)
                       (lp))))
                    (#f
                     (let ((keep (if final? 0 (held-back buffer (append stops tool-markers)))))
                       (emit-content! (substring buffer 0 (- (string-length buffer) keep)))
                       (set! buffer (substring buffer (- (string-length buffer) keep))))))))))
            (define finish
              (let lp ((n 0))
                (cond
                 (cancelled? "cancelled")
                 (stopped? "stop")
                 ((>= n limit) (feed! "" #t) "length")
                 (else
                  (let ((t (step session sampler (chat-request-constraint request))))
                    (set! generated (cons t generated))
                    (cond
                     ((memv t eos)
                      (feed! "" #t)
                      (if cancelled? "cancelled" "stop"))
                     (else
                      ;; Decode everything generated so far (minus the end
                      ;; token) and feed the new part; a trailing U+FFFD may
                      ;; be an incomplete multi-byte character, so wait.
                      (let* ((text (detokenize model (reverse generated)))
                             (ready (if (string-suffix? "\xfffd;" text)
                                        (1- (string-length text))
                                        (string-length text))))
                        (when (> ready decoded)
                          (feed! (substring text decoded ready) #f)
                          (set! decoded ready)))
                      (lp (1+ n)))))))))
            (let* ((reason (cond ((equal? finish "cancelled") "cancelled")
                                 ((and (pair? calls) (equal? finish "stop")) "tool_calls")
                                 (else finish)))
                   (result (make-result reason
                                        (string-concatenate-reverse content)
                                        (reverse calls)
                                        n-prompt
                                        (length generated)
                                        cached)))
              (bump! engine (if (equal? reason "cancelled") 'cancelled 'completed) 1)
              (bump! engine 'prompt-tokens n-prompt)
              (bump! engine 'completion-tokens (length generated))
              (bump! engine 'cached-tokens cached)
              (when (equal? reason "cancelled")
                (stat-set! engine 'last-cancelled-completion-tokens (length generated)))
              result)))
        (lambda ()
          ;; The session now holds the prompt and every sampled token.
          (pool-release! pool entry (append prompt-tokens (reverse generated))))))))

(define (engine-stats engine)
  (with-mutex (engine-lock engine)
    `((queue-limit . ,(engine-queue-limit engine))
      (waiting . ,(length (engine-queue engine)))
      (running . ,(if (engine-running engine) 1 0))
      ,@(map (lambda (k) (cons k (hashq-ref (engine-stats-table engine) k 0)))
             '(requests completed cancelled rejected max-waiting
               prompt-tokens completion-tokens cached-tokens
               last-cancelled-completion-tokens)))))
