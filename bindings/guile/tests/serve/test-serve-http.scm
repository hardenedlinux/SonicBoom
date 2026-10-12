;;; SonicBoom — inference service: end-to-end tests over HTTP
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root (needs GNU Artanis; about half a minute):
;;;   guile -L bindings/guile/scheme bindings/guile/tests/serve/test-serve-http.scm
;;;
;;; Starts serve/app.scm on a free port (queue limit 1, 10 ms per token) and
;;; talks to it with (web client) for plain requests, and over raw sockets
;;; for SSE (to read event by event, and to hang up in the middle) and for
;;; WebSocket (a minimal client: handshake, masked text frames).

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 rdelim)
             (ice-9 binary-ports)
             (rnrs bytevectors)
             (web client)
             (web response)
             (json))

;;; ---------------------------------------------------------------------------
;;; The server

(define (free-port)
  (let ((s (socket PF_INET SOCK_STREAM 0)))
    (bind s AF_INET INADDR_LOOPBACK 0)
    (let ((p (sockaddr:port (getsockname s))))
      (close s)
      p)))

(define port (free-port))
(define base (format #f "http://127.0.0.1:~a" port))
(define tmp (or (getenv "TMPDIR") "/tmp"))
(define log-file (format #f "~a/sonicboom-serve-test-~a.log" tmp (getpid)))

(define server-pid
  (let ((pid (primitive-fork)))
    (if (zero? pid)
        (let ((out (open-output-file log-file)))
          (dup2 (fileno out) 1)
          (dup2 (fileno out) 2)
          (execlp "guile" "guile" "-L" "bindings/guile/scheme"
                  "bindings/guile/scheme/sonicboom/serve/app.scm"
                  "--port" (number->string port) "--queue" "1" "--step-delay" "0.01"
                  "--runtime-dir" (format #f "~a/sonicboom-serve-test-~a" tmp (getpid))))
        pid)))

(define (stop-server)
  (catch #t (lambda () (kill server-pid SIGKILL) (waitpid server-pid)) (lambda _ #f)))

;;; ---------------------------------------------------------------------------
;;; Plain HTTP and JSON

(define (jref obj . path)
  (fold (lambda (k o)
          (cond ((not o) #f)
                ((integer? k) (and (vector? o) (< k (vector-length o)) (vector-ref o k)))
                (else (match (assoc k o) ((_ . v) v) (#f #f)))))
        obj path))

(define* (http method path #:optional body)
  ;; -> (status . json)
  (call-with-values
      (lambda ()
        (http-request (string-append base path) #:method method
                      #:body (and body (string->utf8 (if (string? body)
                                                         body
                                                         (scm->json-string body))))
                      #:headers '((content-type application/json))
                      #:decode-body? #f #:keep-alive? #f))
    (lambda (resp bv)
      (cons (response-code resp)
            (and bv (> (bytevector-length bv) 0)
                 (catch #t (lambda () (json-string->scm (utf8->string bv)))
                   (lambda _ (utf8->string bv))))))))

(define (chat body) (http 'POST "/v1/chat/completions" body))

(define (wait-for-server)
  (let lp ((i 0))
    (cond
     ((catch #t (lambda () (= 200 (car (http 'GET "/health")))) (lambda _ #f)) #t)
     ((> i 240) #f)
     (else (usleep 250000) (lp (1+ i))))))

(define (stats . path) (apply jref (cdr (http 'GET "/sonicboom/stats")) path))

(define (msg role content) `(("role" . ,role) ("content" . ,content)))

(define gemma "sonicboom-mock-gemma")
(define qwen "sonicboom-mock-qwen")

(define* (request model messages #:rest extra)
  `(("model" . ,model) ("messages" . ,(list->vector messages)) ("temperature" . 0) ,@extra))

(define weather-tool
  '(("type" . "function")
    ("function" . (("name" . "get_weather")
                   ("description" . "The weather of a city")
                   ("parameters" . (("type" . "object")
                                    ("properties" . (("city" . (("type" . "string")))))
                                    ("required" . #("city"))))))))

;;; ---------------------------------------------------------------------------
;;; SSE over a raw socket

(define (connect-server)
  (let ((s (socket PF_INET SOCK_STREAM 0)))
    (connect s AF_INET INADDR_LOOPBACK port)
    s))

;; NOTE: Not response-body-port: Guile's chunked input port fills its whole
;;       buffer before returning, so it would wait for later events.
;; -> (values socket response reader); (reader) gives the data of the next
;; event, or #f at the end of the stream.
(define (open-stream body)
  (let ((s (connect-server))
        (payload (string->utf8 (scm->json-string body))))
    (format s "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:~a\r\nContent-Type: application/json\r\nContent-Length: ~a\r\n\r\n"
            port (bytevector-length payload))
    (put-bytevector s payload)
    (force-output s)
    (let ((resp (read-response s)))
      (values s resp (if (response-content-length resp)
                         (lambda () #f)
                         (sse-reader s))))))

;; Decode the chunked body by hand, one chunk at a time, and split events.
(define (sse-reader s)
  (define pending "")
  (define done? #f)
  (define (next-chunk!)
    (let ((line (read-line s)))
      (if (eof-object? line)
          (set! done? #t)
          (let ((size (string->number (string-trim-right line #\return) 16)))
            (cond
             ((or (not size) (zero? size)) (read-line s) (set! done? #t))
             (else
              (set! pending (string-append pending (utf8->string (get-bytevector-n s size))))
              (read-line s)))))))
  (lambda ()
    (let lp ()
      (match (string-contains pending "\n\n")
        (#f (if done? #f (begin (next-chunk!) (lp))))
        (i (let ((event (substring pending 0 i)))
             (set! pending (substring pending (+ i 2)))
             (string-join
              (filter-map (lambda (l) (and (string-prefix? "data: " l) (substring l 6)))
                          (string-split event #\newline))
              "\n")))))))

(define (read-event reader) (reader))

;; All events up to [DONE] -> (values chunk-objects done?)
(define (read-all-events reader)
  (let lp ((out '()))
    (match (read-event reader)
      (#f (values (reverse out) #f))
      ("[DONE]" (values (reverse out) #t))
      (data (lp (cons (json-string->scm data) out))))))

(define (stream body)
  (call-with-values (lambda () (open-stream body))
    (lambda (s resp port)
      (call-with-values (lambda () (read-all-events port))
        (lambda (chunks done?)
          (close s)
          (list resp chunks done?))))))

(define (deltas chunks key)
  (filter-map (lambda (c) (jref c "choices" 0 "delta" key)) chunks))

(define (chunk-content chunks) (string-concatenate (deltas chunks "content")))

(define (finish-reason chunks)
  (any (lambda (c) (match (jref c "choices" 0 "finish_reason")
                     ((? string? r) r)
                     (_ #f)))
       chunks))

;;; ---------------------------------------------------------------------------
;;; A minimal WebSocket client

(define (ws-connect path)
  (let ((s (connect-server)))
    (format s "GET ~a HTTP/1.1\r\nHost: 127.0.0.1:~a\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
            path port)
    (force-output s)
    (let ((resp (read-response s)))
      (and (= 101 (response-code resp)) s))))

(define (ws-send-text s text)
  (let* ((payload (string->utf8 text))
         (n (bytevector-length payload))
         (mask #vu8(1 2 3 4)))
    (put-u8 s #x81)
    (cond ((< n 126) (put-u8 s (logior #x80 n)))
          (else (put-u8 s (logior #x80 126))
                (put-u8 s (ash n -8)) (put-u8 s (logand n #xff))))
    (put-bytevector s mask)
    (put-bytevector s (u8-list->bytevector
                       (map (lambda (b i) (logxor b (bytevector-u8-ref mask (modulo i 4))))
                            (bytevector->u8-list payload) (iota n))))
    (force-output s)))

;; -> (opcode . text), or #f at the end
(define (ws-read s)
  (let ((b0 (get-u8 s)))
    (if (eof-object? b0)
        #f
        (let* ((b1 (get-u8 s))
               (len (logand b1 #x7f))
               (len (cond ((= len 126) (bytevector-u16-ref (get-bytevector-n s 2) 0 (endianness big)))
                          ((= len 127) (bytevector-u64-ref (get-bytevector-n s 8) 0 (endianness big)))
                          (else len)))
               (data (if (zero? len) #vu8() (get-bytevector-n s len))))
          (cons (logand b0 #x0f) (utf8->string data))))))

;; Read text messages up to "[DONE]" (or an error object).
(define (ws-read-reply s)
  (let lp ((out '()))
    (match (ws-read s)
      (#f (values (reverse out) #f))
      ((1 . "[DONE]") (values (reverse out) #t))
      ((1 . text)
       (let ((obj (json-string->scm text)))
         (if (jref obj "error")
             (values (reverse (cons obj out)) #f)
             (lp (cons obj out)))))
      (_ (lp out)))))

;;; ---------------------------------------------------------------------------

(define system-prompt
  "You are the SonicBoom test assistant. Answer briefly and repeat the user's last line.")

(define weather-question (msg "user" "What is the weather in Paris?"))

(define long-line (string-join (make-list 80 "word") " "))   ; ~400 tokens of reply

(define (run-tests)
  (test-assert "the server starts" (wait-for-server))



;;; --- models ------------------------------------------------------------------------

(match (http 'GET "/v1/models")
  ((status . body)
   (test-equal "GET /v1/models" 200 status)
   (test-equal "models list" (list gemma qwen)
     (map (lambda (m) (jref m "id")) (vector->list (jref body "data"))))
   (test-equal "list object" "list" (jref body "object"))))

(match (http 'GET "/v1/models/nope")
  ((status . body)
   (test-equal "unknown model: 404" 404 status)
   (test-equal "unknown model: code" "model_not_found" (jref body "error" "code"))))

;;; --- non-streaming -------------------------------------------------------------------

(match (chat (request gemma (list (msg "user" "Hello from the test"))))
  ((status . body)
   (test-equal "chat: 200" 200 status)
   (test-equal "chat: object" "chat.completion" (jref body "object"))
   (test-equal "chat: content" "You said: Hello from the test"
     (jref body "choices" 0 "message" "content"))
   (test-equal "chat: role" "assistant" (jref body "choices" 0 "message" "role"))
   (test-equal "chat: finish_reason" "stop" (jref body "choices" 0 "finish_reason"))
   (test-equal "chat: completion_tokens (reply + end of turn)"
     (1+ (string-length "You said: Hello from the test"))
     (jref body "usage" "completion_tokens"))
   (test-equal "chat: total_tokens"
     (+ (jref body "usage" "prompt_tokens") (jref body "usage" "completion_tokens"))
     (jref body "usage" "total_tokens"))
   (test-equal "chat: cached_tokens of a fresh prompt" 0
     (jref body "usage" "prompt_tokens_details" "cached_tokens"))))

(match (chat (request gemma (list (msg "user" "Hello there")) '("stop" . "there")))
  ((_ . body)
   (test-equal "stop: content ends before the stop string" "You said: Hello "
     (jref body "choices" 0 "message" "content"))
   (test-equal "stop: finish_reason" "stop" (jref body "choices" 0 "finish_reason"))))

(match (chat (request gemma (list (msg "user" "Hello there")) '("max_tokens" . 4)))
  ((_ . body)
   (test-equal "max_tokens: content" "You " (jref body "choices" 0 "message" "content"))
   (test-equal "max_tokens: finish_reason length" "length" (jref body "choices" 0 "finish_reason"))
   (test-equal "max_tokens: completion_tokens" 4 (jref body "usage" "completion_tokens"))))

(let ((hot (lambda (seed)
             (jref (cdr (chat `(("model" . ,gemma)
                                ("messages" . #((("role" . "user") ("content" . "the quick brown fox"))))
                                ("temperature" . 2.0) ("seed" . ,seed))))
                   "choices" 0 "message" "content"))))
  (test-equal "seed: the same seed gives the same sample" (hot 11) (hot 11))
  (test-equal "top_p: a tiny nucleus is greedy" "You said: the quick brown fox"
    (jref (cdr (chat `(("model" . ,gemma)
                       ("messages" . #((("role" . "user") ("content" . "the quick brown fox"))))
                       ("temperature" . 2.0) ("top_p" . 0.01) ("seed" . 3))))
          "choices" 0 "message" "content")))

;;; --- errors ------------------------------------------------------------------------

(match (chat (request gemma (list (msg "user" "x")) '("n" . 2)))
  ((status . body)
   (test-equal "n=2: 400" 400 status)
   (test-equal "n=2: param" "n" (jref body "error" "param"))))
(test-equal "invalid JSON: 400" 400 (car (chat "{not json")))
(test-equal "missing model: 400" 400 (car (chat '(("messages" . #((("role" . "user") ("content" . "x"))))))))
(match (chat (request "no-such-model" (list (msg "user" "x"))))
  ((status . body)
   (test-equal "unknown model in a chat: 404" 404 status)
   (test-equal "unknown model in a chat: type" "invalid_request_error" (jref body "error" "type"))))

;;; --- prefix reuse --------------------------------------------------------------------


(let* ((first (cdr (chat (request gemma (list (msg "system" system-prompt)
                                              (msg "user" "First question"))))))
       (second (cdr (chat (request gemma (list (msg "system" system-prompt)
                                               (msg "user" "Second question"))))))
       ;; <bos> <start_of_turn> u s e r \n, the system text, \n \n
       (shared (+ 7 (string-length system-prompt) 2))
       (reply (jref second "choices" 0 "message" "content"))
       (third (cdr (chat (request gemma (list (msg "system" system-prompt)
                                              (msg "user" "Second question")
                                              (msg "assistant" reply)
                                              (msg "user" "Third question")))))))
  (test-assert "prefix: the first request finds no system prompt in the cache"
    (< (jref first "usage" "prompt_tokens_details" "cached_tokens") shared))
  (test-assert "prefix: the same system prompt is reused (cached_tokens > 0)"
    (>= (jref second "usage" "prompt_tokens_details" "cached_tokens") shared))
  (test-equal "prefix: a continued conversation reuses the whole previous turn"
    (+ (jref second "usage" "prompt_tokens") (jref second "usage" "completion_tokens"))
    (jref third "usage" "prompt_tokens_details" "cached_tokens"))
  (test-equal "prefix: the third reply" "You said: Third question"
    (jref third "choices" 0 "message" "content")))

(test-assert "stats: the backend counted the reused tokens"
  (> (stats "models" gemma "backend" "reused-tokens") 0))
(test-assert "stats: the pool forked a partial match"
  (> (stats "models" gemma "sessions" "forks") 0))

;;; --- streaming -----------------------------------------------------------------------

(match (stream (request qwen (list (msg "user" "Stream me")) '("stream" . #t)
                        '("stream_options" . (("include_usage" . #t)))))
  ((resp chunks done?)
   (test-equal "stream: 200" 200 (response-code resp))
   (test-equal "stream: text/event-stream" 'text/event-stream
     (car (response-content-type resp)))
   (test-assert "stream: ends with [DONE]" done?)
   (test-assert "stream: every chunk is a chat.completion.chunk"
     (every (lambda (c) (equal? "chat.completion.chunk" (jref c "object"))) chunks))
   (test-equal "stream: one id" 1 (length (delete-duplicates (map (lambda (c) (jref c "id")) chunks))))
   (test-equal "stream: the first delta has the role" "assistant"
     (jref (car chunks) "choices" 0 "delta" "role"))
   (test-equal "stream: content" "You said: Stream me" (chunk-content chunks))
   (test-assert "stream: incremental (one chunk per token here)"
     (> (length (deltas chunks "content")) 10))
   (test-equal "stream: finish_reason" "stop" (finish-reason chunks))
   (let ((usage (jref (last chunks) "usage")))
     (test-assert "stream: the usage chunk has no choices"
       (equal? #() (jref (last chunks) "choices")))
     (test-equal "stream: usage completion_tokens" 20 (jref usage "completion_tokens"))
     (test-assert "stream: usage cached_tokens present"
       (integer? (jref usage "prompt_tokens_details" "cached_tokens"))))))

(match (stream (request qwen (list (msg "user" "Stream me")) '("stream" . #t)
                        '("stop" . #("said:"))))
  ((_ chunks done?)
   (test-equal "stream stop: held-back text is dropped" "You " (chunk-content chunks))
   (test-equal "stream stop: finish_reason" "stop" (finish-reason chunks))
   (test-assert "stream stop: no usage chunk unless asked" (not (jref (last chunks) "usage")))))

;;; --- tool calls ----------------------------------------------------------------------


(match (chat (request qwen (list weather-question) `("tools" . #(,weather-tool))))
  ((_ . body)
   (let ((call (jref body "choices" 0 "message" "tool_calls" 0)))
     (test-equal "tools: finish_reason" "tool_calls" (jref body "choices" 0 "finish_reason"))
     (test-equal "tools: content is null" 'null (jref body "choices" 0 "message" "content"))
     (test-assert "tools: an id" (string-prefix? "call_" (jref call "id")))
     (test-equal "tools: type" "function" (jref call "type"))
     (test-equal "tools: name" "get_weather" (jref call "function" "name"))
     (test-equal "tools: arguments are a JSON string" '(("city" . "Paris"))
       (json-string->scm (jref call "function" "arguments")))
     ;; Answer the call: the tool result goes back, the model reads it.
     (match (chat (request qwen (list weather-question
                                      `(("role" . "assistant") ("content" . null)
                                        ("tool_calls" . #(,call)))
                                      `(("role" . "tool") ("tool_call_id" . ,(jref call "id"))
                                        ("content" . "sunny, 21C")))
                           `("tools" . #(,weather-tool))))
       ((_ . body2)
        (test-equal "tools: the tool result is read" "Tool result: sunny, 21C"
          (jref body2 "choices" 0 "message" "content"))
        (test-equal "tools: the call turn is reused from the cache"
          (+ (jref body "usage" "prompt_tokens") (jref body "usage" "completion_tokens"))
          (jref body2 "usage" "prompt_tokens_details" "cached_tokens")))))))

(match (stream (request gemma (list weather-question) '("stream" . #t)
                        `("tools" . #(,weather-tool))))
  ((_ chunks done?)
   (let* ((calls (append-map (lambda (v) (vector->list v)) (deltas chunks "tool_calls")))
          (head (car calls)))
     (test-assert "stream tools: [DONE]" done?)
     (test-equal "stream tools: header delta" '(0 "get_weather" "")
       (list (jref head "index") (jref head "function" "name") (jref head "function" "arguments")))
     (test-assert "stream tools: header id" (string-prefix? "call_" (jref head "id")))
     (test-assert "stream tools: arguments in several deltas" (> (length calls) 2))
     (test-equal "stream tools: arguments concatenate to JSON" '(("city" . "Paris"))
       (json-string->scm (string-concatenate
                          (map (lambda (c) (jref c "function" "arguments")) calls))))
     (test-equal "stream tools: no content" "" (chunk-content chunks))
     (test-equal "stream tools: finish_reason" "tool_calls" (finish-reason chunks)))))

(match (chat (request gemma (list weather-question) `("tools" . #(,weather-tool))
                      '("tool_choice" . "none")))
  ((_ . body)
   (test-equal "tool_choice none: plain text" "You said: What is the weather in Paris?"
     (jref body "choices" 0 "message" "content"))))

(match (chat (request gemma (list (msg "user" "Tell me about Rome"))
                      `("tools" . #(,weather-tool))
                      '("tool_choice" . (("type" . "function")
                                         ("function" . (("name" . "get_weather")))))))
  ((_ . body)
   (test-equal "tool_choice function: the call is forced" "get_weather"
     (jref body "choices" 0 "message" "tool_calls" 0 "function" "name"))))

;;; --- client disconnect -------------------------------------------------------------


(let ((cancelled-before (stats "engine" "cancelled")))
  (call-with-values
      (lambda () (open-stream (request gemma (list (msg "user" long-line)) '("stream" . #t))))
    (lambda (s resp reader)
      (read-event reader) (read-event reader) (read-event reader) ; role + 2 tokens
      (test-equal "disconnect: chunks arrive while the reply is generated"
        1 (stats "engine" "running"))
      (close s)))
  (test-assert "disconnect: the generation is cancelled"
    (let lp ((i 0))
      (cond ((> (stats "engine" "cancelled") cancelled-before) #t)
            ((> i 100) #f)
            (else (usleep 100000) (lp (1+ i))))))
  (test-assert "disconnect: it stopped early"
    (< (stats "engine" "last-cancelled-completion-tokens") 100))
  (test-equal "disconnect: the server goes on" "You said: still here"
    (jref (cdr (chat (request gemma (list (msg "user" "still here")))))
          "choices" 0 "message" "content")))

;;; --- the queue ------------------------------------------------------------------------

;; queue limit 1: one runs, one waits, the third is turned away.
(let ((medium (string-join (make-list 15 "word") " ")))       ; ~85 tokens
  (define (open body) (call-with-values (lambda () (open-stream body)) list))
  (match (open (request gemma (list (msg "user" medium)) '("stream" . #t)))
    ((sa ra pa)
     (read-event pa)                                          ; admitted
     (match (open (request qwen (list (msg "user" medium)) '("stream" . #t)))
       ((sb rb pb)
        (read-event pb)                                       ; admitted, waiting
        (match (chat (request gemma (list (msg "user" "one too many"))))
          ((status . body)
           (test-equal "queue: full -> 429" 429 status)
           (test-equal "queue: code" "queue_full" (jref body "error" "code"))))
        (call-with-values (lambda () (read-all-events pa))
          (lambda (chunks-a done-a?)
            (let ((a-done (get-internal-real-time)))
              (read-event pb)                                 ; B's first token
              (let ((b-first (get-internal-real-time)))
                (call-with-values (lambda () (read-all-events pb))
                  (lambda (chunks-b done-b?)
                    (test-assert "queue: both finish" (and done-a? done-b?))
                    (test-equal "queue: A's reply" (string-append "You said: " medium)
                      (chunk-content chunks-a))
                    (test-assert "queue: B ran after A (serialized)"
                      (> (- b-first a-done) 0))))))))
        (close sa)
        (close sb)))))
  (test-assert "queue: stats" (and (>= (stats "engine" "rejected") 1)
                                   (= (stats "engine" "max-waiting") 1)
                                   (= 0 (stats "engine" "running")))))

;;; --- WebSocket ------------------------------------------------------------------------

(let ((s (ws-connect "/v1/ws/chat")))
  (test-assert "ws: handshake" s)
  (ws-send-text s (scm->json-string (request qwen (list (msg "user" "Over the socket"))
                                             '("stream_options" . (("include_usage" . #t))))))
  (call-with-values (lambda () (ws-read-reply s))
    (lambda (chunks done?)
      (test-assert "ws: [DONE]" done?)
      (test-equal "ws: chunk objects" "chat.completion.chunk" (jref (car chunks) "object"))
      (test-equal "ws: content" "You said: Over the socket" (chunk-content chunks))
      (test-equal "ws: finish_reason" "stop" (finish-reason chunks))
      (test-assert "ws: usage" (integer? (jref (last chunks) "usage" "completion_tokens")))))
  (ws-send-text s (scm->json-string (request qwen (list (msg "user" "Again")))))
  (call-with-values (lambda () (ws-read-reply s))
    (lambda (chunks done?)
      (test-equal "ws: a second request on the connection" "You said: Again"
        (chunk-content chunks))))
  (ws-send-text s "{\"model\": \"sonicboom-mock-qwen\"}")
  (call-with-values (lambda () (ws-read-reply s))
    (lambda (chunks done?)
      (test-equal "ws: a bad request gets an error object" "invalid_request_error"
        (jref (last chunks) "error" "type"))))
  (close s))

(let ((cancelled-before (stats "engine" "cancelled"))
      (s (ws-connect "/v1/ws/chat")))
  (ws-send-text s (scm->json-string (request gemma (list (msg "user" long-line)))))
  (ws-read s) (ws-read s) (ws-read s)
  (test-equal "ws: chunks arrive while the reply is generated" 1 (stats "engine" "running"))
  (close s)
  (test-assert "ws: disconnect cancels"
    (let lp ((i 0))
      (cond ((> (stats "engine" "cancelled") cancelled-before) #t)
            ((> i 100) #f)
            (else (usleep 100000) (lp (1+ i))))))
  (test-assert "ws: it stopped early"
    (< (stats "engine" "last-cancelled-completion-tokens") 100))))

(test-begin "sonicboom-serve-http")
(dynamic-wind (lambda () #t) run-tests stop-server)
(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-serve-http")
(exit (if (zero? failures) 0 1))
