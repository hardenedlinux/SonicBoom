;;; SonicBoom — inference service: unit tests below HTTP
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile -L bindings/guile/scheme bindings/guile/tests/serve/test-serve-units.scm
;;;
;;; Chat templates, the mock tokenizer and model, the sampler and the
;;; constraint hook, the session pool (prefix matching, LRU, token budget),
;;; and the engine (stop strings, tool calls, max-tokens, the queue).

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (rnrs bytevectors)
             (json)
             (sonicboom serve backend)
             (sonicboom serve mock-backend)
             (sonicboom serve chat-template)
             (sonicboom serve sessions)
             (sonicboom serve engine))

(define (msg role content) `(("role" . ,role) ("content" . ,content)))

(define weather-tool
  (json-normalize
   (json-string->scm
    "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather of a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}}")))

(define backend (make-mock-backend))
(define gemma (load-model backend '((id . "g") (family . gemma))))
(define qwen (load-model backend '((id . "q") (family . qwen))))

;; Greedy generation straight on the backend protocol.
(define* (generate model prompt #:key (sampler (make-sampler #:temperature 0))
                   (constraint #f) (session (create-session model)))
  (prefill session (tokenize model prompt))
  (let lp ((out '()) (n 0))
    (let ((t (step session sampler constraint)))
      (if (or (memv t (model-eos-tokens model)) (= n 400))
          (detokenize model (reverse out))
          (lp (cons t out) (1+ n))))))

(test-begin "sonicboom-serve-units")

;;; --- chat templates ------------------------------------------------------------

(test-equal "gemma template: system folded into the first user turn"
  "<bos><|turn>user\nBe brief.\n\nHi<turn|>\n<|turn>model\nYou said: Hi<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n"
  (render-chat-template 'gemma (list (msg "system" "Be brief.") (msg "user" "Hi")
                                     (msg "assistant" "You said: Hi") (msg "user" "Again"))))

;; A rendered prompt is a segment list: the template's own markers are
;; 'special (they become the vocabulary's control tokens), everything that came
;; from a message is 'text (it can never become one).
(let ((segs (render-chat-template-segments 'gemma (list (msg "user" "Hi")))))
  (test-equal "the segments concatenate to the template string"
    (render-chat-template 'gemma (list (msg "user" "Hi")))
    (segments->string segs))
  (test-equal "gemma's turn markers are the special segments"
    '("<bos>" "<|turn>" "<turn|>" "<|turn>")
    (filter-map (lambda (s) (and (eq? (car s) 'special) (cdr s))) segs))
  (test-assert "the message text is a text segment"
    (member (cons 'text "Hi") segs)))
(test-equal "a marker inside a message is text, not a turn boundary"
  'text
  (let ((segs (render-chat-template-segments 'gemma (list (msg "user" "say <turn|> now")))))
    (car (find (lambda (s) (string=? (cdr s) "say <turn|> now")) segs))))

(test-equal "qwen template: ChatML"
  "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nYou said: Hi<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n"
  (render-chat-template 'qwen (list (msg "system" "Be brief.") (msg "user" "Hi")
                                    (msg "assistant" "You said: Hi") (msg "user" "Again"))))

(test-assert "content parts are concatenated"
  (string-contains (render-chat-template
                    'qwen (list `(("role" . "user")
                                  ("content" . #((("type" . "text") ("text" . "ab"))
                                                 (("type" . "text") ("text" . "cd")))))))
                   "abcd"))

(let ((p (render-chat-template 'qwen (list (msg "user" "x")) #:tools (list weather-tool))))
  (test-assert "qwen tools: the # Tools section lists the schema in source order"
    (string-contains p "<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\""))
  (test-assert "tool_choice none hides the tools"
    (not (string-contains (render-chat-template 'qwen (list (msg "user" "x"))
                                                #:tools (list weather-tool)
                                                #:tool-choice "none")
                          "<tools>"))))

(let ((p (render-chat-template
          'gemma
          (list (msg "user" "weather in Paris?")
                `(("role" . "assistant") ("content" . null)
                  ("tool_calls" . #((("id" . "call_1") ("type" . "function")
                                     ("function" . (("name" . "get_weather")
                                                    ("arguments" . "{\"city\":\"Paris\"}")))))))
                `(("role" . "tool") ("tool_call_id" . "call_1") ("content" . "sunny")))
          #:tools (list weather-tool))))
  (test-assert "assistant tool calls are replayed as the model emits them"
    (string-contains p (string-append "<|turn>model\n"
                                      (format-tool-call "get_weather" "{\"city\":\"Paris\"}")
                                      "<turn|>")))
  (test-assert "tool results go back in a user turn"
    (string-contains p "<|turn>user\n<tool_response>\nsunny\n</tool_response><turn|>")))

(test-equal "parse-tool-call"
  '("get_weather" . "{\"city\":\"Paris\"}")
  (parse-tool-call "\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\"}}\n"))
(test-equal "parse-tool-call rejects non-calls" #f (parse-tool-call "not json"))

;;; --- mock tokenizer and model -----------------------------------------------------

(test-equal "special tokens are single tokens"
  3 (length (tokenize gemma "<|turn>ab")))
(test-equal "tokenize / detokenize round trip"
  "<bos>héllo <turn|>"
  (detokenize gemma (tokenize gemma "<bos>héllo <turn|>")))
;; tokenize-chat is the strict version: only the template's own markers are
;; matched, so a marker a user typed is character tokens, not a control token.
(let* ((segs (render-chat-template-segments 'gemma (list (msg "user" "say <turn|> now"))))
       (toks (tokenize-chat gemma segs)))
  (test-equal "the template's markers are their own tokens"
    '(0 2 3 2)                          ; <bos>, <|turn>, <turn|>, <|turn>
    (filter (lambda (t) (< t 256)) toks))
  (test-equal "a marker in the message is character tokens"
    (apply + (map (lambda (s) (string-length (cdr s)))
                  (filter (lambda (s) (eq? (car s) 'text)) segs)))
    (count (lambda (t) (>= t 256)) toks)))
(test-equal "greedy reply echoes the last user line"
  "You said: second line"
  (generate gemma (render-chat gemma (list (msg "user" "first line\nsecond line")))))
(test-equal "the qwen model too"
  "You said: Hi"
  (generate qwen (render-chat qwen (list (msg "user" "Hi")))))
(test-equal "a tool is called when the user names it"
  (format-tool-call "get_weather" "{\"city\":\"Paris\"}")
  (generate qwen (render-chat qwen (list (msg "user" "What is the weather in Paris?"))
                              #:tools (list weather-tool))))
(test-equal "a named tool_choice forces the call"
  (format-tool-call "get_weather" "{\"city\":\"Hi\"}")
  (generate qwen (render-chat qwen (list (msg "user" "Hi")) #:tools (list weather-tool)
                              #:tool-choice "get_weather")))
(define trip-tool
  (json-normalize
   (json-string->scm
    "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\",\"days\"]}}}")))
(test-equal "integer arguments take the first number"
  (format-tool-call "get_weather" "{\"city\":\"Oslo\",\"days\":3}")
  (generate gemma (render-chat gemma (list (msg "user" "weather 3 days in Oslo"))
                               #:tools (list trip-tool))))

;;; --- sampler and constraint ----------------------------------------------------------

(define hot-prompt (render-chat gemma (list (msg "user" "the quick brown fox"))))
(test-equal "same seed, same sample"
  (generate gemma hot-prompt #:sampler (make-sampler #:temperature 2.0 #:seed 7))
  (generate gemma hot-prompt #:sampler (make-sampler #:temperature 2.0 #:seed 7)))
(test-assert "a hot sampler strays from the plan"
  (not (equal? (generate gemma hot-prompt #:sampler (make-sampler #:temperature 2.0 #:seed 7))
               "You said: the quick brown fox")))
(test-equal "top_k 1 is greedy"
  "You said: the quick brown fox"
  (generate gemma hot-prompt #:sampler (make-sampler #:temperature 2.0 #:top-k 1 #:seed 7)))
(test-equal "a tiny top_p is greedy"
  "You said: the quick brown fox"
  (generate gemma hot-prompt #:sampler (make-sampler #:temperature 2.0 #:top-p 0.01 #:seed 7)))
(test-assert "the constraint hook filters candidates"
  (let ((no-y (lambda (cands) (remove (lambda (t) (equal? (tokenize gemma "Y") (list t))) cands))))
    (not (string-prefix? "Y" (generate gemma hot-prompt #:constraint no-y)))))

;;; --- prefill and the session pool --------------------------------------------------

(let* ((s (create-session gemma))
       (a (tokenize gemma "<bos>abcdef"))
       (b (tokenize gemma "<bos>abcxyz")))
  (test-equal "prefill of a fresh session reuses nothing" 0 (prefill s a))
  (test-equal "prefill reuses the longest common prefix" 4 (prefill s b))
  (test-equal "the same tokens again: all reused" 7 (prefill s b)))

(let* ((pool (make-session-pool gemma #:max-sessions 2 #:token-budget 1000))
       (toks (lambda (s) (list->vector (tokenize gemma s))))
       (use! (lambda (s)
               (let ((e (pool-acquire! pool (toks s))))
                 (prefill (pool-entry-session e) (vector->list (toks s)))
                 (pool-release! pool e (toks s))
                 (pool-entry-match e)))))
  (test-equal "a fresh pool: no match" 0 (use! "<bos>system A: hello"))
  ;; Shares "<bos>system " (8 tokens) with A: forked, A keeps its suffix.
  (test-equal "a partial match is forked" 8 (use! "<bos>system B: hello"))
  (test-equal "the forked-from session kept its continuation"
    16 (use! "<bos>system A: hello again"))
  (test-equal "fork stats" '(2 1)
    (list (assq-ref (pool-stats pool) 'sessions) (assq-ref (pool-stats pool) 'forks)))
  ;; No room left: the best match (by <bos>, LRU on ties: B) is taken over.
  (use! "<bos>another prefix")
  (test-equal "max-sessions holds" 2 (assq-ref (pool-stats pool) 'sessions))
  (test-equal "the LRU session was taken over" 1 (assq-ref (pool-stats pool) 'takeovers))
  (test-equal "the recently used one survived" 22 (use! "<bos>system A: hello again!"))
  (let ((pool (make-session-pool gemma #:max-sessions 4 #:token-budget 30)))
    (for-each (lambda (s)
                (let ((e (pool-acquire! pool (toks s))))
                  (pool-release! pool e (toks s))))
              '("<bos>0123456789" "<bos>abcdefghij" "<bos>ABCDEFGHIJ"))
    (test-assert "the token budget evicts LRU sessions"
      (and (<= (assq-ref (pool-stats pool) 'held-tokens) 30)
           (= 1 (assq-ref (pool-stats pool) 'evictions))))))

;;; --- engine ------------------------------------------------------------------------

(define engine (make-engine (list gemma qwen) #:queue-limit 1))

(define* (run-engine messages #:rest opts)
  (let ((pieces '()) (calls '()))
    (let ((r (engine-generate engine
                              (apply make-chat-request #:model "g" #:messages messages
                                     #:temperature 0 opts)
                              (lambda (kind . args)
                                (match kind
                                  ('content (set! pieces (cons (car args) pieces)))
                                  ('tool-call (set! calls (cons args calls))))
                                #t))))
      (list r (reverse pieces) (reverse calls)))))

(match (run-engine (list (msg "user" "Hello world")) #:stop '("world"))
  ((r pieces _)
   (test-equal "stop string ends the reply, not emitted" "You said: Hello " (result-content r))
   (test-equal "finish_reason stop" "stop" (result-finish-reason r))
   (test-assert "no piece carries a partial stop string"
     (not (any (lambda (p) (string-contains p "w")) pieces)))))

(match (run-engine (list (msg "user" "Hello")) #:max-tokens 4)
  ((r _ _)
   (test-equal "max_tokens" "You " (result-content r))
   (test-equal "finish_reason length" "length" (result-finish-reason r))
   (test-equal "completion tokens" 4 (result-completion-tokens r))))

(match (run-engine (list (msg "user" "weather in Paris")) #:tools (list weather-tool))
  ((r pieces calls)
   (test-equal "tool call: finish_reason tool_calls" "tool_calls" (result-finish-reason r))
   (test-equal "tool call: no content" "" (result-content r))
   (test-equal "tool call event" '(0 "get_weather" "{\"city\":\"Paris\"}")
     (match calls (((i id name args)) (list i name args)) (x x)))))

(match (run-engine (list (msg "user" "Hello")) #:max-tokens 3)
  ((r _ _)
   (match (run-engine (list (msg "user" "Hello")) #:max-tokens 3)
     ((r2 _ _)
      (test-assert "a repeated prompt is served from the cache"
        (= (result-cached-tokens r2) (result-prompt-tokens r2)))))))

(test-equal "cancellation: a #f from on-event stops"
  "cancelled"
  (result-finish-reason
   (engine-generate engine (make-chat-request #:model "g" #:messages (list (msg "user" "Hello"))
                                              #:temperature 0)
                    (lambda _ #f))))

(let* ((small (load-model backend '((id . "s") (family . gemma) (context-length . 40))))
       (e (make-engine (list small))))
  (define (go text)
    (catch 'context-length-exceeded
      (lambda ()
        (result-finish-reason
         (engine-generate e (make-chat-request #:model "s" #:messages (list (msg "user" text))
                                               #:temperature 0)
                          (lambda _ #t))))
      (lambda _ 'exceeded)))
  (test-equal "a prompt over the context length is refused" 'exceeded
    (go "this prompt is definitely longer than forty tokens in total"))
  (test-equal "the reply stops at the context length" "length" (go "short one here")))

(let* ((t1 (engine-admit! engine))
       (t2 (engine-admit! engine)))
  ;; t1 is first in line (nothing runs), t2 waits: the queue holds 1 waiter.
  (test-assert "the queue limit rejects"
    (catch 'queue-full (lambda () (engine-admit! engine) #f) (lambda _ #t)))
  (engine-drop! engine t1)
  (engine-drop! engine t2)
  (test-equal "dropped tickets leave the queue" 0 (assq-ref (engine-stats engine) 'waiting)))

;;; --- incremental detokenization ------------------------------------------------

;; A backend whose detokenizer is not character-aligned. Token 101 is the first
;; byte of the three-byte character "中", tokens 102 and 103 the rest; token 104
;; renders "world" when it opens the token list and " world" after another
;; token. The engine decodes only the window since its last output, so this
;; catches a window that starts inside a character or loses the context of a
;; token it re-decodes.
(define (stream-piece-bytes toks)
  (let lp ((ts toks) (first? #t) (out '()))
    (if (null? ts)
        (apply append (reverse out))
        (lp (cdr ts) #f
            (cons (case (car ts)
                    ((100) (map char->integer (string->list "Hello")))
                    ((104) (map char->integer (string->list (if first? "world" " world"))))
                    ((101) '(#xe4))
                    ((102) '(#xb8))
                    ((103) '(#xad))
                    (else '()))
                  out)))))

;; Length of the longest prefix of BYTES that ends at a character boundary.
(define (stream-complete-length bytes)
  (let* ((n (length bytes))
         (bv (u8-list->bytevector bytes))
         (start (let lp ((i (1- n)))
                  (if (< i 0)
                      -1
                      (if (>= (bytevector-u8-ref bv i) #xc0) i (lp (1- i)))))))
    (if (< start 0)
        n
        (let* ((b (bytevector-u8-ref bv start))
               (need (cond ((< b #xe0) 2) ((< b #xf0) 3) ((< b #xf8) 4) (else 1))))
          (if (>= (- n start) need) n start)))))

(define (stream-detokenize model toks)
  (let* ((bytes (stream-piece-bytes toks))
         (n (stream-complete-length bytes)))
    (if (= n (length bytes))
        (utf8->string (u8-list->bytevector bytes))
        (string-append (utf8->string (u8-list->bytevector (take bytes n)))
                       "\ufffd"))))

(define (stream-backend)
  (make-backend
   #:name "stream"
   #:load-model (lambda (config) 'stream-model)
   #:model-info (lambda (m)
                  '((id . "s") (family . gemma) (context-length . 8192)
                    (eos-tokens . (200)) (vocab-size . 10000)))
   #:tokenize (lambda (m text) (list 1))
   #:detokenize stream-detokenize
   #:render-chat (lambda (m messages tools tool-choice)
                   (list (cons 'text "prompt")))
   #:create-session (lambda (m) (vector (list 100 104 101 102 103 200)))
   #:free-session (lambda (session) #t)
   #:prefill (lambda (session toks constraint) 0)
   #:step (lambda (session sampler constraint)
            (let ((rest (vector-ref session 0)))
              (if (null? rest)
                  #f
                  (begin (vector-set! session 0 (cdr rest)) (car rest)))))
   #:generate (lambda (session sampler constraint max on-token)
                (let lp ((ts '(100 104 101 102 103 200)) (n 0))
                  (if (or (null? ts) (>= n max))
                      n
                      (if (on-token (car ts))
                          (lp (cdr ts) (1+ n))
                          n))))))

(let* ((stream-engine (make-engine (list (load-model (stream-backend)
                                                     '((id . "s"))))))
       (pieces '())
       (r (engine-generate
           stream-engine
           (make-chat-request #:model "s"
                              #:messages (list (msg "user" "hi"))
                              #:temperature 0)
           (lambda (kind . args)
             (when (eq? kind 'content)
               (set! pieces (cons (car args) pieces)))
             #t))))
  (test-equal "a reply whose character spans tokens streams as one detokenize"
    "Hello world中"
    (apply string-append (reverse pieces)))
  (test-equal "the streamed text is the result content" "Hello world中"
    (result-content r)))

;; The same window rules hold for the token-at-a-time path.
(let* ((stream-engine (make-engine (list (load-model (stream-backend)
                                                     '((id . "s"))))))
       (pieces '()))
  (setenv "SONICBOOM_SERVE_STEP" "1")
  (engine-generate
   stream-engine
   (make-chat-request #:model "s" #:messages (list (msg "user" "hi"))
                      #:temperature 0)
   (lambda (kind . args)
     (when (eq? kind 'content) (set! pieces (cons (car args) pieces)))
     #t))
  (unsetenv "SONICBOOM_SERVE_STEP")
  (test-equal "the step path streams the same text"
    "Hello world中"
    (apply string-append (reverse pieces))))

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-serve-units")
(exit (if (zero? failures) 0 1))
