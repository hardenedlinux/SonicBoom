;;; SonicBoom — inference service: a deterministic mock backend
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A stand-in for the libsonicboom backend, for testing the service. It
;;; implements the whole protocol of (sonicboom serve backend):
;;;
;;; Tokenizer   one token per character (id = 256 + code point); the special
;;;             tokens of the family (and <tool_call>, </tool_call>) are whole
;;;             tokens with ids below 256.
;;; Model       the reply is planned from the prompt text when the first step
;;;             after a prefill runs, by fixed rules:
;;;               - the last user turn is tool results -> "Tool result: <last>"
;;;               - tools are offered and the last user line names one of
;;;                 them (its name, or the last part of a snake_case name), or
;;;                 the prompt demands a call -> one tool call, with arguments
;;;                 filled from the user line
;;;               - otherwise -> "You said: <last line of the last user turn>"
;;;             then the end-of-turn token. At each step the planned token
;;;             has logit 8; the successors of the previous character in a
;;;             small character bigram (Markov) table get log(1 + count), at
;;;             most about 3. So greedy decoding (temperature 0) gives exactly
;;;             the plan, and sampling with a seed is reproducible but may
;;;             stray from it.
;;; Sessions    a session holds its token vector; prefill keeps the longest
;;;             common prefix with it and "computes" the rest; fork copies a
;;;             prefix into a new session. The reuse is counted in the stats.
;;; Latency     step-delay and prefill-delay (seconds per computed token) in
;;;             the config simulate the device time.

(define-module (sonicboom serve mock-backend)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (ice-9 threads)
  #:use-module (json)
  #:use-module (sonicboom serve backend)
  #:use-module (sonicboom serve chat-template)
  #:export (make-mock-backend mock-plan-text))

;;; ---------------------------------------------------------------------------
;;; Model handle

(define-record-type <mock-model>
  (make-mock-model id family specials eos context-length step-delay
                   prefill-delay markov stats lock)
  mock-model?
  (id mm-id)
  (family mm-family)
  (specials mm-specials)          ; vector of strings, index = token id
  (eos mm-eos)                    ; list of token ids
  (context-length mm-context-length)
  (step-delay mm-step-delay)
  (prefill-delay mm-prefill-delay)
  (markov mm-markov)              ; hash: char -> alist (char . count)
  (stats mm-stats)                ; hash: symbol -> integer
  (lock mm-lock))

(define (bump! model key n)
  (with-mutex (mm-lock model)
    (hashq-set! (mm-stats model) key (+ n (hashq-ref (mm-stats model) key 0)))))

(define char-base 256)

(define corpus
  "the quick brown fox jumps over the lazy dog. sonicboom is a native tensor \
runtime. you said the weather is fine today, and the tool returned a result. \
a small model answers in a predictable way so that the tests can check it.")

(define (build-markov text)
  (let ((table (make-hash-table)))
    (let lp ((cs (string->list text)))
      (match cs
        ((a b . rest)
         (let* ((row (hashv-ref table a '()))
                (cell (assv b row)))
           (hashv-set! table a (if cell
                                   (begin (set-cdr! cell (1+ (cdr cell))) row)
                                   (cons (cons b 1) row))))
         (lp (cons b rest)))
        (_ table)))))

(define (mock-load-model config)
  (define (cfg key default)
    (match (assq key config) ((_ . v) v) (#f default)))
  (let* ((family (cfg 'family 'gemma))
         (specials (list->vector (template-special-tokens family)))
         (index (lambda (s)
                  (let lp ((i 0))
                    (cond ((= i (vector-length specials)) #f)
                          ((string=? s (vector-ref specials i)) i)
                          (else (lp (1+ i))))))))
    (make-mock-model (cfg 'id (format #f "mock-~a" family))
                     family specials
                     (filter-map index (template-eos-strings family))
                     (cfg 'context-length 8192)
                     (cfg 'step-delay 0)
                     (cfg 'prefill-delay 0)
                     (build-markov corpus)
                     (make-hash-table)
                     (make-mutex))))

(define (mock-model-info model)
  `((id . ,(mm-id model))
    (family . ,(mm-family model))
    (context-length . ,(mm-context-length model))
    (eos-tokens . ,(mm-eos model))
    (vocab-size . ,(+ char-base #x110000))))

;;; ---------------------------------------------------------------------------
;;; Tokenizer

;; TEXT -> tokens, with special matching on or off. With SPECIAL? #f nothing is
;; matched, so every character is a character token — a control marker spelled
;; out in the text stays text.
(define (mock-tokenize-flags model text special?)
  (let ((specials (mm-specials model))
        (n (string-length text)))
    (define (special-at i)
      (and special?
           (char=? (string-ref text i) #\<)
           (let lp ((k 0))
             (cond ((= k (vector-length specials)) #f)
                   ((string-prefix? (vector-ref specials k) text 0
                                    (string-length (vector-ref specials k)) i)
                    k)
                   (else (lp (1+ k)))))))
    (let lp ((i 0) (out '()))
      (if (= i n)
          (reverse out)
          (match (special-at i)
            (#f (lp (1+ i) (cons (+ char-base (char->integer (string-ref text i))) out)))
            (k (lp (+ i (string-length (vector-ref specials k))) (cons k out))))))))

(define (mock-tokenize model text)
  (mock-tokenize-flags model text #t))

;; Like the real backend: the template's own markers ('special segments) are
;; their whole tokens; the message text is always text.
(define (mock-tokenize-chat model segments)
  (append-map (lambda (seg)
                (mock-tokenize-flags model (cdr seg) (eq? (car seg) 'special)))
              segments))

(define (token->string model t)
  (if (< t char-base)
      (vector-ref (mm-specials model) t)
      (string (integer->char (- t char-base)))))

(define (mock-detokenize model tokens)
  (string-concatenate (map (lambda (t) (token->string model t)) tokens)))

;;; ---------------------------------------------------------------------------
;;; The reply plan

(define (last-index-of text pat)
  (let lp ((from 0) (found #f))
    (match (string-contains text pat from)
      (#f found)
      (i (lp (1+ i) i)))))

(define (substring-between text open close)
  ;; The text between the last `open' and the next `close' (or the end).
  (match (last-index-of text open)
    (#f #f)
    (i (let* ((start (+ i (string-length open)))
              (end (or (string-contains text close start) (string-length text))))
         (substring text start end)))))

(define (user-turn-open family)
  (case family
    ((gemma) "<|turn>user\n")
    ((qwen) "<|im_start|>user\n")))

(define (turn-close family)
  (case family
    ((gemma) "<turn|>")
    ((qwen) "<|im_end|>")))

(define (last-line text)
  (let ((lines (filter (lambda (l) (not (string-null? (string-trim-both l))))
                       (string-split text #\newline))))
    (if (null? lines) "" (string-trim-both (last lines)))))

(define (offered-tools prompt)
  (match (substring-between prompt "<tools>\n" "\n</tools>")
    (#f '())
    (block
     (filter-map (lambda (line)
                   (catch #t
                     (lambda () (json-normalize (json-string->scm line)))
                     (lambda _ #f)))
                 (string-split block #\newline)))))

(define (ref obj . keys)
  (fold (lambda (k o) (and (list? o) (match (assoc k o) ((_ . v) v) (#f #f))))
        obj keys))

(define (words line)
  (filter (lambda (w) (not (string-null? w)))
          (map (lambda (w) (string-filter (lambda (c) (or (char-alphabetic? c)
                                                          (char-numeric? c)
                                                          (memv c '(#\- #\_))))
                                          w))
               (string-split line #\space))))

;; Fill the arguments from the user line: strings get the last word, numbers
;; the first number, booleans false. Only the first property and the required
;; ones are filled.
(define (tool-arguments tool line)
  (let* ((params (or (ref tool "function" "parameters") '()))
         (props (or (ref params "properties") '()))
         (required (match (ref params "required")
                     ((? vector? v) (vector->list v))
                     (_ '())))
         (ws (words line))
         (number (find string->number ws)))
    (filter-map
     (lambda (p i)
       (and (or (= i 0) (member (car p) required))
            (cons (car p)
                  (match (ref (cdr p) "type")
                    ("string" (if (null? ws) "" (last ws)))
                    ((or "number" "integer") (if number (string->number number) 0))
                    ("boolean" #f)
                    (_ 'null)))))
     props (iota (length props)))))

;; The line names the tool: its whole name, or the last part of a snake_case
;; name ("weather" for get_weather).
(define (mentions? line name)
  (let ((l (string-downcase line))
        (n (string-downcase name)))
    (or (string-contains l n)
        (string-contains l (last (string-split n #\_))))))

;; The reply for a prompt, as text (without the end-of-turn token).
(define (mock-plan-text family prompt)
  (let* ((turn (or (substring-between prompt (user-turn-open family) (turn-close family))
                   ""))
         (line (last-line turn))
         (tools (offered-tools prompt)))
    (cond
     ((string-prefix? "<tool_response>" (string-trim turn))
      (string-append "Tool result: "
                     (string-trim-both
                      (or (substring-between turn "<tool_response>\n" "\n</tool_response>")
                          ""))))
     ((and (pair? tools)
           (or (find (lambda (t)
                       (let ((name (ref t "function" "name")))
                         (or (string-contains prompt
                                              (string-append "You must call the function "
                                                             name "."))
                             (mentions? line name))))
                     tools)
               (and (string-contains prompt "You must call at least one function.")
                    (car tools))))
      => (lambda (tool)
           (format-tool-call (ref tool "function" "name")
                             (scm->json-string (tool-arguments tool line)))))
     (else (string-append "You said: " line)))))

;;; ---------------------------------------------------------------------------
;;; Sessions

(define-record-type <mock-session>
  (make-mock-session model tokens plan)
  mock-session?
  (model ms-model)
  (tokens ms-tokens set-ms-tokens!)   ; vector
  (plan ms-plan set-ms-plan!))        ; #f, or the remaining planned tokens

(define (mock-create-session model)
  (bump! model 'sessions-created 1)
  (make-mock-session model (vector) #f))

(define (mock-free-session s)
  (bump! (ms-model s) 'sessions-freed 1)
  (set-ms-tokens! s (vector)))

(define (mock-fork-session s n)
  (let ((model (ms-model s)))
    (bump! model 'sessions-created 1)
    (bump! model 'forks 1)
    (make-mock-session model
                       (vector-copy (ms-tokens s) 0 (min n (vector-length (ms-tokens s))))
                       #f)))

;; The mock has no vocabulary and so no grammar support; it ignores the
;; constraint. sample-from-logits still applies a procedure constraint in step.
(define (mock-prefill s tokens _constraint)
  (let* ((model (ms-model s))
         (new (list->vector tokens))
         (reused (common-prefix-length (ms-tokens s) new))
         (computed (- (vector-length new) reused)))
    (when (> (vector-length new) (mm-context-length model))
      (error "prompt exceeds the context length" (vector-length new)))
    (unless (zero? (mm-prefill-delay model))
      (usleep (inexact->exact (round (* 1e6 computed (mm-prefill-delay model))))))
    (set-ms-tokens! s new)
    (set-ms-plan! s #f)
    (bump! model 'prefill-calls 1)
    (bump! model 'prefill-tokens (vector-length new))
    (bump! model 'reused-tokens reused)
    (bump! model 'computed-tokens computed)
    reused))

(define (plan! s)
  (let ((model (ms-model s)))
    (or (ms-plan s)
        (let ((p (append (mock-tokenize model
                                        (mock-plan-text
                                         (mm-family model)
                                         (mock-detokenize model (vector->list (ms-tokens s)))))
                         (list (car (mm-eos model))))))
          (set-ms-plan! s p)
          p))))

(define (logits model plan prev)
  (let* ((target (car plan))
         (row (if (and prev (>= prev char-base))
                  (hashv-ref (mm-markov model) (integer->char (- prev char-base)) '())
                  '()))
         (noise (filter-map (lambda (cell)
                              (let ((t (+ char-base (char->integer (car cell)))))
                                (and (not (= t target))
                                     (cons t (log (1+ (cdr cell)))))))
                            row))
         (eos (car (mm-eos model))))
    (cons (cons target 8.0)
          (if (or (= target eos) (assv eos noise))
              noise
              (cons (cons eos 0.0) noise)))))

(define (mock-step s sampler constraint)
  (let* ((model (ms-model s))
         (plan (plan! s))
         (tokens (ms-tokens s))
         (prev (and (> (vector-length tokens) 0)
                    (vector-ref tokens (1- (vector-length tokens)))))
         (t (sample-from-logits sampler (logits model plan prev) constraint)))
    (unless (zero? (mm-step-delay model))
      (usleep (inexact->exact (round (* 1e6 (mm-step-delay model))))))
    (set-ms-plan! s (if (null? (cdr plan)) (list (car (mm-eos model))) (cdr plan)))
    (set-ms-tokens! s (let ((v (make-vector (1+ (vector-length tokens)) t)))
                        (vector-move-left! tokens 0 (vector-length tokens) v 0)
                        v))
    (bump! model 'steps 1)
    t))

(define (mock-render-chat model messages tools tool-choice)
  (render-chat-template-segments (mm-family model) messages
                                 #:tools tools #:tool-choice tool-choice))

(define (mock-stats model)
  (with-mutex (mm-lock model)
    (sort (hash-map->list cons (mm-stats model))
          (lambda (a b) (string<? (symbol->string (car a)) (symbol->string (car b)))))))

(define (make-mock-backend)
  (make-backend #:name "mock"
                #:load-model mock-load-model
                #:model-info mock-model-info
                #:tokenize mock-tokenize
                #:detokenize mock-detokenize
                #:render-chat mock-render-chat
                #:tokenize-chat mock-tokenize-chat
                #:create-session mock-create-session
                #:free-session mock-free-session
                #:fork-session mock-fork-session
                #:prefill mock-prefill
                #:step mock-step
                #:stats mock-stats
                ;; mock-step honors the sampler's temperature / top-p / top-k.
                #:sampling? #t))
