;;; SonicBoom — inference service: chat templates, written by hand per family
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; No Jinja: each model family has its template as a Scheme procedure.
;;;
;;;   gemma  <bos><|turn>user\n...<turn|>\n<|turn>model\n
;;;          Gemma 4's own turn tokens (<|turn>, <turn|>), which are control
;;;          tokens of its vocabulary. No system role: the system text (and the
;;;          tools block) opens the first user turn. The assistant role is
;;;          `model'.
;;;   qwen   ChatML: <|im_start|>ROLE\n...<|im_end|>\n, with the Qwen2.5 tool
;;;          convention (a # Tools section in the system turn).
;;;
;;; Tool calls use the same convention for both families: the tool signatures
;;; are listed inside <tools></tools>, one JSON object per line, and a call is
;;; <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>. Tool results
;;; go back in a user turn as <tool_response>\n...\n</tool_response>, one per
;;; consecutive tool message. Gemma has no tool tokens of its own; using the
;;; Qwen2.5 wording there is SonicBoom's convention.
;;;
;;; An assistant message is rendered exactly as the model generated it (the
;;; content, then the tool calls via format-tool-call), so a follow-up request
;;; that carries the previous reply shares its tokens with the cached session.
;;;
;;; A rendered prompt is a list of SEGMENTS, each (kind . text) with
;;;   'special  a turn marker of the template (or <bos>): tokenized with
;;;             special parsing, so it becomes the vocabulary's one control
;;;             token;
;;;   'text     text that came from a message, or SonicBoom's own wording:
;;;             tokenized as plain text, so a marker a user typed into a
;;;             message stays text and cannot forge a turn boundary.
;;; render-chat-template returns the concatenated string (for logs and tests);
;;; render-chat-template-segments returns the segments, which is what a backend
;;; has to tokenize (see (sonicboom serve backend) tokenize-chat). The
;;; model-level wrapper the service calls, render-chat-segments, lives in
;;; (sonicboom serve backend) — the two names differ so a module can import both.

(define-module (sonicboom serve chat-template)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-11)
  #:use-module (ice-9 match)
  #:use-module (json)
  #:export (render-chat-template render-chat-template-segments segments->string
            template-special-tokens template-eos-strings
            tool-call-open tool-call-close
            format-tool-call parse-tool-call json-normalize
            message-ref message-text))

(define tool-call-open "<tool_call>")
(define tool-call-close "</tool_call>")

;; --- segments ---------------------------------------------------------------

(define (seg kind text) (cons kind text))
(define (segments->string segments) (string-concatenate (map cdr segments)))

;; Special pieces of each family (the tokenizer keeps them whole when it parses
;; them), and the ones that end an assistant turn. A family's eos string must
;; be one of its special pieces — the end-of-turn marker is the model's own
;; control token, and only special parsing turns it into that one id.
(define (template-special-tokens family)
  (append
   (case family
     ((gemma) '("<bos>" "<eos>" "<|turn>" "<turn|>"))
     ((qwen) '("<|endoftext|>" "<|im_start|>" "<|im_end|>"))
     (else (error "unknown chat template family" family)))
   (list tool-call-open tool-call-close)))

(define (template-eos-strings family)
  (case family
    ((gemma) '("<turn|>" "<eos>"))
    ((qwen) '("<|im_end|>" "<|endoftext|>"))
    (else (error "unknown chat template family" family))))

;;; ---------------------------------------------------------------------------
;;; Messages (decoded JSON objects: alists with string keys)

;; json->scm gives object members in reverse order; restore the source order
;; (tool schemas are shown to the model as JSON, in the order they came).
(define (json-normalize x)
  (cond ((and (pair? x) (every pair? x))
         (reverse (map (lambda (p) (cons (car p) (json-normalize (cdr p)))) x)))
        ((vector? x) (list->vector (map json-normalize (vector->list x))))
        (else x)))

(define (message-ref msg key)
  (match (assoc key msg)
    ((_ . 'null) #f)
    ((_ . v) v)
    (#f #f)))

;; content is a string, null, or an array of parts; only text parts count.
(define (message-text msg)
  (match (message-ref msg "content")
    (#f "")
    ((? string? s) s)
    ((? vector? parts)
     (string-concatenate
      (filter-map (lambda (p)
                    (and (equal? (message-ref p "type") "text")
                         (message-ref p "text")))
                  (vector->list parts))))
    (x (error "unsupported message content" x))))

(define (role msg) (or (message-ref msg "role") ""))

;;; ---------------------------------------------------------------------------
;;; Tools

;; The text of one call, as the model emits it and as an assistant message
;; replays it. arguments is the JSON text of the arguments object.
(define (format-tool-call name arguments)
  (string-append tool-call-open "\n{\"name\": " (scm->json-string name)
                 ", \"arguments\": " arguments "}\n" tool-call-close))

;; The inside of <tool_call>...</tool_call> -> (name . arguments-json-text),
;; or #f when it isn't a call object.
(define (parse-tool-call text)
  (catch #t
    (lambda ()
      (let* ((obj (json-string->scm (string-trim-both text)))
             (name (and (list? obj) (message-ref obj "name")))
             (args (and (list? obj) (assoc "arguments" obj))))
        (and (string? name)
             (cons name
                   (match args
                     (#f "{}")
                     ((_ . (? string? s)) s)
                     ((_ . v) (scm->json-string v)))))))
    (lambda _ #f)))

(define (assistant-tool-calls msg)
  (match (message-ref msg "tool_calls")
    (#f '())
    (calls
     (map (lambda (c)
            (let ((f (message-ref c "function")))
              (format-tool-call (message-ref f "name")
                                (let ((a (message-ref f "arguments")))
                                  (if (string? a) a (scm->json-string (or a '())))))))
          (vector->list calls)))))

(define (assistant-text msg)
  (let ((content (message-text msg))
        (calls (assistant-tool-calls msg)))
    (string-join (filter (lambda (s) (not (string-null? s)))
                         (cons content calls))
                 "\n")))

(define (tools-block tools tool-choice)
  (string-append
   "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
   "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
   (string-join (map scm->json-string tools) "\n")
   "\n</tools>\n\n"
   "For each function call, return a json object with function name and arguments "
   "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
   "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
   (cond
    ((equal? tool-choice "required") "\n\nYou must call at least one function.")
    ((member tool-choice '("auto" "none")) "")
    (else (string-append "\n\nYou must call the function " tool-choice ".")))))

;; Consecutive tool messages become one user turn.
(define (tool-responses msgs)
  (string-join
   (map (lambda (m)
          (string-append "<tool_response>\n" (message-text m) "\n</tool_response>"))
        msgs)
   "\n"))

;; Split leading system messages off, and group consecutive tool messages.
(define (split-system messages)
  (let lp ((ms messages) (sys '()))
    (if (and (pair? ms) (member (role (car ms)) '("system" "developer")))
        (lp (cdr ms) (cons (message-text (car ms)) sys))
        (values (string-join (reverse sys) "\n\n") ms))))

(define (group-turns messages)
  ;; -> list of (role . text) with role one of user / assistant
  (let lp ((ms messages) (out '()))
    (match ms
      (() (reverse out))
      ((m . rest)
       (match (role m)
         ("tool"
          (let-values (((tools rest) (span (lambda (x) (equal? (role x) "tool")) ms)))
            (lp rest (cons (cons 'user (tool-responses tools)) out))))
         ("assistant" (lp rest (cons (cons 'assistant (assistant-text m)) out)))
         ((or "system" "developer")
          ;; A system message after the first turn: keep it as user text.
          (lp rest (cons (cons 'user (message-text m)) out)))
         (_ (lp rest (cons (cons 'user (message-text m)) out))))))))

;;; ---------------------------------------------------------------------------
;;; Families

(define (render-gemma messages tools tool-choice add-generation-prompt?)
  (let-values (((system rest) (split-system messages)))
    (let* ((preamble (string-join
                      (filter (lambda (s) (not (string-null? s)))
                              (list system
                                    (if (null? tools) "" (tools-block tools tool-choice))))
                      "\n\n"))
           (turns (group-turns rest))
           (turns (if (string-null? preamble)
                      turns
                      (match turns
                        ((('user . text) . more)
                         (cons (cons 'user (string-append preamble "\n\n" text)) more))
                        (_ (cons (cons 'user preamble) turns))))))
      (append
       (list (seg 'special "<bos>"))
       (append-map (match-lambda
                     ((r . text)
                      (list (seg 'special "<|turn>")
                            (seg 'text (if (eq? r 'user) "user\n" "model\n"))
                            (seg 'text text)
                            (seg 'special "<turn|>")
                            (seg 'text "\n"))))
                   turns)
       (if add-generation-prompt?
           (list (seg 'special "<|turn>") (seg 'text "model\n"))
           '())))))

(define (render-qwen messages tools tool-choice add-generation-prompt?)
  (let-values (((system rest) (split-system messages)))
    (let ((system (string-join
                   (filter (lambda (s) (not (string-null? s)))
                           (list system
                                 (if (null? tools) "" (tools-block tools tool-choice))))
                   "\n\n")))
      (append
       (if (string-null? system)
           '()
           (list (seg 'special "<|im_start|>") (seg 'text "system\n")
                 (seg 'text system) (seg 'special "<|im_end|>") (seg 'text "\n")))
       (append-map (match-lambda
                     ((r . text)
                      (list (seg 'special "<|im_start|>")
                            (seg 'text (if (eq? r 'user) "user\n" "assistant\n"))
                            (seg 'text text)
                            (seg 'special "<|im_end|>")
                            (seg 'text "\n"))))
                   (group-turns rest))
       (if add-generation-prompt?
           (list (seg 'special "<|im_start|>") (seg 'text "assistant\n"))
           '())))))

;; tools: list of tool objects; tool-choice: "none", "auto", "required" or a
;; function name. With "none" the tools are not shown to the model at all.
(define* (render-chat-template-segments family messages #:key (tools '())
                                        (tool-choice "auto")
                                        (add-generation-prompt? #t))
  (let ((tools (if (equal? tool-choice "none") '() tools)))
    (case family
      ((gemma) (render-gemma messages tools tool-choice add-generation-prompt?))
      ((qwen) (render-qwen messages tools tool-choice add-generation-prompt?))
      (else (error "unknown chat template family" family)))))

;; The same prompt as one string (the segments concatenated).
(define* (render-chat-template family messages #:key (tools '())
                               (tool-choice "auto")
                               (add-generation-prompt? #t))
  (segments->string
   (render-chat-template-segments family messages #:tools tools
                                  #:tool-choice tool-choice
                                  #:add-generation-prompt? add-generation-prompt?)))
