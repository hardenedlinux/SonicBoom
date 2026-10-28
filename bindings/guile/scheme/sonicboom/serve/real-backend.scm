;;; SonicBoom — inference service: the libsonicboom GGUF backend
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>

(define-module (sonicboom serve real-backend)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-34)
  #:use-module (sonicboom error)
  #:use-module ((sonicboom inference) #:prefix inference:)
  #:use-module (sonicboom serve backend)
  #:use-module (sonicboom serve chat-template)
  #:export (make-real-backend))

(define-record-type <real-model>
  (make-real-model id handle context-length family eos-tokens)
  real-model?
  (id real-model-id)
  (handle real-model-handle)
  (context-length real-model-context-length)
  (family real-model-family)
  (eos-tokens real-model-eos-tokens))

(define-record-type <real-session>
  (make-real-session model handle history)
  real-session?
  (model real-session-model)
  (handle real-session-handle)
  (history real-session-history set-real-session-history!))

;; The GGUF architecture the C ABI reports ("gemma4", "qwen2") -> the chat
;; template family. This is the ONE place the two names are matched: everything
;; downstream (prompt formatting, turn markers, stop tokens) is driven by the
;; family the loaded model selected, not by the file name or a flag.
(define (family-for-arch arch)
  (cond ((string=? arch "gemma4") 'gemma)
        ((string=? arch "qwen2") 'qwen)
        (else (error "no chat template for this model architecture" arch))))

(define (load-real-model config)
  (let* ((path (assq-ref config 'path))
         (id (or (assq-ref config 'id) path))
         (context-length (or (assq-ref config 'context-length) 8192))
         (handle (inference:load-model path)))
    ;; Everything after the load can throw (an architecture with no chat
    ;; template, a failed tokenizer call). The handle is not owned by a record
    ;; yet, so it has to be freed on the way out or the weights leak for the
    ;; lifetime of the process.
    (guard (c (else (inference:free-model! handle) (raise c)))
      (let* ((arch (inference:model-arch handle))
             (family (family-for-arch arch))
             ;; Stop tokens, two sources, unioned:
             ;;  - the model's own eos ids, read from its metadata (Gemma 4:
             ;;    <turn|>, Qwen2: <|im_end|>), which is what generation must
             ;;    stop on;
             ;;  - the family template's end-of-turn markers. They are the same
             ;;    control token for these two families, and the union keeps the
             ;;    turn-end guarantee if a model's metadata ever names something
             ;;    else.
             ;; Either way the markers go through the tokenizer's special
             ;; parsing, so each becomes the vocabulary's own control token; as
             ;; plain text they would be subword pieces and never match a
             ;; sampled token.
             (eos (append
                   (inference:model-eos handle)
                   (append-map (lambda (s)
                                 (cdr (inference:tokenize handle s #:special? #t)))
                               (template-eos-strings family)))))
        (make-real-model id handle context-length family
                         (delete-duplicates eos))))))

(define (real-model-info model)
  `((id . ,(real-model-id model))
    (family . ,(real-model-family model))
    (context-length . ,(real-model-context-length model))
    (eos-tokens . ,(real-model-eos-tokens model))))

(define (real-render-chat model messages tools tool-choice)
  (render-chat-template-segments (real-model-family model) messages
                                 #:tools tools #:tool-choice tool-choice))

;; The template's own markers ('special segments) are parsed as the vocabulary's
;; control tokens; everything else — the message text, including a "<turn|>" a
;; user typed — is tokenized as text, so it stays text.
(define (real-tokenize-chat model segments)
  (inference:tokenize-segments (real-model-handle model) segments))

(define (real-create-session model)
  (make-real-session model
                     (inference:make-session (real-model-handle model)
                                             (real-model-context-length model))
                     '()))

(define (real-free-session session)
  (inference:free-session! (real-session-handle session)))

;; A constraint that does not compile is the request's fault — the schema or
;; the grammar text came from the client — so it is rethrown as a typed error
;; and the route answers 400 on the offending parameter, not a 500.
(define (compile-constraint param thunk)
  (guard (c ((sonicboom-error? c)
             (throw 'invalid-constraint param (sonicboom-error-message c))))
    (thunk)))

;; Install (or clear) the session's grammar before a prefill: the core restarts
;; it there, and the first token sampled at the end of the prompt is masked like
;; every decode step. A tool-call constraint is the call object's schema wrapped
;; in the envelope the engine parses back out of the text.
(define (real-apply-constraint session constraint)
  (let ((handle (real-session-handle session)))
    (cond
     ((not constraint) (inference:clear-grammar! handle))
     ((constraint? constraint)
      (case (constraint-kind constraint)
        ((json-schema)
         (compile-constraint "response_format"
          (lambda ()
            (inference:set-json-schema! handle (constraint-text constraint)))))
        ((tool-call)
         (compile-constraint "tool_choice"
          (lambda ()
            (inference:set-grammar!
             handle
             (inference:json-schema->gbnf
              (constraint-text constraint)
              #:prefix (string-append tool-call-open "\n")
              #:suffix (string-append "\n" tool-call-close))))))
        ((gbnf) (inference:set-grammar! handle (constraint-text constraint)))
        (else (error "unknown constraint kind" (constraint-kind constraint)))))
     (else
      ;; A host token filter: this backend samples on the device, so there is
      ;; no candidate list to filter.
      (error "the libsonicboom backend cannot apply a procedure constraint")))))

(define (real-prefill session tokens constraint)
  (real-apply-constraint session constraint)
  (let ((reused (inference:prefill! (real-session-handle session) tokens)))
    (set-real-session-history! session tokens)
    reused))

(define (real-step session sampler constraint)
  ;; One sb_llm_decode call per token on the persistent session (prefill already
  ;; ran). The sampler's parameters go through on every call: the core keys its
  ;; draws by (seed, absolute position, history) and keeps that state in the
  ;; session, so a per-step decode follows the same stream as one decode call of
  ;; the whole reply. temperature 0 is the historical greedy path, unchanged.
  ;;
  ;; With a grammar the core masks every step and finishes the reply itself
  ;; (the stop token is not emitted), so an empty result means the constrained
  ;; output is complete: #f tells the engine there is nothing left to produce.
  (let* ((tokens (inference:generate-tokens
                  (real-session-handle session)
                  (real-session-history session) 1 #f
                  #:temperature (if (sampler-greedy? sampler)
                                    0.0
                                    (sampler-temperature sampler))
                  #:top-k (sampler-top-k sampler)
                  #:top-p (sampler-top-p sampler)
                  #:seed (or (sampler-seed sampler) 0)
                  #:presence-penalty (sampler-presence-penalty sampler)
                  #:frequency-penalty (sampler-frequency-penalty sampler)))
         (token (and (pair? tokens) (car tokens))))
    (cond
     (token
      (set-real-session-history! session
                                 (append (real-session-history session) (list token)))
      token)
     ((constraint? constraint) #f)
     (else (error "SonicBoom backend produced no token")))))

(define (make-real-backend)
  (make-backend
   #:name "libsonicboom"
   #:load-model load-real-model
   #:model-info real-model-info
   #:tokenize (lambda (model text)
                (inference:tokenize (real-model-handle model) text))
   #:detokenize (lambda (model tokens)
                  (inference:detokenize (real-model-handle model) tokens))
   #:render-chat real-render-chat
   #:tokenize-chat real-tokenize-chat
   #:create-session real-create-session
   #:free-session real-free-session
   #:prefill real-prefill
   #:step real-step
   ;; real-step passes the sampler to sb_llm_decode's sampled session path.
   #:sampling? #t))
