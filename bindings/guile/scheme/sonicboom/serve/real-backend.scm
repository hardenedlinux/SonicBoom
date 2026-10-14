;;; SonicBoom — inference service: the libsonicboom GGUF backend
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>

(define-module (sonicboom serve real-backend)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module ((sonicboom inference) #:prefix inference:)
  #:use-module (sonicboom serve backend)
  #:use-module (sonicboom serve chat-template)
  #:export (make-real-backend))

(define-record-type <real-model>
  (make-real-model id handle context-length eos-tokens)
  real-model?
  (id real-model-id)
  (handle real-model-handle)
  (context-length real-model-context-length)
  (eos-tokens real-model-eos-tokens))

(define-record-type <real-session>
  (make-real-session model handle history)
  real-session?
  (model real-session-model)
  (handle real-session-handle)
  (history real-session-history set-real-session-history!))

(define (load-real-model config)
  (let* ((path (assq-ref config 'path))
         (id (or (assq-ref config 'id) path))
         (context-length (or (assq-ref config 'context-length) 8192))
         (handle (inference:load-model path))
         ;; TODO: Read tokenizer.chat_template and EOS IDs from GGUF metadata.
         ;; This service currently serves Gemma 4 with its native turn tokens.
         (eos (append (cdr (inference:tokenize handle "<end_of_turn>"))
                      (cdr (inference:tokenize handle "<eos>")))))
    (make-real-model id handle context-length (delete-duplicates eos))))

(define (real-model-info model)
  `((id . ,(real-model-id model))
    (family . gemma)
    (context-length . ,(real-model-context-length model))
    (eos-tokens . ,(real-model-eos-tokens model))))

(define (real-render-chat model messages tools tool-choice)
  (render-chat-template 'gemma messages #:tools tools #:tool-choice tool-choice))

(define (real-create-session model)
  (make-real-session model
                     (inference:make-session (real-model-handle model)
                                             (real-model-context-length model))
                     '()))

(define (real-free-session session)
  (inference:free-session! (real-session-handle session)))

(define (real-prefill session tokens)
  (let ((reused (inference:prefill! (real-session-handle session) tokens)))
    (set-real-session-history! session tokens)
    reused))

(define (real-step session sampler constraint)
  (let* ((tokens (inference:generate-tokens (real-session-handle session)
                                            (real-session-history session) 1 #f))
         (token (and (pair? tokens) (car tokens))))
    (unless token (error "SonicBoom backend produced no token"))
    (set-real-session-history! session
                               (append (real-session-history session) (list token)))
    token))

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
   #:create-session real-create-session
   #:free-session real-free-session
   #:prefill real-prefill
   #:step real-step))
