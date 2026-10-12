;;; SonicBoom — inference service: the backend protocol
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A backend is a record of procedures. The service (sessions, engine, the
;;; HTTP app) only talks to a backend through the procedures exported here,
;;; so the mock backend can be replaced by a libsonicboom one without touching
;;; anything above this module. design/route2-capi-draft.md maps each
;;; procedure to the C ABI the libsonicboom backend will call.
;;;
;;;   load-model   config alist           -> model
;;;   tokenize     model string           -> list of token ids
;;;   detokenize   model token-ids        -> string
;;;   render-chat  model messages tools tool-choice -> prompt string
;;;   create-session / free-session
;;;   fork-session session n             -> a new session holding the first n
;;;                                         tokens of session (optional)
;;;   prefill      session token-ids      -> number of reused tokens
;;;   step         session sampler constraint -> next token id
;;;
;;; A session is a KV cache: it holds a token sequence. prefill makes the
;;; session hold exactly the given tokens, keeping the longest common prefix
;;; of what it already holds and computing only the rest; it returns how many
;;; tokens it kept. step samples the next token after the held sequence, then
;;; appends it, so the following step continues from it.
;;;
;;; Sampling parameters and the random state live in a sampler (one per
;;; request). A constraint is an optional token filter for constrained
;;; decoding (tool calls, JSON output): a procedure
;;;   (constraint candidates) -> candidates
;;; over a list of candidate token ids, returning the allowed ones. A backend
;;; that samples on the device turns it into a mask; the reference sampler
;;; below applies it to its candidate list.

(define-module (sonicboom serve backend)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:export (make-backend backend? backend-name

            load-model model? model-backend model-handle
            model-id model-family model-context-length model-eos-tokens
            model-info
            tokenize detokenize render-chat
            create-session free-session fork-session can-fork?
            session? session-model session-handle
            prefill step backend-stats

            make-sampler sampler? sampler-temperature sampler-top-p
            sampler-top-k sampler-seed sampler-greedy? sampler-random-state
            sample-from-logits

            common-prefix-length))

;;; ---------------------------------------------------------------------------
;;; The backend record

(define-record-type <backend>
  (%make-backend name load-model model-info tokenize detokenize render-chat
                 create-session free-session fork-session prefill step stats)
  backend?
  (name backend-name)
  (load-model %backend-load-model)
  (model-info %backend-model-info)
  (tokenize %backend-tokenize)
  (detokenize %backend-detokenize)
  (render-chat %backend-render-chat)
  (create-session %backend-create-session)
  (free-session %backend-free-session)
  (fork-session %backend-fork-session)
  (prefill %backend-prefill)
  (step %backend-step)
  (stats %backend-stats))

(define (required who)
  (lambda _ (error "backend procedure not provided:" who)))

;; Every procedure receives the backend's own handles (model handle, session
;; handle), never the wrapper records below.
;;   load-model     (config)                       -> model handle
;;   model-info     (model)                        -> alist, see model-info
;;   tokenize       (model string)                 -> token ids
;;   detokenize     (model token-ids)              -> string
;;   render-chat    (model messages tools tool-choice) -> string
;;   create-session (model)                        -> session handle
;;   free-session   (session)                      -> unspecified
;;   fork-session   (session n)                    -> session handle, or #f
;;                  for a backend that can't copy a KV prefix (optional)
;;   prefill        (session token-ids)            -> reused count
;;   step           (session sampler constraint)   -> token id
;;   stats          (model)                        -> alist (optional)
(define* (make-backend #:key (name "backend")
                       (load-model (required 'load-model))
                       (model-info (required 'model-info))
                       (tokenize (required 'tokenize))
                       (detokenize (required 'detokenize))
                       (render-chat (required 'render-chat))
                       (create-session (required 'create-session))
                       (free-session (required 'free-session))
                       (fork-session #f)
                       (prefill (required 'prefill))
                       (step (required 'step))
                       (stats (lambda (model) '())))
  (%make-backend name load-model model-info tokenize detokenize render-chat
                 create-session free-session fork-session prefill step stats))

;;; ---------------------------------------------------------------------------
;;; Models and sessions

(define-record-type <model>
  (%make-model backend handle info)
  model?
  (backend model-backend)
  (handle model-handle)
  (info model-info))

;; The info alist has at least: id (string), family (symbol), context-length
;; (integer), eos-tokens (list of token ids that end a reply).
(define (info-ref model key)
  (match (assq key (model-info model))
    ((_ . v) v)
    (#f (error "model info lacks" key))))

(define (model-id model) (info-ref model 'id))
(define (model-family model) (info-ref model 'family))
(define (model-context-length model) (info-ref model 'context-length))
(define (model-eos-tokens model) (info-ref model 'eos-tokens))

(define (load-model backend config)
  (let ((handle ((%backend-load-model backend) config)))
    (%make-model backend handle ((%backend-model-info backend) handle))))

(define (tokenize model text)
  ((%backend-tokenize (model-backend model)) (model-handle model) text))

(define (detokenize model tokens)
  ((%backend-detokenize (model-backend model)) (model-handle model) tokens))

;; messages: list of message alists in the OpenAI shape, already decoded from
;; JSON (role, content, tool_calls, tool_call_id, name). tools: list of tool
;; alists or '(). tool-choice: "none", "auto", "required" or a function name.
(define* (render-chat model messages #:key (tools '()) (tool-choice "auto"))
  ((%backend-render-chat (model-backend model))
   (model-handle model) messages tools tool-choice))

(define-record-type <session>
  (%make-session model handle)
  session?
  (model session-model)
  (handle session-handle))

(define (create-session model)
  (%make-session model
                 ((%backend-create-session (model-backend model))
                  (model-handle model))))

(define (free-session session)
  ((%backend-free-session (model-backend (session-model session)))
   (session-handle session)))

(define (can-fork? model)
  (and (%backend-fork-session (model-backend model)) #t))

;; A new session holding the first n tokens of session, without recomputing
;; them (a KV-cache copy on the device).
(define (fork-session session n)
  (let ((model (session-model session)))
    (%make-session model
                   ((%backend-fork-session (model-backend model))
                    (session-handle session) n))))

(define (prefill session tokens)
  ((%backend-prefill (model-backend (session-model session)))
   (session-handle session) tokens))

(define* (step session sampler #:optional (constraint #f))
  ((%backend-step (model-backend (session-model session)))
   (session-handle session) sampler constraint))

(define (backend-stats model)
  ((%backend-stats (model-backend model)) (model-handle model)))

;;; ---------------------------------------------------------------------------
;;; Sampler: the parameters of one request, plus its random state

(define-record-type <sampler>
  (%make-sampler temperature top-p top-k seed greedy? random-state)
  sampler?
  (temperature sampler-temperature)
  (top-p sampler-top-p)
  (top-k sampler-top-k)
  (seed sampler-seed)
  (greedy? sampler-greedy?)
  (random-state sampler-random-state))

;; temperature 0 means greedy, like the OpenAI API. top-k 0 means no top-k.
;; Without a seed the random state is seeded from the clock.
(define* (make-sampler #:key (temperature 1.0) (top-p 1.0) (top-k 0)
                       (seed #f) (greedy? #f))
  (%make-sampler temperature top-p top-k seed
                 (or greedy? (<= temperature 0))
                 (seed->random-state
                  (or seed (+ (* 1000000 (car (gettimeofday)))
                              (cdr (gettimeofday)))))))

;; The reference sampler for backends that produce logits on the host.
;; logits: list of (token . logit). constraint: #f or a token filter.
;; Order: constraint, then greedy, else temperature, top-k, top-p, draw.
(define (sample-from-logits sampler logits constraint)
  (let* ((allowed (if constraint
                      (let ((ok (constraint (map car logits))))
                        (filter (lambda (p) (memv (car p) ok)) logits))
                      logits))
         (sorted (sort allowed (lambda (a b) (> (cdr a) (cdr b))))))
    (when (null? sorted)
      (error "the constraint rejected every candidate token"))
    (if (sampler-greedy? sampler)
        (caar sorted)
        (let* ((k (sampler-top-k sampler))
               (kept (if (and (> k 0) (< k (length sorted)))
                         (take sorted k)
                         sorted))
               (t (sampler-temperature sampler))
               (top (cdar kept))
               (weights (map (lambda (p) (exp (/ (- (cdr p) top) t))) kept))
               (total (apply + weights))
               (probs (map (lambda (w) (/ w total)) weights))
               (nucleus (let lp ((ps probs) (ts kept) (acc 0.0) (out '()))
                          ;; Keep the smallest prefix whose mass reaches top-p.
                          (if (or (null? ps)
                                  (and (pair? out)
                                       (>= acc (sampler-top-p sampler))))
                              (reverse out)
                              (lp (cdr ps) (cdr ts) (+ acc (car ps))
                                  (cons (cons (caar ts) (car ps)) out)))))
               (mass (apply + (map cdr nucleus)))
               (r (* mass (random:uniform (sampler-random-state sampler)))))
          (let lp ((ns nucleus) (acc 0.0))
            (let ((acc (+ acc (cdar ns))))
              (if (or (null? (cdr ns)) (< r acc))
                  (caar ns)
                  (lp (cdr ns) acc))))))))

;;; ---------------------------------------------------------------------------

;; Length of the longest common prefix of two token vectors.
(define (common-prefix-length a b)
  (let ((n (min (vector-length a) (vector-length b))))
    (let lp ((i 0))
      (if (and (< i n) (eqv? (vector-ref a i) (vector-ref b i)))
          (lp (1+ i))
          i))))
