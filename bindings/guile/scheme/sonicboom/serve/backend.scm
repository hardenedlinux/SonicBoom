;;; SonicBoom — inference service: the backend protocol
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A backend is a record of procedures. The service (sessions, engine, the
;;; HTTP app) only talks to a backend through the procedures exported here,
;;; so the mock backend can be replaced by a libsonicboom one without touching
;;; anything above this module. docs/capi.md maps each
;;; procedure to the C ABI the libsonicboom backend will call.
;;;
;;;   load-model   config alist           -> model
;;;   tokenize     model string           -> list of token ids
;;;   detokenize   model token-ids        -> string
;;;   render-chat  model messages tools tool-choice -> prompt segments
;;;   tokenize-chat model segments        -> list of token ids (optional)
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
;;; request). A constraint is an optional filter for constrained decoding (tool
;;; calls, JSON output). Two forms exist:
;;;   - a procedure (candidates) -> candidates over a list of token ids,
;;;     returning the allowed ones. That is what a backend with host logits
;;;     uses, and what sample-from-logits applies; and
;;;   - a <constraint> record, which names a grammar as its source text (a GBNF
;;;     document or a JSON Schema). A backend whose decoder supports grammars
;;;     installs it on the session at prefill; one that does not ignores it.
;;; A <constraint> has no host-side filter (its filter field is #f), because
;;; compiling a grammar to a token mask needs the model's vocabulary.
;;;
;;; The grammar is applied from prefill on: prefill takes the constraint, so
;;; the first token sampled at the end of the prompt is already masked.

(define-module (sonicboom serve backend)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (sonicboom serve chat-template)
  #:export (make-backend backend? backend-name

            load-model model? model-backend model-handle
            model-id model-family model-context-length model-eos-tokens
            model-info model-sampling?
            tokenize detokenize render-chat render-chat-segments tokenize-chat
            create-session free-session fork-session can-fork?
            session? session-model session-handle
            prefill step backend-stats

            make-sampler sampler? sampler-temperature sampler-top-p
            sampler-top-k sampler-seed sampler-greedy? sampler-random-state
            sample-from-logits

            make-json-schema-constraint make-tool-call-constraint
            make-gbnf-constraint constraint? constraint-kind constraint-text
            constraint-filter

            common-prefix-length))

;;; ---------------------------------------------------------------------------
;;; The backend record

(define-record-type <backend>
  (%make-backend name load-model model-info tokenize detokenize render-chat
                 tokenize-chat
                 create-session free-session fork-session prefill step stats
                 sampling?)
  backend?
  (name backend-name)
  (load-model %backend-load-model)
  (model-info %backend-model-info)
  (tokenize %backend-tokenize)
  (detokenize %backend-detokenize)
  (render-chat %backend-render-chat)
  (tokenize-chat %backend-tokenize-chat)
  (create-session %backend-create-session)
  (free-session %backend-free-session)
  (fork-session %backend-fork-session)
  (prefill %backend-prefill)
  (step %backend-step)
  (stats %backend-stats)
  (sampling? backend-sampling?))

(define (required who)
  (lambda _ (error "backend procedure not provided:" who)))

;; Every procedure receives the backend's own handles (model handle, session
;; handle), never the wrapper records below.
;;   load-model     (config)                       -> model handle
;;   model-info     (model)                        -> alist, see model-info
;;   tokenize       (model string)                 -> token ids
;;   detokenize     (model token-ids)              -> string
;;   render-chat    (model messages tools tool-choice) -> segments
;;   tokenize-chat  (model segments)               -> token ids (optional)
;;                  The default tokenizes the segments concatenated, i.e. plain
;;                  text throughout: a backend that owns a real tokenizer
;;                  instead parses the markers it rendered itself (the 'special
;;                  segments) and treats the message text as text.
;;   create-session (model)                        -> session handle
;;   free-session   (session)                      -> unspecified
;;   fork-session   (session n)                    -> session handle, or #f
;;                  for a backend that can't copy a KV prefix (optional)
;;   prefill        (session token-ids constraint) -> reused count
;;   step           (session sampler constraint)   -> token id, or #f when the
;;                  constraint has closed the reply (nothing left to produce)
;;   stats          (model)                        -> alist (optional)
;; The sampling? capability says whether `step` honors the sampler's
;; temperature / top-p / top-k. The libsonicboom backend decodes one token at a
;; time through its session and returns #f here: its step ignores the sampler,
;; and the HTTP layer then rejects an explicit sampling request instead of
;; silently decoding greedily. (The core's sb_llm_decode can sample when the
;; whole reply is decoded in one call; that path is not what this per-step
;; backend uses.) It defaults to #f, so a backend has to claim it.
(define* (make-backend #:key (name "backend")
                       (load-model (required 'load-model))
                       (model-info (required 'model-info))
                       (tokenize (required 'tokenize))
                       (detokenize (required 'detokenize))
                       (render-chat (required 'render-chat))
                       (tokenize-chat #f)
                       (create-session (required 'create-session))
                       (free-session (required 'free-session))
                       (fork-session #f)
                       (prefill (required 'prefill))
                       (step (required 'step))
                       (stats (lambda (model) '()))
                       (sampling? #f))
  (%make-backend name load-model model-info tokenize detokenize render-chat
                 tokenize-chat
                 create-session free-session fork-session prefill step stats
                 sampling?))

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

;; Whether this model's backend can sample (see make-backend's sampling?).
(define (model-sampling? model)
  (backend-sampling? (model-backend model)))

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
;; Returns the prompt as a segment list: (kind . text) with kind 'special for
;; the template's own turn markers and 'text for everything that came from a
;; message (see (sonicboom serve chat-template)).
(define* (render-chat-segments model messages #:key (tools '()) (tool-choice "auto"))
  ((%backend-render-chat (model-backend model))
   (model-handle model) messages tools tool-choice))

;; The same rendered prompt as one string (for logs and tests).
(define* (render-chat model messages #:key (tools '()) (tool-choice "auto"))
  (segments->string
   (render-chat-segments model messages #:tools tools #:tool-choice tool-choice)))

;; A rendered prompt's tokens. A backend that owns a real tokenizer tokenizes
;; the rendered segments itself — the 'special ones with special parsing, the
;; rest as text — so a turn marker is one control token while a marker inside a
;; user's message stays text. Without one, the segments are concatenated and
;; tokenized as plain text, which is what every prompt used to be.
(define (tokenize-chat model segments)
  (let ((proc (%backend-tokenize-chat (model-backend model))))
    (if proc
        (proc (model-handle model) segments)
        (tokenize model (segments->string segments)))))

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

(define* (prefill session tokens #:optional (constraint #f))
  ((%backend-prefill (model-backend (session-model session)))
   (session-handle session) tokens constraint))

(define* (step session sampler #:optional (constraint #f))
  ((%backend-step (model-backend (session-model session)))
   (session-handle session) sampler constraint))

(define (backend-stats model)
  ((%backend-stats (model-backend model)) (model-handle model)))

;;; ---------------------------------------------------------------------------
;;; Constraints
;;;
;;; kind is one of
;;;   'json-schema  text is the JSON Schema the whole reply must be a value of;
;;;   'tool-call    text is the JSON Schema of one call object ({name,
;;;                 arguments}); the backend adds the <tool_call> envelope;
;;;   'gbnf         text is the grammar itself.
;;; filter is #f for every grammar constraint: only the tokenizer's vocabulary
;;; can turn a grammar into a candidate list.

(define-record-type <constraint>
  (%make-constraint kind text filter)
  constraint?
  (kind constraint-kind)
  (text constraint-text)
  (filter constraint-filter))

(define (make-json-schema-constraint schema-json)
  "The reply must be one JSON value of SCHEMA-JSON."
  (%make-constraint 'json-schema schema-json #f))

(define (make-tool-call-constraint call-schema-json)
  "The reply must be one <tool_call> envelope whose object is a value of
CALL-SCHEMA-JSON."
  (%make-constraint 'tool-call call-schema-json #f))

(define (make-gbnf-constraint gbnf)
  "The reply must be text the GBNF grammar accepts."
  (%make-constraint 'gbnf gbnf #f))

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
  ;; A grammar constraint carries no host filter: a backend that cannot apply
  ;; grammars samples unconstrained rather than rejecting every token.
  (let ((keep (if (constraint? constraint)
                  (constraint-filter constraint)
                  constraint)))
    (let* ((allowed (if keep
                      (let ((ok (keep (map car logits))))
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
                  (lp (cdr ns) acc)))))))))

;;; ---------------------------------------------------------------------------

;; Length of the longest common prefix of two token vectors.
(define (common-prefix-length a b)
  (let ((n (min (vector-length a) (vector-length b))))
    (let lp ((i 0))
      (if (and (< i n) (eqv? (vector-ref a i) (vector-ref b i)))
          (lp (1+ i))
          i))))
