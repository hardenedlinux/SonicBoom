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
;;;   unload-model model                  -> release its device memory (optional)
;;;   tokenize     model string           -> list of token ids
;;;   detokenize   model token-ids        -> string
;;;   render-chat  model messages tools tool-choice -> prompt segments
;;;   tokenize-chat model segments        -> list of token ids (optional)
;;;   create-session / free-session
;;;   fork-session session n             -> a new session holding the first n
;;;                                         tokens of session (optional)
;;;   prefill      session token-ids      -> number of reused tokens
;;;   step         session sampler constraint -> next token id
;;;   generate     session sampler constraint max-tokens on-token
;;;                -> number of tokens produced               (optional)
;;;
;;; generate decodes up to max-tokens in one call, handing each produced token
;;; to (on-token token) in order; on-token returns #t to continue and #f to stop
;;; the decode. A backend without generate keeps working through step.
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

            load-model unload-model
            model? model-backend model-handle
            model-id model-family model-arch model-context-length model-eos-tokens
            model-info model-sampling? backend-sampling?
            tokenize detokenize render-chat render-chat-segments tokenize-chat
            create-session free-session fork-session can-fork?
            session? session-model session-handle
            prefill step generate can-generate? backend-stats

            make-sampler sampler? sampler-temperature sampler-top-p
            sampler-top-k sampler-min-p sampler-seed sampler-repeat-penalty
            sampler-greedy? sampler-random-state
            sampler-presence-penalty sampler-frequency-penalty
            sample-from-logits

            make-json-schema-constraint make-tool-call-constraint
            make-gbnf-constraint constraint? constraint-kind constraint-text
            constraint-filter

            common-prefix-length))

;;; ---------------------------------------------------------------------------
;;; The backend record

;; A backend is the service's view of a model runtime.
;;
;; It is a name plus a record of procedures. Every procedure receives the
;; backend's own handles (a model handle, a session handle), so the mock backend
;; and the libsonicboom backend are interchangeable to the code above this module.
;;
;; Example:
;;   (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
;;   (backend? (make-mock-backend))
;;   => #t
;; PyTorch: none
(define-record-type <backend>
  (%make-backend name load-model unload-model model-info tokenize detokenize
                 render-chat tokenize-chat
                 create-session free-session fork-session prefill step
                 generate stats sampling?)
  backend?
  (name backend-name)
  (load-model %backend-load-model)
  (unload-model %backend-unload-model)
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
  (generate %backend-generate)
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
;;   generate       (session sampler constraint max-tokens on-token)
;;                  -> number of tokens produced (optional)
;;                  Decodes in one call, calling (on-token token) per produced
;;                  token; #f from on-token stops the decode. The session
;;                  history ends at the last token handed out.
;;   stats          (model)                        -> alist (optional)
;; The sampling? capability says whether `step` honors the sampler's
;; temperature / top-p / top-k / seed and penalties. The libsonicboom backend
;; claims it: each `step` is one boom_llm_decode call on the persistent session,
;; and the core's sampled path (temperature > 0) keeps its draw state across
;; those calls, so a per-step decode samples exactly like a one-shot decode.
;; A backend that cannot sample returns #f (the default), and the HTTP layer
;; then rejects an explicit sampling request instead of silently decoding
;; greedily.
(define* (make-backend #:key (name "backend")
                       (load-model (required 'load-model))
                       (unload-model #f)
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
                       (generate #f)
                       (stats (lambda (model) '()))
                       (sampling? #f))
  "Create a backend from procedures.

load-model, model-info, tokenize, detokenize, render-chat, create-session,
free-session, prefill and step are required; the other keywords have defaults.
A backend that supplies generate produces a whole reply in one call; one that
does not is driven token by token through step.

Args:
  name: keyword; the backend's name (default \"backend\").
  load-model: keyword; (config) -> a model handle.
  unload-model: keyword; (handle) -> unspecified, releasing the handle so its
    device memory goes back to the driver; the default does nothing (default
    #f).
  model-info: keyword; (handle) -> an info alist (see the model record).
  tokenize: keyword; (handle string) -> a list of token ids.
  detokenize: keyword; (handle token-ids) -> a string.
  render-chat: keyword; (handle messages tools tool-choice) -> prompt segments.
  tokenize-chat: keyword; (handle segments) -> token ids, or #f to concatenate
    the segments and call tokenize (default #f).
  create-session: keyword; (handle) -> a session handle.
  free-session: keyword; (session-handle) -> unspecified.
  fork-session: keyword; (session-handle n) -> a new handle holding the first n
    tokens, or #f when the backend cannot copy a KV prefix (default #f).
  prefill: keyword; (session-handle token-ids constraint) -> the number of
    tokens kept from the previous contents.
  step: keyword; (session-handle sampler constraint) -> a token id, or #f when
    the constraint has closed the reply.
  generate: keyword; (session-handle sampler constraint max-tokens on-token) ->
    the number of tokens produced; on-token returns #t to continue and #f to
    stop (default #f).
  stats: keyword; (handle) -> an alist (default: one returning the empty list).
  sampling?: keyword; whether step honours the sampler's parameters (default
    #f).
Returns: a backend.
Example:
  (use-modules (sonicboom serve backend))
  (backend-name
   (make-backend #:name \"demo\"
                 #:load-model (lambda (config) 'handle)
                 #:model-info (lambda (h) '((id . \"m\") (family . gemma)
                                            (context-length . 8) (eos-tokens . ())))
                 #:tokenize (lambda (model text) '())
                 #:detokenize (lambda (model tokens) \"\")
                 #:render-chat (lambda (model messages tools choice) '())
                 #:create-session (lambda (model) 'session)
                 #:free-session (lambda (session) #f)
                 #:prefill (lambda (session tokens constraint) 0)
                 #:step (lambda (session sampler constraint) 0)))
  => \"demo\"
PyTorch: none"
  (%make-backend name load-model unload-model model-info tokenize detokenize
                 render-chat tokenize-chat
                 create-session free-session fork-session prefill step
                 generate stats sampling?))

;;; ---------------------------------------------------------------------------
;;; Models and sessions

;; A loaded model: the backend that loaded it, the backend's handle and its info alist.
;;
;; The info alist has at least id (string), family (symbol: gemma or qwen),
;; context-length (integer) and eos-tokens (list of token ids that end a reply).
;; model-id, model-family, model-context-length and model-eos-tokens read those
;; entries.
;;
;; Example:
;;   (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
;;   (model-id (load-model (make-mock-backend) '((family . gemma))))
;;   => "mock-gemma"
;; PyTorch: none
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

(define (model-id model)
  "The id string of MODEL.

Args:
  model: a model.
Returns: a string.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (model-id (load-model (make-mock-backend) '()))
  => \"mock-gemma\"
PyTorch: none"
  (info-ref model 'id))
(define (model-family model)
  "The chat-template family of MODEL, as a symbol.

Args:
  model: a model.
Returns: gemma or qwen.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (model-family (load-model (make-mock-backend) '()))
  => gemma
PyTorch: none"
  (info-ref model 'family))

;; The architecture the loader reported, e.g. "qwen2". A backend whose models
;; have none may omit arch or provide #f.
(define (model-arch model)
  "The architecture name of MODEL, or #f when the backend has none.

Args:
  model: a model.
Returns: a string such as \"qwen2\", or #f.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (model-arch (load-model (make-mock-backend) '()))
  => \"mock\"
PyTorch: none"
  (match (assq 'arch (model-info model)) ((_ . v) v) (#f #f)))
(define (model-context-length model)
  "The maximum number of tokens MODEL holds.

Args:
  model: a model.
Returns: a positive integer.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (model-context-length
   (load-model (make-mock-backend) '((context-length . 4096))))
  => 4096
PyTorch: none"
  (info-ref model 'context-length))
(define (model-eos-tokens model)
  "The token ids that end a reply from MODEL.

Args:
  model: a model.
Returns: a list of token ids.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (model-eos-tokens (load-model (make-mock-backend) '()))
  => (3 1)
Note: the mock backend derives them from its template's end-of-turn strings.
PyTorch: none"
  (info-ref model 'eos-tokens))

;; Whether this model's backend can sample (see make-backend's sampling?).
(define (model-sampling? model)
  "True when MODEL's backend honours the sampler's parameters.

Args:
  model: a model.
Returns: a boolean.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (model-sampling? (load-model (make-mock-backend) '()))
  => #t
PyTorch: none"
  (backend-sampling? (model-backend model)))

(define (load-model backend config)
  "Load a model with BACKEND from a config alist.

The config keys are backend specific; the mock backend reads family, id,
context-length, step-delay and prefill-delay.

Args:
  backend: a backend.
  config: an alist.
Returns: a model.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (model-id (load-model (make-mock-backend) '((id . \"demo-model\"))))
  => \"demo-model\"
PyTorch: none"
  (let ((handle ((%backend-load-model backend) config)))
    (%make-model backend handle ((%backend-model-info backend) handle))))

(define (unload-model model)
  "Release MODEL's handle so its device memory returns to the driver.

A backend without an unload procedure does nothing; the handle then goes away
with its record. The model must have no live session: free them first.

Args:
  model: a model.
Returns: unspecified.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (begin (unload-model (load-model (make-mock-backend) '())) #t)
  => #t
PyTorch: none"
  (when (%backend-unload-model (model-backend model))
    ((%backend-unload-model (model-backend model)) (model-handle model))))

(define (tokenize model text)
  "Tokenize TEXT with MODEL's tokenizer.

Special markers are matched as whole tokens.

Args:
  model: a model.
  text: a string.
Returns: a list of token ids.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (tokenize (load-model (make-mock-backend) '()) \"hi\")
  => (360 361)
PyTorch: none"
  ((%backend-tokenize (model-backend model)) (model-handle model) text))

(define (detokenize model tokens)
  "The text of TOKENS under MODEL's tokenizer.

Args:
  model: a model.
  tokens: a list of token ids.
Returns: a string.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (detokenize (load-model (make-mock-backend) '()) '(360 361))
  => \"hi\"
PyTorch: none"
  ((%backend-detokenize (model-backend model)) (model-handle model) tokens))

;; messages: list of message alists in the OpenAI shape, already decoded from
;; JSON (role, content, tool_calls, tool_call_id, name). tools: list of tool
;; alists or '(). tool-choice: "none", "auto", "required" or a function name.
;; Returns the prompt as a segment list: (kind . text) with kind 'special for
;; the template's own turn markers and 'text for everything that came from a
;; message (see (sonicboom serve chat-template)).
(define* (render-chat-segments model messages #:key (tools '()) (tool-choice "auto"))
  "The prompt for MESSAGES as a segment list.

Each segment is (kind . text): special for the template's own turn markers and
text for everything that came from a message. Messages are alists with string
keys in the OpenAI shape.

Args:
  model: a model.
  messages: the list of messages.
  tools: keyword; the tools offered to the model (default '()).
  tool-choice: keyword; \"none\", \"auto\", \"required\" or a function name
    (default \"auto\").
Returns: a list of segments.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (length (render-chat-segments
           (load-model (make-mock-backend) '())
           (list (list (cons \"role\" \"user\") (cons \"content\" \"hi\")))))
  => 8
PyTorch: none"
  ((%backend-render-chat (model-backend model))
   (model-handle model) messages tools tool-choice))

;; The same rendered prompt as one string (for logs and tests).
(define* (render-chat model messages #:key (tools '()) (tool-choice "auto"))
  "The prompt for MESSAGES as one string.

Args:
  model: a model.
  messages: the list of messages.
  tools: keyword; the tools offered to the model (default '()).
  tool-choice: keyword; \"none\", \"auto\", \"required\" or a function name
    (default \"auto\").
Returns: a string.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (render-chat (load-model (make-mock-backend) '())
               (list (list (cons \"role\" \"user\") (cons \"content\" \"hi\"))))
  => \"<bos><|turn>user\\nhi<turn|>\\n<|turn>model\\n\"
PyTorch: none"
  (segments->string
   (render-chat-segments model messages #:tools tools #:tool-choice tool-choice)))

;; A rendered prompt's tokens. A backend that owns a real tokenizer tokenizes
;; the rendered segments itself — the 'special ones with special parsing, the
;; rest as text — so a turn marker is one control token while a marker inside a
;; user's message stays text. Without one, the segments are concatenated and
;; tokenized as plain text, which is what every prompt used to be.
(define (tokenize-chat model segments)
  "The token ids of a rendered prompt.

The template's special segments are tokenized with special parsing, so each
becomes the vocabulary's control token; the text segments are tokenized as
plain text.

Args:
  model: a model.
  segments: a list of prompt segments.
Returns: a list of token ids.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (tokenize-chat (load-model (make-mock-backend) '())
                 (list (cons 'special \"<bos>\")))
  => (0)
PyTorch: none"
  (let ((proc (%backend-tokenize-chat (model-backend model))))
    (if proc
        (proc (model-handle model) segments)
        (tokenize model (segments->string segments)))))

;; A session is one model's KV cache: it holds a token sequence.
;;
;; create-session makes an empty one and free-session releases it. prefill makes
;; it hold a given sequence, keeping the longest common prefix of what it held;
;; step samples the next token and appends it. A session belongs to the model in
;; session-model and carries the backend's handle in session-handle.
;;
;; Example:
;;   (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
;;   (session? (create-session (load-model (make-mock-backend) '())))
;;   => #t
;; PyTorch: none
(define-record-type <session>
  (%make-session model handle)
  session?
  (model session-model)
  (handle session-handle))

(define (create-session model)
  "Create an empty session for MODEL.

Args:
  model: a model.
Returns: a session.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (session? (create-session (load-model (make-mock-backend) '())))
  => #t
PyTorch: none"
  (%make-session model
                 ((%backend-create-session (model-backend model))
                  (model-handle model))))

(define (free-session session)
  "Release SESSION and its held tokens.

Args:
  session: a session.
Returns: unspecified.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (begin (free-session
          (create-session (load-model (make-mock-backend) '())))
         #t)
  => #t
PyTorch: none"
  ((%backend-free-session (model-backend (session-model session)))
   (session-handle session)))

(define (can-fork? model)
  "True when MODEL's backend can copy a session's tokens into a new session.

Args:
  model: a model.
Returns: a boolean.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (can-fork? (load-model (make-mock-backend) '()))
  => #t
PyTorch: none"
  (and (%backend-fork-session (model-backend model)) #t))

;; A new session holding the first n tokens of session, without recomputing
;; them (a KV-cache copy on the device).
(define (fork-session session n)
  "A new session holding the first N tokens of SESSION.

The tokens are copied, not recomputed.

Args:
  session: a session.
  n: how many tokens to keep.
Returns: a session.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (let ((model (load-model (make-mock-backend) '())))
    (session? (fork-session (create-session model) 0)))
  => #t
PyTorch: none"
  (let ((model (session-model session)))
    (%make-session model
                   ((%backend-fork-session (model-backend model))
                    (session-handle session) n))))

(define* (prefill session tokens #:optional (constraint #f))
  "Make SESSION hold TOKENS, computing only what it does not already hold.

The longest common prefix of the session's current tokens and TOKENS is kept;
the rest is computed. With a constraint the grammar is installed before the
first token after the prompt is sampled.

Args:
  session: a session.
  tokens: a list of token ids.
  constraint: optional; a constrained-decoding constraint or #f.
Returns: the number of tokens kept.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (let* ((model (load-model (make-mock-backend) '()))
         (session (create-session model)))
    (prefill session '(104 105))
    (prefill session '(104 105)))
  => 2
PyTorch: none"
  ((%backend-prefill (model-backend (session-model session)))
   (session-handle session) tokens constraint))

(define* (step session sampler #:optional (constraint #f))
  "Sample the next token after the sequence SESSION holds, then append it.

The token is sampled from the sampler's parameters; a constraint may restrict
the candidates. The next step continues from the token.

Args:
  session: a session.
  sampler: a sampler.
  constraint: optional; a constrained-decoding constraint or #f.
Returns: a token id, or #f when the constraint has closed the reply.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (let* ((model (load-model (make-mock-backend) '()))
         (session (create-session model)))
    (prefill session (tokenize model \"hi\"))
    (detokenize model
                (list (step session (make-sampler #:temperature 0)))))
  => \"Y\"
Note: the mock backend plans the reply \"You said: \" for a prompt with no user
turn, so its first token is the \"Y\" of that text.
PyTorch: none"
  ((%backend-step (model-backend (session-model session)))
   (session-handle session) sampler constraint))

;; Whether this model's backend can produce a reply in one call. A backend
;; without generate is driven token by token through step.
(define (can-generate? model)
  "True when MODEL's backend can produce a reply in one call.

Args:
  model: a model.
Returns: a boolean.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (can-generate? (load-model (make-mock-backend) '()))
  => #f
PyTorch: none"
  (and (%backend-generate (model-backend model)) #t))

(define (generate session sampler constraint max-tokens on-token)
  "Produce a reply of at most MAX-TOKENS tokens in one backend call.

Each produced token is passed to ON-TOKEN, which returns #t to continue and #f
to stop the decode.

Args:
  session: a session.
  sampler: a sampler.
  constraint: a constrained-decoding constraint or #f.
  max-tokens: the maximum number of tokens to produce.
  on-token: a procedure of one token id.
Returns: the number of tokens produced.
Example:
  (use-modules (sonicboom serve backend))
  (let ((backend
         (make-backend #:name \"demo\"
                       #:load-model (lambda (config) 'handle)
                       #:model-info (lambda (h) '((id . \"m\") (family . gemma)
                                                  (context-length . 8)
                                                  (eos-tokens . ())))
                       #:tokenize (lambda (model text) '())
                       #:detokenize (lambda (model tokens) \"\")
                       #:render-chat (lambda (model messages tools choice) '())
                       #:create-session (lambda (model) 'session)
                       #:free-session (lambda (session) #f)
                       #:prefill (lambda (session tokens constraint) 0)
                       #:step (lambda (session sampler constraint) 0)
                       #:generate (lambda (session sampler constraint max-tokens
                                          on-token)
                                    (on-token 1) (on-token 2) 2))))
    (generate (create-session (load-model backend '())) (make-sampler) #f 4
              (lambda (token) #t)))
  => 2
PyTorch: none"
  ((%backend-generate (model-backend (session-model session)))
   (session-handle session) sampler constraint max-tokens on-token))

(define (backend-stats model)
  "The backend's counters for MODEL as an alist.

Args:
  model: a model.
Returns: an alist.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend))
  (backend-stats (load-model (make-mock-backend) '()))
  => ()
PyTorch: none"
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

;; A constrained-decoding grammar.
;;
;; kind is one of json-schema, tool-call or gbnf; text is the schema or the
;; grammar source. filter is a host-side token filter, or #f for a grammar: turning
;; a grammar into a token mask needs the model's vocabulary. Build one with
;; make-json-schema-constraint, make-tool-call-constraint or make-gbnf-constraint.
;;
;; Example:
;;   (use-modules (sonicboom serve backend))
;;   (constraint-kind (make-gbnf-constraint "root ::= x"))
;;   => gbnf
;; PyTorch: none
(define-record-type <constraint>
  (%make-constraint kind text filter)
  constraint?
  (kind constraint-kind)
  (text constraint-text)
  (filter constraint-filter))

(define (make-json-schema-constraint schema-json)
  "A constraint that the reply is one JSON value of SCHEMA-JSON.

Args:
  schema-json: a JSON Schema, as JSON text.
Returns: a constraint.
Example:
  (use-modules (sonicboom serve backend))
  (constraint-kind (make-json-schema-constraint \"{}\"))
  => json-schema
PyTorch: none"
  (%make-constraint 'json-schema schema-json #f))

(define (make-tool-call-constraint call-schema-json)
  "A constraint that the reply is one tool-call envelope.

The envelope's object must be a value of CALL-SCHEMA-JSON, a JSON Schema of
{name, arguments}.

Args:
  call-schema-json: the call object's schema, as JSON text.
Returns: a constraint.
Example:
  (use-modules (sonicboom serve backend))
  (constraint-kind (make-tool-call-constraint \"{}\"))
  => tool-call
PyTorch: none"
  (%make-constraint 'tool-call call-schema-json #f))

(define (make-gbnf-constraint gbnf)
  "A constraint that the reply is text the GBNF grammar GBNF accepts.

Args:
  gbnf: the grammar, as GBNF text.
Returns: a constraint.
Example:
  (use-modules (sonicboom serve backend))
  (constraint-kind (make-gbnf-constraint \"root ::= x\"))
  => gbnf
PyTorch: none"
  (%make-constraint 'gbnf gbnf #f))

;;; ---------------------------------------------------------------------------
;;; Sampler: the parameters of one request, plus its random state

;; One request's sampling parameters and its random state.
;;
;; The fields are temperature, top-p, top-k, min-p, seed, repeat-penalty,
;; greedy?, random-state, presence-penalty and frequency-penalty, with an
;; accessor named after each. make-sampler creates one; sample-from-logits
;; draws with it.
;;
;; Example:
;;   (use-modules (sonicboom serve backend))
;;   (sampler-greedy? (make-sampler #:temperature 0))
;;   => #t
;; PyTorch: none
(define-record-type <sampler>
  (%make-sampler temperature top-p top-k min-p seed repeat-penalty greedy?
                 random-state presence-penalty frequency-penalty)
  sampler?
  (temperature sampler-temperature)
  (top-p sampler-top-p)
  (top-k sampler-top-k)
  (min-p sampler-min-p)
  (seed sampler-seed)
  (repeat-penalty sampler-repeat-penalty)
  (greedy? sampler-greedy?)
  (random-state sampler-random-state)
  (presence-penalty sampler-presence-penalty)
  (frequency-penalty sampler-frequency-penalty))

;; temperature 0 means greedy, like the OpenAI API. top-k 0 means no top-k.
;; Without a seed the random state is seeded from the clock.
(define* (make-sampler #:key (temperature 1.0) (top-p 1.0) (top-k 0)
                       (min-p 0.0) (seed #f) (repeat-penalty 1.0) (greedy? #f)
                       (presence-penalty 0.0) (frequency-penalty 0.0))
  "Create a sampler.

A temperature of 0 means greedy, as in the OpenAI API; top-k 0 means no top-k
limit and min-p 0 no min-p cutoff. Without a seed the random state is seeded
from the clock.

Args:
  temperature: keyword; the sampling temperature (default 1.0).
  top-p: keyword; the nucleus mass to keep (default 1.0).
  top-k: keyword; how many candidates to keep, 0 for all (default 0).
  min-p: keyword; minimum relative probability, 0 to keep everything (default
    0.0).
  seed: keyword; the random seed, or #f for one from the clock (default #f).
  repeat-penalty: keyword; the repetition penalty, 1 for off (default 1.0).
    The reference sampler has no token history to penalize; the device
    backends apply it.
  greedy?: keyword; force greedy decoding (default #f).
  presence-penalty: keyword; the presence penalty (default 0.0).
  frequency-penalty: keyword; the frequency penalty (default 0.0).
Returns: a sampler.
Example:
  (use-modules (sonicboom serve backend))
  (sampler-greedy? (make-sampler #:temperature 0))
  => #t
PyTorch: none"
  (%make-sampler temperature top-p top-k min-p seed repeat-penalty
                 (or greedy? (<= temperature 0))
                 (seed->random-state
                  (or seed (+ (* 1000000 (car (gettimeofday)))
                              (cdr (gettimeofday)))))
                 presence-penalty frequency-penalty))

;; The reference sampler for backends that produce logits on the host.
;; logits: list of (token . logit). constraint: #f or a token filter.
;; Order: constraint, then greedy, else temperature, top-k, min-p, top-p, draw.
(define (sample-from-logits sampler logits constraint)
  "Draw one token from LOGITS with SAMPLER.

LOGITS is a list of (token . logit). The constraint is applied first, then a
greedy sampler takes the highest logit; otherwise the candidates are ordered by
logit, limited by top-k, reweighted by temperature, cut below min-p (a
candidate under min-p of the top probability is dropped) and cut to the top-p
mass before one is drawn from the random state.

Args:
  sampler: a sampler.
  logits: a list of (token . logit).
  constraint: #f, a constraint, or a procedure (token-ids) -> the allowed ids.
Returns: a token id.
Example:
  (use-modules (sonicboom serve backend))
  (sample-from-logits (make-sampler #:temperature 0)
                      '((1 . 0.5) (2 . 2.0)) #f)
  => 2
PyTorch: none"
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
               ;; min-p drops every candidate whose probability is under
               ;; min-p of the top one. With softmax the top probability is
               ;; 1 after reweighting, so the logit difference decides.
               (kept (let ((m (sampler-min-p sampler)))
                       (if (> m 0)
                           (let ((cutoff (* t (log m))))
                             (filter (lambda (p) (>= (- (cdr p) top) cutoff))
                                     kept))
                           kept)))
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
  "The length of the longest common prefix of the token vectors A and B.

Args:
  a: a token vector.
  b: a token vector.
Returns: a non-negative integer.
Example:
  (use-modules (sonicboom serve backend))
  (common-prefix-length '#(1 2 3) '#(1 2 4))
  => 2
PyTorch: none"
  (let ((n (min (vector-length a) (vector-length b))))
    (let lp ((i 0))
      (if (and (< i n) (eqv? (vector-ref a i) (vector-ref b i)))
          (lp (1+ i))
          i))))
