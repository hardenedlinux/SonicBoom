;;; SonicBoom — inference service: sessions and the prefix cache
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A pool of backend sessions for one model. Each session is a KV cache and
;;; the pool keeps a mirror of the tokens it holds, so a request can be routed
;;; to the idle session sharing the longest token prefix with its prompt; the
;;; backend's prefill then reuses that prefix.
;;;
;;;   pool-acquire!  prompt tokens -> an entry (its session is busy until
;;;                  released). The idle entry with the longest common prefix
;;;                  (least recently used on ties) is the best match:
;;;                    - the prompt extends all it holds: that session is used;
;;;                    - a partial match, and room for one more session: the
;;;                      prefix is copied into a new session (fork-session), so
;;;                      the matched session keeps its own continuation;
;;;                    - no match and room: a new session;
;;;                    - no room: the best match is taken over (overwritten
;;;                      past the common prefix).
;;;   pool-release!  the tokens the session holds now (prompt + generated).
;;;                  Then the token budget is enforced: least recently used
;;;                  idle sessions are freed until the held tokens fit.
;;;
;;; One session holds one sequence. Without fork-session in the backend, a
;;; partial match takes the session over like the no-room case.

(define-module (sonicboom serve sessions)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 threads)
  #:use-module (sonicboom serve backend)
  #:export (make-session-pool session-pool? pool-model
            pool-acquire! pool-release! pool-entry-session pool-entry-match
            pool-stats pool-clear!))

;; One pooled session and the tokens it holds.
;;
;; pool-entry-session is the backend session; pool-entry-match is the length of
;; the common prefix found when the entry was acquired. The rest of the entry
;; (the mirror, the last-use clock and the busy flag) is internal to the pool.
;;
;; PyTorch: none
(define-record-type <entry>
  (make-entry session tokens last-used busy? match)
  entry?
  (session pool-entry-session)
  (tokens entry-tokens set-entry-tokens!)      ; vector, the mirror
  (last-used entry-last-used set-entry-last-used!)
  (busy? entry-busy? set-entry-busy!)
  (match pool-entry-match set-entry-match!))   ; prefix length found at acquire

;; A pool of backend sessions for one model.
;;
;; session-pool? recognizes it and pool-model returns the model. The pool keeps a
;; mirror of the tokens each session holds, so a request can go to the idle
;; session sharing the longest token prefix with its prompt. make-session-pool
;; creates one; pool-acquire! and pool-release! lend and return entries.
;;
;; Example:
;;   (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
;;                (sonicboom serve sessions))
;;   (session-pool? (make-session-pool (load-model (make-mock-backend) '())))
;;   => #t
;; PyTorch: none
(define-record-type <pool>
  (%make-pool model max-sessions token-budget entries clock lock stats)
  session-pool?
  (model pool-model)
  (max-sessions pool-max-sessions)
  (token-budget pool-token-budget)
  (entries pool-entries set-pool-entries!)
  (clock pool-clock set-pool-clock!)
  (lock pool-lock)
  (stats pool-stats-table))

(define* (make-session-pool model #:key (max-sessions 4) (token-budget 32768))
  "Create a session pool for MODEL.

Args:
  model: a model.
  max-sessions: keyword; how many sessions the pool keeps (default 4).
  token-budget: keyword; how many tokens the sessions may hold together
    (default 32768).
Returns: a session pool.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve sessions))
  (session-pool? (make-session-pool (load-model (make-mock-backend) '())))
  => #t
PyTorch: none"
  (%make-pool model max-sessions token-budget '() 0 (make-mutex) (make-hash-table)))

(define (bump! pool key n)
  (hashq-set! (pool-stats-table pool) key
              (+ n (hashq-ref (pool-stats-table pool) key 0))))

(define (tick! pool)
  (set-pool-clock! pool (1+ (pool-clock pool)))
  (pool-clock pool))

(define (lru entries)
  (and (pair? entries)
       (reduce (lambda (e best)
                 (if (< (entry-last-used e) (entry-last-used best)) e best))
               #f entries)))

(define (evict! pool e)
  (free-session (pool-entry-session e))
  (set-pool-entries! pool (delete e (pool-entries pool) eq?))
  (bump! pool 'evictions 1))

(define (new-entry! pool session)
  (let ((e (make-entry session (vector) 0 #f 0)))
    (set-pool-entries! pool (cons e (pool-entries pool)))
    e))

(define (pool-acquire! pool prompt-tokens)
  "Lend the entry whose session shares the longest prefix with PROMPT-TOKENS.

A prompt that extends everything a session holds reuses that session; a partial
match copies the prefix into a new session when the pool has room and the
backend can fork, so the matched session keeps its own continuation; otherwise
a new session is created, or the best match is taken over when the pool is
full. Ties go to the least recently used session. The entry is busy until
pool-release!.

Args:
  pool: a session pool.
  prompt-tokens: the prompt's token ids, a list or a vector.
Returns: a pool entry.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve sessions))
  (let* ((model (load-model (make-mock-backend) '()))
         (pool (make-session-pool model)))
    (pool-entry-match (pool-acquire! pool '(104 105))))
  => 0
PyTorch: none"
  (let ((prompt (if (vector? prompt-tokens) prompt-tokens (list->vector prompt-tokens)))
        (model (pool-model pool)))
    (with-mutex (pool-lock pool)
      (bump! pool 'lookups 1)
      (let* ((idle (remove entry-busy? (pool-entries pool)))
             (room? (< (length (pool-entries pool)) (pool-max-sessions pool)))
             ;; Best match: the longest prefix, then the least recently used.
             (best (reduce (lambda (a b)
                             (if (or (> (cdr a) (cdr b))
                                     (and (= (cdr a) (cdr b))
                                          (< (entry-last-used (car a))
                                             (entry-last-used (car b)))))
                                 a b))
                           #f
                           (map (lambda (e)
                                  (cons e (common-prefix-length (entry-tokens e) prompt)))
                                idle)))
             (e (and best (car best)))
             (n (if best (cdr best) 0))
             (entry
              (cond
               ;; The prompt extends everything the session holds: use it.
               ((and e (> n 0) (= n (vector-length (entry-tokens e))))
                (bump! pool 'hits 1)
                (set-entry-match! e n)
                e)
               ;; A partial match with room left: copy the prefix, and keep
               ;; the matched session for its own continuation.
               ((and e (> n 0) room? (can-fork? model))
                (bump! pool 'hits 1)
                (bump! pool 'forks 1)
                (let ((f (new-entry! pool (fork-session (pool-entry-session e) n))))
                  (set-entry-tokens! f (vector-copy (entry-tokens e) 0 n))
                  (set-entry-match! f n)
                  f))
               (room?
                (bump! pool 'created 1)
                (new-entry! pool (create-session model)))
               (e
                ;; Overwrite the best match (LRU on ties) past its prefix.
                (when (> n 0) (bump! pool 'hits 1))
                (bump! pool 'takeovers 1)
                (set-entry-match! e n)
                e)
               (else
                ;; Every session is busy: one more, over max-sessions, freed
                ;; again at release.
                (bump! pool 'created 1)
                (new-entry! pool (create-session model))))))
        (set-entry-busy! entry #t)
        (set-entry-last-used! entry (tick! pool))
        entry))))

(define (held-tokens pool)
  (fold + 0 (map (lambda (e) (vector-length (entry-tokens e))) (pool-entries pool))))

(define (pool-release! pool entry tokens)
  "Return ENTRY to the pool with the tokens its session now holds.

The pool then enforces its limits: idle sessions are freed, least recently used
first, until the session count and the token budget are satisfied.

Args:
  pool: a session pool.
  entry: a pool entry.
  tokens: the tokens the session holds (prompt plus generated).
Returns: unspecified.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve sessions))
  (let* ((model (load-model (make-mock-backend) '()))
         (pool (make-session-pool model)))
    (pool-release! pool (pool-acquire! pool '(104 105)) '(104 105))
    (assq-ref (pool-stats pool) 'held-tokens))
  => 2
PyTorch: none"
  (with-mutex (pool-lock pool)
    (set-entry-tokens! entry (if (vector? tokens) tokens (list->vector tokens)))
    (set-entry-busy! entry #f)
    (set-entry-last-used! entry (tick! pool))
    ;; Over max-sessions (all were busy at acquire): drop the oldest idle.
    (let lp ()
      (when (> (length (pool-entries pool)) (pool-max-sessions pool))
        (let ((victim (lru (remove entry-busy? (pool-entries pool)))))
          (when victim (evict! pool victim) (lp)))))
    ;; Token budget.
    (let lp ()
      (when (> (held-tokens pool) (pool-token-budget pool))
        (let ((victim (lru (remove entry-busy? (pool-entries pool)))))
          (when victim (evict! pool victim) (lp)))))))

(define (pool-clear! pool)
  "Free every idle session of POOL.

A session that is busy at the time is left alone.

Args:
  pool: a session pool.
Returns: unspecified.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve sessions))
  (let* ((model (load-model (make-mock-backend) '()))
         (pool (make-session-pool model)))
    (pool-release! pool (pool-acquire! pool '(1)) '(1))
    (pool-clear! pool)
    (assq-ref (pool-stats pool) 'sessions))
  => 0
PyTorch: none"
  (with-mutex (pool-lock pool)
    (for-each (lambda (e) (evict! pool e)) (remove entry-busy? (pool-entries pool)))))

(define (pool-stats pool)
  "The counters of POOL as an alist.

The alist holds sessions, busy, held-tokens, max-sessions, token-budget and the
lookup, hit, fork, creation, takeover and eviction counters.

Args:
  pool: a session pool.
Returns: an alist.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve sessions))
  (assq-ref (pool-stats
             (make-session-pool (load-model (make-mock-backend) '())))
            'sessions)
  => 0
PyTorch: none"
  (with-mutex (pool-lock pool)
    `((sessions . ,(length (pool-entries pool)))
      (busy . ,(count entry-busy? (pool-entries pool)))
      (held-tokens . ,(held-tokens pool))
      (max-sessions . ,(pool-max-sessions pool))
      (token-budget . ,(pool-token-budget pool))
      ,@(map (lambda (k) (cons k (hashq-ref (pool-stats-table pool) k 0)))
             '(lookups hits forks created takeovers evictions)))))
