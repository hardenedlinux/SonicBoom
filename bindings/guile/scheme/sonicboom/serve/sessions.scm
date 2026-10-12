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

(define-record-type <entry>
  (make-entry session tokens last-used busy? match)
  entry?
  (session pool-entry-session)
  (tokens entry-tokens set-entry-tokens!)      ; vector, the mirror
  (last-used entry-last-used set-entry-last-used!)
  (busy? entry-busy? set-entry-busy!)
  (match pool-entry-match set-entry-match!))   ; prefix length found at acquire

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
  (with-mutex (pool-lock pool)
    (for-each (lambda (e) (evict! pool e)) (remove entry-busy? (pool-entries pool)))))

(define (pool-stats pool)
  (with-mutex (pool-lock pool)
    `((sessions . ,(length (pool-entries pool)))
      (busy . ,(count entry-busy? (pool-entries pool)))
      (held-tokens . ,(held-tokens pool))
      (max-sessions . ,(pool-max-sessions pool))
      (token-budget . ,(pool-token-budget pool))
      ,@(map (lambda (k) (cons k (hashq-ref (pool-stats-table pool) k 0)))
             '(lookups hits forks created takeovers evictions)))))
