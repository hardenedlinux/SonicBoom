;;; SonicBoom — Guile byte-level BPE tokenizer tests
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/tokenizer/test-bpe.scm
;;;
;;; The CS336 A1 gate: train on 512 TinyStories-shaped lines (generated
;;; deterministically here — no dataset ships with the repository), then check
;;; the vocabulary size, exact string round trips, token-id round trips on
;;; random samples, the built-in and user special tokens, and save/load.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 format)
             (sonicboom tokenizer bpe))

;;; --- deterministic corpus ---------------------------------------------------

(define story-words
  '("the" "cat" "sat" "on" "mat" "a" "dog" "ran" "to" "ball"
    "girl" "boy" "saw" "small" "big" "red" "blue" "house" "tree"
    "and" "then" "she" "he" "was" "happy" "sad" "played" "with"
    "friend" "little" "puppy" "kitten" "garden" "day" "night"
    "sun" "moon" "star" "sky" "looked" "found" "went" "home"))

(define (make-corpus lines)
  (define seed 20260120)
  (define (rand n)
    (set! seed (modulo (+ (* seed 1103515245) 12345) 2147483648))
    (modulo (quotient seed 65536) n))
  (let loop ((k 0) (out '()))
    (if (>= k lines)
        (string-concatenate (reverse out))
        (let* ((len (+ 6 (rand 9)))
               (ws (let wl ((j 0) (acc '()))
                     (if (>= j len)
                         (reverse acc)
                         (wl (+ j 1)
                             (cons (list-ref story-words (rand (length story-words)))
                                   acc)))))
               (sentence (string-join (cons (string-capitalize (car ws)) (cdr ws))
                                      " ")))
          (loop (+ k 1) (cons (string-append sentence ".\n") out))))))

(define corpus (make-corpus 512))
(define corpus-lines
  (filter (lambda (l) (> (string-length l) 0))
          (string-split corpus #\newline)))

(define tmp-base (format #f "/tmp/sb-bpe-test-~a" (getpid)))
(define corpus-path (string-append tmp-base "-corpus.txt"))
(define saved-base (string-append tmp-base "-tok"))

;; Train from a real text port, the same path a dataset loader takes.
(call-with-output-file corpus-path
  (lambda (port) (display corpus port)))

(define t0 (get-internal-real-time))
(define tok
  (call-with-input-file corpus-path (lambda (port) (bpe-train port 400))))
(define train-seconds
  (/ (- (get-internal-real-time) t0)
     (exact->inexact internal-time-units-per-second)))

(define (random-lines n)
  (define seed 987654321)
  (define (rand k)
    (set! seed (modulo (+ (* seed 1103515245) 12345) 2147483648))
    (modulo (quotient seed 65536) k))
  (map (lambda (_) (list-ref corpus-lines (rand (length corpus-lines))))
       (iota n)))

;;; --- tests ------------------------------------------------------------------

(test-begin "sonicboom-bpe")

(test-assert "vocabulary reaches the requested size"
  (= 400 (bpe-vocab-size tok)))

(test-assert "merges compress the corpus below its byte length"
  (let* ((line (car corpus-lines))
         (ids (bpe-encode tok line)))
    (< (length ids) (string-length line))))

(test-assert "exact string round trip over the whole corpus"
  (string=? corpus (bpe-decode tok (bpe-encode tok corpus))))

(test-assert "exact line round trip on every corpus line"
  (every (lambda (line) (string=? line (bpe-decode tok (bpe-encode tok line))))
         corpus-lines))

(test-assert "UTF-8 text round-trips byte for byte"
  (every (lambda (s) (string=? s (bpe-decode tok (bpe-encode tok s))))
         '("héllo wörld" "猫が座った。" "🐈‍⬛ and 🌙" "a\tb c\r\n")))

(test-assert "encode(decode(ids)) = ids on random samples"
  (every (lambda (s)
           (let ((ids (bpe-encode tok s)))
             (and (equal? ids (bpe-encode tok (bpe-decode tok ids)))
                  (string=? s (bpe-decode tok ids)))))
         (random-lines 25)))

;; Built-in special tokens: <unk> 256, <bos> 257, <eos> 258.
(test-assert "built-in special tokens are present"
  (and (= 256 (bpe-special-id tok "<unk>"))
       (= 257 (bpe-special-id tok "<bos>"))
       (= 258 (bpe-special-id tok "<eos>"))))

(test-equal "special ids decode to their text"
  (list "<unk>" "<bos>" "<eos>")
  (map (lambda (id) (bpe-decode tok (list id))) '(256 257 258)))

(test-equal "an out-of-range id decodes as <unk>"
  "<unk>"
  (bpe-decode tok (list 100000)))

(test-assert "special text inside a string encodes to its id and back"
  (let ((s "hello <eos> world"))
    (let ((ids (bpe-encode tok s)))
      (and (memv 258 ids)
           (string=? s (bpe-decode tok ids))
           (equal? ids (bpe-encode tok (bpe-decode tok ids)))))))

(define story-id (bpe-add-special! tok "<|story|>"))

(test-equal "an added special is appended after the merges"
  400 story-id)

(test-equal "adding a special grows the vocabulary"
  401 (bpe-vocab-size tok))

(test-assert "an added special is matched verbatim"
  (let* ((s "Once <|story|> upon a time")
         (ids (bpe-encode tok s)))
    (and (memv story-id ids)
         (string=? s (bpe-decode tok ids)))))

(test-equal "special ids include the added token"
  '(400 256 257 258)
  (bpe-special-ids tok))

;;; --- save / load ------------------------------------------------------------

(bpe-save tok saved-base)

(test-assert "save writes the vocabulary and the merges"
  (and (file-exists? (string-append saved-base ".vocab"))
       (file-exists? (string-append saved-base ".merges"))))

(define loaded (bpe-load saved-base))

(test-equal "loaded vocabulary size" (bpe-vocab-size tok) (bpe-vocab-size loaded))

(test-equal "loaded special ids" (bpe-special-ids tok) (bpe-special-ids loaded))

(test-assert "loaded tokenizer encodes every line identically"
  (every (lambda (line)
           (equal? (bpe-encode tok line) (bpe-encode loaded line)))
         corpus-lines))

(test-assert "loaded tokenizer keeps the added special"
  (let* ((s "Once <|story|> upon")
         (ids (bpe-encode loaded s)))
    (and (memv story-id ids) (string=? s (bpe-decode loaded ids)))))

;;; --- encode cost ------------------------------------------------------------

(define t1 (get-internal-real-time))
(define pass-tokens
  (fold (lambda (line n) (+ n (length (bpe-encode loaded line))))
        0 corpus-lines))
(define pass-seconds
  (/ (- (get-internal-real-time) t1)
     (exact->inexact internal-time-units-per-second)))

(format #t "bpe: trained in ~,2f s; encoded ~a lines / ~a tokens in ~,3f s (~,0f tok/s)~%"
        train-seconds (length corpus-lines) pass-tokens pass-seconds
        (/ pass-tokens pass-seconds))

;;; --- cleanup + exit ---------------------------------------------------------

(for-each (lambda (p) (when (file-exists? p) (delete-file p)))
          (list corpus-path
                (string-append saved-base ".vocab")
                (string-append saved-base ".merges")))

(let ((failures (test-runner-fail-count (test-runner-current))))
  (test-end "sonicboom-bpe")
  (exit (if (zero? failures) 0 1)))
