;;; SonicBoom — Guile streaming loader tests
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/data/test-loader.scm
;;;
;;; The CS336 A1 gate: 100-line plain text and JSONL files, fixed-shape int32
;;; batches whose target is the input shifted by one, distinct batches on the
;;; first pass, and the epoch wrap at end of file.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 format)
             (sonicboom tokenizer bpe)
             (sonicboom data loader)
             (sonicboom trace tensor))

;;; --- fixtures ---------------------------------------------------------------

(define (make-lines n)
  (map (lambda (i)
         (format #f "line ~a says the little dog ran to the red house" i))
       (iota n)))

(define tmp-base (format #f "/tmp/sb-loader-test-~a" (getpid)))
(define text-path (string-append tmp-base ".txt"))
(define jsonl-path (string-append tmp-base ".jsonl"))
(define small-path (string-append tmp-base "-small.txt"))

(define lines (make-lines 100))
(define small-lines (make-lines 6))

(call-with-output-file text-path
  (lambda (port) (for-each (lambda (l) (display l port) (newline port)) lines)))
(call-with-output-file jsonl-path
  (lambda (port)
    (for-each (lambda (i l)
                (format port "{\"id\": ~a, \"text\": ~s}~%" i l))
              (iota (length lines)) lines)))
(call-with-output-file small-path
  (lambda (port)
    (for-each (lambda (l) (display l port) (newline port)) small-lines)))

;; A byte-level tokenizer (259 tokens, no merges): its ids are the UTF-8 bytes,
;; so the expected stream is easy to build independently of the loader.
(define tok (bpe-train (open-input-string (string-join lines "\n")) 259))
(define eos (bpe-special-id tok "<eos>"))

(define (stream-of ls)
  (append-map (lambda (l) (append (bpe-encode tok l) (list eos))) ls))

(define SEQ 8)
(define BATCH 4)

(define (expected-batch stream b seq-len batch-size)
  "Batch B as (input target), each a flat row-major list."
  (let* ((width (+ seq-len 1))
         (inp '()) (tgt '()))
    (do ((i 0 (+ i 1))) ((= i batch-size))
      (let* ((w (+ (* b batch-size) i))
             (win (take (drop stream (* w width)) width)))
        (set! inp (append inp (take win seq-len)))
        (set! tgt (append tgt (drop win 1)))))
    (list inp tgt)))

(define (batch->lists inp tgt)
  (values (ctensor->list inp) (ctensor->list tgt)))

;;; --- plain text -------------------------------------------------------------

(test-begin "sonicboom-loader")

(define text-stream (stream-of lines))

(define text-loader (text-file-loader text-path tok SEQ BATCH))

(test-equal "epoch starts at 0" 0 (loader-epoch text-loader))

(define first-batches
  (let loop ((b 0) (acc '()))
    (if (= b 5)
        (reverse acc)
        (call-with-values (lambda () (loader-next-batch! text-loader))
          (lambda (inp tgt)
            (test-assert (format #f "batch ~a is int32 [~a ~a]" b BATCH SEQ)
              (and (eq? 'int32 (ctensor-dtype inp))
                   (eq? 'int32 (ctensor-dtype tgt))
                   (equal? (ctensor-shape inp) (list BATCH SEQ))
                   (equal? (ctensor-shape tgt) (list BATCH SEQ))))
            (call-with-values (lambda () (batch->lists inp tgt))
              (lambda (a t)
                (loop (+ b 1) (cons (list a t) acc)))))))))

(test-equal "five batches, then still epoch 0" 0 (loader-epoch text-loader))

(test-assert "every batch matches the packed token stream"
  (every (lambda (got b) (equal? got (expected-batch text-stream b SEQ BATCH)))
         first-batches (iota 5)))

(test-assert "target is input shifted by one inside every row"
  (every (lambda (pair)
           (let ((inp (car pair)) (tgt (cadr pair)))
             (every (lambda (row)
                      (let ((base (* row SEQ)))
                        (every (lambda (j)
                                 (= (list-ref tgt (+ base j))
                                    (list-ref inp (+ base j 1))))
                               (iota (- SEQ 1)))))
                    (iota BATCH))))
         first-batches))

(test-equal "first-pass batches are all distinct"
  5
  (length (delete-duplicates (map car first-batches))))

;;; --- JSONL ------------------------------------------------------------------

(define jsonl-loader* (jsonl-loader jsonl-path "text" tok SEQ BATCH))
(define jsonl-batches
  (let loop ((b 0) (acc '()))
    (if (= b 5)
        (reverse acc)
        (call-with-values (lambda () (loader-next-batch! jsonl-loader*))
          (lambda (inp tgt)
            (call-with-values (lambda () (batch->lists inp tgt))
              (lambda (a t)
                (loop (+ b 1) (cons (list a t) acc)))))))))

(test-assert "JSONL batches match the same text as the plain file"
  (equal? first-batches jsonl-batches))

(test-equal "JSONL loader stays in epoch 0" 0 (loader-epoch jsonl-loader*))

;;; --- wrap-around at end of file ---------------------------------------------

(define small-loader (text-file-loader small-path tok 4 2))
(define small-stream (stream-of small-lines))
(define epoch-start-batch (expected-batch small-stream 0 4 2))

(define (read-until-wrap loader)
  "Read batches until one comes back in epoch 1; return that batch."
  (let loop ((n 0))
    (when (> n 500) (error "loader never wrapped"))
    (call-with-values (lambda () (loader-next-batch! loader))
      (lambda (inp tgt)
        (if (= 1 (loader-epoch loader))
            (call-with-values (lambda () (batch->lists inp tgt)) list)
            (loop (+ n 1)))))))

(define wrapped (read-until-wrap small-loader))

(test-equal "end of file starts the next epoch" 1 (loader-epoch small-loader))

(test-assert "the first batch after a wrap restarts at the top of the file"
  (equal? wrapped epoch-start-batch))

(loader-close! small-loader)
(loader-close! text-loader)
(loader-close! jsonl-loader*)

(for-each (lambda (p) (when (file-exists? p) (delete-file p)))
          (list text-path jsonl-path small-path))

(let ((failures (test-runner-fail-count (test-runner-current))))
  (test-end "sonicboom-loader")
  (exit (if (zero? failures) 0 1)))
