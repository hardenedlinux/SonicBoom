;;; SonicBoom — Guile binding: streaming sequence packer / dataset loader
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Turns a line-per-document text file (plain text or JSONL) into fixed-shape
;;; next-token batches:
;;;
;;;   - lines are read one at a time, tokenized with the caller's tokenizer,
;;;     and separated by the tokenizer's <eos> token;
;;;   - the token stream is cut into non-overlapping windows of SEQ-LEN + 1
;;;     tokens; a window's first SEQ-LEN tokens are the input and the same
;;;     window shifted by one is the target;
;;;   - BATCH-SIZE consecutive windows form one batch, so one batch consumes
;;;     BATCH-SIZE * (SEQ-LEN + 1) tokens;
;;;   - at end of file the incomplete tail is dropped, the file is reopened
;;;     and the epoch increments. No batch ever mixes two epochs, and every
;;;     batch is full.
;;;
;;; Nothing here is bound to a particular corpus: any plain text file (one
;;; document per line) or any JSONL file with a string field works.

(define-module (sonicboom data loader)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 format)
  #:use-module (ice-9 rdelim)
  #:use-module (json)
  #:use-module (sonicboom tokenizer bpe)
  #:use-module (sonicboom trace tensor)
  #:export (text-file-loader
            jsonl-loader
            loader-next-batch!
            loader-epoch
            loader-seq-len
            loader-batch-size
            loader-close!))

(define-record-type <loader>
  (%make-loader path key tokenizer seq-len batch-size eos port pending
                epoch started?)
  loader?
  (path     loader-path)               ; file to (re)read at every epoch
  (key      loader-json-key)           ; #f for plain text; string for JSONL
  (tokenizer loader-tokenizer)         ; a (sonicboom tokenizer bpe) tokenizer
  (seq-len  loader-seq-len)
  (batch-size loader-batch-size)
  (eos      loader-eos)                ; <eos> id, or #f when absent
  (port     loader-port set-loader-port!)
  (pending  loader-pending set-loader-pending!)  ; token ids (list)
  (epoch    loader-epoch set-loader-epoch!)
  (started? loader-started? set-loader-started?!)) ; a batch was emitted

;;; --- construction -----------------------------------------------------------

(define (check-params who seq-len batch-size)
  (unless (and (integer? seq-len) (> seq-len 0))
    (error who "seq-len must be a positive integer" seq-len))
  (unless (and (integer? batch-size) (> batch-size 0))
    (error who "batch-size must be a positive integer" batch-size)))

(define (%open path) (open-input-file path #:encoding "UTF-8"))

(define (%make-from path key tokenizer seq-len batch-size)
  (%make-loader path key tokenizer seq-len batch-size
                (bpe-special-id tokenizer "<eos>")
                (%open path) '() 0 #f))

(define (text-file-loader path tokenizer seq-len batch-size)
  "A loader over a plain text file, one document per line."
  (check-params "text-file-loader" seq-len batch-size)
  (%make-from path #f tokenizer seq-len batch-size))

(define (jsonl-loader path text-key tokenizer seq-len batch-size)
  "A loader over a JSONL file, taking each line's TEXT-KEY string as the
document."
  (unless (string? text-key)
    (error "jsonl-loader: text-key must be a string" text-key))
  (check-params "jsonl-loader" seq-len batch-size)
  (%make-from path text-key tokenizer seq-len batch-size))

(define (loader-close! loader)
  "Close the loader's input port (a following loader-next-batch! reopens it)."
  (let ((port (loader-port loader)))
    (when (and port (not (port-closed? port)))
      (close-port port))
    (set-loader-port! loader #f)))

;;; --- reading ----------------------------------------------------------------

(define (line-text loader line)
  "The document text of LINE (the line itself for plain text; the TEXT-KEY
string of the JSON object for JSONL). #f means skip the line."
  (if (loader-json-key loader)
      (if (string-null? line)
          #f
          (let* ((obj (json->scm (open-input-string line)))
                 (text (and (pair? obj)
                            (assoc-ref obj (loader-json-key loader)))))
            (unless (string? text)
              (error "jsonl-loader: line has no string under"
                     (loader-json-key loader)))
            text))
      line))

(define (line->ids loader line)
  (let ((text (line-text loader line)))
    (if (or (not text) (string-null? text))
        '()
        (let ((ids (bpe-encode (loader-tokenizer loader) text)))
          (if (loader-eos loader)
              (append ids (list (loader-eos loader)))
              ids)))))

(define (%wrap! loader)
  "End the current epoch: drop the incomplete tail, reopen the file, and bump
the epoch. A file that cannot fill a single batch is an error, not a hang."
  (unless (loader-started? loader)
    (error "loader: the file yields fewer tokens than one batch"
           (loader-path loader)
           (* (loader-batch-size loader) (+ (loader-seq-len loader) 1))))
  (loader-close! loader)
  (set-loader-port! loader (%open (loader-path loader)))
  (set-loader-pending! loader '())
  (set-loader-started?! loader #f)
  (set-loader-epoch! loader (+ 1 (loader-epoch loader))))

(define (fill! loader need)
  "Buffer at least NEED token ids. Returns #t when they are buffered and #f
when the epoch ended first (the buffer is then empty and the file reopened)."
  (let loop ()
    (cond ((>= (length (loader-pending loader)) need) #t)
          (else
           (let ((line (read-line (loader-port loader))))
             (cond ((eof-object? line)
                    (%wrap! loader)
                    #f)
                   (else
                    (let ((ids (line->ids loader line)))
                      (unless (null? ids)
                        (set-loader-pending!
                         loader (append (loader-pending loader) ids))))
                    (loop))))))))

;;; --- batching ---------------------------------------------------------------

(define (split-windows tokens seq-len batch-size)
  "Cut the first BATCH-SIZE windows of SEQ-LEN + 1 tokens from TOKENS into an
input and a target vector, each of length BATCH-SIZE * SEQ-LEN."
  (define per (* batch-size seq-len))
  (let ((inp (make-vector per 0))
        (tgt (make-vector per 0)))
    (let window ((w 0) (rest tokens))
      (when (< w batch-size)
        (let ((win (take rest (+ seq-len 1))))
          (let elem ((j 0))
            (when (< j seq-len)
              (vector-set! inp (+ (* w seq-len) j) (list-ref win j))
              (vector-set! tgt (+ (* w seq-len) j) (list-ref win (+ j 1)))
              (elem (+ j 1))))
          (window (+ w 1) (drop rest (+ seq-len 1))))))
    (values inp tgt)))

(define (loader-next-batch! loader)
  "The next batch as (values input target): two int32 ctensors of shape
[BATCH-SIZE SEQ-LEN], where target is input shifted by one token."
  (let* ((seq-len (loader-seq-len loader))
         (batch-size (loader-batch-size loader))
         (need (* batch-size (+ seq-len 1))))
    (let ensure ((ok (fill! loader need)))
      (unless ok (ensure (fill! loader need))))
    (let ((tokens (loader-pending loader)))
      (call-with-values
          (lambda () (split-windows tokens seq-len batch-size))
        (lambda (inp tgt)
          (set-loader-pending! loader (drop tokens need))
          (set-loader-started?! loader #t)
          (values (make-ctensor 'int32 (list batch-size seq-len) inp)
                  (make-ctensor 'int32 (list batch-size seq-len) tgt)))))))
