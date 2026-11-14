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

;; A streaming token packer over a text or JSONL file.
;;
;; A loader keeps the file path, the tokenizer, the window and batch sizes and
;; the token ids read ahead. Lines are read on demand and never all at once, so
;; the file may be larger than memory. loader-seq-len and loader-batch-size
;; return the two sizes, loader-epoch the number of completed passes, and
;; loader-next-batch! produces the batches.
;; PyTorch: none
;; Note: The nearest PyTorch concept is a torch.utils.data.IterableDataset, which
;; leaves the packing to the user.
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
  "A loader over a plain text file, one document per line.

Lines are tokenized with the given tokenizer and separated by its <eos> token.
Args:
  path: the file path, reopened at the start of every epoch.
  tokenizer: a tokenizer from (sonicboom tokenizer bpe).
  seq-len: the number of input tokens per window, a positive integer.
  batch-size: the number of windows per batch, a positive integer.
Returns: a loader.
Example:
  (use-modules (sonicboom data loader) (sonicboom tokenizer bpe))
  (let ((path \"/tmp/sonicboom-docs-loader.txt\"))
    (call-with-output-file path
      (lambda (p) (display \"hello world\\n\" p)))
    (let* ((tok (bpe-train (open-input-string \"hello world\") 259))
           (l (text-file-loader path tok 2 1)))
      (loader-close! l)))
  => #f
PyTorch: torch.utils.data.DataLoader
Note: The DataLoader batches samples; this loader packs a token stream into
fixed next-token windows."
  (check-params "text-file-loader" seq-len batch-size)
  (%make-from path #f tokenizer seq-len batch-size))

(define (jsonl-loader path text-key tokenizer seq-len batch-size)
  "A loader over a JSONL file, taking one string field per line.

Each line must be a JSON object with a string under text-key; blank lines are
skipped. Documents are packed exactly as with text-file-loader.
Args:
  path: the file path, reopened at the start of every epoch.
  text-key: the JSON field holding the document text.
  tokenizer: a tokenizer from (sonicboom tokenizer bpe).
  seq-len: the number of input tokens per window, a positive integer.
  batch-size: the number of windows per batch, a positive integer.
Returns: a loader.
Example:
  (use-modules (sonicboom data loader) (sonicboom tokenizer bpe))
  (let ((path \"/tmp/sonicboom-docs-loader.jsonl\"))
    (call-with-output-file path
      (lambda (p) (display \"{\\\"text\\\": \\\"hello world\\\"}\\n\" p)))
    (let* ((tok (bpe-train (open-input-string \"hello world\") 259))
           (l (jsonl-loader path \"text\" tok 2 1)))
      (loader-close! l)))
  => #f
PyTorch: torch.utils.data.IterableDataset
Note: The IterableDataset supplies samples; this loader supplies packed
next-token batches."
  (unless (string? text-key)
    (error "jsonl-loader: text-key must be a string" text-key))
  (check-params "jsonl-loader" seq-len batch-size)
  (%make-from path text-key tokenizer seq-len batch-size))

(define (loader-close! loader)
  "Close the loader's input port.

A later loader-next-batch! reopens the file; loader-close! is therefore only
a way to release the port between uses.
Args:
  loader: a loader.
Returns: #f.
Example:
  (use-modules (sonicboom data loader) (sonicboom tokenizer bpe))
  (let ((path \"/tmp/sonicboom-docs-loader-close.txt\"))
    (call-with-output-file path
      (lambda (p) (display \"hello world\\n\" p)))
    (let* ((tok (bpe-train (open-input-string \"hello world\") 259))
           (l (text-file-loader path tok 2 1)))
      (loader-close! l)))
  => #f
PyTorch: none"
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
  "The next packed batch from the loader.

A batch is batch-size consecutive windows of seq-len + 1 tokens: the window's
first seq-len tokens are the input and the same tokens shifted by one are the
target. No batch mixes two epochs, and every batch is full.
Args:
  loader: a loader.
Returns: two values, the int32 input and target ctensors, each of shape
[batch-size seq-len].
Example:
  (use-modules (sonicboom data loader) (sonicboom tokenizer bpe)
               (sonicboom trace tensor))
  (let ((path \"/tmp/sonicboom-docs-loader-batch.txt\"))
    (call-with-output-file path
      (lambda (p) (display \"abc\\n\" p)))
    (let* ((tok (bpe-train (open-input-string \"abc\") 259))
           (l (text-file-loader path tok 2 1)))
      (call-with-values (lambda () (loader-next-batch! l))
        (lambda (inp tgt) (ctensor->list inp)))))
  => (97 98)
PyTorch: none
Note: A byte-level tokenizer maps each byte to its own id. The target of the
example is (98 99)."
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
