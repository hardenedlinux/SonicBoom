;;; SonicBoom — Guile binding: byte-level BPE tokenizer
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A generic byte-level BPE tokenizer (the CS336 A1 tokenizer). Nothing here
;;; is bound to a dataset or a model: training reads any text port, and the
;;; tokenizer it returns can be saved, loaded and reused on any string.
;;;
;;; Vocabulary layout:
;;;
;;;   ids 0..255      the 256 byte tokens (one byte each)
;;;   ids 256..258    the built-in special tokens <unk>, <bos>, <eos>
;;;   ids 259..       learned merges, in merge-rank order
;;;   ids after that  special tokens added with bpe-add-special!
;;;
;;; Encoding is greedy lowest-rank BPE over the UTF-8 bytes, with special
;;; tokens matched first (longest match) wherever they occur in the text, so a
;;; string that contains "<eos>" encodes to the eos id and decodes back to the
;;; same text. Every byte is in the vocabulary, so <unk> is only a decode
;;; fallback for an out-of-range id.

(define-module (sonicboom tokenizer bpe)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (rnrs bytevectors)
  #:use-module (ice-9 format)
  #:use-module (ice-9 rdelim)
  #:use-module (ice-9 textual-ports)
  #:export (bpe-train
            bpe-encode
            bpe-decode
            bpe-save
            bpe-load
            bpe-add-special!
            bpe-vocab-size
            bpe-special-id
            bpe-special-ids))

;;; --- tokenizer record -------------------------------------------------------

(define-record-type <bpe>
  (%make-bpe vocab token->id merges merge-rank specials)
  bpe?
  (vocab      bpe-vocab set-bpe-vocab!)  ; vector: id -> token bytes
  (token->id  bpe-token->id)             ; hash: token bytes -> first id
  (merges     bpe-merges)                ; vector: rank -> (left-id . right-id)
  (merge-rank bpe-merge-rank)            ; hash: (left-id . right-id) -> rank
  (specials   bpe-specials set-bpe-specials!)) ; (bytes . id), longest first

(define %builtin-specials
  (map (lambda (s) (string->utf8 s)) '("<unk>" "<bos>" "<eos>")))

(define (%empty-token->id vocab)
  (let ((h (make-hash-table)))
    (let loop ((i 0))
      (when (< i (vector-length vocab))
        (unless (hash-ref h (vector-ref vocab i) #f)
          (hash-set! h (vector-ref vocab i) i))
        (loop (+ i 1))))
    h))

(define (%make-specials vocab ids)
  (sort (map (lambda (id) (cons (vector-ref vocab id) id)) ids)
        (lambda (a b) (> (bytevector-length (car a))
                         (bytevector-length (car b))))))

(define (%finish vocab-list merges-list special-ids)
  "Assemble a tokenizer from a list of token bytevectors, a list of merge
pairs in rank order, and the special-token ids."
  (let* ((vocab (list->vector vocab-list))
         (merges (list->vector merges-list))
         (rank (make-hash-table))
         (specials (%make-specials vocab special-ids)))
    (let loop ((i 0) (ms merges-list))
      (unless (null? ms)
        (hash-set! rank (car ms) i)
        (loop (+ i 1) (cdr ms))))
    (%make-bpe vocab (%empty-token->id vocab) merges rank specials)))

;;; --- input ------------------------------------------------------------------

;; Guile 3.0.9's (rnrs bytevectors) has no bytevector-append; concatenate here.
(define (bytes-append bvs)
  (let ((out (make-bytevector
              (fold (lambda (bv n) (+ n (bytevector-length bv))) 0 bvs)
              0)))
    (let loop ((at 0) (rest bvs))
      (if (null? rest)
          out
          (let ((bv (car rest)))
            (bytevector-copy! bv 0 out at (bytevector-length bv))
            (loop (+ at (bytevector-length bv)) (cdr rest)))))))

(define (port->bytes source)
  "All bytes of SOURCE: a text port is read to EOF as characters (so a
multi-byte character is never split) and a string is taken as its UTF-8."
  (cond ((string? source) (string->utf8 source))
        ((input-port? source) (string->utf8 (get-string-all source)))
        (else (error "bpe: expected a text port or a string" source))))

(define (bytes-at? bv i tok)
  (let ((n (bytevector-length tok)))
    (and (<= (+ i n) (bytevector-length bv))
         (let loop ((k 0))
           (or (= k n)
               (and (= (bytevector-u8-ref bv (+ i k))
                       (bytevector-u8-ref tok k))
                    (loop (+ k 1))))))))

(define (concat-tokens a b)
  (let* ((na (bytevector-length a)) (nb (bytevector-length b))
         (out (make-bytevector (+ na nb))))
    (bytevector-copy! a 0 out 0 na)
    (bytevector-copy! b 0 out na nb)
    out))

;;; --- training ---------------------------------------------------------------

(define (pair-counts ids)
  "Hash of ((left . right) . count) over every adjacent pair of IDS."
  (let ((counts (make-hash-table)))
    (let loop ((i 0))
      (when (< i (- (vector-length ids) 1))
        (let ((p (cons (vector-ref ids i) (vector-ref ids (+ i 1)))))
          (hash-set! counts p (+ 1 (hash-ref counts p 0))))
        (loop (+ i 1))))
    counts))

(define (most-frequent counts)
  "The pair with the highest count; ties break on the smaller (left, right),
so training is deterministic."
  (let loop ((pairs (hash-map->list cons counts))
             (best #f)
             (best-count -1))
    (if (null? pairs)
        best
        (let* ((p (caar pairs))
               (c (cdar pairs)))
          (if (or (> c best-count)
                  (and (= c best-count)
                       (or (< (car p) (car best))
                           (and (= (car p) (car best))
                                (< (cdr p) (cdr best))))))
              (loop (cdr pairs) p c)
              (loop (cdr pairs) best best-count))))))

(define (merge-pair ids left right new-id)
  (let* ((n (vector-length ids)) (out (make-vector n 0)))
    (let loop ((i 0) (w 0))
      (cond ((>= i n) (vector-copy out 0 w))
            ((and (< (+ i 1) n)
                  (= (vector-ref ids i) left)
                  (= (vector-ref ids (+ i 1)) right))
             (vector-set! out w new-id)
             (loop (+ i 2) (+ w 1)))
            (else
             (vector-set! out w (vector-ref ids i))
             (loop (+ i 1) (+ w 1)))))))

(define (bpe-train source vocab-size)
  "Train a byte-level BPE tokenizer on SOURCE, a text port (or a string), with
VOCAB-SIZE tokens in total: 256 byte tokens, the 3 built-in specials and
VOCAB-SIZE - 259 merges (fewer when the corpus runs out of pairs). Each
iteration merges the most frequent adjacent pair; ties break deterministically."
  (unless (and (integer? vocab-size) (>= vocab-size (+ 256 3)))
    (error "bpe-train: vocab-size must be at least 259" vocab-size))
  (let* ((bytes (port->bytes source))
         (n (bytevector-length bytes))
         (ids (make-vector n 0))
         (base-vocab (append (map (lambda (i) (make-bytevector 1 i))
                                  (iota 256))
                             %builtin-specials)))
    (let loop0 ((i 0))
      (when (< i n)
        (vector-set! ids i (bytevector-u8-ref bytes i))
        (loop0 (+ i 1))))
    (let loop ((ids ids)
               (vocab (reverse base-vocab))
               (merges '())
               (next-id (+ 256 3)))
      (if (or (>= next-id vocab-size) (< (vector-length ids) 2))
          (%finish (reverse vocab) (reverse merges)
                   (iota 3 256)) ; <unk> <bos> <eos>
          (let ((pair (most-frequent (pair-counts ids))))
            (if (not pair)
                (%finish (reverse vocab) (reverse merges) (iota 3 256))
                (let* ((left (car pair)) (right (cdr pair))
                       (left-bytes (vector-ref (list->vector (reverse vocab)) left))
                       (right-bytes (vector-ref (list->vector (reverse vocab)) right)))
                  ;; The merged token is left's bytes followed by right's.
                  (loop (merge-pair ids left right next-id)
                        (cons (concat-tokens left-bytes right-bytes) vocab)
                        (cons pair merges)
                        (+ next-id 1)))))))))

;;; --- encoding ---------------------------------------------------------------

(define (best-pair tok ids)
  (let ((rank (bpe-merge-rank tok)))
    (let loop ((i 0) (best #f) (best-rank #f))
      (if (>= i (- (vector-length ids) 1))
          best
          (let* ((p (cons (vector-ref ids i) (vector-ref ids (+ i 1))))
                 (r (hash-ref rank p #f)))
            (if (and r (or (not best-rank) (< r best-rank)))
                (loop (+ i 1) p r)
                (loop (+ i 1) best best-rank)))))))

(define (pair-result-id tok pair)
  "The vocabulary id whose bytes are left's bytes followed by right's. This is
usually 259 + rank, but two different merges can concatenate to the same bytes,
in which case the lower id is canonical."
  (or (hash-ref (bpe-token->id tok)
                (concat-tokens (vector-ref (bpe-vocab tok) (car pair))
                               (vector-ref (bpe-vocab tok) (cdr pair)))
                #f)
      (error "bpe-encode: merge result missing from the vocabulary" pair)))

(define (encode-segment tok bv start end)
  (let* ((n (- end start))
         (ids (make-vector n 0)))
    (let loop0 ((i 0))
      (when (< i n)
        (vector-set! ids i (bytevector-u8-ref bv (+ start i)))
        (loop0 (+ i 1))))
    (let loop ()
      (let ((pair (best-pair tok ids)))
        (if (not pair)
            (vector->list ids)
            (begin
              (set! ids (merge-pair ids (car pair) (cdr pair)
                                    (pair-result-id tok pair)))
              (loop)))))))

(define (bpe-encode tok string)
  "Encode STRING to a list of token ids. Special tokens are matched first
(longest match) wherever they occur; the text between them is byte-level BPE."
  (let ((bv (string->utf8 string))
        (specials (bpe-specials tok)))
    (let loop ((i 0) (acc '()))
      (if (>= i (bytevector-length bv))
          (reverse acc)
          (let ((hit (find (lambda (sp) (bytes-at? bv i (car sp))) specials)))
            (if hit
                (loop (+ i (bytevector-length (car hit)))
                      (cons (cdr hit) acc))
                (let* ((j (let scan ((k (+ i 1)))
                            ;; Find the next position where a special starts.
                            (if (or (>= k (bytevector-length bv))
                                    (any (lambda (sp) (bytes-at? bv k (car sp)))
                                         specials))
                                k
                                (scan (+ k 1)))))
                       (ids (encode-segment tok bv i j)))
                  (loop j (append (reverse ids) acc)))))))))

(define (bpe-decode tok ids)
  "Decode IDS back to a string. An id outside the vocabulary decodes as <unk>."
  (let* ((unk (hash-ref (bpe-token->id tok) (string->utf8 "<unk>") #f))
         (parts (map (lambda (id)
                       (if (and (integer? id) (>= id 0)
                                (< id (vector-length (bpe-vocab tok))))
                           (vector-ref (bpe-vocab tok) id)
                           (vector-ref (bpe-vocab tok) unk)))
                     ids)))
    (utf8->string (bytes-append parts))))

;;; --- special tokens ---------------------------------------------------------

(define (bpe-vocab-size tok) (vector-length (bpe-vocab tok)))

(define (bpe-special-ids tok) (map cdr (bpe-specials tok)))

(define (bpe-special-id tok text)
  "The id of the special token TEXT, or #f."
  (let ((id (hash-ref (bpe-token->id tok) (string->utf8 text) #f)))
    (and id (memv id (bpe-special-ids tok)) id)))

(define (bpe-add-special! tok text)
  "Make TEXT a special token (matched verbatim before BPE) in TOK and return
its id. If the vocabulary already holds those bytes, that token becomes
special; otherwise the token is appended after every existing id."
  (let* ((bytes (string->utf8 text))
         (n (bytevector-length bytes)))
    (when (= n 0) (error "bpe-add-special!: empty token"))
    (let* ((vocab (bpe-vocab tok))
           (existing (hash-ref (bpe-token->id tok) bytes #f))
           (id (or existing (vector-length vocab))))
      (unless existing
        (let ((v (make-vector (+ id 1))))
          (vector-copy! v 0 vocab 0 (vector-length vocab))
          (vector-set! v id bytes)
          (set-bpe-vocab! tok v)
          (hash-set! (bpe-token->id tok) bytes id)))
      (unless (memv id (bpe-special-ids tok))
        (set-bpe-specials!
         tok (%make-specials (bpe-vocab tok) (cons id (bpe-special-ids tok)))))
      id)))

;;; --- save / load ------------------------------------------------------------

;; Save format (PATH is a base name):
;;
;;   PATH.vocab   first line "# sonicboom-bpe v1 specials=<id,id,...>";
;;                then one token per line, hex-escaped (two hex digits a byte)
;;   PATH.merges  one merge per line, in rank order: "<hex left> <hex right>"
;;
;; The merge's result token is vocab[259 + rank], and load verifies that.

(define (hex-byte b)
  (let ((s (number->string b 16)))
    (if (< b 16) (string-append "0" s) s)))

(define (hex-encode bv)
  (apply string-append
         (map (lambda (b) (hex-byte b)) (bytevector->u8-list bv))))

(define (hex-decode s)
  (let* ((n (/ (string-length s) 2))
         (bv (make-bytevector n 0)))
    (let loop ((i 0))
      (when (< i n)
        (bytevector-u8-set! bv i
                            (string->number (substring s (* 2 i) (+ (* 2 i) 2)) 16))
        (loop (+ i 1))))
    bv))

(define (bpe-save tok path)
  "Write TOK's vocabulary to PATH.vocab and its merges to PATH.merges."
  (let ((special-ids (bpe-special-ids tok))
        (vocab (bpe-vocab tok)))
    (call-with-output-file (string-append path ".vocab")
      (lambda (port)
        (format port "# sonicboom-bpe v1 specials=~a~%"
                (string-join (map number->string special-ids) ","))
        (let loop ((i 0))
          (when (< i (vector-length vocab))
            (format port "~a~%" (hex-encode (vector-ref vocab i)))
            (loop (+ i 1))))))
    (call-with-output-file (string-append path ".merges")
      (lambda (port)
        (let loop ((i 0))
          (when (< i (vector-length (bpe-merges tok)))
            (let ((left (car (vector-ref (bpe-merges tok) i)))
                  (right (cdr (vector-ref (bpe-merges tok) i))))
              (format port "~a ~a~%"
                      (hex-encode (vector-ref vocab left))
                      (hex-encode (vector-ref vocab right))))
            (loop (+ i 1)))))))
  path)

(define (read-lines path)
  (call-with-input-file path
    (lambda (port)
      (let loop ((acc '()))
        (let ((line (read-line port)))
          (if (eof-object? line)
              (reverse acc)
              (loop (cons line acc))))))))

(define (bpe-load path)
  "Load a tokenizer written by bpe-save from PATH.vocab and PATH.merges."
  (let* ((lines (read-lines (string-append path ".vocab")))
         (header (and (pair? lines) (string-prefix? "#" (car lines)) (car lines)))
         (token-lines (if header (cdr lines) lines))
         (special-ids
          (let ((at (and header (string-contains header "specials="))))
            (if at
                (map string->number
                     (string-split (substring header (+ at 9)) #\,))
                (iota 3 256))))
         (vocab (list->vector (map hex-decode (filter (lambda (l) (> (string-length l) 0))
                                                      token-lines))))
         (token->id (%empty-token->id vocab))
         (merge-lines (filter (lambda (l) (> (string-length l) 0))
                              (read-lines (string-append path ".merges")))))
    (when (< (vector-length vocab) 259)
      (error "bpe-load: vocabulary is smaller than the 259 fixed tokens"))
    (let loop ((i 0) (ls merge-lines) (merges '()))
      (if (null? ls)
          (%finish (vector->list vocab) (reverse merges) special-ids)
          (let* ((parts (string-split (car ls) #\space))
                 (left-bytes (hex-decode (car parts)))
                 (right-bytes (hex-decode (cadr parts)))
                 (left (hash-ref token->id left-bytes #f))
                 (right (hash-ref token->id right-bytes #f))
                 (new-id (+ 259 i))
                 (want (concat-tokens left-bytes right-bytes)))
            (unless (= 2 (length parts))
              (error "bpe-load: malformed merge line" (car ls)))
            (unless (= (string-length (car parts)) (* 2 (bytevector-length left-bytes)))
              (error "bpe-load: malformed merge line" (car ls)))
            (unless (and left right)
              (error "bpe-load: merge references an unknown token"))
            (unless (and (< new-id (vector-length vocab))
                         (bytevector=? (vector-ref vocab new-id) want))
              (error "bpe-load: merge result is not at its rank position" i))
            (loop (+ i 1) (cdr ls) (cons (cons left right) merges)))))))
