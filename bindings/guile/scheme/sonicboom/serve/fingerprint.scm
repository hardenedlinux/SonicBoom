;;; SonicBoom — inference service: the reproducibility fingerprint
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; A fingerprint names the scope inside which a reply is guaranteed
;;; bit-identical when the request and the seed are the same:
;;;
;;;   engine version and commit   libsonicboom's git tag and commit
;;;   numerics profile            strict or standard
;;;   GPU model, driver           nvidia-smi's name and driver_version
;;;   tune-table digest           libsonicboom's boom_tune_digest (the tables,
;;;                               not the append-only logs)
;;;   model                       SHA-256 of the whole GGUF, first 16 hex
;;;
;;; The digest of that record is its `id`; two servers agree exactly when the
;;; id agrees. The tune digest comes from the library itself, which knows which
;;; files are tuning tables and which are runtime logs; the model hash uses
;;; sha256sum, once per process and model, because a GGUF is far larger.
;;;
;;; None of this is part of the C ABI: it is the serving layer's report about
;;; the library and the machine it is running on.

(define-module (sonicboom serve fingerprint)
  #:use-module (srfi srfi-13)
  #:use-module (ice-9 match)
  #:use-module (ice-9 popen)
  #:use-module (ice-9 rdelim)
  #:use-module (rnrs bytevectors)
  #:use-module (rnrs arithmetic bitwise)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom inference)
  #:export (fingerprint-for-model clear-fingerprint-cache!))

;;; ---------------------------------------------------------------------------
;;; Small helpers

(define (hash-hex bv)
  "FNV-1a (64-bit) of BV, as 16 lowercase hex digits.

Args:
  bv: a bytevector.
Returns: a 16-character string.
Example:
  (use-modules (sonicboom serve fingerprint) (rnrs bytevectors))
  (string-length (hash-hex (string->utf8 \"x\")))
  => 16
PyTorch: none"
  (let loop ((i 0) (h #xcbf29ce484222325))
    (if (>= i (bytevector-length bv))
        (string-pad (number->string h 16) 16 #\0)
        (loop (1+ i)
              (bitwise-and #xffffffffffffffff
                           (* (bitwise-xor h (bytevector-u8-ref bv i))
                              #x100000001b3))))))

(define (string-hash-hex s) (hash-hex (string->utf8 s)))

;; One line of an external command's stdout, trimmed; #f when it fails.
(define (command-line* . argv)
  (catch #t
    (lambda ()
      (let* ((p (apply open-pipe* OPEN_READ argv))
             (line (read-line p))
             (status (close-pipe p)))
        (and (zero? status) line (string-trim-both line))))
    (lambda _ #f)))

;;; ---------------------------------------------------------------------------
;;; Server scope (cached for the process)

(define server-scope #f)

(define (build-identity)
  (let ((info (catch #t build-info (lambda _ '()))))
    (cons (or (assq-ref info 'git_tag) "unknown")
          (or (assq-ref info 'git_commit) "unknown"))))

(define (gpu-identity)
  (let ((line (command-line* "nvidia-smi"
                             "--query-gpu=name,driver_version"
                             "--format=csv,noheader")))
    (match (and line (string-split line #\,))
      ((name driver) (cons (string-trim-both name) (string-trim-both driver)))
      (_ (cons "none" "none")))))

(define (server-fingerprint)
  "The server-scope part of the fingerprint, computed once per process.

It holds engine, commit, numerics, gpu, driver and tune.

Args:
  none.
Returns: an alist with string keys.
Example:
  (use-modules (sonicboom serve fingerprint))
  (assoc-ref (fingerprint-for-model \"m\" #f) \"numerics\")
  => \"standard\"
PyTorch: none"
  (unless server-scope
    (let* ((id (build-identity))
           (gpu (gpu-identity)))
      (set! server-scope
            `(("engine" . ,(car id))
              ("commit" . ,(cdr id))
              ("numerics" . ,(symbol->string (numerics-profile)))
              ("gpu" . ,(car gpu))
              ("driver" . ,(cdr gpu))
              ("tune" . ,(tune-digest))))))
  server-scope)

;;; ---------------------------------------------------------------------------
;;; Model scope + record digest

(define model-hashes (make-hash-table))

;; The model's SHA-256, first 16 hex digits, cached in the process and in /tmp
;; (keyed by name, size and mtime) so a second server does not re-read a
;; multi-gigabyte file. #f for a model with no file (the mock backend).
(define (model-sha-prefix path)
  (and path
       (or (hash-ref model-hashes path)
           (let* ((st (stat path))
                  (key (format #f "~a-~a-~a" (basename path)
                               (stat:size st) (stat:mtime st)))
                  (cache (string-append (or (getenv "TMPDIR") "/tmp")
                                        "/sonicboom-model-sha256-" key))
                  (sha (if (file-exists? cache)
                           (string-trim-both
                            (call-with-input-file cache read-line))
                           (let ((line (command-line* "sha256sum" path)))
                             (when line
                               (let ((h (car (string-split line #\space))))
                                 (catch #t
                                   (lambda ()
                                     (call-with-output-file cache
                                       (lambda (p) (display h p) (newline p))))
                                   (lambda _ #f))
                                 h))))))
             (when sha
               (hash-set! model-hashes path sha)
               (string-take sha 16))))))

(define (fingerprint-for-model model-id model-path)
  "The fingerprint for a reply from model-id whose file is model-path.

The record is the server scope plus model (the file's SHA-256 prefix, or
\"none\") and id, an FNV-1a over the record's canonical text. Two servers agree
exactly when id agrees.

Args:
  model-id: the served model id string.
  model-path: the GGUF path, or #f for a model with no file.
Returns: an alist with string keys.
Example:
  (use-modules (sonicboom serve fingerprint))
  (assoc-ref (fingerprint-for-model \"m\" #f) \"model\")
  => \"none\"
PyTorch: none"
  (let* ((sha (model-sha-prefix model-path))
         (record
          (append (server-fingerprint)
                  `(("model" . ,(if sha (string-append "sha256:" sha)
                                    (string-append "none:" model-id))))))
         (canonical
          (call-with-output-string
            (lambda (out)
              (for-each (lambda (kv)
                          (format out "~a=~a\n" (car kv) (cdr kv)))
                        record)))))
    (append record `(("id" . ,(string-hash-hex canonical))))))

(define (clear-fingerprint-cache!)
  "Forget the cached fingerprints (tests only).

Args:
  none.
Returns: unspecified.
Example:
  (use-modules (sonicboom serve fingerprint))
  (begin (clear-fingerprint-cache!) #t)
  => #t
PyTorch: none"
  (set! server-scope #f)
  (set! model-hashes (make-hash-table)))
