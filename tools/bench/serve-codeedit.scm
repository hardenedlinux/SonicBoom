;; SPDX-License-Identifier: GPL-3.0-or-later
;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;
;; This program is free software: you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.

;; serve-codeedit: the code-edit turns through a running SonicBoom server.
;;
;; Ten frozen edit requests (tools/bench/codeedit: NN.instruction, NN.file and
;; manifest.tsv), each its own request at position 0, generating the file's own
;; token count from codeedit_bench's output for the same model. The stream is
;; greedy (temperature 0, top_k 1) and asks for usage
;; (stream_options.include_usage), so each turn's generated-token count comes
;; from the server's own completion_tokens, not from counting chunks.
;;
;; Endpoint: this tree's server has /v1/chat/completions and no raw
;; /v1/completions, so the turn prompt (the instruction bytes followed by the
;; file bytes) goes as the sole user message and the server's chat template
;; wraps it. There is therefore no add_special flag to send: the chat template
;; owns the special tokens. The token budget still comes from codeedit_bench,
;; which tokenizes the raw turn prompt with the same model.
;;
;; The server is started and stopped by the caller (tools/bench-serve.sh owns
;; the process and the port); this script is one client pass over the ten
;; turns. It waits for /health once so a run against a server still loading
;; fails before any request is timed.
;;
;; Usage:
;;   guile tools/bench/serve-codeedit.scm --port P --model-id ID --data DIR
;;        --gen-tokens FILE --arm NAME --out DIR
;;
;; Output:
;;   turn=NN gen_tokens=G seconds=S hash=<16 hex digits>
;;   RESULT serve arm=NAME turns=10 gen_tokens=TOTAL seconds=TOTAL tok_s=X
;;          hash00=<hex> ... hash09=<hex>
;; and each turn's text in DIR/NAME.NN.txt. The hash is FNV-1a over the
;; turn text's UTF-8 bytes (64-bit, printed as 16 hex digits): a compact,
;; dependency-free identity for "the same text on two arms".

(use-modules (ice-9 binary-ports)
             (ice-9 format)
             (ice-9 rdelim)
             (json)
             ((rnrs bytevectors) #:select (utf8->string string->utf8
                                          bytevector-length
                                          bytevector-u8-ref))
             (srfi srfi-1)
             (srfi srfi-13)
             (web client)
             (web response)
             (web uri))

(define port-number 8000)
(define model-id #f)
(define data-dir #f)
(define gen-file #f)
(define out-dir #f)
(define arm-name #f)

(define (die fmt . args)
  "Report a fatal error on stderr and exit 1.

Args:
  fmt: an (ice-9 format) format string.
  args: its arguments.
Returns: never."
  (apply format (current-error-port) fmt args)
  (newline (current-error-port))
  (exit 1))

(define (slurp path)
  "Return the whole file PATH as a string.

Args:
  path: a file name.
Returns: the file's bytes decoded as UTF-8."
  (call-with-input-file path
    (lambda (p)
      (utf8->string (get-bytevector-all p)))))

;; manifest.tsv: index, source path, sha256, add_special. add_special is read
;; for completeness; the chat endpoint takes no such flag.
(define (read-turns)
  "Read the ten turns from DATA/manifest.tsv.

Returns: a list of (index add-special instruction-path file-path), in file
order. add-special is #t when the manifest's fourth column is 1, as in
llama-codeedit.scm."
  (let ((rows '()))
    (call-with-input-file (string-append data-dir "/manifest.tsv")
      (lambda (p)
        (let loop ()
          (let ((line (read-line p)))
            (unless (eof-object? line)
              (unless (string-null? (string-trim-both line))
                (let* ((f (string-split line #\tab))
                       (index (list-ref f 0))
                       (add-special (string=? (string-trim-both (list-ref f 3))
                                              "1")))
                  (set! rows
                        (cons (list index add-special
                                    (string-append data-dir "/" index
                                                   ".instruction")
                                    (string-append data-dir "/" index ".file"))
                              rows))))
              (loop))))))
    (reverse rows)))

;; codeedit_bench's output: turn=NN ... gen_tokens=G (first rep is enough).
(define (read-gen-tokens path)
  "Read each turn's generation budget from codeedit_bench's output.

Args:
  path: the file codeedit_bench wrote.
Returns: an alist (turn-index . gen_tokens), first occurrence per turn."
  (let ((gens '()))
    (call-with-input-file path
      (lambda (p)
        (let loop ()
          (let ((line (read-line p)))
            (unless (eof-object? line)
              (when (string-prefix? "turn=" line)
                (let* ((fields (string-split (string-trim-both line) #\space))
                       (turn (substring (car fields) 5)))
                  (unless (assoc turn gens)
                    (let ((gen (find (lambda (f)
                                       (string-prefix? "gen_tokens=" f))
                                     fields)))
                      (when gen
                        (set! gens
                              (cons (cons turn
                                          (string->number
                                           (substring gen 11)))
                                    gens)))))))
              (loop))))))
    gens))

(define (jref obj key fallback)
  "Return the value of KEY in the JSON object alist OBJ, or FALLBACK.

Args:
  obj: a JSON object alist (may be #f).
  key: the member name.
  fallback: returned when OBJ is not a pair or has no such member."
  (if (and (pair? obj) (assoc key obj)) (cdr (assoc key obj)) fallback))

(define (fnv1a-64 text)
  "Return the 64-bit FNV-1a hash of TEXT's UTF-8 bytes.

Args:
  text: a string.
Returns: an exact integer below 2^64."
  (let ((bv (string->utf8 text)))
    (let loop ((i 0) (h #xcbf29ce484222325))
      (if (>= i (bytevector-length bv))
          h
          (loop (+ i 1)
                (logand (* (logxor h (bytevector-u8-ref bv i))
                           #x100000001b3)
                        #xffffffffffffffff))))))

(define (hex64 h)
  "Format the 64-bit integer H as 16 lowercase hex digits.

Args:
  h: an exact integer below 2^64.
Returns: a 16-character string."
  (format #f "~16,'0x" h))

(define (server-healthy?)
  "Return #t when the server at PORT-NUMBER answers /health with 200.

Returns: a boolean; any connection error is #f."
  (catch #t
    (lambda ()
      (let ((uri (build-uri 'http #:host "127.0.0.1" #:port port-number
                            #:path "/health")))
        (call-with-values (lambda () (http-get uri))
          (lambda (resp body) (= (response-code resp) 200)))))
    (lambda _ #f)))

;; One streamed /v1/chat/completions request; returns text, tokens, seconds.
(define (complete prompt n-predict)
  "Stream one greedy chat-completion request and collect its text.

Args:
  prompt: the turn's instruction followed by the file, as one user message.
  n-predict: max_tokens, the turn's generation budget.
Returns: (text completion-tokens seconds), where completion-tokens is the
server's usage count and seconds is the wall time from request to [DONE]."
  (let* ((body `(("model" . ,model-id)
                 ("messages" . #((("role" . "user") ("content" . ,prompt))))
                 ("max_tokens" . ,n-predict)
                 ("temperature" . 0.0)
                 ("top_k" . 1)
                 ("stream" . #t)
                 ("stream_options" . (("include_usage" . #t)))))
         (uri (build-uri 'http #:host "127.0.0.1" #:port port-number
                         #:path "/v1/chat/completions"))
         (t0 (get-internal-real-time)))
    (call-with-values
        (lambda ()
          (http-request uri
                        #:method 'POST
                        #:headers '((content-type application/json))
                        #:body (scm->json-string body)
                        #:streaming? #t))
      (lambda (resp body-port)
        (unless (= (response-code resp) 200)
          (die "serve-codeedit: /v1/chat/completions -> ~a"
               (response-code resp)))
        (let loop ((chunks '()) (gen #f) (done #f))
          (let ((line (read-line body-port)))
            (cond
             ((eof-object? line)
              (unless done
                (die "serve-codeedit: stream ended without [DONE]"))
              (let ((seconds
                     (/ (- (get-internal-real-time) t0)
                        (exact->inexact internal-time-units-per-second))))
                (list (apply string-append (reverse chunks))
                      (or gen 0) seconds)))
             ((string-prefix? "data: " line)
              (let ((payload (string-trim-both (substring line 6))))
                (cond
                 ((string-null? payload) (loop chunks gen done))
                 ((string=? payload "[DONE]") (loop chunks gen #t))
                 (else
                  (let ((obj (json->scm (open-input-string payload))))
                    ;; A mid-stream error object ends the reply without the
                    ;; finish and usage chunks; the turn has no server count,
                    ;; so the run must fail rather than record a partial one.
                    (let ((err (jref obj "error" #f)))
                      (when (pair? err)
                        (die "serve-codeedit: the server reported an error: ~a"
                             (jref err "message" "?"))))
                    (let ((usage (jref obj "usage" #f)))
                      (when (pair? usage)
                        (set! gen (jref usage "completion_tokens" gen))))
                    (let* ((choices (jref obj "choices" #f))
                           (choice (and (vector? choices)
                                        (> (vector-length choices) 0)
                                        (vector-ref choices 0)))
                           (delta (jref choice "delta" #f))
                           (content (jref delta "content" #f)))
                      (loop (if (string? content)
                                (cons content chunks)
                                chunks)
                            gen done)))))))
             (else (loop chunks gen done)))))))))

(define (parse-args args)
  "Parse the command line ARGS.

Returns: a list (port model-id data gen-file arm out), all present."
  (let loop ((rest args))
    (cond
     ((null? rest)
      (unless (and model-id data-dir gen-file out-dir arm-name)
        (die "usage: serve-codeedit.scm --port P --model-id ID --data DIR ~
             --gen-tokens FILE --arm NAME --out DIR"))
      (list port-number model-id data-dir gen-file arm-name out-dir))
     ((string=? (car rest) "--port")
      (set! port-number (string->number (cadr rest)))
      (loop (cddr rest)))
     ((string=? (car rest) "--model-id")
      (set! model-id (cadr rest))
      (loop (cddr rest)))
     ((string=? (car rest) "--data")
      (set! data-dir (cadr rest))
      (loop (cddr rest)))
     ((string=? (car rest) "--gen-tokens")
      (set! gen-file (cadr rest))
      (loop (cddr rest)))
     ((string=? (car rest) "--arm")
      (set! arm-name (cadr rest))
      (loop (cddr rest)))
     ((string=? (car rest) "--out")
      (set! out-dir (cadr rest))
      (loop (cddr rest)))
     (else (die "serve-codeedit: unknown option ~a" (car rest))))))

(define (main)
  "Run the ten turns once against the running server and print the record.

Returns: nothing; exits non-zero on a failed request or a missing turn."
  (parse-args (cdr (command-line)))
  (unless (server-healthy?)
    (die "serve-codeedit: no healthy server at 127.0.0.1:~a" port-number))
  (let* ((turns (read-turns))
         (gens (read-gen-tokens gen-file))
         (gen-sum 0)
         (seconds-sum 0.0)
         (hashes '()))
    (system (format #f "mkdir -p ~a"
                    (string-join (list "'" out-dir "'") "")))
    (for-each
     (lambda (turn)
       (let* ((index (list-ref turn 0))
              (prompt (string-append (slurp (list-ref turn 2))
                                     (slurp (list-ref turn 3))))
              (gen-entry (assoc index gens)))
         (unless gen-entry
           (die "serve-codeedit: no gen_tokens for turn ~a" index))
         (let* ((budget (cdr gen-entry))
                (res (complete prompt budget))
                (text (list-ref res 0))
                (gen (list-ref res 1))
                (seconds (list-ref res 2))
                (hash (hex64 (fnv1a-64 text))))
           (call-with-output-file
               (format #f "~a/~a.~a.txt" out-dir arm-name index)
             (lambda (p) (display text p)))
           (format #t "turn=~a gen_tokens=~a seconds=~a hash=~a~%"
                   index gen seconds hash)
           (set! gen-sum (+ gen-sum gen))
           (set! seconds-sum (+ seconds-sum seconds))
           (set! hashes (cons (cons index hash) hashes)))))
     turns)
    (format #t "RESULT serve arm=~a turns=~a gen_tokens=~a seconds=~a ~
                tok_s=~a~{ hash~a=~a~}~%"
            arm-name (length turns) gen-sum seconds-sum
            (if (> seconds-sum 0.0) (/ gen-sum seconds-sum) 0.0)
            (append-map (lambda (h) (list (car h) (cdr h)))
                        (reverse hashes)))))

(main)
