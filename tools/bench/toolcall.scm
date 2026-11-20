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

;; toolcall: the constrained tool-call turns, one side per invocation.
;;
;; Ten frozen prompts (tools/bench/toolcall: NN.prompt and manifest.tsv) each
;; ask for one tool call; the reply is constrained to the shared JSON schema
;; (schema.json) and decoded greedily. Both sides use the same GGUF:
;;
;;   --side boom   SonicBoom through its Guile C-ABI binding: a fresh session
;;                 per turn, set-json-schema!, prefill!, decode 256.
;;   --side llama  llama-server /completion with `json_schema`, one server for
;;                 the run (prompt cache off), started here and stopped at the
;;                 end. Same flags tools/bench.sh uses, plus the caller's
;;                 --moe-args for a routed-expert model.
;;
;; Output, one line per turn per rep, then one line per rep:
;;   turn=NN rep=R side=S tokens=N prefill_ms=P decode_ms=G hash=H valid=0|1
;;   rep=R side=S tok_s=T
;; and each turn's text in OUT/SIDE.repR.NN.txt. The hash is FNV-1a over the
;; text's UTF-8 bytes; valid is 1 when the text parses as the schema's object
;; (name from the enum, arguments an object with the four required typed
;; fields). tools/bench-toolcall.sh compares the two sides' files and rows.
;;
;; Usage:
;;   guile tools/bench/toolcall.scm --side boom|llama --model GGUF
;;        --data DIR --runs N --out DIR [--server PATH --port P]
;;        [--moe-args TEXT]

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
             (web uri)
             (sonicboom inference))

(define side #f)
(define model-path #f)
(define data-dir #f)
(define out-dir #f)
(define runs 3)
(define server-path #f)
(define port-number 18080)
(define moe-args "")
(define n-ctx 2048)

(define (die fmt . args)
  (apply format (current-error-port) fmt args)
  (newline (current-error-port))
  (exit 1))

(define (shell-quote s)
  (string-append "'" (string-join (string-split s #\') "'\\''") "'"))

(define (slurp path)
  (call-with-input-file path
    (lambda (p) (utf8->string (get-bytevector-all p)))))

(define (parse-args rest)
  (let loop ((rest rest))
    (cond
     ((null? rest)
      (unless (and side model-path data-dir out-dir)
        (die "usage: toolcall.scm --side boom|llama --model GGUF --data DIR ~
             --runs N --out DIR [--server PATH --port P] [--moe-args TEXT]"))
      (unless (member side '("boom" "llama"))
        (die "toolcall: --side must be boom or llama"))
      (when (string=? side "llama") (unless server-path (die "toolcall: --server is required for the llama side"))))
     ((string=? (car rest) "--side") (set! side (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--model") (set! model-path (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--data") (set! data-dir (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--runs") (set! runs (string->number (cadr rest))) (loop (cddr rest)))
     ((string=? (car rest) "--out") (set! out-dir (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--server") (set! server-path (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--port") (set! port-number (string->number (cadr rest))) (loop (cddr rest)))
     ((string=? (car rest) "--moe-args") (set! moe-args (cadr rest)) (loop (cddr rest)))
     (else (die "toolcall: unknown option ~a" (car rest))))))

;; manifest.tsv: id, file, n_predict.
(define (read-turns)
  (let ((rows '()))
    (call-with-input-file (string-append data-dir "/manifest.tsv")
      (lambda (p)
        (let loop ()
          (let ((line (read-line p)))
            (unless (eof-object? line)
              (unless (string-null? (string-trim-both line))
                (let* ((f (string-split line #\tab))
                       (id (list-ref f 0))
                       (file (list-ref f 1))
                       (n (string->number (string-trim-both (list-ref f 2)))))
                  (set! rows (cons (list id file n) rows))))
              (loop))))))
    (reverse rows)))

(define (fnv1a-64 text)
  (let ((bv (string->utf8 text)))
    (let loop ((i 0) (h #xcbf29ce484222325))
      (if (>= i (bytevector-length bv))
          h
          (loop (+ i 1)
                (logand (* (logxor h (bytevector-u8-ref bv i))
                           #x100000001b3)
                        #xffffffffffffffff))))))

;; The schema's object, as text (boom compiles it) and as data (llama sends it).
(define schema-text (delay (slurp (string-append data-dir "/schema.json"))))

;; valid? checks the one object shape schema.json admits: a name from the
;; enum and an arguments object carrying the four required, typed fields.
(define (valid-tool-call? text)
  (catch #t
    (lambda ()
      (let ((obj (json-string->scm text)))
        (and (list? obj)
             (let ((name (assoc-ref obj "name"))
                   (args (assoc-ref obj "arguments")))
               (and (member name '("search_web" "read_file" "write_file"
                                   "run_command"))
                    (list? args)
                    (string? (assoc-ref args "path"))
                    (string? (assoc-ref args "query"))
                    (string? (assoc-ref args "content"))
                    (integer? (assoc-ref args "timeout_s")))))))
    (lambda _ #f)))

(define (report-turn id rep tokens prefill-ms decode-ms text)
  (call-with-output-file
      (format #f "~a/~a.rep~a.~a.txt" out-dir side rep id)
    (lambda (p) (display text p)))
  (format #t "turn=~a rep=~a side=~a tokens=~a prefill_ms=~a decode_ms=~a hash=~16,'0x valid=~a~%"
          id rep side tokens prefill-ms decode-ms (fnv1a-64 text)
          (if (valid-tool-call? text) 1 0)))

;; --- SonicBoom side --------------------------------------------------------

(define (now-t) (get-internal-real-time))
(define (secs a b) (/ (- b a) 1.0 internal-time-units-per-second))

(define (boom-turn model turn rep)
  (let* ((id (list-ref turn 0))
         (text (slurp (string-append data-dir "/" (list-ref turn 1))))
         (n (list-ref turn 2))
         (session (make-session model n-ctx))
         (prompt (tokenize model text)))
    (set-json-schema! session (force schema-text))
    (let* ((t0 (now-t))
           (_ (prefill! session prompt))
           (t1 (now-t))
           (tokens (generate-tokens session '() n #f))
           (t2 (now-t)))
      (free-session! session)
      (report-turn id rep (length tokens) (* 1000.0 (secs t0 t1))
                   (* 1000.0 (secs t1 t2)) (detokenize model tokens))
      (cons (length tokens) (* 1000.0 (secs t1 t2))))))

(define (run-boom turns)
  (let ((model (load-model model-path)))
    ;; One untimed warm-up pass: builds the decode runtime and the session
    ;; set; its timing describes process setup, not the measured turns.
    (let* ((warm (car turns))
           (session (make-session model n-ctx)))
      (set-json-schema! session (force schema-text))
      (prefill! session (tokenize model (slurp (string-append data-dir "/" (list-ref warm 1)))))
      (generate-tokens session '() 16 #f)
      (free-session! session))
    (let loop ((rep 0))
      (when (< rep runs)
        (let ((gen 0) (ms 0.0))
          (for-each
           (lambda (turn)
             (let ((r (boom-turn model turn rep)))
               (set! gen (+ gen (car r)))
               (set! ms (+ ms (cdr r)))))
           turns)
          (format #t "rep=~a side=~a tok_s=~a~%"
                  rep side (if (> ms 0.0) (/ (* gen 1000.0) ms) 0.0)))
        (loop (+ rep 1))))
    (free-model! model)))

;; --- llama-server side -----------------------------------------------------

(define (http-json method path body)
  (let ((uri (build-uri 'http #:host "127.0.0.1" #:port port-number
                        #:path path)))
    (call-with-values
        (lambda ()
          (http-request uri
                        #:method method
                        #:headers '((content-type application/json))
                        #:body (and body (scm->json-string body))))
      (lambda (resp body) (values (response-code resp) body)))))

(define (jref obj key fallback)
  (let ((p (and (list? obj) (assoc key obj))))
    (if p (cdr p) fallback)))

(define (server-healthy?)
  (catch #t
    (lambda ()
      (call-with-values (lambda () (http-json 'GET "/health" #f))
        (lambda (code body) (= code 200))))
    (lambda _ #f)))

(define (read-pid pidfile)
  (let loop ((tries 0))
    (cond ((file-exists? pidfile)
           (let ((line (string-trim-both (slurp pidfile))))
             (if (string-null? line)
                 (begin (usleep 50000) (loop (+ tries 1)))
                 (string->number line))))
          ((> tries 100) #f)
          (else (usleep 50000) (loop (+ tries 1))))))

(define (pid-alive? pid)
  (catch #t (lambda () (kill pid 0) #t) (lambda _ #f)))

(define (start-server pidfile logfile)
  (let* ((args (append (list server-path "-m" model-path "-ngl" "99"
                             "-fa" "on" "-np" "1" "--no-webui"
                             "--seed" "1234" "-c" (number->string n-ctx)
                             "--port" (number->string port-number))
                       (if (string-null? moe-args)
                           '()
                           (string-split moe-args #\space))))
         (launcher (string-append out-dir "/server-toolcall.sh")))
    (when (file-exists? pidfile) (delete-file pidfile))
    (call-with-output-file launcher
      (lambda (p)
        (format p "#!/bin/sh\nexport LD_LIBRARY_PATH=~a:\"${LD_LIBRARY_PATH-}\"\n"
                (shell-quote (dirname server-path)))
        (format p "echo $$ > ~a\nexec ~a\n"
                (shell-quote pidfile)
                (string-join (map shell-quote args) " "))))
    (system (format #f "sh ~a > ~a 2>&1 &"
                    (shell-quote launcher) (shell-quote logfile)))
    (let ((pid (read-pid pidfile)))
      (unless pid (die "toolcall: llama-server did not start (~a)" logfile))
      (let loop ((waited 0))
        (cond ((server-healthy?) pid)
              ((>= waited 3000) (die "toolcall: llama-server did not become healthy"))
              ((> waited 0)
               (unless (pid-alive? pid) (die "toolcall: llama-server exited"))
               (usleep 200000) (loop (+ waited 1)))
              (else (usleep 200000) (loop (+ waited 1))))))))

(define (stop-server pid)
  (when pid
    (catch #t (lambda () (kill pid 15)) (lambda _ #f))
    (let loop ((tries 0))
      (when (and (< tries 300) (pid-alive? pid))
        (usleep 100000)
        (loop (+ tries 1))))
    (catch #t (lambda () (kill pid 9)) (lambda _ #f))))

(define (llama-turn turn rep)
  (let* ((id (list-ref turn 0))
         (text (slurp (string-append data-dir "/" (list-ref turn 1))))
         (n (list-ref turn 2))
         (body `(("prompt" . ,text)
                 ("n_predict" . ,n)
                 ("temperature" . 0.0)
                 ("top_k" . 1)
                 ("top_p" . 1.0)
                 ("min_p" . 0.0)
                 ("repeat_penalty" . 1.0)
                 ("seed" . 1234)
                 ("cache_prompt" . #f)
                 ("add_special" . #f)
                 ("stream" . #f)
                 ("json_schema" . ,(json-string->scm (force schema-text)))))
         (res (call-with-values (lambda () (http-json 'POST "/completion" body))
                (lambda (code body) (cons code body)))))
    (unless (= (car res) 200)
      (die "toolcall: /completion returned ~a" (car res)))
    (let* ((obj (json-string->scm (utf8->string (cdr res))))
           (content (jref obj "content" ""))
           (tm (jref obj "timings" '()))
           (predicted-n (jref tm "predicted_n" 0))
           (prompt-ms (jref tm "prompt_ms" 0.0))
           (predicted-ms (jref tm "predicted_ms" 0.0)))
      (report-turn id rep predicted-n prompt-ms predicted-ms content)
      (cons predicted-n predicted-ms))))

(define (run-llama turns)
  (system (format #f "mkdir -p ~a" (shell-quote out-dir)))
  (let* ((pidfile (format #f "~a/server.pid" out-dir))
         (logfile (format #f "~a/server.log" out-dir))
         (pid (start-server pidfile logfile)))
    (let loop ((rep 0))
      (when (< rep runs)
        (let ((gen 0) (ms 0.0))
          (for-each
           (lambda (turn)
             (let ((r (llama-turn turn rep)))
               (set! gen (+ gen (car r)))
               (set! ms (+ ms (cdr r)))))
           turns)
          (format #t "rep=~a side=~a tok_s=~a~%"
                  rep side (if (> ms 0.0) (/ (* gen 1000.0) ms) 0.0)))
        (loop (+ rep 1))))
    (stop-server pid)
    (when (file-exists? pidfile) (delete-file pidfile))))

(define (main)
  (parse-args (cdr (command-line)))
  (system (format #f "mkdir -p ~a" (shell-quote out-dir)))
  (let ((turns (read-turns)))
    (if (string=? side "boom")
        (run-boom turns)
        (run-llama turns))))

(main)
