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

;; llama-codeedit: the code-edit turns through llama-server.
;;
;; Ten frozen edit requests (tools/bench/codeedit: NN.instruction, NN.file
;; and manifest.tsv), each its own stateless request whose prompt is the
;; instruction bytes followed by the file bytes.
;;
;; Two request shapes:
;;
;;   --chat   /v1/chat/completions with the turn as one user message, so the
;;            model's own chat template wraps it exactly as SonicBoom's server
;;            wraps the same turn. max_tokens is the file's own token count
;;            from codeedit_bench's output for the same model; generation
;;            stops at the turn-end token, as SonicBoom's does.
;;   default  raw /completion of the unwrapped prompt with ignore_eos on
;;            (the reference runner's shape).
;;
;; Greedy (temperature 0, seed 1234), cache_prompt on. A chat run passes
;; enable_thinking:false both to the server and per request, which is what a
;; Qwen3.5/3.6 template needs to emit its empty think block.
;;
;; --canonical-dir DIR makes every turn prove its prompt before it is timed:
;; the server renders the same message through /apply-template and tokenizes
;; the result through /tokenize, and both the bytes and the ids must equal
;; DIR/NN.txt and DIR/NN.ids (written by the cross-engine comparison's prompt
;; snapshot). The server's reported prompt_tokens must equal the id count too.
;;
;; The server is started here and stopped at the end:
;;
;;   llama-server -m GGUF -ngl 99 -fa on -np 1 --no-webui --seed 1234
;;                -c CTX --port PORT [--spec-type ngram-mod]
;;
;; ngram-mod keeps one n-gram container per server and grows it with every
;; request, so a second rep on the same server could draft from the first
;; rep's own output. That arm therefore runs one server per rep, as the
;; reference runner does; the no-spec arm keeps one server for the run.
;;
;; Usage:
;;   guile tools/bench/llama-codeedit.scm --server PATH --model GGUF
;;        --data DIR --ctx CTX --gen-tokens FILE --arm off|ngram-mod
;;        --reps N --out DIR [--moe-args TEXT] [--chat] [--canonical-dir DIR]
;;
;; Output, one line per turn per rep, then one line per rep:
;;   turn=NN rep=R prompt_n=P predicted_n=G predicted_ms=T
;;   rep=R tok_s=<sum G / sum T over the ten turns * 1000>
;; and each turn's content is written to OUT/ARM.repR.NN.txt.

(use-modules (ice-9 binary-ports)
             (ice-9 format)
             (ice-9 rdelim)
             (json)
             ((rnrs bytevectors) #:select (utf8->string string->utf8))
             (srfi srfi-1)
             (srfi srfi-13)
             (web client)
             (web response)
             (web uri))

(define port-number 18080)
(define data-dir #f)
(define out-dir #f)
(define arm-name #f)
(define server-path #f)
(define model-path #f)
(define ctx-size #f)
(define gen-file #f)
(define reps 1)
(define moe-args "")
(define chat-mode? #f)
(define canonical-dir #f)

(define (die fmt . args)
  (apply format (current-error-port) fmt args)
  (newline (current-error-port))
  (exit 1))

(define (shell-quote s)
  (string-append "'" (string-join (string-split s #\') "'\\''") "'"))

(define (slurp path)
  (call-with-input-file path
    (lambda (p)
      (utf8->string (get-bytevector-all p)))))

(define (split-tab line) (string-split line #\tab))

;; manifest.tsv: index, source path, sha256, add_special.
(define (read-turns)
  (let ((rows '()))
    (call-with-input-file (string-append data-dir "/manifest.tsv")
      (lambda (p)
        (let loop ()
          (let ((line (read-line p)))
            (unless (eof-object? line)
              (unless (string-null? (string-trim-both line))
                (let* ((f (split-tab line))
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
  (let ((p (assoc key obj)))
    (if p (cdr p) fallback)))

(define (http-json method path body)
  (let ((uri (build-uri 'http #:host "127.0.0.1" #:port port-number
                        #:path path)))
    (call-with-values
        (lambda ()
          ;; (web client) validates a typed header, so the media type is given
          ;; unquoted, as (web http) expects it.
          (http-request uri
                        #:method method
                        #:headers '((content-type application/json))
                        #:body (and body (scm->json-string body))))
      (lambda (resp body)
        (values (response-code resp) body)))))

;; The same request, parsed as JSON. (web client) hands the body back as a
;; string or as bytes depending on the response headers; both are UTF-8 JSON.
(define (http-json-body method path body)
  (call-with-values (lambda () (http-json method path body))
    (lambda (code resp-body)
      (unless (= code 200)
        (die "llama-codeedit: ~a -> ~a" path code))
      (json->scm (open-input-string
                  (if (string? resp-body)
                      resp-body
                      (utf8->string resp-body)))))))

(define (jref-ids obj)
  (map (lambda (x) (if (number? x) x (jref x "id" #f)))
       (vector->list (jref obj "tokens" #()))))

(define (read-ids path)
  (call-with-input-file path
    (lambda (p)
      (let lp ((acc '()))
        (let ((x (read p)))
          (if (eof-object? x) (reverse acc) (lp (cons x acc))))))))

;; The chat body a timed turn sends. The template kwargs are the same on the
;; server's command line; repeating them here keeps the body the one the
;; snapshot tool renders.
(define (chat-body content max-tokens)
  `(("messages" . #((("role" . "user") ("content" . ,content))))
    ("max_tokens" . ,max-tokens)
    ("temperature" . 0.0)
    ("top_k" . 1)
    ("top_p" . 1.0)
    ("min_p" . 0.0)
    ("repeat_penalty" . 1.0)
    ("seed" . 1234)
    ("cache_prompt" . #t)
    ("stream" . #t)
    ("stream_options" . (("include_usage" . #t)))
    ("chat_template_kwargs" . (("enable_thinking" . #f)))))

;; Prove this turn's prompt is the canonical one: the server's own /apply-template
;; and /tokenize must reproduce DIR/NN.txt and DIR/NN.ids byte for byte.
;; Returns the canonical id count for the prompt_tokens check.
(define (canonical-check index content)
  (let* ((base (format #f "~a/~a" canonical-dir index))
         (want-text (slurp (string-append base ".txt")))
         (want-ids (read-ids (string-append base ".ids")))
         (got-text (jref (http-json-body 'POST "/apply-template"
                                         (chat-body content 1))
                         "prompt" #f))
         (got-ids (jref-ids (http-json-body 'POST "/tokenize"
                                            `(("content" . ,got-text)
                                              ("add_special" . #t)
                                              ("parse_special" . #t))))))
    (unless (string=? got-text want-text)
      (die "llama-codeedit: turn ~a rendered a different prompt than ~a"
           index (string-append base ".txt")))
    (unless (equal? got-ids want-ids)
      (die "llama-codeedit: turn ~a tokenized to different ids than ~a"
           index (string-append base ".ids")))
    (length want-ids)))

(define (server-healthy?)
  (catch #t
    (lambda ()
      (call-with-values (lambda () (http-json 'GET "/health" #f))
        (lambda (code body) (= code 200))))
    (lambda _ #f)))

(define (wait-health proc-stop? timeout)
  (let loop ((waited 0))
    (cond ((server-healthy?) #t)
          ((>= waited timeout) #f)
          ((and proc-stop? (proc-stop?)) #f)
          (else (usleep 200000) (loop (+ waited 1))))))

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

;; MOE-ARGS is the llama.cpp offload argument text for a routed-expert model
;; (e.g. "-ncmoe 16"), or "" when the model is dense. It holds no spaces
;; inside an argument, so splitting on spaces recovers argv.
(define (start-server server model ctx arm pidfile logfile moe-args)
  (let* ((args (append (list server "-m" model "-ngl" "99" "-fa" "on"
                             "-np" "1" "--no-webui" "--seed" "1234"
                             "-c" (number->string ctx)
                             "--port" (number->string port-number))
                       (if (string=? arm "ngram-mod")
                           (list "--spec-type" "ngram-mod")
                           '())
                       ;; A thinking model answers directly only when its
                       ;; template is told thinking is off; the flag is the
                       ;; same one run-llama-chat.sh passes.
                       (if chat-mode?
                           (list "--chat-template-kwargs"
                                 "{\"enable_thinking\":false}")
                           '())
                       (if (string-null? moe-args)
                           '()
                           (string-split moe-args #\space))))
         (launcher (string-append out-dir "/server-" arm ".sh")))
    ;; The launcher records its own pid and becomes the server, so the pid
    ;; read back is the server's. It also puts the server's own directory on
    ;; the loader path the way the reference runner does.
    (when (file-exists? pidfile) (delete-file pidfile))
    (call-with-output-file launcher
      (lambda (p)
        (format p "#!/bin/sh\nexport LD_LIBRARY_PATH=~a:\"${LD_LIBRARY_PATH-}\"\n"
                (shell-quote (dirname server)))
        (format p "echo $$ > ~a\nexec ~a\n"
                (shell-quote pidfile)
                (string-join (map shell-quote args) " "))))
    (system (format #f "sh ~a > ~a 2>&1 &"
                    (shell-quote launcher) (shell-quote logfile)))
    (let ((pid (read-pid pidfile)))
      (unless pid
        (die "llama-codeedit: server did not start (~a)" logfile))
      (unless (wait-health (lambda () (not (pid-alive? pid))) 1500)
        (die "llama-codeedit: server did not become healthy (~a)" logfile))
      pid)))

;; The launcher exec'd the server and the shell that started it has exited, so
;; the server is not this process's child: it cannot be waitpid'd here, and its
;; exit is observed by polling the pid.
(define (stop-server pid)
  (when pid
    (catch #t (lambda () (kill pid SIGTERM)) (lambda _ #f))
    (let loop ((tries 0))
      (when (and (< tries 300) (pid-alive? pid))
        (usleep 100000)
        (loop (+ tries 1))))
    (catch #t (lambda () (kill pid SIGKILL)) (lambda _ #f))))

;; One /completion request, streamed; returns the turn's text and timings.
(define (complete prompt n-predict add-special)
  (let ((body `(("prompt" . ,prompt)
                ("n_predict" . ,n-predict)
                ("temperature" . 0.0)
                ("top_k" . 1)
                ("top_p" . 1.0)
                ("min_p" . 0.0)
                ("repeat_penalty" . 1.0)
                ("seed" . 1234)
                ("cache_prompt" . #t)
                ("ignore_eos" . #t)
                ("stream" . #t)
                ("add_special" . ,(if add-special #t #f))))
        (uri (build-uri 'http #:host "127.0.0.1" #:port port-number
                        #:path "/completion")))
    (call-with-values
        (lambda ()
          (http-request uri
                        #:method 'POST
                        #:headers '((content-type application/json))
                        #:body (scm->json-string body)
                        #:streaming? #t))
      (lambda (resp body-port)
        (let ((code (response-code resp)))
          (unless (= code 200)
            (die "llama-codeedit: /completion -> ~a" code))
          ;; Under LC_ALL=C the default port encoding is ASCII; the stream is
          ;; UTF-8 JSON and the answers may be any script.
          (set-port-encoding! body-port "UTF-8")
          (let loop ((chunks '()) (final #f))
            (let ((line (read-line body-port)))
              (cond
               ((eof-object? line)
                (unless final
                  (die "llama-codeedit: stream ended without a stop chunk"))
                (let ((tm (jref final "timings" '())))
                  (list (apply string-append (reverse chunks))
                        (jref tm "prompt_n" 0)
                        (jref tm "predicted_n" 0)
                        (jref tm "predicted_ms" 0.0))))
               ((string-prefix? "data: " line)
                (let* ((payload (string-trim-both (substring line 6)))
                       (obj (and (not (string-null? payload))
                                 (json->scm (open-input-string payload)))))
                  (cond
                   ((not obj) (loop chunks final))
                   ((jref obj "stop" #f) (loop chunks obj))
                   (else
                    (let ((c (jref obj "content" #f)))
                      (loop (if c (cons c chunks) chunks) final))))))
               (else (loop chunks final))))))))))

;; One /v1/chat/completions request, streamed, with the turn as one user
;; message. Returns (text predicted-n prompt-tokens predicted-ms); the
;; predicted figures are the server's own timings for the request, and
;; prompt-tokens is the count it evaluated (cached + processed).
(define (complete-chat content n-predict)
  (let ((uri (build-uri 'http #:host "127.0.0.1" #:port port-number
                        #:path "/v1/chat/completions")))
    (call-with-values
        (lambda ()
          (http-request uri
                        #:method 'POST
                        #:headers '((content-type application/json))
                        #:body (scm->json-string (chat-body content n-predict))
                        #:streaming? #t))
      (lambda (resp body-port)
        (unless (= (response-code resp) 200)
          (die "llama-codeedit: /v1/chat/completions -> ~a"
               (response-code resp)))
        ;; Under LC_ALL=C the default port encoding is ASCII; the stream is
        ;; UTF-8 JSON and the answers may be any script.
        (set-port-encoding! body-port "UTF-8")
        (let loop ((chunks '()) (usage #f) (timings #f) (done #f))
          (let ((line (read-line body-port)))
            (cond
             ((eof-object? line)
              (unless done
                (die "llama-codeedit: stream ended without a stop chunk"))
              (unless timings
                (die "llama-codeedit: the server sent no timings"))
              (list (apply string-append (reverse chunks))
                    (or (and (pair? usage) (jref usage "prompt_tokens" #f))
                        (jref timings "prompt_n" 0))
                    (jref timings "predicted_n" 0)
                    (jref timings "predicted_ms" 0.0)))
             ((string-prefix? "data: " line)
              (let* ((payload (string-trim-both (substring line 6)))
                     ;; [DONE] is not JSON; it ends the OpenAI SSE stream and
                     ;; must be recognised before anything parses the line.
                     (obj (and (not (string-null? payload))
                               (not (string=? payload "[DONE]"))
                               (json->scm (open-input-string payload)))))
                (cond
                 ((string=? payload "[DONE]") (loop chunks usage timings #t))
                 ((not obj) (loop chunks usage timings done))
                 ((jref obj "stop" #f) (loop chunks usage timings obj))
                 (else
                  (let* ((u (jref obj "usage" #f))
                         (tm (jref obj "timings" #f))
                         (choices (jref obj "choices" #()))
                         (choice (and (vector? choices)
                                      (> (vector-length choices) 0)
                                      (vector-ref choices 0)))
                         (delta (and choice (jref choice "delta" #f)))
                         (c (and delta (jref delta "content" #f))))
                    (loop (if (string? c) (cons c chunks) chunks)
                          (if (pair? u) u usage)
                          (if (pair? tm) tm timings)
                          done))))))
             (else (loop chunks usage timings done)))))))))

(define (parse-args args)
  (let loop ((rest args))
    (cond
     ((null? rest)
      (unless (and server-path model-path data-dir ctx-size gen-file arm-name)
        (die "usage: llama-codeedit.scm --server PATH --model GGUF --data DIR ~
             --ctx CTX --gen-tokens FILE --arm off|ngram-mod --reps N --out DIR ~
             [--moe-args TEXT] [--chat] [--canonical-dir DIR]"))
      (unless (or (string=? arm-name "off") (string=? arm-name "ngram-mod"))
        (die "llama-codeedit: --arm must be off or ngram-mod")))
     ((string=? (car rest) "--server")
      (set! server-path (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--model")
      (set! model-path (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--data")
      (set! data-dir (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--ctx")
      (set! ctx-size (string->number (cadr rest))) (loop (cddr rest)))
     ((string=? (car rest) "--gen-tokens")
      (set! gen-file (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--arm")
      (set! arm-name (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--reps")
      (set! reps (string->number (cadr rest))) (loop (cddr rest)))
     ((string=? (car rest) "--out")
      (set! out-dir (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--moe-args")
      (set! moe-args (cadr rest)) (loop (cddr rest)))
     ((string=? (car rest) "--chat")
      (set! chat-mode? #t) (loop (cdr rest)))
     ((string=? (car rest) "--canonical-dir")
      (set! canonical-dir (cadr rest)) (loop (cddr rest)))
     (else (die "llama-codeedit: unknown option ~a" (car rest))))))

;; One rep: every turn, in order; returns (gen . ms) sums and the contents.
(define (run-rep turns gens rep)
  (let ((gen-sum 0) (ms-sum 0.0))
    (for-each
     (lambda (turn)
       (let* ((index (list-ref turn 0))
              (add-special (list-ref turn 1))
              (prompt (string-append (slurp (list-ref turn 2))
                                     (slurp (list-ref turn 3))))
              (gen-entry (assoc index gens)))
         (unless gen-entry
           (die "llama-codeedit: no gen_tokens for turn ~a" index))
         (let* ((gen (cdr gen-entry))
                ;; With --canonical-dir the timed server itself must render and
                ;; tokenize this turn to the snapshot's bytes and ids.
                (want-n (and canonical-dir (canonical-check index prompt)))
                (res (if chat-mode?
                         (complete-chat prompt gen)
                         (complete prompt gen add-special)))
                (text (list-ref res 0))
                (prompt-n (list-ref res 1))
                (predicted-n (list-ref res 2))
                (predicted-ms (list-ref res 3)))
           (when (and want-n (not (= prompt-n want-n)))
             (die "llama-codeedit: turn ~a: the server evaluated ~a prompt ~
                  tokens, the canonical prompt has ~a"
                  index prompt-n want-n))
           ;; Bytes, not a textual port: the default output encoding under
           ;; LC_ALL=C is ASCII and would replace every non-ASCII character
           ;; with '?' (which would also collapse the comparison's text).
           (call-with-output-file
               (format #f "~a/~a.rep~a.~a.txt" out-dir arm-name rep index)
             (lambda (p) (put-bytevector p (string->utf8 text)))
             #:binary #t)
           (format #t "turn=~a rep=~a prompt_n=~a predicted_n=~a predicted_ms=~a~%"
                   index rep prompt-n predicted-n predicted-ms)
           (set! gen-sum (+ gen-sum predicted-n))
           (set! ms-sum (+ ms-sum predicted-ms)))))
     turns)
    (format #t "rep=~a tok_s=~a~%"
            rep (if (> ms-sum 0.0) (/ (* gen-sum 1000.0) ms-sum) 0.0))))

(define (main)
  (parse-args (cdr (command-line)))
  (when (and canonical-dir (not chat-mode?))
    (die "llama-codeedit: --canonical-dir needs --chat (the snapshot is a chat prompt)"))
  (let* ((turns (read-turns))
         (gens (read-gen-tokens gen-file)))
    (system (format #f "mkdir -p ~a" (shell-quote out-dir)))
    (let ((pidfile (format #f "~a/server.pid" out-dir))
          (logfile (format #f "~a/server_~a.log" out-dir arm-name)))
      (if (string=? arm-name "ngram-mod")
          ;; A fresh server per rep: the n-gram container is per server and
          ;; would otherwise carry one rep's output into the next.
          (let loop ((rep 0) (pid #f))
            (when (< rep reps)
              (when pid (stop-server pid))
              (let ((pid (start-server server-path model-path ctx-size arm-name
                                       pidfile logfile moe-args)))
                (run-rep turns gens rep)
                (loop (+ rep 1) pid)))
            (when pid (stop-server pid)))
          ;; The no-spec arm keeps one server for the run; it starts with an
          ;; empty cache.
          (let ((pid (start-server server-path model-path ctx-size arm-name
                                   pidfile logfile moe-args)))
            (let loop ((rep 0))
              (when (< rep reps)
                (run-rep turns gens rep)
                (loop (+ rep 1))))
            (stop-server pid)))
      (when (file-exists? pidfile) (delete-file pidfile)))))

(main)
