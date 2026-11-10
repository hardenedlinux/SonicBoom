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
;; and manifest.tsv), each its own stateless /completion request: the prompt
;; is the instruction bytes followed by the file bytes, and n_predict is the
;; file's own token count from codeedit_bench's output for the same model.
;; Greedy (temperature 0, seed 1234), cache_prompt on, ignore_eos on.
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
;;        --reps N --out DIR
;;
;; Output, one line per turn per rep, then one line per rep:
;;   turn=NN rep=R prompt_n=P predicted_n=G predicted_ms=T
;;   rep=R tok_s=<sum G / sum T over the ten turns * 1000>
;; and each turn's content is written to OUT/ARM.repR.NN.txt.

(use-modules (ice-9 binary-ports)
             (ice-9 format)
             (ice-9 rdelim)
             (json)
             ((rnrs bytevectors) #:select (utf8->string))
             (srfi srfi-1)
             (srfi srfi-13)
             (web client)
             (web response)
             (web uri))

(define port-number 18080)
(define data-dir #f)
(define out-dir #f)
(define arm-name #f)

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

(define (start-server server model ctx arm pidfile logfile)
  (let* ((args (append (list server "-m" model "-ngl" "99" "-fa" "on"
                             "-np" "1" "--no-webui" "--seed" "1234"
                             "-c" (number->string ctx)
                             "--port" (number->string port-number))
                       (if (string=? arm "ngram-mod")
                           (list "--spec-type" "ngram-mod")
                           '())))
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

(define (parse-args args)
  (let loop ((rest args) (server #f) (model #f) (ctx #f) (gen-file #f)
             (arm #f) (reps #f))
    (cond
     ((null? rest)
      (unless (and server model data-dir ctx gen-file arm reps)
        (die "usage: llama-codeedit.scm --server PATH --model GGUF --data DIR ~
             --ctx CTX --gen-tokens FILE --arm off|ngram-mod --reps N --out DIR"))
      (unless (or (string=? arm "off") (string=? arm "ngram-mod"))
        (die "llama-codeedit: --arm must be off or ngram-mod"))
      (list server model ctx gen-file arm reps))
     ((string=? (car rest) "--server")
      (loop (cddr rest) (cadr rest) model ctx gen-file arm reps))
     ((string=? (car rest) "--model")
      (loop (cddr rest) server (cadr rest) ctx gen-file arm reps))
     ((string=? (car rest) "--data")
      (set! data-dir (cadr rest))
      (loop (cddr rest) server model ctx gen-file arm reps))
     ((string=? (car rest) "--ctx")
      (loop (cddr rest) server model (string->number (cadr rest)) gen-file arm
            reps))
     ((string=? (car rest) "--gen-tokens")
      (loop (cddr rest) server model ctx (cadr rest) arm reps))
     ((string=? (car rest) "--arm")
      (loop (cddr rest) server model ctx gen-file (cadr rest) reps))
     ((string=? (car rest) "--reps")
      (loop (cddr rest) server model ctx gen-file arm
            (string->number (cadr rest))))
     ((string=? (car rest) "--out")
      (set! out-dir (cadr rest))
      (loop (cddr rest) server model ctx gen-file arm reps))
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
                (res (complete prompt gen add-special))
                (text (list-ref res 0))
                (prompt-n (list-ref res 1))
                (predicted-n (list-ref res 2))
                (predicted-ms (list-ref res 3)))
           (call-with-output-file
               (format #f "~a/~a.rep~a.~a.txt" out-dir arm-name rep index)
             (lambda (p) (display text p)))
           (format #t "turn=~a rep=~a prompt_n=~a predicted_n=~a predicted_ms=~a~%"
                   index rep prompt-n predicted-n predicted-ms)
           (set! gen-sum (+ gen-sum predicted-n))
           (set! ms-sum (+ ms-sum predicted-ms)))))
     turns)
    (format #t "rep=~a tok_s=~a~%"
            rep (if (> ms-sum 0.0) (/ (* gen-sum 1000.0) ms-sum) 0.0))))

(define (main)
  (let* ((parsed (parse-args (cdr (command-line))))
         (server (list-ref parsed 0))
         (model (list-ref parsed 1))
         (ctx (list-ref parsed 2))
         (gen-file (list-ref parsed 3))
         (arm (list-ref parsed 4))
         (reps (list-ref parsed 5))
         (turns (read-turns))
         (gens (read-gen-tokens gen-file)))
    (set! arm-name arm)
    (system (format #f "mkdir -p ~a" (shell-quote out-dir)))
    (let ((pidfile (format #f "~a/server.pid" out-dir))
          (logfile (format #f "~a/server_~a.log" out-dir arm)))
      (if (string=? arm "ngram-mod")
          ;; A fresh server per rep: the n-gram container is per server and
          ;; would otherwise carry one rep's output into the next.
          (let loop ((rep 0) (pid #f))
            (when (< rep reps)
              (when pid (stop-server pid))
              (let ((pid (start-server server model ctx arm pidfile logfile)))
                (run-rep turns gens rep)
                (loop (+ rep 1) pid)))
            (when pid (stop-server pid)))
          ;; The no-spec arm keeps one server for the run; it starts with an
          ;; empty cache.
          (let ((pid (start-server server model ctx arm pidfile logfile)))
            (let loop ((rep 0))
              (when (< rep reps)
                (run-rep turns gens rep)
                (loop (+ rep 1))))
            (stop-server pid)))
      (when (file-exists? pidfile) (delete-file pidfile)))))

(main)
