;;; SonicBoom — inference service: /v1/images/generations end-to-end test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root (needs GNU Artanis, the MOSS-Image model and a
;;; CUDA device; skips cleanly when any is absent):
;;;
;;;   LD_LIBRARY_PATH=build/core guile -L bindings/guile/scheme \
;;;       bindings/guile/tests/serve/test-serve-images.scm
;;;
;;; Starts serve/app.scm against the mock chat backend (--backend mock), so no
;;; GGUF is loaded next to the image model, and checks:
;;;
;;;   1. a generation returns {"created", "data": [{"b64_json"}]}, and the
;;;      decoded base64 is a PNG of the requested size;
;;;   2. the same request twice returns the same PNG (a second generation in
;;;      the same process, so the DiT re-upload path is exercised);
;;;   3. image 0 of n=2 equals the single-image result;
;;;   4. response_format other than b64_json, a missing prompt, and a size that
;;;      is not a multiple of 16 are 400s with the right param;
;;;   5. stream=true sends one {"step","total","image"} event per denoising
;;;      step, then the same images object the non-stream reply holds, then
;;;      "[DONE]", under a sonicboom-job-id response header;
;;;   6. DELETE /sonicboom/jobs/<id> cancels a running generation (the stream
;;;      ends {"id","status":"cancelled"}), the cancelled job is then a 404,
;;;      and the next generation is unaffected (the device was released);
;;;   7. a client that goes away cancels its generation too.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (ice-9 rdelim)
             (ice-9 binary-ports)
             (rnrs bytevectors)
             (web client)
             (web response)
             (json)
             (sonicboom ffi))

;;; ---------------------------------------------------------------------------
;;; Preconditions

(define root (or (getenv "SONICBOOM_SRC_DIR") "."))
(define model-dir
  (or (getenv "SONICBOOM_MOSS_IMAGE_DIR")
      (string-append root "/models/MOSS-Image-SFT")))

(define (artanis-available?)
  (catch #t (lambda () (resolve-interface '(artanis artanis)) #t) (lambda _ #f)))

(define (cuda-build?)
  ;; A CPU-only libsonicboom.so has no VAE decoder even on a GPU host.
  (let ((cuda (assq-ref (build-info) 'cuda)))
    (and cuda (eq? (assq-ref cuda 'enabled) #t))))

(unless (and (artanis-available?)
             (file-exists? (string-append model-dir "/transformer"))
             (file-exists? "/dev/nvidia0")
             (cuda-build?))
  (format (current-error-port)
          "SKIP: needs GNU Artanis, ~a/transformer and a CUDA-enabled build~%"
          model-dir)
  (exit 0))

;;; ---------------------------------------------------------------------------
;;; The server: the mock chat backend, the real image model

(define (free-port)
  (let ((s (socket PF_INET SOCK_STREAM 0)))
    (bind s AF_INET INADDR_LOOPBACK 0)
    (let ((p (sockaddr:port (getsockname s))))
      (close s)
      p)))

(define port (free-port))
(define base (format #f "http://127.0.0.1:~a" port))
(define tmp (or (getenv "TMPDIR") "/tmp"))
(define log-file (format #f "~a/sonicboom-serve-images-~a.log" tmp (getpid)))

(define server-pid
  (let ((pid (primitive-fork)))
    (if (zero? pid)
        (let ((out (open-output-file log-file)))
          (dup2 (fileno out) 1)
          (dup2 (fileno out) 2)
          (execlp "guile" "guile" "-L" "bindings/guile/scheme"
                  "bindings/guile/scheme/sonicboom/serve/app.scm"
                  "--backend" "mock" "--port" (number->string port)
                  "--image-model" model-dir
                  "--runtime-dir" (format #f "~a/sonicboom-serve-images-~a"
                                          tmp (getpid))))
        pid)))

(define (stop-server)
  (catch #t (lambda () (kill server-pid SIGKILL) (waitpid server-pid)) (lambda _ #f)))

;;; ---------------------------------------------------------------------------
;;; HTTP + base64

(define* (http method path #:optional body)
  ;; -> (status . json)
  (call-with-values
      (lambda ()
        (http-request (string-append base path) #:method method
                      #:body (and body (string->utf8 (if (string? body)
                                                         body
                                                         (scm->json-string body))))
                      #:headers '((content-type application/json))
                      #:decode-body? #f #:keep-alive? #f))
    (lambda (resp bv)
      (cons (response-code resp)
            (and bv (> (bytevector-length bv) 0)
                 (catch #t (lambda () (json-string->scm (utf8->string bv)))
                   (lambda _ (utf8->string bv))))))))

(define (images body) (http 'POST "/v1/images/generations" body))

;;; --- raw HTTP, for the streamed (SSE) response -----------------------------

;; Guile's HTTP client buffers a whole response; the stream tests read the
;; chunked SSE body as it arrives, over a plain socket.
(define* (send-request method path #:optional body)
  (let* ((bv (and body (string->utf8 (scm->json-string body))))
         (request (string-append
                   method " " path " HTTP/1.1\r\n"
                   "Host: 127.0.0.1\r\n"
                   "Content-Type: application/json\r\n"
                   (if bv (format #f "Content-Length: ~a\r\n"
                                  (bytevector-length bv)) "")
                   "Connection: close\r\n\r\n"))
         (s (socket PF_INET SOCK_STREAM 0)))
    (connect s AF_INET (inet-pton AF_INET "127.0.0.1") port)
    (put-bytevector s (string->utf8 request))
    (when bv (put-bytevector s bv))
    (force-output s)
    s))

(define (strip-cr s)
  (if (and (> (string-length s) 0)
           (char=? (string-ref s (1- (string-length s))) #\return))
      (substring s 0 (1- (string-length s)))
      s))

;; -> (values the status line, headers as an alist of strings)
(define (read-headers port)
  (let ((status (read-line port)))
    (let loop ((line (read-line port)) (acc '()))
      (if (eof-object? line)
          (values status (reverse acc))
          (let ((l (string-trim (strip-cr line))))
            (if (string-null? l)
                (values status (reverse acc))
                (let ((i (string-index l #\:)))
                  (loop (read-line port)
                        (cons (cons (string-downcase (string-trim
                                                      (substring l 0 i)))
                                    (string-trim (substring l (1+ i))))
                              acc)))))))))

;; One chunk of a chunked-transfer body, or #f at the terminating chunk.
(define (read-chunk port)
  (let ((line (read-line port)))
    (if (eof-object? line)
        #f
        (let ((n (string->number (string-trim (strip-cr line)) 16)))
          (if (or (not n) (zero? n))
              #f
              (let ((data (utf8->string (get-bytevector-n port n))))
                (read-line port)                ; the CRLF after the data
                data))))))

;; The data of one SSE event, or #f when the stream ended.
(define (read-event port)
  (let ((chunk (read-chunk port)))
    (and chunk
         (string-prefix? "data: " chunk)
         (let ((end (string-length chunk)))
           (substring chunk 6
                      (if (string-suffix? "\n\n" chunk) (- end 2) end))))))

;; The stream's events, in order, up to and including "[DONE]" (or the
;; connection ending). The last element is "[DONE]" when it arrived.
(define (read-events port)
  (let loop ((acc '()))
    (let ((e (read-event port)))
      (cond ((not e) (reverse acc))
            ((equal? e "[DONE]") (reverse (cons e acc)))
            (else (loop (cons e acc)))))))

;; After a stream ends the producer's cleanup may still be running; poll until
;; the job is gone (404) or the bound elapses.
(define (wait-for-404 path)
  (let loop ((i 0))
    (cond ((= 404 (car (http 'DELETE path))) #t)
          ((> i 100) #f)
          (else (usleep 20000) (loop (1+ i))))))

(define (wait-for-server)
  (let lp ((i 0))
    (cond
     ((catch #t (lambda () (= 200 (car (http 'GET "/health")))) (lambda _ #f)) #t)
     ((> i 240) #f)
     (else (usleep 250000) (lp (1+ i))))))

(define (jref obj . path)
  (fold (lambda (k o)
          (cond ((not o) #f)
                ((integer? k) (and (vector? o) (< k (vector-length o)) (vector-ref o k)))
                (else (match (assoc k o) ((_ . v) v) (#f #f)))))
        obj path))

(define (b64-value c)
  (cond ((char-upper-case? c) (- (char->integer c) 65))
        ((char-lower-case? c) (+ 26 (- (char->integer c) 97)))
        ((char-numeric? c) (+ 52 (- (char->integer c) 48)))
        ((char=? c #\+) 62)
        ((char=? c #\/) 63)
        (else (error "not a base64 character" c))))

(define (base64-decode str)
  (let* ((n (string-length str))
         (pad (cond ((and (>= n 2) (char=? (string-ref str (- n 1)) #\=)
                           (char=? (string-ref str (- n 2)) #\=))
                     2)
                    ((and (>= n 1) (char=? (string-ref str (- n 1)) #\=)) 1)
                    (else 0)))
         (out (make-bytevector (- (* 3 (quotient n 4)) pad) 0)))
    (let loop ((i 0) (o 0))
      (if (>= i n)
          out
          (let* ((ch (lambda (j)
                       (let ((c (string-ref str (+ i j))))
                         (if (char=? c #\=) 0 (b64-value c)))))
                 (v (+ (* (ch 0) 262144) (* (ch 1) 4096) (* (ch 2) 64) (ch 3))))
            (when (< o (bytevector-length out))
              (bytevector-u8-set! out o (quotient v 65536)))
            (when (< (+ o 1) (bytevector-length out))
              (bytevector-u8-set! out (+ o 1) (remainder (quotient v 256) 256)))
            (when (< (+ o 2) (bytevector-length out))
              (bytevector-u8-set! out (+ o 2) (remainder v 256)))
            (loop (+ i 4) (+ o 3)))))))

(define (u32be bv off)
  (+ (* (bytevector-u8-ref bv off) 16777216)
     (* (bytevector-u8-ref bv (+ off 1)) 65536)
     (* (bytevector-u8-ref bv (+ off 2)) 256)
     (bytevector-u8-ref bv (+ off 3))))

(define side 64)
(define (request seed) `(("prompt" . "a red fox") ("size" . ,(format #f "~ax~a" side side))
                         ("steps" . 2) ("seed" . ,seed) ("response_format" . "b64_json")))

(define (run-tests)
  (test-assert "the server starts" (wait-for-server))

  (define single
    (match (images (request 5))
      ((status . body)
       (test-equal "images: 200" 200 status)
       (test-assert "images: created is a timestamp" (integer? (jref body "created")))
       (test-equal "images: one image" 1 (vector-length (jref body "data")))
       (let ((png (base64-decode (jref body "data" 0 "b64_json"))))
         (test-equal "images: the PNG signature"
           #vu8(137 80 78 71 13 10 26 10)
           (let ((sig (make-bytevector 8 0)))
             (bytevector-copy! png 0 sig 0 8)
             sig))
         (test-equal "images: the PNG width" side (u32be png 16))
         (test-equal "images: the PNG height" side (u32be png 20))
         png))))

  (match (images (request 5))
    ((_ . body)
     (test-assert "images: the same request twice is deterministic"
       (equal? single (base64-decode (jref body "data" 0 "b64_json"))))))

  (match (images `(("prompt" . "a red fox") ("size" . "64x64") ("steps" . 2)
                   ("seed" . 5) ("n" . 2)))
    ((_ . body)
     (test-equal "images: n=2 gives two images" 2 (vector-length (jref body "data")))
     (test-assert "images: image 0 of n=2 equals the single image"
       (equal? single (base64-decode (jref body "data" 0 "b64_json"))))))

  (match (images `(("prompt" . "a red fox") ("response_format" . "url")))
    ((status . body)
     (test-equal "images: response_format url is 400" 400 status)
     (test-equal "images: the param is response_format" "response_format"
       (jref body "error" "param"))))

  (match (images `(("size" . "64x64")))
    ((status . body)
     (test-equal "images: a missing prompt is 400" 400 status)
     (test-equal "images: the param is prompt" "prompt" (jref body "error" "param"))))

  (match (images `(("prompt" . "a red fox") ("size" . "65x64")))
    ((status . body)
     (test-equal "images: a size that is not a multiple of 16 is 400" 400 status)
     (test-equal "images: the param is size" "size" (jref body "error" "param"))))

  ;; --- streaming progress and cancellation --------------------------------

  (define (stream-request seed steps)
    `(("prompt" . "a red fox") ("size" . "64x64") ("steps" . ,steps)
      ("seed" . ,seed) ("stream" . #t)))

  ;; Both replies name their job, so a client can cancel it from another
  ;; connection.
  (test-assert "images: the non-stream reply names its job"
    (let ((id (call-with-values
                  (lambda ()
                    (http-request (string-append base "/v1/images/generations")
                                  #:method 'POST
                                  #:body (string->utf8 (scm->json-string (request 5)))
                                  #:headers '((content-type application/json))
                                  #:decode-body? #f #:keep-alive? #f))
                (lambda (resp _)
                  (assoc-ref (response-headers resp) 'sonicboom-job-id)))))
      (and (string? id) (string-prefix? "job-" id))))

  (let* ((port (send-request "POST" "/v1/images/generations"
                             (stream-request 5 2))))
    (call-with-values (lambda () (read-headers port))
      (lambda (status headers)
        (test-assert "images: stream 200" (string-prefix? "HTTP/1.1 200" status))
        (test-equal "images: the stream is an event stream" "text/event-stream"
          (car (string-split (string-trim (assoc-ref headers "content-type"))
                             #\;)))
        (test-assert "images: the stream names its job"
          (let ((id (assoc-ref headers "sonicboom-job-id")))
            (and id (string-prefix? "job-" id))))))
    (let* ((raw (read-events port))
           (events (map json-string->scm
                        (remove (lambda (e) (equal? e "[DONE]")) raw)))
           (final (and (equal? "[DONE]" (car (reverse raw)))
                      (car (reverse events)))))
      (test-assert "images: the stream ends with [DONE]"
        (equal? "[DONE]" (car (reverse raw))))
      (test-equal "images: one event per step, then the images object"
        3 (length events))
      (test-equal "images: the first step" 1 (jref (car events) "step"))
      (test-equal "images: the step total" 2 (jref (car events) "total"))
      (test-equal "images: the step's image index" 0 (jref (car events) "image"))
      (test-equal "images: the last step" 2 (jref (cadr events) "step"))
      (test-assert "images: the streamed image equals the non-stream one"
        (equal? single (base64-decode (jref final "data" 0 "b64_json"))))))

  ;; Cancel a running generation by id; the stream then ends with the
  ;; cancelled status, and the job is gone from the registry.
  (let* ((port (send-request "POST" "/v1/images/generations"
                             (stream-request 9 12)))
         (headers (call-with-values (lambda () (read-headers port))
                    (lambda (_ h) h)))
         (id (assoc-ref headers "sonicboom-job-id")))
    (test-assert "images: a cancel stream starts with a step"
      (jref (json-string->scm (read-event port)) "step"))
    (match (call-with-values
               (lambda ()
                 (http-request (string-append base "/sonicboom/jobs/" id)
                               #:method 'DELETE
                               #:decode-body? #f #:keep-alive? #f))
             (lambda (resp bv)
               (cons (response-code resp) (json-string->scm (utf8->string bv)))))
      ((status . body)
       (test-equal "images: DELETE is 200" 200 status)
       (test-equal "images: DELETE names the job" id (jref body "id"))
       (test-equal "images: DELETE says cancelled" "cancelled"
         (jref body "status"))))
    (let* ((raw (read-events port))
           (events (map json-string->scm
                        (remove (lambda (e) (equal? e "[DONE]")) raw)))
           (last (car (reverse events))))
      (test-equal "images: a cancelled stream ends cancelled" "cancelled"
        (jref last "status")))
    (test-assert "images: the cancelled job is gone"
      (wait-for-404 (string-append "/sonicboom/jobs/" id)))
    (test-assert "images: a generation after a cancel still works"
      (match (images (request 5))
        ((status . body)
         (and (= 200 status)
              (equal? single
                      (base64-decode (jref body "data" 0 "b64_json"))))))))

  ;; A client that goes away cancels its generation, so the queue and the
  ;; device are free for the next request. The server notices at its next
  ;; event write, so after the wait below a 20-step job (about 70 ms a step)
  ;; is only gone if the disconnect stopped it; a still-running job would
  ;; answer the DELETE with 200.
  (let* ((port (send-request "POST" "/v1/images/generations"
                             (stream-request 13 20)))
         (headers (call-with-values (lambda () (read-headers port))
                    (lambda (_ h) h)))
         (id (assoc-ref headers "sonicboom-job-id")))
    (read-event port)
    (close port)
    (usleep 500000)
    (test-assert "images: a client going away cancels its job"
      (= 404 (car (http 'DELETE (string-append "/sonicboom/jobs/" id)))))
    (test-assert "images: a generation after a disconnect still works"
      (match (images (request 5))
        ((status . body)
         (and (= 200 status)
              (equal? single
                      (base64-decode (jref body "data" 0 "b64_json")))))))))

(test-begin "sonicboom-serve-images")
(dynamic-wind (lambda () #t) run-tests stop-server)
(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-serve-images")
(exit (if (zero? failures) 0 1))
