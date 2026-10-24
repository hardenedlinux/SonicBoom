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
;;;      is not a multiple of 16 are 400s with the right param.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (ice-9 match)
             (rnrs bytevectors)
             (web client)
             (web response)
             (json))

;;; ---------------------------------------------------------------------------
;;; Preconditions

(define root (or (getenv "SONICBOOM_SRC_DIR") "."))
(define model-dir
  (or (getenv "SONICBOOM_MOSS_IMAGE_DIR")
      (string-append root "/models/MOSS-Image-SFT")))

(define (artanis-available?)
  (catch #t (lambda () (resolve-interface '(artanis artanis)) #t) (lambda _ #f)))

(unless (and (artanis-available?)
             (file-exists? (string-append model-dir "/transformer"))
             (file-exists? "/dev/nvidia0"))
  (format (current-error-port)
          "SKIP: needs GNU Artanis, ~a/transformer and a CUDA device~%"
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
     (test-equal "images: the param is size" "size" (jref body "error" "param")))))

(test-begin "sonicboom-serve-images")
(dynamic-wind (lambda () #t) run-tests stop-server)
(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-serve-images")
(exit (if (zero? failures) 0 1))
