;;; SonicBoom — inference service: a minimal SSE writer for Artanis
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Artanis writes a response body as a bytevector or a file-sender, always
;;; with a Content-Length: there is no chunked or streaming body. This module
;;; streams Server-Sent Events without modifying Artanis, through the hook
;;; Artanis uses for WebSocket: a connection can be switched to another
;;; Ragnarok protocol, whose write procedure then writes the response.
;;;
;;;   1. sse-install! adds the protocol `sonicboom-sse': open, read and close
;;;      are those of HTTP; write is sse-write below.
;;;   2. A route handler calls sse-response with a producer. It switches the
;;;      connection to `sonicboom-sse' (with the producer as the state) and
;;;      returns an empty response for Artanis' handler-render.
;;;   3. Ragnarok calls sse-write instead of http-write. It switches the
;;;      connection back to HTTP, writes its own status line and headers
;;;      (text/event-stream, Transfer-Encoding: chunked, no Content-Length),
;;;      and runs the producer in a runner thread (call-with-runner, no
;;;      timeout), like Artanis does for a file-sender. Each event is one
;;;      chunk, flushed to the socket at once.
;;;   4. When the producer returns, the terminating chunk is written; the
;;;      connection then stays alive for the next request, as HTTP/1.1 does.
;;;
;;; The producer gets a `send' procedure: (send data) writes the event
;;; "data: <data>\n\n" and returns #t, or #f once the client is gone (the
;;; write failed with EPIPE / ECONNRESET), so the producer can stop. The
;;; producer is called even when the headers could not be written (then its
;;; first send returns #f), so it can always release what it holds.
;;;
;;; This relies on Artanis internals (register-proto!, the Ragnarok protocol
;;; record), not on a public API; design/artanis-streaming-notes.md describes
;;; what Artanis would need to provide instead.

(define-module (sonicboom serve sse)
  #:use-module (rnrs bytevectors)
  #:use-module (ice-9 binary-ports)
  #:use-module (web http)
  #:use-module (web response)
  #:use-module (artanis artanis)
  #:use-module ((artanis env) #:select (protocol-add! lookup-protocol))
  #:use-module ((artanis runner) #:select (call-with-runner))
  #:use-module ((artanis server http) #:select (new-http-protocol))
  #:use-module ((artanis server server-context)
                #:select (make-ragnarok-protocol ragnarok-protocol-open
                          ragnarok-protocol-read ragnarok-protocol-close
                          register-proto! unregister-proto! proto-conn-state
                          client-sockport current-client))
  #:export (sse-install! sse-response))

(define protocol-name 'sonicboom-sse)

(define sse-headers
  '((content-type text/event-stream (charset . "utf-8"))
    (cache-control no-cache)
    (transfer-encoding (chunked))))

;; One event: every line of data gets its own "data:" field.
(define (event-bytes data)
  (string->utf8
   (string-append
    (string-concatenate
     (map (lambda (line) (string-append "data: " line "\n"))
          (string-split data #\newline)))
    "\n")))

(define (run-producer producer port alive?)
  (let ((chunked (make-chunked-output-port port #:keep-alive? #t)))
    (define (send data)
      (and alive?
           ;; EPIPE / ECONNRESET, or the port already closed by Artanis
           ;; after it saw the peer go away.
           (catch #t
             (lambda ()
               (put-bytevector chunked (event-bytes data))
               (force-output chunked)
               (force-output port)
               #t)
             (lambda _ (set! alive? #f) #f))))
    (catch #t
      (lambda () (producer send))
      (lambda (key . args)
        (format (current-error-port) "sse: producer failed: ~a ~s~%" key args)))
    (when alive?
      (catch #t
        (lambda ()
          (close-port chunked)           ; the terminating 0-length chunk
          (force-output port))
        (lambda _ #f)))))

(define (sse-write server client response body method-is-head?)
  (let ((producer (proto-conn-state client))
        (port (client-sockport client)))
    ;; Back to HTTP first: the next request on this connection is plain HTTP.
    (unregister-proto! client)
    (let ((alive? (catch #t
                    (lambda ()
                      (write-response (build-response #:code 200 #:headers sse-headers)
                                      port)
                      (force-output port)
                      (not method-is-head?))
                    (lambda _ #f))))
      ;; The producer always runs, so it can release what it holds; when the
      ;; headers could not be written its first send already returns #f.
      (call-with-runner (lambda () (run-producer producer port alive?)) #:timeout 0))))

(define (sse-install!)
  (unless (lookup-protocol protocol-name)
    (let ((http (new-http-protocol)))
      (protocol-add! protocol-name
                     (make-ragnarok-protocol protocol-name
                                             (ragnarok-protocol-open http)
                                             (ragnarok-protocol-read http)
                                             sse-write
                                             (ragnarok-protocol-close http))))))

;; In a route handler: stream the events of producer. The handler returns
;; what this returns.
(define (sse-response producer)
  (register-proto! (current-client) protocol-name producer)
  (response-emit "" #:headers '((content-type . (text/event-stream)))))
