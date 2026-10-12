# GNU Artanis: streaming support, and the changes it would need

Findings from building the OpenAI-compatible service
(`bindings/guile/scheme/sonicboom/serve/`, see `design/serve-openai.md`) on
GNU Artanis. Artanis itself was **not** modified; the service works around
each point inside `serve/`. This note lists what Artanis would need so the
workarounds can go.

Source examined: `https://gitlab.com/hardenedlinux/artanis`, commit `9100e1d`
("Add text type to postgresql"). `github.com/HardenedLinux/artanis` was not
reachable from the build environment (no such repository visible), so the
GitLab tree was used. Built with Guile 3.0.9, guile-json 4.7.3, guile-redis
(git), guile-curl (git), libargon2.

## 1. Streaming HTTP responses: not supported

**Finding.** A route handler's body ends up in `http-write`
(`artanis/server/http.scm`), which accepts a bytevector or a `file-sender`
and nothing else. `handler-render` (`artanis/page.scm`) always adds
`Content-Length` (`gen-content-length` in `artanis/utils.scm`): from the
bytevector, or from `file-sender-size`. There is no chunked transfer encoding
and no way to leave the length out, so a body whose size is unknown in advance
(Server-Sent Events) cannot be written through the public API. Adding a
`Transfer-Encoding` header by hand would send both headers, which RFC 9112
forbids.

**Workaround in `serve/sse.scm`.** The connection is switched, for one
response, to a private Ragnarok protocol, the hook Artanis uses for WebSocket:

1. `sse-install!` registers the protocol `sonicboom-sse` with
   `protocol-add!`: `open`, `read` and `close` of HTTP, its own `write`;
2. the handler calls `register-proto!` on `(current-client)` with the producer
   as the state, and returns an empty response;
3. Ragnarok then calls the protocol's `write` instead of `http-write`. It
   unregisters the protocol, writes its own status line and headers
   (`text/event-stream`, `Transfer-Encoding: chunked`, no `Content-Length`),
   and runs the producer through `call-with-runner` with no timeout, as
   `http-write` does for a file-sender. Each event is one chunk, flushed to the
   socket at once (`force-output` of the chunked port and of the socket port:
   the socket port is block-buffered with `server.bufsize`, so flushing only
   the chunked port batches the events);
4. the terminating chunk is written; the connection stays alive.

A write to a peer that went away fails with `EPIPE`/`ECONNRESET`, or with a
wrong-type error once Ragnarok has closed the port, and `send` returns `#f`;
the generation is cancelled at the next token.

This depends on internals (`register-proto!`, `make-ragnarok-protocol`,
`proto-conn-state`, `client-sockport`, `new-http-protocol`), and on
`protocol-add!` being callable before `run` with a protocol built from a fresh
`(new-http-protocol)`, because `lookup-protocol 'http` is `#f` until
`init-server-core` runs.

**Proposed change.** A streamed body type next to `file-sender`:

```scheme
;; artanis/utils.scm
(define-record-type stream-sender (fields thunk))   ; thunk: (lambda (send) ...)

;; artanis/page.scm, handler-render: no Content-Length for a stream-sender,
;; and Transfer-Encoding: chunked instead.

;; artanis/server/http.scm, http-write:
((stream-sender? body)
 (call-with-runner
  (lambda ()
    (let ((chunked (make-chunked-output-port port #:keep-alive? #t)))
      ((stream-sender-thunk body)
       (lambda (bv)            ; send: #t, or #f when the peer is gone
         (catch #t
           (lambda () (put-bytevector chunked bv) (force-output chunked)
                      (force-output port) #t)
           (lambda _ #f))))
      (close-port chunked)))
  #:timeout 0))
```

plus a public helper, e.g. `(response-emit-stream producer #:headers ...)`,
and an SSE layer on top: `(sse-emit rc producer)` where the producer's `send`
takes the event data (and optionally `#:event`, `#:id`, `#:retry`), writes the
`data:` lines and the blank line, and returns `#f` once the client is gone.
The HEAD case writes the headers only.

## 2. Single-file apps outside an `art create` project

**Finding.** `(init-server)` fails in a plain script, like
`examples/test.scm`, run outside a project directory:
`mkdir "#f/sys/i18n/json"`. `init-i18n` builds its path from
`current-toplevel`, which is `#f` when no `ENTRY` file is found, and `mkdir`
is not recursive.

**Workaround in `serve/app.scm`.** Create `<runtime-dir>/sys/i18n` and set
`(%current-toplevel runtime-dir)` before `init-server` (`--runtime-dir`,
default `$TMPDIR/sonicboom-serve-<pid>`).

**Proposed change.** Skip i18n initialisation (or use `current-tmp`) when
there is no toplevel, and create the directory with its parents.

## 3. WebSocket: replies streamed from a blocking handler

**Finding.** As documented in `artanis/server/websocket.scm`, the task of a
connection is the only writer of its socket, and it writes the messages
queued by `ws-send` only between two messages or while it waits for the peer.
A handler that calls `call-with-runner` inside `on-message` (as
`examples/ENTRY.websocket`'s `/slow` does) keeps the task busy until the
runner returns, so every `ws-send` made during a long generation is written
at the end in one burst, and a peer that disconnects meanwhile is not noticed
(`ws-send` keeps returning `#t`).

**Workaround in `serve/app.scm`.** `on-message` admits the request and
returns at once; the reply is generated in a Guile thread, which calls
`ws-send` per chunk. The connection task, waiting for the peer, writes each
chunk as `ws-send` wakes it, and sees the close, after which `ws-send` returns
`'closed` and the generation is cancelled. The thread runs in the dynamic
state the app had at load time (like the runners' `*clean-dynamic-state*`),
not in the Ragnarok task's one whose port waiters would suspend the task.

**Proposed change.** One of:

- a non-suspending runner submission, e.g. `(call-with-runner thunk
  #:wait? #f)` or `spawn-runner`, so long work started from `on-message` uses
  the runner pool (bounded) instead of ad-hoc threads;
- or let the waiting loop of `call-with-runner` flush the connection's
  outbound queue and watch for the peer's close while the runner works.

## 4. Smaller points

- `:mime` cannot set a status: an error with a JSON body needs
  `response-emit` with an explicit `content-type`. A `#:status` keyword on
  `:mime` would cover it.
- `server.runners` (default 2) bounds the concurrently streaming responses,
  since each stream holds a runner for its whole duration. The service sets it
  to the queue limit + 2.
- Not Artanis: Guile's `make-chunked-input-port` fills its whole buffer before
  returning, so a client reading SSE through `response-body-port` blocks until
  later events arrive. The tests decode chunks by hand.
