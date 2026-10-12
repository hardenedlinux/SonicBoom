;;; SonicBoom — tracing prototype: run a test file, validating every graph it executes
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Not a test on its own; used by test-trace-validate.scm:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/validate-graphs.scm <test-file> <report-file>
;;;
;;; Loads <test-file> with current-executor wrapping the reference interpreter:
;;; every distinct graph text handed to the executor is first checked by
;;; (sonicboom trace validate); one line per graph is appended to
;;; <report-file>: (ok <bytes>) or (fail <errors>). An invalid graph also
;;; raises, so the loaded test fails. The process exits with the loaded test's
;;; own status.

(use-modules (sonicboom trace tracer)
             (sonicboom trace interp)
             (sonicboom trace validate))

(define test-file (cadr (command-line)))
(define report (open-output-file (caddr (command-line))))

(define seen '())          ; graph texts already validated (by identity)

(define (validating-executor text inputs)
  (unless (memq text seen)
    (set! seen (cons text seen))
    (let ((errors (validate-sexpr-text text)))
      (write (if (null? errors) (list 'ok (string-length text)) (cons 'fail errors)) report)
      (newline report)
      (force-output report)
      (unless (null? errors)
        (error "graph does not validate against the v0.2 draft" errors))))
  (interpret-sexpr-text text inputs))

(parameterize ((current-executor validating-executor))
  (load test-file))
