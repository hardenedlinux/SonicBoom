;;; SonicBoom — Guile explain binding test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; (sonicboom explain) exposes the planner's decision registry through the
;;; C ABI's two views. Checks:
;;;
;;;   1. explain-table returns the human table tools/sb-explain prints, with a
;;;      decision and its reason, and no embedded NUL;
;;;   2. explain-json returns the same registry as a parsed Scheme value:
;;;      version, device, and a decision list carrying names and costs.
;;;
;;; Needs no model and no GPU: the static knobs are seeded without one. Skips
;;; cleanly if libsonicboom.so cannot be loaded.

(use-modules (srfi srfi-1)
             (srfi srfi-13)
             (srfi srfi-64)
             (sonicboom ffi)
             (sonicboom explain))

(test-begin "sonicboom-explain")

(if (not (libsonicboom-available?))
    (test-skip 10)
    (begin
      (let ((table (explain-table)))
        (test-assert "the table is a non-empty string"
          (and (string? table) (> (string-length table) 0)))
        (test-assert "the table names a decision and its reason"
          (and (string-contains table "mmq.enabled device = ")
               (string-contains table "reason=\"")))
        (test-equal "the table ends with a newline"
          #\newline (string-ref table (- (string-length table) 1)))
        (test-assert "the string ends at the NUL terminator"
          (not (string-index table #\nul))))

      (let* ((doc (explain-json))
             (decisions (assq-ref doc 'decisions)))
        (test-equal "the JSON top-level version" 1 (assq-ref doc 'version))
        (test-assert "the JSON names the device"
          (string? (assq-ref doc 'device)))
        (test-assert "the JSON has decisions"
          (and (list? decisions) (pair? decisions)))
        (test-assert "a decision carries its declared costs"
          (let ((d (car decisions)))
            (and (assq-ref d 'name) (assq-ref d 'value)
                 (string? (assq-ref d 'reason))
                 (integer? (assq-ref d 'vram_bytes))
                 (integer? (assq-ref d 'tuning_ms)))))
        (test-assert "every decision has a name and a reason"
          (every (lambda (d)
                   (and (assq-ref d 'name) (string? (assq-ref d 'reason))))
                 decisions))
        (test-assert "the JSON lists the mmq.enabled decision"
          (any (lambda (d) (equal? "mmq.enabled" (assq-ref d 'name)))
               decisions)))))

(test-end "sonicboom-explain")
