;;; SonicBoom — C ABI (Guile) prefill/decode measurement for v1 acceptance.
;;;
;;; Measures the CUDA fast-path numbers the C ABI exposes to Guile, for the
;;; item-2 acceptance gate (Guile tg64 / pp4100 vs the C++ HEAD measurements):
;;;   tg64   — 64 greedy decode steps from seed 2 (steady, after a warm-up),
;;;            reported as tokens/sec (prefill of the seed token reported
;;;            separately so the decode number matches the C++ decode-only ref).
;;;   pp4100 — device-resident chunked prefill of a 4100-token prompt,
;;;            reported as milliseconds.
;;;
;;; Usage:
;;;   LD_LIBRARY_PATH=build/core SONICBOOM_GEMMA_GGUF=models/gemma-4-E4B-it-Q3_K_M.gguf \
;;;     guile --no-auto-compile -L bindings/guile/scheme tools/baseline/measure_capi.scm

(use-modules (system foreign)
             (rnrs bytevectors)
             (srfi srfi-1)
             (ice-9 format)
             (sonicboom ffi)
             (sonicboom inference))

;; sb_llm_sampling as a native-layout bytevector (mirror inference.scm):
;;   struct_size u64@0, temperature f32@8, top_p f32@12, top_k u32@16,
;;   seed u64@24, max_tokens u32@32  → 40 bytes. Greedy decode.
(define (make-sampling-bv max-tokens)
  (let ((bv (make-bytevector 40 0)))
    (bytevector-u64-native-set! bv 0 40)
    (bytevector-ieee-single-native-set! bv 8 0.0)
    (bytevector-ieee-single-native-set! bv 12 1.0)
    (bytevector-u32-native-set! bv 16 0)
    (bytevector-u64-native-set! bv 24 0)
    (bytevector-u32-native-set! bv 32 max-tokens)
    bv))

(define (sb-check! st err-slot what)
  (unless (sb-status-ok? st)
    (throw 'sb-measure
           (string-append what ": " (capture-error err-slot)))))

(define (now-t) (get-internal-real-time))
(define (secs a b)
  (/ (- b a) 1.0 internal-time-units-per-second))

;; Decode up to MAX-TOKENS tokens from a pre-filled session, returning the count.
(define (decode! session max-tokens)
  (let* ((count 0)
         (cb (lambda (user token) (set! count (+ count 1)) 0))
         (cb-ptr (procedure->pointer int cb (list '* int32)))
         (smp-bv (make-sampling-bv max-tokens))
         (n-slot (make-bytevector 4 0))
         (err-slot (make-slot))
         (st ((proc 'sb_llm_decode) session (bytevector->pointer smp-bv)
              cb-ptr %null-pointer (bytevector->pointer n-slot)
              (slot-ptr err-slot))))
    (sb-check! st err-slot "decode")
    count))

(define (main)
  (let ((path (getenv "SONICBOOM_GEMMA_GGUF")))
    (unless path
      (error "set SONICBOOM_GEMMA_GGUF to the Gemma 4 GGUF path"))
    (let ((model (load-model path)))

      ;; ---- tg64: 64 decode steps from seed 2 --------------------------------
      (let ((seed '(2)))
        ;; Warm-up: builds the decode runtime (graph/plan/capture) and the
        ;; persistent cache; its timing is discarded.
        (let ((s (make-session model 512)))
          (prefill! s seed)
          (decode! s 64)
          (free-session! s))

        ;; Timed run: prefill (un-timed for the tok/s figure) then 64-token
        ;; decode. Decode tok/s is 64 / decode-seconds.
        (let* ((s (make-session model 512))
               (tp0 (now-t)))
          (prefill! s seed)
          (let ((tp1 (now-t)))
            (let ((td0 (now-t)))
              (decode! s 64)
              (let ((td1 (now-t)))
                (format #t "tg64 prefill(seed): ~,1f ms | decode64: ~,1f ms = ~,2f tok/s~%"
                        (* 1000.0 (secs tp0 tp1))
                        (* 1000.0 (secs td0 td1))
                        (/ 64.0 (secs td0 td1))))))
          (free-session! s)))

      ;; ---- pp4100: prefill a 4100-token prompt ------------------------------
      (let* ((nctx 8192)
             (prompt (list-tabulate 4100 (lambda (i) (+ 2 (modulo i 500))))))
        ;; Warm-up (builds the prefill segment graphs + device buffers).
        (let ((s (make-session model nctx)))
          (prefill! s prompt)
          (free-session! s))

        ;; Fresh session for a clean timed prefill.
        (let* ((s (make-session model nctx))
               (t0 (now-t)))
          (prefill! s prompt)
          (let ((t1 (now-t)))
            (format #t "pp4100 prefill: ~,1f ms~%" (* 1000.0 (secs t0 t1))))
          (free-session! s)))

      (free-model! model))))

(main)
