;;; SonicBoom — inference service: one model resident on the device
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The serve layer's model catalog and the switch between its models. Khala's
;;; pipeline is linear: one step finishes and the next starts, often on another
;;; model (text, then an image). The device holds one model at a time, so a
;;; request for another model makes the device unload the resident one and
;;; load its own; the switch happens only between jobs, with the device idle
;;; (the engine's queue gives the job its turn first).
;;;
;;; An entry is one served model:
;;;   chat   loaded through the backend protocol (load-model), with a session
;;;          pool; unloaded by freeing its idle sessions, then the model.
;;;   image  loaded through the image loader; the handle is the model.
;;;
;;; The manager does not load anything by itself at construction; the app
;;; loads its first model before accepting requests, exactly as a single-model
;;; server always did. The rest load on demand.
;;;
;;; Times are milliseconds: last_load_ms is the backend load plus its session
;;; pool, last_switch_ms the unload-and-load of the switch that made the model
;;; resident (the first load counts as a switch from no model). vram_mib is
;;; the device free-memory drop measured around the load (0 when it cannot be
;;; measured); it is 0 once the model is unloaded.
;;;
;;;   residency-ensure!   make an entry resident (switch if another one is)
;;;   residency-load!     the same, for the load endpoint
;;;   residency-unload!   free the entry's device memory
;;;   residency-info      one entry as an alist, for /sonicboom/models

(define-module (sonicboom serve residency)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (ice-9 rdelim)
  #:use-module (ice-9 popen)
  #:use-module (sonicboom serve backend)
  #:use-module (sonicboom serve engine)
  #:use-module (sonicboom serve sessions)
  #:export (make-residency-entry residency-entry? residency-entry-id
            make-residency residency? residency-entries residency-entry
            residency-ids residency-known? residency-kind
            residency-loaded? residency-active residency-model residency-family
            residency-ensure! residency-load! residency-unload! residency-info))

;; One served model: its identity, how to load it, and its residency state.
;;
;; kind is chat or image; config is what the loader takes (the backend's load
;; config alist for chat, the model directory for image); arch and family are
;; learned when the model loads and kept afterwards, so the catalog still
;; describes an unloaded model.
;;
;; PyTorch: none
(define-record-type <residency-entry>
  (make-residency-entry* id kind config arch family
                         model pool loaded? load-ms switch-ms vram-mib)
  residency-entry?
  (id residency-entry-id)
  (kind residency-entry-kind)
  (config residency-entry-config)
  (arch residency-entry-arch set-residency-entry-arch!)
  (family residency-entry-family set-residency-entry-family!)
  (model residency-entry-model set-residency-entry-model!)
  (pool residency-entry-pool set-residency-entry-pool!)
  (loaded? residency-entry-loaded? set-residency-entry-loaded?!)
  (load-ms residency-entry-load-ms set-residency-entry-load-ms!)
  (switch-ms residency-entry-switch-ms set-residency-entry-switch-ms!)
  (vram-mib residency-entry-vram-mib set-residency-entry-vram-mib!))

(define* (make-residency-entry id kind config #:key (arch #f) (family #f))
  "Create a catalog entry for one served model.

Args:
  id: the model id string.
  kind: chat or image.
  config: the backend's load config alist for chat, the model directory for
    an image model.
  arch: keyword; the architecture name when it is already known (default #f;
    a loaded model's own architecture is used then).
  family: keyword; the chat-template family when it is already known
    (default #f).
Returns: a residency entry.
Example:
  (use-modules (sonicboom serve residency))
  (residency-entry-id (make-residency-entry \"m\" 'chat '((path . \"m.gguf\"))))
  => \"m\"
PyTorch: none"
  (make-residency-entry* id kind config arch family #f #f #f #f #f 0))

;; The catalog and the one resident model.
;;
;; entries is the list of <residency-entry> in serving order. The manager uses
;; the backend to load and unload chat models, creates each model's session
;; pool, and keeps the engine's slots in step. image-loader / image-unloader /
;; image-arch are the image binding's procedures, passed in so this module
;; stays free of the C ABI; trim returns freed device buffers to the driver
;; (the Guile binding's device-trim!). One job at a time runs through the
;; engine's queue, so the manager needs no lock of its own: its state is only
;; changed between jobs, and readers see one field at a time.
;;
;; PyTorch: none
(define-record-type <residency>
  (make-residency* engine backend entries max-sessions token-budget
                   image-loader image-unloader image-arch trim)
  residency?
  (engine residency-engine)
  (backend residency-backend)
  (entries residency-entries)
  (max-sessions residency-max-sessions)
  (token-budget residency-token-budget)
  (image-loader residency-image-loader)
  (image-unloader residency-image-unloader)
  (image-arch residency-image-arch)
  (trim residency-trim))

(define* (make-residency engine entries
                         #:key backend (max-sessions 4) (token-budget 32768)
                         image-loader image-unloader
                         (image-arch (lambda (handle) "moss-image"))
                         (trim (lambda () #f)))
  "Create the residency manager of ENGINE over ENTRIES.

Nothing is loaded here; residency-load! (or the first residency-ensure!)
loads a model, and every switch unloads the resident one first.

Args:
  engine: the engine the entries are served through; their ids must be among
    its model ids.
  entries: the list of residency entries, in serving order.
  backend: keyword; the backend that loads chat models.
  max-sessions: keyword; the session pool size per chat model (default 4).
  token-budget: keyword; the pool's token budget per chat model (default
    32768).
  image-loader: keyword; (config) -> image model handle.
  image-unloader: keyword; (handle) -> unspecified.
  image-arch: keyword; (handle) -> architecture string (default \"moss-image\").
  trim: keyword; a procedure that returns freed device buffers to the driver
    (default: do nothing).
Returns: a residency manager.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve residency))
  (let ((engine (make-engine '() #:ids '(\"m\"))))
    (residency?
     (make-residency engine (list (make-residency-entry \"m\" 'chat '((id . \"m\"))))
                     #:backend (make-mock-backend))))
  => #t
PyTorch: none"
  (make-residency* engine backend entries max-sessions token-budget
                   image-loader image-unloader image-arch trim))

(define (residency-entry resid id)
  "The entry of model ID, or #f.

Args:
  resid: a residency manager.
  id: a model id string.
Returns: a residency entry or #f.
PyTorch: none"
  (find (lambda (e) (equal? id (residency-entry-id e)))
        (residency-entries resid)))

(define (residency-ids resid)
  "The ids of every catalog entry, in serving order.

Args:
  resid: a residency manager.
Returns: a list of strings.
PyTorch: none"
  (map residency-entry-id (residency-entries resid)))

(define (residency-known? resid id)
  "True when ID names a catalog entry.

Args:
  resid: a residency manager.
  id: a model id string.
Returns: a boolean.
PyTorch: none"
  (and (residency-entry resid id) #t))

(define (residency-kind resid id)
  "The kind (chat or image) of model ID, or #f.

Args:
  resid: a residency manager.
  id: a model id string.
Returns: chat, image or #f.
PyTorch: none"
  (match (residency-entry resid id) (#f #f) (e (residency-entry-kind e))))

(define (residency-loaded? resid id)
  "True when model ID is resident on the device.

Args:
  resid: a residency manager.
  id: a model id string.
Returns: a boolean.
PyTorch: none"
  (match (residency-entry resid id) (#f #f) (e (residency-entry-loaded? e))))

;; The resident entry, or #f when nothing is loaded.
(define (residency-active resid)
  "The resident entry, or #f when no model is on the device.

Args:
  resid: a residency manager.
Returns: a residency entry or #f.
PyTorch: none"
  (find residency-entry-loaded? (residency-entries resid)))

(define (residency-family resid id)
  "The chat-template family of model ID, as the last load learned it.

It is #f for a model that has not been loaded since the server started, and
for an image model.

Args:
  resid: a residency manager.
  id: a model id string.
Returns: a family symbol or #f.
PyTorch: none"
  (match (residency-entry resid id) (#f #f) (e (residency-entry-family e))))

(define (residency-model resid id)
  "The loaded model handle of model ID (backend handle or image handle).

Args:
  resid: a residency manager.
  id: a model id string.
Returns: the handle, or #f when the model is not loaded.
PyTorch: none"
  (match (residency-entry resid id) (#f #f) (e (residency-entry-model e))))

;;; ---------------------------------------------------------------------------
;;; Loading and switching

(define (now-ms)
  (quotient (* 1000 (get-internal-real-time)) internal-time-units-per-second))

;; Free device memory in MiB, or #f when it cannot be read (no NVIDIA driver,
;; no nvidia-smi). It is asked once per load and once per unload, not per
;; request.
(define (free-vram-mib)
  (catch #t
    (lambda ()
      (let* ((port (open-input-pipe
                    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"))
             (line (read-line port)))
        (close-pipe port)
        (and (not (eof-object? line)) (string->number (string-trim-both line)))))
    (lambda _ #f)))

;; What the load took, from the free-memory readings around it.
(define (vram-delta before after)
  (if (and before after) (max 0 (- before after)) 0))

(define (load-entry! resid e)
  (let* ((id (residency-entry-id e))
         (free0 (free-vram-mib))
         (t0 (now-ms)))
    (case (residency-entry-kind e)
      ((chat)
       (let* ((model (load-model (residency-backend resid)
                                 (residency-entry-config e)))
              (pool (make-session-pool model
                                       #:max-sessions (residency-max-sessions resid)
                                       #:token-budget (residency-token-budget resid))))
         (engine-model-install! (residency-engine resid) id model pool)
         (set-residency-entry-model! e model)
         (set-residency-entry-pool! e pool)
         (set-residency-entry-family! e (model-family model))
         (set-residency-entry-arch! e (or (model-arch model)
                                          (residency-entry-arch e)))))
      ((image)
       (let ((handle ((residency-image-loader resid)
                      (residency-entry-config e))))
         (set-residency-entry-model! e handle)
         (set-residency-entry-arch! e (or ((residency-image-arch resid) handle)
                                          (residency-entry-arch e)))))
      (else (error "unknown model kind" (residency-entry-kind e))))
    (set-residency-entry-load-ms! e (- (now-ms) t0))
    (set-residency-entry-vram-mib! e (vram-delta free0 (free-vram-mib)))
    (set-residency-entry-loaded?! e #t)
    e))

(define (unload-entry! resid e)
  (case (residency-entry-kind e)
    ((chat)
     ;; The manager switches only with the device idle, so every pooled session
     ;; is idle and the pool frees them; only then do the weights go.
     (let ((pool (residency-entry-pool e))
           (model (residency-entry-model e)))
       (when pool (pool-clear! pool))
       (when model (unload-model model))
       (engine-model-clear! (residency-engine resid) (residency-entry-id e))))
    ((image)
     (let ((handle (residency-entry-model e)))
       (when handle ((residency-image-unloader resid) handle)))))
  (set-residency-entry-loaded?! e #f)
  (set-residency-entry-model! e #f)
  (set-residency-entry-pool! e #f)
  (set-residency-entry-vram-mib! e 0)
  ((residency-trim resid))
  e)

;; Make e the resident model, unloading the one there is. The timing covers
;; the whole switch.
(define (switch-to! resid e)
  (let ((active (residency-active resid)))
    (if (eq? active e)
        e
        (let ((t0 (now-ms)))
          (when active (unload-entry! resid active))
          (load-entry! resid e)
          (set-residency-entry-switch-ms! e (- (now-ms) t0))
          e))))

(define (residency-ensure! resid id)
  "Make model ID the one on the device, and return its entry.

Another resident model is unloaded first; the call is expected to run between
jobs, with the device idle. Raises (model-not-found id) for an unknown id.

Args:
  resid: a residency manager.
  id: a model id string.
Returns: a residency entry.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve residency))
  (let* ((engine (make-engine '() #:ids '(\"m\")))
         (resid (make-residency
                 engine (list (make-residency-entry \"m\" 'chat '((id . \"m\"))))
                 #:backend (make-mock-backend))))
    (residency-entry-id (residency-ensure! resid \"m\")))
  => \"m\"
PyTorch: none"
  (match (residency-entry resid id)
    (#f (throw 'model-not-found id))
    (e (switch-to! resid e))))

(define (residency-load! resid id)
  "Load model ID (the load endpoint's operation).

This is residency-ensure!: a resident model is unloaded first.

Args:
  resid: a residency manager.
  id: a model id string.
Returns: a residency entry.
PyTorch: none"
  (residency-ensure! resid id))

(define (residency-unload! resid id)
  "Unload model ID, freeing its device memory.

Unloading a model that is not resident does nothing; an unknown id raises
(model-not-found id).

Args:
  resid: a residency manager.
  id: a model id string.
Returns: a residency entry.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve residency))
  (let* ((engine (make-engine '() #:ids '(\"m\")))
         (resid (make-residency
                 engine (list (make-residency-entry \"m\" 'chat '((id . \"m\"))))
                 #:backend (make-mock-backend))))
    (residency-loaded? resid
                       (residency-entry-id
                        (residency-unload! resid \"m\"))))
  => #f
PyTorch: none"
  (match (residency-entry resid id)
    (#f (throw 'model-not-found id))
    (e (if (residency-entry-loaded? e) (unload-entry! resid e) e))))

;; ---------------------------------------------------------------------------
;; Introspection

(define (null-or v) (if v v 'null))

(define (residency-info resid id)
  "The state of model ID as an alist with symbol keys.

It holds id, kind, arch, loaded, vram_mib, last_load_ms and last_switch_ms.
arch is null until a load learns it; the times are 0 until the model has been
loaded (a model that never loaded has never switched either).

Args:
  resid: a residency manager.
  id: a model id string.
Returns: an alist, or #f for an unknown id.
Example:
  (use-modules (sonicboom serve backend) (sonicboom serve mock-backend)
               (sonicboom serve engine) (sonicboom serve residency))
  (let* ((engine (make-engine '() #:ids '(\"m\")))
         (resid (make-residency
                 engine (list (make-residency-entry \"m\" 'chat '((id . \"m\"))))
                 #:backend (make-mock-backend))))
    (assq-ref (residency-info resid \"m\") 'loaded))
  => #f
PyTorch: none"
  (match (residency-entry resid id)
    (#f #f)
    (e `((id . ,(residency-entry-id e))
         (kind . ,(symbol->string (residency-entry-kind e)))
         (arch . ,(null-or (residency-entry-arch e)))
         (loaded . ,(residency-entry-loaded? e))
         (vram_mib . ,(residency-entry-vram-mib e))
         (last_load_ms . ,(null-or (residency-entry-load-ms e)))
         (last_switch_ms . ,(null-or (residency-entry-switch-ms e)))))))
