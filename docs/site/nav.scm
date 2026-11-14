;;; The structure of the documentation site.
;;;
;;; A list of sections; each section is (TITLE PAGE ...), and each page is
;;; (SLUG TITLE). SLUG names the page: a page in docs/site/pages/<SLUG>.md, an
;;; article docs/<SLUG>.md, or the generated page "api/index". The title is what
;;; the navigation shows.
;;;
;;; build.scm appends, at the end of Concepts, every docs/*.md that is not
;;; listed here.
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>

(define site-nav
  '(("SonicBoom"
     ("index" "Home"))
    ("Getting started"
     ("getting-started" "Getting started")
     ("build" "Building SonicBoom"))
    ("Guides"
     ("serve-openai" "The OpenAI-compatible server")
     ("artanis-streaming-notes" "Streaming with GNU Artanis")
     ("onnx-to-s-expr-importer" "Importing ONNX")
     ("novel-revision-final-paragraph" "Novel revision: a chapter's final paragraph"))
    ("API reference"
     ("api/index" "Guile binding")
     ("capi" "The C API"))
    ("Concepts"
     ("architecture" "Architecture")
     ("scope" "Scope")
     ("boom-ir" "Boom IR")
     ("s-expr-v0-spec" "S-expression format v0")
     ("s-expr-v0.2-draft" "S-expression format v0.2 (draft)")
     ("static-execution-plan-format-v0" "Static execution plan format v0")
     ("weight-quant-format" "Weight quantization formats")
     ("gguf-reader" "Reading GGUF")
     ("backend-hal" "The backend HAL")
     ("backend-author-guide" "Writing a backend")
     ("fattn-autotune" "Attention autotuning")
     ("decision-registry" "The decision registry"))
    ("Coming from PyTorch"
     ("pytorch" "Coming from PyTorch"))
    ("Benchmarks"
     ("benchmark" "Benchmarks"))
    ("Project"
     ("roadmap" "Roadmap"))))
