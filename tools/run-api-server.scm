#!/bin/bash

guile -L bindings/guile/scheme \
      bindings/guile/scheme/sonicboom/serve/app.scm --port 8000
