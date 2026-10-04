# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# `python -m onnx2sx` entry point.

import sys

from .importer import main

if __name__ == "__main__":
    sys.exit(main())
