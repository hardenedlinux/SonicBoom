#!/usr/bin/env python3
"""Check the generated pages for inline styles.

Every page's <head> may hold, directly, only meta, title, link and script
elements and whitespace, and no page may hold a <style> element at all: the
highlighting rules live in highlight.css, which the template links.

Usage: check-html.py PAGE...
Prints one line per problem and exits 1 when any page fails.

SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
"""

import sys
from html.parser import HTMLParser

VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
ALLOWED_IN_HEAD = {"meta", "title", "link", "script"}


class Checker(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.problems = []
        self.in_head = False
        self.depth = 0

    def problem(self, message):
        self.problems.append("%s: %s" % (self.path, message))

    def handle_starttag(self, tag, attrs):
        if tag == "style":
            self.problem("has a <style> element")
        if self.in_head:
            if self.depth == 0 and tag not in ALLOWED_IN_HEAD:
                self.problem("the <head> holds a <%s> element" % tag)
            if tag not in VOID:
                self.depth += 1
        elif tag == "head":
            self.in_head = True
            self.depth = 0

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.in_head and tag not in VOID:
            self.depth -= 1

    def handle_endtag(self, tag):
        if not self.in_head:
            return
        if tag == "head" and self.depth == 0:
            self.in_head = False
        elif self.depth > 0:
            self.depth -= 1

    def handle_data(self, data):
        if self.in_head and self.depth == 0 and data.strip():
            self.problem("text directly in the <head>: %r"
                         % data.strip()[:40])

    def handle_comment(self, data):
        if self.in_head and self.depth == 0 and data.strip():
            self.problem("a comment directly in the <head>")


def main(argv):
    problems = []
    for path in argv[1:]:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as error:
            problems.append("%s: %s" % (path, error))
            continue
        checker = Checker(path)
        checker.feed(text)
        checker.close()
        problems.extend(checker.problems)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
