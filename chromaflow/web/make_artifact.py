#!/usr/bin/env python3
"""Strip index.html down to the body content some hosts want.

`web/index.html` is a complete, standalone page — open it from disk, serve it,
put it on GitHub Pages.  Hosts that wrap your markup in their own document
skeleton want only what goes *inside* the body, plus the title and styles.
This produces that, without keeping a second copy of the page in the repo.

    python web/make_artifact.py [out.html]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "index.html"


def to_body_fragment(html: str) -> str:
    title = re.search(r"<title>.*?</title>", html, re.S)
    style = re.search(r"<style>.*?</style>", html, re.S)
    body = re.search(r"<body[^>]*>(.*)</body>", html, re.S)
    if not (title and style and body):
        raise SystemExit("index.html is not shaped the way this script expects")
    return "\n".join([title.group(0), style.group(0), body.group(1).strip(), ""])


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "chromaflow-fragment.html")
    out.write_text(to_body_fragment(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
