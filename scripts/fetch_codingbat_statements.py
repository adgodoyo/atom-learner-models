#!/usr/bin/env python
"""Fetch the 50 CodeWorkout problem statements from their public CodingBat pages.

Input: artifacts/cw_coding_bat_sources.json (method name -> source URL, the character spans the
thesis's version omits from the page text, e.g. cross-references and trailing examples, and the
statement's SHA-256).
Output: artifacts/cw_coding_bat_recovery_raw.json (method name -> {statement, source_url}), the
file `scripts/cw_4b_recover_problem_statements.py` reads. Every fetched statement is checked
against its published hash, so the rebuilt text is identical to the text the thesis used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

from bs4 import BeautifulSoup

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCES = REPO / "artifacts" / "cw_coding_bat_sources.json"
OUT = REPO / "artifacts" / "cw_coding_bat_recovery_raw.json"


def fetch_statement(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research reproduction)"})
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    div = BeautifulSoup(html, "html.parser").find("div", class_="minh")
    if div is None:
        raise ValueError(f"no statement block found at {url}")
    return " ".join(div.get_text(" ", strip=True).replace("\xa0", " ").split())


def apply_deletions(text: str, spans: list[list[int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()
    sources = json.loads(SOURCES.read_text())
    out, bad = {}, []
    for method, entry in sources.items():
        text = apply_deletions(fetch_statement(entry["source_url"]), entry.get("delete_spans", []))
        if hashlib.sha256(text.encode()).hexdigest() != entry["statement_sha256"]:
            bad.append(method)
        out[method] = {"statement": text, "source_url": entry["source_url"]}
        time.sleep(args.delay)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {len(out)} statements to {OUT.relative_to(REPO)}")
    if bad:
        print(f"{len(bad)} statement(s) differ from the published hashes (the page changed): {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
