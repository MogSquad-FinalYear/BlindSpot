#!/usr/bin/env python3
"""
Self-hosted similarity check for the manuscript.

This is NOT a substitute for Turnitin/iThenticate, which compare against a
licensed corpus we do not have. What it CAN do is check the thing most
likely to trip a real similarity report for this particular paper: verbatim
or near-verbatim overlap with the small set of source documents the authors
read closely while writing (the OMAT source paper, detector repository
READMEs, and the project's own internal analysis notes).

Method: strip LaTeX markup, lowercase, normalise whitespace, then report
every shared word n-gram of length >= N between the manuscript and each
source. Quoted material is expected to appear and is reported separately so
it can be confirmed as properly attributed rather than silently flagged.
"""
import argparse
import os
import re
import subprocess
import sys


def strip_latex(text):
    text = re.sub(r"(?m)^\s*%.*$", " ", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(cite|ref|label|includegraphics|input)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$&~^_\\]", " ", text)
    return text


def normalise(text):
    text = text.lower()
    text = re.sub(r"``|''|[\"']", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ngrams(tokens, n):
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def read_pdf(path):
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not read {path}: {e}", file=sys.stderr)
        return ""


def load_source(path):
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    try:
        with open(path, errors="ignore") as f:
            return f.read()
    except Exception:  # noqa: BLE001
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="/home/student/data/paper/paper_draft.tex")
    ap.add_argument("--n", type=int, default=8, help="minimum shared n-gram length to flag")
    ap.add_argument("--sources", nargs="+", required=True)
    args = ap.parse_args()

    paper_raw = open(args.paper).read()
    paper_tokens = normalise(strip_latex(paper_raw)).split()
    print(f"manuscript: {len(paper_tokens)} words after markup stripping\n")

    # quoted spans are legitimate if attributed; track them separately
    quoted = re.findall(r"``(.+?)''", paper_raw, flags=re.S)
    quoted_norm = [normalise(strip_latex(q)) for q in quoted]

    total_flagged = 0
    for src in args.sources:
        if not os.path.exists(src):
            print(f"--- {os.path.basename(src)}: NOT FOUND, skipped")
            continue
        stoks = normalise(strip_latex(load_source(src))).split()
        if len(stoks) < args.n:
            print(f"--- {os.path.basename(src)}: too short/unreadable, skipped")
            continue
        shared = ngrams(paper_tokens, args.n) & ngrams(stoks, args.n)
        # collapse overlapping n-grams into maximal runs for readability
        hits = sorted(shared, key=len, reverse=True)
        merged = []
        for h in hits:
            if not any(h in m for m in merged):
                merged.append(h)
        in_quote = [h for h in merged if any(h in q for q in quoted_norm)]
        outside = [h for h in merged if h not in in_quote]
        pct = 100.0 * sum(len(h.split()) for h in outside) / max(len(paper_tokens), 1)
        print(f"--- {os.path.basename(src)} ({len(stoks)} words)")
        print(f"    shared {args.n}+ word sequences: {len(merged)} "
              f"({len(in_quote)} inside attributed quotations, {len(outside)} outside)")
        print(f"    unattributed overlap: ~{pct:.2f}% of manuscript words")
        for h in outside[:12]:
            print(f"      * \"{h}\"")
        if len(outside) > 12:
            print(f"      ... and {len(outside)-12} more")
        total_flagged += len(outside)
        print()

    print(f"TOTAL unattributed {args.n}+ word overlaps across all sources: {total_flagged}")


if __name__ == "__main__":
    main()
