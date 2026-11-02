# ICMLDE 2026 revised submission

Revised manuscript and the point-by-point reviewer response, supplied both as
separate documents and combined into the single file the revision
instructions ask for.

## Three documents

| File | What it is | Pages |
|---|---|---|
| `ICMLDE_manuscript.tex/.pdf` | revised manuscript on its own | 14 |
| `ICMLDE_response.tex/.pdf` | point-by-point reviewer response on its own | 4 |
| `ICMLDE_revised.tex/.pdf` | both in one file, manuscript then response | 17 |

The conference instructions ask for the response and the manuscript in a
**single** file named `<PaperID>_revised_version.pdf`, which is what
`ICMLDE_revised.pdf` is. The two split documents are there for convenience;
if you upload those instead, check with the chairs first.

`ICMLDE_response.tex` is standalone: its section, table and figure numbers are
hardcoded to match the compiled manuscript, so it needs no `.aux` from the
other document. If you renumber anything in the manuscript, re-check those
numbers in the response.

## Build

Compile with **pdfLaTeX** (the Procedia `ecrc.sty` template expects it):

```
pdflatex ICMLDE_manuscript.tex
pdflatex ICMLDE_manuscript.tex   # second pass resolves \ref and \cite
pdflatex ICMLDE_response.tex     # plain article class, one pass is enough
pdflatex ICMLDE_revised.tex      # the combined version, twice
pdflatex ICMLDE_revised.tex
```

The ScienceDirect and Elsevier logo images ship in the distributed archive but
are kept out of the public repository, to avoid redistributing Elsevier
branding. `ecrc.sty` falls back to a plain rule when they are absent, so the
documents compile either way; if you cloned the repo and want the full
masthead, copy `Procs.pdf`, `SDlogo-3p.pdf` and `elsevier-logo-3p.pdf` from the
official Procedia author template into this folder.

Overleaf: upload the whole folder and set the main document to whichever of
the three you want, compiler pdfLaTeX. The bibliography is a literal
`thebibliography` block, so there is no BibTeX step.

The PDFs shipped here were built with Tectonic (XeTeX) as a structural check.
pdfLaTeX output is what to submit; pagination may shift by a line or two
between the engines, and bold in the combined file's response section renders
correctly under pdfLaTeX (Tectonic substituted a non-bold shape).

## Layout

```
ICMLDE_manuscript.tex     manuscript alone (Sections 1-10 + Appendix A)
ICMLDE_response.tex       reviewer response alone
ICMLDE_revised.tex        both, manuscript then response
figures/                  fig_reliability_diagrams.{pdf,png}  (Figure 1)
                          fig_score_dist_ufd.png  (see note below)
elsarticle.cls ecrc.sty framed.sty elsarticle-harv.bst
                          Procedia CRC template files (LPPL)
review_additions/         data behind the four new experiments
```

`figures/fig_score_dist_ufd.png` is the per-image score histogram that was in
the submitted version. It was cut for length; the reliability diagrams carry
the same evidence more rigorously. A commented-out `figure` block sits in
Section 4.7 of the `.tex` if you want it back.

## What changed

Four new experiments, all run for this revision:

| Experiment | Where | Asked for by |
|---|---|---|
| Ensembling by training lineage vs. architecture | §4.4, Table 6 | R2.6 |
| Per-detector reliability diagrams | §4.7, Figure 1 | R3.3 |
| Matched-compression estimate of the PNG/JPEG confound | §6, Table 8 | R2.5 |
| Comparison against published numbers for the same checkpoints | §3.5, Table 3 | R3.5 |

New prose: generator-selection rationale (§3.2), practitioner recommendations
(§7), diagnostic-study framing (title, abstract, §1), the qualification of the
two-detector claim (§4.3), and an itemised data/code release statement
(Appendix A).

Cut to make room: the three preprocessing subsections merged into one;
spectral-deviation analysis reduced to a paragraph and its figure removed;
"generator easiness is detector-relative" folded into §4.1; the score
histogram removed; related work, methodology, limitations, ethics and
conclusion tightened throughout. Roughly a fifth of the carried-over text is
gone, and about 62% of the revised body is newly written or rewritten.

## review_additions/

Every number in the four new experiments is reproducible from the per-image
prediction CSVs:

- `ensemble_pairs.csv` — all ten detector pairs, per-generator fake-accuracy
  under mean-probability fusion, plus mean/worst/spread and the BigGAN and
  GLIDE real-accuracy and AUC
- `ensemble_summary.json` — group means by lineage and representation
- `ensemble_mcnemar.csv` — the paired test behind the §4.4 headline contrast
- `single_detectors.csv` — the same statistics for each detector alone
- `reliability_ece.csv` — ECE per detector/generator (matches Table 4)
- `compression_conditions.csv` — the three compression conditions (Table 8)
- `scripts/` — the two scripts that produce all of the above

Both scripts read the per-image prediction dumps and need no GPU. Paths at the
top of each file point at the prediction CSVs in the main project tree.

## One thing to check before submitting

The conference asks for a similarity report under 10%. Overlap with your own
earlier *submitted* (unpublished) version is normally excluded from that
figure, but configure the check to exclude it explicitly if the tool allows,
since about 38% of the revised body is carried over verbatim from the
submission the reviewers read.
