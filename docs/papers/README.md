# Iterative-vs-static edge-criticality papers

Twelve standalone LaTeX conference papers (network × method), following the
AsiaSim2024/JSST2024 structure of the two reference `.docx` files in the repo
root (iterative metric re-estimation thesis).

| Network ↓ / Method → | LDC | LLBCe | LLBMEe1 |
|---|---|---|---|
| Jazz | `jazz_LDC.tex` | `jazz_LLBCe.tex` | `jazz_LLBMEe1.tex` |
| Dolphins | `dolphins_LDC.tex` | `dolphins_LLBCe.tex` | `dolphins_LLBMEe1.tex` |
| Baseball | `baseball_LDC.tex` | `baseball_LLBCe.tex` | `baseball_LLBMEe1.tex` |
| Transport | `transport_LDC.tex` | `transport_LLBCe.tex` | `transport_LLBMEe1.tex` |

Each paper: Abstract → Keywords → Introduction → Method + iterative variant →
Network & experimental setup (stats table) → Results (static-vs-iterative RGC
figure + AUC table + all-metrics context table) → Discussion → Conclusion →
References.

## Headline results (dismantling AUC, lower = better)

| Network | LDC static→iter | LLBCe static→iter | LLBMEe1 static→iter |
|---|---|---|---|
| Jazz | 0.914 → 0.936 (**−2.4%**) | 0.585 → 0.419 (+28.4%) | 0.770 → 0.434 (+43.7%) |
| Dolphins | 0.675 → 0.657 (+2.7%) | 0.455 → 0.324 (+28.8%) | 0.610 → 0.430 (+29.5%) |
| Baseball | 0.936 → 0.960 (**−2.6%**) | 0.936 → 0.668 (+28.6%) | 0.936 → 0.670 (+28.4%) |
| Transport | 0.239 → 0.196 (+18.1%) | 0.398 → 0.368 (+7.5%) | 0.225 → 0.160 (+29.1%) |

Iterative re-estimation helps LLBCe and LLBMEe1 on every network. For LDC it
helps only on the sparse networks (Transport, Dolphins) and slightly *hurts* on
the dense Jazz/Baseball graphs — reported honestly in the papers.

## Regenerating

```bash
.venv/Scripts/python.exe docs/papers/make_figures.py   # 16 figures + stats.json
.venv/Scripts/python.exe docs/papers/build_papers.py   # 12 .tex files
```

`make_figures.py` recomputes the static RGC curve with the repo's own
`NetworkDismantler` (verified to reproduce the recorded `aucs_*.csv` exactly)
and reuses the saved iterative RGC CSVs. No data files are modified.

## Compiling

No LaTeX engine is installed in this environment. On a machine with TeX Live /
MiKTeX:

```bash
cd docs/papers
pdflatex jazz_LDC.tex && pdflatex jazz_LDC.tex   # twice for cleveref refs
```

Figures resolve via `\graphicspath{{figures/}}`. The author block contains a
`[author names to be completed]` placeholder to fill in before submission.
