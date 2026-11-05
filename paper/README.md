# Paper: "Memory in Lévy walks from the dynamic-entropy formalism"

A. Barra, A. Ladiana (corresponding) — draft for *Physical Review E* (REVTeX 4.2, two columns).

## Build

```
cd scripts
python compute.py all          # new computations -> ../data (planted-vs-inferred, phase coordinates; seconds)
python make_figs.py            # all figures -> ../figs (pdf + png), from ../../results and ../data
cd ..
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Layout

The text follows model → physics → inference → data, for single walkers and groups together.

| path | content |
|---|---|
| `main.tex` | front matter (abstract to be written last), data availability, acknowledgments |
| `sections/01_intro.tex` | I. Introduction (no figure) |
| `sections/02_model.tex` | II. Flights with memory: one flight, path entropy, causal chain, many walkers (Figs. 1, 2 defined here) |
| `sections/03_statmech.tex` | III. Physics: stationary law, duality and condensation, dressed exponent, collective critical point |
| `sections/04_inference.tex` | IV. Inference: fitting and depth, what looks like memory, reading interactions from a group |
| `sections/05_data.tex` | V. Data: larvae, termites, baboons and pigeons, T cells as negative control (Figs. 3, 4) |
| `sections/06_discussion.tex` | VI. Discussion |
| `sections/A_appendices.tex` | A one flight and K=1 chain · B large range · C estimation · D selection and validation · E what looks like memory · F data sets and single-walker controls · G collective model (Figs. 5–9, Tables I–III) |
| `refs.bib` | bibliography (apsrev4-2) |
| `scripts/style.py` | one look for all figures: Arial, PRE sizes 86/178 mm, semantic palette |
| `scripts/common.py` | paths, labels, exact stationary law of the K=1 chain, drawing helpers |
| `scripts/panels.py` | one function per panel, shared by the figure modules |
| `scripts/fig*.py` | one module per figure, `make()` |
| `scripts/compute.py` | the only new computations: `validation` (200 planted walkers), `phase` (K=1 coordinates of larvae and termites) |
| `data/` | outputs of `compute.py` and cached theory maps |
| `figs/` | final figures |
| `archive/` | earlier versions (e.g. `2026-11-05_with_TODOs/`, the annotated draft before the restructure) |

## Figures

Main text
1. `fig1_overview` — larval track, flights as Heisenberg spins, phase diagram with larvae and termites, stationary laws, free energy.
2. `fig2_many` — own vs others' past, aggregation of own memory by the group average, mean-field phase diagram of the collective speed, three regimes.
3. `fig3_walkers` — larval couplings per condition, termite depth and kernel, dressed exponent (μ̂ vs μ_bare).
4. `fig4_groups` — baboons (lags, rank, ahead/behind), pigeons (lags, rank), group-average depth on data vs time-shifted units for T cells, baboons, pigeons.

Appendix
5. `figA1_theory` — Langevin response, finite-size crossing of Π, dressed-exponent curves.
6. `figA2_validation` — planted walkers: Fisher calibration, depth selection, AIC/BIC false positives.
7. `figA3_pitfalls` — variable, heterogeneity, segmentation, signatures of mechanisms.
8. `figA4_controls` — larval robustness heatmap, termite wall field, T-cell heterogeneity.
9. `figA5_collective` — common drives, critical fluctuations with and without own memory.

Colours: indigo genuine memory/data, coral artefacts and flawed readings, teal fields/common drives, plum social coupling, greys nulls; larvae indigo, termites ochre, T cells slate. Fig. 1(a) is readable in black and white.
