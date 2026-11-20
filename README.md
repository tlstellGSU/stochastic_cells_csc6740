# Stochastic cell walls: CSC 6740 project

Tommy Stell, Georgia State University

Red blood cells flicker. Their walls move by tens of nanometres at random. This project mines public
1 kHz videos of flickering cells for the statistical law of that motion. It then asks whether the moving
wall changes transport inside the cell. There are two questions:

- **Q1 (inverse).** What can contour videos identify about the wall's stochastic law? Is a passive
  membrane enough, or is a colored (active) forcing needed?
- **Q2 (forward).** With the fitted law, does the wall's motion change how a molecule travels from the
  membrane to the cell centre?

Each shape mode of the wall is modelled as an Ornstein–Uhlenbeck (OU) process. The models are fitted
by spectral (Whittle) likelihood, checked by profile likelihood and by a calibrated model comparison,
and then used to drive a transport simulation.

## Contents

| Path | What it is |
|---|---|
| `stochastic_cells.py` | The whole analysis pipeline in one file, from the raw videos to the report and figures (below) |
| `requirements.txt` | Python dependencies (versions verified with Python 3.12) |
| `paper/ieee_paper.tex` | The report (IEEE conference format, 8 pages); built: `paper/ieee_paper.pdf`. `IEEEtran.cls`/`.bst` are included |
| `paper/figures/` | The report's figures, as finished PDFs |
| `paper/ieee_refs.bib`, `paper/ieee_extra.bib` | References cited in the report |
| `slides/deck.tex`, `slides/deck_notes.tex` | The talk as a Beamer deck (19 pages); `deck_notes.tex` adds the speaker notes on a second screen. Built: `slides/deck.pdf`, `slides/deck_notes.pdf` |
| `slides/figures/` | The deck's images, as finished PNGs |
| `slides/class_deck_2026-11-18.pptx` | The current version of the talk in PowerPoint. It has animated GIFs, a representative video, the implementation-scheme slide and the travel-time-with-decay results |

## The pipeline (`stochastic_cells.py`)

Each section of the file is one stage of the data-mining process, and the section names follow the
report.

| Stage | Code section | What it does | Report |
|---|---|---|---|
| 1. Data | `1. DATA` | Loads the 11 videos (6 free, 5 optically trapped; 300 × 300 px, 47.2 nm/px, 1 kHz, 5000 frames each) | §II-A |
| 2. Preprocessing | `2. PREPROCESSING` | Finds the contour as a per-ray level crossing on 2048 rays (`LevelCrossingTracker`). Validates it on synthetic frames rendered from each video's own intensity profile and pixel noise (noise floor and gain), and bounds the tracking noise | §II-B to II-D, Fig. 2 |
| 3. Analysis | `3. ANALYSIS` | Computes Fourier contour modes q = 2–6 and their periodograms. Tests Gaussianity against phase-randomized surrogates and measures the memory (autocorrelation time). Builds the generative model: passive OU modes (P), or OU modes driven by a colored OU forcing (A), projected onto the equator and exposure-averaged. Fits both by the Whittle likelihood | §II-E, III, IV |
| 4. Evaluation | `4. EVALUATION` | 95% profile-likelihood intervals (identifiability). ΔAIC = AIC(P) − AIC(A), calibrated by a parametric bootstrap under three passive null generators. A goodness-of-fit statistic | §IV, V |
| 5. Interpretation | `5. INTERPRETATION`, `REPORT` | Paired molecules in a moving and a frozen copy of a fitted wall, with the cytosol carried by the wall's Stokes flow. Measures the first passage from a membrane channel to r < 0.3R with first-order decay. Writes a report whose answers are computed from the results | §VI, VII; slides |

### Get the data

The raw videos are the public deposit behind Caselli et al., *iScience* 27, 109915 (2024):
<https://doi.org/10.6084/m9.figshare.24018015> (CC BY 4.0; `Free_cells.rar`, 4.0 GB, and
`Trapped Cells.rar`, 3.3 GB). Unpack both into `data/` next to `stochastic_cells.py`:

```
data/Free_cells/RBC1/captura_p0001.tif ... captura_p5000.tif     (RBC1, 2, 3, 10, 11, 13)
data/Trapped Cells/Trapped_RBC1/captura_p0001.tif ...           (RBC1, 2, 4, 10, 13)
```

`--data PATH` points the script at another location.

### Run

```bash
python3 -m pip install -r requirements.txt
python3 stochastic_cells.py --quick        # whole pipeline, reduced sizes: 2 videos, 1000 frames
python3 stochastic_cells.py                # full run, all 11 videos (2.7 h on 23 processes; the bootstrap null dominates)
python3 stochastic_cells.py --stages fit profile   # selected stages; each stage caches its output
```

The stages, in order, are `track explore fit profile select null transport report`. `--workers N` sets
the number of processes (the default is all cores but one). `--transport-cell`, `--D` and `--decay`
choose the transport case.

Outputs go to `output/`:

- `tracks/*.npz` (contours and validation)
- one JSON file per stage
- `report.txt`
- `figures/fig1_data.png` to `fig6_transport.png`

### What it reproduces, and where it differs from the report

- **Tracking** reproduces the research repository's contours exactly. The per-video rms amplitudes are
  identical (23.3–31.8 nm) and the noise floors agree within 0.3 nm (the synthetic frames use different random seeds).
- **Fits** use one protocol throughout: binned Whittle likelihood with l_max = 40 for estimates and
  profiles, and unbinned with l_max = 80 for ΔAIC. With this protocol:
  - The passive fits P match the report's estimates in all 11 videos.
  - The colored fits A match in 10 of 11.
  - For free RBC 1, a wider set of starting points finds a better optimum of A than the one in the
    report (Limitations, §VIII).
- **Transport** uses the measure the talk ends on: the time to reach the centre with decay. The report's
  share-at-a-fixed-time measure Φ comes from a grid advection–diffusion scheme, which is not included
  here.
- **Transport walls.** Model A's best fit is degenerate: the bending modulus runs to zero along a flat
  likelihood ridge, which leaves the instantaneous wall undefined. The transport walls therefore keep the
  parameters the videos do identify (A's forcing strength and memory, P's bending modulus) and refit the
  rest. The report prints the simulated wall amplitude beside the measured one as a check (full run,
  free RBC 1: 25.2 nm simulated, instantaneous; 22.2 nm measured, exposure-averaged and including tracking
  noise).
- **ΔAIC and its null.** The full run gives ΔAIC = +796 to +1479, the report's range. Its null
  calibration has the report's design (four cells, 20 replicates each, 80 per generator), but its 95th
  percentiles are +2.9 (G0), +6.1 (G1) and +567 (G2), against the report's 26.9, 36.3 and 554.
  The cause of the G0 and G1 difference has not been traced. The conclusion does not change: all 11 videos
  exceed every null by at least 229.

### Full-run results (`output/report.txt`)

| | |
|---|---|
| Wall motion | 23–32 nm rms; shape modes Gaussian in 11/11 videos; memory τ(q=2) = 17–160 ms |
| Identifiable (closed 95% intervals) | P: κ 11/11, η 11/11. A: a 11/11, τ_a 11/11, κ 0/11 |
| A's forcing | τ_a = 12–60 ms, a = 53–3180 |
| ΔAIC (P − A) | +796 to +1479; above all three nulls in 11/11 videos |
| Transport, free RBC 1, D = 5 µm²/s, decay 0.4–11.8 /s | wall motion changes the share arriving and the travel time by < 0.10% (2 SE); 0/8 changes differ from zero by more than 2 SE |

## Build the report and the slides

```bash
cd paper  && latexmk -pdf ieee_paper.tex
cd slides && latexmk -pdf deck.tex && latexmk -pdf deck_notes.tex
```

Both need only a standard TeX Live (pdfLaTeX; Beamer, textpos, helvet, IEEEtran is included).

### Figure provenance

The figures are shipped as finished files. They were produced in the research repository
(<https://github.com/tlstellGSU/brownian_cells>) from its experiment results:

- `paper/figures/`: `paper/paper_figures/make_figures.py` and `fig_contour.py` there.
- `slides/figures/`:
  - `gs_*.png` are the figures of the Google Slides version of the talk.
  - `stochastic_stats.png` and `simple_simulation.png` come from
    `results/journal_club/simple_simulation_2026-11-18.py`.
  - `model_selection.png` and `misfit.png` come from `slide_figures_class_2026-11-18.py`.
  - `pipeline_2d.png` comes from `pipeline_overview.py`.

`stochastic_cells.py` regenerates its own versions of the main analysis figures from the raw videos.

## License and data

The video data are © Caselli et al. (2024), CC BY 4.0, and are not redistributed here.
