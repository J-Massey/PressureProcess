# SU pressure — report figures, section & upload data

Self-contained report directory for the SU pressure subset (products **B/C/E/F**
of `tab:dataSU`): the results section, every figure, the plotting code that
regenerates them, and — under `data/SU/` — the upload products themselves.

## Contents
- `results_section.tex` — the results section (paste into the manuscript, or set
  `\graphicspath{{./}}` so the local PNGs resolve).
- `plot_report.py` — regenerates all figures and prints the quantitative
  numbers quoted in the text (rejection variance fractions, baseline drift).
- `*.png` — the figures, produced by `plot_report.py`.
- `data/SU/` — the framework upload products (built by
  `python -m src.upload.build_framework_products`; gitignored, ~12.6 GiB).

## Consistency check
`plot_report.py` reads **only the upload products** in `data/SU/` and rebuilds
every report figure from them. If the figures match the manuscript, the
uploaded data is self-consistent with the report.

    python report/plot_report.py     # run from the repo root

| Figure | Source file (`data/SU/`) | Layout / normalisation |
|---|---|---|
| `B1_wallp_raw.png` / `B2_wallp_raw.png` | `{B,E}_*_wallp_SU_raw.hdf5` | 3 panels (per psig, titled Re_τ^smooth) × 4 stations; PSD φ_pp [Pa²/Hz] vs f, log–log (not premultiplied) |
| `B1_wallp_production.png` / `B2_wallp_production.png` | `{B,E}_*_wallp_SU_production.hdf5` | 3 panels × 4 stations; premultiplied inner-scaled fφ⁺ vs T⁺, semilog-x (per-station u_τ) |
| `B1_freestream.png` / `B2_freestream.png` | `{C,F}_*_freestreamp_SU_production.hdf5` | 1 panel, 3 psig; PSD φ_pp [Pa²/Hz] vs f, log–log (not premultiplied) |
| `baseline_verification.png` | phase-1 & phase-2 pipeline (figure-only, not in the table) | 3 panels (per psig); premultiplied inner-scaled fφ⁺ vs T⁺ |

Convention: log–log dimensional spectra are plain PSDs; premultiplication is
used only on the semilog-x inner-scaled figures, where equal areas correspond
to equal energy.

The baseline figure is the one exception to the data/SU rule: the smooth-wall
drift check has no slot in `tab:dataSU`, so it reads
`../data/phase{1,2}/pressure/G_wallp_SU_production.hdf5`.

## Noise rejection (what the Wiener filter is and is not)
The production wall pressure is facility-noise rejected with a **fixed Wiener
FIR kernel identified on the smooth-wall baseline** (phase 2), where the
nose-cone freestream reference carries facility noise only. On B1/B2 that fixed
kernel filters the simultaneously recorded nose-cone signal to form the noise
estimate that is subtracted. The rejection is **not** an adaptive cancellation
referenced to the case's own freestream microphone — on a protrusion case the
nose-cone also hears the body's dipole/quadrupole radiation, and training
against it would delete genuine flow-generated wall-pressure content.

## Band limiting
No digital band filtering is applied to the signals anywhere in the pipeline
(the analog anti-alias low-pass acts upstream in hardware; the Wiener stage
sees only demeaned signals). The reported band is enforced purely as a
**spectral mask when plotting/using the spectra**: every wall-pressure curve
stops at its per-station cut

    f_cut = min( f_H = 1.2 kHz ,  u_τ² / (ν · T⁺_cut) ),   T⁺_cut = 10

read from the dataset attribute `f_cut_Hz` (see `src/band_limits.py`). The
pinhole Helmholtz cavity resonance sits at ≈1.4 kHz at **every**
pressurisation (set by cavity geometry and sound speed — it does not scale
with Re_τ), so the fixed 1.2 kHz guard binds almost everywhere; the T⁺
resolution term binds only at the lowest-u_τ fence stations at 0 psig
(≈0.95–1.15 kHz). The nose-cone **freestream** mic is not pinhole-mounted,
so it is not clipped. `f_cut_Hz` is distinct from `analog_LP_filter_Hz`
(the per-condition analog anti-alias filter at 2100/4700/14100 Hz).
