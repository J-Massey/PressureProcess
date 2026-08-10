PressureProcess
===============

Processing pipeline for the SAPPHiRe wall-pressure and freestream-pressure
measurements. It reads raw MATLAB acquisitions, builds semi-anechoic
calibration transfer functions, applies them to the measured time series,
removes facility noise with a Wiener canceller, and writes HDF5 products plus
sanity-check figures.

The repo processes **five datasets**, each with its own module tree and its own
hard-coded configuration. They share one processing backbone but differ in
which calibration each borrows and how noise rejection is done — those
differences are the substance of this document.


Contents
--------

- [Setup](#setup)
- [The measurement](#the-measurement)
- [Input layout](#input-layout)
- [Processing stages](#processing-stages)
- [The five cases](#the-five-cases)
- [Running a case](#running-a-case)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Testing](#testing)
- [Known defects](#known-defects)


Setup
-----

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Every command in this document must be run from the repository root.** All
paths in every module are relative, and each case's `Config` is instantiated at
module import time, so environment variables must precede the interpreter on
the same command line.


The measurement
---------------

Three microphones are recorded simultaneously at 50 kHz, 24-bit:

| Channel | Sensor | Role |
| --- | --- | --- |
| `PH1`, `PH2` | pinhole-masked B&K 1/2" Type 4964 | wall pressure, two streamwise stations |
| `NC` | nose-cone-treated B&K 1/2" Type 4964 | freestream reference, carries facility noise |

A fourth signal, `nkd` ("naked", untreated), appears only in the semi-anechoic
calibration recordings as the reference the `NC` mic is calibrated against.

Each dataset is measured at three tunnel pressurisations and, for most cases,
at two pinhole spacings (`close`, `far`). Pressurising the tunnel raises density
and therefore `Re_tau` at fixed velocity — that is the point of the sweep.

Two separate calibrations are involved, and keeping them straight is essential:

- **PH → NC**: corrects the pinhole's frequency response (the Helmholtz cavity
  behind the 70 µm pinhole) onto the nose-cone mic's response. Measured in a
  semi-anechoic chamber with a white-noise source, in two runs
  (`_1` positions PH1, `_2` positions PH2).
- **NC → nkd**: corrects the nose-cone treatment onto a bare microphone.
  Measured separately, with the facility fan off.

### The reported band

Wall-pressure content above `f_cut` is not reported. `f_cut` is a **reporting
limit applied as a spectral mask at plotting time, not a filter** — the
pipelines never digitally band-pass the signals. It is

```
f_cut = min( 1200 Hz , u_tau^2 / (nu * T+_cut) )      with T+_cut = 10
```

The 1200 Hz term guards the pinhole cavity resonance at ~1.4 kHz. That
resonance is set by cavity geometry and the speed of sound, so it sits at the
**same frequency at every pressurisation** — only its damping falls as density
rises. It therefore binds as a fixed guard in Hz, not as a viscous-scaled cut.
The second term is the pinhole's spatial-resolution limit, which does scale
with the local friction velocity.

With each case's nominal `u_tau`, the resonance guard binds everywhere, so
every case currently reports `F_CUTS == (1200.0, 1200.0, 1200.0)`. See
[`src/band_limits.py`](src/band_limits.py).

> `f_cut` is **not** `ANALOG_LP_FILTER`, which is the acquisition anti-alias
> filter (2100 / 4700 / 14100 Hz). An older `(1200, 4000, 10000)` triple, which
> raised the cut with `Re_tau` and left the resonance inside the band, was
> wrong and has been removed.


Input layout
------------

All paths are relative to the case's `Config.ROOT_DIR`. `<label>` ranges over
`Config.LABELS` and `<spacing>` over `Config.SPACINGS` — both are case-specific
(see [The five cases](#the-five-cases)).

```text
ROOT_DIR/
  raw_wallp/<spacing>/<label>.mat            wall-pressure + freestream run
  raw_calib/PH/calib_<label>_1.mat           PH1 semi-anechoic calibration
  raw_calib/PH/calib_<label>_2.mat           PH2 semi-anechoic calibration
  raw_calib/NC/<label>/nkd-ns_nofacilitynoise.mat    NC semi-anechoic calibration
```

### MATLAB variables and channel order

| File | Variable | Shape | Column order |
| --- | --- | --- | --- |
| `raw_wallp/<spacing>/<label>.mat` | `channelData` | (N, 3) | `PH1, PH2, NC` |
| `raw_calib/PH/calib_<label>_{1,2}.mat` | `channelData_WN`, else `channelData` | (N, ≥3) | `PH1, PH2, NC, …` |
| `raw_calib/NC/<label>/nkd-ns_…mat` | `channelData` | (N, 2) | `nkd, NC` |

Two quirks that are not typos:

- The NC calibration file at **100 psig only** stores its array under
  `channelData_nofacitynoise` (missing "li"). Every other pressure uses
  `channelData`. The loaders hard-code this special case.
- The NC calibration file's column order is `nkd, NC` — the reverse of the
  intuitive reading, and different from the other two file types.

All `.mat` payloads are in **volts**. The raw stages divide by
`SENSITIVITIES_V_PER_PA` (~50 mV/Pa) to get pascals.


Processing stages
-----------------

Each case runs five stages in this order. `fs_proc` must precede `pw_proc`,
because `pw_proc` reads the freestream production file as its Wiener reference.

**1. `calibs.save_PH_calibs()` → `calibration/PH/calibs_<psig>.h5`**

Loads the two PH calibration runs, converts volts → Pa, applies the
sensitivity-drift correction, estimates `H1` (PH1→NC) and `H2` (PH2→NC) with
Welch/CSD at `nperseg=4096`, then fuses them on a common grid with
coherence weights `w = γ²/(1-γ²)` gated at `γ² ≥ 0.4`, followed by 1/6-octave
complex log-frequency smoothing. Writes `frequencies`, `H1`, `H2`, `H_fused`,
`gamma2_fused`.

**2. `calibs.save_NC_calibs()` → `calibration/NC/calibs_<psig>.h5`**

Same estimator, `x = NC`, `y = nkd`. Only runs where `RUN_NC_CALIBS` is true.

**3. `fs_raw.save_raw_fs_pressure()` → `pressure/F_freestreamp_SU_raw.hdf5`**

Extracts column 2 (`NC`) from each wall-pressure run, converts to Pa, applies
the sensitivity-drift correction. Optionally stores the raw NC calibration
traces under `FRF_NC_to_nkd`.

**4. `pw_raw.save_raw_ph_pressure()` → `pressure/G_wallp_SU_raw.hdf5`**

Same for columns 0 and 1 (`PH1`, `PH2`), plus the PH calibration traces under
`FRF_PH_to_NC/Run{1,2}`.

**5. `fs_proc.save_prod_fs_pressure()` → `pressure/F_freestreamp_SU_production.hdf5`**

Re-reads the `.mat` files (not the raw HDF5), converts NC to Pa, and applies
the NC→nkd FRF. This is the reference the Wiener canceller uses.

**6. `pw_proc.save_corrected_pressure()` → `pressure/G_wallp_SU_production.hdf5`**

Per `(label, spacing, channel)`, at float32:

```
raw PH  →  apply PH→NC FRF  →  apply NC→nkd FRF  →  frf_corrected_signals
                                        ↓
                                     demean
                                        ↓
                        Wiener cancel against the demeaned NC
                                        ↓
                            fs_noise_rejected_signals
```

Both stages are written, so the effect of noise rejection is always inspectable.

### What the noise rejection actually is

The freestream `NC` microphone is **not** an adaptive reference that tracks the
wall signal. A **fixed FIR Wiener kernel** `c[k]` is solved from the
autocorrelation of the reference and its cross-correlation with the wall
signal (Toeplitz system, solved by conjugate gradient with FFT matrix-vector
products), then the estimated noise `c * pn` is subtracted:

```
R_pn c = r_{p0,pn}          p_clean = p0 - alpha * (c * pn)
```

Default FIR order 1024, `alpha = 1.0`, relative ridge `1e-3`.
See [`src/core/wiener_filter_torch.py`](src/core/wiener_filter_torch.py).

For a **smooth wall**, the reference is dominated by facility noise, so this
correctly removes facility content and leaves canonical TBL pressure behind.
For a **bump or fence**, the protrusion radiates as a dipole into the
freestream, so the reference also carries flow content that belongs on the
wall — training a kernel locally would subtract real signal. Those cases
therefore transplant a kernel trained on the smooth-wall donor, which
identifies only the facility-to-wall acoustic channel (approximately
independent of what is in the working section).


The five cases
--------------

| | phase1 | phase2 | bump1 | fence | iso_re |
| --- | --- | --- | --- | --- | --- |
| Module | `src/save` | `src/save_phase2` | `src/save_bump` | `src/save_fence` | `src/save_iso_re` |
| `ROOT_DIR` | `data/phase1` | `data/phase2` | `data/bump1` | `data/fence` | `data/iso_re` |
| Pressures | 0/50/100 | 0/50/100 | 0/50/100 | 0/50/100 | **0/30/50** |
| Spacings | close, far | **close only** | close, far | close, far | **close only** |
| **PH → NC TF** | own root | **phase1's** | **phase1's** | **phase1's** | own root |
| **NC → nkd TF** | own root, identity if absent | own root | own root | own root | **identity, pinned** |
| **Wiener** | trained locally | trained locally | **phase2 donor kernel** | **phase2 donor kernel** | trained locally |
| Config | `src/config_params.py` | own | own | own | own |
| Env-overridable root | **yes** | no | no | no | no |

### Why the borrowing

- **phase2, bump1 and fence all apply phase1's `H_fused`** from
  `data/phase1/calibration/PH/calibs_<psig>.h5`, to *both* channels. Their own
  PH calibrations are still computed and written, but are used only for
  diagnostics. phase2's own PH recordings sit ~15–20 dB low in the
  100–1000 Hz ROI; bump1's and fence's pinholes are degraded. phase1's fused
  TF is the one that is trusted.
- **bump1 and fence apply a Wiener kernel extracted from phase2**
  (`data/phase2/calibration/wiener_kernels.h5`), for the dipole-contamination
  reason above. phase2 is close-only, so **both** spacings of bump1 and fence
  receive phase2's `close` kernel — the facility transfer path is taken to be
  spacing-independent.
- **iso_re borrows nothing.** Its pressures are 0/30/50 psig, so no NC
  calibration at a matching pressure exists; its NC→nkd stage is pinned to an
  identity FRF by construction, not reached by a fallback. (An identity FRF is
  still not a no-op: `apply_frf` demeans and zeroes the DC and Nyquist bins.)

### Other per-case deviations

- **bump1** and **fence** stamp a per-sensor-position `u_tau` and `Re_tau` on
  each dataset, in addition to the per-pressure value on the condition group.
  bump1 reads them from `U_TAU_BY_POSITION` in its config; fence reads them
  from `data/fence/A_B1_shear_SU_production.hdf5` at
  `/Production/<ATM|50_psig|100_psig>/<P1..P4>/u_tau`, falling back to its
  config and then to the per-pressure scalar. Position map: `close/PH1→P1`,
  `close/PH2→P2`, `far/PH1→P3`, `far/PH2→P4`.
- **bump1** applies `H_smooth` (if present) for the freestream NC correction
  but raw `H_fused` for the wall correction — an asymmetry no other case has.
- **phase1** is the only case whose root responds to
  `PRESSUREPROCESS_ROOT_DIR`, and the only case that falls back to an identity
  NC FRF with a warning when its NC calibration is missing.
- **iso_re** has no plot package.


Running a case
--------------

Dependency order — do not reorder:

```
phase1 PH calibs  →  phase2 full run  →  kernel extraction  →  bump1 / fence
```

`iso_re` is independent of all of the above.

```bash
# ---- phase1 (the PH transfer-function donor) ----
python -m src.save.run_all                    # process
python -m src.checks.plot.run_all             # plots
python -m src.run_pipeline                    # both, for the env-selected root

# ---- phase2 (the Wiener kernel donor) ----
python -m src.save_phase2.run_all
python -m src.checks.plot_phase2.run_all
python -m src.run_phase2_plots                # both
python -m src.checks.plot_phase2.baseline_verification   # not in run_all

# ---- extract the donor Wiener kernels (required before bump1/fence) ----
python -m src.core.wiener_kernel_donor        # → data/phase2/calibration/wiener_kernels.h5

# ---- bump1 ----
python -m src.save_bump.run_all
python -m src.checks.plot_bump.run_all

# ---- fence ----
python -m src.save_fence.run_all
python -m src.checks.plot_fence.run_all

# ---- iso_re (no plot package) ----
python -m src.save_iso_re.run_all

# ---- diagnostics / packaging ----
python -m src.checks.compare_phase1_phase2_PH_tfs   # phase1-vs-phase2 PH TF comparison
python -m src.upload.build_framework_products       # repackages fence + bump1 into report/data/SU/
python report/plot_report.py                        # manuscript figures
```

Note that `src.checks.plot.run_all` selects its plot set from the *basename* of
`ROOT_DIR`: a name starting with `bump` gets the bump plots, anything else gets
the generic set. It reaches into `plot_bump.raw` and `plot_bump.production`
only — the bump calibration, coherence, freestream and wall-stage figures come
from `src.checks.plot_bump.run_all`.


Outputs
-------

```text
ROOT_DIR/
  calibration/PH/calibs_<psig>.h5     frequencies, H1, H2, H_fused, gamma2_fused
  calibration/NC/calibs_<psig>.h5     frequencies, H_fused, H_smooth, gamma2_fused
  pressure/F_freestreamp_SU_raw.hdf5
  pressure/F_freestreamp_SU_production.hdf5
  pressure/G_wallp_SU_raw.hdf5
  pressure/G_wallp_SU_production.hdf5
```

`data/phase2/calibration/wiener_kernels.h5` is written by the separate kernel
extraction step, not by any `run_all`.

### HDF5 structure

```text
G_wallp_SU_production.hdf5
└── wallp_production/
    └── <label>/                             attrs: psig, u_tau, nu, rho, mu,
        │                                           Re_tau, delta, T_K,
        │                                           analog_LP_filter_Hz, Ue_m_per_s
        ├── frf_corrected_signals/<spacing>/{PH1_Pa, PH2_Pa}    float32
        └── fs_noise_rejected_signals/<spacing>/{PH1_Pa, PH2_Pa} float32

F_freestreamp_SU_production.hdf5
└── freestream_production/
    └── <label>/
        ├── FRF_NC_to_nkd/{fcal_Hz, Hcal}
        └── <spacing>/NC_Pa

G_wallp_SU_raw.hdf5
└── wallp_raw/
    └── <label>/
        ├── <spacing>/{PH1_Pa, PH2_Pa}
        └── FRF_PH_to_NC/{Run1/{PH1_Pa,NC_Pa}, Run2/{PH2_Pa,NC_Pa}}
```

Figures are written under `figures/<ROOT_DIR>/…` (e.g. `figures/data/bump1/`),
**except** the calibration TF check plots emitted by the save stage, which go
to `figures/<basename>/calibration/` (e.g. `figures/bump1/calibration/`).


Configuration
-------------

`src/config_params.py` holds settings for the **phase1 pipeline only**. Each
other case has its own `src/save_<case>/config_params.py` with hard-coded
values that ignore the environment.

### Environment variables

| Variable | Read by | Effect |
| --- | --- | --- |
| `PRESSUREPROCESS_ROOT_DIR` | `src/config_params.py` | Dataset root for the **phase1 pipeline and the generic plots only**. Default `data/phase1`. The other four cases ignore it. |
| `PRESSUREPROCESS_PW_DENOISER` | `src/save/pw_proc.py` | `auto` (default), `wiener`, or `hybrid`. Affects **phase1 only** — see below. |
| `PRESSUREPROCESS_USE_TEX` | `src/checks/plot/_style.py` | `auto` (default; uses TeX only if a `latex` binary is on PATH), `1`/`true`/`yes`/`on`, or `0`/`false`/`no`/`off`. Anything else raises `ValueError`. |
| `PRESSUREPROCESS_PROFILE` | `src/config_params.py` | **Dead** — read by `_profile_name()`, which has no callers. Setting it does nothing. |

Under `auto`, `src/save/pw_proc.py` selects the `hybrid` canceller only when
`sys.platform == "darwin"` **and** the signal has ≥ 1,000,000 samples;
otherwise it uses the exact Wiener solver. The check is per-signal, not
per-run. `save_phase2` hard-codes the same platform rule and ignores the
environment variable; `save_iso_re` reads a hard-coded `"auto"`;
`save_bump`/`save_fence` bypass the selector entirely.

> Prefer `PRESSUREPROCESS_PW_DENOISER=wiener` on macOS. The `hybrid` branch has
> a defect — see [Known defects](#known-defects).

### Key config fields

| Field | Notes |
| --- | --- |
| `LABELS`, `PSIGS` | Must describe the same conditions; filenames are built from both. |
| `SPACINGS` | `("close","far")`, or `("close",)` for phase2 and iso_re. Two-point plots are skipped unless both are present. |
| `U_TAU`, `U_TAU_REL_UNC`, `U_E`, `DELTA`, `TDEG` | Per-condition, indexed positionally alongside `LABELS`. |
| `ANALOG_LP_FILTER` | Acquisition anti-alias filter, in Hz. Tunable. |
| `F_CUTS` | `field(init=False)` — **derived** in `__post_init__` from `PSIGS`, `TDEG`, `U_TAU` and `TPLUS_CUT`. Not settable. |
| `RUN_NC_CALIBS`, `INCLUDE_NC_CALIB_RAW` | `False` for phase1 and iso_re; `True` for phase2, bump1 and fence. |
| `SENSITIVITIES_V_PER_PA` | V/Pa per channel. Identical across all five cases today. |
| `FS`, `NPERSEG`, `WINDOW` | 50 kHz, 4096, hann. Identical across cases — spectra are not comparable otherwise. |


Testing
-------

```bash
pytest
```

159 tests, ~4 s, no access to the real dataset required — every test builds
synthetic `.mat` and HDF5 fixtures with the same layout, variable names and
channel ordering the pipelines expect.

| File | Covers |
| --- | --- |
| `test_apply_frf.py` | FRF application: identity, gain, pure delay, out-of-band taper, DC/Nyquist zeroing, float32/float64 agreement |
| `test_tf_definition.py` | H1 estimator against an analytic FIR response, coherence behaviour under added noise, coherence-weighted fusion, `gmin` gating, log-frequency smoothing |
| `test_wiener.py` | Kernel identification against a known FIR path, variance reduction *and* signal preservation, no-op on an uncorrelated reference, donor-kernel equivalence, `alpha`/`preserve_mean` semantics |
| `test_band_limits.py` | Sutherland viscosity against a hand computation, guard-vs-`T+` crossover, monotonicity, per-case `F_CUTS` regression lock |
| `test_configs.py` | Path derivation for all five configs, per-condition tuple lengths, env-override scoping, case-specific labels/spacings |
| `test_case_wiring.py` | Every stage entry point exists; borrowed paths pinned as literals; plot dispatch targets exist |
| `test_end_to_end.py` | Full **phase2 → kernel extraction → bump1** chain on synthetic data; asserts the HDF5 tree, dtypes, finiteness, volts→Pa conversion, thermodynamic consistency, per-position `u_tau`, and that noise rejection reduced variance without destroying the signal |
| `test_plot_dispatch.py` | Dataset-name routing, absolute-path roots, two-point skip for close-only cases |
| `test_plot_style.py` | TeX resolution and figure-directory handling |

Three tests are explicitly labelled `CHARACTERIZATION TEST`. They pin behaviour
that is currently **wrong** so it cannot drift silently; see below.


Known defects
-------------

These are live and verified. None is fixed in code, because each changes
scientific output and the decision to reprocess is yours.

**1. PH calibration is truncated to the first pressure — phase1 and iso_re.**
[`src/save/calibs.py:105`](src/save/calibs.py#L105) and
[`src/save_iso_re/calibs.py:99`](src/save_iso_re/calibs.py#L99) do
`pressures = [int(p) for p in cfg.PSIGS[:1]]`. A clean run regenerates only
`calibs_0.h5`. The 50 and 100 psig files on disk are stale artefacts of an
earlier run — and phase2, bump1 and fence *all borrow all three of phase1's PH
TFs*. Two thirds of the borrowed calibration cannot currently be reproduced.
This is the most consequential item here. For iso_re it is also fatal:
`pw_proc` opens `calibs_0/30/50.h5` with no existence guard, so a clean run
raises at the 30 psig label.

**2. `estimate_frf` returns the complex conjugate of the true FRF.**
[`src/core/tf_definition.py:92`](src/core/tf_definition.py#L92) computes
`H = conj(Sxy)/Sxx` on the stated assumption that
`scipy.signal.csd(x, y) == E{X·conj(Y)}`. SciPy's actual convention is
`E{conj(X)·Y}`, so `Sxy/Sxx` is *already* H1(x→y) and the extra conjugation
negates the phase. Magnitudes — and therefore the reported PSDs — are correct;
group delay and anything phase-derived downstream of an FRF correction are not.
Pinned by `test_estimate_frf_phase_is_the_conjugate_of_the_x_to_y_response`.

**3. The `hybrid` Wiener branch removes only the zero-lag noise term.**
[`src/core/wiener_filter_torch.py:324`](src/core/wiener_filter_torch.py#L324)
has the same conjugation error, which time-reverses the impulse response; the
code then keeps only the first `m` taps, discarding everything except lag 0. On
the test case it recovers the signal at correlation 0.73 versus 0.998 for the
exact solver. Selected by `PRESSUREPROCESS_PW_DENOISER=hybrid`, and by `auto`
for ≥1e6-sample signals on macOS — which real 50 kHz records exceed. Nothing in
the output records which branch ran. Pinned by
`test_hybrid_canceller_only_removes_the_zero_lag_component`.

**4. The NC → nkd transfer function is 0.25 dB low.**
`save_PH_calibs` converts both channels to Pa and applies the
sensitivity-drift correction before estimating the FRF; `save_NC_calibs`
estimates on the **raw volt traces** in all four cases that run it. Since
`H1 = S_xy/S_xx`, scaling `x` by `1/a` and `y` by `1/b` scales `H` by `a/b`,
so the omitted conversion leaves the FRF low by the nkd/NC sensitivity ratio
`52.4e-3 / 50.9e-3 = 1.0295`, i.e. **0.25 dB**. (The drift correction cancels
in the ratio, so only the sensitivity ratio is missing.) It propagates into
the freestream production signal, the wall production signal and the Wiener
reference. `src/save/calibs.py`'s `save_NC_calibs` also carries a docstring
copy-pasted from `save_PH_calibs` claiming a conversion it does not perform.
Pinned by `test_nc_calibration_estimates_the_frf_on_raw_volts`.

**5. Fence raw plots are a silent no-op.**
[`src/checks/plot_fence/raw.py:30`](src/checks/plot_fence/raw.py#L30) pins
`SPACING = "combined"`, but fence's `SPACINGS` is `("close", "far")`. Every
label is skipped and no figure is emitted, without error.

**6. Provenance attributes written into the HDF5 files are wrong.**
phase2's production file records
`"Per-channel PH → NC FRF (H1 on PH1, H2 on PH2, no fusion)"`, and the
calibration files record `application = "per-channel: apply H1_smooth to PH1,
H2_smooth to PH2"` and `"fs_proc/pw_proc apply H_smooth"`. **No case currently
applies per-channel PH TFs** — all four borrowing/own-root cases apply a single
fused `H_fused` to both channels, and the NC correction uses `H_fused`, not
`H_smooth`. These strings are already on disk and will mislead any downstream
consumer. Neither bump1 nor fence records that it borrowed phase1's TF or
phase2's kernel at all.

### Stale comments (harmless, but misleading)

- Four docstrings claim fence uses `SPACINGS = ("combined",)` and reads
  `ATM_Rev1.mat`; it uses `("close","far")` and per-spacing files. The
  `ATM_FILE`/`ATM_PATH`/`ATM_CHANNEL_COLUMNS`/`ATM_NOFLOW_KEY` config fields
  are dead and `data/fence/ATM_Rev1.mat` does not exist.
- bump1's and fence's `pw_proc` warn about "falling through with bandpassed
  signal"; digital band-passing was removed.
- `src/save_iso_re/config_params.py` calls the identity NC FRF a "fallback";
  it is pinned unconditionally.
- `src/checks/plot_bump/production.py` calls itself a "back-compat shim"; it is
  a 451-line implementation.

### Unwired modules

`src/checks/models.py` (boundary-layer / pipe / channel spectral models) and
`src/checks/hdf5_tree.py` (HDF5 tree printer) have no importers. The plot
modules carry their own inline copies of `bl_model` and `channel_model`;
`pipe_model` and `cf_approx` exist only in `models.py`.
`src/upload/build_framework_products.py` packages only fence (B1) and bump1
(B2). `BASELINE_CASE` (phase2/B0) is defined but deliberately not built —
the smooth-wall baseline has no slot in the framework's SU pressure table and
appears only as the drift-verification figure.
