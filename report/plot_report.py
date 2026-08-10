"""
Single-source report plotting and final consistency check.

This module reads ONLY the framework upload products in ``report/data/SU/`` --
the exact files that will be uploaded, kept inside the report directory so the
report is fully self-contained -- and regenerates every figure in the results
section from them. If these plots reproduce the manuscript figures, the
uploaded data is self-consistent with the report.

    B/C -> B1 (fence)     E/F -> B2 (bump)
    <letter>_<id>_wallp_SU_raw.hdf5         -> raw wall pressure     (dimensional, Pa)
    <letter>_<id>_wallp_SU_production.hdf5  -> production wall press. (inner-scaled)
    <letter>_<id>_freestreamp_SU_production.hdf5 -> freestream        (dimensional, Pa)

The one exception is the smooth-wall baseline-drift figure, which is figure-only
(no slot in the SU table): it compares the phase-1 and phase-2 baselines and so
reads the phase-1/phase-2 pipeline files directly.

Figure layout (wall pressure): one panel per Reynolds condition, titled with the
smooth-wall reference Re_tau; all four pinhole stations (P1..P4, ordered by
their streamwise position x) in each panel.

Normalisation (see the results section):
* production wall pressure -- premultiplied, inner-scaled:
      abscissa T+ = u_tau^2/(nu f),  ordinate f*phi/(rho^2 u_tau^4),
      using the measured per-station u_tau on each dataset.
* raw wall pressure and freestream -- dimensional PSD phi_pp [Pa^2/Hz] vs
      f [Hz] on log-log axes, NOT premultiplied (premultiplication is reserved
      for the semilog inner-scaled figures; inner scaling is not meaningful for
      an uncalibrated or non-wall signal).

Band limiting: every wall-pressure curve stops at its per-station 'f_cut_Hz'
attribute, f_cut = min(pinhole Helmholtz-resonance guard = 1.2 kHz, spatial-
resolution limit T+ = u_tau^2/(nu f) >= 10). The cavity resonance sits at
~1.4 kHz at EVERY pressurisation (geometry x sound speed, not viscous units),
so the resonance guard binds almost everywhere. The nose-cone freestream mic is
not pinhole-mounted and is not clipped.

Run from the repo root:  python report/plot_report.py
Figures are written next to this file; the quantitative numbers quoted in the
results section (rejection variance fractions, baseline drift) are printed to
stdout.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import welch, get_window

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data" / "SU"

FS = 50_000.0
WINDOW = "hann"

RE_ORDER = ("Re1", "Re2", "Re3")
RE_TITLE = {"Re1": "0 psig", "Re2": "50 psig", "Re3": "100 psig"}
RE_COLOUR = {"Re1": "#1e8ad8", "Re2": "#ff7f0e", "Re3": "#26bd26"}

# Station colours: P1..P4 are holder positions; curves are drawn ordered by the
# streamwise position x_m (P4, P1, P2, P3), not by station number.
STATION_COLOUR = {"P1": "#0b4eb2", "P2": "#d62728", "P3": "#2ca02c", "P4": "#9467bd"}

# (wall-pressure letter, freestream letter, framework id, case title)
CASES = (
    ("B", "C", "B1", r"Case B1: $2\delta$ protrusions (fence)"),
    ("E", "F", "B2", "Case B2: Eaton bump"),
)

# Reported-band edge, mirrored from src/band_limits.py for the baseline figure
# (the phase-1/phase-2 pipeline files predate the 'f_cut_Hz' attribute carried
# by the upload products): f_cut = min(Helmholtz guard, T+ >= TPLUS_CUT limit).
F_HELMHOLTZ_GUARD_HZ = 1200.0
TPLUS_CUT = 10.0


def _f_cut_from_scales(u_tau: float, nu: float) -> float:
    return float(min(F_HELMHOLTZ_GUARD_HZ, u_tau ** 2 / (nu * TPLUS_CUT)))


def _attr(obj, key: str) -> float:
    return float(np.atleast_1d(obj.attrs[key])[0])


def _fcut(dset: h5py.Dataset, cond: h5py.Group) -> float:
    """Per-station reported-band edge; dataset attr, else the condition attr."""
    if "f_cut_Hz" in dset.attrs:
        return _attr(dset, "f_cut_Hz")
    return _attr(cond, "f_cut_Hz")


def _station_id(dset: h5py.Dataset) -> str:
    v = dset.attrs["station_id"]
    return v.decode() if isinstance(v, bytes) else str(v)


def _fmt_re_tau_smooth(re_tau: float) -> str:
    exp = int(np.floor(np.log10(re_tau)))
    mant = re_tau / 10 ** exp
    return rf"$Re_\tau^{{\mathrm{{smooth}}}} \approx {mant:.1f}\times10^{{{exp}}}$"


def _apply_style() -> None:
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "grid"])
    except Exception:
        plt.style.use("default")
    # usetex off keeps the directory runnable without a LaTeX install; mathtext
    # renders the sqrt/subscript labels fine.
    plt.rcParams.update({"font.size": 10.5, "text.usetex": False,
                         "axes.grid": True, "grid.linestyle": "--",
                         "grid.linewidth": 0.4, "grid.alpha": 0.7})


def _spec(x: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = min(nperseg, x.size)
    w = get_window(WINDOW, n, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=n, noverlap=n // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


def _smooth_psd_logf(f: np.ndarray, p: np.ndarray, *,
                     span_oct: float = 0.5, ppo: int = 96):
    """Cosmetic log-f smoothing. The moving average is normalised by the number
    of points actually inside the window, so the band edges are NOT dragged
    toward zero (a plain 'same'-mode convolution zero-pads beyond the edges,
    which faked a roll-off over the last ~half window before f_cut)."""
    pos = f > 0.0
    if not np.any(pos):
        return f, p
    fpos, ppos = f[pos], p[pos]
    f_lo, f_hi = float(max(fpos[0], 1e-12)), float(fpos[-1])
    n_pts = max(int(np.ceil(np.log2(f_hi / f_lo) * ppo)), 8)
    fgrid = 2.0 ** np.linspace(np.log2(f_lo), np.log2(f_hi), n_pts)
    pgrid = np.interp(fgrid, fpos, ppos)
    wlen = max(int(round(span_oct * ppo)), 1)
    if wlen % 2 == 0:
        wlen += 1
    ker = np.ones(wlen)
    support = np.convolve(np.ones_like(pgrid), ker, mode="same")
    return fgrid, np.convolve(pgrid, ker, mode="same") / support


def _wall_stations(cond: h5py.Group, stage: str | None = None):
    """Yield (station_id, x_m, dataset) for all pinhole stations of a condition,
    ordered by streamwise position x_m (P1..P4 are holder labels, not an x
    ordering). stage=None reads the raw layout (<spacing>/<ch>_Pa)."""
    entries = []
    base = cond if stage is None else cond[stage]
    for sp in ("close", "far"):
        if sp not in base:
            continue
        for ch in ("PH1", "PH2"):
            name = f"{sp}/{ch}_Pa"
            if name not in base:
                continue
            d = base[name]
            entries.append((_attr(d, "x_m"), _station_id(d), d))
    entries.sort(key=lambda e: e[0])
    return [(st, x, d) for x, st, d in entries]


# --------------------------------------------------------------------------- #
#  Raw wall pressure: 3 panels (per condition), 4 stations each, dimensional
# --------------------------------------------------------------------------- #
def _plot_wallp_raw(path: Path, title: str, out: Path,
                    nperseg: int = 2 ** 14) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                             sharey=True, tight_layout=True)
    with h5py.File(path, "r") as h:
        for ax, re_id in zip(axes, RE_ORDER):
            cond = h[f"Observations/{re_id}"]
            for st, x_m, d in _wall_stations(cond):
                f, p = _spec(d[:], nperseg)
                m = (f > 0.0) & (f <= _fcut(d, cond))   # reported band only
                ax.loglog(f[m], p[m],
                          color=STATION_COLOUR[st], lw=0.9,
                          label=rf"{st} ($x={x_m*1e3:.0f}$ mm)")
            ax.set_title(_fmt_re_tau_smooth(
                _attr(cond, "Re_tau_reference_smooth_wall")))
            ax.set_xlabel(r"$f$ [Hz]")
        axes[0].set_ylabel(r"$\phi_{pp}$ [Pa$^2$/Hz]")
        axes[0].legend(fontsize=7, loc="upper left", title="station",
                       title_fontsize=7)
    fig.suptitle(title, y=1.02)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}  <-  {path.name}")


# --------------------------------------------------------------------------- #
#  Production wall pressure: 3 panels, 4 stations, inner-scaled premultiplied
# --------------------------------------------------------------------------- #
def _plot_wallp_production(path: Path, title: str, out: Path,
                           nperseg: int = 2 ** 12) -> None:
    """Inner-scaled premultiplied production (Wiener-rejected) wall pressure,
    per-station u_tau. The pre-rejection state is the companion raw figure."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                             sharey=True, tight_layout=True)
    with h5py.File(path, "r") as h:
        for ax, re_id in zip(axes, RE_ORDER):
            cond = h[f"Observations/{re_id}"]
            nu = _attr(cond, "nu")
            rho = _attr(cond, "rho")
            for st, x_m, d in _wall_stations(cond, "fs_noise_rejected_signals"):
                u_tau = _attr(d, "u_tau")
                f, p = _spec(d[:], nperseg)
                keep = (f > 0.0) & (f <= _fcut(d, cond))  # band-limit, then smooth
                f_s, p_s = _smooth_psd_logf(f[keep], p[keep])
                ax.semilogx((u_tau ** 2) / (nu * f_s),
                            (f_s * p_s) / (rho ** 2 * u_tau ** 4),
                            color=STATION_COLOUR[st], lw=1.1,
                            label=rf"{st} ($x={x_m*1e3:.0f}$ mm)")
            ax.set_title(_fmt_re_tau_smooth(
                _attr(cond, "Re_tau_reference_smooth_wall")))
            ax.set_xlabel(r"$T^+$")
            ax.set_xlim(7, 7_000)
        axes[0].set_ylabel(r"$(f\,\phi_{pp}^{+})_{\mathrm{corr.}}$")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(fontsize=7, loc="upper right", title="station",
                       title_fontsize=7)
    fig.suptitle(title, y=1.02)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out.name}  <-  {path.name}")


# --------------------------------------------------------------------------- #
#  Dimensional freestream spectra (nose-cone mic; not pinhole-mounted -> no cut)
# --------------------------------------------------------------------------- #
def _plot_freestream(path: Path, title: str, out: Path,
                     nperseg: int = 2 ** 14) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2), tight_layout=True)
    with h5py.File(path, "r") as h:
        for re_id in RE_ORDER:
            cond = h[f"Observations/{re_id}"]
            if "close/NC_Pa" not in cond:
                continue
            f, p = _spec(cond["close/NC_Pa"][:], nperseg)
            m = f > 0.0
            ax.loglog(f[m], p[m],
                      color=RE_COLOUR[re_id], lw=1.0, label=RE_TITLE[re_id])
    ax.set_title(title)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$\phi_{pp}$ [Pa$^2$/Hz]")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"[ok] {out.name}  <-  {path.name}")


# --------------------------------------------------------------------------- #
#  Baseline drift (figure-only): phase-1 vs phase-2 smooth wall
# --------------------------------------------------------------------------- #
def _plot_baseline(out: Path, nperseg: int = 2 ** 12) -> None:
    phase1 = ROOT / "data/phase1/pressure/G_wallp_SU_production.hdf5"
    phase2 = ROOT / "data/phase2/pressure/G_wallp_SU_production.hdf5"
    labels = ("0psig", "50psig", "100psig")
    campaigns = ((phase1, "phase 1", "-"), (phase2, "phase 2", "--"))
    channels = (("PH1_Pa", "PH1 (close)", "#0b4eb2"),
                ("PH2_Pa", "PH2 (close)", "#d62728"))
    handles = [h5py.File(p, "r") for p, _, _ in campaigns]
    try:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6),
                                 sharey=True, tight_layout=True)
        for ax, L in zip(axes, labels):
            for (_, _, ls), hf in zip(campaigns, handles):
                g = hf[f"wallp_production/{L}"]
                nu = _attr(g, "nu")
                rho = _attr(g, "rho")
                u_tau = _attr(g, "u_tau")     # smooth wall: physical everywhere
                fcut = _f_cut_from_scales(u_tau, nu)
                for ch, _lbl, col in channels:
                    ds = f"fs_noise_rejected_signals/close/{ch}"
                    if ds not in g:
                        continue
                    f, p = _spec(g[ds][:], nperseg)
                    keep = (f > 0.0) & (f <= fcut)
                    f_s, p_s = _smooth_psd_logf(f[keep], p[keep])
                    ax.semilogx((u_tau ** 2) / (nu * f_s),
                                (f_s * p_s) / (rho ** 2 * u_tau ** 4),
                                color=col, linestyle=ls, lw=1.1)
            g = handles[0][f"wallp_production/{L}"]
            ax.set_title(_fmt_re_tau_smooth(_attr(g, "Re_tau")))
            ax.set_xlabel(r"$T^+$")
            ax.set_xlim(7, 7_000)
            ax.set_ylim(0, 6)
        axes[0].set_ylabel(r"$(f\,\phi_{pp}^{+})_{\mathrm{corr.}}$")
        axes[0].legend([Line2D([0], [0], color="black", ls=ls)
                        for _, _, ls in campaigns],
                       [c for _, c, _ in campaigns],
                       loc="upper left", fontsize=8, title="campaign")
        axes[-1].legend([Line2D([0], [0], color=col, ls="-")
                         for _, _, col in channels],
                        [lbl for _, lbl, _ in channels],
                        loc="upper right", fontsize=8, title="channel")
        fig.savefig(out, dpi=600)
        plt.close(fig)
        print(f"[ok] {out.name}  <-  phase1/phase2 pipeline (figure-only)")
    finally:
        for hf in handles:
            hf.close()


# --------------------------------------------------------------------------- #
#  Quantitative numbers quoted in the results section
# --------------------------------------------------------------------------- #
def _band_var(f: np.ndarray, p: np.ndarray, f_lo: float, f_hi: float) -> float:
    m = (f >= f_lo) & (f <= f_hi)
    return float(np.trapezoid(p[m], f[m]))


def _rejection_stats(path: Path, fid: str, nperseg: int = 2 ** 14) -> None:
    """Fraction of wall-pressure variance removed by the smooth-wall Wiener
    rejection, within the reported band and within the sub-100 Hz band."""
    print(f"\n--- {fid}: variance removed by the facility-noise rejection "
          f"({path.name}) ---")
    with h5py.File(path, "r") as h:
        for re_id in RE_ORDER:
            cond = h[f"Observations/{re_id}"]
            rows = []
            for st, _x, d_rej in _wall_stations(cond, "fs_noise_rejected_signals"):
                sp_ch = d_rej.name.split("/")[-2:]
                d_cor = cond[f"frf_corrected_signals/{sp_ch[0]}/{sp_ch[1]}"]
                fcut = _fcut(d_rej, cond)
                f_r, p_r = _spec(d_rej[:], nperseg)
                f_c, p_c = _spec(d_cor[:], nperseg)
                band = 1.0 - (_band_var(f_r, p_r, 2.0, fcut) /
                              _band_var(f_c, p_c, 2.0, fcut))
                low = 1.0 - (_band_var(f_r, p_r, 2.0, 100.0) /
                             _band_var(f_c, p_c, 2.0, 100.0))
                rows.append((st, band, low))
            msg = "  ".join(f"{st}: band {b*100:4.0f}%  <100Hz {l*100:4.0f}%"
                            for st, b, l in rows)
            print(f"  {re_id} ({RE_TITLE[re_id]:>8}): {msg}")


def _baseline_stats(nperseg: int = 2 ** 14) -> None:
    """Phase-1 vs phase-2 smooth-wall drift: reported-band variance ratio and
    median inner-spectrum deviation."""
    print("\n--- baseline drift: phase 2 vs phase 1 (reported band) ---")
    p1 = ROOT / "data/phase1/pressure/G_wallp_SU_production.hdf5"
    p2 = ROOT / "data/phase2/pressure/G_wallp_SU_production.hdf5"
    with h5py.File(p1, "r") as h1, h5py.File(p2, "r") as h2:
        for L in ("0psig", "50psig", "100psig"):
            g1, g2 = h1[f"wallp_production/{L}"], h2[f"wallp_production/{L}"]
            u_tau, nu = _attr(g1, "u_tau"), _attr(g1, "nu")
            fcut = _f_cut_from_scales(u_tau, nu)
            for ch in ("PH1_Pa", "PH2_Pa"):
                ds = f"fs_noise_rejected_signals/close/{ch}"
                if ds not in g1 or ds not in g2:
                    continue
                f1, s1 = _spec(g1[ds][:], nperseg)
                f2, s2 = _spec(g2[ds][:], nperseg)
                v1 = _band_var(f1, s1, 2.0, fcut)
                v2 = _band_var(f2, s2, 2.0, fcut)
                fg1, sg1 = _smooth_psd_logf(*(lambda m: (f1[m], s1[m]))((f1 > 0) & (f1 <= fcut)))
                fg2, sg2 = _smooth_psd_logf(*(lambda m: (f2[m], s2[m]))((f2 > 0) & (f2 <= fcut)))
                sg2i = np.interp(fg1, fg2, sg2)
                dev = np.abs(sg2i - sg1) / sg1
                print(f"  {L:>7} {ch[:3]}: var ratio p2/p1 = {v2/v1:5.2f}   "
                      f"median |dPhi|/Phi = {np.median(dev)*100:4.1f}%   "
                      f"p90 = {np.percentile(dev, 90)*100:4.1f}%")


def main() -> None:
    _apply_style()
    for wall, fs, fid, title in CASES:
        _plot_wallp_raw(
            DATA / f"{wall}_{fid}_wallp_SU_raw.hdf5",
            f"{fid} wall pressure (raw)", HERE / f"{fid}_wallp_raw.png")
        _plot_wallp_production(
            DATA / f"{wall}_{fid}_wallp_SU_production.hdf5",
            f"{fid} wall pressure (production)", HERE / f"{fid}_wallp_production.png")
        _plot_freestream(
            DATA / f"{fs}_{fid}_freestreamp_SU_production.hdf5",
            f"{fid} freestream pressure", HERE / f"{fid}_freestream.png")
    _plot_baseline(HERE / "baseline_verification.png")
    for wall, _fs, fid, _title in CASES:
        _rejection_stats(DATA / f"{wall}_{fid}_wallp_SU_production.hdf5", fid)
    _baseline_stats()
    print("\nAll report figures regenerated from the upload products "
          "(baseline from phase1/phase2). Consistency check complete.")


if __name__ == "__main__":
    main()
