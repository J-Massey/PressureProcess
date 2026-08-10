"""
Baseline drift verification (phase-2 figure-only deliverable).

Overlays the phase-1 smooth-wall wall-pressure spectra against the phase-2
re-acquisition of the *same* smooth-wall baseline, taken after the fence/bump
campaign, on identical inner-scaled axes. Both are the delivered production
(FRF-corrected + Wiener facility-noise-rejected) signals.

Close spacing only (phase-2 is close-only); PH1 and PH2 are shown. The wall is
smooth, so the group-level u_tau is the physical friction velocity and is used
for the inner scaling. Minimal separation between the two campaigns confirms the
smooth-wall reference has not drifted.

Run: python -m src.checks.plot_phase2.baseline_verification
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import welch, get_window

from src.band_limits import f_cut_hz
from src.checks.plot._style import apply_plot_style

apply_plot_style()

FS = 50_000.0
NPERSEG = 2**12          # matches the production wall-pressure plotter
WINDOW = "hann"
LABELS = ("0psig", "50psig", "100psig")
TPLUS_XLIM = (7.0, 7_000.0)
PHI_YLIM = (0.0, 6.0)

PHASE1 = "data/phase1/pressure/G_wallp_SU_production.hdf5"
PHASE2 = "data/phase2/pressure/G_wallp_SU_production.hdf5"
OUT_DIR = Path("figures/phase2")

CAMPAIGNS = ((PHASE1, "phase 1", "-"), (PHASE2, "phase 2", "--"))
CHANNELS = (("PH1_Pa", "PH1 (close)", "#0b4eb2"), ("PH2_Pa", "PH2 (close)", "#d62728"))

# Reported-band upper edge, f_cut = min(pinhole Helmholtz-resonance guard,
# T+ >= TPLUS_CUT spatial-resolution limit); see src/band_limits.py. The cavity
# resonance sits at ~1.4 kHz at EVERY pressurisation (geometry x sound speed --
# it does not scale with Re_tau), so the guard binds at all three conditions.
# The phase-1/phase-2 pipeline files predate the 'f_cut_Hz' attribute carried by
# the upload products, so the cut is computed here from the group attrs.
TPLUS_CUT = 10.0


def _spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = min(NPERSEG, x.size)
    w = get_window(WINDOW, n, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=n, noverlap=n // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


def _smooth_psd_logf(f: np.ndarray, p: np.ndarray, *, span_oct: float, ppo: int):
    """Resample a one-sided PSD onto a log-f grid and smooth with a uniform
    kernel of width span_oct (octaves). Cosmetic only; matches the production
    wall-pressure plotter so this verification figure reads the same way.
    The average is normalised by the window support so the band edges are not
    dragged toward zero by the implicit zero-padding of 'same'-mode."""
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


def _inner(f: np.ndarray, p: np.ndarray, u_tau: float, nu: float, rho: float):
    f, p = _smooth_psd_logf(f, p, span_oct=1 / 2, ppo=96)
    t_plus = (u_tau ** 2) / (nu * f)
    y = (f * p) / (rho ** 2 * u_tau ** 4)
    return t_plus, y


def plot_baseline_verification() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handles = [h5py.File(p, "r") for p, _, _ in CAMPAIGNS]
    try:
        fig, axes = plt.subplots(1, len(LABELS), figsize=(10.5, 3.6),
                                 sharey=True, tight_layout=True)
        for ax, L in zip(axes, LABELS):
            re_tau_smooth = None
            for (path, campaign, ls), hf in zip(CAMPAIGNS, handles):
                g = hf[f"wallp_production/{L}"]
                u_tau = float(np.atleast_1d(g.attrs["u_tau"])[0])
                nu = float(np.atleast_1d(g.attrs["nu"])[0])
                rho = float(np.atleast_1d(g.attrs["rho"])[0])
                fcut = f_cut_hz(u_tau, nu, tplus_cut=TPLUS_CUT)
                if re_tau_smooth is None:
                    re_tau_smooth = float(np.atleast_1d(g.attrs["Re_tau"])[0])
                for ch, _lbl, col in CHANNELS:
                    ds = f"fs_noise_rejected_signals/close/{ch}"
                    if ds not in g:
                        continue
                    f, p = _spec(g[ds][:])
                    keep = (f > 0.0) & (f <= fcut)   # reported band only
                    t_plus, y = _inner(f[keep], p[keep], u_tau, nu, rho)
                    ax.semilogx(t_plus, y, color=col, linestyle=ls, lw=1.1)
            exp = int(np.floor(np.log10(re_tau_smooth)))
            ax.set_title(rf"$Re_\tau^{{\mathrm{{smooth}}}} \approx "
                         rf"{re_tau_smooth / 10**exp:.1f}\times10^{{{exp}}}$")
            ax.set_xlabel(r"$T^+$")
            ax.set_xlim(*TPLUS_XLIM)
            ax.set_ylim(*PHI_YLIM)
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
        axes[0].set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")

        campaign_lines = [Line2D([0], [0], color="black", linestyle=ls)
                          for _, _, ls in CAMPAIGNS]
        axes[0].legend(campaign_lines, [c for _, c, _ in CAMPAIGNS],
                       loc="upper left", fontsize=8, title="campaign",
                       title_fontsize=8)
        ch_lines = [Line2D([0], [0], color=col, linestyle="-")
                    for _, _, col in CHANNELS]
        axes[-1].legend(ch_lines, [lbl for _, lbl, _ in CHANNELS],
                        loc="upper right", fontsize=8, title="channel",
                        title_fontsize=8)

        out = OUT_DIR / "baseline_verification_phase1_vs_phase2.png"
        fig.savefig(out, dpi=600)
        plt.close(fig)
        print(f"[ok] wrote {out}")
        return out
    finally:
        for hf in handles:
            hf.close()


if __name__ == "__main__":
    plot_baseline_verification()
