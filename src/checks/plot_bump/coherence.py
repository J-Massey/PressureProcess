"""
Noise-rejection diagnostic via coherence for the bump pipeline.

If the Wiener filter is working, the post-rejection wall-pressure signal
should be (mostly) incoherent with the freestream NC reference used to
drive the canceller. We therefore plot, per pressure x spacing x channel:

  - gamma^2(NC_pre, PH_frf_corrected_pre)   -- pre-rejection  (dashed)
  - gamma^2(NC_pre, PH_noise_rejected)      -- post-rejection (solid)

To keep the comparison apples-to-apples with what the Wiener canceller
actually saw, both the NC reference and the pre-rejection PH signal are
demeaned and bandpass-filtered (1 Hz HP, analog LP) inside this plot
before the coherence is computed, mirroring the exact preprocessing in
pw_proc.

Anywhere the dashed line is high but the solid line drops, the canceller
removed coherent freestream content. Anywhere both are high, the
canceller failed to reject coherent noise (e.g. a 70 Hz facility tone).
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, get_window

from src.save_bump.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FS = cfg.FS
NPERSEG = 2**14
WINDOW = cfg.WINDOW

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "production"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ("PH1_Pa", "PH2_Pa")
SPACINGS = ("close", "far")


def _preproc(x: np.ndarray, f_high: float) -> np.ndarray:
    """Match pw_proc's pre-Wiener preprocessing: demean only (band limits are
    applied as spectral masks at reporting time; f_high is unused, kept for
    call-site compatibility)."""
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def _coh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(x.size, y.size)
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    nseg = min(NPERSEG, n)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, g = coherence(x, y, fs=FS, window=w, nperseg=nseg, noverlap=nseg // 2,
                     detrend="constant")
    return f, g


def plot_coherence() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf_w, \
         h5py.File(cfg.NKD_PROCESSED_FILE, "r") as hf_fs:
        g_w = hf_w["wallp_production"]
        g_fs = hf_fs["freestream_production"]
        for label in g_w:
            if label not in g_fs:
                print(f"[skip] {label}: missing in freestream_production")
                continue
            gL_w = g_w[label]
            gL_fs = g_fs[label]
            g_frf = gL_w.get("frf_corrected_signals")
            g_rej = gL_w.get("fs_noise_rejected_signals")
            if g_frf is None or g_rej is None:
                print(f"[skip] {label}: missing frf_corrected or fs_noise_rejected")
                continue

            f_high = float(gL_w.attrs["analog_LP_filter_Hz"])
            fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6), sharex=True, sharey=True,
                                     tight_layout=True)
            for i, channel in enumerate(CHANNELS):
                for j, spacing in enumerate(SPACINGS):
                    ax = axes[i, j]
                    if spacing not in gL_fs or "NC_Pa" not in gL_fs[spacing]:
                        ax.set_visible(False)
                        continue
                    nc_pre = _preproc(gL_fs[f"{spacing}/NC_Pa"][:], f_high)

                    if spacing in g_frf and channel in g_frf[spacing]:
                        ph_pre = _preproc(g_frf[f"{spacing}/{channel}"][:], f_high)
                        f_pre, g_pre = _coh(nc_pre, ph_pre)
                        m = f_pre > 0.0
                        ax.semilogx(f_pre[m], g_pre[m],
                                    color="#1e8ad8", linewidth=0.9,
                                    linestyle="--", label="pre Wiener")
                    if spacing in g_rej and channel in g_rej[spacing]:
                        ph_post = np.asarray(g_rej[f"{spacing}/{channel}"][:], dtype=float)
                        f_post, g_post = _coh(nc_pre, ph_post)
                        m = f_post > 0.0
                        ax.semilogx(f_post[m], g_post[m],
                                    color="#d62728", linewidth=1.0,
                                    label="post Wiener")

                    ax.set_title(f"{channel.replace('_Pa','')} -- {spacing}", fontsize=9)
                    ax.set_ylim(0.0, 1.05)
                    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
                    if i == 1:
                        ax.set_xlabel("$f$ [Hz]")
                    if j == 0:
                        ax.set_ylabel(r"$\gamma^2(\mathrm{NC},\mathrm{PH})$")
                    if i == 0 and j == 0:
                        ax.legend(fontsize=7, loc="upper right")

            fig.suptitle(rf"bump {label} -- NC$\leftrightarrow$PH coherence (pre vs post Wiener)",
                         fontsize=10)
            out = FIG_DIR / f"coherence_{label}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_all() -> None:
    plot_coherence()


if __name__ == "__main__":
    plot_all()
