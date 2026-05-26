"""
Wall-pressure end-to-end stage overlay for the phase2 pipeline.

For each pressure, a grid (rows = PH1/PH2, cols = cfg.SPACINGS) showing the
three pw_proc stages on one axis per channel/spacing:
  1. raw            -- straight from pw_raw (wallp_raw)
  2. frf_corrected  -- after PH->NC (per-channel) then NC->nkd FRFs
  3. noise_rejected -- after detrend + bandpass + Wiener cancellation
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window

from src.save_phase2.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FS = cfg.FS
NPERSEG = 2**14
WINDOW = cfg.WINDOW

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "production"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ("PH1_Pa", "PH2_Pa")

STAGE_STYLE = {
    "raw":            ("#888888", 0.8, "--"),
    "frf_corrected":  ("#1e8ad8", 1.0, "-"),
    "noise_rejected": ("#d62728", 1.0, "-"),
}


def _spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nseg = min(NPERSEG, x.size)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=nseg, noverlap=nseg // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


def _draw(ax, x: np.ndarray, stage: str) -> None:
    color, lw, ls = STAGE_STYLE[stage]
    f, p = _spec(x)
    m = f > 0.0
    ax.loglog(f[m], np.sqrt(f[m] * p[m]),
              color=color, linewidth=lw, linestyle=ls, label=stage)


def plot_wall_stages() -> None:
    spacings = tuple(cfg.SPACINGS)
    ncols = max(1, len(spacings))
    with h5py.File(cfg.PH_RAW_FILE, "r") as hf_raw, \
         h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf_prod:
        g_raw = hf_raw["wallp_raw"]
        g_prod = hf_prod["wallp_production"]
        labels = [L for L in g_prod if L in g_raw]
        for label in labels:
            gLr = g_raw[label]
            gLp = g_prod[label]
            g_frf = gLp.get("frf_corrected_signals")
            g_rej = gLp.get("fs_noise_rejected_signals")
            if g_frf is None or g_rej is None:
                print(f"[skip] {label}: missing frf_corrected or fs_noise_rejected groups")
                continue

            fig, axes = plt.subplots(2, ncols, figsize=(3.8 * ncols, 5.6),
                                     sharex=True, sharey=True, tight_layout=True,
                                     squeeze=False)
            for i, channel in enumerate(CHANNELS):
                for j, spacing in enumerate(spacings):
                    ax = axes[i, j]
                    raw_ok = spacing in gLr and channel in gLr[spacing]
                    frf_ok = spacing in g_frf and channel in g_frf[spacing]
                    rej_ok = spacing in g_rej and channel in g_rej[spacing]

                    if raw_ok:
                        _draw(ax, gLr[f"{spacing}/{channel}"][:], "raw")
                    if frf_ok:
                        _draw(ax, g_frf[f"{spacing}/{channel}"][:], "frf_corrected")
                    if rej_ok:
                        _draw(ax, g_rej[f"{spacing}/{channel}"][:], "noise_rejected")

                    ax.set_title(f"{channel.replace('_Pa','')} -- {spacing}", fontsize=9)
                    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
                    if i == 1:
                        ax.set_xlabel("$f$ [Hz]")
                    if j == 0:
                        ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
                    if i == 0 and j == 0:
                        ax.legend(fontsize=7, loc="lower left")

            fig.suptitle(f"phase2 {label} -- wall pipeline stages", fontsize=10)
            out = FIG_DIR / f"wall_stages_{label}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_all() -> None:
    plot_wall_stages()


if __name__ == "__main__":
    plot_all()
