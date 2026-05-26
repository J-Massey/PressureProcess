"""
Raw-spectrum plots for the phase2 pipeline.

Per pressure:
  - Wall pressure: raw PH1/PH2 at the spacings present in cfg.SPACINGS.
  - Freestream: raw NC at the same spacings.
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

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "raw"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPACING_COLORS = {
    "close": ("#0b4eb2", "#d62728"),
    "far":   ("#26bd26", "#a51990"),
}


def _spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nseg = min(NPERSEG, x.size)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=nseg, noverlap=nseg // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


def plot_wall_raw() -> None:
    with h5py.File(cfg.PH_RAW_FILE, "r") as hf:
        g = hf["wallp_raw"]
        for label in g:
            gL = g[label]
            fig, ax = plt.subplots(figsize=(4.2, 3.4), tight_layout=True)
            for spacing in cfg.SPACINGS:
                if spacing not in gL:
                    continue
                colors = SPACING_COLORS.get(spacing, ("#1f77b4", "#ff7f0e"))
                for channel, color in (("PH1_Pa", colors[0]), ("PH2_Pa", colors[1])):
                    if channel not in gL[spacing]:
                        continue
                    f, p = _spec(gL[f"{spacing}/{channel}"][:])
                    mask = f > 0.0
                    name = f"{spacing} {channel.replace('_Pa','')}"
                    ax.loglog(f[mask], np.sqrt(f[mask] * p[mask]),
                              color=color, linewidth=1.0, label=name)
            ax.set_title(f"phase2 {label} -- wall raw")
            ax.set_xlabel("$f$ [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.legend(fontsize=8)
            out = FIG_DIR / f"wall_raw_{label}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_fs_raw() -> None:
    with h5py.File(cfg.NKD_RAW_FILE, "r") as hf:
        g = hf["freestream_raw"]
        for label in g:
            gL = g[label]
            fig, ax = plt.subplots(figsize=(4.2, 3.4), tight_layout=True)
            for spacing in cfg.SPACINGS:
                if spacing not in gL or "NC_Pa" not in gL[spacing]:
                    continue
                color = SPACING_COLORS.get(spacing, ("#1f77b4",))[0]
                f, p = _spec(gL[f"{spacing}/NC_Pa"][:])
                mask = f > 0.0
                ax.loglog(f[mask], np.sqrt(f[mask] * p[mask]),
                          color=color, linewidth=1.0, label=f"{spacing} NC")
            ax.set_title(f"phase2 {label} -- freestream raw (NC)")
            ax.set_xlabel("$f$ [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.legend(fontsize=8)
            out = FIG_DIR / f"fs_raw_{label}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_all() -> None:
    plot_wall_raw()
    plot_fs_raw()


if __name__ == "__main__":
    plot_all()
