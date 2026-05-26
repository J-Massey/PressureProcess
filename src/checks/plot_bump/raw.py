"""
Raw-spectrum plots for the bump pipeline.

Per pressure (density):
  - Wall pressure: raw PH1/PH2 at "close" and "far" spacings (4 traces).
  - Freestream: raw NC at "close" and "far" spacings (2 traces).
Uses the save_bump config (per-case ROOT_DIR and file paths).
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window

from src.save_bump.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FS = cfg.FS
NPERSEG = 2**14
WINDOW = cfg.WINDOW

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "raw"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WALL_TRACES = (
    ("close", "PH1_Pa", "P1 (close, PH1)", "#0b4eb2"),
    ("close", "PH2_Pa", "P2 (close, PH2)", "#d62728"),
    ("far",   "PH1_Pa", "P3 (far,  PH1)", "#26bd26"),
    ("far",   "PH2_Pa", "P4 (far,  PH2)", "#a51990"),
)
FS_TRACES = (
    ("close", "NC_Pa", "close NC", "#0b4eb2"),
    ("far",   "NC_Pa", "far  NC", "#d62728"),
)


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
            for spacing, channel, name, color in WALL_TRACES:
                if spacing not in gL or channel not in gL[spacing]:
                    continue
                f, p = _spec(gL[f"{spacing}/{channel}"][:])
                mask = f > 0.0
                ax.loglog(f[mask], np.sqrt(f[mask] * p[mask]),
                          color=color, linewidth=1.0, label=name)
            ax.set_title(f"bump {label} -- wall raw")
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
            for spacing, channel, name, color in FS_TRACES:
                if spacing not in gL or channel not in gL[spacing]:
                    continue
                f, p = _spec(gL[f"{spacing}/{channel}"][:])
                mask = f > 0.0
                ax.loglog(f[mask], np.sqrt(f[mask] * p[mask]),
                          color=color, linewidth=1.0, label=name)
            ax.set_title(f"bump {label} -- freestream raw (NC)")
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
