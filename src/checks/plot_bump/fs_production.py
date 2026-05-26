"""
Production freestream plots for the bump pipeline.

Per pressure (density): PSDs of the freestream NC channel BEFORE (raw, dashed)
and AFTER (production, solid) the NC->nkd FRF correction from fs_proc. One
panel; two spacings (close, far) overlaid so you can see that the FRF acts
consistently across spacing and across psig.
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

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "production"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FS_TRACES = (
    ("close", "#0b4eb2"),
    ("far",   "#d62728"),
)


def _spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nseg = min(NPERSEG, x.size)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=nseg, noverlap=nseg // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


def plot_fs_production() -> None:
    with h5py.File(cfg.NKD_RAW_FILE, "r") as hf_raw, \
         h5py.File(cfg.NKD_PROCESSED_FILE, "r") as hf_prod:
        g_raw = hf_raw["freestream_raw"]
        g_prod = hf_prod["freestream_production"]
        labels = [L for L in g_prod if L in g_raw]
        for label in labels:
            gLr = g_raw[label]
            gLp = g_prod[label]
            fig, ax = plt.subplots(figsize=(4.6, 3.4), tight_layout=True)
            for spacing, color in FS_TRACES:
                if spacing in gLr and "NC_Pa" in gLr[spacing]:
                    fr, pr = _spec(gLr[f"{spacing}/NC_Pa"][:])
                    mr = fr > 0.0
                    ax.loglog(fr[mr], np.sqrt(fr[mr] * pr[mr]),
                              color=color, linewidth=0.8, alpha=0.45,
                              linestyle="--", label=f"{spacing} raw NC")
                if spacing in gLp and "NC_Pa" in gLp[spacing]:
                    fp, pp = _spec(gLp[f"{spacing}/NC_Pa"][:])
                    mp = fp > 0.0
                    ax.loglog(fp[mp], np.sqrt(fp[mp] * pp[mp]),
                              color=color, linewidth=1.0,
                              label=f"{spacing} NC$\\to$nkd")

            ax.set_title(f"bump {label} -- freestream NC raw (dashed) vs production (solid)")
            ax.set_xlabel("$f$ [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.legend(fontsize=7, loc="lower left")
            out = FIG_DIR / f"fs_production_{label}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_all() -> None:
    plot_fs_production()


if __name__ == "__main__":
    plot_all()
