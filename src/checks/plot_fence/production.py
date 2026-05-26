"""
Production plots for the fence pipeline.

Per pressure (density): PSDs of the FRF-corrected wall pressure. The
`fs_noise_rejected_signals` group holds the bandpassed (1 Hz, analog-LP)
FRF-corrected signal -- Wiener noise rejection against the freestream
reference is currently disabled in pw_proc.py. Two traces (PH1, PH2 in the
"combined" spacing); full-bandwidth `frf_corrected_signals` shown dashed
for reference.
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window

from src.save_fence.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FS = cfg.FS
NPERSEG = 2**14
WINDOW = cfg.WINDOW

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "production"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPACING = "combined"
WALL_TRACES = (
    ("PH1_Pa", "PH1", "#0b4eb2"),
    ("PH2_Pa", "PH2", "#d62728"),
)


def _spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nseg = min(NPERSEG, x.size)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=nseg, noverlap=nseg // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


def plot_noise_rejected() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g = hf["wallp_production"]
        for label in g:
            gL = g[label]
            if "fs_noise_rejected_signals" not in gL:
                print(f"[skip] {label}: no fs_noise_rejected_signals group")
                continue
            g_rej = gL["fs_noise_rejected_signals"]
            g_pre = gL.get("frf_corrected_signals")

            fig, ax = plt.subplots(figsize=(4.6, 3.4), tight_layout=True)
            for channel, name, color in WALL_TRACES:
                if SPACING not in g_rej or channel not in g_rej[SPACING]:
                    continue
                f_rej, p_rej = _spec(g_rej[f"{SPACING}/{channel}"][:])
                mask = f_rej > 0.0
                ax.loglog(f_rej[mask], np.sqrt(f_rej[mask] * p_rej[mask]),
                          color=color, linewidth=1.0, label=name)

                if g_pre is not None and SPACING in g_pre and channel in g_pre[SPACING]:
                    f_pre, p_pre = _spec(g_pre[f"{SPACING}/{channel}"][:])
                    mp = f_pre > 0.0
                    ax.loglog(f_pre[mp], np.sqrt(f_pre[mp] * p_pre[mp]),
                              color=color, linewidth=0.7, alpha=0.35, linestyle="--")

            ax.set_title(f"fence {label} -- wall pressure (bandpassed solid, full-bw dashed)")
            ax.set_xlabel("$f$ [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.legend(fontsize=8)
            out = FIG_DIR / f"noise_rejected_{label}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_all() -> None:
    plot_noise_rejected()


if __name__ == "__main__":
    plot_all()
