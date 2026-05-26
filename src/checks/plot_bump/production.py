"""
Inner-normalised production plots for the bump pipeline.

Per pressure: pre-multiplied PSDs of the FRF-corrected wall pressure on
inner-scaled axes (same scaling as src/checks/plot/G_wallp_SU_production.py).

  x = T^+ = u_tau^2 / (f * nu)        (logarithmic, inner period)
  y = f * phi_pp / (rho^2 * u_tau^4)  (linear, dimensionless)

`fs_noise_rejected_signals` holds the bandpassed (1 Hz, analog-LP)
FRF-corrected signal -- Wiener noise rejection against the freestream
reference is currently disabled in pw_proc.py. The full-bandwidth
`frf_corrected_signals` are still drawn dashed at lower alpha so the
out-of-band content is visible. Four wall traces P1..P4 (close PH1,
close PH2, far PH1, far PH2).
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

POINT_TRACES = (
    ("close", "PH1_Pa", "P1 (close, PH1)", "#0b4eb2"),
    ("close", "PH2_Pa", "P2 (close, PH2)", "#d62728"),
    ("far",   "PH1_Pa", "P3 (far,  PH1)", "#26bd26"),
    ("far",   "PH2_Pa", "P4 (far,  PH2)", "#a51990"),
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

            rho = float(gL.attrs["rho"])
            u_tau = float(gL.attrs["u_tau"])
            nu = float(gL.attrs["nu"])
            t_to_tplus = (u_tau * u_tau) / nu
            phi_norm = 1.0 / (rho * rho * u_tau**4)

            fig, ax = plt.subplots(figsize=(4.8, 3.4), tight_layout=True)
            for spacing, channel, name, color in POINT_TRACES:
                if spacing not in g_rej or channel not in g_rej[spacing]:
                    continue
                f_rej, p_rej = _spec(g_rej[f"{spacing}/{channel}"][:])
                mask = f_rej > 0.0
                ax.semilogx(t_to_tplus / f_rej[mask],
                            f_rej[mask] * p_rej[mask] * phi_norm,
                            color=color, linewidth=1.0, label=name)

                if g_pre is not None and spacing in g_pre and channel in g_pre[spacing]:
                    f_pre, p_pre = _spec(g_pre[f"{spacing}/{channel}"][:])
                    mp = f_pre > 0.0
                    ax.semilogx(t_to_tplus / f_pre[mp],
                                f_pre[mp] * p_pre[mp] * phi_norm,
                                color=color, linewidth=0.7, alpha=0.35, linestyle="--")

            ax.set_title(f"bump {label}")
            ax.set_xlabel(r"$T^+$")
            ax.set_ylabel(r"$f \phi_{pp} / (\rho^2 u_\tau^4)$")
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.legend(fontsize=7, loc="upper right")
            out = FIG_DIR / f"noise_rejected_{label}.png"
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_all() -> None:
    plot_noise_rejected()


if __name__ == "__main__":
    plot_all()
