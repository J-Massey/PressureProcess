"""
Inner-normalised production plot for the bump pipeline.

A single 3-panel figure (0/50/100 psig, sharey) showing the four sensor
positions (P1..P4) and the three pipeline stages (raw, FRF-corrected,
Wiener-rejected) overlaid on inner-scaled axes. Each line is normalised by
its own position-specific u_tau (read from the per-channel dataset attrs
stamped by save_bump/pw_proc.py).

Smoothing: PSDs are resampled onto a log-frequency grid and convolved with a
uniform kernel before inner-scaling -- exposed in plot_stages_3panel as
SMOOTH_SPAN_OCT and SMOOTH_PPO so the smoothing is visible at the call site.
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import welch, get_window

from src.save_bump.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FS = cfg.FS
NPERSEG = 2**12
WINDOW = cfg.WINDOW

# ROI + axes matching phase1/phase2.
F_CUTL = 100.0
F_CUTH = 1_000.0
TPLUS_XLIM = (7.0, 7_000.0)
PHI_YLIM = (0.0, 14.0)

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "production"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# P1..P4 positions: matches plot_bump/raw.py legend mapping.
POSITION_TRACES = (
    ("close", "PH1_Pa", "P1 (close, PH1)", "#0b4eb2"),
    ("close", "PH2_Pa", "P2 (close, PH2)", "#d62728"),
    ("far",   "PH1_Pa", "P3 (far,  PH1)",  "#26bd26"),
    ("far",   "PH2_Pa", "P4 (far,  PH2)",  "#a51990"),
)


def _spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nseg = min(NPERSEG, x.size)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=nseg, noverlap=nseg // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


def _smooth_psd_logf(
    f: np.ndarray,
    p: np.ndarray,
    *,
    span_oct: float,
    ppo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a one-sided PSD onto a log-f grid (ppo points-per-octave)
    and smooth with a uniform kernel of width span_oct. Real-valued only --
    used as a cosmetic step before inner-scaling so 12 overlaid lines stay
    legible."""
    f = np.asarray(f, dtype=float)
    p = np.asarray(p, dtype=float)
    pos = f > 0.0
    if not np.any(pos):
        return f, p
    fpos = f[pos]
    ppos = p[pos]
    f_lo = float(max(fpos[0], 1e-12))
    f_hi = float(fpos[-1])
    n_oct = np.log2(f_hi / f_lo)
    n_pts = max(int(np.ceil(n_oct * ppo)), 8)
    flog = np.linspace(np.log2(f_lo), np.log2(f_hi), n_pts)
    fgrid = 2.0 ** flog
    pgrid = np.interp(fgrid, fpos, ppos)
    wlen = max(int(round(span_oct * ppo)), 1)
    if wlen % 2 == 0:
        wlen += 1
    ker = np.ones(wlen) / wlen
    psm = np.convolve(pgrid, ker, mode="same")
    return fgrid, psm


def bl_model(Tplus, Re_tau: float, cf_2: float):
    A1 = 2.2
    sig1 = 3.9
    mean_Tplus = 20
    A2 = 1.4 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.2
    mean_To = 0.82
    r1 = 0.5
    r2 = 7
    z = np.clip(r1 * (Tplus - r2), -60.0, 60.0)
    rv = 1.0 / (1.0 + np.exp(-z))
    mean_To_plus = mean_To * Re_tau * np.sqrt(cf_2)
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus)) ** 2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus)) ** 2)
    return g1, g2, rv


def channel_model(Tplus, Re_tau: float, u_tau: float, u_cl: float):
    A1 = 2.1 * (1 - 100 / Re_tau)
    sig1 = 4.4
    mean_Tplus = 12
    A2 = 0.9 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.0
    mean_To = 0.6
    r1 = 0.5
    r2 = 3
    z = np.clip(r1 * (Tplus - r2), -60.0, 60.0)
    rv = 1.0 / (1.0 + np.exp(-z))
    mean_To_plus = mean_To * Re_tau * u_tau / u_cl
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus)) ** 2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus)) ** 2)
    return g1, g2, rv


def _get_ue(hf: h5py.File, gL: h5py.Group, idx: int, default: float = 14.0) -> float:
    if "Ue_m_per_s" in gL.attrs:
        return float(np.atleast_1d(gL.attrs["Ue_m_per_s"])[0])
    ue_attr = hf.attrs.get("Ue_m_per_s", default)
    ue_arr = np.atleast_1d(ue_attr)
    if ue_arr.size > idx:
        return float(ue_arr[idx])
    return float(ue_arr[0])


def _local_u_tau_re_tau(
    g_corr: h5py.Group,
    sp: str,
    channel_key: str,
    gL: h5py.Group,
) -> tuple[float, float]:
    """Per-(spacing, channel) u_tau and Re_tau when pw_proc has stamped them
    on the dataset attrs; otherwise fall back to the per-label values."""
    try:
        ds = g_corr[sp][channel_key]
        u_tau_v = float(np.atleast_1d(ds.attrs["u_tau"])[0])
        re_tau_v = float(np.atleast_1d(ds.attrs["Re_tau"])[0])
        return u_tau_v, re_tau_v
    except (KeyError, ValueError, TypeError):
        return (
            float(np.atleast_1d(gL.attrs["u_tau"])[0]),
            float(np.atleast_1d(gL.attrs["Re_tau"])[0]),
        )


def plot_stages_3panel(
    *,
    smooth_span_oct: float = 1 / 2,
    smooth_ppo: int = 96,
) -> None:
    """One 3-panel figure (0/50/100 psig) overlaying all 4 positions (P1..P4)
    and all 3 pipeline stages (raw, FRF-corrected, Wiener-rejected).

    Each line is normalised by its position-specific u_tau (read from the
    dataset attrs stamped by pw_proc). Inner-scaled axes T^+ in (7, 7000),
    y in (0, 6), with the data masked to the 100 < f < 1000 Hz ROI.

    Smoothing (cosmetic): PSDs are resampled onto a `smooth_ppo`-points-per-
    octave log-frequency grid and convolved with a uniform kernel of width
    `smooth_span_oct` octaves before inner-scaling. Defaults: 1/6 octave,
    96 ppo.

    Style key:
        position -> colour  (P1 blue, P2 red, P3 green, P4 magenta)
        stage    -> linestyle  (raw dotted, FRF-corrected dashed,
                                Wiener-rejected solid)
    """
    with h5py.File(cfg.PH_RAW_FILE, "r") as hf_raw, \
         h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf_prod:
        g_raw_root = hf_raw["wallp_raw"]
        g_prod_root = hf_prod["wallp_production"]
        labels = [L for L in cfg.LABELS if L in g_prod_root]
        if not labels:
            print("[skip] no labels in wallp_production")
            return

        fig, axes = plt.subplots(
            1, len(labels),
            figsize=(3.6 * len(labels), 4.2),
            sharey=True, tight_layout=True, squeeze=False,
        )
        axes = axes[0]

        for i, L in enumerate(labels):
            ax = axes[i]
            gL_prod = g_prod_root[L]
            gL_raw = g_raw_root[L] if L in g_raw_root else None
            g_frf = gL_prod.get("frf_corrected_signals")
            g_rej = gL_prod.get("fs_noise_rejected_signals")
            if g_rej is None:
                ax.set_visible(False)
                continue

            Ue = _get_ue(hf_prod, gL_prod, i)
            nu = float(np.atleast_1d(gL_prod.attrs["nu"])[0])
            rho = float(np.atleast_1d(gL_prod.attrs["rho"])[0])

            u_taus: list[float] = []
            re_taus: list[float] = []

            for sp, ch, _lbl, col in POSITION_TRACES:
                if sp not in g_rej or ch not in g_rej[sp]:
                    continue
                u_tau_p, re_tau_p = _local_u_tau_re_tau(g_rej, sp, ch, gL_prod)
                u_taus.append(u_tau_p)
                re_taus.append(re_tau_p)

                def _draw_stage(group, linestyle: str, lw: float, alpha: float):
                    if group is None or sp not in group or ch not in group[sp]:
                        return
                    f_raw, p_raw = _spec(group[f"{sp}/{ch}"][:])
                    if group is g_rej:
                        f_s, p_s = _smooth_psd_logf(
                            f_raw, p_raw,
                            span_oct=smooth_span_oct, ppo=smooth_ppo,
                        )
                    else:
                        f_s, p_s = f_raw, p_raw
                    m = (f_s > F_CUTL) & (f_s < F_CUTH)
                    if not np.any(m):
                        return
                    T = (u_tau_p ** 2) / (nu * f_s[m])
                    Y = (f_s[m] * p_s[m]) / (rho ** 2 * u_tau_p ** 4)
                    ax.semilogx(T, Y, color=col, linestyle=linestyle,
                                lw=lw, alpha=alpha)

                _draw_stage(gL_raw, ":", 0.9, 0.45)
                _draw_stage(g_frf,  "--", 0.9, 0.45)
                _draw_stage(g_rej,  "-", 1.4, 1.0)

            # Reference models at the median u_tau / Re_tau across positions.
            if u_taus:
                u_tau_ref = float(np.median(u_taus))
                Re_tau_ref = float(np.median(re_taus))
                cf_2 = (u_tau_ref / Ue) ** 2
                T_grid = np.geomspace(TPLUS_XLIM[0], TPLUS_XLIM[1], 256)
                g1_b, g2_b, rv_b = bl_model(T_grid, Re_tau_ref, cf_2)
                g1_c, g2_c, rv_c = channel_model(T_grid, Re_tau_ref, u_tau_ref, u_cl=Ue)
                ax.semilogx(T_grid, rv_b * (g1_b + g2_b),
                            linestyle="--", color="black", lw=0.7)
                ax.semilogx(T_grid, rv_c * (g1_c + g2_c),
                            linestyle="-.", color="black", lw=0.7)

            ax.set_title(f"bump {L}")
            ax.set_xlabel(r"$T^+$")
            if i == 0:
                ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
            ax.set_xlim(*TPLUS_XLIM)
            ax.set_ylim(*PHI_YLIM)
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)

        stage_lines = [
            Line2D([0], [0], color="black", linestyle=":", lw=0.9, alpha=0.45),
            Line2D([0], [0], color="black", linestyle="--", lw=1.0, alpha=0.7),
            Line2D([0], [0], color="black", linestyle="-", lw=1.4),
            Line2D([0], [0], color="black", linestyle="--", lw=0.7),
            Line2D([0], [0], color="black", linestyle="-.", lw=0.7),
        ]
        stage_labels = ["raw", "FRF-corrected", "Wiener-rejected",
                        "BL model", "Channel model"]
        axes[0].legend(stage_lines, stage_labels, loc="upper right",
                       fontsize=7, title="stage", title_fontsize=7)

        pos_lines = [Line2D([0], [0], color=col, linestyle="-", lw=1.4)
                     for _, _, _lbl, col in POSITION_TRACES]
        pos_labels = [lbl for _, _, lbl, _ in POSITION_TRACES]
        axes[-1].legend(pos_lines, pos_labels, loc="upper right",
                        fontsize=7, title="position", title_fontsize=7)

        # fig.suptitle(
        #     "bump -- wall pipeline stages, inner-scaled "
        #     f"(P1..P4, smoothed {smooth_span_oct:g} oct @ {smooth_ppo} ppo)",
        #     fontsize=10,
        # )

        out = FIG_DIR / "G_wallp_SU_production_stages.png"
        fig.savefig(out, dpi=600)
        plt.close(fig)
        print(f"[ok] {out}")


def plot_all() -> None:
    plot_stages_3panel()


def plot_cleaned_by_case() -> None:
    """Back-compat shim for src/checks/plot/run_all.py (bump branch)."""
    plot_all()


if __name__ == "__main__":
    plot_all()
