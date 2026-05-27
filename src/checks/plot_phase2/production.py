"""
Inner-normalised production plot for the phase2 pipeline.

Single 3-panel figure (0/50/100 psig, sharey) overlaying the available
sensor positions and the three pipeline stages (raw, FRF-corrected,
Wiener-rejected) on inner-scaled axes. Mirrors plot_bump.production.

Phase2 is close-only, so only P1 (close, PH1) and P2 (close, PH2) are
drawn; P3 and P4 are filtered out of the position legend automatically.
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import welch, get_window

from src.save_phase2.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FS = cfg.FS
NPERSEG = 2**12
WINDOW = cfg.WINDOW

F_CUTL = 100.0
F_CUTH = 1_000.0
TPLUS_XLIM = (7.0, 7_000.0)
PHI_YLIM = (0.0, 14.0)

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "production"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    and smooth with a uniform kernel of width span_oct. Cosmetic only --
    applied just before inner-scaling on the Wiener-rejected stage."""
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
    try:
        ds = g_corr[sp][channel_key]
        return (
            float(np.atleast_1d(ds.attrs["u_tau"])[0]),
            float(np.atleast_1d(ds.attrs["Re_tau"])[0]),
        )
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
    """One 3-panel figure (0/50/100 psig) overlaying the available positions
    (phase2 = close-only, so just P1/P2) and the three pipeline stages.

    Same axes, ROI, and smoothing convention as plot_bump.production.
    Position legend filters down to positions that actually produced data
    on disk.
    """
    drawn_positions: set[tuple[str, str]] = set()

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
                drawn_positions.add((sp, ch))

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

            ax.set_title(f"phase2 {L}")
            ax.set_xlabel(r"$T^+$")
            if i == 0:
                ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
            ax.set_xlim(*TPLUS_XLIM)
            ax.set_ylim(*PHI_YLIM)
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)

        stage_lines = [
            Line2D([0], [0], color="black", linestyle=":", lw=0.9, alpha=0.45),
            Line2D([0], [0], color="black", linestyle="--", lw=0.9, alpha=0.45),
            Line2D([0], [0], color="black", linestyle="-", lw=1.4),
            Line2D([0], [0], color="black", linestyle="--", lw=0.7),
            Line2D([0], [0], color="black", linestyle="-.", lw=0.7),
        ]
        stage_labels = ["raw", "FRF-corrected", "Wiener-rejected",
                        "BL model", "Channel model"]
        axes[0].legend(stage_lines, stage_labels, loc="upper right",
                       fontsize=7, title="stage", title_fontsize=7)

        # Position legend: only positions that actually rendered on at least
        # one panel (phase2's close-only data drops P3/P4 silently).
        drawn = [
            (lbl, col)
            for sp, ch, lbl, col in POSITION_TRACES
            if (sp, ch) in drawn_positions
        ]
        if drawn:
            pos_lines = [Line2D([0], [0], color=col, linestyle="-", lw=1.4)
                         for _, col in drawn]
            pos_labels = [lbl for lbl, _ in drawn]
            axes[-1].legend(pos_lines, pos_labels, loc="upper right",
                            fontsize=7, title="position", title_fontsize=7)

        out = FIG_DIR / "G_wallp_SU_production_stages.png"
        fig.savefig(out, dpi=600)
        plt.close(fig)
        print(f"[ok] {out}")


PRESSURE_COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26")


def _fade_alpha(u: float, *, mid: float, span: float) -> float:
    if span <= 0.0:
        return 0.9
    w = 1.0 - np.abs(u - mid) / (0.5 * span)
    return 0.15 + 0.75 * np.clip(w, 0.0, 1.0)


def plot_stages_PH2_3panel() -> None:
    """Phase2 equivalent of phase1's per-channel PH2 stages figure.

    One 3-panel figure (0/50/100 psig) for the PH2 channel only, pressure-
    coloured. Each panel: BL + Channel models, u_tau uncertainty fan, and the
    three pipeline stages (raw dotted grey, FRF-corrected dashed colour,
    Wiener-rejected solid colour). Phase2 is close-only so sp_roi = "close"
    for every panel; matches phase1's older `G_wallp_SU_production_stages_PH2.png`
    style.
    """
    from matplotlib.colors import to_rgba

    f_cutl, f_cuth = F_CUTL, F_CUTH
    channel_key = "PH2_Pa"

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
            figsize=(3.0 * len(labels), 3.1),
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

            available_spacings = [sp for sp in ("far", "close") if sp in g_rej]
            if not available_spacings:
                ax.set_visible(False)
                continue
            sp_model = "far" if "far" in available_spacings else available_spacings[0]
            sp_roi = "close" if "close" in available_spacings else available_spacings[0]
            if channel_key not in g_rej[sp_roi]:
                ax.set_visible(False)
                continue

            Ue = _get_ue(hf_prod, gL_prod, i)
            u_tau, Re_tau = _local_u_tau_re_tau(g_rej, sp_roi, channel_key, gL_prod)
            nu = float(np.atleast_1d(gL_prod.attrs["nu"])[0])
            rho = float(np.atleast_1d(gL_prod.attrs["rho"])[0])
            u_tau_rel_unc = float(
                np.atleast_1d(gL_prod.attrs.get("u_tau_rel_unc", 0.0))[0]
            )
            colour = PRESSURE_COLOURS[i % len(PRESSURE_COLOURS)]

            sig_model = g_rej[sp_model][channel_key][:]
            f_model, _ = _spec(sig_model)
            model_mask = f_model > 0.0
            if not np.any(model_mask):
                ax.set_visible(False)
                continue
            f_model = f_model[model_mask]
            t_plus_model = (u_tau ** 2) / (nu * f_model)
            cf_2 = (u_tau / Ue) ** 2

            g1_b, g2_b, rv_b = bl_model(t_plus_model, Re_tau, cf_2)
            g1_c, g2_c, rv_c = channel_model(t_plus_model, Re_tau, u_tau, u_cl=Ue)
            ax.semilogx(t_plus_model, rv_b * (g1_b + g2_b),
                        linestyle="--", color=colour, lw=0.7)
            ax.semilogx(t_plus_model, rv_c * (g1_c + g2_c),
                        linestyle="-.", color=colour, lw=0.7)

            f_roi, p_rej = _spec(g_rej[sp_roi][channel_key][:])
            mask = (f_roi > f_cutl) & (f_roi < f_cuth)
            if not np.any(mask):
                ax.set_visible(False)
                continue
            f_m = f_roi[mask]
            P_rej_m = p_rej[mask]

            u_nom = u_tau
            u_lo = u_nom * (1.0 - u_tau_rel_unc)
            u_hi = u_nom * (1.0 + u_tau_rel_unc)
            u_grid = np.linspace(u_lo, u_hi, 16)
            mid = 0.5 * (u_lo + u_hi)
            span = (u_hi - u_lo)
            order = np.argsort(np.abs(u_grid - u_nom))[::-1]
            for j in order:
                u = u_grid[j]
                T = (u ** 2) / (nu * f_m)
                Y = (f_m * P_rej_m) / (rho ** 2 * u ** 4)
                ax.semilogx(
                    T, Y,
                    color=to_rgba("gray", _fade_alpha(float(u), mid=mid, span=span)),
                    linewidth=1.0,
                )

            T_nom = (u_nom ** 2) / (nu * f_m)

            if gL_raw is not None and sp_roi in gL_raw and channel_key in gL_raw[sp_roi]:
                f_r, p_r = _spec(gL_raw[sp_roi][channel_key][:])
                mr = (f_r > f_cutl) & (f_r < f_cuth)
                Y_raw = (f_r[mr] * p_r[mr]) / (rho ** 2 * u_nom ** 4)
                ax.semilogx((u_nom ** 2) / (nu * f_r[mr]), Y_raw,
                            color="#555555", linestyle=":", lw=0.9, alpha=0.7)

            if g_frf is not None and sp_roi in g_frf and channel_key in g_frf[sp_roi]:
                f_c, p_c = _spec(g_frf[sp_roi][channel_key][:])
                mc = (f_c > f_cutl) & (f_c < f_cuth)
                Y_frf = (f_c[mc] * p_c[mc]) / (rho ** 2 * u_nom ** 4)
                ax.semilogx((u_nom ** 2) / (nu * f_c[mc]), Y_frf,
                            color=colour, linestyle="--", lw=0.9, alpha=0.7)

            Y_rej = (f_m * P_rej_m) / (rho ** 2 * u_nom ** 4)
            ax.semilogx(T_nom, Y_rej, color=colour, linewidth=1.2, zorder=10)

            ax.set_title(f"phase2 {L}")
            ax.set_xlabel(r"$T^+$")
            if i == 0:
                ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
            ax.set_xlim(7, 7_000)
            ax.set_ylim(0, 6)
            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)

        legend_lines = [
            Line2D([0], [0], color="black", linestyle="--", lw=0.7),
            Line2D([0], [0], color="black", linestyle="-.", lw=0.7),
            Line2D([0], [0], color="#555555", linestyle=":", lw=0.9, alpha=0.7),
            Line2D([0], [0], color="black", linestyle="--", lw=0.9, alpha=0.7),
            Line2D([0], [0], color="black", linestyle="-", lw=1.2),
        ]
        legend_labels = [
            "BL model", "Channel model", "raw",
            "FRF-corrected", "Wiener-rejected",
        ]
        axes[0].legend(legend_lines, legend_labels, loc="upper right", fontsize=7)

        out = FIG_DIR / "G_wallp_SU_production_stages_PH2.png"
        fig.savefig(out, dpi=600)
        plt.close(fig)
        print(f"[ok] {out}")


def plot_all() -> None:
    plot_stages_3panel()
    plot_stages_PH2_3panel()


if __name__ == "__main__":
    plot_all()
