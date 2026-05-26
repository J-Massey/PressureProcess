"""
Inner-normalised production plots for the phase2 pipeline.

Mirrors phase1's G_wallp_SU_production.plot_model_comparison_roi:
  - One figure per (pressure, channel).
  - BL model (dashed) + Channel model (dash-dot) curves drawn in inner units.
  - u_tau uncertainty fan: a grey fan of curves swept across +-u_tau_rel_unc,
    fading at the edges; the nominal-u_tau curve is drawn on top in colour.
  - ROI for the data: F_CUTL < f < F_CUTH (100 < f < 1000 Hz).
  - Fixed axes: T^+ in (7, 7000), y in (0, 6).

Phase2 is close-only, so sp_model and sp_roi both resolve to close.
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.signal import welch, get_window

from src.save_phase2.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FS = cfg.FS
NPERSEG = 2**12
WINDOW = cfg.WINDOW

# ROI + axes matching phase1's G_wallp_SU_production.
F_CUTL = 100.0
F_CUTH = 1_000.0
TPLUS_XLIM = (7.0, 7_000.0)
PHI_YLIM = (0.0, 6.0)

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "production"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Pressure-indexed colours; matches phase1 (C0/C1/C2 hex).
PRESSURE_COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26")


def _spec(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nseg = min(NPERSEG, x.size)
    w = get_window(WINDOW, nseg, fftbins=True)
    f, p = welch(x, fs=FS, window=w, nperseg=nseg, noverlap=nseg // 2,
                 detrend="constant", scaling="density", return_onesided=True)
    return f, p


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


def _fade_alpha(u: float, *, mid: float, span: float) -> float:
    if span <= 0.0:
        return 0.9
    w = 1.0 - np.abs(u - mid) / (0.5 * span)
    return 0.15 + 0.75 * np.clip(w, 0.0, 1.0)


def _get_ue(hf: h5py.File, gL: h5py.Group, idx: int, default: float = 14.0) -> float:
    if "Ue_m_per_s" in gL.attrs:
        return float(np.atleast_1d(gL.attrs["Ue_m_per_s"])[0])
    ue_attr = hf.attrs.get("Ue_m_per_s", default)
    ue_arr = np.atleast_1d(ue_attr)
    if ue_arr.size > idx:
        return float(ue_arr[idx])
    return float(ue_arr[0])


def plot_noise_rejected() -> None:
    channel_keys = ("PH1_Pa", "PH2_Pa")

    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_fs = hf["wallp_production"]
        labels = list(g_fs.keys())
        if not labels:
            print("[skip] no labels in wallp_production")
            return

        fs = float(hf.attrs.get("fs_Hz", FS))

        for i, L in enumerate(labels):
            gL = g_fs[L]
            if "fs_noise_rejected_signals" not in gL:
                print(f"[skip] {L}: no fs_noise_rejected_signals group")
                continue
            g_corr = gL["fs_noise_rejected_signals"]

            available_spacings = [sp for sp in ("far", "close") if sp in g_corr]
            if not available_spacings:
                print(f"[skip] {L}: no corrected spacing groups")
                continue
            sp_model = "far" if "far" in available_spacings else available_spacings[0]
            sp_roi = "close" if "close" in available_spacings else available_spacings[0]

            Ue = _get_ue(hf, gL, i)
            u_tau = float(np.atleast_1d(gL.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(gL.attrs["nu"])[0])
            rho = float(np.atleast_1d(gL.attrs["rho"])[0])
            Re_tau = float(np.atleast_1d(gL.attrs["Re_tau"])[0])
            u_tau_rel_unc = float(
                np.atleast_1d(gL.attrs.get("u_tau_rel_unc", 0.0))[0]
            )

            colour = PRESSURE_COLOURS[i % len(PRESSURE_COLOURS)]

            for channel_key in channel_keys:
                if channel_key not in g_corr[sp_model] or channel_key not in g_corr[sp_roi]:
                    print(f"[skip] {L}: missing {channel_key} in {sp_model}/{sp_roi}")
                    continue

                channel_tag = channel_key.split("_")[0]
                fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.1), tight_layout=True)

                sig_model = g_corr[sp_model][channel_key][:]
                sig_roi = g_corr[sp_roi][channel_key][:]

                f_model, _ = _spec(sig_model)
                f_roi, pyy_roi = _spec(sig_roi)
                model_mask = f_model > 0.0
                if not np.any(model_mask):
                    print(f"[skip] {L} {channel_tag}: no positive model frequencies")
                    plt.close(fig)
                    continue
                f_model = f_model[model_mask]

                t_plus_model = (u_tau ** 2) / (nu * f_model)
                cf_2 = (u_tau / Ue) ** 2

                g1_b, g2_b, rv_b = bl_model(t_plus_model, Re_tau, cf_2)
                g1_c, g2_c, rv_c = channel_model(t_plus_model, Re_tau, u_tau, u_cl=Ue)

                bl_fphipp_plus = rv_b * (g1_b + g2_b)
                channel_fphipp_plus = rv_c * (g1_c + g2_c)

                ax.semilogx(t_plus_model, bl_fphipp_plus,
                            linestyle="--", color=colour, lw=0.7)
                ax.semilogx(t_plus_model, channel_fphipp_plus,
                            linestyle="-.", color=colour, lw=0.7)

                # ROI mask for the data line.
                mask = (f_roi > F_CUTL) & (f_roi < F_CUTH)
                if not np.any(mask):
                    print(f"[skip] {L} {channel_tag}: no frequencies in ROI")
                    plt.close(fig)
                    continue
                f_m = f_roi[mask]
                P_m = pyy_roi[mask]

                u_nom = u_tau
                u_lo = u_nom * (1.0 - u_tau_rel_unc)
                u_hi = u_nom * (1.0 + u_tau_rel_unc)
                u_grid = np.linspace(u_lo, u_hi, 16)
                mid = 0.5 * (u_lo + u_hi)
                span = (u_hi - u_lo)

                # u_tau uncertainty fan: draw edges first, centre last.
                order = np.argsort(np.abs(u_grid - u_nom))[::-1]
                for j in order:
                    u = u_grid[j]
                    T = (u ** 2) / (nu * f_m)
                    Y = (f_m * P_m) / (rho ** 2 * u ** 4)
                    ax.semilogx(
                        T, Y,
                        color=to_rgba("gray", _fade_alpha(float(u), mid=mid, span=span)),
                        linewidth=1.0,
                    )

                # Nominal-u_tau data curve on top.
                T_nom = (u_nom ** 2) / (nu * f_m)
                Y_nom = (f_m * P_m) / (rho ** 2 * u_nom ** 4)
                ax.semilogx(
                    T_nom, Y_nom,
                    color=colour, linewidth=1.0, label=L, zorder=10,
                )

                ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
                ax.set_title(f"phase2 {L} ({channel_tag})")
                ax.set_xlabel(r"$T^+$")
                ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
                ax.set_xlim(*TPLUS_XLIM)
                ax.set_ylim(*PHI_YLIM)

                legend_lines = [
                    Line2D([0], [0], color="black", linestyle="--"),
                    Line2D([0], [0], color="black", linestyle="-."),
                    Line2D([0], [0], color=colour, linestyle="-"),
                ]
                legend_labels = [
                    "BL model",
                    "Channel model",
                    f"Data ({sp_roi}, {channel_tag})",
                ]
                ax.legend(legend_lines, legend_labels, loc="upper center", fontsize=8)

                # Match phase1: G_wallp_SU_production_{L}_{channel}.png at dpi=600;
                # additionally write the historical PH2 filename for compatibility.
                out_paths = [FIG_DIR / f"G_wallp_SU_production_{L}_{channel_tag}.png"]
                if channel_tag == "PH2":
                    out_paths.append(FIG_DIR / f"G_wallp_SU_production_{L}.png")
                for out in out_paths:
                    fig.savefig(out, dpi=600)
                    print(f"[ok] {out}")
                plt.close(fig)


def plot_stages_3panel() -> None:
    """
    One 3-panel figure per channel (PH1, PH2), with one panel per pressure
    (0/50/100 psig). Each panel shows on the same inner-scaled axes:
      - BL model (dashed) and Channel model (dash-dot)
      - u_tau uncertainty fan (grey)
      - raw wall pressure (light, dotted)
      - frf_corrected (PH->NC->nkd, no Wiener) -- pressure colour, dashed
      - fs_noise_rejected (after Wiener) -- pressure colour, solid (nominal)

    Same ROI (100 < f < 1000 Hz), same axes (T^+ in (7, 7000), y in (0, 6)),
    same NPERSEG=2**12 used elsewhere.
    """
    channel_keys = ("PH1_Pa", "PH2_Pa")

    with h5py.File(cfg.PH_RAW_FILE, "r") as hf_raw, \
         h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf_prod:
        g_raw_root = hf_raw["wallp_raw"]
        g_prod_root = hf_prod["wallp_production"]
        labels = [L for L in cfg.LABELS if L in g_prod_root]
        if not labels:
            print("[skip] no labels in wallp_production")
            return

        fs = float(hf_prod.attrs.get("fs_Hz", FS))

        for channel_key in channel_keys:
            channel_tag = channel_key.split("_")[0]
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
                u_tau = float(np.atleast_1d(gL_prod.attrs["u_tau"])[0])
                nu = float(np.atleast_1d(gL_prod.attrs["nu"])[0])
                rho = float(np.atleast_1d(gL_prod.attrs["rho"])[0])
                Re_tau = float(np.atleast_1d(gL_prod.attrs["Re_tau"])[0])
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
                bl_fphipp_plus = rv_b * (g1_b + g2_b)
                channel_fphipp_plus = rv_c * (g1_c + g2_c)

                ax.semilogx(t_plus_model, bl_fphipp_plus,
                            linestyle="--", color=colour, lw=0.7)
                ax.semilogx(t_plus_model, channel_fphipp_plus,
                            linestyle="-.", color=colour, lw=0.7)

                # Pick the ROI mask once and reuse for all three stages so the
                # ROI is identical across raw/frf/rejected.
                f_roi, p_rej = _spec(g_rej[sp_roi][channel_key][:])
                mask = (f_roi > F_CUTL) & (f_roi < F_CUTH)
                if not np.any(mask):
                    ax.set_visible(False)
                    continue

                u_nom = u_tau
                u_lo = u_nom * (1.0 - u_tau_rel_unc)
                u_hi = u_nom * (1.0 + u_tau_rel_unc)
                u_grid = np.linspace(u_lo, u_hi, 16)
                mid = 0.5 * (u_lo + u_hi)
                span = (u_hi - u_lo)
                f_m = f_roi[mask]
                P_rej_m = p_rej[mask]
                order = np.argsort(np.abs(u_grid - u_nom))[::-1]
                for j in order:
                    u = u_grid[j]
                    T = (u ** 2) / (nu * f_m)
                    Y = (f_m * P_rej_m) / (rho ** 2 * u ** 4)
                    # ax.semilogx(
                    #     T, Y,
                    #     color=to_rgba("gray", _fade_alpha(float(u), mid=mid, span=span)),
                    #     linewidth=1.0,
                    # )

                T_nom = (u_nom ** 2) / (nu * f_m)

                # Raw (pre-FRF, pre-bandpass, pre-Wiener)
                if gL_raw is not None and sp_roi in gL_raw and channel_key in gL_raw[sp_roi]:
                    f_r, p_r = _spec(gL_raw[sp_roi][channel_key][:])
                    mr = (f_r > F_CUTL) & (f_r < F_CUTH)
                    Y_raw = (f_r[mr] * p_r[mr]) / (rho ** 2 * u_nom ** 4)
                    ax.semilogx((u_nom ** 2) / (nu * f_r[mr]), Y_raw,
                                color="#555555", linestyle=":", lw=0.9, alpha=0.7)

                # FRF-corrected (PH->NC->nkd, no Wiener)
                if g_frf is not None and sp_roi in g_frf and channel_key in g_frf[sp_roi]:
                    f_c, p_c = _spec(g_frf[sp_roi][channel_key][:])
                    mc = (f_c > F_CUTL) & (f_c < F_CUTH)
                    Y_frf = (f_c[mc] * p_c[mc]) / (rho ** 2 * u_nom ** 4)
                    ax.semilogx((u_nom ** 2) / (nu * f_c[mc]), Y_frf,
                                color='grey', linestyle="--", lw=3, alpha=0.7)

                # Wiener-rejected (nominal data line, on top)
                Y_rej = (f_m * P_rej_m) / (rho ** 2 * u_nom ** 4)
                ax.semilogx(T_nom, Y_rej, color=colour, linewidth=1.2, zorder=10)

                ax.set_title(f"phase2 {L}")
                ax.set_xlabel(r"$T^+$")
                if i == 0:
                    ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
                ax.set_xlim(*TPLUS_XLIM)
                ax.set_ylim(*PHI_YLIM)
                ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)

            # Single legend on the first panel; legend keys are stage-agnostic
            # (pressure colour is per-panel) so we use neutral black for clarity.
            legend_lines = [
                Line2D([0], [0], color="black", linestyle="--", lw=0.7),
                Line2D([0], [0], color="black", linestyle="-.", lw=0.7),
                Line2D([0], [0], color="#555555", linestyle=":", lw=0.9, alpha=0.7),
                Line2D([0], [0], color="black", linestyle="--", lw=0.9, alpha=0.7),
                Line2D([0], [0], color="black", linestyle="-", lw=1.2),
            ]
            legend_labels = [
                "BL model",
                "Channel model",
                "raw",
                "FRF-corrected",
                "Wiener-rejected",
            ]
            axes[0].legend(legend_lines, legend_labels, loc="upper right", fontsize=7)
            

            out = FIG_DIR / f"G_wallp_SU_production_stages_{channel_tag}.png"
            fig.savefig(out, dpi=600)
            plt.close(fig)
            print(f"[ok] {out}")


def plot_all() -> None:
    plot_noise_rejected()
    plot_stages_3panel()


if __name__ == "__main__":
    plot_all()
