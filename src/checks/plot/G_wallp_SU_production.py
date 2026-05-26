import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from icecream import ic

from src.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()

apply_plot_style()

# -------------------- constants --------------------
FS = cfg.FS
NPERSEG = 2**12          # keep one value for all runs
WINDOW  = cfg.WINDOW

LABELS = ("0psig", "50psig", "100psig")
PSIGS  = (0.0, 50.0, 100.0)
COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26")  # hex equivalents of C0, C1, C2
FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR)


def compute_spec(x: np.ndarray, fs: float = FS, nperseg: int = NPERSEG):
    """Welch PSD with consistent settings. Returns f [Hz], Pxx [Pa^2/Hz]."""
    x = np.asarray(x, float)
    nseg = min(nperseg, x.size)
    if nseg < 16:
        raise ValueError(f"Signal too short for Welch: n={x.size}, nperseg={nperseg}")
    w = get_window(WINDOW, nseg, fftbins=True)
    f, Pxx = welch(
        x, fs=fs, window=w, nperseg=nseg, noverlap=nseg//2,
        detrend="constant", scaling="density", return_onesided=True,
    )
    return f, Pxx

def _get_ue(hf: h5py.File, gL: h5py.Group, idx: int, default: float = 14.0) -> float:
    if "Ue_m_per_s" in gL.attrs:
        return float(np.atleast_1d(gL.attrs["Ue_m_per_s"])[0])
    ue_attr = hf.attrs.get("Ue_m_per_s", default)
    ue_arr = np.atleast_1d(ue_attr)
    if ue_arr.size > idx:
        return float(ue_arr[idx])
    return float(ue_arr[0])

def bl_model(Tplus, Re_tau: float, cf_2: float) -> np.ndarray:
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
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv

def channel_model(Tplus, Re_tau: float, u_tau: float, u_cl) -> np.ndarray:
    A1 = 2.1*(1 - 100/Re_tau)
    sig1 = 4.4
    mean_Tplus = 12
    A2 = 0.9 * (np.log10(Re_tau) - 2.2)
    sig2 = 1.0
    mean_To = 0.6
    r1 = 0.5
    r2 = 3
    z = np.clip(r1 * (Tplus - r2), -60.0, 60.0)
    rv = 1.0 / (1.0 + np.exp(-z))
    mean_To_plus = mean_To * Re_tau * u_tau/u_cl
    g1 = A1 * np.exp(-sig1 * (np.log10(Tplus) - np.log10(mean_Tplus))**2)
    g2 = A2 * np.exp(-sig2 * (np.log10(Tplus) - np.log10(mean_To_plus))**2)
    return g1, g2, rv

def _fade_alpha(u: float, *, mid: float, span: float) -> float:
    if span <= 0.0:
        return 0.9
    w = 1.0 - np.abs(u - mid) / (0.5 * span)  # 1 at centre, 0 at edges
    return 0.15 + 0.75 * np.clip(w, 0.0, 1.0)


def plot_model_comparison_roi():
    f_cutl, f_cuth = 100.0, 1_000.0  # Hz
    channel_keys = ("PH1_Pa", "PH2_Pa")

    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_fs = hf["wallp_production"]
        labels = list(g_fs.keys())
        if not labels:
            raise KeyError("No labels found in wallp_production")

        # fall back to global FS if attribute is missing
        fs = float(hf.attrs.get("fs_Hz", FS))
        model_labels = ["BL model", "Channel model"]
        model_lines = [
            Line2D([0], [0], color="black", linestyle="--"),
            Line2D([0], [0], color="black", linestyle="-."),
        ]

        for i, L in enumerate(labels):
            gL = g_fs[L]
            Ue = _get_ue(hf, gL, i)

            # scalarise attrs in case h5py gives small arrays
            u_tau = float(np.atleast_1d(gL.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(gL.attrs["nu"])[0])
            ic(u_tau**2 / (nu * 700), u_tau, nu)
            rho = float(np.atleast_1d(gL.attrs["rho"])[0])
            Re_tau = float(np.atleast_1d(gL.attrs["Re_tau"])[0])
            u_tau_rel_unc = float(
                np.atleast_1d(gL.attrs.get("u_tau_rel_unc", 0.0))[0]
            )

            g_corr = gL["fs_noise_rejected_signals"]
            available_spacings = [sp for sp in ("far", "close") if sp in g_corr]
            if not available_spacings:
                print(f"[skip] no corrected spacing groups for {L}")
                continue

            sp_model = "far" if "far" in available_spacings else available_spacings[0]
            sp_roi = "close" if "close" in available_spacings else available_spacings[0]

            for channel_key in channel_keys:
                if channel_key not in g_corr[sp_model] or channel_key not in g_corr[sp_roi]:
                    print(f"[skip] {L}: missing {channel_key} in {sp_model}/{sp_roi}")
                    continue

                channel_tag = channel_key.split("_")[0]
                fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.1), tight_layout=True)

                sig_model = g_corr[sp_model][channel_key][:]
                sig_roi = g_corr[sp_roi][channel_key][:]

                # spectra
                f_model, _ = compute_spec(sig_model, fs=fs, nperseg=NPERSEG)
                f_roi, pyy_roi = compute_spec(sig_roi, fs=fs, nperseg=NPERSEG)
                model_mask = f_model > 0.0
                if not np.any(model_mask):
                    print(f"[skip] non-positive model frequencies for {L} {channel_tag}")
                    plt.close(fig)
                    continue
                f_model = f_model[model_mask]

                # T^+ based on model spacing spectrum for the model curves
                t_plus_model = (u_tau**2) / (nu * f_model)

                # friction coefficient "cf_2" from u_tau and Ue (cf/2 = (u_tau/Ue)^2)
                cf_2 = (u_tau / Ue) ** 2

                # models
                g1_b, g2_b, rv_b = bl_model(t_plus_model, Re_tau, cf_2)
                g1_c, g2_c, rv_c = channel_model(t_plus_model, Re_tau, u_tau, u_cl=Ue)

                bl_fphipp_plus = rv_b * (g1_b + g2_b)
                channel_fphipp_plus = rv_c * (g1_c + g2_c)

                ax.semilogx(
                    t_plus_model,
                    bl_fphipp_plus,
                    linestyle="--",
                    color=COLOURS[i % len(COLOURS)],
                    lw=0.7,
                )
                ax.semilogx(
                    t_plus_model,
                    channel_fphipp_plus,
                    linestyle="-.",
                    color=COLOURS[i % len(COLOURS)],
                    lw=0.7,
                )

                # ROI & u_tau-uncertainty fan
                mask = (f_roi > f_cutl) & (f_roi < f_cuth)
                if not np.any(mask):
                    print(f"[skip] no frequencies in ROI for {L} {channel_tag}")
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

                # draw edges first, centre (nominal) last
                order = np.argsort(np.abs(u_grid - u_nom))[::-1]
                for j in order:
                    u = u_grid[j]
                    T = (u**2) / (nu * f_m)
                    Y = (f_m * P_m) / (rho**2 * u**4)
                    ax.semilogx(
                        T,
                        Y,
                        color=to_rgba("gray", _fade_alpha(float(u), mid=mid, span=span)),
                        linewidth=1.0,
                    )

                # nominal curve on top
                T_nom = (u_nom**2) / (nu * f_m)
                Y_nom = (f_m * P_m) / (rho**2 * u_nom**4)
                ax.semilogx(
                    T_nom,
                    Y_nom,
                    color=COLOURS[i % len(COLOURS)],
                    linewidth=1.0,
                    label=L,
                    zorder=10,
                )

                ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
                ax.set_title(f"{L} ({channel_tag})")
                ax.set_xlabel(r"$T^+$")
                ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
                ax.set_xlim(7, 7_000)
                ax.set_ylim(0, 6)

                legend_lines = model_lines + [
                    Line2D([0], [0], color=COLOURS[i % len(COLOURS)], linestyle="-")
                ]
                legend_labels = model_labels + [f"Data ({sp_roi}, {channel_tag})"]
                ax.legend(legend_lines, legend_labels, loc="upper center", fontsize=8)

                out_paths = [FIG_DIR / f"G_wallp_SU_production_{L}_{channel_tag}.png"]
                if channel_tag == "PH2":
                    # Keep the historical filename for compatibility.
                    out_paths.append(FIG_DIR / f"G_wallp_SU_production_{L}.png")
                for out in out_paths:
                    plt.savefig(out, dpi=600)
                    print(f"[ok] wrote {out}")
                plt.close(fig)

def plot_stages_3panel():
    """
    One 3-panel figure per channel (PH1, PH2) for phase1, one panel per
    pressure (0/50/100 psig). Same inner-scaled axes, ROI, models, and
    u_tau uncertainty fan as plot_model_comparison_roi, but with three
    stages overlaid per panel:
      - raw            (grey, dotted)        from wallp_raw
      - FRF-corrected  (pressure colour, dashed)  from frf_corrected_signals
      - Wiener-rejected (pressure colour, solid)  from fs_noise_rejected_signals
    """
    f_cutl, f_cuth = 100.0, 1_000.0
    channel_keys = ("PH1_Pa", "PH2_Pa")

    with h5py.File(cfg.PH_RAW_FILE, "r") as hf_raw, \
         h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf_prod:
        g_raw_root = hf_raw["wallp_raw"]
        g_prod_root = hf_prod["wallp_production"]
        labels = [L for L in LABELS if L in g_prod_root]
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
                colour = COLOURS[i % len(COLOURS)]

                sig_model = g_rej[sp_model][channel_key][:]
                f_model, _ = compute_spec(sig_model, fs=fs, nperseg=NPERSEG)
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

                # ROI for the data line (same as plot_model_comparison_roi).
                f_roi, p_rej = compute_spec(g_rej[sp_roi][channel_key][:],
                                            fs=fs, nperseg=NPERSEG)
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

                # raw (pre-FRF, pre-bandpass)
                if gL_raw is not None and sp_roi in gL_raw and channel_key in gL_raw[sp_roi]:
                    f_r, p_r = compute_spec(gL_raw[sp_roi][channel_key][:],
                                            fs=fs, nperseg=NPERSEG)
                    mr = (f_r > f_cutl) & (f_r < f_cuth)
                    Y_raw = (f_r[mr] * p_r[mr]) / (rho ** 2 * u_nom ** 4)
                    ax.semilogx((u_nom ** 2) / (nu * f_r[mr]), Y_raw,
                                color="#555555", linestyle=":", lw=0.9, alpha=0.7)

                # FRF-corrected (PH->NC->nkd, no Wiener)
                if g_frf is not None and sp_roi in g_frf and channel_key in g_frf[sp_roi]:
                    f_c, p_c = compute_spec(g_frf[sp_roi][channel_key][:],
                                            fs=fs, nperseg=NPERSEG)
                    mc = (f_c > f_cutl) & (f_c < f_cuth)
                    Y_frf = (f_c[mc] * p_c[mc]) / (rho ** 2 * u_nom ** 4)
                    ax.semilogx((u_nom ** 2) / (nu * f_c[mc]), Y_frf,
                                color=colour, linestyle="--", lw=0.9, alpha=0.7)

                # Wiener-rejected (nominal-u_tau data line, on top)
                Y_rej = (f_m * P_rej_m) / (rho ** 2 * u_nom ** 4)
                ax.semilogx(T_nom, Y_rej, color=colour, linewidth=1.2, zorder=10)

                ax.set_title(f"phase1 {L}")
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
            print(f"[ok] wrote {out}")


if __name__ == "__main__":
    # plot_model_comparison_roi()
    plot_stages_3panel()
