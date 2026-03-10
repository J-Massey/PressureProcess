from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window

from src.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()

apply_plot_style()

# -------------------- constants --------------------
FS = cfg.FS
NPERSEG = 2**12
WINDOW = cfg.WINDOW

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR)
POINT_COLOURS = ("#1e8ad8", "#ff7f0e", "#26bd26", "#d62728")


def compute_spec(x: np.ndarray, fs: float = FS, nperseg: int = NPERSEG):
    """Welch PSD with consistent settings. Returns f [Hz], Pxx [Pa^2/Hz]."""
    x = np.asarray(x, float)
    nseg = min(nperseg, x.size)
    if nseg < 16:
        raise ValueError(f"Signal too short for Welch: n={x.size}, nperseg={nperseg}")
    w = get_window(WINDOW, nseg, fftbins=True)
    f, pxx = welch(
        x,
        fs=fs,
        window=w,
        nperseg=nseg,
        noverlap=nseg // 2,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, pxx


def _scalar_attr(group: h5py.Group, name: str) -> float:
    if name not in group.attrs:
        return float("nan")
    return float(np.atleast_1d(group.attrs[name])[0])


def _collect_point_signals(g_corr: h5py.Group) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for spacing in g_corr.keys():
        g_spacing = g_corr[spacing]
        for channel_key in ("PH1_Pa", "PH2_Pa"):
            if channel_key not in g_spacing:
                continue
            channel = channel_key.split("_")[0]
            x_pos = _scalar_attr(g_spacing, f"x_{channel}")
            points.append(
                {
                    "spacing": spacing,
                    "channel": channel,
                    "x_pos": x_pos,
                    "signal": g_spacing[channel_key][:],
                }
            )

    if not points:
        return points

    if all(np.isfinite(float(p["x_pos"])) for p in points):
        points.sort(key=lambda p: float(p["x_pos"]))
    else:
        points.sort(key=lambda p: (str(p["spacing"]), str(p["channel"])))

    for idx, point in enumerate(points, start=1):
        point["point"] = f"P{idx}"
    return points


def _configure_axes(ax: plt.Axes, *, title: str) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(r"$T^+$")
    ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
    ax.set_xlim(7, 7_000)
    # ax.set_ylim(0, 6)


def plot_cleaned_by_case() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        labels = list(g_root.keys())
        if not labels:
            raise KeyError("No labels found in wallp_production")

        fs = float(hf.attrs.get("fs_Hz", FS))
        for label in labels:
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])

            g_corr = g_label["fs_noise_rejected_signals"]

            point_signals = _collect_point_signals(g_corr)
            if not point_signals:
                print(f"[skip] no corrected spacing groups for {label}")
                continue
            curve_data: list[dict[str, object]] = []
            for point in point_signals:
                f, pxx = compute_spec(np.asarray(point["signal"]), fs=fs, nperseg=NPERSEG)
                mask = f > 0.0
                if not np.any(mask):
                    continue

                point_name = str(point["point"])
                t_plus = (u_tau**2) / (nu * f[mask])
                y = f[mask] * pxx[mask] / (rho**2 * u_tau**4)
                curve_data.append(
                    {
                        "point": point_name,
                        "t_plus": t_plus,
                        "y": y,
                    }
                )
                print(
                    f"[info] {label}: {point_name} <- "
                    f"{point['spacing']}/{point['channel']}"
                )

            if not curve_data:
                print(f"[skip] no usable corrected signals for {label}")
                continue

            fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.2), tight_layout=True)
            for idx, curve in enumerate(curve_data):
                ax.semilogx(
                    np.asarray(curve["t_plus"]),
                    np.asarray(curve["y"]),
                    color=POINT_COLOURS[idx % len(POINT_COLOURS)],
                    linewidth=1.0,
                    label=str(curve["point"]),
                )
            _configure_axes(ax, title=f"{label} (P1-P4)")
            ax.legend(title="Point", fontsize=8)

            combined_outs = [
                FIG_DIR / f"G_wallp_SU_production_{label}_P_all.png",
                FIG_DIR / f"G_wallp_SU_production_{label}.png",
            ]
            for out in combined_outs:
                plt.savefig(out, dpi=600)
                print(f"[ok] wrote {out}")
            plt.close(fig)

            for idx, curve in enumerate(curve_data):
                point_name = str(curve["point"])
                fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.1), tight_layout=True)
                ax.semilogx(
                    np.asarray(curve["t_plus"]),
                    np.asarray(curve["y"]),
                    color=POINT_COLOURS[idx % len(POINT_COLOURS)],
                    linewidth=1.0,
                    label=point_name,
                )
                _configure_axes(ax, title=f"{label} ({point_name})")
                ax.legend(fontsize=8)

                out = FIG_DIR / f"G_wallp_SU_production_{label}_{point_name}.png"
                plt.savefig(out, dpi=600)
                plt.close(fig)
                print(f"[ok] wrote {out}")


if __name__ == "__main__":
    plot_cleaned_by_case()
