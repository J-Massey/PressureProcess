from __future__ import annotations

from icecream import ic

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window, welch

from src.checks.plot._style import apply_plot_style, resolve_figure_dir
from src.config_params import Config

cfg = Config()

apply_plot_style()

# -------------------- constants --------------------
FS = cfg.FS
NPERSEG = 2**12
WINDOW = cfg.WINDOW

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR)
ic(cfg.ROOT_DIR, FIG_DIR)

POINT_COLOURS = ("#0b4eb2", "#d62728", "#26bd26", "#a51990")
SPACING_ORDER = ("close", "far")
CHANNEL_ORDER = ("PH1", "PH2")


def compute_spec(x: np.ndarray, fs: float = FS, nperseg: int = NPERSEG):
    """Welch PSD with consistent settings. Returns f [Hz], Pxx [Pa^2/Hz]."""
    x = np.asarray(x, dtype=float)
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


def _configure_axes(ax: plt.Axes, *, title: str) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(r"$T^+$")
    ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{corr.}}$")
    ax.set_xlim(7, 7_000)
    # ax.set_ylim(0, 6)


def _iter_point_signals(g_corr: h5py.Group):
    ordered_spacings = [spacing for spacing in SPACING_ORDER if spacing in g_corr]
    ordered_spacings.extend(
        spacing for spacing in g_corr.keys() if spacing not in ordered_spacings
    )

    for spacing in ordered_spacings:
        g_spacing = g_corr[spacing]
        for channel in CHANNEL_ORDER:
            key = f"{channel}_Pa"
            if key in g_spacing:
                yield spacing, channel, np.asarray(g_spacing[key][:], dtype=float)


def _compute_inner_curve(
    signal: np.ndarray, *, fs: float, rho: float, u_tau: float, nu: float, f_cut: float
) -> tuple[np.ndarray, np.ndarray] | None:

    f, pxx = compute_spec(signal, fs=fs, nperseg=NPERSEG)

    mask = f > 0.0
    if not np.any(mask):
        return None

    f = f[mask]
    pxx = pxx[mask]

    t_plus = (u_tau**2) / (nu * f)
    y = f * pxx / (rho**2 * u_tau**4)

    t_plus_cut = (u_tau**2) / (nu * f_cut)
    ic(f_cut, t_plus_cut)

    mask = t_plus >= t_plus_cut
    return t_plus[mask], y[mask]


def plot_cleaned_by_case() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        ic(g_root.keys())

        if not list(g_root.keys()):
            raise KeyError("No labels found in wallp_production")

        fs = float(hf.attrs.get("fs_Hz", FS))

        for i, (label, g_label) in enumerate(g_root.items()):
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
            f_cut = [20, 70, 200][i]
            ic(f_cut)

            g_corr = g_label["fs_noise_rejected_signals"]
            point_signals = list(_iter_point_signals(g_corr))
            if not point_signals:
                print(f"[skip] no corrected spacing groups for {label}")
                continue

            curve_data: list[tuple[str, np.ndarray, np.ndarray]] = []
            for idx, (spacing, channel, signal) in enumerate(point_signals, start=1):
                try:
                    curve = _compute_inner_curve(
                        signal,
                        fs=fs,
                        rho=rho,
                        u_tau=u_tau,
                        nu=nu,
                        f_cut=f_cut   # i = label index
                    )
                except ValueError:
                    print(f"[skip] {label}: {spacing}/{channel} signal too short")
                    continue
                if curve is None:
                    continue

                point_name = f"P{idx}"
                curve_data.append((point_name, curve[0], curve[1]))
                print(f"[info] {label}: {point_name} <- {spacing}/{channel}")

            if not curve_data:
                print(f"[skip] no usable corrected signals for {label}")
                continue

            fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0), tight_layout=True)
            for idx, (point_name, t_plus, y) in enumerate(curve_data):
                ax.semilogx(
                    t_plus,
                    y,
                    color=POINT_COLOURS[idx % len(POINT_COLOURS)],
                    linewidth=1.0,
                    label=point_name,
                )
            _configure_axes(ax, title=f"{label} (P1-P4)")
            ax.legend(title="Point", fontsize=8)

            fig.savefig(FIG_DIR / f"G_wallp_SU_production_{label}_P_all.png", dpi=600)
            plt.close(fig)

            # for idx, (point_name, t_plus, y) in enumerate(curve_data):
            #     fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0), tight_layout=True)
            #     ax.semilogx(
            #         t_plus,
            #         y,
            #         color=POINT_COLOURS[idx % len(POINT_COLOURS)],
            #         linewidth=1.0,
            #         label=point_name,
            #     )
            #     _configure_axes(ax, title=f"{label} ({point_name})")
            #     ax.legend(fontsize=8)

            #     output = FIG_DIR / f"G_wallp_SU_production_{label}_{point_name}.png"
            #     fig.savefig(output, dpi=600)
            #     plt.close(fig)
            #     print(f"[ok] wrote {output}")


if __name__ == "__main__":
    plot_cleaned_by_case()
