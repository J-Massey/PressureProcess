import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
from icecream import ic

from src.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()

apply_plot_style()

# -------------------- constants --------------------
FS = cfg.FS
NPERSEG = 2**14          # keep one value for all runs
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

def plot_fs_raw():
    with h5py.File(cfg.NKD_RAW_FILE, "r") as f_raw:
        g_root = f_raw["freestream_raw"]
        labels = list(g_root.keys())
        if not labels:
            raise KeyError("No labels found in freestream_raw")

        sample = g_root[labels[0]]
        spacing_order = [sp for sp in ("close", "far") if sp in sample]
        if not spacing_order:
            spacing_order = list(sample.keys())
        if not spacing_order:
            raise KeyError("No spacing groups found in freestream_raw")

        fig, axes = plt.subplots(
            1,
            len(spacing_order),
            figsize=(3 * len(spacing_order), 3),
            sharey=True,
        )
        axes = np.atleast_1d(axes)
        for ax, sp in zip(axes, spacing_order):
            ax.set_title(f"NC--{sp} run")
            ax.set_xlabel(r"$T^+$")
        axes[0].set_ylabel(r"${f \phi_{pp}}_{\mathrm{raw}}^+$")

        for i, label in enumerate(labels):
            g_label = g_root[label]
            ic(g_label.attrs.keys())
            rho = g_label.attrs["rho"][()]
            u_tau = g_label.attrs["u_tau"][()]
            nu = g_label.attrs["nu"][()]

            for j, sp in enumerate(spacing_order):
                if sp not in g_label:
                    continue
                nc_raw = g_label[f"{sp}/NC_Pa"][:]
                f, pxx = compute_spec(nc_raw, fs=FS, nperseg=NPERSEG)
                t_plus = (u_tau**2) / (nu * f)
                norm_factor = (rho**2) * (u_tau**4)
                axes[j].loglog(
                    t_plus,
                    f * pxx / norm_factor,
                    label=label,
                    color=COLOURS[i % len(COLOURS)],
                )

    for ax in axes:
        ax.legend()
    plt.savefig(FIG_DIR / "freestreamp_bump_raw.png", dpi=600)


def plot_raw():
    with h5py.File(cfg.PH_RAW_FILE, "r") as f_raw:
        g_root = f_raw["wallp_raw"]
        labels = list(g_root.keys())
        if not labels:
            raise KeyError("No labels found in wallp_raw")

        sample = g_root[labels[0]]
        spacing_order = [sp for sp in ("close", "far") if sp in sample]
        if not spacing_order:
            spacing_order = list(sample.keys())
        if not spacing_order:
            raise KeyError("No spacing groups found in wallp_raw")

        fig, axes = plt.subplots(
            1,
            len(spacing_order),
            figsize=(3 * len(spacing_order), 3),
            sharey=True,
        )
        axes = np.atleast_1d(axes)
        for ax, sp in zip(axes, spacing_order):
            ax.set_title(f"PH2--{sp} run")
            ax.set_xlabel(r"$T^+$")
        axes[0].set_ylabel(r"${f \phi_{pp}}_{\mathrm{raw}}^+$")
        axes[0].set_ylim(0, 15)

        for i, label in enumerate(labels):
            g_label = g_root[label]
            ic(g_label.attrs.keys())
            rho = g_label.attrs["rho"][()]
            u_tau = g_label.attrs["u_tau"][()]
            nu = g_label.attrs["nu"][()]

            for j, sp in enumerate(spacing_order):
                if sp not in g_label:
                    continue
                ph2_raw = g_label[f"{sp}/PH2_Pa"][:]
                f, pxx = compute_spec(ph2_raw, fs=FS, nperseg=NPERSEG)
                t_plus = (u_tau**2) / (nu * f)
                norm_factor = (rho**2) * (u_tau**4)
                axes[j].semilogx(
                    t_plus,
                    f * pxx / norm_factor,
                    label=label,
                    color=COLOURS[i % len(COLOURS)],
                )

    for ax in axes:
        ax.legend()
    plt.savefig(FIG_DIR / "wallp_bump_rw.png", dpi=600)


if __name__ == "__main__":
    plot_fs_raw()
    plot_raw()
