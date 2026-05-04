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
FIG_DIR = "figures/fence"


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
                mask = f > 0.0
                if not np.any(mask):
                    continue
                t_plus = (u_tau**2) / (nu * f[mask])
                norm_factor = (rho**2) * (u_tau**4)
                axes[j].loglog(
                    t_plus,
                    f[mask] * pxx[mask] / norm_factor,
                    label=label,
                    color=COLOURS[i % len(COLOURS)],
                )

    for ax in axes:
        ax.legend()
    plt.savefig(FIG_DIR / "freestreamp_bump_raw.png", dpi=600)



def plot_raw():
    with h5py.File("/home/masseyj/Workspace/SAPPHiRe/PressureProcess/data/fence/ATM_Rev1.mat", "r") as f_raw:
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
                mask = f > 0.0
                if not np.any(mask):
                    continue
                t_plus = (u_tau**2) / (nu * f[mask])
                norm_factor = (rho**2) * (u_tau**4)
                axes[j].semilogx(
                    t_plus,
                    f[mask] * pxx[mask] / norm_factor,
                    label=label,
                    color=COLOURS[i % len(COLOURS)],
                )

    for ax in axes:
        ax.legend()
    plt.savefig(FIG_DIR / "wallp_bump_raw.png", dpi=600)


def plot_fence_raw_ts():
    # load .mat, define ph1, ph2 and nc, plot raw time series and spectrafn = "data/fence/ATM_Rev1.mat"
    # Load .mat file and print its structure
    import scipy.io
    fn = "data/fence/ATM_Rev1.mat"
    mat_data = scipy.io.loadmat(fn)
    channelData = mat_data.get("channelData")
    PH1 = channelData[:, 0]
    PH2 = channelData[:, 1]
    NC = channelData[:, 2]
    t = np.arange(channelData.shape[0]) / FS
    fig, ax = plt.subplots()
    ax.plot(t[1000:10000], PH1[1000:10000], lw=0.25)
    ax.plot(t[1000:10000], PH2[1000:10000], lw=0.25)
    ax.plot(t[1000:10000], NC[1000:10000], lw=0.25)
    plt.savefig(FIG_DIR + "/raw_time_series.png", dpi=600)

def plot_fence_raw_spec():
    # load .mat, define ph1, ph2 and nc, plot raw time series and spectrafn = "data/fence/ATM_Rev1.mat"
    # Load .mat file and print its structure
    import scipy.io
    fn = "data/fence/ATM_Rev1.mat"
    mat_data = scipy.io.loadmat(fn)
    channelData = mat_data.get("channelData")
    PH1 = channelData[:, 0]
    PH2 = channelData[:, 1]
    NC = channelData[:, 2]
    f, Pxx_PH1 = compute_spec(PH1, fs=FS, nperseg=NPERSEG)
    f, Pxx_PH2 = compute_spec(PH2, fs=FS, nperseg=NPERSEG)
    f, Pxx_NC = compute_spec(NC, fs=FS, nperseg=NPERSEG)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(f, f*Pxx_PH1, lw=0.7, label="PH1")
    ax.plot(f, f*Pxx_PH2, lw=0.7, label="PH2")
    ax.plot(f, f*Pxx_NC, lw=0.5, label="NC")

    ax.set_ylim(0, 0.01)

    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel(r"$f \phi_{pp}$[V$^2$ s$^{-1}$]")
    ax.set_xscale("log")
    ax.legend()
    plt.savefig(FIG_DIR + "/raw_spectra.png", dpi=600)



if __name__ == "__main__":
    plot_fence_raw_ts()
    plot_fence_raw_spec()
