from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import get_window, welch

from src.checks.plot._style import apply_plot_style
from src.config_params import Config

cfg = Config()

apply_plot_style()

FS = cfg.FS
NPERSEG = 2**13
WINDOW = cfg.WINDOW

DIAG_DIR = Path("figures") / Path(cfg.ROOT_DIR).name
DIAG_DIR.mkdir(parents=True, exist_ok=True)

POINT_COLOURS = ("#0b4eb2", "#d62728", "#26bd26", "#a51990")
FREESTREAM_COLOURS = ("#0b4eb2", "#d62728")
SPACING_ORDER = ("close", "far")
WALL_CHANNELS = ("PH1_Pa", "PH2_Pa")
WALL_LABELS = {
    ("close", "PH1"): "P1",
    ("close", "PH2"): "P2",
    ("far", "PH1"): "P3",
    ("far", "PH2"): "P4",
}
FREESTREAM_LABELS = {
    "close": "P1-P2 run",
    "far": "P3-P4 run",
}
RAW_MAT_COLUMNS = {
    "PH1": 0,
    "PH2": 1,
    "NC": 2,
}


def _ordered_labels(candidates: list[str]) -> list[str]:
    labels = [label for label in cfg.LABELS if label in candidates]
    labels.extend(label for label in candidates if label not in labels)
    return labels


def _channel_name(channel: str) -> str:
    return channel.removesuffix("_Pa").removesuffix("_V")


def _wall_label(spacing: str, channel: str) -> str:
    channel_name = _channel_name(channel)
    return WALL_LABELS.get((spacing, channel_name), f"{spacing}/{channel_name}")


def _raw_mat_path(label: str, spacing: str) -> Path:
    return Path(cfg.RAW_BASE) / spacing / f"{label}.mat"


def _load_raw_mat_channel_data(mat_path: Path) -> np.ndarray:
    mat_obj = sio.loadmat(mat_path)
    if "channelData" not in mat_obj:
        raise KeyError(f"{mat_path} is missing 'channelData'")
    return np.asarray(mat_obj["channelData"], dtype=float)


def compute_spec(x: np.ndarray, fs: float = FS, nperseg: int = NPERSEG):
    """Welch PSD with consistent settings. Returns f [Hz], Pxx [Pa^2/Hz]."""
    x = np.asarray(x, dtype=float)
    nseg = min(nperseg, x.size)
    window = get_window(WINDOW, nseg, fftbins=True)
    f, pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nseg,
        noverlap=nseg // 2,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return f, pxx


def plot_wall_hz_pa() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        for label in labels:
            g_label = g_root[label]
            g_signals = g_label["fs_noise_rejected_signals"]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for spacing in SPACING_ORDER:
                if spacing not in g_signals:
                    continue
                for channel in WALL_CHANNELS:
                    if channel not in g_signals[spacing]:
                        continue

                    f, pxx = compute_spec(g_signals[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.loglog(
                        f[mask],
                        np.sqrt(f[mask] * pxx[mask]),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_hz_pa.png", dpi=300)
            plt.close(fig)


def plot_wall_plus_loglin() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        u_taus = [[[0.5414, 0.5818], [0.5949, 0.5752]],
                  [[0.4836, 0.5615], [0.5221, 0.4986]],
                  [[0.4854, 0.5232], [0.4831, 0.4858]]]

        for i1, label in enumerate(labels):
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            # u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
            # u_tau = u_taus[i]
            print(nu)
            g_signals = g_label["fs_noise_rejected_signals"]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for i2, spacing in enumerate(SPACING_ORDER):
                for i3, channel in enumerate(WALL_CHANNELS):
                    u_tau = u_taus[i1][i2][i3]
                    print(u_tau)

                    f, pxx = compute_spec(g_signals[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.semilogx(
                        (u_tau**2) / (nu * f[mask]),
                        f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel(r"$T^+$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{wall}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_plus_loglin.png", dpi=600)
            plt.close(fig)

def plot_wall_plus_loglog() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        u_taus = [[[0.5414, 0.5818], [0.5949, 0.5752]],
                  [[0.4836, 0.5615], [0.5221, 0.4986]],
                  [[0.4854, 0.5232], [0.4831, 0.4858]]]

        for i1, label in enumerate(labels):
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            # u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
            # u_tau = u_taus[i]
            print(nu)
            g_signals = g_label["fs_noise_rejected_signals"]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for i2, spacing in enumerate(SPACING_ORDER):
                for i3, channel in enumerate(WALL_CHANNELS):
                    u_tau = u_taus[i1][i2][i3]
                    print(u_tau)

                    f, pxx = compute_spec(g_signals[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.loglog(
                        (u_tau**2) / (nu * f[mask]),
                        f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel(r"$T^+$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{wall}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_plus_loglog.png", dpi=600)
            plt.close(fig)


def plot_wall_out_loglin() -> None:
    """
    Outer-scale wall pressure 
    """
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        deltas = [[[55.76e-3, 42.63e-3], [38.82e-3, 39.36e-3]],]
        Ues = [[[14.0, 14.0], [14.0, 14.0]]]

        u_taus = [[[0.5414, 0.5818], [0.5949, 0.5752]],
                  [[0.4836, 0.5615], [0.5221, 0.4986]],
                  [[0.4854, 0.5232], [0.4831, 0.4858]]]

        for i1, label in enumerate(labels[:1]):
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            # u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
            # u_tau = u_taus[i]
            print(nu)
            g_signals = g_label["fs_noise_rejected_signals"]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for i2, spacing in enumerate(SPACING_ORDER):
                for i3, channel in enumerate(WALL_CHANNELS):
                    u_tau = u_taus[i1][i2][i3]
                    delta = deltas[i1][i2][i3]
                    Ue = Ues[i1][i2][i3]

                    f, pxx = compute_spec(g_signals[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.semilogx(
                        (Ue) / (delta * f[mask]),
                        f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel(r"$T^o$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{wall}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_out_loglin.png", dpi=600)
            plt.close(fig)

def plot_wall_out_loglog() -> None:
    """
    Outer-scale wall pressure 
    """
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        deltas = [[[55.76e-3, 42.63e-3], [38.82e-3, 39.36e-3]],]
        Ues = [[[14.0, 14.0], [14.0, 14.0]]]

        u_taus = [[[0.5414, 0.5818], [0.5949, 0.5752]],
                  [[0.4836, 0.5615], [0.5221, 0.4986]],
                  [[0.4854, 0.5232], [0.4831, 0.4858]]]

        for i1, label in enumerate(labels[:1]):
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            # u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
            # u_tau = u_taus[i]
            print(nu)
            g_signals = g_label["fs_noise_rejected_signals"]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for i2, spacing in enumerate(SPACING_ORDER):
                for i3, channel in enumerate(WALL_CHANNELS):
                    u_tau = u_taus[i1][i2][i3]
                    delta = deltas[i1][i2][i3]
                    Ue = Ues[i1][i2][i3]

                    f, pxx = compute_spec(g_signals[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.loglog(
                        (Ue) / (delta * f[mask]),
                        f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel(r"$T^o$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{wall}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_out_loglog.png", dpi=600)
            plt.close(fig)



def plot_wall_bump_loglin() -> None:
    """
    Outer-scale wall pressure 
    """
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        deltas = [[[55.76e-3, 42.63e-3], [38.82e-3, 39.36e-3]],]
        Ues = [[[14.0, 14.0], [14.0, 14.0]]]

        u_taus = [[[0.5414, 0.5818], [0.5949, 0.5752]],
                  [[0.4836, 0.5615], [0.5221, 0.4986]],
                  [[0.4854, 0.5232], [0.4831, 0.4858]]]

        for i1, label in enumerate(labels[:1]):
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            # u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
            # u_tau = u_taus[i]
            print(nu)
            g_signals = g_label["fs_noise_rejected_signals"]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for i2, spacing in enumerate(SPACING_ORDER):
                for i3, channel in enumerate(WALL_CHANNELS):
                    u_tau = u_taus[i1][i2][i3]
                    delta = deltas[i1][i2][i3]
                    Ue = Ues[i1][i2][i3]

                    f, pxx = compute_spec(g_signals[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.semilogx(
                        (Ue) / (delta * f[mask]),
                        f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel(r"$T^o$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{wall}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_bump_loglin.png", dpi=600)
            plt.close(fig)

def plot_wall_bump_loglog() -> None:
    """
    Outer-scale wall pressure 
    """
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        deltas = [[[20e-3, 20e-3], [20e-3, 20e-3]],]
        Ues = [[[14.0, 14.0], [14.0, 14.0]]]

        u_taus = [[[0.5414, 0.5818], [0.5949, 0.5752]],
                  [[0.4836, 0.5615], [0.5221, 0.4986]],
                  [[0.4854, 0.5232], [0.4831, 0.4858]]]

        for i1, label in enumerate(labels[:1]):
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            # u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
            # u_tau = u_taus[i]
            print(nu)
            g_signals = g_label["fs_noise_rejected_signals"]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for i2, spacing in enumerate(SPACING_ORDER):
                for i3, channel in enumerate(WALL_CHANNELS):
                    u_tau = u_taus[i1][i2][i3]
                    delta = deltas[i1][i2][i3]
                    Ue = Ues[i1][i2][i3]

                    f, pxx = compute_spec(g_signals[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.loglog(
                        (Ue) / (delta * f[mask]),
                        f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel(r"$T^o$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{wall}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_bump_loglog.png", dpi=600)
            plt.close(fig)


def plot_wall_raw_hz_pa() -> None:
    with h5py.File(cfg.PH_RAW_FILE, "r") as hf:
        g_root = hf["wallp_raw"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        for label in labels:
            g_label = g_root[label]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for spacing in SPACING_ORDER:
                if spacing not in g_label:
                    continue
                for channel in WALL_CHANNELS:
                    if channel not in g_label[spacing]:
                        continue

                    f, pxx = compute_spec(g_label[spacing][channel][:], fs=fs)
                    mask = f > 0.0
                    if not np.any(mask):
                        continue

                    ax.loglog(
                        f[mask],
                        np.sqrt(f[mask] * pxx[mask]),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=_wall_label(spacing, channel),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall raw")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_raw_hz_pa.png", dpi=300)
            plt.close(fig)


def plot_freestream_hz_pa() -> None:
    with h5py.File(cfg.NKD_PROCESSED_FILE, "r") as hf:
        g_root = hf["freestream_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        for label in labels:
            g_label = g_root[label]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for spacing in SPACING_ORDER:
                if spacing not in g_label or "NC_Pa" not in g_label[spacing]:
                    continue

                f, pxx = compute_spec(g_label[spacing]["NC_Pa"][:], fs=fs)
                mask = f > 0.0
                if not np.any(mask):
                    continue

                ax.loglog(
                    f[mask],
                    np.sqrt(f[mask] * pxx[mask]),
                    color=FREESTREAM_COLOURS[colour_idx % len(FREESTREAM_COLOURS)],
                    linewidth=1.0,
                    label=FREESTREAM_LABELS.get(spacing, spacing),
                )
                colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} freestream production")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_freestream_hz_pa.png", dpi=300)
            plt.close(fig)


def plot_freestream_plus() -> None:
    with h5py.File(cfg.NKD_PROCESSED_FILE, "r") as hf:
        g_root = hf["freestream_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        for label in labels:
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for spacing in SPACING_ORDER:
                if spacing not in g_label or "NC_Pa" not in g_label[spacing]:
                    continue

                f, pxx = compute_spec(g_label[spacing]["NC_Pa"][:], fs=fs)
                mask = f > 0.0
                if not np.any(mask):
                    continue

                ax.semilogx(
                    (u_tau**2) / (nu * f[mask]),
                    f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                    color=FREESTREAM_COLOURS[colour_idx % len(FREESTREAM_COLOURS)],
                    linewidth=1.0,
                    label=FREESTREAM_LABELS.get(spacing, spacing),
                )
                colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} freestream production")
            ax.set_xlabel(r"$T^+$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{fs}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_freestream_plus.png", dpi=300)
            plt.close(fig)


def plot_freestream_raw_hz_pa() -> None:
    with h5py.File(cfg.NKD_RAW_FILE, "r") as hf:
        g_root = hf["freestream_raw"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = _ordered_labels(list(g_root.keys()))

        for label in labels:
            g_label = g_root[label]

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
            colour_idx = 0
            for spacing in SPACING_ORDER:
                if spacing not in g_label or "NC_Pa" not in g_label[spacing]:
                    continue

                f, pxx = compute_spec(g_label[spacing]["NC_Pa"][:], fs=fs)
                mask = f > 0.0
                if not np.any(mask):
                    continue

                ax.loglog(
                    f[mask],
                    np.sqrt(f[mask] * pxx[mask]),
                    color=FREESTREAM_COLOURS[colour_idx % len(FREESTREAM_COLOURS)],
                    linewidth=1.0,
                    label=FREESTREAM_LABELS.get(spacing, spacing),
                )
                colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} freestream raw")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_freestream_raw_hz_pa.png", dpi=300)
            plt.close(fig)


def plot_raw_mat_hz_v() -> None:
    labels = _ordered_labels(
        [
            label
            for label in cfg.LABELS
            if any(_raw_mat_path(label, spacing).exists() for spacing in SPACING_ORDER)
        ]
    )

    for label in labels:
        wall_fig, wall_ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
        fs_fig, fs_ax = plt.subplots(1, 1, figsize=(4.2, 3.4), tight_layout=True)
        wall_colour_idx = 0
        fs_colour_idx = 0
        saw_wall = False
        saw_freestream = False

        for spacing in SPACING_ORDER:
            mat_path = _raw_mat_path(label, spacing)
            if not mat_path.exists():
                continue

            signals_v = _load_raw_mat_channel_data(mat_path)

            for channel_name in ("PH1", "PH2"):
                f, pxx = compute_spec(signals_v[:, RAW_MAT_COLUMNS[channel_name]], fs=FS)
                mask = f > 0.0
                if not np.any(mask):
                    continue

                wall_ax.loglog(
                    f[mask],
                    np.sqrt(f[mask] * pxx[mask]),
                    color=POINT_COLOURS[wall_colour_idx % len(POINT_COLOURS)],
                    linewidth=1.0,
                    label=_wall_label(spacing, f"{channel_name}_V"),
                )
                wall_colour_idx += 1
                saw_wall = True

            f, pxx = compute_spec(signals_v[:, RAW_MAT_COLUMNS["NC"]], fs=FS)
            mask = f > 0.0
            if np.any(mask):
                fs_ax.loglog(
                    f[mask],
                    np.sqrt(f[mask] * pxx[mask]),
                    color=FREESTREAM_COLOURS[fs_colour_idx % len(FREESTREAM_COLOURS)],
                    linewidth=1.0,
                    label=FREESTREAM_LABELS.get(spacing, spacing),
                )
                fs_colour_idx += 1
                saw_freestream = True

        if saw_wall:
            wall_ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            wall_ax.set_title(f"{label} wall raw (.mat)")
            wall_ax.set_xlabel("f [Hz]")
            wall_ax.set_ylabel(r"$\sqrt{f \phi_{vv}}$ [V]")
            wall_ax.legend(fontsize=8)
            wall_fig.savefig(DIAG_DIR / f"{label}_wall_raw_mat_hz_v.png", dpi=300)
        plt.close(wall_fig)

        if saw_freestream:
            fs_ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            fs_ax.set_title(f"{label} freestream raw (.mat)")
            fs_ax.set_xlabel("f [Hz]")
            fs_ax.set_ylabel(r"$\sqrt{f \phi_{vv}}$ [V]")
            fs_ax.legend(fontsize=8)
            fs_fig.savefig(DIAG_DIR / f"{label}_freestream_raw_mat_hz_v.png", dpi=300)
        plt.close(fs_fig)


def plot_cleaned_by_case() -> None:
    # plot_wall_hz_pa()
    # plot_wall_plus_loglin()
    # plot_wall_plus_loglog()
    # plot_wall_out_loglin()
    # plot_wall_out_loglog()
    plot_wall_bump_loglin()
    plot_wall_bump_loglog()
    # plot_wall_raw_hz_pa()
    # plot_freestream_hz_pa()
    # plot_freestream_plus()
    # plot_freestream_raw_hz_pa()
    # plot_raw_mat_hz_v()


if __name__ == "__main__":
    plot_cleaned_by_case()
