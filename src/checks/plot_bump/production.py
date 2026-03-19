from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window, welch

from src.checks.plot._style import apply_plot_style
from src.config_params import Config

cfg = Config()

apply_plot_style()

FS = cfg.FS
NPERSEG = 2**13
WINDOW = cfg.WINDOW

DIAG_DIR = Path("figures") / Path(cfg.ROOT_DIR).name / "diag"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

POINT_COLOURS = ("#0b4eb2", "#d62728", "#26bd26", "#a51990")
FREESTREAM_COLOURS = ("#0b4eb2", "#d62728")
SPACING_ORDER = ("close", "far")
WALL_CHANNELS = ("PH1_Pa", "PH2_Pa")
WALL_LABELS = {
    ("close", "PH1_Pa"): "PH1",
    ("close", "PH2_Pa"): "PH2",
    ("far", "PH1_Pa"): "PH3",
    ("far", "PH2_Pa"): "PH4",
}
FREESTREAM_LABELS = {
    "close": "PH1-PH2 run",
    "far": "PH3-PH4 run",
}


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
        labels = [label for label in cfg.LABELS if label in g_root]
        labels.extend(label for label in g_root.keys() if label not in labels)

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
                        label=WALL_LABELS.get(
                            (spacing, channel), f"{spacing}/{channel.removesuffix('_Pa')}"
                        ),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel(r"$\sqrt{f \phi_{pp}}$ [Pa]")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_hz_pa.png", dpi=300)
            plt.close(fig)


def plot_wall_plus() -> None:
    with h5py.File(cfg.PH_PROCESSED_FILE, "r") as hf:
        g_root = hf["wallp_production"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = [label for label in cfg.LABELS if label in g_root]
        labels.extend(label for label in g_root.keys() if label not in labels)

        for label in labels:
            g_label = g_root[label]
            rho = float(np.atleast_1d(g_label.attrs["rho"])[0])
            u_tau = float(np.atleast_1d(g_label.attrs["u_tau"])[0])
            nu = float(np.atleast_1d(g_label.attrs["nu"])[0])
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

                    ax.semilogx(
                        (u_tau**2) / (nu * f[mask]),
                        f[mask] * pxx[mask] / (rho**2 * u_tau**4),
                        color=POINT_COLOURS[colour_idx % len(POINT_COLOURS)],
                        linewidth=1.0,
                        label=WALL_LABELS.get(
                            (spacing, channel), f"{spacing}/{channel.removesuffix('_Pa')}"
                        ),
                    )
                    colour_idx += 1

            ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
            ax.set_title(f"{label} wall production")
            ax.set_xlabel(r"$T^+$")
            ax.set_ylabel(r"$({f \phi_{pp}}^+)_{\mathrm{wall}}$")
            ax.legend(fontsize=8)
            fig.savefig(DIAG_DIR / f"{label}_wall_plus.png", dpi=300)
            plt.close(fig)


def plot_wall_raw_hz_pa() -> None:
    with h5py.File(cfg.PH_RAW_FILE, "r") as hf:
        g_root = hf["wallp_raw"]
        fs = float(hf.attrs.get("fs_Hz", FS))
        labels = [label for label in cfg.LABELS if label in g_root]
        labels.extend(label for label in g_root.keys() if label not in labels)

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
                        label=WALL_LABELS.get(
                            (spacing, channel), f"{spacing}/{channel.removesuffix('_Pa')}"
                        ),
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
        labels = [label for label in cfg.LABELS if label in g_root]
        labels.extend(label for label in g_root.keys() if label not in labels)

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
        labels = [label for label in cfg.LABELS if label in g_root]
        labels.extend(label for label in g_root.keys() if label not in labels)

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
        labels = [label for label in cfg.LABELS if label in g_root]
        labels.extend(label for label in g_root.keys() if label not in labels)

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


def plot_cleaned_by_case() -> None:
    plot_wall_hz_pa()
    plot_wall_plus()
    plot_wall_raw_hz_pa()
    plot_freestream_hz_pa()
    plot_freestream_plus()
    plot_freestream_raw_hz_pa()


if __name__ == "__main__":
    plot_cleaned_by_case()
