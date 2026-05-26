"""
Calibration plots for the bump pipeline.

Per pressure (density):
  - PH calibration: |H1|, |H2|, and the diagnostic fused |H| in dB, plus the
    per-run coherence (gamma^2). Per-channel TFs are what pw_proc applies
    (PH1 -> H1, PH2 -> H2); the fused TF is shown only as a sanity check.
  - NC calibration: |H_fused| and gamma^2 for the NC -> nkd mapping.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.save_bump.config_params import Config
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

cfg = Config()
apply_plot_style()

FIG_DIR = resolve_figure_dir(cfg.ROOT_DIR) / "calibration"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PH_TF_PATH = Path(cfg.TF_BASE) / "PH"
NC_TF_PATH = Path(cfg.TF_BASE) / "NC"


def _mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(z)), 1e-12))


def plot_ph_calibration() -> None:
    pressures = [int(p) for p in cfg.PSIGS]
    for p_si in pressures:
        h5 = PH_TF_PATH / f"calibs_{p_si}.h5"
        if not h5.exists():
            print(f"[skip] missing PH calib {h5}")
            continue
        with h5py.File(h5, "r") as hf:
            f1, H1, g1 = hf["f1"][:], hf["H1"][:], hf["gamma2_1"][:]
            f2, H2, g2 = hf["f2"][:], hf["H2"][:], hf["gamma2_2"][:]
            ff, Hf, gf = hf["frequencies"][:], hf["H_fused"][:], hf["gamma2_fused"][:]
            H1s = hf["H1_smooth"][:] if "H1_smooth" in hf else None
            H2s = hf["H2_smooth"][:] if "H2_smooth" in hf else None

        fig, (ax_mag, ax_coh) = plt.subplots(2, 1, figsize=(6.0, 4.8), tight_layout=True)

        mag_vals: list[np.ndarray] = []
        for f, H, label, color, dashed in (
            (f1, H1,  r"PH1$\to$NC raw",                "#1e8ad8", True),
            (f1, H1s, r"PH1$\to$NC smoothed (applied)", "#1e8ad8", False),
            (f2, H2,  r"PH2$\to$NC raw",                "#ff7f0e", True),
            (f2, H2s, r"PH2$\to$NC smoothed (applied)", "#ff7f0e", False),
            (ff, Hf,  "fused (diagnostic only)",    "#777777", True),
        ):
            if H is None:
                continue
            mask = np.asarray(f) > 0.0
            if not np.any(mask):
                continue
            mag = _mag_db(np.asarray(H)[mask])
            ax_mag.semilogx(
                np.asarray(f)[mask], mag,
                color=color, linewidth=1.0 if not dashed else 0.7,
                alpha=0.45 if dashed else 1.0,
                label=label,
                linestyle="--" if dashed else "-",
            )
            mag_vals.append(mag[np.isfinite(mag) & (mag > -100.0)])

        for f, g, label, color in (
            (f1, g1, r"$\gamma^2_{PH1,NC}$", "#1e8ad8"),
            (f2, g2, r"$\gamma^2_{PH2,NC}$", "#ff7f0e"),
            (ff, gf, r"$\gamma^2$ fused", "#777777"),
        ):
            mask = np.asarray(f) > 0.0
            ax_coh.semilogx(
                np.asarray(f)[mask], np.asarray(g)[mask],
                color=color, linewidth=1.0, label=label,
                linestyle="--" if "fused" in label else "-",
            )

        ax_mag.set_title(f"PH calibration -- bump {p_si} psig")
        ax_mag.set_ylabel(r"$|H(f)|$ [dB]")
        if mag_vals:
            all_mag = np.concatenate(mag_vals)
            if all_mag.size:
                ax_mag.set_ylim(float(all_mag.min()) - 3.0, float(all_mag.max()) + 3.0)
        ax_mag.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
        ax_mag.legend(fontsize=8)

        ax_coh.set_xlabel("$f$ [Hz]")
        ax_coh.set_ylabel(r"$\gamma^2$")
        ax_coh.set_ylim(0.0, 1.05)
        ax_coh.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
        ax_coh.legend(fontsize=8)

        out = FIG_DIR / f"PH_calib_{p_si}psig.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"[ok] {out}")


def plot_nc_calibration() -> None:
    pressures = [int(p) for p in cfg.PSIGS]
    for p_si in pressures:
        h5 = NC_TF_PATH / f"calibs_{p_si}.h5"
        if not h5.exists():
            print(f"[skip] missing NC calib {h5}")
            continue
        with h5py.File(h5, "r") as hf:
            f, H, g = hf["frequencies"][:], hf["H_fused"][:], hf["gamma2_fused"][:]
            Hs = hf["H_smooth"][:] if "H_smooth" in hf else None

        fig, (ax_mag, ax_coh) = plt.subplots(2, 1, figsize=(6.0, 4.8), tight_layout=True)
        f_sq = np.asarray(f).squeeze()
        mask = f_sq > 0.0
        ax_mag.semilogx(f_sq[mask], _mag_db(np.asarray(H).squeeze()[mask]),
                        color="#26bd26", linewidth=0.7, alpha=0.45,
                        linestyle="--", label=r"NC$\to$nkd raw")
        if Hs is not None:
            ax_mag.semilogx(f_sq[mask], _mag_db(np.asarray(Hs).squeeze()[mask]),
                            color="#26bd26", linewidth=1.0,
                            label=r"NC$\to$nkd smoothed (applied)")
        ax_coh.semilogx(np.asarray(f).squeeze()[mask], np.asarray(g).squeeze()[mask],
                        color="#26bd26", linewidth=1.0, label=r"$\gamma^2$ NC,nkd")

        ax_mag.set_title(f"NC calibration -- bump {p_si} psig")
        ax_mag.set_ylabel(r"$|H(f)|$ [dB]")
        ax_mag.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
        ax_mag.legend(fontsize=8)

        ax_coh.set_xlabel("$f$ [Hz]")
        ax_coh.set_ylabel(r"$\gamma^2$")
        ax_coh.set_ylim(0.0, 1.05)
        ax_coh.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
        ax_coh.legend(fontsize=8)

        out = FIG_DIR / f"NC_calib_{p_si}psig.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"[ok] {out}")


def plot_all() -> None:
    plot_ph_calibration()
    plot_nc_calibration()


if __name__ == "__main__":
    plot_all()
