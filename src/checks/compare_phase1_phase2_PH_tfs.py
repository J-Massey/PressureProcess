"""
Side-by-side comparison of the PH calibration TFs actually applied in pw_proc
for phase1 (fused, single H per pressure) and phase2 (per-channel H1, H2).

Layout: 3 rows (0, 50, 100 psig) x 1 col. Magnitude only (dB). The phase1
TF is what gets applied to both PH1 and PH2; the phase2 TFs are applied
per-channel (H1_smooth -> PH1, H2_smooth -> PH2). Phase2's H_fused is also
overlaid (dashed grey) as a diagnostic equivalent of phase1's applied TF.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from src.checks.plot._style import apply_plot_style, resolve_figure_dir


PSIGS = (0, 50, 100)
PHASE1_CAL = Path("data/phase1/calibration/PH")
PHASE2_CAL = Path("data/phase2/calibration/PH")
OUT_DIR = Path("figures/comparisons/phase1_vs_phase2")


def _mag_db(z: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(z)), 1e-12))


def _load_phase1(p_si: int):
    path = PHASE1_CAL / f"calibs_{p_si}.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as hf:
        # phase1 applies the fused TF (and saves the same TF for all
        # pressures, since pw_proc reads calibs_{psig}.h5 per pressure).
        f = np.asarray(hf["frequencies"][:]).squeeze()
        H_fused = np.asarray(hf["H_fused"][:]).squeeze()
    return f, H_fused


def _load_phase2(p_si: int):
    path = PHASE2_CAL / f"calibs_{p_si}.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as hf:
        f1 = np.asarray(hf["f1"][:]).squeeze()
        f2 = np.asarray(hf["f2"][:]).squeeze()
        H1 = np.asarray(hf["H1_smooth" if "H1_smooth" in hf else "H1"][:]).squeeze()
        H2 = np.asarray(hf["H2_smooth" if "H2_smooth" in hf else "H2"][:]).squeeze()
        ff = np.asarray(hf["frequencies"][:]).squeeze()
        Hf = np.asarray(hf["H_fused"][:]).squeeze()
    return f1, H1, f2, H2, ff, Hf


def plot_compare(*, zoom_roi: bool = False) -> None:
    apply_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(PSIGS), 1, figsize=(6.0, 2.6 * len(PSIGS)),
        sharex=True, tight_layout=True,
    )

    for i, p_si in enumerate(PSIGS):
        ax = axes[i]
        p1 = _load_phase1(p_si)
        p2 = _load_phase2(p_si)
        if p1 is None and p2 is None:
            ax.set_visible(False)
            continue

        mag_pool: list[np.ndarray] = []

        if p1 is not None:
            f, Hf = p1
            mask = f > 0.0
            mag = _mag_db(Hf[mask])
            ax.semilogx(f[mask], mag,
                        color="#111111", linewidth=1.2,
                        label="phase1 H_fused (applied)")
            mag_pool.append(mag[np.isfinite(mag)])

        if p2 is not None:
            f1, H1, f2, H2, ff, Hf = p2
            for fch, Hch, lbl, col in (
                (f1, H1, "phase2 H1_smooth $\\to$ PH1 (applied)", "#1e8ad8"),
                (f2, H2, "phase2 H2_smooth $\\to$ PH2 (applied)", "#ff7f0e"),
            ):
                mask = fch > 0.0
                mag = _mag_db(Hch[mask])
                ax.semilogx(fch[mask], mag,
                            color=col, linewidth=1.0, label=lbl)
                mag_pool.append(mag[np.isfinite(mag)])

            # phase2's fused TF as a diagnostic comparison against phase1's.
            mask = ff > 0.0
            mag = _mag_db(Hf[mask])
            ax.semilogx(ff[mask], mag,
                        color="#777777", linewidth=0.8, alpha=0.7,
                        linestyle="--",
                        label="phase2 H_fused (diagnostic)")
            mag_pool.append(mag[np.isfinite(mag)])

        ax.set_title(f"{p_si} psig")
        ax.set_ylabel(r"$|H_{PH \to NC}(f)|$ [dB]")
        ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
        if zoom_roi:
            ax.set_xlim(100.0, 1000.0)
            ax.set_ylim(-25.0, 10.0)
        elif mag_pool:
            all_mag = np.concatenate(mag_pool)
            if all_mag.size:
                ax.set_ylim(float(all_mag.min()) - 3.0, float(all_mag.max()) + 3.0)
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")

    axes[-1].set_xlabel("$f$ [Hz]")
    suffix = " (zoom: 100-1000 Hz ROI)" if zoom_roi else ""
    fig.suptitle(f"PH calibration TFs: phase1 vs phase2{suffix}", fontsize=11)

    out_name = "PH_tf_phase1_vs_phase2_ROI.png" if zoom_roi else "PH_tf_phase1_vs_phase2.png"
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    plot_compare(zoom_roi=False)
    plot_compare(zoom_roi=True)
