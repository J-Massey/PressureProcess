# Per-pressure physical FRFs (PH to NC) from semi-anechoic runs — phase2 variant.
# PH calibrations kept per-channel (no fusion); H_fused stored only as a
# diagnostic backup so plots can show it dashed.
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from src.core.phys_helpers import volts_to_pa
from src.core.pressure_sensitivity import correct_pressure_sensitivity
from src.core.tf_definition import estimate_frf, combine_anechoic_calibrations, smooth_frf_logfreq
from src.checks.plot._style import apply_plot_style, resolve_figure_dir

from src.save_phase2.config_params import Config

cfg = Config()


def _extract_channel_data(mat_obj: dict) -> object:
    if "channelData_WN" in mat_obj:
        return mat_obj["channelData_WN"]
    if "channelData" in mat_obj:
        return mat_obj["channelData"]
    raise KeyError("Expected one of calibration keys: channelData_WN, channelData")


def _save_ph_tf_plot(
    *,
    p_si: int,
    f1: np.ndarray,
    H1: np.ndarray,
    f2: np.ndarray,
    H2: np.ndarray,
    f_fused: np.ndarray,
    H_fused: np.ndarray,
    g2_fused: np.ndarray,
) -> None:
    apply_plot_style()
    fig_dir = resolve_figure_dir(Path(cfg.ROOT_DIR).name) / "calibration"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_mag, ax_coh) = plt.subplots(2, 1, figsize=(6.0, 4.8), tight_layout=True)

    for f_i, h_i, label, color in (
        (f1, H1, "PH1-NC", "#1e8ad8"),
        (f2, H2, "PH2-NC", "#ff7f0e"),
        (f_fused, H_fused, "Fused (diagnostic)", "#111111"),
    ):
        mask = np.asarray(f_i) > 0.0
        if not np.any(mask):
            continue
        mag_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(h_i)[mask]), 1e-12))
        ax_mag.semilogx(np.asarray(f_i)[mask], mag_db, color=color, linewidth=1.0, label=label)

    mask_fused = np.asarray(f_fused) > 0.0
    if np.any(mask_fused):
        ax_coh.semilogx(
            np.asarray(f_fused)[mask_fused],
            np.asarray(g2_fused)[mask_fused],
            color="#26bd26",
            linewidth=1.0,
            label="Fused coherence",
        )

    ax_mag.set_title(f"PH Calibration TF Check ({p_si} psig) [phase2]")
    ax_mag.set_ylabel(r"$|H(f)|$ [dB]")
    ax_mag.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
    ax_mag.legend(fontsize=8)
    ax_mag.set_ylim(-7.0, 5.0)

    ax_coh.set_xlabel("$f$ [Hz]")
    ax_coh.set_ylabel(r"$\gamma^2$")
    ax_coh.set_ylim(0.0, 1.05)
    ax_coh.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.7)
    ax_coh.legend(fontsize=8)

    out = fig_dir / f"PH_tf_{p_si}psig.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[ok] TF check plot {out}")


def save_PH_calibs(
    *,
    gmin: float = 0.4,
    smooth_oct: float = 1 / 6,
    points_per_oct: int = 32,
    eps: float = 1e-12,
) -> None:
    base = Path(cfg.RAW_CAL_BASE) / "PH"
    out_dir = Path(cfg.TF_BASE) / "PH"
    out_dir.mkdir(parents=True, exist_ok=True)
    pressures = [int(p) for p in cfg.PSIGS]

    for p_si, fcut in zip(pressures, cfg.F_CUTS):
        psig = float(p_si)
        m1 = loadmat(base / f"calib_{p_si}psig_1.mat")
        ph1_v, _, nc_v, *_ = _extract_channel_data(m1).T
        ph1_pa = volts_to_pa(ph1_v, "PH1")
        nc1_pa = volts_to_pa(nc_v, "NC")
        ph1_pa = correct_pressure_sensitivity(ph1_pa, psig)
        nc1_pa = correct_pressure_sensitivity(nc1_pa, psig)
        f1, H1, g2_1 = estimate_frf(
            ph1_pa, nc1_pa,
            fs=cfg.FS, window=cfg.WINDOW, nperseg=cfg.NPERSEG,
        )

        m2 = loadmat(base / f"calib_{p_si}psig_2.mat")
        _, ph2_v, nc_v2, *_ = _extract_channel_data(m2).T
        ph2_pa = volts_to_pa(ph2_v, "PH2")
        nc2_pa = volts_to_pa(nc_v2, "NC")
        ph2_pa = correct_pressure_sensitivity(ph2_pa, psig)
        nc2_pa = correct_pressure_sensitivity(nc2_pa, psig)
        f2, H2, g2_2 = estimate_frf(
            ph2_pa, nc2_pa,
            fs=cfg.FS, window=cfg.WINDOW, nperseg=cfg.NPERSEG,
        )

        # Fused TF kept only as a diagnostic so plots can show what fusion
        # would have produced. pw_proc applies H1/H2 per-channel.
        f_fused, H_fused, g2_fused = combine_anechoic_calibrations(
            f1, H1, g2_1, f2, H2, g2_2,
            gmin=gmin, smooth_oct=smooth_oct, points_per_oct=points_per_oct, eps=eps,
        )

        H1_s = smooth_frf_logfreq(
            f1, H1, span_oct=cfg.TF_SMOOTH_OCT, ppo=cfg.TF_SMOOTH_PPO,
        )
        H2_s = smooth_frf_logfreq(
            f2, H2, span_oct=cfg.TF_SMOOTH_OCT, ppo=cfg.TF_SMOOTH_PPO,
        )

        out = out_dir / f"calibs_{p_si}.h5"
        with h5py.File(out, "w") as hf:
            hf.create_dataset("f1", data=f1)
            hf.create_dataset("H1", data=H1)
            hf.create_dataset("H1_smooth", data=H1_s)
            hf.create_dataset("gamma2_1", data=g2_1)
            hf.create_dataset("f2", data=f2)
            hf.create_dataset("H2", data=H2)
            hf.create_dataset("H2_smooth", data=H2_s)
            hf.create_dataset("gamma2_2", data=g2_2)
            hf.create_dataset("frequencies", data=f_fused)
            hf.create_dataset("H_fused", data=H_fused)
            hf.create_dataset("gamma2_fused", data=g2_fused)
            hf.attrs["psig"] = psig
            hf.attrs["orientation"] = "H = NC/PH (H1 = conj(Sxy)/Sxx with x=PH, y=NC)"
            hf.attrs["fs_Hz"] = cfg.FS
            hf.attrs["fcut_Hz"] = fcut
            hf.attrs["case"] = cfg.CASE
            hf.attrs["application"] = "per-channel: apply H1_smooth to PH1, H2_smooth to PH2"
            hf.attrs["smooth_oct"] = float(cfg.TF_SMOOTH_OCT)
            hf.attrs["smooth_ppo"] = int(cfg.TF_SMOOTH_PPO)

        try:
            _save_ph_tf_plot(
                p_si=p_si,
                f1=f1, H1=H1_s, f2=f2, H2=H2_s,
                f_fused=f_fused, H_fused=H_fused, g2_fused=g2_fused,
            )
        except Exception as exc:
            print(f"[warn] failed TF check plot for {p_si} psig: {exc}")

        print(f"[ok] {p_si:>3} psig to {out}")


def save_NC_calibs(
    *,
    fs: float | None = None,
    f_cuts: Sequence[float] | None = None,
) -> None:
    base = Path(cfg.RAW_CAL_BASE) / "NC"
    out_dir = Path(cfg.TF_BASE) / "NC"
    out_dir.mkdir(parents=True, exist_ok=True)

    pressures = [int(p) for p in cfg.PSIGS]
    fs = cfg.FS if fs is None else fs
    f_cuts = cfg.F_CUTS if f_cuts is None else f_cuts
    if len(f_cuts) != len(pressures):
        raise ValueError("f_cuts length must match number of pressures")

    for p_si, fcut in zip(pressures, f_cuts):
        psig = float(p_si)
        m1 = loadmat(base / f"{p_si}psig/nkd-ns_nofacilitynoise.mat")
        if p_si == 100:
            nkd, nc = m1["channelData_nofacitynoise"].T
        else:
            nkd, nc = m1["channelData"].T

        f1, H1, g2_1 = estimate_frf(
            nc, nkd,
            fs=fs, window=cfg.WINDOW, nperseg=cfg.NPERSEG,
        )
        H1_s = smooth_frf_logfreq(
            f1, H1, span_oct=cfg.TF_SMOOTH_OCT, ppo=cfg.TF_SMOOTH_PPO,
        )
        out = out_dir / f"calibs_{p_si}.h5"
        with h5py.File(out, "w") as hf:
            hf.create_dataset("frequencies", data=f1)
            hf.create_dataset("H_fused", data=H1)
            hf.create_dataset("H_smooth", data=H1_s)
            hf.create_dataset("gamma2_fused", data=g2_1)
            hf.attrs["psig"] = psig
            hf.attrs["orientation"] = "H = nkd/nc (H1 = conj(Sxy)/Sxx with x=nc, y=nkd)"
            hf.attrs["fs_Hz"] = fs
            hf.attrs["fcut_Hz"] = fcut
            hf.attrs["case"] = cfg.CASE
            hf.attrs["application"] = "fs_proc/pw_proc apply H_smooth"
            hf.attrs["smooth_oct"] = float(cfg.TF_SMOOTH_OCT)
            hf.attrs["smooth_ppo"] = int(cfg.TF_SMOOTH_PPO)
        print(f"[ok] {p_si:>3} psig to {out}")


if __name__ == "__main__":
    save_PH_calibs()
    save_NC_calibs()
