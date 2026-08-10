# Processed wall-pressure (pinhole, FRF-corrected) — bump variant.
#
# Pipeline per (label, spacing, channel):
#   raw PH -> apply phase1 H_fused (PH->NC, same TF for PH1 and PH2)
#   -> apply bump's own NC->nkd FRF
#   -> demean -> apply phase2-donor Wiener FIR kernel against the demeaned NC
#   reference. No digital band filtering; band limits (f_cut_Hz) are applied
#   only as spectral masks at reporting time.
#
# Why these sources:
#   - phase1 PH TF: bump's own PH anechoic calibrations are pinhole-degraded,
#     and phase2's are ~15-20 dB low in the ROI; phase1's fused TF is the
#     reference we trust.
#   - phase2 Wiener kernel (donor): training Wiener locally on bump corrupts
#     the wall signal because the freestream NC carries flow content from the
#     bump dipole. The kernel from phase2 (smooth wall, clean NC reference)
#     is purely the facility-to-wall acoustic channel and can be transplanted.
#     Phase2 is close-only; bump's `far` spacing reuses phase2's close kernel
#     (the kernel is approximately spacing-independent).
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import h5py

from src.core.apply_frf import apply_frf
from src.core.wiener_filter_torch import apply_wiener_kernel
from src.save_bump.config_params import Config

cfg = Config()

# Borrowed-source paths (deliberate, not derived from cfg).
PH_CALIB_SOURCE = Path("data/phase1/calibration/PH")
WIENER_KERNELS_FILE = Path("data/phase2/calibration/wiener_kernels.h5")
WIENER_KERNEL_FALLBACK_SPACING = "close"  # phase2 only has 'close'

WORK_DTYPE = np.float32


def air_props_from_gauge(psi_gauge: float, T_K: float):
    p_abs = cfg.P_ATM + psi_gauge * cfg.PSI_TO_PA
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    mu = mu0 * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)
    rho = p_abs / (cfg.R * T_K)
    nu = mu / rho
    return rho, mu, nu


def save_corrected_pressure(
    *,
    spacings: tuple[str, ...] | None = None,
):
    labels = cfg.LABELS
    psigs = cfg.PSIGS
    u_tau = cfg.U_TAU
    u_tau_unc = cfg.U_TAU_REL_UNC
    Tk = [273.15 + t for t in cfg.TDEG]
    FS = cfg.FS
    Ue = cfg.U_E
    analog_LP_filter = cfg.ANALOG_LP_FILTER
    spacings = cfg.SPACINGS if spacings is None else spacings

    ph_processed = cfg.PH_PROCESSED_FILE
    ph_raw = cfg.PH_RAW_FILE
    nkd_raw = cfg.NKD_PROCESSED_FILE
    cal_base = cfg.TF_BASE
    os.makedirs(Path(ph_processed).parent, exist_ok=True)

    with h5py.File(ph_processed, "w") as hf:
        hf.attrs["title"] = "Wall-pressure (pin-hole) - production [bump]"
        hf.attrs["fs_Hz"] = FS
        hf.attrs["Ue_m_per_s"] = np.asarray(Ue, float)
        hf.attrs["DAQ"] = "24-bit"
        hf.attrs["mic_details"] = "HB&K 1/2'' Type 4964"
        hf.attrs["case"] = cfg.CASE
        hf.attrs["description"] = (
            "Processed wall-pressure (bump, rough surface). PH -> NC FRF from "
            "semi-anechoic calibration, then NC -> nkd from semi-anechoic NC calibration."
        )

        g_fs = hf.create_group("wallp_production")

        for i, L in enumerate(labels):
            gL = g_fs.create_group(L)
            rho, mu, nu = air_props_from_gauge(psigs[i], Tk[i])
            delta_i = float(cfg.DELTA[i])
            gL.attrs["psig"] = psigs[i]
            gL.attrs["u_tau"] = u_tau[i]
            gL.attrs["nu"] = nu
            gL.attrs["rho"] = rho
            gL.attrs["mu"] = mu
            gL.attrs["Re_tau"] = u_tau[i] * delta_i / nu
            gL.attrs["delta"] = delta_i
            gL.attrs["u_tau_rel_unc"] = u_tau_unc[i]
            gL.attrs["T_K"] = Tk[i]
            gL.attrs["analog_LP_filter_Hz"] = analog_LP_filter[i]
            gL.attrs["Ue_m_per_s"] = float(Ue[i])
            gL.attrs["k_s"] = cfg.K_S
            gL.attrs["bump_height_m"] = cfg.BUMP_HEIGHT
            gL.attrs["units"] = [
                "psig: psi(g)", "u_tau: m/s", "nu: m^2/s", "rho: kg/m^3",
                "mu: Pa*s", "T_K: K", "analog_LP_filter_Hz: Hz",
            ]

            with h5py.File(ph_raw, "r") as f_raw, h5py.File(nkd_raw, "r") as f_nkd:
                g_raw = f_raw[f"wallp_raw/{L}"]
                g_nkd = f_nkd[f"freestream_production/{L}"]
                available = [sp for sp in spacings if sp in g_raw and sp in g_nkd]
                if not available:
                    raise FileNotFoundError(f"No matching spacings for {L} in raw files")

                # PH calibration: phase1's fused TF for both PH1 and PH2.
                ph_cal_path = PH_CALIB_SOURCE / f"calibs_{int(psigs[i])}.h5"
                if not ph_cal_path.exists():
                    raise FileNotFoundError(
                        f"bump pw_proc needs phase1 PH calibration: {ph_cal_path}"
                    )
                with h5py.File(ph_cal_path, "r") as hf_cal:
                    f_ph_cal = np.asarray(hf_cal["frequencies"][:], dtype=WORK_DTYPE)
                    H_ph_cal = np.asarray(hf_cal["H_fused"][:], dtype=np.complex64)
                f_per_channel = {"PH1": f_ph_cal, "PH2": f_ph_cal}
                H_per_channel = {"PH1": H_ph_cal, "PH2": H_ph_cal}

                nc_cal_path = Path(cal_base) / "NC" / f"calibs_{int(psigs[i])}.h5"
                if not nc_cal_path.exists():
                    raise FileNotFoundError(
                        f"bump pipeline requires NC calibration but missing: {nc_cal_path}"
                    )
                # Match phase1/phase2: NC->nkd FRF applied raw (H_fused).
                with h5py.File(nc_cal_path, "r") as hf_nc:
                    f_cal_nkd = hf_nc["frequencies"][:].squeeze().astype(WORK_DTYPE)
                    H_fused_nkd = hf_nc["H_fused"][:].squeeze().astype(np.complex64)

                # Load phase2 Wiener kernels for this pressure once per label.
                if not WIENER_KERNELS_FILE.exists():
                    raise FileNotFoundError(
                        f"bump pw_proc needs phase2 Wiener kernels: {WIENER_KERNELS_FILE}"
                    )

                g_corrected = gL.create_group("frf_corrected_signals")
                g_rejected = gL.create_group("fs_noise_rejected_signals")

                for sp in available:
                    spacing_meta = {
                        "close": {
                            "spacing_m": 2.8 * delta_i,
                            "x_PH1": 15e-3 + 0.2 * delta_i,
                            "x_PH2": 15e-3 + 0.2 * delta_i + 2.8 * delta_i,
                        },
                        "far": {
                            "spacing_m": 3.2 * delta_i,
                            "x_PH2": 15e-3,
                            "x_PH1": 15e-3 + 3.2 * delta_i,
                        },
                    }
                    g_corr = g_corrected.create_group(sp)
                    meta = spacing_meta.get(sp)
                    if meta:
                        g_corr.attrs["spacing_m"] = meta["spacing_m"]
                        g_corr.attrs["x_PH1"] = meta["x_PH1"]
                        g_corr.attrs["x_PH2"] = meta["x_PH2"]

                    g_rej = g_rejected.create_group(sp)
                    if meta:
                        g_rej.attrs["spacing_m"] = meta["spacing_m"]
                        g_rej.attrs["x_PH1"] = meta["x_PH1"]
                        g_rej.attrs["x_PH2"] = meta["x_PH2"]

                    # NC reference for Wiener: bumped's own NC (production),
                    # demeaned identically to the donor (band limits
                    # are applied only as spectral masks at reporting time).
                    nkd = np.asarray(g_nkd[f"{sp}/NC_Pa"][:], dtype=WORK_DTYPE)
                    nkd = np.ascontiguousarray(nkd - nkd.mean(dtype=WORK_DTYPE))

                    # Phase2 kernels exist for "close" only; reuse for "far".
                    kernel_sp = sp if sp in (WIENER_KERNEL_FALLBACK_SPACING,) else WIENER_KERNEL_FALLBACK_SPACING
                    with h5py.File(WIENER_KERNELS_FILE, "r") as hf_k:
                        g_k_label = hf_k.get(L)
                        if g_k_label is None or kernel_sp not in g_k_label:
                            raise FileNotFoundError(
                                f"missing phase2 kernel for {L}/{kernel_sp}"
                            )
                        c_per_channel = {
                            ch: np.asarray(g_k_label[f"{kernel_sp}/{ch}_Pa/c"][:],
                                           dtype=WORK_DTYPE)
                            for ch in ("PH1", "PH2")
                            if f"{kernel_sp}/{ch}_Pa" in g_k_label
                        }

                    for channel in ("PH1", "PH2"):
                        # Per-position friction velocity (and Re_tau derived
                        # from it via the per-pressure delta + nu). Fall back
                        # to the per-pressure u_tau if the position is missing
                        # from the config dict.
                        try:
                            u_tau_pos = float(cfg.U_TAU_BY_POSITION[L][sp][channel])
                        except KeyError:
                            u_tau_pos = float(u_tau[i])
                        re_tau_pos = u_tau_pos * delta_i / nu

                        signal = np.asarray(g_raw[f"{sp}/{channel}_Pa"][:], dtype=WORK_DTYPE)
                        signal = apply_frf(
                            signal, FS,
                            f_per_channel[channel], H_per_channel[channel],
                            dtype=WORK_DTYPE,
                        )
                        signal = apply_frf(signal, FS, f_cal_nkd, H_fused_nkd, dtype=WORK_DTYPE)
                        signal = np.ascontiguousarray(signal)
                        d_corr = g_corr.create_dataset(f"{channel}_Pa", data=signal, dtype="f4")
                        d_corr.attrs["u_tau"] = u_tau_pos
                        d_corr.attrs["Re_tau"] = re_tau_pos

                        signal = np.ascontiguousarray(signal - signal.mean(dtype=WORK_DTYPE))

                        if channel in c_per_channel:
                            clean = apply_wiener_kernel(
                                signal, nkd, c_per_channel[channel],
                                alpha=1.0, preserve_mean=False,
                            )
                            clean = np.ascontiguousarray(clean, dtype=WORK_DTYPE)
                        else:
                            print(f"[warn] no phase2 kernel for {L}/{kernel_sp}/{channel}; "
                                  f"falling through with bandpassed signal")
                            clean = signal
                        d_rej = g_rej.create_dataset(f"{channel}_Pa", data=clean, dtype="f4")
                        d_rej.attrs["u_tau"] = u_tau_pos
                        d_rej.attrs["Re_tau"] = re_tau_pos

                        del signal, clean
                        gc.collect()

                    del nkd
                    gc.collect()


if __name__ == "__main__":
    save_corrected_pressure()
