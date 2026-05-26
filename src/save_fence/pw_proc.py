# Processed wall-pressure (pinhole, FRF-corrected) — fence variant.
# Uses NC semi-anechoic calibration FRF.
#
# Wiener noise rejection against the freestream NC reference is intentionally
# disabled here (see save_bump/pw_proc.py for the rationale). The
# "fs_noise_rejected_signals" group is retained for backward compatibility
# with downstream plotting/checks and holds the bandpassed FRF-corrected
# signal.
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

from src.core.apply_frf import apply_frf
from src.save_fence.config_params import Config

cfg = Config()

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
        hf.attrs["title"] = "Wall-pressure (pin-hole) - production [fence]"
        hf.attrs["fs_Hz"] = FS
        hf.attrs["Ue_m_per_s"] = np.asarray(Ue, float)
        hf.attrs["DAQ"] = "24-bit"
        hf.attrs["mic_details"] = "HB&K 1/2'' Type 4964"
        hf.attrs["case"] = cfg.CASE
        hf.attrs["description"] = (
            "Processed wall-pressure (fence). PH -> NC FRF from semi-anechoic "
            "calibration, then NC -> nkd from NC semi-anechoic calibration."
        )

        g_fs = hf.create_group("wallp_production")

        # Discover which labels actually have raw wall-pressure data on disk.
        with h5py.File(ph_raw, "r") as _f_raw_probe:
            available_labels = [L for L in labels if L in _f_raw_probe["wallp_raw"]]
        if not available_labels:
            raise FileNotFoundError("fence pw_proc: no labels present in raw wall-pressure file")

        for i, L in enumerate(labels):
            if L not in available_labels:
                print(f"[skip] fence {L}: no raw wall-pressure entry")
                continue
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
            gL.attrs["fence_height_m"] = cfg.FENCE_HEIGHT
            gL.attrs["fence_thickness_m"] = cfg.FENCE_THICKNESS
            gL.attrs["x_fence_m"] = cfg.X_FENCE
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

                with h5py.File(f"{cal_base}/PH/calibs_{int(psigs[i])}.h5", "r") as hf_cal:
                    # Per-channel TFs (PH1 uses H1 on f1; PH2 uses H2 on f2).
                    # The fused TF in this same file is a diagnostic backup
                    # and is NOT applied in this fence pipeline.
                    f_per_channel = {
                        "PH1": np.asarray(hf_cal["f1"][:], dtype=WORK_DTYPE),
                        "PH2": np.asarray(hf_cal["f2"][:], dtype=WORK_DTYPE),
                    }
                    H_per_channel = {
                        "PH1": np.asarray(hf_cal["H1"][:], dtype=np.complex64),
                        "PH2": np.asarray(hf_cal["H2"][:], dtype=np.complex64),
                    }

                nc_cal_path = Path(cal_base) / "NC" / f"calibs_{int(psigs[i])}.h5"
                if not nc_cal_path.exists():
                    raise FileNotFoundError(
                        f"fence pipeline requires NC calibration but missing: {nc_cal_path}"
                    )
                with h5py.File(nc_cal_path, "r") as hf_nc:
                    f_cal_nkd = hf_nc["frequencies"][:].squeeze().astype(WORK_DTYPE)
                    H_fused_nkd = hf_nc["H_fused"][:].squeeze().astype(np.complex64)

                g_corrected = gL.create_group("frf_corrected_signals")
                g_rejected = gL.create_group("fs_noise_rejected_signals")

                def bandpass_filter(data, fs, f_low, f_high, order=3):
                    sos = butter(order, [f_low, f_high], btype="band", fs=fs, output="sos")
                    filtered = sosfiltfilt(sos, data)
                    filtered = np.nan_to_num(filtered, nan=0.0, copy=False)
                    return np.ascontiguousarray(filtered, dtype=WORK_DTYPE)

                for sp in available:
                    g_corr = g_corrected.create_group(sp)
                    g_rej = g_rejected.create_group(sp)

                    for channel in ("PH1", "PH2"):
                        signal = np.asarray(g_raw[f"{sp}/{channel}_Pa"][:], dtype=WORK_DTYPE)
                        signal = apply_frf(
                            signal, FS,
                            f_per_channel[channel], H_per_channel[channel],
                            dtype=WORK_DTYPE,
                        )
                        signal = apply_frf(signal, FS, f_cal_nkd, H_fused_nkd, dtype=WORK_DTYPE)
                        signal = np.ascontiguousarray(signal)
                        g_corr.create_dataset(f"{channel}_Pa", data=signal, dtype="f4")

                        signal = np.ascontiguousarray(signal - signal.mean(dtype=WORK_DTYPE))
                        signal = bandpass_filter(signal, FS, 1, analog_LP_filter[i])
                        g_rej.create_dataset(f"{channel}_Pa", data=signal, dtype="f4")

                        del signal
                        gc.collect()


if __name__ == "__main__":
    save_corrected_pressure()
