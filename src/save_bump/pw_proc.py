# Processed wall-pressure (pinhole, FRF-corrected) — bump variant.
# Uses NC semi-anechoic calibration FRF (full NC calibs available for bump).
#
# Wiener noise rejection against the freestream NC reference is intentionally
# disabled here: the NC<->PH coherence collapsed in the bump runs (and was
# already weak in the smooth donor), so both local Wiener and the
# smooth-donor kernel inject low-frequency phantom content rather than
# removing facility noise. The "fs_noise_rejected_signals" group is kept for
# backward compatibility with downstream plotting/checks and now simply holds
# the bandpassed FRF-corrected signal.
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

from src.core.apply_frf import apply_frf
from src.save_bump.config_params import Config

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

                with h5py.File(f"{cal_base}/PH/calibs_{int(psigs[i])}.h5", "r") as hf_cal:
                    # Per-channel smoothed TFs (PH1 uses H1_smooth on f1;
                    # PH2 uses H2_smooth on f2). The fused TF in this same
                    # file is a diagnostic backup and is NOT applied here.
                    H1_key = "H1_smooth" if "H1_smooth" in hf_cal else "H1"
                    H2_key = "H2_smooth" if "H2_smooth" in hf_cal else "H2"
                    f_per_channel = {
                        "PH1": np.asarray(hf_cal["f1"][:], dtype=WORK_DTYPE),
                        "PH2": np.asarray(hf_cal["f2"][:], dtype=WORK_DTYPE),
                    }
                    H_per_channel = {
                        "PH1": np.asarray(hf_cal[H1_key][:], dtype=np.complex64),
                        "PH2": np.asarray(hf_cal[H2_key][:], dtype=np.complex64),
                    }

                nc_cal_path = Path(cal_base) / "NC" / f"calibs_{int(psigs[i])}.h5"
                if not nc_cal_path.exists():
                    raise FileNotFoundError(
                        f"bump pipeline requires NC calibration but missing: {nc_cal_path}"
                    )
                with h5py.File(nc_cal_path, "r") as hf_nc:
                    nc_key = "H_smooth" if "H_smooth" in hf_nc else "H_fused"
                    f_cal_nkd = hf_nc["frequencies"][:].squeeze().astype(WORK_DTYPE)
                    H_fused_nkd = hf_nc[nc_key][:].squeeze().astype(np.complex64)

                g_corrected = gL.create_group("frf_corrected_signals")
                g_rejected = gL.create_group("fs_noise_rejected_signals")

                def bandpass_filter(data, fs, f_low, f_high, order=3):
                    sos = butter(order, [f_low, f_high], btype="band", fs=fs, output="sos")
                    filtered = sosfiltfilt(sos, data)
                    filtered = np.nan_to_num(filtered, nan=0.0, copy=False)
                    return np.ascontiguousarray(filtered, dtype=WORK_DTYPE)

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
