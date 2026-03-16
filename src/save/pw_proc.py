# tf_compute.py
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

from pathlib import Path

from src.core.apply_frf import apply_frf
from src.core.wiener_filter_torch import wiener_cancel_background, wiener_cancel_hybrid
from src.config_params import Config

cfg = Config()

# =============================================================================
# Constants & styling (exported so tf_plot.py can import them)
# =============================================================================
FS = cfg.FS
NPERSEG = cfg.NPERSEG
WINDOW = cfg.WINDOW

# --- constants (keep once, top of file) ---
R = cfg.R        # J/kg/K
PSI_TO_PA = cfg.PSI_TO_PA
P_ATM = cfg.P_ATM
DELTA = cfg.DELTA  # m, bl-height of channel
TDEG = cfg.TDEG

# =============================================================================
# Units & optional conversions (kept for compatibility with other workflows)
# =============================================================================

SENSITIVITIES_V_PER_PA = cfg.SENSITIVITIES_V_PER_PA
PREAMP_GAIN = cfg.PREAMP_GAIN
CAL_BASE = cfg.TF_BASE
RAW_BASE = cfg.RAW_BASE
WORK_DTYPE = np.float32


def _resolve_noise_canceller(signal_len: int) -> str:
    method = cfg.PW_NOISE_CANCELLER
    if method == "auto":
        if sys.platform == "darwin" and signal_len >= 1_000_000:
            return "hybrid"
        return "wiener"
    if method in {"hybrid", "wiener"}:
        return method
    raise ValueError(
        "PRESSUREPROCESS_PW_DENOISER must be one of: auto, wiener, hybrid"
    )


def _cancel_noise(signal: np.ndarray, nkd: np.ndarray, fs: float) -> np.ndarray:
    method = _resolve_noise_canceller(signal.size)
    if method == "hybrid":
        return wiener_cancel_hybrid(signal, nkd, fs, m=2**10, dtype=WORK_DTYPE)
    return wiener_cancel_background(signal, nkd, fs)


def volts_to_pa(x_volts: np.ndarray, channel: str) -> np.ndarray:
    sens = SENSITIVITIES_V_PER_PA[channel]  # V/Pa
    return x_volts / sens


def air_props_from_gauge(psi_gauge: float, T_K: float):
    """
    Return rho [kg/m^3], mu [Pa*s], nu [m^2/s] from gauge pressure [psi] and temperature [K].
    Sutherland's law for mu; nu = mu/rho.
    """
    p_abs = P_ATM + psi_gauge * PSI_TO_PA
    # Sutherland's
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    mu = mu0 * (T_K/T0)**1.5 * (T0 + S)/(T_K + S)
    rho = p_abs / (R * T_K)
    nu = mu / rho
    return rho, mu, nu


def save_corrected_pressure(
    *,
    spacings: tuple[str, ...] | None = None,
):
    """
    Apply the (rho, f)-scaled calibration FRF to measured time series and plot
    pre-multiplied, normalized spectra:  f * Pyy / (rho^2 * u_tau^4).
    """
    # --- fit rho-f scaling once from your saved target + calibration ---
    labels = cfg.LABELS
    psigs = cfg.PSIGS
    u_tau = cfg.U_TAU
    u_tau_unc = cfg.U_TAU_REL_UNC
    Tdeg = cfg.TDEG
    Tk = [273.15 + t for t in Tdeg]
    FS = cfg.FS
    Ue = cfg.U_E
    analog_LP_filter = cfg.ANALOG_LP_FILTER
    spacings = cfg.SPACINGS if spacings is None else spacings

    ph_processed = cfg.PH_PROCESSED_FILE
    ph_raw = cfg.PH_RAW_FILE
    os.makedirs(Path(ph_processed).parent, exist_ok=True)

    with h5py.File(ph_processed, 'w') as hf:
        # --- file-level metadata ---
        hf.attrs['title'] = "Wall-pressure (pin-hole) - processed & FRF from calibration"
        hf.attrs['fs_Hz'] = FS
        hf.attrs['Ue_m_per_s'] = np.asarray(Ue, float)
        hf.attrs['DAQ'] = "24-bit"
        hf.attrs['mic_details'] = "HB&K 1/2'' Type 4964"
        # gL.attrs['sensor_serial'] = sensor_serial[i % len(sensor_serial)]


        # --- helpful top-level description ---
        hf.attrs['description'] = (
            "Processed wall-pressure signals from pinhole treated microphone applying FRF from calibration. "
            "Two measurements per condition: close/far correspond to the "
            "pinhole spacings used in wall-pressure dataset G. "
            "Includes frequency response function from semi-anechoic calibration signal with a white noise "
            "source measuring the pinhole and nosecone treated mic simultaneously."
        )

        g_fs = hf.create_group("wallp_production")

        for i, L in enumerate(labels):
            gL = g_fs.create_group(L)
            # condition-level metadata (numeric + units separate)
            rho, mu, nu = air_props_from_gauge(psigs[i], Tk[i])
            delta_i = float(DELTA[i])
            gL.attrs['psig'] = psigs[i]          # unit: psi(g)
            gL.attrs['u_tau'] = u_tau[i]         # unit: m/s
            gL.attrs['nu'] = nu
            gL.attrs['rho'] = rho
            gL.attrs['mu'] = mu
            gL.attrs['Re_tau'] = u_tau[i] * delta_i / nu         # unit: m/s
            gL.attrs['delta'] = delta_i
            gL.attrs['u_tau_rel_unc'] = u_tau_unc[i]
            gL.attrs['T_K'] = Tk[i]
            gL.attrs['analog_LP_filter_Hz'] = analog_LP_filter[i]
            gL.attrs['Ue_m_per_s'] = float(Ue[i])
            gL.attrs['units'] = ['psig: psi(g)', 'u_tau: m/s', 'nu: m^2/s', 'rho: kg/m^3', 'mu: Pa*s', 'T_K: K', 'analog_LP_filter_Hz: Hz']
            ph_raw = cfg.PH_RAW_FILE
            nkd_raw = cfg.NKD_PROCESSED_FILE

            with h5py.File(ph_raw, "r") as f_raw, h5py.File(nkd_raw, "r") as f_nkd:
                g_raw = f_raw[f"wallp_raw/{L}"]
                g_nkd = f_nkd[f"freestream_production/{L}"]
                available = [sp for sp in spacings if sp in g_raw and sp in g_nkd]
                if not available:
                    raise FileNotFoundError(f"No matching spacings for {L} in raw files")

                with h5py.File(f"{CAL_BASE}/PH/calibs_{int(psigs[i])}.h5", "r") as hf:
                    f_cal = np.asarray(hf["frequencies"][:], dtype=WORK_DTYPE)
                    H_cal = np.asarray(hf["H_fused"][:], dtype=np.complex64)

                nc_cal_path = Path(CAL_BASE) / "NC" / f"calibs_{int(psigs[i])}.h5"
                if nc_cal_path.exists():
                    with h5py.File(nc_cal_path, "r") as hf:
                        f_cal_nkd = hf["frequencies"][:].squeeze().astype(WORK_DTYPE)
                        H_fused_nkd = hf["H_fused"][:].squeeze().astype(np.complex64)
                else:
                    # iso_re workflow: run without NC semi-anechoic files.
                    print(f"[warn] missing NC calibration: {nc_cal_path}; using identity FRF")
                    f_cal_nkd = np.array([0.0, FS / 2.0], dtype=WORK_DTYPE)
                    H_fused_nkd = np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64)

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

                    nkd = np.asarray(g_nkd[f"{sp}/NC_Pa"][:], dtype=WORK_DTYPE)
                    nkd = np.ascontiguousarray(nkd - nkd.mean(dtype=WORK_DTYPE))
                    nkd = bandpass_filter(nkd, FS, 1, analog_LP_filter[i])

                    for channel in ("PH1", "PH2"):
                        signal = np.asarray(g_raw[f"{sp}/{channel}_Pa"][:], dtype=WORK_DTYPE)
                        signal = apply_frf(signal, FS, f_cal, H_cal, dtype=WORK_DTYPE)
                        signal = apply_frf(signal, FS, f_cal_nkd, H_fused_nkd, dtype=WORK_DTYPE)
                        signal = np.ascontiguousarray(signal)
                        g_corr.create_dataset(f"{channel}_Pa", data=signal, dtype="f4")

                        signal = np.ascontiguousarray(signal - signal.mean(dtype=WORK_DTYPE))
                        signal = bandpass_filter(signal, FS, 1, analog_LP_filter[i])
                        clean = _cancel_noise(signal, nkd, FS)
                        g_rej.create_dataset(f"{channel}_Pa", data=clean, dtype="f4")

                        del signal, clean
                        gc.collect()

                    del nkd
                    gc.collect()


if __name__ == "__main__":
    save_corrected_pressure()
