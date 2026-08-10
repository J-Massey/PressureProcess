# Processed wall-pressure (pinhole, FRF-corrected) — phase2 variant.
#
# Pipeline per (label, spacing, channel):
#   raw PH -> apply phase1 H_fused (PH->NC) -> NC reference
#   -> apply NC->nkd FRF (from phase2 NC calibs) -> demean -> Wiener noise
#   rejection against the demeaned freestream NC (production) reference.
#   No digital band filtering; band limits (f_cut_Hz) are applied only as
#   spectral masks at reporting time.
#
# PH calibration source: phase1's data/phase1/calibration/PH/calibs_{psig}.h5
# applied to BOTH PH1 and PH2 (same fused TF). Phase2's own PH calibration
# recordings are systematically depressed ~15-20 dB in the 100-1000 Hz ROI,
# so they are not used for the wall-pressure correction here; they are still
# generated and saved by save_phase2.calibs.save_PH_calibs for diagnostics.
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import h5py

from src.core.apply_frf import apply_frf
from src.core.wiener_filter_torch import wiener_cancel_background, wiener_cancel_hybrid
from src.save_phase2.config_params import Config

cfg = Config()

WORK_DTYPE = np.float32


def _resolve_noise_canceller(signal_len: int) -> str:
    if sys.platform == "darwin" and signal_len >= 1_000_000:
        return "hybrid"
    return "wiener"


def _cancel_noise(signal: np.ndarray, nkd: np.ndarray, fs: float) -> np.ndarray:
    method = _resolve_noise_canceller(signal.size)
    if method == "hybrid":
        return wiener_cancel_hybrid(signal, nkd, fs, m=2**10, dtype=WORK_DTYPE)
    return wiener_cancel_background(signal, nkd, fs)


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
        hf.attrs["title"] = "Wall-pressure (pin-hole) - production [phase2]"
        hf.attrs["fs_Hz"] = FS
        hf.attrs["Ue_m_per_s"] = np.asarray(Ue, float)
        hf.attrs["DAQ"] = "24-bit"
        hf.attrs["mic_details"] = "HB&K 1/2'' Type 4964"
        hf.attrs["case"] = cfg.CASE
        hf.attrs["description"] = (
            "Processed wall-pressure (phase2). Per-channel PH -> NC FRF "
            "(H1 on PH1, H2 on PH2, no fusion), then NC -> nkd from NC "
            "semi-anechoic calibration, then Wiener noise rejection against "
            "the freestream NC reference."
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

                # PH calibration: use phase1's fused TF for both channels.
                # Phase2's own PH calibration recordings are ~15-20 dB below
                # phase1's in the 100-1000 Hz ROI (see
                # figures/comparisons/phase1_vs_phase2/PH_tf_phase1_vs_phase2_ROI.png),
                # which depresses the corrected wall pressure by ~10x in PSD.
                # Phase2's own PH calibs are still generated for diagnostics
                # but are not applied.
                ph_cal_path = Path("data/phase1/calibration/PH") / f"calibs_{int(psigs[i])}.h5"
                if not ph_cal_path.exists():
                    raise FileNotFoundError(
                        f"phase2 pw_proc requires phase1 PH calibration: {ph_cal_path}"
                    )
                with h5py.File(ph_cal_path, "r") as hf_cal:
                    f_ph_cal = np.asarray(hf_cal["frequencies"][:], dtype=WORK_DTYPE)
                    H_ph_cal = np.asarray(hf_cal["H_fused"][:], dtype=np.complex64)
                f_per_channel = {"PH1": f_ph_cal, "PH2": f_ph_cal}
                H_per_channel = {"PH1": H_ph_cal, "PH2": H_ph_cal}

                nc_cal_path = Path(cal_base) / "NC" / f"calibs_{int(psigs[i])}.h5"
                if not nc_cal_path.exists():
                    raise FileNotFoundError(
                        f"phase2 pipeline requires NC calibration but missing: {nc_cal_path}"
                    )
                # Match phase1: NC->nkd FRF applied raw (H_fused), no smoothing.
                with h5py.File(nc_cal_path, "r") as hf_nc:
                    f_cal_nkd = hf_nc["frequencies"][:].squeeze().astype(WORK_DTYPE)
                    H_fused_nkd = hf_nc["H_fused"][:].squeeze().astype(np.complex64)

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

                    # Freestream NC (production) reference for the Wiener
                    # canceller, demeaned to match the PH path (band limits are
                    # applied only as spectral masks at reporting time).
                    nkd = np.asarray(g_nkd[f"{sp}/NC_Pa"][:], dtype=WORK_DTYPE)
                    nkd = np.ascontiguousarray(nkd - nkd.mean(dtype=WORK_DTYPE))

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
                        clean = _cancel_noise(signal, nkd, FS)
                        g_rej.create_dataset(f"{channel}_Pa", data=clean, dtype="f4")

                        del signal, clean
                        gc.collect()

                    del nkd
                    gc.collect()


if __name__ == "__main__":
    save_corrected_pressure()
