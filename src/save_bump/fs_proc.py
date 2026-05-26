# Processed freestream pressure (NC -> nkd) — bump variant.
# Uses NC semi-anechoic calibration FRF (full NC calibs available for bump).
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import h5py
import scipy.io as sio

from src.core.apply_frf import apply_frf
from src.save_bump.config_params import Config

cfg = Config()


def correct_pressure_sensitivity(p, psig, alpha: float = 0.01):
    p_corr = p * 10 ** (psig * cfg.PSI_TO_PA / 1000 * alpha / 20)
    return p_corr


def volts_to_pa(x_volts: np.ndarray, channel: str) -> np.ndarray:
    sens = cfg.SENSITIVITIES_V_PER_PA[channel]
    return x_volts / sens


def air_props_from_gauge(psi_gauge: float, T_K: float):
    p_abs = cfg.P_ATM + psi_gauge * cfg.PSI_TO_PA
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    mu = mu0 * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)
    rho = p_abs / (cfg.R * T_K)
    nu = mu / rho
    return rho, mu, nu


def save_prod_fs_pressure(
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

    fs_prod = cfg.NKD_PROCESSED_FILE
    os.makedirs(Path(fs_prod).parent, exist_ok=True)

    with h5py.File(fs_prod, "w") as hf:
        hf.attrs["title"] = "Freestream pressure (nose-cone) - production [bump]"
        hf.attrs["fs_Hz"] = FS
        hf.attrs["Ue_m_per_s"] = np.asarray(Ue, float)
        hf.attrs["DAQ"] = "24-bit"
        hf.attrs["mic_details"] = "HB&K 1/2'' Type 4964"
        hf.attrs["case"] = cfg.CASE
        hf.attrs["description"] = (
            "Freestream NC signals (bump, rough surface), corrected with NC -> "
            "nkd FRF from semi-anechoic calibration."
        )

        g_fs = hf.create_group("freestream_production")

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

            nc_cal_path = Path(cfg.TF_BASE) / "NC" / f"calibs_{int(psigs[i])}.h5"
            if not nc_cal_path.exists():
                raise FileNotFoundError(
                    f"bump pipeline requires NC calibration but missing: {nc_cal_path}"
                )
            with h5py.File(nc_cal_path, "r") as hf_nc:
                nc_key = "H_smooth" if "H_smooth" in hf_nc else "H_fused"
                f_cal_nkd = hf_nc["frequencies"][:].squeeze().astype(float)
                H_fused_nkd = hf_nc[nc_key][:].squeeze().astype(complex)

            gFRF = gL.create_group("FRF_NC_to_nkd")
            gFRF.create_dataset("fcal_Hz", data=f_cal_nkd)
            gFRF.create_dataset("Hcal", data=H_fused_nkd)
            gFRF.attrs["from"] = "NC"
            gFRF.attrs["to"] = "nkd"
            gFRF.attrs["note"] = "Semi-anechoic calibration mapping"

            seen_any = False
            for sp in spacings:
                mat_path = Path(cfg.RAW_BASE) / f"{sp}/{L}.mat"
                if not mat_path.exists():
                    print(f"[skip] missing raw mat file: {mat_path}")
                    continue
                dat = sio.loadmat(mat_path)
                X = np.asarray(dat["channelData"])
                NC_v = X[:, 2]
                NC = correct_pressure_sensitivity(volts_to_pa(NC_v, "NC"), psigs[i])
                NC = apply_frf(NC, FS, f_cal_nkd, H_fused_nkd)

                gS = gL.create_group(sp)
                gS.create_dataset("NC_Pa", data=NC)
                seen_any = True
            if not seen_any:
                raise FileNotFoundError(f"No raw files found for {L} in {cfg.RAW_BASE}")


if __name__ == "__main__":
    save_prod_fs_pressure()
