# Wall-pressure raw (pinhole) — fence variant.
#
# NOTE: fence delivers wall-pressure as a single combined MAT file
# (cfg.ATM_PATH). The ingest scaffold below mirrors the bump version but the
# PH1/PH2 channel pull from the combined file needs to be filled in once the
# ATM_Rev1.mat layout is known.
import os
from pathlib import Path

import numpy as np
import h5py
import scipy.io as sio

from src.save_fence.config_params import Config

cfg = Config()


def _extract_channel_data(mat_obj: dict) -> object:
    if "channelData_WN" in mat_obj:
        return mat_obj["channelData_WN"]
    if "channelData" in mat_obj:
        return mat_obj["channelData"]
    raise KeyError("Expected one of calibration keys: channelData_WN, channelData")


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


def _load_atm_combined() -> dict:
    atm_path = Path(cfg.ATM_PATH)
    if not atm_path.exists():
        raise FileNotFoundError(f"fence ATM file missing: {atm_path}")
    return sio.loadmat(atm_path)


def _per_pressure_mat_path(label: str) -> Path:
    """<RAW_BASE>/combined/<label>.mat for non-atmospheric runs (when they land)."""
    return Path(cfg.RAW_BASE) / "combined" / f"{label}.mat"


def _extract_PH_for_pressure(atm: dict, label: str, psig: float):
    """
    Return (ph1_v, ph2_v) volts for one pressure, or (None, None) if missing.

    psig == 0 reads ATM_Rev1.mat. Other pressures look for a per-pressure
    file with the same 4-column layout as ATM_Rev1.mat.
    """
    cols = cfg.ATM_CHANNEL_COLUMNS
    if float(psig) == 0.0:
        cd = np.asarray(atm["channelData"])
        return cd[:, cols["PH1"]].astype(float), cd[:, cols["PH2"]].astype(float)
    mat_path = _per_pressure_mat_path(label)
    if not mat_path.exists():
        return None, None
    mat = sio.loadmat(mat_path)
    cd = np.asarray(mat["channelData"])
    return cd[:, cols["PH1"]].astype(float), cd[:, cols["PH2"]].astype(float)


def save_raw_ph_pressure(
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

    cal_base = Path(cfg.RAW_CAL_BASE) / "PH"
    os.makedirs(cal_base, exist_ok=True)

    ph_raw = cfg.PH_RAW_FILE
    os.makedirs(Path(ph_raw).parent, exist_ok=True)

    atm = _load_atm_combined()

    with h5py.File(ph_raw, "w") as hf:
        hf.attrs["title"] = "Wall-pressure (pin-hole) - raw \& calibration [fence]"
        hf.attrs["fs_Hz"] = FS
        hf.attrs["Ue_m_per_s"] = np.asarray(Ue, float)
        hf.attrs["DAQ"] = "24-bit NI-USB-6363"
        hf.attrs["mic_details"] = "HB&K 1/2'' Type 4964"
        hf.attrs["case"] = cfg.CASE
        hf.attrs["atm_file"] = cfg.ATM_FILE
        hf.attrs["description"] = (
            "Raw wall-pressure signals (fence). psig=0 from ATM_Rev1.mat; "
            "non-atmospheric pressures expect <RAW_BASE>/combined/<label>.mat "
            "with the same 4-column layout. Includes PH<->NC semi-anechoic "
            "calibration."
        )

        g_fs = hf.create_group("wallp_raw")

        seen_any = False
        for i, L in enumerate(labels):
            ph1_v, ph2_v = _extract_PH_for_pressure(atm, L, psigs[i])
            if ph1_v is None:
                print(f"[skip] fence {L}: no per-pressure wall-pressure file yet")
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

            ph1 = correct_pressure_sensitivity(volts_to_pa(ph1_v, "PH1"), psigs[i])
            ph2 = correct_pressure_sensitivity(volts_to_pa(ph2_v, "PH2"), psigs[i])

            for sp in spacings:
                gS = gL.create_group(sp)
                gS.create_dataset("PH1_Pa", data=ph1)
                gS.create_dataset("PH2_Pa", data=ph2)
            seen_any = True

            # PH<->NC semi-anechoic calibration — same shape as bump/iso_re.
            m1_path = cal_base / f"calib_{L}_1.mat"
            m2_path = cal_base / f"calib_{L}_2.mat"
            if not (m1_path.exists() and m2_path.exists()):
                print(f"[skip] missing PH calibration for {L} at {cal_base}")
                continue

            m1 = sio.loadmat(m1_path)
            ph1_v_cal, _, nc_v, *_ = _extract_channel_data(m1).T
            ph1_pa = correct_pressure_sensitivity(volts_to_pa(ph1_v_cal, "PH1"), psigs[i])
            nc1_pa = correct_pressure_sensitivity(volts_to_pa(nc_v, "NC"), psigs[i])

            m2 = sio.loadmat(m2_path)
            _, ph2_v_cal, nc_v2, *_ = _extract_channel_data(m2).T
            ph2_pa = correct_pressure_sensitivity(volts_to_pa(ph2_v_cal, "PH2"), psigs[i])
            nc2_pa = correct_pressure_sensitivity(volts_to_pa(nc_v2, "NC"), psigs[i])

            gFRF = gL.create_group("FRF_PH_to_NC")
            gFRF.attrs["from"] = "NC"
            gFRF.attrs["to"] = "nkd"
            gFRF.attrs["note"] = "Semi-anechoic calibration mapping the pinhole mic to nosecone mic"
            gR1 = gFRF.create_group("Run1")
            gR1.create_dataset("PH1_Pa", data=ph1_pa)
            gR1.create_dataset("NC_Pa", data=nc1_pa)
            gR2 = gFRF.create_group("Run2")
            gR2.create_dataset("PH2_Pa", data=ph2_pa)
            gR2.create_dataset("NC_Pa", data=nc2_pa)

        if not seen_any:
            raise FileNotFoundError(
                "fence wall-pressure ingest found no readable data — check ATM_Rev1.mat path"
            )


if __name__ == "__main__":
    save_raw_ph_pressure()
