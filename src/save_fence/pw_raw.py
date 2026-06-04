# Wall-pressure raw (pinhole) — fence variant.
# Pipeline-equivalent to save_bump.pw_raw; only the Config source and the
# case-specific group attrs differ.
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
    os.makedirs(Path(cfg.RAW_BASE), exist_ok=True)

    ph_raw = cfg.PH_RAW_FILE
    os.makedirs(Path(ph_raw).parent, exist_ok=True)

    with h5py.File(ph_raw, "w") as hf:
        hf.attrs["title"] = "Wall-pressure (pin-hole) - raw & calibration [fence]"
        hf.attrs["fs_Hz"] = FS
        hf.attrs["Ue_m_per_s"] = np.asarray(Ue, float)
        hf.attrs["DAQ"] = "24-bit NI-USB-6363"
        hf.attrs["mic_details"] = "HB&K 1/2'' Type 4964"
        hf.attrs["case"] = cfg.CASE
        hf.attrs["description"] = (
            "Raw wall-pressure signals (fence). Pressures (0, 50, 100) psig "
            "with close+far pinhole spacings. Includes PH<->NC semi-anechoic "
            "calibration."
        )

        g_fs = hf.create_group("wallp_raw")

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
            gL.attrs["fence_height_m"] = cfg.FENCE_HEIGHT
            gL.attrs["fence_thickness_m"] = cfg.FENCE_THICKNESS
            gL.attrs["x_fence_m"] = cfg.X_FENCE
            gL.attrs["units"] = [
                "psig: psi(g)", "u_tau: m/s", "nu: m^2/s", "rho: kg/m^3",
                "mu: Pa*s", "T_K: K", "analog_LP_filter_Hz: Hz",
            ]

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
            seen_any = False
            for sp in spacings:
                mat_path = Path(cfg.RAW_BASE) / f"{sp}/{L}.mat"
                if not mat_path.exists():
                    print(f"[skip] missing raw mat file: {mat_path}")
                    continue
                dat = sio.loadmat(mat_path)
                X = np.asarray(dat["channelData"])
                ph1_v = X[:, 0]
                ph2_v = X[:, 1]

                ph1 = correct_pressure_sensitivity(volts_to_pa(ph1_v, "PH1"), psigs[i])
                ph2 = correct_pressure_sensitivity(volts_to_pa(ph2_v, "PH2"), psigs[i])

                gS = gL.create_group(sp)
                meta = spacing_meta.get(sp)
                if meta:
                    gS.attrs["spacing_m"] = meta["spacing_m"]
                    gS.attrs["x_PH1"] = meta["x_PH1"]
                    gS.attrs["x_PH2"] = meta["x_PH2"]
                gS.create_dataset("PH1_Pa", data=ph1)
                gS.create_dataset("PH2_Pa", data=ph2)
                seen_any = True
            if not seen_any:
                raise FileNotFoundError(f"No raw files found for {L} in {cfg.RAW_BASE}")

            m1 = sio.loadmat(cal_base / f"calib_{L}_1.mat")
            ph1_v, _, nc_v, *_ = _extract_channel_data(m1).T
            ph1_pa = correct_pressure_sensitivity(volts_to_pa(ph1_v, "PH1"), psigs[i])
            nc1_pa = correct_pressure_sensitivity(volts_to_pa(nc_v, "NC"), psigs[i])

            m2 = sio.loadmat(cal_base / f"calib_{L}_2.mat")
            _, ph2_v, nc_v2, *_ = _extract_channel_data(m2).T
            ph2_pa = correct_pressure_sensitivity(volts_to_pa(ph2_v, "PH2"), psigs[i])
            nc2_pa = correct_pressure_sensitivity(volts_to_pa(nc_v2, "NC"), psigs[i])

            gFRF = gL.create_group("FRF_PH_to_NC")
            gFRF.attrs["from"] = "PH"
            gFRF.attrs["to"] = "NC"
            gFRF.attrs["note"] = "Semi-anechoic calibration mapping the pinhole mic to nosecone mic"
            gR1 = gFRF.create_group("Run1")
            gR1.create_dataset("PH1_Pa", data=ph1_pa)
            gR1.create_dataset("NC_Pa", data=nc1_pa)
            gR2 = gFRF.create_group("Run2")
            gR2.create_dataset("PH2_Pa", data=ph2_pa)
            gR2.create_dataset("NC_Pa", data=nc2_pa)


if __name__ == "__main__":
    save_raw_ph_pressure()
