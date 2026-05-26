# Freestream raw pressure (NC channel) — fence variant.
#
# NOTE: fence delivers wall-pressure as a single combined MAT file
# (cfg.ATM_PATH = "<ROOT_DIR>/ATM_Rev1.mat"), NOT as per-spacing per-pressure
# files. The ingest code below is a scaffold structured the same way as bump
# so you can fill it in once the ATM_Rev1.mat layout is settled. The
# placeholder reads the combined file once and extracts NC per pressure.
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import h5py
import scipy.io as sio

from src.save_fence.config_params import Config

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


def _load_atm_combined() -> dict:
    """Load the atmospheric (0 psig) fence wall-pressure file."""
    atm_path = Path(cfg.ATM_PATH)
    if not atm_path.exists():
        raise FileNotFoundError(f"fence ATM file missing: {atm_path}")
    return sio.loadmat(atm_path)


def _extract_channel(atm: dict, channel: str, *, noflow: bool = False) -> np.ndarray:
    """
    Return one column of ATM_Rev1.mat as float volts.

    The 4 columns of `channelData` are mapped by cfg.ATM_CHANNEL_COLUMNS:
    {"PH1": 0, "PH2": 1, "NC": 2, "ref": 3}. Pass noflow=True to read from
    the no-flow baseline (cfg.ATM_NOFLOW_KEY) instead.
    """
    key = cfg.ATM_NOFLOW_KEY if noflow else "channelData"
    if key not in atm:
        raise KeyError(f"fence ATM file missing key {key!r}")
    col = cfg.ATM_CHANNEL_COLUMNS[channel]
    return np.asarray(atm[key][:, col], dtype=float)


def _per_pressure_mat_path(label: str) -> Path:
    """
    Path to a per-pressure wall-pressure file when those land later.
    Mirrors bump's <RAW_BASE>/<spacing>/<label>.mat layout but fence's single
    pseudo-spacing is "combined".
    """
    return Path(cfg.RAW_BASE) / "combined" / f"{label}.mat"


def _extract_NC_for_pressure(atm: dict, label: str, psig: float) -> np.ndarray | None:
    """
    Return the NC volts for one pressure, or None if the data isn't on disk.

    psig == 0 reads ATM_Rev1.mat. Other pressures look for a per-pressure
    file at _per_pressure_mat_path(label) with the same 4-column layout;
    return None if missing so the caller can skip and continue.
    """
    if float(psig) == 0.0:
        return _extract_channel(atm, "NC")
    mat_path = _per_pressure_mat_path(label)
    if not mat_path.exists():
        return None
    mat = sio.loadmat(mat_path)
    col = cfg.ATM_CHANNEL_COLUMNS["NC"]
    return np.asarray(mat["channelData"][:, col], dtype=float)


def _extract_ref_for_pressure(atm: dict, label: str, psig: float) -> np.ndarray | None:
    """Same shape as _extract_NC_for_pressure but returns the tunnel-Re reference channel."""
    if float(psig) == 0.0:
        return _extract_channel(atm, "ref")
    mat_path = _per_pressure_mat_path(label)
    if not mat_path.exists():
        return None
    mat = sio.loadmat(mat_path)
    col = cfg.ATM_CHANNEL_COLUMNS["ref"]
    return np.asarray(mat["channelData"][:, col], dtype=float)


def save_raw_fs_pressure(
    *,
    spacings: tuple[str, ...] | None = None,
    include_nc_calib: bool | None = None,
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
    include_nc_calib = cfg.INCLUDE_NC_CALIB_RAW if include_nc_calib is None else include_nc_calib

    fs_raw = cfg.NKD_RAW_FILE
    os.makedirs(Path(fs_raw).parent, exist_ok=True)

    atm = _load_atm_combined()

    with h5py.File(fs_raw, "w") as hf:
        hf.attrs["title"] = "Freestream pressure (nose-cone) - raw \& calibration [fence]"
        hf.attrs["fs_Hz"] = FS
        hf.attrs["Ue_m_per_s"] = np.asarray(Ue, float)
        hf.attrs["DAQ"] = "24-bit"
        hf.attrs["mic_details"] = "HB&K 1/2'' Type 4964"
        hf.attrs["case"] = cfg.CASE
        hf.attrs["atm_file"] = cfg.ATM_FILE
        hf.attrs["description"] = (
            "Raw freestream NC signals (fence). psig=0 comes from "
            "ATM_Rev1.mat; non-atmospheric pressures expect "
            "<RAW_BASE>/combined/<label>.mat with the same 4-column layout."
        )

        g_fs = hf.create_group("freestream_raw")

        seen_any = False
        for i, L in enumerate(labels):
            NC_v = _extract_NC_for_pressure(atm, L, psigs[i])
            if NC_v is None:
                print(f"[skip] fence {L}: no per-pressure file yet")
                continue
            ref_v = _extract_ref_for_pressure(atm, L, psigs[i])

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

            NC = correct_pressure_sensitivity(volts_to_pa(NC_v, "NC"), psigs[i])

            # fence has a single "combined" pseudo-spacing.
            for sp in spacings:
                gS = gL.create_group(sp)
                gS.create_dataset("NC_Pa", data=NC)
                if ref_v is not None:
                    # Reference (tunnel static) channel — used to set inflow Re.
                    gS.create_dataset("ref_V", data=ref_v)
            seen_any = True

            if include_nc_calib:
                base = Path(cfg.RAW_CAL_BASE) / "NC"
                mat_path = base / f"{L}/nkd-ns_nofacilitynoise.mat"
                if not mat_path.exists():
                    print(f"[skip] missing NC calib file: {mat_path}")
                else:
                    m1 = sio.loadmat(mat_path)
                    if L == "100psig":
                        nkd_cal, nc_cal = m1["channelData_nofacitynoise"].T
                    else:
                        nkd_cal, nc_cal = m1["channelData"].T

                    nc_cal = correct_pressure_sensitivity(volts_to_pa(nc_cal, "NC"), psigs[i])
                    nkd_cal = correct_pressure_sensitivity(volts_to_pa(nkd_cal, "nkd"), psigs[i])

                    gFRF = gL.create_group("FRF_NC_to_nkd")
                    gFRF.create_dataset("NC_Pa", data=nc_cal)
                    gFRF.create_dataset("nkd_Pa", data=nkd_cal)
                    gFRF.attrs["from"] = "NC"
                    gFRF.attrs["to"] = "nkd"
                    gFRF.attrs["note"] = "Semi-anechoic calibration mapping"

        if not seen_any:
            raise FileNotFoundError(
                "fence freestream ingest found no readable data — check ATM_Rev1.mat path"
            )


if __name__ == "__main__":
    save_raw_fs_pressure()
