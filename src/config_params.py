"""
Config parameters for the processing pipeline.
Keep all hard-coded paths and constants here so the rest of the pipeline stays consistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _profile_name() -> str:
    return os.getenv("PRESSUREPROCESS_PROFILE", "").strip().lower()


def _use_iso_profile() -> bool:
    return _profile_name() in {"iso_re", "iso", "sarah_iso_re"}


@dataclass(frozen=True)
class Config:
    # --- Experiment/run metadata defaults ---
    ROOT_DIR: str = field(
        default_factory=lambda: os.getenv(
            "PRESSUREPROCESS_ROOT_DIR",
            "data/bump1",
        )
    )
    LABELS: tuple[str, str, str] = field(
        default_factory=lambda: (
            ("0psig", "30psig", "50psig")
            if _use_iso_profile()
            else ("0psig", "50psig", "100psig")
        )
    )
    PSIGS: tuple[float, float, float] = field(
        default_factory=lambda: (
            (0.0, 30.0, 50.0) if _use_iso_profile() else (0.0, 50.0, 100.0)
        )
    )
    U_TAU: tuple[float, float, float] = (0.537, 0.522, 0.506)
    U_E: tuple[float, float, float] = (14., 14., 14.)
    ANALOG_LP_FILTER: tuple[int, int, int] = (2100, 4700, 14100)
    F_CUTS: tuple[float, float, float] = (1200.0, 4000.0, 10000.0)  # per-label anti-alias lowpass in Hz
    U_TAU_REL_UNC: tuple[float, float, float] = (0.2, 0.1, 0.05)
    SPACINGS: tuple[str, ...] = field(
        default_factory=lambda: ("close",) if _use_iso_profile() else ("close", "far")
    )
    RUN_NC_CALIBS: bool = field(default_factory=lambda: not _use_iso_profile())
    INCLUDE_NC_CALIB_RAW: bool = field(default_factory=lambda: not _use_iso_profile())

    # --- Sampling / spectral defaults ---
    FS: float = 50_000.0
    NPERSEG: int = 2**12
    WINDOW: str = "hann"

    # --- Physical constants ---
    R: float = 287.05        # J/kg/K
    PSI_TO_PA: float = 6_894.76
    P_ATM: float = 101_325.0
    DELTA: tuple[float, float, float] = (0.035, 0.035, 0.035)  # m, bl-height of channel
    TDEG: tuple[float, float, float] = (18.0, 20.0, 22.0)
    TPLUS_CUT: float = 10.0  # picked so that we cut at half the inner peak

    # --- Data paths ---
    RAW_CAL_BASE: str = field(init=False)
    RAW_BASE: str = field(init=False)
    
    # --- Output paths ---
    TF_BASE: str = field(init=False)
    PH_RAW_FILE: str = field(init=False)
    PH_PROCESSED_FILE: str = field(init=False)
    NKD_RAW_FILE: str = field(init=False)
    NKD_PROCESSED_FILE: str = field(init=False)


    # --- Sensor constants ---
    SENSITIVITIES_V_PER_PA: dict[str, float] = field(
        default_factory=lambda: {
            "PH1": 50.9e-3,
            "PH2": 51.7e-3,
            "NC": 52.4e-3,
            "nkd": 50.9e-3,
        }
    )
    PREAMP_GAIN: dict[str, float] = field(
        default_factory=lambda: {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0}
    )

    def __post_init__(self) -> None:
        root = self.ROOT_DIR.rstrip("/")
        object.__setattr__(self, "RAW_CAL_BASE", f"{root}/raw_calib")
        object.__setattr__(self, "RAW_BASE", f"{root}/raw_wallp")
        object.__setattr__(self, "TF_BASE", f"{root}/calibration")
        object.__setattr__(self, "PH_RAW_FILE", f"{root}/pressure/G_wallp_SU_raw.hdf5")
        object.__setattr__(
            self,
            "PH_PROCESSED_FILE",
            f"{root}/pressure/G_wallp_SU_production.hdf5",
        )
        object.__setattr__(
            self,
            "NKD_RAW_FILE",
            f"{root}/pressure/F_freestreamp_SU_raw.hdf5",
        )
        object.__setattr__(
            self,
            "NKD_PROCESSED_FILE",
            f"{root}/pressure/F_freestreamp_SU_production.hdf5",
        )

    
