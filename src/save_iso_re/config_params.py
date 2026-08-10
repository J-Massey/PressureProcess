"""
Config parameters for the iso_re pipeline.

Hard-coded for the iso_re case. Tune values here, not via env vars.

Notable differences from the default (phase1) pipeline:
  - Pressures: (0, 30, 50) psig — NOT (0, 50, 100).
  - Only the "close" spacing exists (no "far" pinhole spacing run).
  - NC semi-anechoic calibration runs are NOT used here (the available NC
    calib pressures don't match the run pressures), so RUN_NC_CALIBS and
    INCLUDE_NC_CALIB_RAW are both False, and fs_proc / pw_proc fall back to
    an identity FRF for the NC -> nkd mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.band_limits import reported_band_f_cuts


@dataclass(frozen=False)
class Config:
    # --- Case identity ---
    CASE: str = "iso_re"
    ROOT_DIR: str = "data/iso_re"

    # --- Experiment/run metadata ---
    LABELS: tuple[str, str, str] = ("0psig", "30psig", "50psig")
    PSIGS: tuple[float, float, float] = (0.0, 30.0, 50.0)
    U_TAU: tuple[float, float, float] = (0.537, 0.522, 0.506)
    U_E: tuple[float, float, float] = (14.0, 14.0, 14.0)
    ANALOG_LP_FILTER: tuple[int, int, int] = (2100, 4700, 14100)
    # Reported-band upper edge per condition; computed in __post_init__ as
    # min(Helmholtz-resonance guard, T+ = TPLUS_CUT limit), see src/band_limits.py.
    F_CUTS: tuple[float, float, float] = field(init=False)
    U_TAU_REL_UNC: tuple[float, float, float] = (0.2, 0.1, 0.05)

    # iso_re has only the "close" pinhole spacing.
    SPACINGS: tuple[str, ...] = ("close",)

    # NC semi-anechoic calibration is not run for iso_re — pressures don't match.
    RUN_NC_CALIBS: bool = False
    INCLUDE_NC_CALIB_RAW: bool = False

    # Noise-canceller backend for pw_proc. "auto" picks per-platform; otherwise
    # one of {"wiener", "hybrid"}.
    PW_NOISE_CANCELLER: str = "auto"

    # --- Sampling / spectral defaults ---
    FS: float = 50_000.0
    NPERSEG: int = 2**12
    WINDOW: str = "hann"

    # --- Physical constants ---
    R: float = 287.05           # J/kg/K
    PSI_TO_PA: float = 6_894.76
    P_ATM: float = 101_325.0
    DELTA: tuple[float, float, float] = (0.035, 0.035, 0.035)   # m
    TDEG: tuple[float, float, float] = (18.0, 20.0, 22.0)
    TPLUS_CUT: float = 10.0

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

    # --- Derived data paths (built from ROOT_DIR) ---
    RAW_CAL_BASE: str = field(init=False)
    RAW_BASE: str = field(init=False)
    TF_BASE: str = field(init=False)
    PH_RAW_FILE: str = field(init=False)
    PH_PROCESSED_FILE: str = field(init=False)
    NKD_RAW_FILE: str = field(init=False)
    NKD_PROCESSED_FILE: str = field(init=False)

    def __post_init__(self) -> None:
        root = self.ROOT_DIR.rstrip("/")
        object.__setattr__(self, "RAW_CAL_BASE", f"{root}/raw_calib")
        object.__setattr__(self, "RAW_BASE", f"{root}/raw_wallp")
        object.__setattr__(self, "TF_BASE", f"{root}/calibration")
        object.__setattr__(self, "PH_RAW_FILE", f"{root}/pressure/G_wallp_SU_raw.hdf5")
        object.__setattr__(self, "PH_PROCESSED_FILE", f"{root}/pressure/G_wallp_SU_production.hdf5")
        object.__setattr__(self, "NKD_RAW_FILE", f"{root}/pressure/F_freestreamp_SU_raw.hdf5")
        object.__setattr__(self, "NKD_PROCESSED_FILE", f"{root}/pressure/F_freestreamp_SU_production.hdf5")
        object.__setattr__(self, "F_CUTS", reported_band_f_cuts(
            psigs=self.PSIGS, tdegs=self.TDEG, u_taus=self.U_TAU,
            tplus_cut=self.TPLUS_CUT, p_atm=self.P_ATM,
            psi_to_pa=self.PSI_TO_PA, R=self.R))
