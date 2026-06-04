"""
Config parameters for the bump (rough surface) pipeline.

Hard-coded for the bump1 case. Tune values here, not via env vars.

The bump1 data has the same outer layout as the phase1 (smooth) case
— (0, 50, 100) psig, close+far spacings, and full NC semi-anechoic calibration
— but the surface is rough, so quantities like u_tau, delta, and the various
length scales need tuning here independently of the smooth case.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=False)
class Config:
    # --- Case identity ---
    CASE: str = "bump"
    ROOT_DIR: str = "data/bump1"

    # --- Experiment/run metadata (TUNE these for the rough surface) ---
    LABELS: tuple[str, str, str] = ("0psig", "50psig", "100psig")
    PSIGS: tuple[float, float, float] = (0.0, 50.0, 100.0)
    # Friction velocities for the rough wall — DIFFERENT from smooth; tune.
    U_TAU: tuple[float, float, float] = (0.537, 0.522, 0.506)
    U_E: tuple[float, float, float] = (14.0, 14.0, 14.0)
    ANALOG_LP_FILTER: tuple[int, int, int] = (2100, 4700, 14100)
    F_CUTS: tuple[float, float, float] = (1200.0, 4000.0, 10000.0)
    U_TAU_REL_UNC: tuple[float, float, float] = (0.2, 0.1, 0.05)

    SPACINGS: tuple[str, ...] = ("close", "far")

    # Per-sensor-position friction velocity [m/s], indexed as
    # U_TAU_BY_POSITION[label][spacing][channel]. Mapping mirrors
    # plot_bump/raw.py's P1..P4 legend:
    #   P1 = (close, PH1)   P2 = (close, PH2)
    #   P3 = (far,   PH1)   P4 = (far,   PH2)
    U_TAU_BY_POSITION: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: {
            # ATM (0 psig): P1=0.5414, P2=0.5818, P3=0.5949, P4=0.5752
            "0psig":   {"close": {"PH1": 0.5414, "PH2": 0.5818},
                         "far":   {"PH1": 0.5949, "PH2": 0.5752}},
            # 50 psig:    P1=0.4836, P2=0.5615, P3=0.5221, P4=0.4986
            "50psig":  {"close": {"PH1": 0.4836, "PH2": 0.5615},
                         "far":   {"PH1": 0.5221, "PH2": 0.4986}},
            # 100 psig:   P1=0.4854, P2=0.5232, P3=0.4831, P4=0.4858
            "100psig": {"close": {"PH1": 0.4854, "PH2": 0.5232},
                         "far":   {"PH1": 0.4831, "PH2": 0.4858}},
        }
    )

    # Bump has full NC semi-anechoic calibration data.
    RUN_NC_CALIBS: bool = True
    INCLUDE_NC_CALIB_RAW: bool = True

    # --- TF smoothing (applied at save time; raw H also kept for diagnostics) ---
    TF_SMOOTH_OCT: float = 1 / 6
    TF_SMOOTH_PPO: int = 48

    # --- Sampling / spectral defaults ---
    FS: float = 50_000.0
    NPERSEG: int = 2**12
    WINDOW: str = "hann"

    # --- Physical constants ---
    R: float = 287.05
    PSI_TO_PA: float = 6_894.76
    P_ATM: float = 101_325.0
    # delta: rough-wall boundary-layer thickness — TUNE per pressure.
    DELTA: tuple[float, float, float] = (0.035, 0.035, 0.035)
    TDEG: tuple[float, float, float] = (18.0, 20.0, 22.0)
    TPLUS_CUT: float = 10.0

    # Roughness-specific knobs.
    K_S: float = 0.0           # equivalent sand-grain roughness [m]
    BUMP_HEIGHT: float = 20.01e-3  # nominal bump height [m] (20.01 mm)

    # Per-sensor-position local BL thickness [m], indexed as
    # DELTA_BY_POSITION[label][spacing][channel]. Same P1..P4 mapping as
    # U_TAU_BY_POSITION (P1=close PH1, P2=close PH2, P3=far PH1, P4=far PH2).
    DELTA_BY_POSITION: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: {
            # 0 psig: P1=56.98 mm, P2=51.60, P3=42.73, P4=43.32
            "0psig":   {"close": {"PH1": 56.98e-3, "PH2": 51.60e-3},
                         "far":   {"PH1": 42.73e-3, "PH2": 43.32e-3}},
            # 50 psig: P1=55.76, P2=42.63, P3=38.82, P4=39.36
            "50psig":  {"close": {"PH1": 55.76e-3, "PH2": 42.63e-3},
                         "far":   {"PH1": 38.82e-3, "PH2": 39.36e-3}},
            # 100 psig: P1=44.51, P2=42.07, P3=38.50, P4=39.06
            "100psig": {"close": {"PH1": 44.51e-3, "PH2": 42.07e-3},
                         "far":   {"PH1": 38.50e-3, "PH2": 39.06e-3}},
        }
    )

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
