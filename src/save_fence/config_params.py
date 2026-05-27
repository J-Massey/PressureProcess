"""
Config parameters for the fence (rough surface, single combined run) pipeline.

Hard-coded for the fence case. Tune values here, not via env vars.

The fence dataset differs from bump/iso_re:
  - Wall-pressure data is delivered as a single combined MAT file
    (ATM_Rev1.mat at the case root), not per-spacing per-pressure files.
  - raw_calib has the usual PH/ and NC/ subtrees for FRF construction.
  - There are no separate "close"/"far" spacings; this case currently has
    SPACINGS = ("combined",) as a single placeholder.

Most save files for fence will diverge more from the canonical pipeline
than iso_re/bump did, because the raw wall-pressure ingest step has to
read ATM_Rev1.mat instead of looping over `<spacing>/<label>.mat`. The
scaffolding here keeps the same five-step shape so it's easy to fill in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=False)
class Config:
    # --- Case identity ---
    CASE: str = "fence"
    ROOT_DIR: str = "data/fence"

    # --- Experiment/run metadata ---
    LABELS: tuple[str, str, str] = ("0psig", "50psig", "100psig")
    PSIGS: tuple[float, float, float] = (0.0, 50.0, 100.0)
    U_TAU: tuple[float, float, float] = (0.537, 0.522, 0.506)
    U_E: tuple[float, float, float] = (14.0, 14.0, 14.0)
    ANALOG_LP_FILTER: tuple[int, int, int] = (2100, 4700, 14100)
    F_CUTS: tuple[float, float, float] = (1200.0, 4000.0, 10000.0)
    U_TAU_REL_UNC: tuple[float, float, float] = (0.2, 0.1, 0.05)

    # fence delivers one combined wall-pressure file rather than per-spacing.
    SPACINGS: tuple[str, ...] = ("combined",)

    # Per-sensor-position friction velocity [m/s], indexed as
    # U_TAU_BY_POSITION[label][spacing][channel] -- mirrors save_bump.
    # Fence has SPACINGS=("combined",), so the position legend in plots is
    # P1=(combined, PH1), P2=(combined, PH2). Defaults seeded with the
    # per-pressure U_TAU value; fill in real per-position values when known.
    U_TAU_BY_POSITION: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: {
            "0psig":   {"combined": {"PH1": 0.537, "PH2": 0.537}},
            "50psig":  {"combined": {"PH1": 0.522, "PH2": 0.522}},
            "100psig": {"combined": {"PH1": 0.506, "PH2": 0.506}},
        }
    )

    RUN_NC_CALIBS: bool = True
    INCLUDE_NC_CALIB_RAW: bool = True

    # --- Fence raw-mat channel layout ---
    # channelData in ATM_Rev1.mat is (N, 4): [PH1, PH2, NC, ref].
    # The 4th channel is a static reference pressure used to determine the
    # inflow Re (tunnel static pressure).
    ATM_CHANNEL_COLUMNS: dict[str, int] = field(
        default_factory=lambda: {"PH1": 0, "PH2": 1, "NC": 2, "ref": 3}
    )
    # Sub-key inside ATM_Rev1.mat that holds the no-flow baseline (same shape).
    ATM_NOFLOW_KEY: str = "channelData_noflow"

    # --- Sampling / spectral defaults ---
    FS: float = 50_000.0
    NPERSEG: int = 2**12
    WINDOW: str = "hann"

    # --- Physical constants ---
    R: float = 287.05
    PSI_TO_PA: float = 6_894.76
    P_ATM: float = 101_325.0
    DELTA: tuple[float, float, float] = (0.035, 0.035, 0.035)
    TDEG: tuple[float, float, float] = (18.0, 20.0, 22.0)
    TPLUS_CUT: float = 10.0

    # Fence-specific knobs — placeholders, fill in per the fence geometry.
    FENCE_HEIGHT: float = 0.0       # fence height [m]
    FENCE_THICKNESS: float = 0.0    # fence thickness [m]
    X_FENCE: float = 0.0            # streamwise location of fence [m]

    # --- Sensor constants ---
    SENSITIVITIES_V_PER_PA: dict[str, float] = field(
        default_factory=lambda: {
            "PH1": 50.9e-3,
            "PH2": 51.7e-3,
            "NC": 52.4e-3,
            "nkd": 50.9e-3,
            "ref": 1.0,  # tunnel static reference, units of V/Pa — TUNE.
        }
    )
    PREAMP_GAIN: dict[str, float] = field(
        default_factory=lambda: {"nc": 1.0, "PH1": 1.0, "PH2": 1.0, "NC": 1.0, "ref": 1.0}
    )

    # --- Fence-specific ingest path ---
    # The combined wall-pressure file lives at <ROOT_DIR>/<ATM_FILE>.
    ATM_FILE: str = "ATM_Rev1.mat"

    # --- Derived data paths (built from ROOT_DIR) ---
    RAW_CAL_BASE: str = field(init=False)
    RAW_BASE: str = field(init=False)
    ATM_PATH: str = field(init=False)
    TF_BASE: str = field(init=False)
    PH_RAW_FILE: str = field(init=False)
    PH_PROCESSED_FILE: str = field(init=False)
    NKD_RAW_FILE: str = field(init=False)
    NKD_PROCESSED_FILE: str = field(init=False)

    def __post_init__(self) -> None:
        root = self.ROOT_DIR.rstrip("/")
        object.__setattr__(self, "RAW_CAL_BASE", f"{root}/raw_calib")
        object.__setattr__(self, "RAW_BASE", f"{root}/raw_wallp")
        object.__setattr__(self, "ATM_PATH", f"{root}/{self.ATM_FILE}")
        object.__setattr__(self, "TF_BASE", f"{root}/calibration")
        object.__setattr__(self, "PH_RAW_FILE", f"{root}/pressure/G_wallp_SU_raw.hdf5")
        object.__setattr__(self, "PH_PROCESSED_FILE", f"{root}/pressure/G_wallp_SU_production.hdf5")
        object.__setattr__(self, "NKD_RAW_FILE", f"{root}/pressure/F_freestreamp_SU_raw.hdf5")
        object.__setattr__(self, "NKD_PROCESSED_FILE", f"{root}/pressure/F_freestreamp_SU_production.hdf5")
