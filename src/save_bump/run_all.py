"""
Run the bump (rough surface) processing pipeline from raw .mat files.

Order:
1) calibs (PH + NC)
2a) fs_raw
2b) pw_raw
3a) fs_proc
3b) pw_proc
"""

from __future__ import annotations

from src.save_bump import calibs, fs_raw, pw_raw, fs_proc, pw_proc
from src.save_bump.config_params import Config


def run_all() -> None:
    cfg = Config()

    print("[1] calibs: PH + NC (bump)")
    calibs.save_PH_calibs()
    if cfg.RUN_NC_CALIBS:
        calibs.save_NC_calibs()

    print("[2] fs_raw")
    fs_raw.save_raw_fs_pressure(
        spacings=cfg.SPACINGS,
        include_nc_calib=cfg.INCLUDE_NC_CALIB_RAW,
    )

    print("[3] pw_raw")
    pw_raw.save_raw_ph_pressure(spacings=cfg.SPACINGS)

    print("[4a] fs_proc")
    fs_proc.save_prod_fs_pressure(spacings=cfg.SPACINGS)

    print("[4b] pw_proc")
    pw_proc.save_corrected_pressure(spacings=cfg.SPACINGS)


if __name__ == "__main__":
    run_all()
