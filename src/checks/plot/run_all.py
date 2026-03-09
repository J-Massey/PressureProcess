"""
Run all plotting sanity checks.
"""

from __future__ import annotations

from src.config_params import Config
from src.checks.plot import (
    F_freestreamp_SU_raw,
    F_freestreamp_SU_production,
    G_wallp_SU_raw,
    G_wallp_SU_production,
    SU_two_point,
)


def run_all() -> None:
    cfg = Config()

    F_freestreamp_SU_raw.plot_fs_raw()
    F_freestreamp_SU_production.plot_fs_raw()
    G_wallp_SU_raw.plot_raw()
    G_wallp_SU_production.plot_model_comparison_roi()

    if {"close", "far"}.issubset(set(cfg.SPACINGS)):
        SU_two_point.plot_2pt_inner()
        SU_two_point.plot_2pt_outer()
        SU_two_point.plot_2pt_speed_outer()
        SU_two_point.plot_2pt_speed_inner()
    else:
        print("[skip] two-point plots require close and far spacings")


if __name__ == "__main__":
    run_all()
