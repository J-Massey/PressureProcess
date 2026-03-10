"""
Run all plotting sanity checks.
"""

from __future__ import annotations

from pathlib import Path

from src.config_params import Config
from src.checks.plot import (
    F_freestreamp_SU_raw,
    F_freestreamp_SU_production,
    G_wallp_SU_raw,
    G_wallp_SU_production,
    SU_two_point,
)
from src.checks.plot_bump import raw as bump_raw
from src.checks.plot_bump import production as bump_production


def run_all() -> None:
    cfg = Config()
    dataset_name = Path(cfg.ROOT_DIR).name.lower()

    if dataset_name.startswith("bump"):
        bump_raw.plot_fs_raw()
        bump_raw.plot_raw()
    else:
        F_freestreamp_SU_raw.plot_fs_raw()
        G_wallp_SU_raw.plot_raw()

    F_freestreamp_SU_production.plot_fs_raw()
    if dataset_name.startswith("bump"):
        bump_production.plot_cleaned_by_case()
    else:
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
