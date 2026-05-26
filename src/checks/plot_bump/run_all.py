"""
Run all bump plot sanity checks: calibration, raw, noise-rejected.
"""

from __future__ import annotations

from src.checks.plot_bump import (
    calibration, raw, production, fs_production, wall_stages, coherence,
)


def run_all() -> None:
    print("[plot] bump calibration")
    calibration.plot_all()
    print("[plot] bump raw")
    raw.plot_all()
    print("[plot] bump freestream production (NC->nkd)")
    fs_production.plot_all()
    print("[plot] bump wall pipeline stages")
    wall_stages.plot_all()
    print("[plot] bump noise-rejected production (inner-scaled)")
    production.plot_all()
    print("[plot] bump NC<->PH coherence (Wiener diagnostic)")
    coherence.plot_all()


if __name__ == "__main__":
    run_all()
