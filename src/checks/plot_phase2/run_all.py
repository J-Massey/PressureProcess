"""
Run all phase2 plot sanity checks: calibration, raw, fs_production,
wall_stages, noise-rejected production, NC<->PH coherence.
"""

from __future__ import annotations

from src.checks.plot_phase2 import (
    calibration, raw, production, fs_production, wall_stages, coherence,
)


def run_all() -> None:
    print("[plot] phase2 calibration")
    calibration.plot_all()
    print("[plot] phase2 raw")
    raw.plot_all()
    print("[plot] phase2 freestream production (NC->nkd)")
    fs_production.plot_all()
    print("[plot] phase2 wall pipeline stages")
    wall_stages.plot_all()
    print("[plot] phase2 noise-rejected production (inner-scaled)")
    production.plot_all()
    print("[plot] phase2 NC<->PH coherence (Wiener diagnostic)")
    coherence.plot_all()


if __name__ == "__main__":
    run_all()
