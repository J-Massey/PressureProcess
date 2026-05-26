"""
Run all fence plot sanity checks: calibration, raw, noise-rejected.
"""

from __future__ import annotations

from src.checks.plot_fence import calibration, raw, production


def run_all() -> None:
    print("[plot] fence calibration")
    calibration.plot_all()
    print("[plot] fence raw")
    raw.plot_all()
    print("[plot] fence noise-rejected production")
    production.plot_all()


if __name__ == "__main__":
    run_all()
