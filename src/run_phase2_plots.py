"""
Run the phase2 processing pipeline and then all phase2 plots.
"""

from __future__ import annotations

from src.save_phase2.run_all import run_all as run_processing
from src.checks.plot_phase2.run_all import run_all as run_plots


def run_pipeline() -> None:
    run_processing()
    run_plots()


if __name__ == "__main__":
    run_pipeline()
