"""Dispatch contract for src/checks/plot/run_all.py.

run_all picks its plot set from the *name* of the configured dataset root, so
a renamed dataset directory silently changes which figures get produced. These
tests patch the leaf plot calls (drawing is exercised by test_plot_style and by
the real runs) and assert only the routing.

Unlike the previous version of this file, the patched attribute names are the
ones that actually exist -- see test_case_wiring.py for the check that keeps
them honest.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.checks.plot import run_all as plot_run_all


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    recorded: list[str] = []

    def record(module, attribute: str, tag: str) -> None:
        monkeypatch.setattr(module, attribute, lambda: recorded.append(tag))

    record(plot_run_all.bump_raw, "plot_fs_raw", "bump:fs_raw")
    record(plot_run_all.bump_raw, "plot_wall_raw", "bump:wall_raw")
    record(plot_run_all.bump_production, "plot_cleaned_by_case", "bump:wall_prod")

    record(plot_run_all.F_freestreamp_SU_raw, "plot_fs_raw", "generic:fs_raw")
    record(plot_run_all.G_wallp_SU_raw, "plot_raw", "generic:wall_raw")
    record(plot_run_all.F_freestreamp_SU_production, "plot_fs_raw", "generic:fs_prod")
    record(plot_run_all.G_wallp_SU_production, "plot_model_comparison_roi",
           "generic:wall_prod_roi")
    record(plot_run_all.G_wallp_SU_production, "plot_stages_3panel",
           "generic:wall_prod_stages")

    for name in ("plot_2pt_inner", "plot_2pt_outer",
                 "plot_2pt_speed_outer", "plot_2pt_speed_inner"):
        record(plot_run_all.SU_two_point, name, f"twopt:{name}")

    return recorded


def use_dataset(monkeypatch, root_dir: str, spacings=("close", "far")) -> None:
    monkeypatch.setattr(
        plot_run_all, "Config",
        lambda: SimpleNamespace(ROOT_DIR=root_dir, SPACINGS=spacings),
    )


def test_bump_dataset_uses_the_bump_plot_set(monkeypatch, calls) -> None:
    use_dataset(monkeypatch, "data/bump1")

    plot_run_all.run_all()

    assert "bump:fs_raw" in calls
    assert "bump:wall_raw" in calls
    assert "bump:wall_prod" in calls
    assert not [c for c in calls if c.startswith("generic:")]


def test_non_bump_dataset_uses_the_generic_plot_set(monkeypatch, calls) -> None:
    use_dataset(monkeypatch, "data/iso_re", spacings=("close",))

    plot_run_all.run_all()

    assert "generic:fs_raw" in calls
    assert "generic:wall_raw" in calls
    assert "generic:fs_prod" in calls
    assert "generic:wall_prod_roi" in calls
    assert "generic:wall_prod_stages" in calls
    assert not [c for c in calls if c.startswith("bump:")]


def test_dispatch_is_by_directory_name_not_full_path(monkeypatch, calls) -> None:
    """An absolute root such as /app/data/bump1 (the Docker layout) must still
    select the bump plots."""
    use_dataset(monkeypatch, "/app/data/bump1")

    plot_run_all.run_all()

    assert "bump:wall_raw" in calls
    assert not [c for c in calls if c.startswith("generic:")]


def test_two_point_plots_require_both_spacings(monkeypatch, calls) -> None:
    """The two-point correlations compare the close and far pinhole pairs, so
    they must be skipped for close-only datasets (phase2, iso_re) rather than
    crashing on a missing group."""
    use_dataset(monkeypatch, "data/phase1", spacings=("close", "far"))
    plot_run_all.run_all()
    assert len([c for c in calls if c.startswith("twopt:")]) == 4

    calls.clear()
    use_dataset(monkeypatch, "data/phase2", spacings=("close",))
    plot_run_all.run_all()
    assert not [c for c in calls if c.startswith("twopt:")]
