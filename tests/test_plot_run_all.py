from __future__ import annotations

from types import SimpleNamespace

from src.checks.plot import run_all as plot_run_all


def _patch_plot_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    monkeypatch.setattr(
        plot_run_all.bump_raw,
        "plot_fs_raw",
        lambda: calls.append("bump_fs_raw"),
    )
    monkeypatch.setattr(
        plot_run_all.bump_raw,
        "plot_raw",
        lambda: calls.append("bump_wall_raw"),
    )
    monkeypatch.setattr(
        plot_run_all.F_freestreamp_SU_raw,
        "plot_fs_raw",
        lambda: calls.append("generic_fs_raw"),
    )
    monkeypatch.setattr(
        plot_run_all.G_wallp_SU_raw,
        "plot_raw",
        lambda: calls.append("generic_wall_raw"),
    )
    monkeypatch.setattr(
        plot_run_all.F_freestreamp_SU_production,
        "plot_fs_raw",
        lambda: calls.append("fs_prod"),
    )
    monkeypatch.setattr(
        plot_run_all.G_wallp_SU_production,
        "plot_model_comparison_roi",
        lambda: calls.append("wall_prod"),
    )
    monkeypatch.setattr(
        plot_run_all.bump_production,
        "plot_cleaned_by_case",
        lambda: calls.append("bump_wall_prod"),
    )
    monkeypatch.setattr(
        plot_run_all.SU_two_point,
        "plot_2pt_inner",
        lambda: calls.append("2pt_inner"),
    )
    monkeypatch.setattr(
        plot_run_all.SU_two_point,
        "plot_2pt_outer",
        lambda: calls.append("2pt_outer"),
    )
    monkeypatch.setattr(
        plot_run_all.SU_two_point,
        "plot_2pt_speed_outer",
        lambda: calls.append("2pt_speed_outer"),
    )
    monkeypatch.setattr(
        plot_run_all.SU_two_point,
        "plot_2pt_speed_inner",
        lambda: calls.append("2pt_speed_inner"),
    )
    return calls


def test_run_all_uses_bump_raw_plots_for_bump_dataset(monkeypatch) -> None:
    calls = _patch_plot_calls(monkeypatch)
    monkeypatch.setattr(
        plot_run_all,
        "Config",
        lambda: SimpleNamespace(ROOT_DIR="data/bump1", SPACINGS=("close", "far")),
    )

    plot_run_all.run_all()

    assert "bump_fs_raw" in calls
    assert "bump_wall_raw" in calls
    assert "bump_wall_prod" in calls
    assert "fs_prod" not in calls
    assert "generic_fs_raw" not in calls
    assert "generic_wall_raw" not in calls
    assert "wall_prod" not in calls


def test_run_all_uses_generic_raw_plots_for_non_bump_dataset(monkeypatch) -> None:
    calls = _patch_plot_calls(monkeypatch)
    monkeypatch.setattr(
        plot_run_all,
        "Config",
        lambda: SimpleNamespace(ROOT_DIR="data/iso_re", SPACINGS=("close", "far")),
    )

    plot_run_all.run_all()

    assert "generic_fs_raw" in calls
    assert "fs_prod" in calls
    assert "generic_wall_raw" in calls
    assert "wall_prod" in calls
    assert "bump_fs_raw" not in calls
    assert "bump_wall_raw" not in calls
    assert "bump_wall_prod" not in calls
