from __future__ import annotations

import importlib
import sys
import types


def test_run_pipeline_calls_processing_only(monkeypatch) -> None:
    calls: list[str] = []

    save_run_all = types.ModuleType("src.save.run_all")
    save_run_all.run_all = lambda: calls.append("processing")

    plot_run_all = types.ModuleType("src.checks.plot.run_all")
    plot_run_all.run_all = lambda: calls.append("plots")

    monkeypatch.setitem(sys.modules, "src.save.run_all", save_run_all)
    monkeypatch.setitem(sys.modules, "src.checks.plot.run_all", plot_run_all)
    sys.modules.pop("src.run_pipeline", None)

    mod = importlib.import_module("src.run_pipeline")
    mod.run_pipeline()

    assert calls == ["processing"]


def test_run_bump_plots_calls_processing_then_plots(monkeypatch) -> None:
    calls: list[str] = []

    save_run_all = types.ModuleType("src.save.run_all")
    save_run_all.run_all = lambda: calls.append("processing")

    plot_run_all = types.ModuleType("src.checks.plot.run_all")
    plot_run_all.run_all = lambda: calls.append("plots")

    monkeypatch.setitem(sys.modules, "src.save.run_all", save_run_all)
    monkeypatch.setitem(sys.modules, "src.checks.plot.run_all", plot_run_all)
    sys.modules.pop("src.run_bump_plots", None)

    mod = importlib.import_module("src.run_bump_plots")
    mod.run_pipeline()

    assert calls == ["processing", "plots"]
