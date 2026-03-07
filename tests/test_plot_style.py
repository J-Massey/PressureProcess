from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.checks.plot._style import apply_plot_style


def test_apply_plot_style_auto_without_latex(monkeypatch) -> None:
    monkeypatch.setenv("PRESSUREPROCESS_USE_TEX", "auto")
    monkeypatch.setattr("src.checks.plot._style.shutil.which", lambda _: None)

    apply_plot_style()

    assert plt.rcParams["text.usetex"] is False


def test_apply_plot_style_explicit_false(monkeypatch) -> None:
    monkeypatch.setenv("PRESSUREPROCESS_USE_TEX", "false")

    apply_plot_style()

    assert plt.rcParams["text.usetex"] is False


def test_apply_plot_style_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("PRESSUREPROCESS_USE_TEX", "definitely-not-valid")

    with pytest.raises(ValueError, match="PRESSUREPROCESS_USE_TEX"):
        apply_plot_style()
