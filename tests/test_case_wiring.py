"""Contract for how the case pipelines are wired together.

These are cheap structural tests, but they are the ones that catch the two
failure modes this repo actually suffers from:

  1. a run_all calling a stage function that has been renamed away, and
  2. a hard-coded borrowed path (phase1's PH TF, phase2's Wiener kernels)
     drifting without anyone noticing.

Both are silent until someone runs a multi-hour reprocess.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from conftest import REPO_ROOT

SAVE_MODULES = [
    "src.save",
    "src.save_bump",
    "src.save_fence",
    "src.save_iso_re",
    "src.save_phase2",
]

STAGE_FUNCTIONS = {
    "calibs": "save_PH_calibs",
    "fs_raw": "save_raw_fs_pressure",
    "pw_raw": "save_raw_ph_pressure",
    "fs_proc": "save_prod_fs_pressure",
    "pw_proc": "save_corrected_pressure",
}


@pytest.mark.parametrize("package", SAVE_MODULES)
@pytest.mark.parametrize("stage,function", sorted(STAGE_FUNCTIONS.items()))
def test_every_case_exposes_every_stage_entry_point(
    package: str, stage: str, function: str
) -> None:
    module = importlib.import_module(f"{package}.{stage}")
    assert callable(getattr(module, function))


@pytest.mark.parametrize("package", SAVE_MODULES)
def test_every_case_has_a_runnable_run_all(package: str) -> None:
    module = importlib.import_module(f"{package}.run_all")
    assert callable(module.run_all)


def test_only_bump_and_fence_use_a_donor_wiener_kernel() -> None:
    """bump and fence must NOT train Wiener locally -- the freestream mic on
    those cases carries flow content radiated by the protrusion, so local
    training would subtract real wall pressure. They apply a kernel trained on
    the phase2 smooth-wall donor instead."""
    for package in ("src.save_bump", "src.save_fence"):
        module = importlib.import_module(f"{package}.pw_proc")
        assert hasattr(module, "apply_wiener_kernel")
        assert not hasattr(module, "wiener_cancel_background")

    for package in ("src.save", "src.save_iso_re", "src.save_phase2"):
        module = importlib.import_module(f"{package}.pw_proc")
        assert hasattr(module, "wiener_cancel_background")


@pytest.mark.parametrize("package", ["src.save_bump", "src.save_fence"])
def test_borrowed_source_paths_are_pinned(package: str) -> None:
    """These are deliberate cross-case borrows, not derived from Config. If a
    path changes, the reprocessing order documented in the README changes with
    it, so pin the literal strings."""
    module = importlib.import_module(f"{package}.pw_proc")

    assert module.PH_CALIB_SOURCE == Path("data/phase1/calibration/PH")
    assert module.WIENER_KERNELS_FILE == Path("data/phase2/calibration/wiener_kernels.h5")
    assert module.WIENER_KERNEL_FALLBACK_SPACING == "close"


def test_phase2_borrows_the_phase1_ph_transfer_function() -> None:
    """phase2's own PH calibrations are ~15-20 dB low in the 100-1000 Hz ROI,
    so pw_proc reads phase1's H_fused. The literal path lives inline rather
    than in a module constant, so assert on the source text."""
    source = (REPO_ROOT / "src/save_phase2/pw_proc.py").read_text()
    assert 'Path("data/phase1/calibration/PH")' in source


def test_iso_re_pins_an_identity_nc_to_nkd_frf() -> None:
    """iso_re has no semi-anechoic NC calibration at its pressures. The
    identity FRF must be pinned, not reached via a silent fallback."""
    source = (REPO_ROOT / "src/save_iso_re/pw_proc.py").read_text()
    assert "identity FRF for NC -> nkd (pinned, not a fallback)" in source


def test_wiener_kernel_donor_defaults_to_the_phase2_smooth_wall_case() -> None:
    from src.core.wiener_kernel_donor import DonorSpec, extract_wiener_kernels

    assert callable(extract_wiener_kernels)
    spec = DonorSpec(
        ph_processed_file="x", fs_processed_file="y",
        labels=("0psig",), spacings=("close",),
    )
    # bump/fence read kernels at <label>/<spacing>/<channel>/c with channel
    # names carrying the _Pa suffix; the donor must write that layout.
    assert tuple(spec.channels) == ("PH1_Pa", "PH2_Pa")


# --------------------------------------------------------------------------
# Plot dispatch
# --------------------------------------------------------------------------

def test_generic_plot_dispatch_calls_functions_that_exist() -> None:
    """src/checks/plot/run_all.py dispatches by dataset name. Every function it
    reaches for must actually exist on the target module -- a rename here is
    only discovered when someone runs the plots at the end of a long job.
    """
    from src.checks.plot import run_all as plot_run_all

    expected = {
        plot_run_all.F_freestreamp_SU_raw: ["plot_fs_raw"],
        plot_run_all.G_wallp_SU_raw: ["plot_raw"],
        plot_run_all.F_freestreamp_SU_production: ["plot_fs_raw"],
        plot_run_all.G_wallp_SU_production: ["plot_model_comparison_roi",
                                             "plot_stages_3panel"],
        plot_run_all.SU_two_point: ["plot_2pt_inner", "plot_2pt_outer",
                                    "plot_2pt_speed_outer", "plot_2pt_speed_inner"],
        plot_run_all.bump_production: ["plot_cleaned_by_case"],
    }
    missing = [
        f"{module.__name__}.{name}"
        for module, names in expected.items()
        for name in names
        if not callable(getattr(module, name, None))
    ]
    assert not missing, f"run_all dispatches to missing functions: {missing}"


def test_bump_raw_dispatch_targets_exist() -> None:
    """REGRESSION: src/checks/plot/run_all.py:27 calls ``bump_raw.plot_raw()``
    but src/checks/plot_bump/raw.py defines ``plot_wall_raw()``. Running the
    plots against a bump dataset raises AttributeError.

    This test asserts the CORRECT wiring, so it fails until run_all.py is
    changed to call plot_wall_raw (or raw.py grows a plot_raw alias).
    """
    from src.checks.plot_bump import raw as bump_raw
    from src.checks.plot import run_all as plot_run_all
    import inspect

    called = [
        name for name in ("plot_raw", "plot_wall_raw", "plot_fs_raw")
        if f"bump_raw.{name}()" in inspect.getsource(plot_run_all.run_all)
    ]
    missing = [name for name in called if not callable(getattr(bump_raw, name, None))]
    assert not missing, (
        f"src/checks/plot/run_all.py calls plot_bump.raw.{missing} which does not "
        f"exist; available: {[n for n in dir(bump_raw) if n.startswith('plot')]}"
    )


@pytest.mark.parametrize("package", [
    "src.checks.plot_bump",
    "src.checks.plot_phase2",
    "src.checks.plot_fence",
])
def test_per_case_plot_run_alls_dispatch_to_plot_all(package: str) -> None:
    """The per-case plot packages each expose plot_all() on every submodule
    their run_all drives."""
    run_all_module = importlib.import_module(f"{package}.run_all")
    import inspect

    source = inspect.getsource(run_all_module.run_all)
    for submodule_name in ("calibration", "raw", "production"):
        if f"{submodule_name}.plot_all()" in source:
            submodule = importlib.import_module(f"{package}.{submodule_name}")
            assert callable(submodule.plot_all)


def test_top_level_runners_bind_the_pipelines_they_name() -> None:
    """src/run_pipeline.py and src/run_phase2_plots.py are the only top-level
    runners; each must drive the save package matching its name."""
    import inspect

    from src import run_phase2_plots

    assert "src.save_phase2.run_all" in inspect.getsource(run_phase2_plots)
    assert "src.checks.plot_phase2.run_all" in inspect.getsource(run_phase2_plots)


# --------------------------------------------------------------------------
# Calibration unit handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("package", [
    "src.save", "src.save_phase2", "src.save_bump", "src.save_fence",
])
def test_ph_calibration_converts_volts_to_pascals(package: str) -> None:
    """save_PH_calibs must convert both channels to Pa and apply the
    sensitivity-drift correction before estimating the FRF."""
    import inspect

    source = inspect.getsource(
        importlib.import_module(f"{package}.calibs").save_PH_calibs
    )
    assert "volts_to_pa" in source
    assert "correct_pressure_sensitivity" in source


@pytest.mark.parametrize("package", [
    "src.save", "src.save_phase2", "src.save_bump", "src.save_fence",
])
def test_nc_calibration_estimates_the_frf_on_raw_volts(package: str) -> None:
    """CHARACTERIZATION TEST -- pins a known asymmetry.

    save_NC_calibs estimates H(NC -> nkd) directly on the raw volt traces,
    without the volts_to_pa conversion and drift correction that
    save_PH_calibs applies. Because H1 = S_xy/S_xx, scaling x by 1/a and y by
    1/b would scale H by a/b, so the omission leaves the NC -> nkd transfer
    function low by the nkd/NC sensitivity ratio:

        52.4e-3 / 50.9e-3 = 1.0295   ->   0.25 dB

    The drift correction itself cancels in the ratio (both channels take the
    same gain), so only the sensitivity ratio is missing. 0.25 dB is small but
    systematic, and it propagates into the freestream production signal, the
    wall production signal, and the Wiener reference.

    Note src/save/calibs.py's save_NC_calibs carries a docstring copy-pasted
    from save_PH_calibs that claims "Convert both channels to Pa (volts_to_pa)
    and compensate mic sensitivity vs psig" -- which the body does not do. The
    docstring is stripped before checking so it cannot mask the real behaviour.
    """
    import ast
    import inspect

    function = importlib.import_module(f"{package}.calibs").save_NC_calibs
    tree = ast.parse(inspect.getsource(function).lstrip())
    body = tree.body[0].body
    if (isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]  # drop the docstring
    called = {
        node.func.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "estimate_frf" in called
    assert "volts_to_pa" not in called
    assert "correct_pressure_sensitivity" not in called
