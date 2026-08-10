"""Contract for the five Config dataclasses.

Every dataset case has its own Config (src/config_params.py for phase1, plus
src/save_<case>/config_params.py). They must all derive the same output paths
from ROOT_DIR and keep their per-condition tuples in lockstep, because every
stage indexes them positionally by condition.
"""

from __future__ import annotations

import importlib

import pytest

from src.config_params import Config as PhaseOneConfig

# (module path, expected ROOT_DIR, expected LABELS, expected SPACINGS)
CASE_CONFIGS = [
    ("src.save_bump.config_params", "data/bump1",
     ("0psig", "50psig", "100psig"), ("close", "far")),
    ("src.save_fence.config_params", "data/fence",
     ("0psig", "50psig", "100psig"), ("close", "far")),
    ("src.save_iso_re.config_params", "data/iso_re",
     ("0psig", "30psig", "50psig"), ("close",)),
    ("src.save_phase2.config_params", "data/phase2",
     ("0psig", "50psig", "100psig"), ("close",)),
]

ALL_CONFIG_MODULES = ["src.config_params"] + [m for m, *_ in CASE_CONFIGS]


def load(module: str):
    return importlib.import_module(module).Config()


# --------------------------------------------------------------------------
# Path derivation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_CONFIG_MODULES)
def test_derived_paths_compose_from_root_dir(module: str) -> None:
    """Every case must lay its products out identically under its own root --
    this is the layout the README documents and the plot modules assume."""
    cfg = load(module)
    root = cfg.ROOT_DIR.rstrip("/")

    assert cfg.RAW_CAL_BASE == f"{root}/raw_calib"
    assert cfg.RAW_BASE == f"{root}/raw_wallp"
    assert cfg.TF_BASE == f"{root}/calibration"
    assert cfg.PH_RAW_FILE == f"{root}/pressure/G_wallp_SU_raw.hdf5"
    assert cfg.PH_PROCESSED_FILE == f"{root}/pressure/G_wallp_SU_production.hdf5"
    assert cfg.NKD_RAW_FILE == f"{root}/pressure/F_freestreamp_SU_raw.hdf5"
    assert cfg.NKD_PROCESSED_FILE == f"{root}/pressure/F_freestreamp_SU_production.hdf5"


def test_phase1_root_dir_defaults_to_phase1(monkeypatch) -> None:
    monkeypatch.delenv("PRESSUREPROCESS_ROOT_DIR", raising=False)
    cfg = PhaseOneConfig()

    assert cfg.ROOT_DIR == "data/phase1"
    assert cfg.PH_RAW_FILE == "data/phase1/pressure/G_wallp_SU_raw.hdf5"


def test_phase1_root_dir_honours_the_environment(monkeypatch) -> None:
    """Only the phase1 Config is env-driven; the case configs are hard-coded."""
    monkeypatch.setenv("PRESSUREPROCESS_ROOT_DIR", "/tmp/custom/root/")
    cfg = PhaseOneConfig()

    assert cfg.ROOT_DIR == "/tmp/custom/root/"
    assert cfg.RAW_CAL_BASE == "/tmp/custom/root/raw_calib"
    assert cfg.PH_PROCESSED_FILE == "/tmp/custom/root/pressure/G_wallp_SU_production.hdf5"


@pytest.mark.parametrize("module,root,_labels,_spacings", CASE_CONFIGS)
def test_case_configs_ignore_the_root_dir_env_var(
    monkeypatch, module: str, root: str, _labels, _spacings
) -> None:
    """A case pipeline must always write into its own dataset root. If this
    ever started honouring PRESSUREPROCESS_ROOT_DIR, running two cases in one
    shell would silently cross-contaminate their outputs."""
    monkeypatch.setenv("PRESSUREPROCESS_ROOT_DIR", "/tmp/should-be-ignored")
    assert load(module).ROOT_DIR == root


# --------------------------------------------------------------------------
# Per-condition tuples
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_CONFIG_MODULES)
def test_per_condition_tuples_are_the_same_length(module: str) -> None:
    """Stages index LABELS/PSIGS/U_TAU/... positionally by condition, so a
    mismatched length silently pairs the wrong pressure with the wrong u_tau."""
    cfg = load(module)
    n = len(cfg.LABELS)

    for name in ("PSIGS", "U_TAU", "U_E", "ANALOG_LP_FILTER", "U_TAU_REL_UNC",
                 "DELTA", "TDEG", "F_CUTS"):
        assert len(getattr(cfg, name)) == n, f"{module}.{name} has length != {n}"


@pytest.mark.parametrize("module,_root,labels,spacings", CASE_CONFIGS)
def test_case_labels_and_spacings_are_pinned(
    module: str, _root: str, labels, spacings
) -> None:
    """iso_re runs (0, 30, 50) psig, not (0, 50, 100); phase2 and iso_re are
    close-only. These drive which raw files are looked for on disk."""
    cfg = load(module)
    assert cfg.LABELS == labels
    assert cfg.SPACINGS == spacings


@pytest.mark.parametrize("module", ALL_CONFIG_MODULES)
def test_labels_and_psigs_agree(module: str) -> None:
    """Calibration filenames are built from PSIGS (calibs_<int psig>.h5) while
    raw filenames are built from LABELS (<label>.mat), so they must describe
    the same conditions."""
    cfg = load(module)
    assert tuple(f"{int(p)}psig" for p in cfg.PSIGS) == tuple(cfg.LABELS)


@pytest.mark.parametrize("module", ALL_CONFIG_MODULES)
def test_sensitivities_cover_every_channel(module: str) -> None:
    cfg = load(module)
    assert set(cfg.SENSITIVITIES_V_PER_PA) >= {"PH1", "PH2", "NC", "nkd"}
    assert all(v > 0 for v in cfg.SENSITIVITIES_V_PER_PA.values())


@pytest.mark.parametrize("module", ALL_CONFIG_MODULES)
def test_spectral_settings_are_consistent_across_cases(module: str) -> None:
    """Welch parameters must match across cases or spectra are not comparable
    between datasets."""
    cfg = load(module)
    assert cfg.FS == 50_000.0
    assert cfg.NPERSEG == 2**12
    assert cfg.WINDOW == "hann"


# --------------------------------------------------------------------------
# Case-specific structure
# --------------------------------------------------------------------------

def test_bump_and_fence_carry_per_position_friction_velocities() -> None:
    """pw_proc writes a per-sensor-position u_tau attribute for these cases;
    the lookup is U_TAU_BY_POSITION[label][spacing][channel]."""
    for module in ("src.save_bump.config_params", "src.save_fence.config_params"):
        cfg = load(module)
        for label in cfg.LABELS:
            for spacing in cfg.SPACINGS:
                for channel in ("PH1", "PH2"):
                    value = cfg.U_TAU_BY_POSITION[label][spacing][channel]
                    assert 0.0 < float(value) < 5.0


def test_iso_re_pins_the_noise_canceller_and_skips_nc_calibs() -> None:
    """iso_re has no matching semi-anechoic NC calibration at (0, 30, 50) psig,
    so its NC -> nkd stage uses an identity FRF and no NC calib step runs."""
    cfg = load("src.save_iso_re.config_params")
    assert cfg.PW_NOISE_CANCELLER == "auto"
    assert not hasattr(cfg, "RUN_NC_CALIBS") or cfg.RUN_NC_CALIBS is False


def test_bump_enables_its_own_nc_calibration() -> None:
    cfg = load("src.save_bump.config_params")
    assert cfg.RUN_NC_CALIBS is True
    assert cfg.INCLUDE_NC_CALIB_RAW is True


def test_phase1_denoiser_env_var_is_honoured(monkeypatch) -> None:
    monkeypatch.setenv("PRESSUREPROCESS_PW_DENOISER", "HYBRID")
    assert PhaseOneConfig().PW_NOISE_CANCELLER == "hybrid"

    monkeypatch.delenv("PRESSUREPROCESS_PW_DENOISER")
    assert PhaseOneConfig().PW_NOISE_CANCELLER == "auto"
