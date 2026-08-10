"""End-to-end runs of two real case pipelines on synthetic data.

These exercise the documented reprocessing chain:

    phase2  ->  wiener kernel extraction  ->  bump1

which is the order the borrowed-source design forces (bump1 consumes phase2's
kernels and phase1's PH transfer function). Everything runs against synthetic
.mat inputs in a temp workspace, so no part of the 44 GB dataset is needed.

The pipelines build their paths relative to a hard-coded ROOT_DIR, so the
fixture changes directory into the workspace while running them.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import (
    FS,
    chdir,
    write_ph_calibration_h5,
    write_raw_case_root,
)

LABELS = ("0psig", "50psig", "100psig")
PSIGS = (0.0, 50.0, 100.0)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Path:
    """Build a synthetic workspace and run phase2 -> kernels -> bump1 once."""
    root = tmp_path_factory.mktemp("pressure_workspace")

    write_raw_case_root(root / "data" / "phase2", LABELS, ("close",), seed=1)
    write_raw_case_root(root / "data" / "bump1", LABELS, ("close", "far"), seed=2)
    # phase2 and bump1 both correct PH with phase1's fused TF.
    write_ph_calibration_h5(root / "data" / "phase1" / "calibration" / "PH", PSIGS)

    with chdir(root):
        from src.save_phase2 import (
            calibs as p2_calibs,
            fs_proc as p2_fs_proc,
            fs_raw as p2_fs_raw,
            pw_proc as p2_pw_proc,
            pw_raw as p2_pw_raw,
        )

        p2_calibs.save_PH_calibs()
        p2_calibs.save_NC_calibs()
        p2_fs_raw.save_raw_fs_pressure(spacings=("close",), include_nc_calib=True)
        p2_pw_raw.save_raw_ph_pressure(spacings=("close",))
        p2_fs_proc.save_prod_fs_pressure(spacings=("close",))
        p2_pw_proc.save_corrected_pressure(spacings=("close",))

        from src.core.wiener_kernel_donor import DonorSpec, extract_wiener_kernels

        extract_wiener_kernels(
            DonorSpec(
                ph_processed_file="data/phase2/pressure/G_wallp_SU_production.hdf5",
                fs_processed_file="data/phase2/pressure/F_freestreamp_SU_production.hdf5",
                labels=LABELS,
                spacings=("close",),
                fs_hz=FS,
            ),
            out_h5="data/phase2/calibration/wiener_kernels.h5",
            filter_order=256,
        )

        from src.save_bump import (
            calibs as b_calibs,
            fs_proc as b_fs_proc,
            fs_raw as b_fs_raw,
            pw_proc as b_pw_proc,
            pw_raw as b_pw_raw,
        )

        b_calibs.save_PH_calibs()
        b_calibs.save_NC_calibs()
        b_fs_raw.save_raw_fs_pressure(spacings=("close", "far"), include_nc_calib=True)
        b_pw_raw.save_raw_ph_pressure(spacings=("close", "far"))
        b_fs_proc.save_prod_fs_pressure(spacings=("close", "far"))
        b_pw_proc.save_corrected_pressure(spacings=("close", "far"))

    return root


# --------------------------------------------------------------------------
# Products land where the README says they do
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case,psigs", [("phase2", PSIGS), ("bump1", PSIGS)])
def test_every_documented_product_is_written(workspace: Path, case: str, psigs) -> None:
    root = workspace / "data" / case

    for name in ("G_wallp_SU_raw.hdf5", "G_wallp_SU_production.hdf5",
                 "F_freestreamp_SU_raw.hdf5", "F_freestreamp_SU_production.hdf5"):
        assert (root / "pressure" / name).is_file(), f"{case}: missing {name}"

    for psig in psigs:
        for sensor in ("PH", "NC"):
            path = root / "calibration" / sensor / f"calibs_{int(psig)}.h5"
            assert path.is_file(), f"{case}: missing {path}"


def test_calibration_files_carry_the_documented_datasets(workspace: Path) -> None:
    """pw_proc/fs_proc read 'frequencies' and 'H_fused'; if a calibs writer
    renames either, every downstream stage fails at read time."""
    for sensor in ("PH", "NC"):
        path = workspace / "data" / "phase2" / "calibration" / sensor / "calibs_0.h5"
        with h5py.File(path, "r") as hf:
            assert "frequencies" in hf
            assert "H_fused" in hf
            assert np.iscomplexobj(hf["H_fused"][:])
            assert hf["frequencies"].shape == hf["H_fused"].shape
            assert hf.attrs["fs_Hz"] == FS


# --------------------------------------------------------------------------
# The wall-pressure production tree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case,spacings", [
    ("phase2", ("close",)),
    ("bump1", ("close", "far")),
])
def test_wall_production_tree_has_both_processing_stages(
    workspace: Path, case: str, spacings
) -> None:
    """The production file must expose both stages for every condition:
    frf_corrected_signals (calibration applied) and fs_noise_rejected_signals
    (Wiener applied on top). Plot and upload code indexes exactly these names.
    """
    path = workspace / "data" / case / "pressure" / "G_wallp_SU_production.hdf5"

    with h5py.File(path, "r") as hf:
        assert set(hf["wallp_production"].keys()) == set(LABELS)

        for label in LABELS:
            group = hf[f"wallp_production/{label}"]
            for stage in ("frf_corrected_signals", "fs_noise_rejected_signals"):
                assert stage in group
                assert set(group[stage].keys()) == set(spacings)
                for spacing in spacings:
                    for channel in ("PH1_Pa", "PH2_Pa"):
                        dataset = group[f"{stage}/{spacing}/{channel}"]
                        assert dataset.dtype == np.float32
                        assert dataset.size > 0
                        assert np.isfinite(dataset[:]).all()


@pytest.mark.parametrize("case", ["phase2", "bump1"])
def test_condition_metadata_is_physically_consistent(workspace: Path, case: str) -> None:
    """Each condition group carries the thermodynamic state used to scale the
    spectra. Re_tau must equal u_tau*delta/nu, and density must rise with
    gauge pressure -- a units slip in air_props_from_gauge shows up here."""
    path = workspace / "data" / case / "pressure" / "G_wallp_SU_production.hdf5"

    with h5py.File(path, "r") as hf:
        densities = []
        for label in LABELS:
            attrs = hf[f"wallp_production/{label}"].attrs
            u_tau = float(attrs["u_tau"])
            delta = float(attrs["delta"])
            nu = float(attrs["nu"])
            rho = float(attrs["rho"])

            assert float(attrs["Re_tau"]) == pytest.approx(u_tau * delta / nu, rel=1e-9)
            assert nu == pytest.approx(float(attrs["mu"]) / rho, rel=1e-9)
            assert 1.0e-6 < nu < 2.0e-5
            assert float(attrs["T_K"]) > 273.15
            densities.append(rho)

        assert densities == sorted(densities), "rho must increase with gauge pressure"


@pytest.mark.parametrize("case", ["phase2", "bump1"])
def test_noise_rejection_reduces_variance_without_destroying_the_signal(
    workspace: Path, case: str
) -> None:
    """The synthetic wall signal is (known FIR * NC) + uncorrelated content, so
    a working rejection stage must cut variance substantially while leaving a
    strongly correlated remnant of the corrected signal."""
    path = workspace / "data" / case / "pressure" / "G_wallp_SU_production.hdf5"

    with h5py.File(path, "r") as hf:
        group = hf["wallp_production/0psig"]
        corrected = group["frf_corrected_signals/close/PH1_Pa"][:]
        rejected = group["fs_noise_rejected_signals/close/PH1_Pa"][:]

    assert rejected.var() < 0.8 * corrected.var()
    assert rejected.var() > 0.05 * corrected.var(), "rejection ate the whole signal"
    assert np.corrcoef(corrected, rejected)[0, 1] > 0.4


def test_bump_records_per_position_friction_velocity(workspace: Path) -> None:
    """Only bump/fence write u_tau and Re_tau per sensor position; phase1 and
    phase2 carry a single per-pressure value on the condition group."""
    path = workspace / "data" / "bump1" / "pressure" / "G_wallp_SU_production.hdf5"

    with h5py.File(path, "r") as hf:
        from src.save_bump.config_params import Config

        cfg = Config()
        for label in LABELS:
            for spacing in ("close", "far"):
                for channel in ("PH1", "PH2"):
                    attrs = hf[
                        f"wallp_production/{label}/frf_corrected_signals/"
                        f"{spacing}/{channel}_Pa"
                    ].attrs
                    expected = cfg.U_TAU_BY_POSITION[label][spacing][channel]
                    assert float(attrs["u_tau"]) == pytest.approx(expected)
                    assert float(attrs["Re_tau"]) > 0.0


def test_bump_file_is_tagged_with_its_case(workspace: Path) -> None:
    path = workspace / "data" / "bump1" / "pressure" / "G_wallp_SU_production.hdf5"
    with h5py.File(path, "r") as hf:
        assert hf.attrs["case"] == "bump"
        assert hf.attrs["fs_Hz"] == FS


# --------------------------------------------------------------------------
# The freestream product
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case,spacings", [
    ("phase2", ("close",)),
    ("bump1", ("close", "far")),
])
def test_freestream_production_tree(workspace: Path, case: str, spacings) -> None:
    """pw_proc reads its Wiener reference from this file, so its layout is a
    hard dependency of the wall-pressure stage."""
    path = workspace / "data" / case / "pressure" / "F_freestreamp_SU_production.hdf5"

    with h5py.File(path, "r") as hf:
        assert set(hf["freestream_production"].keys()) == set(LABELS)
        for label in LABELS:
            group = hf[f"freestream_production/{label}"]
            assert "FRF_NC_to_nkd" in group
            assert group["FRF_NC_to_nkd"].attrs["from"] == "NC"
            assert group["FRF_NC_to_nkd"].attrs["to"] == "nkd"
            for spacing in spacings:
                signal = group[f"{spacing}/NC_Pa"][:]
                assert signal.size > 0
                assert np.isfinite(signal).all()


def test_raw_wall_file_keeps_the_calibration_runs(workspace: Path) -> None:
    path = workspace / "data" / "phase2" / "pressure" / "G_wallp_SU_raw.hdf5"

    with h5py.File(path, "r") as hf:
        group = hf["wallp_raw/0psig"]
        assert "close/PH1_Pa" in group
        assert "close/PH2_Pa" in group
        assert group["FRF_PH_to_NC"].attrs["from"] == "PH"
        assert group["FRF_PH_to_NC"].attrs["to"] == "NC"
        assert "Run1/PH1_Pa" in group["FRF_PH_to_NC"]
        assert "Run2/PH2_Pa" in group["FRF_PH_to_NC"]


def test_volts_are_converted_to_pascals(workspace: Path) -> None:
    """The raw stage divides by SENSITIVITIES_V_PER_PA (~50 mV/Pa), so stored
    values must be ~20x the volt-level input. A missing conversion would leave
    the data 20x too small and every reported spectrum 400x low."""
    from conftest import SENSITIVITIES_V_PER_PA

    mat_path = workspace / "data" / "phase2" / "raw_wallp" / "close" / "0psig.mat"
    from scipy.io import loadmat

    volts = np.asarray(loadmat(mat_path)["channelData"])[:, 0]

    path = workspace / "data" / "phase2" / "pressure" / "G_wallp_SU_raw.hdf5"
    with h5py.File(path, "r") as hf:
        pascals = hf["wallp_raw/0psig/close/PH1_Pa"][:]

    # 0 psig => the sensitivity-drift correction is exactly unity.
    assert pascals.std() == pytest.approx(
        volts.std() / SENSITIVITIES_V_PER_PA["PH1"], rel=1e-5
    )


# --------------------------------------------------------------------------
# The donor Wiener kernels
# --------------------------------------------------------------------------

def test_donor_kernels_are_written_for_every_condition(workspace: Path) -> None:
    path = workspace / "data" / "phase2" / "calibration" / "wiener_kernels.h5"

    with h5py.File(path, "r") as hf:
        assert hf.attrs["filter_order"] == 256
        assert hf.attrs["fs_hz"] == FS
        for label in LABELS:
            for channel in ("PH1_Pa", "PH2_Pa"):
                kernel = hf[f"{label}/close/{channel}/c"][:]
                assert kernel.shape == (256,)
                assert np.isfinite(kernel).all()
                assert np.abs(kernel).max() > 0.0


def test_bump_far_spacing_reuses_the_close_kernel(workspace: Path) -> None:
    """phase2 is close-only, so bump's 'far' spacing falls back to the 'close'
    kernel. Both spacings must therefore be produced even though only one
    kernel exists."""
    path = workspace / "data" / "bump1" / "pressure" / "G_wallp_SU_production.hdf5"

    with h5py.File(path, "r") as hf:
        rejected = hf["wallp_production/0psig/fs_noise_rejected_signals"]
        assert set(rejected.keys()) == {"close", "far"}
        for spacing in ("close", "far"):
            values = rejected[f"{spacing}/PH1_Pa"][:]
            assert np.isfinite(values).all()
            assert values.std() > 0.0


def test_donor_kernel_actually_changed_the_bump_signal(workspace: Path) -> None:
    """Guards the stale-kernel failure mode: if the kernel were never applied,
    the rejected stage would be a byte-copy of the corrected stage."""
    path = workspace / "data" / "bump1" / "pressure" / "G_wallp_SU_production.hdf5"

    with h5py.File(path, "r") as hf:
        group = hf["wallp_production/50psig"]
        for spacing in ("close", "far"):
            corrected = group[f"frf_corrected_signals/{spacing}/PH1_Pa"][:]
            rejected = group[f"fs_noise_rejected_signals/{spacing}/PH1_Pa"][:]
            assert not np.allclose(corrected, rejected)
            assert rejected.var() < corrected.var()
