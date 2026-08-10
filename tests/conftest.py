"""Shared fixtures.

The real dataset is ~44 GB and is not available on CI, so every test here
builds *synthetic* inputs with the same on-disk layout, MATLAB variable names
and channel ordering that the pipelines expect. Anything that cannot be
checked without the real data is asserted structurally (tree shape, dtypes,
finiteness) rather than numerically.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import numpy as np
import pytest
import torch
from scipy.io import savemat
from scipy.signal import lfilter

# Matches Config.FS / Config.NPERSEG across every case config.
FS = 50_000.0
NPERSEG = 2**12
# Long enough for Welch at NPERSEG=4096, short enough to stay fast.
N_SAMPLES = 1 << 15

# Matches Config.SENSITIVITIES_V_PER_PA in every case config.
SENSITIVITIES_V_PER_PA = {
    "PH1": 50.9e-3,
    "PH2": 51.7e-3,
    "NC": 52.4e-3,
    "nkd": 50.9e-3,
}

# The facility-noise path NC -> wall that the synthetic data is built with.
# The Wiener canceller is expected to identify (approximately) this FIR.
FACILITY_PATH_FIR = np.array([0.6, -0.3, 0.15, 0.05])

CPU = torch.device("cpu")

# Tests that read source text must not depend on the invoking cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _deterministic_torch():
    """Pin torch to single-threaded CPU so Wiener results are reproducible."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@contextmanager
def chdir(path: Path):
    """Case pipelines build paths relative to a hard-coded ROOT_DIR, so tests
    drive them by changing directory into a synthetic workspace."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def pink_noise(rng: np.random.Generator, n: int = N_SAMPLES) -> np.ndarray:
    """Cheap 1/f-ish noise, so synthetic spectra are not flat and the
    coherence-weighted machinery has something frequency-dependent to chew on."""
    return lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))


def write_raw_case_root(
    root: Path,
    labels,
    spacings,
    *,
    seed: int = 0,
    with_nc_calib: bool = True,
) -> None:
    """Write a synthetic raw dataset root in the layout the pipelines expect.

    Produces, under ``root``:
      raw_wallp/<spacing>/<label>.mat            channelData (N,3) = [PH1, PH2, NC]
      raw_calib/PH/calib_<label>_{1,2}.mat       channelData (N,4) = [PH1, PH2, NC, aux]
      raw_calib/NC/<label>/nkd-ns_nofacilitynoise.mat
                                                 channelData (N,2) = [nkd, NC]
                                                 (key is channelData_nofacitynoise
                                                  for the 100psig label only)

    All .mat payloads are in VOLTS; the pipelines divide by
    SENSITIVITIES_V_PER_PA to get Pa.
    """
    rng = np.random.default_rng(seed)
    (root / "raw_wallp").mkdir(parents=True, exist_ok=True)
    (root / "raw_calib" / "PH").mkdir(parents=True, exist_ok=True)

    for label in labels:
        for spacing in spacings:
            out_dir = root / "raw_wallp" / spacing
            out_dir.mkdir(parents=True, exist_ok=True)
            nc = pink_noise(rng)
            # Wall signal = facility noise through a known FIR + uncorrelated TBL.
            facility = lfilter(FACILITY_PATH_FIR, [1.0], nc)
            ph1 = facility + 0.4 * pink_noise(rng)
            ph2 = facility + 0.4 * pink_noise(rng)
            channel_data = np.column_stack([
                ph1 * SENSITIVITIES_V_PER_PA["PH1"],
                ph2 * SENSITIVITIES_V_PER_PA["PH2"],
                nc * SENSITIVITIES_V_PER_PA["NC"],
            ])
            savemat(out_dir / f"{label}.mat", {"channelData": channel_data})

        for run in (1, 2):
            source = pink_noise(rng)
            pinhole = lfilter([0.9, 0.1], [1.0], source)
            channel_data = np.column_stack([
                pinhole * SENSITIVITIES_V_PER_PA["PH1"],
                pinhole * SENSITIVITIES_V_PER_PA["PH2"],
                source * SENSITIVITIES_V_PER_PA["NC"],
                rng.standard_normal(N_SAMPLES),
            ])
            savemat(
                root / "raw_calib" / "PH" / f"calib_{label}_{run}.mat",
                {"channelData": channel_data},
            )

        if with_nc_calib:
            out_dir = root / "raw_calib" / "NC" / label
            out_dir.mkdir(parents=True, exist_ok=True)
            nc_cal = pink_noise(rng)
            nkd_cal = lfilter([0.8, 0.2], [1.0], nc_cal)
            channel_data = np.column_stack([
                nkd_cal * SENSITIVITIES_V_PER_PA["nkd"],
                nc_cal * SENSITIVITIES_V_PER_PA["NC"],
            ])
            # The loaders special-case the 100psig file's variable name.
            key = "channelData_nofacitynoise" if label == "100psig" else "channelData"
            savemat(out_dir / "nkd-ns_nofacilitynoise.mat", {key: channel_data})


def write_ph_calibration_h5(out_dir: Path, psigs, *, gain: complex = 1.0 + 0.0j) -> None:
    """Write donor PH calibration files with the datasets pw_proc reads.

    phase2/bump/fence all read ``frequencies`` and ``H_fused`` from
    ``data/phase1/calibration/PH/calibs_<psig>.h5``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(0.0, FS / 2.0, 2049)
    for psig in psigs:
        with h5py.File(out_dir / f"calibs_{int(psig)}.h5", "w") as hf:
            hf.create_dataset("frequencies", data=freqs)
            hf.create_dataset("H_fused", data=np.full(freqs.shape, gain, dtype=complex))
            hf.attrs["psig"] = float(psig)
