"""
Extract Wiener noise-cancellation kernels from a "donor" smooth-wall case
and save them per (psig, spacing, channel) so they can be transplanted into
bump/fence pipelines.

Rationale
---------
For a smooth wall, the freestream-mounted reference microphone (NC/nkd) is
dominated by facility noise; running an adaptive Wiener filter against it
correctly removes that facility noise from the wall-pressure signal and
leaves canonical TBL content behind. For bump/fence cases the protrusion
itself radiates as a dipole, so the freestream reference also carries
flow content we want to keep on the wall. Running Wiener locally on those
cases therefore eats real signal.

The transfer-function path the Wiener filter is identifying on the smooth
case is purely the facility-to-wall acoustic channel, which is (to first
order) independent of the test article in the working section. So we can
train Wiener on smooth, save the FIR kernel c[k], and apply that same
kernel to bump/fence frf-corrected PH and freestream-production NC.

What this module does
---------------------
Reads the donor case's processed HDF5s, demeans the PH and NC signals
exactly the way pw_proc does pre-Wiener (no digital band filtering -- band
limits are applied as spectral masks at reporting time), runs Wiener with
return_kernel=True, and stores c[k] per (psig, spacing, channel) into an
output HDF5. Attributes record the FIR order, ridge, leak factor, and
source files so the recipient pipeline can verify compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from src.core.wiener_filter_torch import wiener_cancel_background

WORK_DTYPE = np.float32


@dataclass(frozen=True)
class DonorSpec:
    """Where to read the donor signals."""
    ph_processed_file: str          # e.g. data/phase2/pressure/G_wallp_SU_production.hdf5
    fs_processed_file: str          # e.g. data/phase2/pressure/F_freestreamp_SU_production.hdf5
    labels: Iterable[str]            # e.g. ("0psig", "50psig", "100psig")
    spacings: Iterable[str]          # e.g. ("close",)
    channels: Iterable[str] = ("PH1_Pa", "PH2_Pa")
    fs_hz: float = 50_000.0


def _demean(x: np.ndarray) -> np.ndarray:
    # Matches the pw_proc pipelines' pre-Wiener conditioning exactly: demean
    # only. No digital band filtering anywhere; band limits are applied as
    # spectral masks at reporting time.
    x = np.asarray(x, dtype=WORK_DTYPE)
    return np.ascontiguousarray(x - x.mean(dtype=WORK_DTYPE))


def extract_wiener_kernels(
    donor: DonorSpec,
    out_h5: str,
    *,
    filter_order: int = 2**10,
    alpha: float = 1.0,
    ridge_rel: float = 1e-3,
) -> None:
    """
    For every (label, spacing, channel) tuple in `donor`, retrain Wiener on
    the donor's bandpassed PH (frf_corrected) and NC (freestream_production)
    and save the FIR kernel c[k] into `out_h5`.

    Output layout:
        /<label>/<spacing>/<channel>/c  (filter_order-long float array)
        attrs: filter_order, alpha, ridge_rel, fs_hz,
               donor_ph_file, donor_fs_file
    """
    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(donor.labels)
    spacings = list(donor.spacings)
    channels = list(donor.channels)

    with h5py.File(donor.ph_processed_file, "r") as hf_w, \
         h5py.File(donor.fs_processed_file, "r") as hf_fs, \
         h5py.File(out_path, "w") as hf_out:
        hf_out.attrs["filter_order"] = int(filter_order)
        hf_out.attrs["alpha"] = float(alpha)
        hf_out.attrs["ridge_rel"] = float(ridge_rel)
        hf_out.attrs["fs_hz"] = float(donor.fs_hz)
        hf_out.attrs["donor_ph_file"] = str(donor.ph_processed_file)
        hf_out.attrs["donor_fs_file"] = str(donor.fs_processed_file)
        hf_out.attrs["note"] = (
            "Wiener FIR kernels c[k] trained on donor (label, spacing, channel). "
            "Apply via apply_wiener_kernel on demeaned frf-corrected PH and "
            "freestream-production NC. No digital band filtering anywhere; "
            "band limits are applied as spectral masks at reporting time."
        )

        g_w = hf_w["wallp_production"]
        g_fs = hf_fs["freestream_production"]

        for label in labels:
            if label not in g_w or label not in g_fs:
                print(f"[skip] {label}: missing in donor groups")
                continue
            gL_w = g_w[label]
            gL_fs = g_fs[label]
            g_frf = gL_w.get("frf_corrected_signals")
            if g_frf is None:
                print(f"[skip] {label}: no frf_corrected_signals")
                continue

            g_out_label = hf_out.create_group(label)

            for spacing in spacings:
                if spacing not in g_frf or spacing not in gL_fs:
                    print(f"[skip] {label}/{spacing}: missing in donor")
                    continue
                if "NC_Pa" not in gL_fs[spacing]:
                    print(f"[skip] {label}/{spacing}: missing NC_Pa")
                    continue

                pn = _demean(gL_fs[f"{spacing}/NC_Pa"][:])

                g_out_sp = g_out_label.create_group(spacing)
                for channel in channels:
                    if channel not in g_frf[spacing]:
                        print(f"[skip] {label}/{spacing}/{channel}: missing")
                        continue
                    p0 = _demean(g_frf[f"{spacing}/{channel}"][:])

                    _, c = wiener_cancel_background(
                        p0, pn, donor.fs_hz,
                        filter_order=filter_order,
                        alpha=alpha,
                        ridge_rel=ridge_rel,
                        preserve_mean=False,
                        return_kernel=True,
                    )
                    g_ch = g_out_sp.create_group(channel)
                    g_ch.create_dataset("c", data=np.asarray(c, dtype=np.float32))
                    g_ch.attrs["filter_order"] = int(filter_order)
                    print(f"[ok] kernel {label}/{spacing}/{channel} -> {out_path}")


if __name__ == "__main__":
    # Canonical donor: the phase2 smooth-wall baseline (close-only). These are
    # the kernels transplanted into the bump/fence pipelines.
    donor = DonorSpec(
        ph_processed_file="data/phase2/pressure/G_wallp_SU_production.hdf5",
        fs_processed_file="data/phase2/pressure/F_freestreamp_SU_production.hdf5",
        labels=("0psig", "50psig", "100psig"),
        spacings=("close",),
    )
    extract_wiener_kernels(donor, out_h5="data/phase2/calibration/wiener_kernels.h5")
