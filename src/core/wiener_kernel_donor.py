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
Reads the donor case's processed HDF5s, demeans + bandpass-filters the
PH and NC signals exactly the way pw_proc does pre-Wiener, runs Wiener
with return_kernel=True, and stores c[k] per (psig, spacing, channel)
into an output HDF5. Attributes record the FIR order, ridge, leak factor,
bandpass cutoffs, and source files so the recipient pipeline can verify
compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.core.wiener_filter_torch import wiener_cancel_background

WORK_DTYPE = np.float32


@dataclass(frozen=True)
class DonorSpec:
    """Where to read donor signals and what each label's analog-LP cutoff is."""
    ph_processed_file: str          # e.g. data/phase1/pressure/G_wallp_SU_production.hdf5
    fs_processed_file: str          # e.g. data/phase1/pressure/F_freestreamp_SU_production.hdf5
    labels: Iterable[str]            # e.g. ("0psig", "50psig", "100psig")
    analog_lp_hz: Iterable[float]    # per-label analog-LP cutoff used in bandpass
    spacings: Iterable[str]          # e.g. ("close", "far")
    channels: Iterable[str] = ("PH1_Pa", "PH2_Pa")
    fs_hz: float = 50_000.0


def _bandpass(x: np.ndarray, fs: float, f_high: float) -> np.ndarray:
    x = np.asarray(x, dtype=WORK_DTYPE)
    x = x - x.mean(dtype=WORK_DTYPE)
    sos = butter(3, [1.0, float(f_high)], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return np.ascontiguousarray(np.nan_to_num(y, nan=0.0, copy=False), dtype=WORK_DTYPE)


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
        attrs: filter_order, alpha, ridge_rel, fs_hz, bandpass_lo_hz, bandpass_hi_hz,
               donor_ph_file, donor_fs_file
    """
    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(donor.labels)
    analog_lp = list(donor.analog_lp_hz)
    spacings = list(donor.spacings)
    channels = list(donor.channels)
    if len(labels) != len(analog_lp):
        raise ValueError("labels and analog_lp_hz must have equal length")

    with h5py.File(donor.ph_processed_file, "r") as hf_w, \
         h5py.File(donor.fs_processed_file, "r") as hf_fs, \
         h5py.File(out_path, "w") as hf_out:
        hf_out.attrs["filter_order"] = int(filter_order)
        hf_out.attrs["alpha"] = float(alpha)
        hf_out.attrs["ridge_rel"] = float(ridge_rel)
        hf_out.attrs["fs_hz"] = float(donor.fs_hz)
        hf_out.attrs["bandpass_lo_hz"] = 1.0
        hf_out.attrs["donor_ph_file"] = str(donor.ph_processed_file)
        hf_out.attrs["donor_fs_file"] = str(donor.fs_processed_file)
        hf_out.attrs["note"] = (
            "Wiener FIR kernels c[k] trained on donor (label, spacing, channel). "
            "Apply via apply_wiener_kernel on bandpassed (1 Hz, analog_LP) "
            "frf-corrected PH and freestream-production NC."
        )

        g_w = hf_w["wallp_production"]
        g_fs = hf_fs["freestream_production"]

        for label, f_high in zip(labels, analog_lp):
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
            g_out_label.attrs["bandpass_hi_hz"] = float(f_high)

            for spacing in spacings:
                if spacing not in g_frf or spacing not in gL_fs:
                    print(f"[skip] {label}/{spacing}: missing in donor")
                    continue
                if "NC_Pa" not in gL_fs[spacing]:
                    print(f"[skip] {label}/{spacing}: missing NC_Pa")
                    continue

                nc_raw = gL_fs[f"{spacing}/NC_Pa"][:]
                pn = _bandpass(nc_raw, donor.fs_hz, f_high)

                g_out_sp = g_out_label.create_group(spacing)
                for channel in channels:
                    if channel not in g_frf[spacing]:
                        print(f"[skip] {label}/{spacing}/{channel}: missing")
                        continue
                    ph_raw = g_frf[f"{spacing}/{channel}"][:]
                    p0 = _bandpass(ph_raw, donor.fs_hz, f_high)

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
                    g_ch.attrs["bandpass_hi_hz"] = float(f_high)
                    print(f"[ok] kernel {label}/{spacing}/{channel} -> {out_path}")


if __name__ == "__main__":
    donor = DonorSpec(
        ph_processed_file="data/phase1/pressure/G_wallp_SU_production.hdf5",
        fs_processed_file="data/phase1/pressure/F_freestreamp_SU_production.hdf5",
        labels=("0psig", "50psig", "100psig"),
        analog_lp_hz=(2100.0, 4700.0, 14100.0),
        spacings=("close", "far"),
    )
    extract_wiener_kernels(donor, out_h5="data/phase1/calibration/wiener_kernels.h5")
