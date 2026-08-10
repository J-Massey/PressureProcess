"""
Repackage the SU pressure production products into the framework upload format.

Reads (never writes) ``data/<case>/`` and emits self-contained files under
``report/data/SU/`` (the report directory contains everything: tex, figures,
plotting code and the upload data). The framework SU table (tab:dataSU)
defines products A-F;
A and D are wall SHEAR STRESS (a separate deliverable). This script builds the
PRESSURE subset -- B/C (fence, B1) and E/F (bump, B2) -- each in Raw and
Production form::

    fence  -> B1     B_B1_wallp_SU_raw.hdf5        B_B1_wallp_SU_production.hdf5
                     C_B1_freestreamp_SU_raw.hdf5  C_B1_freestreamp_SU_production.hdf5
    bump1  -> B2     E_B2_wallp_SU_raw.hdf5        E_B2_wallp_SU_production.hdf5
                     F_B2_freestreamp_SU_raw.hdf5  F_B2_freestreamp_SU_production.hdf5

The smooth-wall baseline (phase2) has NO slot in tab:dataSU: it appears only as
the drift-verification figure, so it is not built by default (see BASELINE_CASE).

Raw products carry the uncalibrated pressure in Pa plus the raw semi-anechoic
calibration recordings; the applied FRFs, Wiener kernels and processed signals
are in the companion production file.

The framework layout is::

    /                       root attributes (the 7 required categories)
    /Calibration data/ReN/  transfer functions + Wiener kernels actually applied
    /Observations/ReN/      the pressure time series

Sample arrays are copied bit-for-bit. Only metadata is rebuilt:

* Group-level ``u_tau``/``Re_tau`` in the source are the *smooth-wall* triple
  (0.537, 0.522, 0.506) even for the rough cases, while the per-dataset attrs
  hold the *measured* per-station values. Here the measured values are promoted
  to the group as ``u_tau_stations``/``Re_tau_stations`` and the smooth-wall
  number is kept only under an explicitly-named reference attribute.
* Facility and resolution-constraint attributes (absent from every source file)
  are filled in.
* Geometry that is genuinely unset in config is written as 0.0 alongside a
  ``*_status`` attribute saying so, never as a bare number.

Run: ``python -m src.upload.build_framework_products``
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.band_limits import F_HELMHOLTZ_GUARD_HZ, f_cut_hz
from src.save_bump.config_params import Config as BumpConfig
from src.save_fence.config_params import Config as FenceConfig
from src.save_phase2.config_params import Config as Phase2Config

# Upload products live inside the report directory so the report is fully
# self-contained (tex + figures + plotting code + the uploaded data).
OUT_ROOT = Path("report/data/SU")

FACILITY = "Stanford University High Pressure Wind Tunnel (HPWT)"
FACILITY_CODE = "SU"

# Framework: "Reynolds number #1/#2/#3". Ordered by increasing Re_tau.
REYNOLDS_INDEX = {"0psig": "Re1", "50psig": "Re2", "100psig": "Re3"}

# The dual-sensor holder is repositioned between runs; P1..P4 are holder-position
# labels, NOT a streamwise ordering (see STATION_ORDERING_NOTE).
STATIONS = {
    ("close", "PH1"): "P1",
    ("close", "PH2"): "P2",
    ("far", "PH1"): "P3",
    ("far", "PH2"): "P4",
}
# Per-case station order is derived from each case's cfg.SPACINGS in build_wallp
# (B0 baseline is close-only -> P1/P2; B1/B2 -> P1..P4).

PINHOLE_DIAMETER_M = 70e-6  # data/phase1/raw_calib/readme.txt

STATION_ORDERING_NOTE = (
    "P1..P4 are dual-sensor-holder position labels, not a streamwise ordering. "
    "The authoritative streamwise coordinate is the per-dataset attribute 'x_m'. "
    "Sorted by x_m the order is P4 (0.015 m), P1 (0.022 m), P2 (0.120 m), "
    "P3 (0.127 m). Sort by x_m, not by station_id."
)

X_POSITION_NOTE = (
    "Streamwise positions are derived formulaically from the nominal smooth-wall "
    "boundary-layer thickness delta = 0.035 m (close: x_PH1 = 15 mm + 0.2*delta, "
    "spacing 2.8*delta; far: x_PH2 = 15 mm, spacing 3.2*delta). They are not "
    "independently surveyed positions."
)

BASELINE_STATION_ORDERING_NOTE = (
    "P1 and P2 are the two dual-sensor-holder positions at the single (close) "
    "spacing of this baseline; there is no 'far' station. The authoritative "
    "streamwise coordinate is the per-dataset attribute 'x_m' (P1 = 0.022 m, "
    "P2 = 0.120 m). Sort by x_m, not by station_id."
)

BASELINE_X_POSITION_NOTE = (
    "Streamwise positions are derived formulaically from the nominal smooth-wall "
    "boundary-layer thickness delta = 0.035 m (close: x_PH1 = 15 mm + 0.2*delta, "
    "spacing 2.8*delta). Close spacing only. They are not independently surveyed "
    "positions."
)

USE_NOTES = (
    "Wall-pressure signals are provided in two forms per station: "
    "'frf_corrected_signals' (FRF-corrected only) and 'fs_noise_rejected_signals' "
    "(FRF-corrected, mean-removed, then facility-noise rejected; no digital "
    "band filtering is applied to the signals). "
    "The rejection is NOT adaptive against this case's freestream "
    "microphone: a fixed Wiener FIR rejection kernel, identified once on the "
    "smooth-wall baseline (where the nose-cone reference carries facility noise "
    "only), filters the simultaneously recorded nose-cone signal to form the "
    "facility-noise estimate that is subtracted. Use the noise-rejected form "
    "for spectra below the convective range; use the FRF-corrected form to "
    "apply your own noise rejection. Report only frequencies at or below the "
    "per-station 'f_cut_Hz' band edge. Inner-normalise with the per-station "
    "u_tau (dataset attribute), never with the group-level smooth-wall reference."
)

F_CUT_NOTE = (
    "Upper edge of the reported wall-pressure band. Per station, f_cut_Hz = "
    f"min(pinhole Helmholtz-resonance guard = {F_HELMHOLTZ_GUARD_HZ:.0f} Hz, "
    "spatial-resolution limit u_tau^2/(nu*T_plus_cut) with T_plus_cut = 10), "
    "evaluated with the measured per-station u_tau; each pressure dataset "
    "carries its own 'f_cut_Hz' attribute, and this group attribute is the "
    "minimum over the condition's stations. The pinhole cavity resonance sits "
    "at ~1.4 kHz at every pressurisation (set by cavity geometry and sound "
    "speed, it does not scale with Re_tau), so the resonance guard binds at "
    "almost every station. Wall-pressure content above f_cut_Hz is "
    "resonance-contaminated or spatially unresolved and must not be used. "
    "Distinct from 'analog_LP_filter_Hz', the analog anti-alias filter."
)


@dataclass(frozen=True)
class CaseSpec:
    source: str          # directory under data/
    framework_id: str    # B1 / B2
    description: str
    wallp_letter: str
    freestream_letter: str
    cfg: object


def _git_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{head}+dirty" if dirty else head
    except Exception:
        return "unknown"


def _geometry(spec: CaseSpec) -> dict[str, object]:
    """Case geometry, with an explicit status string for every unset value."""
    c = spec.cfg
    if spec.framework_id == "B0":
        return {
            "surface": "smooth wall",
            "geometry_status": (
                "Smooth-wall baseline: no protrusion, no roughness. This case exists "
                "to verify the smooth-wall reference does not drift between campaigns. "
                "No fence/bump geometry applies."
            ),
        }
    if spec.framework_id == "B1":
        return {
            "fence_height_m": float(c.FENCE_HEIGHT),
            "fence_thickness_m": float(c.FENCE_THICKNESS),
            "x_fence_m": float(c.X_FENCE),
            "geometry_status": (
                "UNSET. fence_height_m, fence_thickness_m and x_fence_m are "
                "placeholder zeros in src/save_fence/config_params.py and have "
                "never been populated. Do NOT interpret them as measured values."
            ),
        }
    return {
        "bump_height_m": float(c.BUMP_HEIGHT),
        "k_s_m": float(c.K_S),
        "geometry_status": (
            "bump_height_m = 20.01 mm is the nominal Eaton-bump height from "
            "src/save_bump/config_params.py. NOTE: the source production file "
            "data/bump1/pressure/G_wallp_SU_production.hdf5 carries a stale "
            "bump_height_m = 0.0, written before the config was updated; the "
            "config value is used here. k_s_m = 0.0 is UNCONFIRMED: it is either "
            "correct (a smooth 2-D bump contour) or an unfilled placeholder. "
            "k_s+ is therefore not reported."
        ),
    }


def _root_attrs(spec: CaseSpec, kind: str, src_path: Path, src: h5py.File,
                n_samples: int, datatype: str = "production") -> dict:
    """The seven framework attribute categories, at file root."""
    fs_hz = float(src.attrs["fs_Hz"])
    is_wall = kind == "wallp"
    is_baseline = spec.framework_id == "B0"
    is_raw = datatype == "raw"
    a: dict[str, object] = {
        # --- Facility ---
        "facility": FACILITY,
        "facility_code": FACILITY_CODE,

        # --- Run conditions ---
        "framework_id": spec.framework_id,
        "case_description": spec.description,
        "n_reynolds_conditions": len(REYNOLDS_INDEX),
        "reynolds_index_map": "Re1 = 0 psig, Re2 = 50 psig, Re3 = 100 psig",
        "run_conditions_note": "Per-condition run state is on each 'Observations/ReN' group.",
        "Ue_m_per_s": np.asarray(src.attrs["Ue_m_per_s"], dtype=float),

        # --- Diagnostic / sensor ---
        "diagnostic": (
            "Pinhole-mounted condenser microphone, wall pressure"
            if is_wall else
            "Nose-cone condenser microphone, freestream pressure"
        ),
        "sensor_model": str(src.attrs["mic_details"]),
        "DAQ": str(src.attrs["DAQ"]),
        "adc_bits": 24,

        # --- Sampling parameters ---
        "sampling_rate_Hz": fs_hz,
        "record_length_samples": int(n_samples),
        "record_duration_s": float(n_samples) / fs_hz,

        # --- Processing parameters ---
        "datatype": datatype,
        "processing_chain": (
            "volts -> Pa (sensitivity, psig-compensated). No FRF correction or "
            "noise rejection applied; this is the uncalibrated wall/freestream "
            "pressure. The applied FRFs and Wiener kernels, and the fully processed "
            "signals, are in the companion '..._production.hdf5' file."
            if is_raw else
            ("volts -> Pa (sensitivity, psig-compensated) -> PH->NC FRF -> NC->nkd FRF "
             "-> mean removal -> Wiener facility-noise rejection (fixed FIR kernel "
             "identified on the smooth-wall baseline, applied to the simultaneous NC "
             "recording). No digital band filtering is applied to the signals; the "
             "analog anti-alias acts upstream in hardware and the reported band is a "
             "spectral mask at f_cut_Hz"
             if is_wall else
             "volts -> Pa (sensitivity, psig-compensated) -> NC->nkd FRF")
        ),
        "frf_ph_to_nc_source": (
            "n/a for raw. The raw semi-anechoic PH<->NC calibration recordings are "
            "under 'Calibration data/ReN/FRF_PH_to_NC_recordings'; the FRF actually "
            "applied is in the companion production file."
            if (is_raw and is_wall) else
            "data/phase1/calibration/PH (borrowed: the phase-1 pinhole calibration "
            "is used for this case; its own PH calibration is 15-20 dB low in the ROI)"
            if is_wall else "n/a"
        ),
        "wiener_kernel_source": (
            "n/a for raw (no noise rejection applied). See the companion production file."
            if (is_raw and is_wall) else
            ("data/phase2/calibration/wiener_kernels.h5 (native kernels trained on "
             "this smooth-wall case; close spacing only. This case is the donor whose "
             "kernels B1/B2 borrow.)"
             if is_baseline else
             "data/phase2/calibration/wiener_kernels.h5 (borrowed donor kernels; "
             "phase2 is close-spacing only, so the 'far' spacing reuses the 'close' "
             "kernel, which is approximately spacing-independent)")
            if is_wall else "n/a"
        ),

        # --- Resolution constraints ---
        "pinhole_diameter_m": PINHOLE_DIAMETER_M if is_wall else 0.0,
        "resolution_note": (
            f"Pinhole orifice diameter d = {PINHOLE_DIAMETER_M*1e6:.0f} um. Viscous-scaled "
            "resolution d+ = d*u_tau/nu. For rough cases the local d+ depends on the "
            "measured per-station u_tau; see 'd_plus_stations' in the companion "
            "production file. Attenuation of the wall-pressure spectrum becomes "
            "significant for d+ > ~20. The analog anti-alias lowpass "
            "('analog_LP_filter_Hz') truncates the high-frequency tail and differs "
            "per Reynolds condition."
            if (is_raw and is_wall) else
            f"Pinhole orifice diameter d = {PINHOLE_DIAMETER_M*1e6:.0f} um. Viscous-scaled "
            "resolution d+ = d*u_tau/nu is reported per station as 'd_plus' on each "
            "dataset and as 'd_plus_stations' on each Reynolds group. Attenuation of "
            "the wall-pressure spectrum becomes significant for d+ > ~20. The analog "
            "anti-alias lowpass ('analog_LP_filter_Hz') truncates the high-frequency "
            "tail and differs per Reynolds condition. The reported band is limited "
            "above by the per-station 'f_cut_Hz' (see 'f_cut_note' on each "
            "Reynolds group)."
            if is_wall else
            "Not applicable: the nose-cone microphone is not pinhole-mounted."
        ),

        # --- Additional use notes ---
        "use_notes": USE_NOTES if is_wall else
            "Freestream nose-cone pressure. On the smooth-wall baseline this "
            "channel is the reference from which the Wiener facility-noise "
            "rejection kernel is identified; on B1/B2 the fixed smooth-wall "
            "kernel filters this simultaneous recording to form the noise "
            "estimate subtracted from the wall pressure in the companion wallp "
            "product. On B1/B2 it also carries protrusion-radiated flow content "
            "(dipole/quadrupole), which is why the rejection is deliberately "
            "not trained adaptively on this case's own records.",
        "station_ordering_note": (
            (BASELINE_STATION_ORDERING_NOTE if is_baseline else STATION_ORDERING_NOTE)
            if is_wall else "n/a"
        ),
        "x_position_note": (
            (BASELINE_X_POSITION_NOTE if is_baseline else X_POSITION_NOTE)
            if is_wall else "n/a"
        ),
        "known_issues": (
            ("Smooth-wall baseline: the group-level u_tau/Re_tau (0.537/0.522/0.506 m/s) "
             "ARE the physical friction values here, because a smooth wall has no "
             "station-to-station variation. 'u_tau_stations' therefore equals "
             "'u_tau_reference_smooth_wall' by construction. Close spacing only -- there "
             "is no 'far' station (P3/P4)."
             if is_baseline else
             "Group-level u_tau/Re_tau in the SOURCE pipeline files are the smooth-wall "
             "phase-1 triple (0.537, 0.522, 0.506 m/s) for every case. They are NOT the "
             "local friction velocities of this rough case. In this file they appear only "
             "as 'u_tau_reference_smooth_wall'; the measured per-station values are in "
             "'u_tau_stations' and on each dataset.")
            if is_wall else
            "Group-level u_tau in the source pipeline is the smooth-wall phase-1 value "
            "and is retained here only as 'u_tau_reference_smooth_wall'."
        ),

        # --- Provenance ---
        "title": f"{spec.framework_id} {'wall' if is_wall else 'freestream'} pressure "
                 f"({spec.description}) - {datatype}",
        "description": str(src.attrs["description"]),
        "source_case": spec.source,
        "source_file": str(src_path),
        "generated_by": "src/upload/build_framework_products.py",
        "source_commit": _git_commit(),
    }
    if is_wall:
        a.update(_geometry(spec))
    return a


def _condition_attrs(src_grp: h5py.Group, spec: CaseSpec, idx: int,
                     phys: dict | None = None) -> dict:
    """Run-condition attrs, minus the misleading smooth-wall u_tau scalar.

    ``phys`` overrides mu/nu/rho: the raw pipeline files leave mu and nu unset
    (0.0), so raw products borrow the physical properties from the matching
    production condition."""
    c = spec.cfg
    keep = ("psig", "T_K", "rho", "mu", "nu", "Ue_m_per_s", "u_tau_rel_unc",
            "analog_LP_filter_Hz")
    out = {k: src_grp.attrs[k] for k in keep if k in src_grp.attrs}
    if phys is not None:
        out["mu"] = phys["mu"]
        out["nu"] = phys["nu"]
        out["rho"] = phys["rho"]
    out["reynolds_index"] = REYNOLDS_INDEX[list(REYNOLDS_INDEX)[idx]]
    out["p_abs_Pa"] = c.P_ATM + float(src_grp.attrs["psig"]) * c.PSI_TO_PA
    out["delta_inflow_m"] = float(src_grp.attrs["delta"])
    out["u_tau_reference_smooth_wall"] = float(src_grp.attrs["u_tau"])
    out["Re_tau_reference_smooth_wall"] = float(src_grp.attrs["Re_tau"])
    out["reference_u_tau_note"] = (
        "Smooth-wall friction velocity for this baseline case. It IS the physical "
        "u_tau at every station (a smooth wall has no station-to-station variation), "
        "so it equals u_tau_stations here."
        if spec.framework_id == "B0" else
        "Smooth-wall inflow value inherited from the phase-1 configuration. It is "
        "NOT the local friction velocity at any station of this rough case. Use "
        "u_tau_stations / the per-dataset u_tau attribute instead."
    )
    out["units"] = np.asarray([
        "psig: psi(g)", "u_tau: m/s", "nu: m^2/s", "rho: kg/m^3", "mu: Pa*s",
        "T_K: K", "analog_LP_filter_Hz: Hz", "p_abs_Pa: Pa", "x_m: m",
        "pressure datasets: Pa",
    ], dtype=h5py.special_dtype(vlen=str))
    return out


def _copy_series(src: h5py.Dataset, dst_grp: h5py.Group, name: str, extra: dict) -> h5py.Dataset:
    """Bit-exact chunked copy of a long 1-D record, preserving source attrs."""
    d = dst_grp.create_dataset(name, shape=src.shape, dtype=src.dtype)
    step = 2_000_000
    for i in range(0, src.shape[0], step):
        d[i:i + step] = src[i:i + step]
    for k, v in src.attrs.items():
        d.attrs[k] = v
    for k, v in extra.items():
        d.attrs[k] = v
    return d


def _station_u_tau(case: str, label: str, framework_id: str) -> dict[tuple[str, str], float]:
    """Measured per-station u_tau, read from the production file (the raw files
    carry no calibrated metadata of their own). B0 smooth wall: the group-level
    u_tau is physical and identical at every station."""
    out: dict[tuple[str, str], float] = {}
    with h5py.File(f"data/{case}/pressure/G_wallp_SU_production.hdf5", "r") as f:
        g = f[f"wallp_production/{label}"]
        for sp in g["frf_corrected_signals"]:
            for ch in ("PH1", "PH2"):
                ds_path = f"frf_corrected_signals/{sp}/{ch}_Pa"
                if ds_path not in g:
                    continue
                if framework_id == "B0":
                    out[(sp, ch)] = float(g.attrs["u_tau"])
                else:
                    out[(sp, ch)] = float(g[ds_path].attrs["u_tau"])
    return out


def _spacing_meta(case: str, spacing: str) -> dict:
    """x_PH1 / x_PH2 / spacing_m. fence's production stage drops these; the raw
    file still has them, so read from raw for both cases (identical geometry)."""
    with h5py.File(f"data/{case}/pressure/G_wallp_SU_raw.hdf5", "r") as f:
        return dict(f[f"wallp_raw/0psig/{spacing}"].attrs)


def build_wallp(spec: CaseSpec) -> Path:
    src_path = Path(f"data/{spec.source}/pressure/G_wallp_SU_production.hdf5")
    out = OUT_ROOT / f"{spec.wallp_letter}_{spec.framework_id}_wallp_SU_production.hdf5"
    labels = list(REYNOLDS_INDEX)
    spacings = spec.cfg.SPACINGS
    geom = {sp: _spacing_meta(spec.source, sp) for sp in spacings}
    order = [STATIONS[(sp, ch)] for sp in spacings for ch in ("PH1", "PH2")]
    kernels = Path("data/phase2/calibration/wiener_kernels.h5")

    with h5py.File(src_path, "r") as src, h5py.File(out, "w") as dst:
        root = src["wallp_production"]
        n = root[f"{labels[0]}/frf_corrected_signals/close/PH1_Pa"].shape[0]
        for k, v in _root_attrs(spec, "wallp", src_path, src, n).items():
            dst.attrs[k] = v

        g_cal = dst.create_group("Calibration data")
        g_obs = dst.create_group("Observations")

        for idx, label in enumerate(labels):
            re_id = REYNOLDS_INDEX[label]
            s_cond = root[label]
            nu = float(s_cond.attrs["nu"])

            # ---- Observations ----
            d_cond = g_obs.create_group(re_id)
            for k, v in _condition_attrs(s_cond, spec, idx).items():
                d_cond.attrs[k] = v

            u_tau_st: dict[str, float] = {}
            re_tau_st: dict[str, float] = {}
            d_plus_st: dict[str, float] = {}
            x_st: dict[str, float] = {}
            f_cut_st: dict[str, float] = {}

            for stage in ("frf_corrected_signals", "fs_noise_rejected_signals"):
                d_stage = d_cond.create_group(stage)
                for spacing in spacings:
                    s_sp = s_cond[f"{stage}/{spacing}"]
                    d_sp = d_stage.create_group(spacing)
                    for k, v in geom[spacing].items():   # restore fence's dropped attrs
                        d_sp.attrs[k] = v
                    for ch in ("PH1", "PH2"):
                        s_d = s_sp[f"{ch}_Pa"]
                        station = STATIONS[(spacing, ch)]
                        x_m = float(geom[spacing][f"x_{ch}"])
                        # Smooth-wall baseline (B0) has no per-dataset u_tau/Re_tau;
                        # the smooth-wall value on the condition group is physical and
                        # identical at every station.
                        if spec.framework_id == "B0":
                            u_tau = float(s_cond.attrs["u_tau"])
                            re_tau = float(s_cond.attrs["Re_tau"])
                        else:
                            u_tau = float(s_d.attrs["u_tau"])
                            re_tau = float(s_d.attrs["Re_tau"])
                        d_plus = PINHOLE_DIAMETER_M * u_tau / nu
                        f_cut = f_cut_hz(u_tau, nu, tplus_cut=spec.cfg.TPLUS_CUT)
                        extra = {
                            "station_id": station,
                            "x_m": x_m,
                            "d_plus": d_plus,
                            "f_cut_Hz": f_cut,
                            "units": "Pa",
                        }
                        if spec.framework_id == "B2":
                            dl = float(spec.cfg.DELTA_BY_POSITION[label][spacing][ch])
                            extra["delta_local_m"] = dl
                            extra["Re_tau_local"] = u_tau * dl / nu
                            extra["Re_tau_note"] = (
                                "'Re_tau' is the value written by the pipeline, formed with "
                                "the nominal inflow delta = 0.035 m. 'Re_tau_local' uses the "
                                "measured local boundary-layer thickness 'delta_local_m' and "
                                "is the physically appropriate one for this station."
                            )
                        _copy_series(s_d, d_sp, f"{ch}_Pa", extra)

                        if stage == "frf_corrected_signals":
                            u_tau_st[station] = u_tau
                            re_tau_st[station] = re_tau
                            d_plus_st[station] = d_plus
                            x_st[station] = x_m
                            f_cut_st[station] = f_cut

            # promote the measured per-station values to the condition group
            d_cond.attrs["station_ids"] = np.asarray(
                order, dtype=h5py.special_dtype(vlen=str))
            d_cond.attrs["u_tau_stations"] = np.asarray(
                [u_tau_st[s] for s in order], dtype=float)
            d_cond.attrs["Re_tau_stations"] = np.asarray(
                [re_tau_st[s] for s in order], dtype=float)
            d_cond.attrs["d_plus_stations"] = np.asarray(
                [d_plus_st[s] for s in order], dtype=float)
            d_cond.attrs["station_x_m"] = np.asarray(
                [x_st[s] for s in order], dtype=float)
            d_cond.attrs["f_cut_Hz_stations"] = np.asarray(
                [f_cut_st[s] for s in order], dtype=float)
            d_cond.attrs["f_cut_Hz"] = float(min(f_cut_st.values()))
            d_cond.attrs["f_cut_note"] = F_CUT_NOTE
            if spec.framework_id == "B1":
                u_tau_src = (
                    "Measured, from data/fence/A_B1_shear_SU_production.hdf5 "
                    "(/Production/<label>/<P>/u_tau)."
                )
            elif spec.framework_id == "B2":
                u_tau_src = "Measured, from src/save_bump/config_params.py:U_TAU_BY_POSITION."
            else:  # B0 smooth-wall baseline
                u_tau_src = (
                    "Smooth-wall reference triple from src/save_phase2/config_params.py:"
                    "U_TAU. Physical for a smooth wall and identical at every station; "
                    "equals the group-level u_tau."
                )
            d_cond.attrs["u_tau_stations_source"] = u_tau_src

            # ---- Calibration data ----
            c_cond = g_cal.create_group(re_id)
            c_cond.attrs["reynolds_index"] = re_id
            c_cond.attrs["psig"] = float(s_cond.attrs["psig"])

            with h5py.File(f"data/phase1/calibration/PH/calibs_{int(s_cond.attrs['psig'])}.h5", "r") as f:
                g = c_cond.create_group("FRF_PH_to_NC")
                for k, v in f.attrs.items():
                    g.attrs[k] = v
                for k in f:
                    g.create_dataset(k, data=f[k][()])
                g.attrs["source_file"] = f"data/phase1/calibration/PH/calibs_{int(s_cond.attrs['psig'])}.h5"
                g.attrs["borrowed"] = (
                    "Borrowed from phase1. This case's own PH calibration is "
                    "15-20 dB low in the region of interest and is not used."
                )
                g.attrs["applied"] = "H_fused, applied identically to PH1 and PH2."

            nc = Path(f"data/{spec.source}/calibration/NC/calibs_{int(s_cond.attrs['psig'])}.h5")
            with h5py.File(nc, "r") as f:
                g = c_cond.create_group("FRF_NC_to_nkd")
                for k, v in f.attrs.items():
                    g.attrs[k] = v
                for k in f:
                    g.create_dataset(k, data=f[k][()])
                g.attrs["source_file"] = str(nc)

            if kernels.exists():
                with h5py.File(kernels, "r") as f:
                    g = c_cond.create_group("wiener_kernel")
                    for k, v in f.attrs.items():
                        g.attrs[k] = v
                    g.attrs["source_file"] = str(kernels)
                    g.attrs["note"] = (
                        "Native FIR kernels c[k] trained on this smooth-wall case "
                        "(close spacing only). This case is the donor whose kernels "
                        "B1/B2 borrow."
                        if spec.framework_id == "B0" else
                        "Donor FIR kernels c[k] trained on phase2. phase2 is close-spacing "
                        "only; the 'far' spacing of this case reuses the 'close' kernel."
                    )
                    for ch in ("PH1", "PH2"):
                        p = f"{label}/close/{ch}_Pa/c"
                        if p in f:
                            for spacing in spacings:   # far reuses the close kernel
                                g.create_dataset(f"{spacing}/{ch}_Pa/c", data=f[p][()])
    return out


def build_freestream(spec: CaseSpec) -> Path:
    src_path = Path(f"data/{spec.source}/pressure/F_freestreamp_SU_production.hdf5")
    out = OUT_ROOT / f"{spec.freestream_letter}_{spec.framework_id}_freestreamp_SU_production.hdf5"
    labels = list(REYNOLDS_INDEX)

    with h5py.File(src_path, "r") as src, h5py.File(out, "w") as dst:
        root = src["freestream_production"]
        n = root[f"{labels[0]}/close/NC_Pa"].shape[0]
        for k, v in _root_attrs(spec, "freestreamp", src_path, src, n).items():
            dst.attrs[k] = v

        g_cal = dst.create_group("Calibration data")
        g_obs = dst.create_group("Observations")

        for idx, label in enumerate(labels):
            re_id = REYNOLDS_INDEX[label]
            s_cond = root[label]

            d_cond = g_obs.create_group(re_id)
            for k, v in _condition_attrs(s_cond, spec, idx).items():
                d_cond.attrs[k] = v
            for spacing in ("close", "far"):
                if spacing not in s_cond:
                    continue
                d_sp = d_cond.create_group(spacing)
                _copy_series(s_cond[f"{spacing}/NC_Pa"], d_sp, "NC_Pa",
                             {"units": "Pa", "sensor": "NC (nose-cone reference microphone)"})

            c_cond = g_cal.create_group(re_id)
            c_cond.attrs["reynolds_index"] = re_id
            c_cond.attrs["psig"] = float(s_cond.attrs["psig"])
            if "FRF_NC_to_nkd" in s_cond:
                g = c_cond.create_group("FRF_NC_to_nkd")
                s_frf = s_cond["FRF_NC_to_nkd"]
                for k, v in s_frf.attrs.items():
                    g.attrs[k] = v
                for k in s_frf:
                    g.create_dataset(k, data=s_frf[k][()])
    return out


def _phys(case: str, label: str) -> dict:
    """Physical properties (mu, nu, rho) for a condition, read from the production
    file. The raw pipeline files leave mu and nu unset (0.0), so raw products
    borrow them from the matching production condition (identical run state)."""
    with h5py.File(f"data/{case}/pressure/G_wallp_SU_production.hdf5", "r") as f:
        a = f[f"wallp_production/{label}"].attrs
        return {k: float(a[k]) for k in ("mu", "nu", "rho")}


def build_wallp_raw(spec: CaseSpec) -> Path:
    """Repackage the raw (uncalibrated, Pa) wall-pressure signals plus the raw
    semi-anechoic PH<->NC calibration recordings into the framework layout."""
    src_path = Path(f"data/{spec.source}/pressure/G_wallp_SU_raw.hdf5")
    out = OUT_ROOT / f"{spec.wallp_letter}_{spec.framework_id}_wallp_SU_raw.hdf5"
    labels = list(REYNOLDS_INDEX)
    spacings = spec.cfg.SPACINGS
    geom = {sp: _spacing_meta(spec.source, sp) for sp in spacings}

    with h5py.File(src_path, "r") as src, h5py.File(out, "w") as dst:
        root = src["wallp_raw"]
        n = root[f"{labels[0]}/close/PH1_Pa"].shape[0]
        for k, v in _root_attrs(spec, "wallp", src_path, src, n, datatype="raw").items():
            dst.attrs[k] = v

        g_cal = dst.create_group("Calibration data")
        g_obs = dst.create_group("Observations")

        for idx, label in enumerate(labels):
            re_id = REYNOLDS_INDEX[label]
            s_cond = root[label]
            phys = _phys(spec.source, label)

            # ---- Observations: the raw Pa series ----
            d_cond = g_obs.create_group(re_id)
            for k, v in _condition_attrs(s_cond, spec, idx, phys=phys).items():
                d_cond.attrs[k] = v
            d_cond.attrs["stage_note"] = (
                "Raw, uncalibrated wall pressure in Pa (sensitivity + psig only). "
                "Per-station calibrated u_tau/Re_tau/d_plus are in the companion "
                "production file."
            )
            st_u_tau = _station_u_tau(spec.source, label, spec.framework_id)
            f_cut_st: dict[str, float] = {}
            for spacing in spacings:
                d_sp = d_cond.create_group(spacing)
                for k, v in geom[spacing].items():
                    d_sp.attrs[k] = v
                for ch in ("PH1", "PH2"):
                    station = STATIONS[(spacing, ch)]
                    f_cut = f_cut_hz(st_u_tau[(spacing, ch)], phys["nu"],
                                     tplus_cut=spec.cfg.TPLUS_CUT)
                    f_cut_st[station] = f_cut
                    extra = {
                        "station_id": station,
                        "x_m": float(geom[spacing][f"x_{ch}"]),
                        "f_cut_Hz": f_cut,
                        "units": "Pa",
                    }
                    _copy_series(s_cond[f"{spacing}/{ch}_Pa"], d_sp, f"{ch}_Pa", extra)
            d_cond.attrs["f_cut_Hz"] = float(min(f_cut_st.values()))
            d_cond.attrs["f_cut_note"] = F_CUT_NOTE

            # ---- Calibration data: the raw PH<->NC calibration recordings ----
            c_cond = g_cal.create_group(re_id)
            c_cond.attrs["reynolds_index"] = re_id
            c_cond.attrs["psig"] = float(s_cond.attrs["psig"])
            s_frf = s_cond["FRF_PH_to_NC"]
            g = c_cond.create_group("FRF_PH_to_NC_recordings")
            for k, v in s_frf.attrs.items():
                g.attrs[k] = v
            g.attrs["note"] = (
                "Raw semi-anechoic calibration recordings (time series, Pa) that map "
                "the pinhole microphone to the nose-cone microphone. Run1 pairs NC "
                "with PH1; Run2 pairs NC with PH2. The PH->NC FRF is derived from "
                "these; the FRF actually applied in production is in the companion "
                "production file."
            )
            for run in s_frf:
                gr = g.create_group(run)
                for ds in s_frf[run]:
                    _copy_series(s_frf[run][ds], gr, ds, {"units": "Pa"})
    return out


def build_freestream_raw(spec: CaseSpec) -> Path:
    """Repackage the raw (uncalibrated, Pa) nose-cone freestream signals plus the
    raw semi-anechoic NC<->nkd calibration recordings into the framework layout."""
    src_path = Path(f"data/{spec.source}/pressure/F_freestreamp_SU_raw.hdf5")
    out = OUT_ROOT / f"{spec.freestream_letter}_{spec.framework_id}_freestreamp_SU_raw.hdf5"
    labels = list(REYNOLDS_INDEX)

    with h5py.File(src_path, "r") as src, h5py.File(out, "w") as dst:
        root = src["freestream_raw"]
        n = root[f"{labels[0]}/close/NC_Pa"].shape[0]
        for k, v in _root_attrs(spec, "freestreamp", src_path, src, n, datatype="raw").items():
            dst.attrs[k] = v

        g_cal = dst.create_group("Calibration data")
        g_obs = dst.create_group("Observations")

        for idx, label in enumerate(labels):
            re_id = REYNOLDS_INDEX[label]
            s_cond = root[label]
            phys = _phys(spec.source, label)

            d_cond = g_obs.create_group(re_id)
            for k, v in _condition_attrs(s_cond, spec, idx, phys=phys).items():
                d_cond.attrs[k] = v
            d_cond.attrs["stage_note"] = (
                "Raw, uncalibrated nose-cone freestream pressure in Pa "
                "(sensitivity + psig only). NC->nkd FRF is applied in production."
            )
            for spacing in ("close", "far"):
                if spacing not in s_cond:
                    continue
                d_sp = d_cond.create_group(spacing)
                _copy_series(s_cond[f"{spacing}/NC_Pa"], d_sp, "NC_Pa",
                             {"units": "Pa", "sensor": "NC (nose-cone reference microphone)"})

            # ---- Calibration data: the raw NC<->nkd calibration recordings ----
            c_cond = g_cal.create_group(re_id)
            c_cond.attrs["reynolds_index"] = re_id
            c_cond.attrs["psig"] = float(s_cond.attrs["psig"])
            if "FRF_NC_to_nkd" in s_cond:
                s_frf = s_cond["FRF_NC_to_nkd"]
                g = c_cond.create_group("FRF_NC_to_nkd_recordings")
                for k, v in s_frf.attrs.items():
                    g.attrs[k] = v
                g.attrs["note"] = (
                    "Raw semi-anechoic calibration recordings (time series, Pa) that "
                    "map the nose-cone microphone to the naked reference. The NC->nkd "
                    "FRF is derived from these; the FRF applied in production is in "
                    "the companion production file."
                )
                for ds in s_frf:
                    _copy_series(s_frf[ds], g, ds, {"units": "Pa"})
    return out


# The framework SU table (tab:dataSU) defines products A-F. A/D are wall shear
# stress (separate deliverable); the pressure subset owned here is B/C/E/F, each
# in Raw and Production form. Phase2 (smooth-wall baseline) has NO table slot: it
# appears only as the drift-verification figure, so it is not built by default.
CASES = (
    CaseSpec("fence", "B1", "2-delta protrusions (fence)", "B", "C", FenceConfig()),
    CaseSpec("bump1", "B2", "Eaton bump", "E", "F", BumpConfig()),
)

# Kept available (not in tab:dataSU) should the baseline ever need repackaging.
BASELINE_CASE = CaseSpec("phase2", "B0", "smooth-wall baseline", "A", "D", Phase2Config())


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in CASES:
        for fn in (build_wallp_raw, build_wallp, build_freestream_raw, build_freestream):
            p = fn(spec)
            print(f"  wrote {p}  ({p.stat().st_size / 2**30:.2f} GiB)")
    print(
        "\nNote: A/D (wall shear stress) are a separate deliverable; phase1, phase2 "
        "and iso_re have no slot in the framework's SU pressure table and were not "
        "repackaged here."
    )


if __name__ == "__main__":
    main()
