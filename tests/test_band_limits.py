"""Contract for src/band_limits.py -- the reported-band upper edge f_cut.

f_cut is a *reporting* limit, not a signal filter: the pipelines never
band-pass with it, the plots mask with it. It is
``min(pinhole Helmholtz guard, T+ spatial-resolution limit)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.band_limits import (
    F_HELMHOLTZ_GUARD_HZ,
    air_nu,
    f_cut_hz,
    reported_band_f_cuts,
)

P_ATM = 101_325.0
PSI_TO_PA = 6_894.76
R_AIR = 287.05


def test_sutherland_viscosity_matches_a_hand_computation() -> None:
    """nu = mu/rho with Sutherland's mu and the ideal gas law, at 0 psig, 18 C."""
    T = 273.15 + 18.0
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    rho = P_ATM / (R_AIR * T)

    nu = air_nu(0.0, T, p_atm=P_ATM, psi_to_pa=PSI_TO_PA, R=R_AIR)

    assert nu == pytest.approx(mu / rho, rel=1e-12)
    # Sanity: air at room conditions is ~1.5e-5 m^2/s.
    assert 1.4e-5 < nu < 1.6e-5


def test_pressurising_the_tunnel_lowers_kinematic_viscosity() -> None:
    """Density rises with gauge pressure while mu does not, so nu must fall
    roughly as 1/p_abs. This is what pushes the T+ limit up with Re_tau."""
    T = 293.15
    nu_0 = air_nu(0.0, T, p_atm=P_ATM, psi_to_pa=PSI_TO_PA, R=R_AIR)
    nu_100 = air_nu(100.0, T, p_atm=P_ATM, psi_to_pa=PSI_TO_PA, R=R_AIR)

    expected_ratio = P_ATM / (P_ATM + 100.0 * PSI_TO_PA)
    assert nu_100 / nu_0 == pytest.approx(expected_ratio, rel=1e-12)
    assert nu_100 < nu_0


def test_helmholtz_guard_binds_when_the_spatial_limit_is_higher() -> None:
    """The pinhole cavity resonance (~1.4 kHz) is pressure-independent, so the
    1.2 kHz guard is the binding constraint at ordinary conditions."""
    nu = 1.5e-5
    u_tau = 0.537  # f_spatial ~ 1.9 kHz, well above the guard
    assert f_cut_hz(u_tau, nu, tplus_cut=10.0) == F_HELMHOLTZ_GUARD_HZ


def test_spatial_limit_binds_when_it_falls_below_the_guard() -> None:
    """Low local u_tau (e.g. downstream of the fence at 0 psig) undercuts the
    resonance guard and becomes the reported edge."""
    nu = 1.5e-5
    u_tau = 0.1  # f_spatial = 0.01/(1.5e-5*10) ~ 67 Hz
    cut = f_cut_hz(u_tau, nu, tplus_cut=10.0)

    assert cut < F_HELMHOLTZ_GUARD_HZ
    assert cut == pytest.approx(u_tau**2 / (nu * 10.0), rel=1e-12)


def test_f_cut_is_monotonic_in_u_tau() -> None:
    nu = 1.5e-5
    cuts = [f_cut_hz(u, nu, tplus_cut=10.0) for u in (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert cuts == sorted(cuts)
    assert cuts[-1] == F_HELMHOLTZ_GUARD_HZ  # saturates at the guard


def test_tighter_tplus_cut_lowers_the_spatial_limit() -> None:
    nu, u_tau = 1.5e-5, 0.1
    assert f_cut_hz(u_tau, nu, tplus_cut=20.0) < f_cut_hz(u_tau, nu, tplus_cut=10.0)


def test_reported_band_returns_one_cut_per_condition() -> None:
    cuts = reported_band_f_cuts(
        psigs=(0.0, 50.0, 100.0), tdegs=(18.0, 20.0, 22.0),
        u_taus=(0.537, 0.522, 0.506), tplus_cut=10.0,
        p_atm=P_ATM, psi_to_pa=PSI_TO_PA, R=R_AIR,
    )
    assert len(cuts) == 3
    assert all(isinstance(c, float) for c in cuts)


@pytest.mark.parametrize(
    "module",
    [
        "src.config_params",
        "src.save_bump.config_params",
        "src.save_fence.config_params",
        "src.save_iso_re.config_params",
        "src.save_phase2.config_params",
    ],
)
def test_every_case_currently_reports_the_helmholtz_guard(module: str) -> None:
    """Regression lock on the shipped numbers.

    With each case's nominal per-pressure u_tau, the T+ limit sits above
    1.2 kHz everywhere, so every reported edge is the resonance guard. If a
    u_tau or TPLUS_CUT is retuned such that this stops being true, the change
    should be deliberate -- this test makes it visible.
    """
    import importlib

    cfg = importlib.import_module(module).Config()

    assert cfg.F_CUTS == (F_HELMHOLTZ_GUARD_HZ,) * len(cfg.PSIGS)


def test_f_cut_is_not_the_analog_antialias_filter() -> None:
    """Guards against the old (1200, 4000, 10000) triple creeping back in:
    f_cut must NOT track ANALOG_LP_FILTER, which is a different thing."""
    from src.config_params import Config

    cfg = Config()
    assert cfg.F_CUTS != tuple(float(v) for v in cfg.ANALOG_LP_FILTER)
    assert all(cut < lp for cut, lp in zip(cfg.F_CUTS, cfg.ANALOG_LP_FILTER))
