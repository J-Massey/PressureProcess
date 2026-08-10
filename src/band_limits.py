"""
Reported-band upper edge (f_cut) for the pinhole wall-pressure measurements.

Two effects contaminate the wall-pressure spectrum at high frequency and set
the edge of the *reported* band (they are reporting limits, not signal
filters -- the production pipeline band-passes at [1 Hz, analog_LP] only):

1. Pinhole Helmholtz cavity resonance. The cavity behind the 70-um pinhole
   resonates at ~1.4 kHz. The resonance frequency is set by the cavity
   geometry and the speed of sound, so it is the SAME at every tunnel
   pressurisation (c = sqrt(gamma*R*T) does not change with pressure at
   fixed temperature); only its damping falls -- the peak sharpens -- as
   density rises. It therefore binds as a fixed guard in Hz, NOT as a
   viscous-scaled cut: raising the cut with Re_tau (e.g. 4/10 kHz at
   50/100 psig) leaves the entire resonance inside the band. The
   semi-anechoic FRF cannot be trusted to invert the resonance either
   (high Q, mounting-sensitive), so the band is cut BELOW it.

2. Spatial resolution / attenuation of the pinhole (finite d+). Expressed
   as a viscous time-scale limit T+ = u_tau^2/(nu f) >= TPLUS_CUT; in Hz,
   f <= u_tau^2 / (nu * TPLUS_CUT). This one DOES scale with the local
   friction velocity, so it is evaluated per station where a measured
   per-station u_tau exists.

The reported cut is the minimum of the two. With the guard at 1.2 kHz the
resonance binds at essentially every condition/station; the T+ limit only
undercuts it where the local u_tau is low (e.g. downstream of the fence at
0 psig).
"""

from __future__ import annotations

# Onset of the pinhole cavity (Helmholtz) resonance, identified from the
# semi-anechoic PH<->NC calibration TFs (|H| dip centred ~1.4 kHz at every
# psig) and the raw wall spectra (broad peak at 1.2-1.7 kHz in every case).
# Pressure-independent; guard placed just below the onset.
F_HELMHOLTZ_GUARD_HZ = 1200.0

# Sutherland's law constants (match air_props_from_gauge in the pipelines).
_MU0, _T0, _S = 1.716e-5, 273.15, 110.4


def air_nu(psig: float, T_K: float, *, p_atm: float, psi_to_pa: float,
           R: float) -> float:
    """Kinematic viscosity [m^2/s] at gauge pressure [psi] and T [K]."""
    mu = _MU0 * (T_K / _T0) ** 1.5 * (_T0 + _S) / (T_K + _S)
    rho = (p_atm + psig * psi_to_pa) / (R * T_K)
    return mu / rho


def f_cut_hz(u_tau: float, nu: float, *, tplus_cut: float,
             helmholtz_guard_hz: float = F_HELMHOLTZ_GUARD_HZ) -> float:
    """Reported-band upper edge: min(resonance guard, T+ resolution limit)."""
    f_spatial = u_tau ** 2 / (nu * tplus_cut)
    return float(min(helmholtz_guard_hz, f_spatial))


def reported_band_f_cuts(*, psigs, tdegs, u_taus, tplus_cut, p_atm,
                         psi_to_pa, R,
                         helmholtz_guard_hz: float = F_HELMHOLTZ_GUARD_HZ,
                         ) -> tuple[float, ...]:
    """Per-condition f_cut for a case config (per-pressure nominal u_tau)."""
    cuts = []
    for psig, tdeg, u_tau in zip(psigs, tdegs, u_taus):
        nu = air_nu(float(psig), 273.15 + float(tdeg),
                    p_atm=p_atm, psi_to_pa=psi_to_pa, R=R)
        cuts.append(f_cut_hz(float(u_tau), nu, tplus_cut=tplus_cut,
                             helmholtz_guard_hz=helmholtz_guard_hz))
    return tuple(cuts)
