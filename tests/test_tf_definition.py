"""Numerical contract for src/core/tf_definition.py.

This module produces every calibration transfer function in the repo
(PH -> NC and NC -> nkd), so its estimator orientation, coherence weighting
and smoothing are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from src.core.tf_definition import (
    combine_anechoic_calibrations,
    estimate_frf,
    smooth_frf_logfreq,
)

FS = 2_000.0
NPERSEG = 1024


def _fir_response(b: np.ndarray, f: np.ndarray, fs: float = FS) -> np.ndarray:
    """Analytic frequency response of a causal FIR filter b."""
    return sum(bk * np.exp(-2j * np.pi * f / fs * k) for k, bk in enumerate(b))


@pytest.fixture
def known_fir_pair():
    """y = b * x with a known causal FIR b, long enough for tight Welch stats."""
    rng = np.random.default_rng(0)
    b = np.array([0.5, 0.3, 0.2])
    x = rng.standard_normal(200_000)
    y = lfilter(b, [1.0], x)
    return b, x, y


def test_estimate_frf_recovers_the_magnitude(known_fir_pair) -> None:
    b, x, y = known_fir_pair
    f, H, _ = estimate_frf(x, y, fs=FS, nperseg=NPERSEG)

    expected = np.abs(_fir_response(b, f))
    rel_err = np.abs(np.abs(H) - expected) / expected
    assert np.median(rel_err) < 1e-3
    assert rel_err.max() < 1e-2


def test_estimate_frf_phase_is_the_conjugate_of_the_x_to_y_response(known_fir_pair) -> None:
    """CHARACTERIZATION TEST -- pins current behaviour, which is NOT the
    textbook H1.

    estimate_frf computes ``H = conj(Sxy) / Sxx`` (src/core/tf_definition.py:92)
    on the stated assumption that ``scipy.signal.csd(x, y) == E{X conj(Y)}``.
    SciPy's actual convention is ``csd(x, y) == E{conj(X) Y}``, so
    ``Sxy / Sxx`` is already H1(x -> y) and the extra conjugation returns the
    COMPLEX CONJUGATE of the true response: correct magnitude, negated phase.

    Consequence: every saved H_fused has a sign-flipped group delay. PSD
    magnitudes (what the production plots report) are unaffected; anything
    phase-derived downstream of an FRF correction is.

    This test asserts the conjugate so the behaviour cannot drift silently.
    Flipping it to the textbook convention is a one-character change
    (drop the np.conjugate) but changes every calibration file on disk and
    requires reprocessing, so it is deliberately not done here.
    """
    b, x, y = known_fir_pair
    f, H, _ = estimate_frf(x, y, fs=FS, nperseg=NPERSEG)

    true_response = _fir_response(b, f)

    assert np.median(np.abs(H - np.conj(true_response)) / np.abs(true_response)) < 1e-3
    # And it is genuinely NOT the un-conjugated response (guards against a
    # future "fix" landing without this test being updated).
    assert np.median(np.abs(H - true_response) / np.abs(true_response)) > 0.1


def test_estimate_frf_coherence_is_unity_for_a_noise_free_pair(known_fir_pair) -> None:
    _, x, y = known_fir_pair
    _, _, gamma2 = estimate_frf(x, y, fs=FS, nperseg=NPERSEG)

    assert gamma2.min() > 0.99
    assert gamma2.max() <= 1.0


def test_independent_noise_drops_coherence_but_not_the_gain(known_fir_pair) -> None:
    """Uncorrelated noise on the output must degrade gamma^2 while leaving the
    H1 magnitude estimate ~unbiased -- that is the whole point of using H1."""
    b, x, y = known_fir_pair
    rng = np.random.default_rng(1)
    y_noisy = y + 0.5 * y.std() * rng.standard_normal(y.size)

    f, H_clean, g2_clean = estimate_frf(x, y, fs=FS, nperseg=NPERSEG)
    _, H_noisy, g2_noisy = estimate_frf(x, y_noisy, fs=FS, nperseg=NPERSEG)

    assert g2_noisy.mean() < 0.9 * g2_clean.mean()
    expected = np.abs(_fir_response(b, f))
    assert np.median(np.abs(np.abs(H_noisy) - expected) / expected) < 0.02


def test_fusion_follows_the_high_coherence_input() -> None:
    """A coherence-weighted fusion of a trusted and an untrusted FRF must
    return the trusted one."""
    f = np.linspace(0.0, 1_000.0, 513)
    H_good = np.full(f.shape, 2.0 + 0.0j)
    H_bad = np.full(f.shape, 10.0 + 0.0j)
    g_good = np.full(f.shape, 0.99)
    g_bad = np.full(f.shape, 0.05)

    _, H_fused, _ = combine_anechoic_calibrations(
        f, H_good, g_good, f, H_bad, g_bad, smooth_oct=None
    )

    assert np.median(np.abs(H_fused)) == pytest.approx(2.0, rel=1e-6)


def test_fusion_gates_everything_below_gmin() -> None:
    """gmin=0.4 by default: if neither input clears it, both weights are zero
    and the fused FRF collapses to 0 rather than to a garbage average."""
    f = np.linspace(0.0, 1_000.0, 513)
    H1 = np.full(f.shape, 2.0 + 0.0j)
    H2 = np.full(f.shape, 10.0 + 0.0j)
    low = np.full(f.shape, 0.05)

    _, H_fused, _ = combine_anechoic_calibrations(f, H1, low, f, H2, low, smooth_oct=None)

    assert np.abs(H_fused).max() == pytest.approx(0.0, abs=1e-12)


def test_fusion_targets_the_finer_frequency_grid() -> None:
    f_fine = np.linspace(0.0, 1_000.0, 513)
    f_coarse = np.linspace(0.0, 1_000.0, 129)
    high = lambda n: np.full(n, 0.99)  # noqa: E731

    f_out, H_out, _ = combine_anechoic_calibrations(
        f_coarse, np.full(129, 1.0 + 0j), high(129),
        f_fine, np.full(513, 1.0 + 0j), high(513),
        smooth_oct=None,
    )

    assert f_out.size == f_fine.size
    assert H_out.size == f_fine.size


def test_smoothing_preserves_the_level_but_removes_ripple() -> None:
    """Log-frequency smoothing must reduce bin-to-bin ripple without shifting
    the mean level of the FRF."""
    f = np.linspace(1.0, 1_000.0, 2048)
    rng = np.random.default_rng(3)
    ripple = 1.0 + 0.2 * rng.standard_normal(f.size)
    H = ripple.astype(complex)

    H_smooth = smooth_frf_logfreq(f, H, span_oct=1 / 3, ppo=48)

    assert np.std(np.diff(np.abs(H_smooth))) < 0.5 * np.std(np.diff(np.abs(H)))
    assert np.mean(np.abs(H_smooth)) == pytest.approx(np.mean(np.abs(H)), rel=0.05)


def test_smoothing_is_a_noop_for_a_disabled_span() -> None:
    f = np.linspace(1.0, 100.0, 64)
    H = np.exp(1j * f / 10.0)
    assert np.array_equal(smooth_frf_logfreq(f, H, span_oct=0.0), H)


def test_estimate_frf_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        smooth_frf_logfreq(np.arange(10.0), np.arange(5, dtype=complex))
