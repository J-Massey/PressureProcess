"""Numerical contract for the Wiener background canceller.

This is the noise-rejection stage: a fixed FIR kernel identified from the
freestream reference microphone (NC/nkd) is convolved with that reference and
subtracted from the wall-pressure signal. The tests below build a signal whose
"facility noise" really is a known FIR-filtered copy of the reference, so the
canceller has a ground truth to be measured against.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

from src.core.wiener_filter_torch import (
    apply_wiener_kernel,
    wiener_cancel_background,
    wiener_cancel_hybrid,
)

from conftest import CPU

FS = 2_000.0
N = 20_000
TRUE_PATH = np.array([0.8, -0.4, 0.25, 0.1, -0.05])


@pytest.fixture
def correlated_case():
    """p0 = (known FIR * pn) + uncorrelated signal."""
    rng = np.random.default_rng(7)
    reference = rng.standard_normal(N)
    facility_noise = lfilter(TRUE_PATH, [1.0], reference)
    true_signal = 0.5 * rng.standard_normal(N)
    return reference, true_signal, facility_noise + true_signal


def test_wiener_identifies_the_true_transfer_path(correlated_case) -> None:
    """The solved FIR kernel must match the FIR that actually generated the
    noise. This is the single strongest statement that the canceller works."""
    reference, _, p0 = correlated_case

    _, kernel = wiener_cancel_background(
        p0, reference, FS, filter_order=64, preserve_mean=False,
        return_kernel=True, device=CPU,
    )

    assert np.allclose(kernel[: TRUE_PATH.size], TRUE_PATH, atol=0.02)
    # Beyond the true path length the kernel must decay to ~nothing.
    assert np.abs(kernel[TRUE_PATH.size:]).max() < 0.05


def test_wiener_removes_the_noise_and_keeps_the_signal(correlated_case) -> None:
    reference, true_signal, p0 = correlated_case

    clean = wiener_cancel_background(
        p0, reference, FS, filter_order=64, preserve_mean=False, device=CPU,
    )

    # Variance collapses to the uncorrelated signal's variance...
    assert clean.var() == pytest.approx(true_signal.var(), rel=0.05)
    assert clean.var() < 0.3 * p0.var()
    # ...and what is left really is the signal, not a lucky variance match.
    assert np.corrcoef(clean, true_signal)[0, 1] > 0.99


def test_wiener_with_an_uncorrelated_reference_does_not_eat_the_signal() -> None:
    """The failure mode that matters scientifically: if the reference carries
    no shared content, the canceller must be ~a no-op rather than subtracting
    real wall pressure."""
    rng = np.random.default_rng(11)
    signal = rng.standard_normal(N)
    unrelated = rng.standard_normal(N)

    clean = wiener_cancel_background(
        signal, unrelated, FS, filter_order=64, preserve_mean=False, device=CPU,
    )

    assert np.corrcoef(clean, signal)[0, 1] > 0.99
    assert clean.var() / signal.var() == pytest.approx(1.0, rel=0.02)


def test_alpha_scales_the_subtraction(correlated_case) -> None:
    """alpha is the leak factor: alpha=0 must leave the input untouched."""
    reference, _, p0 = correlated_case

    untouched = wiener_cancel_background(
        p0, reference, FS, filter_order=64, alpha=0.0, preserve_mean=False, device=CPU,
    )

    assert np.allclose(untouched, p0 - p0.mean(), atol=1e-5)


def test_preserve_mean_restores_the_input_mean(correlated_case) -> None:
    reference, _, p0 = correlated_case
    offset = p0 + 3.0

    kept = wiener_cancel_background(
        offset, reference, FS, filter_order=64, preserve_mean=True, device=CPU,
    )
    dropped = wiener_cancel_background(
        offset, reference, FS, filter_order=64, preserve_mean=False, device=CPU,
    )

    assert kept.mean() == pytest.approx(offset.mean(), rel=1e-3)
    assert abs(dropped.mean()) < 1e-3 * dropped.std()


def test_donor_kernel_reproduces_local_training_exactly(correlated_case) -> None:
    """The bump and fence pipelines do not train Wiener locally: they load a
    kernel trained on the phase2 smooth-wall donor and apply it with
    apply_wiener_kernel. Applying a kernel to the same data it was trained on
    must be identical to training in place, otherwise the two code paths have
    diverged."""
    reference, _, p0 = correlated_case

    local, kernel = wiener_cancel_background(
        p0, reference, FS, filter_order=64, preserve_mean=False,
        return_kernel=True, device=CPU,
    )
    transplanted = apply_wiener_kernel(
        p0, reference, kernel, alpha=1.0, preserve_mean=False, device=CPU,
    )

    assert np.allclose(transplanted, local, atol=1e-6)


def test_apply_wiener_kernel_rejects_an_oversized_kernel(correlated_case) -> None:
    reference, _, p0 = correlated_case
    with pytest.raises(ValueError, match="kernel length"):
        apply_wiener_kernel(p0[:32], reference[:32], np.zeros(64), device=CPU)


def test_wiener_accepts_non_contiguous_input(correlated_case) -> None:
    """h5py slices and reversed views reach this function in the pipelines;
    torch.as_tensor rejects negative strides, so the copy must happen inside."""
    reference, _, p0 = correlated_case
    reversed_p0 = p0[::-1]
    reversed_ref = reference[::-1]

    expected = wiener_cancel_background(
        np.ascontiguousarray(reversed_p0), np.ascontiguousarray(reversed_ref),
        FS, filter_order=32, device=CPU,
    )
    actual = wiener_cancel_background(
        reversed_p0, reversed_ref, FS, filter_order=32, device=CPU,
    )

    assert np.isfinite(actual).all()
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_filter_order_is_clamped_to_the_signal_length() -> None:
    rng = np.random.default_rng(5)
    short = rng.standard_normal(128)
    out = wiener_cancel_background(short, short, FS, filter_order=4096, device=CPU)
    assert out.shape == short.shape
    assert np.isfinite(out).all()


def test_hybrid_canceller_only_removes_the_zero_lag_component(correlated_case) -> None:
    """CHARACTERIZATION TEST -- pins a known defect in the 'hybrid' branch.

    wiener_cancel_hybrid computes ``H = csd(p0, pn) / Snn``
    (src/core/wiener_filter_torch.py:324). With SciPy's actual convention
    ``csd(a, b) == E{conj(A) B}``, the numerator needed for the pn -> p0 path
    is ``csd(pn, p0)``; using ``csd(p0, pn)`` yields the complex conjugate,
    whose impulse response is TIME-REVERSED. The code then keeps only the
    first m taps (``h_full[:m]``), so the entire anti-causal part -- i.e.
    everything except the zero-lag tap -- is discarded.

    Net effect: hybrid removes only the instantaneous component of the
    facility noise, while the exact solver removes the whole path. On the
    reference case here that is ~73% vs ~100% of the noise energy.

    This branch is selected by PRESSUREPROCESS_PW_DENOISER=hybrid, and by
    'auto' for signals >= 1e6 samples on macOS -- which real 50 kHz records
    exceed. See test_donor_kernel_reproduces_local_training_exactly for the
    exact-solver contract this is being compared against.
    """
    reference, true_signal, p0 = correlated_case

    hybrid = wiener_cancel_hybrid(p0, reference, FS, m=256, nperseg=1024)
    exact = wiener_cancel_background(
        p0, reference, FS, filter_order=256, preserve_mean=False, device=CPU,
    )

    # It does remove *something* -- the zero-lag term.
    assert hybrid.var() < p0.var()
    # But it recovers the true signal markedly worse than the exact solver.
    corr_hybrid = np.corrcoef(hybrid, true_signal)[0, 1]
    corr_exact = np.corrcoef(exact, true_signal)[0, 1]
    assert corr_exact > 0.99
    assert corr_hybrid < 0.85
    assert corr_exact - corr_hybrid > 0.1


def test_wiener_returns_the_noise_estimate_when_asked(correlated_case) -> None:
    reference, _, p0 = correlated_case

    clean, noise = wiener_cancel_background(
        p0, reference, FS, filter_order=64, preserve_mean=False,
        return_noise_estimate=True, device=CPU,
    )

    # p_clean = p0_demeaned - noise, by construction.
    assert np.allclose(clean + noise, p0 - p0.mean(), atol=1e-4)
    assert np.corrcoef(noise, lfilter(TRUE_PATH, [1.0], reference))[0, 1] > 0.99


def test_torch_and_numpy_inputs_agree(correlated_case) -> None:
    reference, _, p0 = correlated_case

    from_numpy = wiener_cancel_background(
        p0, reference, FS, filter_order=32, preserve_mean=False, device=CPU,
    )
    from_torch = wiener_cancel_background(
        torch.as_tensor(p0), torch.as_tensor(reference), FS,
        filter_order=32, preserve_mean=False, device=CPU,
    )

    assert np.allclose(from_numpy, from_torch, atol=1e-6)
