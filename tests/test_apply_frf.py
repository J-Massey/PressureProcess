"""Numerical contract for src/core/apply_frf.py.

apply_frf is the workhorse of every pw_proc/fs_proc stage: it applies a
measured FRF to a time series. If its interpolation, out-of-band handling or
DC/Nyquist policy silently changes, every production HDF5 in the repo changes
with it, so those behaviours are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.apply_frf import apply_frf

FS = 2_000.0
N = 4096


@pytest.fixture
def signal() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(N)


def test_identity_frf_returns_the_demeaned_input(signal: np.ndarray) -> None:
    """H == 1 everywhere is a no-op apart from the demean and the explicitly
    zeroed DC/Nyquist bins. Asserted in the frequency domain because the
    discarded Nyquist term shows up in the time domain as a constant-magnitude
    alternating residual that no sensible time-domain tolerance describes."""
    f = np.array([0.0, FS / 2.0])
    H = np.array([1.0 + 0.0j, 1.0 + 0.0j])

    out = apply_frf(signal, FS, f, H)

    expected = signal - signal.mean()
    spec_out = np.fft.rfft(out)
    spec_expected = np.fft.rfft(expected)

    # Every bin except DC and Nyquist is preserved.
    assert np.abs(spec_out[1:-1] - spec_expected[1:-1]).max() < 1e-8 * np.abs(spec_expected).max()
    # DC and Nyquist are zeroed, by design.
    assert abs(spec_out[0]) < 1e-8 * np.abs(spec_expected).max()
    assert abs(spec_out[-1]) < 1e-8 * np.abs(spec_expected).max()


def test_constant_gain_scales_amplitude(signal: np.ndarray) -> None:
    f = np.array([0.0, FS / 2.0])
    out = apply_frf(signal, FS, f, np.array([2.0 + 0.0j, 2.0 + 0.0j]))

    assert out.std() == pytest.approx(2.0 * (signal - signal.mean()).std(), rel=1e-3)


def test_pure_delay_frf_shifts_the_signal(signal: np.ndarray) -> None:
    """H = exp(-j 2 pi f tau) must delay by tau. This pins the sign convention
    of the phase that apply_frf reconstructs from unwrap(angle(H))."""
    delay_samples = 5
    f = np.linspace(0.0, FS / 2.0, 2049)
    H = np.exp(-2j * np.pi * f * (delay_samples / FS))

    out = apply_frf(signal, FS, f, H)

    demeaned = signal - signal.mean()
    shifted = np.roll(demeaned, delay_samples)
    interior = slice(delay_samples + 5, -5)
    assert np.corrcoef(out[interior], shifted[interior])[0, 1] > 0.999


def test_dc_and_nyquist_bins_are_zeroed(signal: np.ndarray) -> None:
    """zero_dc=True (the default, and what every caller uses) must remove the
    mean exactly, not approximately."""
    f = np.array([0.0, FS / 2.0])
    H = np.array([3.0 + 0.0j, 3.0 + 0.0j])

    out = apply_frf(signal + 100.0, FS, f, H)

    assert abs(out.mean()) < 1e-3 * out.std()


def test_out_of_band_magnitude_is_held_at_unity(signal: np.ndarray) -> None:
    """Outside the measured band apply_frf holds |H| at 1.0 (np.interp
    left/right), it does not extrapolate the in-band gain. A change here would
    silently rescale everything above the calibration's top frequency."""
    f_band = np.linspace(0.0, FS / 8.0, 257)
    H = np.full(f_band.shape, 4.0 + 0.0j)

    out = apply_frf(signal, FS, f_band, H)

    spec_in = np.abs(np.fft.rfft(out))
    spec_ref = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / FS)

    in_band = (freqs > 10.0) & (freqs < FS / 8.0 * 0.9)
    out_band = freqs > FS / 8.0 * 1.2
    assert np.median(spec_in[in_band] / spec_ref[in_band]) == pytest.approx(4.0, rel=0.05)
    assert np.median(spec_in[out_band] / spec_ref[out_band]) == pytest.approx(1.0, rel=0.05)


def test_float32_and_float64_agree(signal: np.ndarray) -> None:
    """The pipelines run at float32 (WORK_DTYPE) to keep memory down; that must
    not change the answer."""
    f = np.linspace(0.0, FS / 2.0, 513)
    H = (1.0 / (1.0 + 1j * f / 200.0)).astype(np.complex128)

    out32 = apply_frf(signal.astype(np.float32), FS, f.astype(np.float32),
                      H.astype(np.complex64), dtype=np.float32)
    out64 = apply_frf(signal, FS, f, H, dtype=np.float64)

    assert out32.dtype == np.float32
    assert np.abs(out32 - out64).max() < 1e-3 * out64.std()


def test_unsupported_dtype_is_rejected(signal: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        apply_frf(signal, FS, np.array([0.0, FS / 2.0]),
                  np.array([1.0 + 0j, 1.0 + 0j]), dtype=np.float16)


def test_output_length_matches_input(signal: np.ndarray) -> None:
    """apply_frf zero-pads to the next power of two internally and must trim
    back; N here is already a power of two, so use a length that is not."""
    x = signal[:3000]
    out = apply_frf(x, FS, np.array([0.0, FS / 2.0]), np.array([1.0 + 0j, 1.0 + 0j]))
    assert out.shape == x.shape
