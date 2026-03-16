from __future__ import annotations

import numpy as np
import torch

from src.core.apply_frf import apply_frf
from src.core.wiener_filter_torch import wiener_cancel_background


def test_apply_frf_supports_float32_working_dtype() -> None:
    x = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    f = np.array([0.0, 100.0, 200.0], dtype=np.float32)
    H = np.array([1.0 + 0.0j, 0.75 + 0.1j, 0.5 + 0.0j], dtype=np.complex64)

    y32 = apply_frf(x, fs=400.0, f=f, H=H, dtype=np.float32)
    y64 = apply_frf(x.astype(np.float64), fs=400.0, f=f, H=H, dtype=np.float64)

    assert y32.dtype == np.float32
    assert np.allclose(y32, y64.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_wiener_cancel_background_accepts_negative_stride_inputs() -> None:
    t = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    p0 = (np.sin(2.0 * np.pi * 7.0 * t) + 0.2 * np.sin(2.0 * np.pi * 17.0 * t))[::-1]
    pn = (0.5 * np.sin(2.0 * np.pi * 7.0 * t + 0.4))[::-1]

    expected = wiener_cancel_background(
        np.ascontiguousarray(p0),
        np.ascontiguousarray(pn),
        FS=256.0,
        filter_order=16,
        device=torch.device("cpu"),
    )
    actual = wiener_cancel_background(
        p0,
        pn,
        FS=256.0,
        filter_order=16,
        device=torch.device("cpu"),
    )

    assert actual.shape == p0.shape
    assert np.isfinite(actual).all()
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)
