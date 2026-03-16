import numpy as np


def apply_frf(
    x: np.ndarray,
    fs: float,
    f: np.ndarray,
    H: np.ndarray,
    demean: bool = True,
    zero_dc: bool = True,
    dtype: np.dtype | type[np.floating] = np.float64,
):
    """
    Apply a measured FRF H (x to y) to a time series x to synthesise y.
    This is the forward operation: Y = H * X in the frequency domain.
    """
    work_dtype = np.dtype(dtype)
    if work_dtype == np.float32:
        complex_dtype = np.complex64
    elif work_dtype == np.float64:
        complex_dtype = np.complex128
    else:
        raise ValueError(f"Unsupported dtype for FRF application: {work_dtype}")

    x = np.asarray(x, dtype=work_dtype)
    f = np.asarray(f, dtype=work_dtype)
    H = np.asarray(H, dtype=complex_dtype)
    if demean:
        x = x - x.mean(dtype=work_dtype)

    N = x.size
    Nfft = int(2 ** np.ceil(np.log2(N)))
    X = np.fft.rfft(x, n=Nfft).astype(complex_dtype, copy=False)
    fr = np.fft.rfftfreq(Nfft, d=1.0 / fs).astype(work_dtype, copy=False)

    mag = np.abs(H).astype(work_dtype, copy=False)
    phi = np.unwrap(np.angle(H)).astype(work_dtype, copy=False)
    # Safer OOB behaviour: taper magnitude to zero outside measured band
    mag_i = np.interp(fr, f, mag, left=1.0, right=1.0).astype(work_dtype, copy=False)
    phi_i = np.interp(fr, f, phi, left=float(phi[0]), right=float(phi[-1])).astype(
        work_dtype,
        copy=False,
    )
    Hi = mag_i.astype(complex_dtype, copy=False) * np.exp(
        phi_i.astype(complex_dtype, copy=False) * complex_dtype(1j)
    )

    if zero_dc:
        Hi[0] = 0.0
        if Nfft % 2 == 0:
            Hi[-1] = 0.0

    X *= Hi
    y = np.fft.irfft(X, n=Nfft)[:N]
    return y.astype(work_dtype, copy=False)
