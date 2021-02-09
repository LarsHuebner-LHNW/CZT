"""Test CZT package.

To run:

    pytest test_czt.py -v

"""

import numpy as np

import czt_L as czt

# DFT for comparison


def dft(t, x, f=None):
    """Convert signal from time-domain to frequency-domain using a Discrete
    Fourier Transform (DFT).

    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal

    Returns:
        np.ndarray: frequency-domain signal

    """

    if f is None:
        dt = t[1] - t[0]  # time step
        Fs = 1 / dt  # sample frequency
        f = np.linspace(-Fs / 2, Fs / 2, len(t))

    X = np.zeros(len(f), dtype=complex)
    for k in range(len(X)):
        X[k] = np.sum(x * np.exp(-2j * np.pi * f[k] * t))

    return f, X


def idft(f, X, t=None):
    """Convert signal from time-domain to frequency-domain using an Inverse
    Discrete Fourier Transform (IDFT).

    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal

    Returns:
        np.ndarray: time-domain signal

    """

    if t is None:
        bw = f.max() - f.min()
        t = np.linspace(0, bw / 2, len(f))

    N = len(t)
    x = np.zeros(N, dtype=complex)
    for n in range(len(x)):
        x[n] = np.sum(X * np.exp(2j * np.pi * f * t[n]))
        # for k in range(len(X)):
        #     x[n] += X[k] * np.exp(2j * np.pi * f[k] * t[n])
    x /= N

    return t, x


# actual tests following


def test_compare_different_czt_methods(debug=False):
    """Compare different CZT calculation methods."""

    # Create time-domain data
    t = np.arange(0, 20e-3, 1e-4)

    # Signal
    def model(t):
        output = (
            1.0 * np.sin(2 * np.pi * 1e3 * t)
            + 0.3 * np.sin(2 * np.pi * 2e3 * t)
            + 0.1 * np.sin(2 * np.pi * 3e3 * t)
        ) * np.exp(-1e3 * t)
        return output

    x = model(t)

    # Calculate CZT using different methods
    X_czt2 = czt.czt(x)

    # Plot for debugging purposes
    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.abs(X_czt2))
        plt.figure()
        plt.plot(X_czt2.real)
        plt.figure()
        plt.plot(X_czt2.imag)
        plt.show()


def test_compare_czt_fft_dft(debug=False):
    """Compare CZT, FFT and DFT."""

    # Create time-domain data
    t = np.arange(0, 20e-3 + 1e-10, 1e-4)

    # Signal
    def model(t):
        output = (
            1.0 * np.sin(2 * np.pi * 1e3 * t)
            + 0.3 * np.sin(2 * np.pi * 2e3 * t)
            + 0.1 * np.sin(2 * np.pi * 3e3 * t)
        ) * np.exp(-1e3 * t)
        return output

    x = model(t)

    # CZT (defaults to FFT)
    X_czt = np.fft.fftshift(czt.czt(x))

    # FFT
    X_fft = np.fft.fftshift(np.fft.fft(x))

    # DFT
    _, X_dft = dft(t, x)

    # Plot for debugging purposes
    if debug:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.title("Absolute")
        plt.plot(np.abs(X_czt), label="CZT")
        plt.plot(np.abs(X_fft), label="FFT", ls="--")
        plt.plot(np.abs(X_dft), label="DFT", ls="--")
        plt.legend()
        plt.figure(figsize=(10, 8))
        plt.title("Real")
        plt.plot(X_czt.real, label="CZT")
        plt.plot(X_fft.real, label="FFT", ls="--")
        plt.plot(X_dft.real, label="DFT", ls="--")
        plt.legend()
        plt.figure(figsize=(10, 8))
        plt.title("Imaginary")
        plt.plot(X_czt.imag, label="CZT")
        plt.plot(X_fft.imag, label="FFT", ls="--")
        plt.plot(X_dft.imag, label="DFT", ls="--")
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(X_czt, X_fft, decimal=12)
    np.testing.assert_allclose(X_czt, X_dft, atol=0.2)


def test_czt_to_iczt(debug=False):
    """Test CZT -> ICZT."""

    # Create time-domain data
    t = np.arange(0, 20e-3, 1e-4)

    # Signal
    def model(t):
        output = (
            1.0 * np.sin(2 * np.pi * 1e3 * t)
            + 0.3 * np.sin(2 * np.pi * 2e3 * t)
            + 0.1 * np.sin(2 * np.pi * 3e3 * t)
        ) * np.exp(-1e3 * t)
        return output

    x = model(t)

    # CZT (defaults to FFT)
    X_czt = czt.czt(x)

    # ICZT
    x_iczt = czt.iczt(X_czt)

    # Plot for debugging purposes
    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x.real)
        plt.plot(x_iczt.real)
        plt.figure()
        plt.plot(x.imag)
        plt.plot(x_iczt.imag)
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x, x_iczt, decimal=12)


def test_time_to_freq_to_time(debug=False):
    """Test time -> frequency -> time domain conversions."""

    # Create time-domain data
    t1 = np.arange(0, 20e-3, 1e-4)

    # Signal
    def model(t):
        output = (
            1.0 * np.sin(2 * np.pi * 1e3 * t)
            + 0.3 * np.sin(2 * np.pi * 2e3 * t)
            + 0.1 * np.sin(2 * np.pi * 3e3 * t)
        ) * np.exp(-1e3 * t)
        return output

    x1 = model(t1)

    # Frequency domain
    f, X = czt.time2freq(t1, x1)

    # Back to time domain
    t2, x2 = czt.freq2time(f, X)

    # Plot for debugging purposes
    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Absolute")
        plt.plot(t1, np.abs(x1), label="Original")
        plt.plot(t2, np.abs(x2), label="Recovered", ls="--")
        plt.legend()
        plt.figure()
        plt.title("Real")
        plt.plot(t1, x1.real, label="Original")
        plt.plot(t2, x2.real, label="Recovered", ls="--")
        plt.legend()
        plt.figure()
        plt.title("Imaginary")
        plt.plot(t1, x1.imag, label="Original")
        plt.plot(t2, x2.imag, label="Recovered", ls="--")
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_allclose(x1, x2, atol=0.01)


def test_compare_iczt_idft(debug=False):
    """Compare ICZT to IDFT."""

    # Create time-domain data
    t = np.arange(0, 20e-3, 1e-4)

    # Signal
    def model(t):
        output = (
            1.0 * np.sin(2 * np.pi * 1e3 * t)
            + 0.3 * np.sin(2 * np.pi * 2e3 * t)
            + 0.1 * np.sin(2 * np.pi * 3e3 * t)
        ) * np.exp(-1e3 * t)
        return output

    x = model(t)

    # Frequency domain using CZT
    f, X = czt.time2freq(t, x)

    # Get time-domain using ICZT
    _, x_iczt = czt.freq2time(f, X, t)

    # Get time-domain using IDFT
    _, x_idft = idft(f, X, t)

    # Plot for debugging purposes
    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(t, x.real, "k", label="Original")
        plt.plot(t, x_iczt.real, "g:", label="ICZT")
        plt.plot(t, x_idft.real, "r--", label="IDFT")
        plt.legend()
        plt.figure()
        plt.plot(t, x.imag, "k", label="Original")
        plt.plot(t, x_iczt.imag, "g:", label="ICZT")
        plt.plot(t, x_idft.imag, "r--", label="IDFT")
        plt.legend()
        plt.show()

    # Compare
    np.testing.assert_almost_equal(x_iczt, x_idft, decimal=12)


def test_frequency_zoom(debug=False):
    """Test frequency zoom."""

    # Create time-domain data
    t = np.arange(0, 20e-3 + 1e-10, 1e-4)

    # Signal
    def model(t):
        output = (
            1.0 * np.sin(2 * np.pi * 1e3 * t)
            + 0.3 * np.sin(2 * np.pi * 2e3 * t)
            + 0.1 * np.sin(2 * np.pi * 3e3 * t)
        ) * np.exp(-1e3 * t)
        return output

    x = model(t)

    # CZT
    f_czt1, X_czt1 = czt.time2freq(t, x)

    # DFT
    f_dft1, X_dft1 = dft(t, x)

    # Truncate
    idx1, idx2 = 110, 180
    f_czt1, X_czt1 = f_czt1[idx1:idx2], X_czt1[idx1:idx2]
    f_dft1, X_dft1 = f_dft1[idx1:idx2], X_dft1[idx1:idx2]

    # Zoom CZT
    f_czt2, X_czt2 = czt.time2freq(t, x, f_czt1)

    # Zoom DFT
    f_dft2, X_dft2 = dft(t, x, f_dft1)

    # Plot for debugging purposes
    if debug:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.plot(f_czt1, np.abs(X_czt1))
        plt.plot(f_czt2, np.abs(X_czt2))
        plt.plot(f_dft1, np.abs(X_dft1))
        plt.plot(f_dft2, np.abs(X_dft2))
        plt.show()

    # All frequencies should be the same
    np.testing.assert_almost_equal(f_czt1, f_czt2, decimal=12)
    np.testing.assert_almost_equal(f_czt1, f_dft1, decimal=12)
    np.testing.assert_almost_equal(f_czt1, f_dft2, decimal=12)

    # Compare
    np.testing.assert_almost_equal(X_czt1, X_czt2, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_dft1, decimal=12)
    np.testing.assert_almost_equal(X_czt1, X_dft2, decimal=12)


def test_compare_czt_to_analytic_expression(debug=False):
    """Compare CZT to analytic expression."""

    # Create time-domain data
    t = np.arange(0, 50e-3 + 1e-10, 1e-5)

    # Signal
    def model(t):
        output = (
            1.0 * np.sin(2 * np.pi * 1e3 * t)
            + 0.3 * np.sin(2 * np.pi * 2e3 * t)
            + 0.1 * np.sin(2 * np.pi * 3e3 * t)
        ) * np.exp(-1e3 * t)
        return output

    x = model(t)

    # CZT
    f = np.arange(-20, 20 + 1e-10, 0.01) * 1e3
    _, X_czt = czt.time2freq(t, x, f)

    # Build frequency domain signal
    X1 = np.zeros_like(f, dtype=complex)
    idx = np.abs(f - 1e3).argmin()
    X1[idx] = 1 / 2j
    idx = np.abs(f + 1e3).argmin()
    X1[idx] = -1 / 2j
    idx = np.abs(f - 2e3).argmin()
    X1[idx] = 0.3 / 2j
    idx = np.abs(f + 2e3).argmin()
    X1[idx] = -0.3 / 2j
    idx = np.abs(f - 3e3).argmin()
    X1[idx] = 0.1 / 2j
    idx = np.abs(f + 3e3).argmin()
    X1[idx] = -0.1 / 2j
    X2 = 1 / (1e3 + 2j * np.pi * f)
    X = np.convolve(X1, X2)
    X = X[len(X) // 4 : -len(X) // 4 + 1]
    X *= 2 * (f[1] - f[0]) * len(t)

    # Truncate
    mask = (0 < f) & (f < 5e3)
    f, X, X_czt = f[mask], X[mask], X_czt[mask]

    # Plot for debugging purposes
    if debug:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Absolute")
        plt.plot(f / 1e3, np.abs(X_czt))
        plt.plot(f / 1e3, np.abs(X), "r--")
        plt.figure()
        plt.title("Real")
        plt.plot(f / 1e3, X_czt.real)
        plt.plot(f / 1e3, X.real, "r--")
        plt.figure()
        plt.title("Imaginary")
        plt.plot(f / 1e3, X_czt.imag)
        plt.plot(f / 1e3, X.imag, "r--")
        plt.show()

    # Compare
    np.testing.assert_allclose(X, X_czt, atol=0.02)


if __name__ == "__main__":

    test_compare_different_czt_methods()
    test_compare_czt_fft_dft()
    test_czt_to_iczt()
    test_time_to_freq_to_time()
    test_compare_iczt_idft()
    test_frequency_zoom()
    test_compare_czt_to_analytic_expression()
