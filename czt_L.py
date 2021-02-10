"""Chirp Z-transform.

Main reference:

    Sukhoy, V., Stoytchev, A. Generalizing the inverse FFT off the unit
    circle. Sci Rep 9, 14443 (2019).
    https://doi.org/10.1038/s41598-019-50234-9

"""

import numpy as np
from scipy.linalg import matmul_toeplitz
from scipy.signal import kaiser
from math import gcd
from warnings import warn

__all__ = [
    "czt",
    "iczt",
    "fourier",
    "inverse_fourier",
    "time2freq",
    "freq2time",
    "acft",
    "iacft",
]

# CZT TRANSFORM --------------------------------------------------------------


def czt(x, M=None, W=None, A=1.0):
    """Calculate Chirp Z-transform (CZT).

    Using an efficient algorithm. Solves in O(n log n) time.

    See algorithm 1 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        x (np.ndarray): input array
        M (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point

    Returns:
        np.ndarray: Chirp Z-transform

    """

    N = len(x)
    if M is None:
        M = N
    if W is None:
        W = np.exp(-2j * np.pi / M)

    # bugfix for A or W is an int (int**-np.array(dtype=int) raises an ValueError)
    A = complex(A)
    W = complex(W)

    k = np.arange(N)
    X = W ** (k ** 2 / 2) * A ** -k * x
    r = W ** (-(k ** 2) / 2)
    k = np.arange(M)
    c = W ** (-(k ** 2) / 2)
    X = matmul_toeplitz((c, r), X)
    for k in range(M):
        X[k] = W ** (k ** 2 / 2) * X[k]

    return X


def iczt(X, N=None, W=None, A=1.0):
    """Calculate inverse Chirp Z-transform (ICZT).

    Args:
        X (np.ndarray): input array
        N (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point

    Returns:
        np.ndarray: Inverse Chirp Z-transform

    """
    return np.conj(czt(np.conj(X), M=N, W=W, A=A)) / len(X)


# New FFTs with Fourier-Parameters
# Similar to
# https://reference.wolfram.com/language/ref/Fourier.html


def fourier(x, FourierParameters=(1, -1), check_coprime=False):
    """Calculate the Discrete Fourier transform of a list of complex numbers.

    Similar to https://reference.wolfram.com/language/ref/Fourier.html

    With FourierParameters = (a,b), this is equivalent to ::

        X[k] = 1/N**((1-a)/2) * np.sum(x*np.exp(2j * np.pi * b * k * np.arange(N)/N))

    The default FourierParameters = (1,-1) is equivalent to default scipy.fft.fft.

    Args:
        x (np.ndarray): input array
        FourierParameters: List of length 2
            Some common choices are
                (1,-1) : default, signal processing
                (0,1) : Mathematica
                (-1,1) : data analysis

    Returns:
        np.ndarray: Discrete Fourier Transform with given FourierParameters.
    """
    a, b = FourierParameters
    N = len(x)
    if check_coprime:
        try:
            divisor = gcd(abs(b), N)
            if divisor != 1:
                warn(
                    f"Array length N={N} and Fourier Parameter |b|={abs(b)} "
                    f"are not coprime, but have common divisor {divisor}. "
                    "Unique Inverse Fourier Transform cannot be ensured."
                )
        except TypeError:
            warn("check_coprime only works for integer FourierParameters.")
    W = np.exp(2.0j * np.pi * b / N)

    return N ** (-(1.0 - a) / 2.0) * czt(x, W=W)


def inverse_fourier(X, FourierParameters=(1, -1), check_coprime=False):
    """Calculate the Discrete Inverse Fourier transform of a list of complex numbers.

    Similar to https://reference.wolfram.com/language/ref/InverseFourier.html

    With FourierParameters = (a,b), this is equivalent to ::

        x[n] = 1/N**((1+a)/2) * np.sum(X*np.exp(2j * np.pi * b * n * np.arange(N)/N))

    The default FourierParameters = (1,-1) is equivalent to default scipy.fft.fft.

    Args:
        X (np.ndarray): input array
        FourierParameters: List of length 2
            Some common choices are
                (1,-1) : default, signal processing
                (0,1) : Mathematica
                (-1,1) : data analysis

    Returns:
        np.ndarray: Discrete Inverse Fourier Transform with given FourierParameters.

    Notes:
        Equivalent to a call to fourier with FourierParameters=(-a,-b).
        Error Messages relate to arguments of fourier.
    """
    a, b = FourierParameters
    return fourier(X, FourierParameters=(-a, -b), check_coprime=check_coprime)


# Continuous Fourier Transform approximation by DFT


def acft(t, x, f=None, FourierParameters=(0, -2 * np.pi)):
    """Convert signal from time-domain to frequency-domain.

    Approximated Continuous Fourier Transformation from discrete time signal to
    discrete frequncy signal.

    Similar to https://reference.wolfram.com/language/ref/FourierTransform.html
    but with discrete input.

    With FourierParameters = (a,b), this is (roughly) equivalent to ::

        X[k] = np.sqrt(abs(b)/(2*np.pi)**(1-a)) * np.sum(x*np.exp(1j * b * f[k] * t)

    The first and last entry of x are weighted with 1/2 to use trapezoidal rule,
    which in some cases improves the result.

    The default FourierParameters = (0,-2*np.pi) is for signal processing.

    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal
            If None, then numpy.fft.fftfreq(len(t),dt) will be used.
        FourierParameters: List of length 2
            Some common choices are
                (0,-2*np.pi) : default, signal processing (unitary, ordinary frequency)
                (0,1) : Mathematica, modern physics (unitary, angular frequency)
                (-1,1) : classical physics (non-unitary, angular frequency)

    Returns:
        np.ndarray: Approximated Fourier Transform
    """
    t = np.array(t, dtype=float)
    x = np.array(x, dtype=complex)
    if t.shape != x.shape:
        raise ValueError(
            f"t and x must have same shape, but have shapes {t.shape} and {x.shape}."
        )
    # calculate dt
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        raise ValueError("t has no constant time step.")
    dt = dt[0]
    # calculate df
    if f is None:
        f = np.fft.fftshift(np.fft.fftfreq(len(t), dt))
        df = f[1] - f[0]
    else:
        df = np.diff(f)
        if not np.allclose(df, df[0]):
            raise ValueError("f has no constant time step.")
        df = df[0]

    Nf = len(f)  # number of frequency points

    a, b = FourierParameters
    # Step
    W = np.exp(1j * b * dt * df)

    # Starting point
    A = np.exp(-1j * b * f.min() * dt)

    # Frequency-domain transform
    prefactor = dt * (abs(b) * (2 * np.pi) ** (a - 1.0)) ** 0.5
    phase = np.exp(1j * b * t[0] * f)
    freq_data = czt(x, Nf, W, A)

    # trapezoidal rule
    I0 = (
        -dt
        / 2
        * (x[0] * np.exp(1j * b * t[0] * f) + x[-1] * np.exp(1j * b * t[-1] * f))
    )

    return prefactor * (I0 + phase * freq_data)


def iacft(f, X, t=None, FourierParameters=(0, -2 * np.pi)):
    """Convert signal from frequency-domain to time-domain.

    Approximated Continuous Inverse Fourier Transformation from discrete
    frequency signal to discrete time signal.

    Similar to https://reference.wolfram.com/language/ref/InverseFourierTransform.html
    but with discrete input.

    With FourierParameters = (a,b), this is (roughly) equivalent to ::

        x[n] = np.sqrt(abs(b)/(2*np.pi)**(1+a)) * np.sum(-x*np.exp(1j * b * f * t[n])

    The first and last entry of x are weighted with 1/2 to use trapezoidal rule,
    which in some cases improves the result.

    The default FourierParameters = (0,-2*np.pi) is for signal processing.
    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal
            If None, then numpy.fft.fftfreq(len(t),dt) will be used.
        FourierParameters: List of length 2
            Some common choices are
                (0,-2*np.pi) : default, signal processing (unitary, ordinary frequency)
                (0,1) : Mathematica, modern physics (unitary, angular frequency)
                (-1,1) : classical physics (non-unitary, angular frequency)

    Returns:
        np.ndarray: Approximated Inverse Fourier Transform

    Notes:
        Equivalent to a call to acft with FourierParameters=(-a,-b).
        Error Messages relate to arguments of acft.
    """
    a, b = FourierParameters
    return acft(f, X, t, FourierParameters=(-a, -b))


# FREQ <--> TIME-DOMAIN CONVERSION -------------------------------------------


def time2freq(t, x, f=None, f_orig=None):
    """Convert signal from time-domain to frequency-domain.

    Args:
        t (np.ndarray): time
        x (np.ndarray): time-domain signal
        f (np.ndarray): frequency for output signal
        f_orig (np.ndarray): frequency sweep of the original signal, necessary
            for normalization if the new frequency sweep is different from the
            original

    Returns:
        np.ndarray: frequency-domain signal

    """

    # Input time array
    t1, t2 = t.min(), t.max()  # start / stop time
    dt = t[1] - t[0]  # time step
    Nt = len(t)  # number of time points
    Fs = 1 / dt  # sampling frequency

    # Output frequency array
    if f is None:
        f = np.linspace(-Fs / 2, Fs / 2, Nt)
    f1, f2 = f.min(), f.max()  # start / stop
    df = f[1] - f[0]  # frequency step
    bw = f2 - f1  # bandwidth
    Nf = len(f)  # number of frequency points

    # Correction factor (normalization)
    if f_orig is not None:
        k = 1 / (dt * (f_orig.max() - f_orig.min()))
    else:
        k = 1 / (dt * (f.max() - f.min()))

    # Step
    # W = np.exp(-2j * np.pi * bw / Nf / Fs)
    W = np.exp(-2j * np.pi * bw / (Nf - 1) / Fs)

    # Starting point
    A = np.exp(2j * np.pi * f1 / Fs)

    # Frequency-domain transform
    freq_data = czt(x, Nf, W, A)

    return f, freq_data / k


def freq2time(f, X, t=None, t_orig=None):
    """Convert signal from frequency-domain to time-domain.

    Args:
        f (np.ndarray): frequency
        X (np.ndarray): frequency-domain signal
        t (np.ndarray): time for output signal

    Returns:
        np.ndarray: time-domain signal

    """

    # Input frequency
    f1, f2 = f.min(), f.max()  # start / stop frequency
    df = f[1] - f[0]  # frequency step
    bw = f2 - f1  # bandwidth
    fc = (f1 + f2) / 2  # center frequency
    Nf = len(f)  # number of frequency points
    t_alias = 1 / df  # alias-free interval

    # Output time
    if t is None:
        t = np.linspace(0, t_alias, Nf)
    t1, t2 = t.min(), t.max()  # start / stop time
    dt = t[1] - t[0]  # time step
    Nt = len(t)  # number of time points
    Fs = 1 / dt  # sampling frequency

    # Correction factor (normalization)
    if t_orig is not None:
        k = (t.max() - t.min()) / df / (t_orig.max() - t_orig.min()) ** 2
    else:
        k = 1

    # Step
    # W = np.exp(-2j * np.pi * bw / Nf / Fs)
    W = np.exp(-2j * np.pi * bw / (Nf - 1) / Fs)

    # Starting point
    A = np.exp(2j * np.pi * t1 / t_alias)

    # Time-domain transform
    time_data = iczt(X, N=Nt, W=W, A=A)

    # Phase shift
    n = np.arange(len(time_data))
    phase = np.exp(2j * np.pi * f1 * n * dt)
    # phase = np.exp(2j * np.pi * (f1 + df / 2) * n * dt)

    return t, time_data * phase / k


# WINDOW ---------------------------------------------------------------------


def get_window(f, f_start=None, f_stop=None, beta=6):
    """Get Kaiser-Bessel window.

    See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.kaiser.html

    Args:
        f (np.ndarray): frequency
        f_start (float): start frequency
        f_stop (float): stop frequency
        beta (float): KB parameter for Kaiser Bessel filter

    Returns:
        np.ndarray: windowed S-parameter

    """

    # Frequency limits
    if f_start is None:
        f_start = f.min()
    if f_stop is None:
        f_stop = f.max()

    # Get corresponding indices
    idx_start = np.abs(f - f_start).argmin()
    idx_stop = np.abs(f - f_stop).argmin()
    idx_span = idx_stop - idx_start

    # Make window
    window = np.r_[
        np.zeros(idx_start), kaiser(idx_span, beta), np.zeros(len(f) - idx_stop)
    ]

    return window


def window(f, s, f_start=None, f_stop=None, beta=6, normalize=True):
    """Window frequency-domain data using Kaiser-Bessel filter.

    Args:
        f (np.ndarray): frequency
        s (np.ndarray): S-parameter
        f_start (float): start frequency
        f_stop (float): stop frequency
        beta (float): KB parameter for Kaiser Bessel filter

    Returns:
        np.ndarray: windowed S-parameter

    """

    _window = get_window(f, f_start=f_start, f_stop=f_stop, beta=beta)

    # Normalize
    if normalize:
        w0 = np.mean(_window)
    else:
        w0 = 1

    return s * _window / w0
