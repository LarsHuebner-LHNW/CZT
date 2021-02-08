"""Chirp Z-transform.

Main reference:

    Sukhoy, V., Stoytchev, A. Generalizing the inverse FFT off the unit
    circle. Sci Rep 9, 14443 (2019).
    https://doi.org/10.1038/s41598-019-50234-9

"""

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import kaiser


# CZT TRANSFORM --------------------------------------------------------------


def czt(x, M=None, W=None, A=1.0, simple=False, t_method="ce"):
    """Calculate Chirp Z-transform (CZT).

    Using an efficient algorithm. Solves in O(n log n) time.

    See algorithm 1 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        x (np.ndarray): input array
        M (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point
        simple (bool): use simple algorithm?
        t_method (str): Toeplitz matrix multiplication method. 'ce' for
            circulant embedding, 'pd' for Pustylnikov's decomposition, 'mm'
            for simple matrix multiplication

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

    if simple:
        k = np.arange(M)
        X = np.zeros(M, dtype=complex)
        z = A * W ** -k
        for n in range(N):
            X += x[n] * z ** -n
        return X

    k = np.arange(N)
    X = W ** (k ** 2 / 2) * A ** -k * x
    r = W ** (-(k ** 2) / 2)
    k = np.arange(M)
    c = W ** (-(k ** 2) / 2)
    if t_method.lower() == "ce":
        X = _toeplitz_mult_ce(r, c, x)
    elif t_method.lower() == "pd":
        X = _toeplitz_mult_pd(r, c, X)
    elif t_method.lower() == "mm":
        X = np.matmul(toeplitz(r, c), X)
    else:
        print("t_method not recognized.")
        raise ValueError
    for k in range(M):
        X[k] = W ** (k ** 2 / 2) * X[k]

    return X


def iczt(X, N=None, W=None, A=1.0, t_method="ce"):
    """Calculate inverse Chirp Z-transform (ICZT).

    Args:
        X (np.ndarray): input array
        N (int): length of output array
        W (complex): complex ratio between points
        A (complex): complex starting point
        t_method (str): Toeplitz matrix multiplication method. 'ce' for
            circulant embedding, 'pd' for Pustylnikov's decomposition, 'mm'
            for simple matrix multiplication

    Returns:
        np.ndarray: Inverse Chirp Z-transform

    """

    M = len(X)
    if N is None:
        N = M
    if W is None:
        W = np.exp(-2j * np.pi / M)

    return np.conj(czt(np.conj(X), M=N, W=W, A=A, t_method=t_method)) / M


# OTHER TRANSFORMS -----------------------------------------------------------


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


# HELPER FUNCTIONS -----------------------------------------------------------


def _toeplitz_mult_ce(r, c, x):
    """Multiply Toeplitz matrix by vector using circulant embedding.

    "Compute the product y = Tx of a Toeplitz matrix T and a vector x, where T
    is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and its
    first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0]."

    See algorithm S1 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix

    Returns:
        np.ndarray: product of Toeplitz matrix and vector x

    """
    N = len(r)
    M = len(c)
    assert r[0] == c[0]
    assert len(x) == N
    n = int(2 ** np.ceil(np.log2(M + N - 1)))
    assert n >= M
    assert n >= N
    chat = np.r_[c, np.zeros(n - (M + N - 1)), r[-(N - 1) :][::-1]]
    xhat = _zero_pad(x, n)
    yhat = _circulant_multiply(chat, xhat)
    y = yhat[:M]
    return y


def _toeplitz_mult_pd(r, c, x):
    """Multiply Toeplitz matrix by vector using Pustylnikov's decomposition.

    Compute the product y = Tx of a Toeplitz matrix T and a vector x, where T
    is specified by its first row r = (r[0], r[1], r[2],...,r[N-1]) and its
    first column c = (c[0], c[1], c[2],...,c[M-1]), where r[0] = c[0].

    See algorithm S3 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        r (np.ndarray): first row of Toeplitz matrix
        c (np.ndarray): first column of Toeplitz matrix
        x (np.ndarray): vector to multiply the Toeplitz matrix

    Returns:
        np.ndarray: product of Toeplitz matrix and vector x

    """
    N = len(r)
    M = len(c)
    assert r[0] == c[0]
    assert len(x) == N
    n = int(2 ** np.ceil(np.log2(M + N - 1)))
    if N != n:
        r = _zero_pad(r, n)
        x = _zero_pad(x, n)
    if M != n:
        c = _zero_pad(c, n)
    c1 = np.empty(n, dtype=complex)
    c2 = np.empty(n, dtype=complex)
    c1[0] = 0.5 * c[0]
    c2[0] = 0.5 * c[0]
    for k in range(1, n):
        c1[k] = 0.5 * (c[k] + r[n - k])
        c2[k] = 0.5 * (c[k] - r[n - k])
    y1 = _circulant_multiply(c1, x)
    y2 = _skew_circulant_multiply(c2, x)
    y = y1[:M] + y2[:M]
    return y


def _zero_pad(x, n):
    """Zero pad an array x to length n by appending zeros.

    See algorithm S2 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        x (np.ndarray): array x
        n (int): length of output array

    Returns:
        np.ndarray: array x with padding

    """
    m = len(x)
    assert m <= n
    xhat = np.zeros(n, dtype=complex)
    xhat[:m] = x
    return xhat


def _circulant_multiply(c, x, f_method="std"):
    """Multiply a circulat matrix by a vector.

    Compute the product y = Gx of a circulat matrix G and a vector x, where G
    is generated by its first column c=(c[0], c[1],...,c[n-1]).

    Runs in O(n log n) time.

    See algorithm S4 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        c (np.ndarray): first column of circulant matrix G
        x (np.ndarray): vector x
    Returns:
        np.ndarray: product Gx

    """
    n = len(c)
    assert len(x) == n
    C = np.fft.fft(c)
    X = np.fft.fft(x)
    Y = np.empty(n, dtype=complex)
    for k in range(n):
        Y[k] = C[k] * X[k]
    y = np.fft.ifft(Y)
    return y


def _skew_circulant_multiply(c, x):
    """Multiply a skew-circulant matrix by a vector.

    Runs in O(n log n) time.

    See algorithm S7 in Sukhoy & Stoytchev 2019 (full reference in README).

    Args:
        c (np.ndarray): first column of skew-circulant matrix G
        x (np.ndarray): vector x

    Returns:
        np.ndarray: product Gx

    """
    n = len(c)
    assert len(x) == n
    chat = np.empty(n, dtype=complex)
    xhat = np.empty(n, dtype=complex)
    for k in range(n):
        chat[k] = c[k] * np.exp(-1j * k * np.pi / n)
        xhat[k] = x[k] * np.exp(-1j * k * np.pi / n)
    # k = np.arange(n, dtype=complex)
    # chat = c * np.exp(-1j * k * np.pi / n)
    # xhat = c * np.exp(-1j * k * np.pi / n)
    y = _circulant_multiply(chat, xhat)
    for k in range(n):
        y[k] = y[k] * np.exp(1j * k * np.pi / n)
    return y
