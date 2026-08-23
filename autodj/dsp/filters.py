"""The 3-band DJ EQ, and time-varying filter sweeps.

The band split is built so that low + mid + high reconstructs the input
*exactly*. That property is what makes an EQ transparent at unity gain, and it
is not something a naive three-filter bank gives you.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

CROSSOVER_LOW = 200.0     # kick and bassline live below this
CROSSOVER_HIGH = 2000.0   # hats and air live above this
ORDER = 4


def _sos(cut, sample_rate, btype, order=ORDER):
    nyq = sample_rate / 2.0
    return butter(order, np.clip(cut / nyq, 1e-5, 0.999), btype=btype, output="sos")


def split(audio, sample_rate, low_cut=CROSSOVER_LOW, high_cut=CROSSOVER_HIGH):
    """Split into (low, mid, high) that sum back to the original exactly.

    Note what mid is: not a bandpass, but *everything the other two did not
    take*. Three independent filters would each impose their own phase shift
    and the sum would not reconstruct -- you would hear comb filtering at the
    crossovers even with all gains at 1.0. Defining mid by subtraction makes
    perfect reconstruction true by construction.

    `sosfiltfilt` runs the filter forwards and backwards, which cancels phase
    shift entirely. That is a luxury only offline rendering can afford, and it
    is the reason our EQ sounds cleaner than a live DJ mixer's.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    low = sosfiltfilt(_sos(low_cut, sample_rate, "low"), a, axis=-1)
    high = sosfiltfilt(_sos(high_cut, sample_rate, "high"), a, axis=-1)
    return low, a - low - high, high


def apply_eq(audio, sample_rate, g_low=1.0, g_mid=1.0, g_high=1.0,
             bands=None):
    """Apply three band gains. Each gain may be a scalar or a per-sample curve."""
    low, mid, high = bands if bands is not None else split(audio, sample_rate)
    return (low * np.asarray(g_low) + mid * np.asarray(g_mid)
            + high * np.asarray(g_high)).astype(np.float32)


def sweep(audio, sample_rate, cutoffs, btype="low", resonance=0.7, blocks=None):
    """Filter with a cutoff that moves over time.

    A biquad's coefficients depend on its cutoff, so a moving cutoff means the
    filter is no longer time-invariant and cannot be applied in one pass. The
    standard offline answer: cut into overlapping blocks, filter each at its own
    fixed cutoff, and crossfade with a Hann window so the coefficient changes
    are never heard as clicks.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    n = a.shape[1]
    cutoffs = np.asarray(cutoffs, dtype=np.float64)
    if cutoffs.size == 1:
        return sosfiltfilt(_sos(float(cutoffs), sample_rate, btype), a, axis=-1)

    blocks = blocks or max(8, int(n / (sample_rate * 0.05)))   # ~50 ms blocks
    size = int(np.ceil(n / blocks)) * 2                        # 50% overlap
    hop = size // 2
    win = np.hanning(size)

    out = np.zeros_like(a)
    norm = np.zeros(n)
    grid = np.linspace(0, 1, len(cutoffs))
    for start in range(0, n, hop):
        stop = min(start + size, n)
        seg = a[:, start:stop]
        if seg.shape[1] < 8:
            break
        pos = (start + seg.shape[1] / 2) / max(1, n)
        cut = float(np.interp(pos, grid, cutoffs))
        # Filter a padded slice so each block's edges are not filter transients.
        pad = min(size, start)
        wide = a[:, start - pad:stop]
        filt = sosfiltfilt(_sos(cut, sample_rate, btype), wide, axis=-1)[:, pad:]
        w = win[:seg.shape[1]] if seg.shape[1] < size else win
        out[:, start:stop] += filt * w
        norm[start:stop] += w
    return (out / np.maximum(norm, 1e-9)).astype(np.float32)


def _biquad(b0, b1, b2, a0, a1, a2):
    """Normalise RBJ cookbook coefficients into one second-order section."""
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def shelf_sos(sample_rate, freq, gain_db, kind="high", q=0.707):
    """Shelving filter (RBJ cookbook).

    A shelf lifts or cuts everything above (or below) a corner frequency,
    unlike our band split which carves fixed regions. It is the right shape for
    'add air' or 'tame the sub' because the change is gradual and has no
    audible edge.
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * np.clip(freq, 1.0, sample_rate / 2 - 1) / sample_rate
    cos_w, sin_w = np.cos(w0), np.sin(w0)
    alpha = sin_w / (2.0 * q)
    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha
    if kind == "low":
        return _biquad(
            A * ((A + 1) - (A - 1) * cos_w + two_sqrtA_alpha),
            2 * A * ((A - 1) - (A + 1) * cos_w),
            A * ((A + 1) - (A - 1) * cos_w - two_sqrtA_alpha),
            (A + 1) + (A - 1) * cos_w + two_sqrtA_alpha,
            -2 * ((A - 1) + (A + 1) * cos_w),
            (A + 1) + (A - 1) * cos_w - two_sqrtA_alpha)
    return _biquad(
        A * ((A + 1) + (A - 1) * cos_w + two_sqrtA_alpha),
        -2 * A * ((A - 1) + (A + 1) * cos_w),
        A * ((A + 1) + (A - 1) * cos_w - two_sqrtA_alpha),
        (A + 1) - (A - 1) * cos_w + two_sqrtA_alpha,
        2 * ((A - 1) - (A + 1) * cos_w),
        (A + 1) - (A - 1) * cos_w - two_sqrtA_alpha)


def peaking_sos(sample_rate, freq, gain_db, q=1.0):
    """Parametric bell: boost or cut a band around `freq`."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * np.clip(freq, 1.0, sample_rate / 2 - 1) / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w = np.cos(w0)
    return _biquad(1 + alpha * A, -2 * cos_w, 1 - alpha * A,
                   1 + alpha / A, -2 * cos_w, 1 - alpha / A)


def apply_static_eq(audio, sample_rate, bands):
    """Apply a mastering `eq_bands` block. Skips anything set to unity.

    Every gain is designed at HALF the requested value. `sosfiltfilt` runs each
    filter forwards and backwards to cancel phase, which means the magnitude
    response is applied twice -- a shelf designed for +2.5 dB measured +4.48 dB
    until this was corrected. Zero phase is worth having; you just have to pay
    for it by halving the design gain.
    """
    a = np.atleast_2d(audio).astype(np.float64)
    stages = []
    half = 0.5

    hpf = bands.get("high_pass_filter", {})
    if hpf.get("enabled") and hpf.get("frequency_hz", 0) > 10:
        order = max(2, int(hpf.get("slope_db_oct", 12) // 6))
        stages.append(_sos(hpf["frequency_hz"], sample_rate, "high", order))

    ls = bands.get("low_shelf", {})
    if abs(ls.get("gain_db", 0.0)) > 0.05:
        stages.append(shelf_sos(sample_rate, ls["frequency_hz"],
                                ls["gain_db"] * half, "low",
                                ls.get("q_factor", 0.7)))

    pm = bands.get("parametric_mid", {})
    if abs(pm.get("gain_db", 0.0)) > 0.05:
        stages.append(peaking_sos(sample_rate, pm["frequency_hz"],
                                  pm["gain_db"] * half,
                                  pm.get("q_factor", 1.2)))

    hs = bands.get("high_shelf", {})
    if abs(hs.get("gain_db", 0.0)) > 0.05:
        stages.append(shelf_sos(sample_rate, hs["frequency_hz"],
                                hs["gain_db"] * half, "high",
                                hs.get("q_factor", 0.7)))

    lpf = bands.get("low_pass_filter", {})
    if lpf.get("enabled") and lpf.get("frequency_hz", 0) < sample_rate / 2 - 500:
        order = max(1, int(lpf.get("slope_db_oct", 6) // 6))
        stages.append(_sos(lpf["frequency_hz"], sample_rate, "low", order))

    for sos in stages:
        a = sosfiltfilt(sos, a, axis=-1)
    return a.astype(np.float32)


def highpass(audio, sample_rate, cut):
    return sosfiltfilt(_sos(cut, sample_rate, "high"),
                       np.atleast_2d(audio), axis=-1).astype(np.float32)


def lowpass(audio, sample_rate, cut):
    return sosfiltfilt(_sos(cut, sample_rate, "low"),
                       np.atleast_2d(audio), axis=-1).astype(np.float32)
