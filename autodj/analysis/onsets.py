"""Onset detection: turning the spectrogram into 'something happened here'."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .. import spectral


def flux(mag, sample_rate, band=None, n_fft=spectral.N_FFT):
    """Spectral flux: the positive frame-to-frame change in log magnitude.

    Only *increases* count. A note ending is not an onset; a note starting is.
    """
    if band is not None:
        freqs = spectral.bin_freqs(sample_rate, n_fft)
        rows = (freqs >= band[0]) & (freqs < band[1])
        mag = mag[rows]

    # Log compression: matches hearing, and stops the loud chorus from
    # drowning out the quiet intro when we normalise later.
    logmag = np.log1p(1000.0 * mag)
    diff = np.diff(logmag, axis=1)
    diff = np.maximum(diff, 0.0)
    env = diff.sum(axis=0)
    # np.diff loses one frame; pad the front so env[i] lines up with mag[:, i].
    return np.concatenate([[0.0], env])


def normalise(env, smooth_frames=0):
    """Remove the slow-moving floor and scale to a 0..1 peak."""
    env = np.asarray(env, dtype=np.float64)
    if smooth_frames > 1:
        env = uniform_filter1d(env, int(smooth_frames))
    # Subtracting a *local* median kills the drifting baseline without
    # flattening genuine dynamics the way a global mean would.
    floor = uniform_filter1d(env, max(3, int(len(env) * 0.02)))
    env = np.maximum(env - floor, 0.0)
    peak = float(env.max())
    return env / peak if peak > 0 else env


def peaks(env, sample_rate, hop=spectral.HOP, min_gap_s=0.05, threshold=0.15):
    """Frame indices of onset peaks."""
    if env.max() <= 0:
        return np.array([], dtype=int)
    distance = max(1, int(round(min_gap_s * sample_rate / hop)))
    idx, _ = find_peaks(env, height=threshold * float(env.max()), distance=distance)
    return idx


def envelope(mag, sample_rate, band=None, smooth_frames=0):
    """The standard onset envelope used everywhere else in the project."""
    return normalise(flux(mag, sample_rate, band), smooth_frames)
