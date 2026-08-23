"""Automation curves.

Every transition in this project is the same operation -- multiply a track by a
gain that changes over time -- and differs only in the *shape* of the curves.
Building them here, once, means a new transition is a few lines rather than a
new signal path.
"""

import numpy as np


def ramp(n, start=0.0, end=1.0, shape="linear", power=2.0):
    """A curve of `n` samples going from `start` to `end`."""
    if n <= 0:
        return np.zeros(0)
    t = np.linspace(0.0, 1.0, int(n))
    if shape == "linear":
        f = t
    elif shape == "ease_in":
        f = t ** power                      # slow start, fast finish
    elif shape == "ease_out":
        f = 1.0 - (1.0 - t) ** power        # fast start, slow finish
    elif shape == "s_curve":
        f = t * t * (3.0 - 2.0 * t)         # smoothstep: flat at both ends
    elif shape == "equal_power_in":
        f = np.sin(t * np.pi / 2)
    elif shape == "equal_power_out":
        f = np.cos(t * np.pi / 2)
    else:
        raise ValueError(f"unknown curve shape: {shape}")
    return start + (end - start) * f


def equal_power(n):
    """A crossfade pair whose summed *power* stays constant.

    A linear crossfade dips in the middle: at the halfway point both tracks are
    at 0.5 amplitude, so combined power is 0.5, about 3 dB down -- audible as a
    sag right where the mix should feel strongest. sin/cos keeps
    out^2 + in^2 == 1 throughout.
    """
    t = np.linspace(0.0, 1.0, int(max(n, 1)))
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)


def hold(n, value=1.0):
    return np.full(int(max(n, 0)), float(value))


def chain(*segments):
    """Glue curve segments end to end."""
    parts = [np.asarray(s, dtype=np.float64) for s in segments if len(s)]
    return np.concatenate(parts) if parts else np.zeros(0)


def fit(curve, n):
    """Resample a curve to exactly n samples (curves are defined in bars)."""
    curve = np.asarray(curve, dtype=np.float64)
    n = int(n)
    if len(curve) == n:
        return curve
    if len(curve) == 0:
        return np.zeros(n)
    if len(curve) == 1:
        return np.full(n, curve[0])
    return np.interp(np.linspace(0, len(curve) - 1, n),
                     np.arange(len(curve)), curve)


def db(gain_db):
    """Decibels to linear gain."""
    return float(10.0 ** (np.asarray(gain_db, dtype=np.float64) / 20.0)) \
        if np.isscalar(gain_db) else 10.0 ** (np.asarray(gain_db) / 20.0)


def by_bar(bars, points, samples_per_bar, shape="linear"):
    """Build a curve from (bar, value) breakpoints.

    Lets a transition be written the way a DJ describes it -- "bass out at bar
    4, back in at bar 12" -- instead of in sample indices.
    """
    total = int(round(bars * samples_per_bar))
    if not points:
        return np.ones(total)
    pts = sorted(points)
    out = np.empty(0)
    if pts[0][0] > 0:
        out = hold(int(pts[0][0] * samples_per_bar), pts[0][1])
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        out = chain(out, ramp(int((b1 - b0) * samples_per_bar), v0, v1, shape))
    tail = total - len(out)
    if tail > 0:
        out = chain(out, hold(tail, pts[-1][1]))
    return fit(out, total)
