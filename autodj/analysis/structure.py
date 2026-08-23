"""Structural segmentation and cue points.

Finds where the track changes character, snaps those boundaries to phrase
lines, and derives the four positions a DJ actually needs: where the intro
ends, where the outro starts, where the drops are, and where the breakdowns are.
"""

import numpy as np
from scipy.signal import find_peaks

PHRASE = 8          # bars; dance music is built in 8- and 16-bar phrases
MIN_SEGMENT = 8     # bars


def similarity(bars):
    """Cosine self-similarity between every pair of bars."""
    if len(bars) < 2:
        return np.zeros((len(bars), len(bars)))
    x = (bars - bars.mean(axis=0)) / (bars.std(axis=0) + 1e-9)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x @ x.T


def _checkerboard(size):
    """Foote's novelty kernel: +1 on the diagonal blocks, -1 off them.

    Slid down the diagonal of the similarity matrix, it responds where bars
    before a point resemble each other, bars after resemble each other, but the
    two groups do not resemble each other -- which is exactly a section change.
    """
    k = np.ones((2 * size, 2 * size))
    k[:size, size:] = -1.0
    k[size:, :size] = -1.0
    taper = np.outer(np.hanning(2 * size), np.hanning(2 * size))
    return k * taper


def novelty(ssm, size=4):
    """How strongly each bar looks like a section boundary."""
    n = ssm.shape[0]
    if n < 2 * size + 1:
        return np.zeros(n)
    kern = _checkerboard(size)
    padded = np.pad(ssm, size, mode="edge")
    out = np.array([float((padded[i:i + 2 * size, i:i + 2 * size] * kern).sum())
                    for i in range(n)])
    out = np.maximum(out, 0.0)
    return out / (out.max() + 1e-12)


def boundaries(nov, n_bars, phrase=PHRASE, min_gap=MIN_SEGMENT):
    """Section boundaries in bars, snapped to phrase lines."""
    if n_bars < 2 * phrase:
        return [0, n_bars]
    peaks, _ = find_peaks(nov, height=0.25, distance=max(2, min_gap // 2))
    snapped = {0, n_bars}
    for p in peaks:
        b = int(round(p / phrase) * phrase)
        if 0 < b < n_bars:
            snapped.add(b)
    out = sorted(snapped)
    # Drop boundaries that would create a segment shorter than min_gap.
    merged = [out[0]]
    for b in out[1:]:
        if b - merged[-1] >= min_gap or b == n_bars:
            merged.append(b)
    if len(merged) > 1 and merged[-1] != n_bars:
        merged.append(n_bars)
    return merged


def _runs(mask):
    """(start, stop) index pairs for each run of True."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def cue_points(curve, phrase=PHRASE):
    """The positions a DJ mixes on, all in bars.

    `full` means the track is at working energy -- drums and bass in. The intro
    is everything before the first sustained full run; the outro is everything
    after the last one.
    """
    n = len(curve)
    if n < 4:
        return {"intro_bars": 0, "outro_start_bar": max(0, n - 1),
                "first_full_bar": 0, "last_full_bar": max(0, n - 1),
                "drop_bars": [], "breakdown_bars": []}

    ref = float(np.percentile(curve, 75))
    full = curve >= 0.62 * ref
    runs = [(a, b) for a, b in _runs(full) if b - a >= 4]
    if not runs:
        runs = [(0, n)]

    first_full = runs[0][0]
    last_full = runs[-1][1] - 1

    # Snap the mix-in point *down* to a phrase line: better to come in early on
    # a phrase than late in the middle of one.
    intro_bars = int(first_full // phrase * phrase)
    outro_start = int(np.ceil((last_full + 1) / phrase) * phrase)
    outro_start = min(outro_start, n - 1)

    # A drop is a large jump in energy landing on a phrase line.
    drops = []
    for b in range(phrase, n - 1, phrase):
        before = curve[max(0, b - phrase):b].mean()
        after = curve[b:min(n, b + phrase)].mean()
        if after - before > 0.18:
            drops.append(b)

    # A breakdown is a sustained dip in the middle of the track.
    breakdowns = [a for a, b in _runs(~full) if b - a >= 4 and a > first_full + 4
                  and b < last_full]

    return {
        "intro_bars": intro_bars,
        "outro_start_bar": outro_start,
        "first_full_bar": int(first_full),
        "last_full_bar": int(last_full),
        "drop_bars": [int(d) for d in drops],
        "breakdown_bars": [int(b) for b in breakdowns],
    }


def label(curve, bounds):
    """Name each segment from its energy relative to the whole track."""
    out = []
    if len(curve) == 0:
        return out
    peak = float(curve.max()) + 1e-12
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = curve[a:min(b, len(curve))]
        if len(seg) == 0:
            continue
        rel = float(seg.mean()) / peak
        if a == 0 and rel < 0.7:
            name = "intro"
        elif b >= len(curve) - 1 and rel < 0.7:
            name = "outro"
        elif rel > 0.88:
            name = "drop"
        elif rel < 0.55:
            name = "breakdown"
        else:
            name = "build" if len(out) and out[-1]["name"] == "breakdown" else "verse"
        out.append({"name": name, "start_bar": int(a), "end_bar": int(b),
                    "energy": round(rel, 3)})
    return out


def analyse(bars, curve):
    """Full structural analysis from per-bar features."""
    ssm = similarity(bars)
    nov = novelty(ssm)
    bounds = boundaries(nov, len(curve))
    segments = label(curve, bounds)
    cues = cue_points(curve)
    cues["segments"] = segments
    cues["n_segments"] = len(segments)
    return cues
