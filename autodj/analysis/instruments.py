"""What is actually *playing* in each bar, so transitions can avoid collisions.

Beatmatching gets the two tracks into the same time. It does nothing about
whether they should be sounding together at all. Two tracks can be perfectly
locked to one grid, in compatible keys, and still sound like a mess -- because
both of them are singing.

That is the difference between "aligned" and "seamless". A professional does
not overlap two choruses; they overlap A's outro *groove* with B's intro
*groove*, and let exactly one lead voice be present at any moment. This module
measures which roles are occupied in each bar so the renderer can choose an
overlap where the two tracks fit together instead of competing.

Two backends, same output shape:

stems     exact. RMS of each separated source per bar. Requires the track to
          have been separated (~60 s on the GPU), so it is used when the stem
          cache already has it.
spectral  free. Derived from the spectrogram we already computed during
          analysis. Less exact, but it only has to answer "is there a lead
          voice in this bar", and for that a mid-band-dominance measure is
          good enough.
"""

import numpy as np

from .. import audio as audio_mod
from .. import spectral
from . import track as track_mod

ROLES = ("vocals", "drums", "bass", "other")

# A lead voice lives here. Below is bass and kick body, above is mostly air and
# cymbals -- neither tells you whether someone is singing.
VOCAL_BAND = (300.0, 3500.0)


def _bar_frames(n_frames, meta, sample_rate, hop=spectral.HOP):
    """Frame index of each bar line, from the fitted grid."""
    period = 4 * 60.0 / meta["bpm"]
    t0 = meta["first_downbeat"]
    times = np.arange(t0, t0 + period * (meta["n_bars"] + 1), period)
    idx = np.round(times * sample_rate / hop).astype(int)
    return idx[(idx >= 0) & (idx < n_frames)]


def from_spectrogram(mag, sample_rate, meta):
    """Per-bar vocal likelihood, with no separation required.

    Uses the ratio of vocal-band power to total power. It is a proxy, not a
    detector: a bright synth lead reads much like a voice. That is acceptable
    here, because for transition planning a synth lead and a vocal are the same
    problem -- both are a lead voice, and two of them at once is the thing to
    avoid.
    """
    edges = _bar_frames(mag.shape[1], meta, sample_rate)
    freqs = spectral.bin_freqs(sample_rate, spectral.N_FFT)
    rows = (freqs >= VOCAL_BAND[0]) & (freqs < VOCAL_BAND[1])
    power = mag.astype(np.float64) ** 2
    out = []
    for i in range(len(edges) - 1):
        seg = power[:, edges[i]:edges[i + 1]]
        if seg.size == 0:
            out.append(0.0)
            continue
        out.append(float(seg[rows].sum() / (seg.sum() + 1e-20)))
    return np.asarray(out)


def from_stems(path, meta, sample_rate=None):
    """Per-bar RMS of each separated source. Exact, but needs the stem cache."""
    from ..stems import mashup, separate
    if not separate.is_cached(path):
        return None
    sample_rate = sample_rate or audio_mod.RENDER_RATE
    stems = mashup.load_stems(path, sample_rate=sample_rate)
    period = 4 * 60.0 / meta["bpm"]
    t0 = meta["first_downbeat"]
    out = {}
    for role, a in stems.items():
        mono = a.mean(axis=0)
        vals = []
        for b in range(meta["n_bars"]):
            s = int((t0 + b * period) * sample_rate)
            e = int((t0 + (b + 1) * period) * sample_rate)
            seg = mono[max(0, s):max(0, e)]
            vals.append(float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0)
        out[role] = np.asarray(vals)
    return out


def vocal_activity(path, meta, mag=None, sample_rate=None, prefer_stems=True):
    """Per-bar 0..1 measure of "is there a lead voice in this bar".

    Normalised within the track, deliberately. The question is not how loud
    this track's vocal is against another track's -- it is which of *this*
    track's bars have the voice in them, so a quiet verse still reads as
    occupied.
    """
    if prefer_stems:
        st = from_stems(path, meta, sample_rate)
        if st is not None:
            v = st["vocals"]
            rest = st["drums"] + st["bass"] + st["other"] + 1e-12
            score = v / rest
            return _unit(score), "stems"
    if mag is None:
        a, sr = audio_mod.load(path, audio_mod.ANALYSIS_RATE, mono=True)
        mag = spectral.magnitude(a[0])
        sample_rate = sr
    return _unit(from_spectrogram(mag, sample_rate or audio_mod.ANALYSIS_RATE,
                                  meta)), "spectral"


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def collision(out_act, out_start, in_act, in_start, bars):
    """How much the two tracks fight over the lead voice across an overlap.

    The product, not the sum: a bar only costs something when *both* sides have
    a voice in it. One track singing over the other's groove is exactly what a
    good transition sounds like, and must score zero.
    """
    a = _window(out_act, out_start, bars)
    b = _window(in_act, in_start, bars)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(a[:n] * b[:n]))


def _window(act, start, bars):
    act = np.asarray(act)
    if act.size == 0:
        return np.zeros(bars)
    idx = np.arange(int(start), int(start) + int(bars))
    idx = np.clip(idx, 0, len(act) - 1)
    return act[idx]


def best_entry(out_act, out_start, in_act, candidates, bars):
    """Pick the incoming track's entry bar that collides least.

    `candidates` must all be phrase-aligned -- this chooses between musically
    legal entry points, it never invents one. Ties keep the earliest, which is
    the one the structural analysis already preferred.
    """
    best, best_score = None, None
    for c in candidates:
        s = collision(out_act, out_start, in_act, c, bars)
        if best_score is None or s < best_score - 1e-9:
            best, best_score = c, s
    return best, (best_score or 0.0)
