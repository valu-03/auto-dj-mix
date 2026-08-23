"""Per-bar energy and timbre.

Everything downstream reasons in *bars*, not seconds. A bar is the musical
unit a DJ actually thinks in, and once the beatgrid exists we can average any
frame-level feature over each bar and get a compact, tempo-independent
description of how the track evolves.
"""

import numpy as np

from .. import spectral

FEATURE_NAMES = ("rms", "low", "mid", "high", "centroid")

# Bands used for tonal profiling rather than for mixing.
SUB_BAND = (20.0, 60.0)        # what a club subwoofer reproduces
PRESENCE_BAND = (1000.0, 5000.0)   # vocal intelligibility
HARSH_BAND = (2000.0, 4000.0)      # where ear fatigue lives
AIR_BAND = (8000.0, 11000.0)       # "modern" top end (capped by Nyquist)


def tonal_balance(mag, sample_rate):
    """Fractions of total energy in the bands a mastering engineer reaches for."""
    total = spectral.band_power(mag, sample_rate, (20.0, 11000.0)) + 1e-12

    def frac(band):
        return spectral.band_power(mag, sample_rate, band) / total

    presence = frac(PRESENCE_BAND)
    harsh = frac(HARSH_BAND)
    return {
        "sub_ratio": round(frac(SUB_BAND), 5),
        "mid_ratio": round(frac(spectral.MID_BAND), 5),
        "presence_ratio": round(presence, 5),
        # Harshness is the 2-4 kHz share *relative to* the presence band it sits
        # inside. A track can be bright without being harsh; what fatigues the
        # ear is 2-4 kHz dominating the rest of the presence range.
        "harshness": round(harsh / (presence + 1e-9), 4),
        "air_ratio": round(frac(AIR_BAND), 5),
    }


def frame_features(mag, sample_rate):
    """Frame-level features, all shaped (n_frames,)."""
    rms = np.sqrt(np.mean(mag ** 2, axis=0))
    low = spectral.band_energy(mag, sample_rate, spectral.LOW_BAND)
    mid = spectral.band_energy(mag, sample_rate, spectral.MID_BAND)
    high = spectral.band_energy(mag, sample_rate, spectral.HIGH_BAND)

    # Spectral centroid: the "centre of mass" of the spectrum in Hz. High when
    # hats and vocals dominate, low during a bass-only breakdown. It separates
    # sections that have similar loudness but different character.
    freqs = spectral.bin_freqs(sample_rate)
    centroid = (freqs[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-12)
    return np.vstack([rms, low, mid, high, centroid])


def per_bar(mag, sample_rate, downbeats, hop=spectral.HOP):
    """Average each frame feature over every bar. Shape (n_bars, 5)."""
    feats = frame_features(mag, sample_rate)
    n_frames = feats.shape[1]
    idx = np.clip((np.asarray(downbeats) * sample_rate / hop).astype(int),
                  0, n_frames - 1)
    rows = []
    for a, b in zip(idx[:-1], idx[1:]):
        b = max(b, a + 1)
        rows.append(feats[:, a:b].mean(axis=1))
    if not rows:
        return np.zeros((0, feats.shape[0]))
    return np.vstack(rows)


def energy_curve(bars):
    """One number per bar, 0..1, describing how 'full' the track is there.

    Weighted towards the low band because in dance music the kick and bass
    entering or leaving is what actually reads as an energy change; the
    centroid column is excluded because it measures character, not level.
    """
    if len(bars) == 0:
        return np.zeros(0)
    rms, low, mid, high = bars[:, 0], bars[:, 1], bars[:, 2], bars[:, 3]

    def norm(x):
        hi = np.percentile(x, 95) if len(x) else 1.0
        return np.clip(x / (hi + 1e-12), 0.0, 1.0)

    return 0.35 * norm(rms) + 0.35 * norm(low) + 0.20 * norm(mid) + 0.10 * norm(high)


def loudness(audio_array, sample_rate):
    """EBU R128 integrated loudness in LUFS. Absolute, so comparable across files."""
    y = np.asarray(spectral.as_mono_1d(audio_array), dtype=np.float64)
    try:
        import pyloudnorm
        return round(float(pyloudnorm.Meter(sample_rate).integrated_loudness(y)), 2)
    except Exception:
        rms = float(np.sqrt(np.mean(y ** 2))) + 1e-12
        return round(20.0 * np.log10(rms), 2)


def intro_character(bars, first_full_bar):
    """Is the intro an isolated vocal, or drums?

    A vocal-led intro has presence-band energy with almost no kick under it.
    That is exactly the setup for a slam drop: there is nothing in the low end
    to fight the outgoing track, so the outgoing track can run until the vocal
    lands and then be cut dead on the beat.
    """
    n = len(bars)
    if n == 0 or first_full_bar <= 0:
        return {"vocal_forward_intro": False, "vocal_entry_bar": 0,
                "intro_vocal_ratio": 0.0}
    stop = int(min(first_full_bar, n))
    intro = bars[:stop]
    if len(intro) == 0:
        return {"vocal_forward_intro": False, "vocal_entry_bar": 0,
                "intro_vocal_ratio": 0.0}

    mid = intro[:, 2]
    low = intro[:, 1]
    ratio = float(np.mean(mid) / (np.mean(low) + 1e-9))
    # Where the voice actually starts: first intro bar carrying real mid energy.
    thresh = 0.45 * float(np.max(mid)) if np.max(mid) > 0 else 0.0
    entry = int(np.argmax(mid >= thresh)) if thresh > 0 else 0
    return {
        "vocal_forward_intro": bool(ratio > 1.6 and stop >= 4),
        "vocal_entry_bar": entry,
        "intro_vocal_ratio": round(ratio, 3),
    }


def summarise(bars, curve, lufs=None):
    """Track-level numbers used by the planner.

    Everything here is deliberately *absolute* except `density`. `energy_curve`
    is normalised per track -- which is right for finding that track's own
    drops and breakdowns, and useless for comparing two tracks, because it
    forces every track to peak at 1.0. The planner needs to know that one song
    hits harder than another, so it gets raw levels and z-scores them itself
    across whatever set is actually being mixed.
    """
    if len(bars) == 0:
        return {"loudness": -70.0, "density": 0.0, "low_level": 0.0,
                "brightness": 0.0, "energy": 0.0}
    return {
        "loudness": lufs if lufs is not None else -70.0,
        # Fraction of the track spent at working energy: a relentless banger
        # scores high, a track that is mostly breakdown scores low.
        "density": round(float((curve > 0.6).mean()), 4),
        "low_level": round(float(np.median(bars[:, 1])), 6),
        "brightness": round(float(np.median(bars[:, 4])), 2),
        # Provisional single number for display only; the planner recomputes.
        "energy": round(float(np.mean(curve)), 4),
    }
