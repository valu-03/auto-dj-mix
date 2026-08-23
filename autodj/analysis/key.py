"""Key detection and the Camelot wheel.

Camelot notation exists so that harmonic mixing needs no music theory at the
decks: neighbours on the wheel sound good together. 8A -> 8B, 9A or 7A are all
safe; 8A -> 2A is a train wreck. Converting to it here means the planner in
Lesson 17 only ever compares two short strings.
"""

import librosa
import numpy as np

from .. import spectral

# Krumhansl-Schmuckler key profiles: how strongly each of the 12 pitch classes
# is expected to appear in a major / minor key, from listener experiments.
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Relative major and minor share a Camelot number: A minor = 8A, C major = 8B.
CAMELOT_MAJOR = {0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B",
                 6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B"}
CAMELOT_MINOR = {0: "5A", 1: "12A", 2: "7A", 3: "2A", 4: "9A", 5: "4A",
                 6: "11A", 7: "6A", 8: "1A", 9: "8A", 10: "3A", 11: "10A"}


# How much the bass register's vote counts. Measured at 0.0 / 0.25 / 0.35 / 0.5
# on 24 synthetic progressions of known key: 20/24, 20/24, 20/24, 19/24. It buys
# nothing and costs a second CQT, so it is off -- but the mechanism is sound and
# worth reaching for if real-world minors ever start reading as relative majors.
BASS_WEIGHT = 0.0


def chroma(audio_array, sample_rate, harmonic=False, hop=spectral.HOP):
    """12-bin pitch-class energy over time.

    A CQT is used rather than the linear STFT because pitch is logarithmic: the
    CQT puts one bin per semitone, so folding to 12 pitch classes is exact
    rather than approximate.

    `harmonic=True` runs HPSS to strip percussion first. Measured on the test
    set it costs 11x the runtime (12.4 s vs 1.1 s per track) and *reduces* the
    confidence gap on 8 of 10 tracks, so it is off by default.
    """
    y = spectral.as_mono_1d(audio_array)
    if harmonic:
        y = librosa.effects.harmonic(y, margin=3.0)
    return librosa.feature.chroma_cqt(y=y, sr=sample_rate, hop_length=hop)


def bass_chroma(audio_array, sample_rate, hop=spectral.HOP):
    """Pitch-class energy of the bass register only (C1-C3).

    This is what breaks the relative major/minor tie. A Krumhansl profile sees
    only *which* pitches occur, and C minor and Eb major contain exactly the
    same seven -- so pitch content alone cannot separate them. The bassline can:
    it sits on the tonic. Weighting low-register pitch classes tells us which
    of the two candidates the track is actually resting on.
    """
    y = spectral.as_mono_1d(audio_array)
    c = librosa.feature.chroma_cqt(
        y=y, sr=sample_rate, hop_length=hop,
        fmin=librosa.note_to_hz("C1"), n_octaves=2)
    w = np.median(c, axis=1)
    if w.std() < 1e-9:
        return np.zeros(12)
    return (w - w.mean()) / w.std()


def detect(audio_array, sample_rate, chroma_matrix=None, harmonic=False,
           bass=None, bass_weight=BASS_WEIGHT):
    """Best-matching key, as pitch class + mode + Camelot code."""
    if chroma_matrix is None:
        chroma_matrix = chroma(audio_array, sample_rate, harmonic)
    if bass is None and bass_weight:
        bass = bass_chroma(audio_array, sample_rate)
    if bass is None:
        bass = np.zeros(12)

    # Median, not mean: one loud atonal crash should not move the estimate.
    profile = np.median(chroma_matrix, axis=1)
    if profile.std() < 1e-9:
        return {"key": "?", "mode": "?", "camelot": "1A", "key_confidence": 0.0}
    profile = (profile - profile.mean()) / profile.std()

    best, second = (-9.0, None), (-9.0, None)
    for pc in range(12):
        for template, mode in ((KRUMHANSL_MAJOR, "maj"), (KRUMHANSL_MINOR, "min")):
            t = np.roll(template, pc)
            t = (t - t.mean()) / t.std()
            r = float(np.dot(profile, t) / 12.0) + bass_weight * float(bass[pc])
            if r > best[0]:
                second, best = best, (r, (pc, mode))
            elif r > second[0]:
                second = (r, (pc, mode))

    r, (pc, mode) = best
    camelot = CAMELOT_MAJOR[pc] if mode == "maj" else CAMELOT_MINOR[pc]
    return {
        "key": PITCH_NAMES[pc] + ("" if mode == "maj" else "m"),
        "mode": mode,
        "camelot": camelot,
        "key_correlation": round(r, 4),
        # Gap to the runner-up: a confident key beats its rival clearly.
        "key_confidence": round(float(r - second[0]), 4),
    }


def camelot_distance(a, b):
    """Harmonic distance between two Camelot codes. 0 is a perfect match."""
    na, la = int(a[:-1]), a[-1]
    nb, lb = int(b[:-1]), b[-1]
    ring = min((na - nb) % 12, (nb - na) % 12)   # steps around the wheel
    if la == lb:
        return float(ring)                       # same mode: pure wheel distance
    if ring == 0:
        return 1.0                               # relative major/minor
    return float(ring) + 1.5                     # different mode AND far away


def compatible(a, b):
    """The classic DJ rule: same key, one step round the wheel, or relative."""
    return camelot_distance(a, b) <= 1.0
