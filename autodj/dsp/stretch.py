"""Time-stretching: change tempo without changing pitch.

Everything lives behind `time_stretch`, so the phase-vocoder backend can be
swapped for Rubber Band later without touching a single caller.
"""

import warnings

import librosa
import numpy as np

# Beyond this the phase vocoder's transient smearing becomes audible on drums.
# The planner's job is to never ask for more; this is the safety net.
SAFE_STRETCH = 0.06


def rate_for(from_bpm, to_bpm):
    """Playback rate that turns `from_bpm` into `to_bpm`. >1 means faster."""
    return float(to_bpm) / float(from_bpm)


def time_stretch(audio, rate, backend="phase_vocoder"):
    """Stretch a (channels, frames) array by `rate`. Pitch is preserved.

    A phase vocoder resynthesises from the STFT with the hop changed between
    analysis and synthesis. Because `magnitude()` threw phase away for analysis
    but resynthesis *needs* it, the vocoder has to invent consistent phase --
    and that invention is exactly what smears drum transients.
    """
    if abs(rate - 1.0) < 1e-6:
        return audio
    a = np.atleast_2d(audio)
    if backend == "phase_vocoder":
        with warnings.catch_warnings():
            # librosa 1.0 warns about hop_length/n_fft from inside its own call.
            warnings.simplefilter("ignore", FutureWarning)
            out = [librosa.effects.time_stretch(np.ascontiguousarray(ch), rate=rate)
                   for ch in a]
    elif backend == "resample":
        # Cheap and transient-perfect, but it shifts pitch: only ever useful
        # for tiny corrections or deliberate tape-style effects.
        n = int(round(a.shape[1] / rate))
        out = [np.interp(np.linspace(0, ch.shape[0] - 1, n),
                         np.arange(ch.shape[0]), ch) for ch in a]
    else:
        raise ValueError(f"unknown stretch backend: {backend}")

    n = min(len(o) for o in out)
    return np.ascontiguousarray(np.vstack([o[:n] for o in out]).astype(np.float32))


def stretch_to_bpm(audio, from_bpm, to_bpm, backend="phase_vocoder"):
    """Retime a track to a target tempo, returning (audio, rate_used)."""
    rate = rate_for(from_bpm, to_bpm)
    return time_stretch(audio, rate, backend), rate


def is_safe(from_bpm, to_bpm, limit=SAFE_STRETCH):
    """Is this tempo change small enough to be inaudible?"""
    return abs(rate_for(from_bpm, to_bpm) - 1.0) <= limit
