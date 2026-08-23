"""One track, fully analysed and cached.

The cache stores the *model* (bpm, offset, phase) rather than the derived beat
list. Beats regenerate exactly from three numbers, so there is no point writing
600 floats to disk -- and no risk of the stored list disagreeing with the model.
"""

import numpy as np

from .. import audio as audio_mod
from .. import corrections, library, spectral
from . import beats, energy, key, structure


def analyse_file(path):
    """Every analyser, from a single decode and a single spectrogram."""
    a, sr = audio_mod.load(path, audio_mod.ANALYSIS_RATE, mono=True)
    mag = spectral.magnitude(a)
    duration = audio_mod.duration(a, sr)

    # Manual grid overrides enter HERE, before anything is measured per bar.
    # Applied afterwards they would leave the energy curve, the segments and
    # the cue points all indexed against the grid the user just rejected.
    over = corrections.get(path)
    beat = corrections.apply_grid(beats.analyse(a, sr, mag), duration, over)
    downbeats = downbeat_times(beat, duration)

    bars = energy.per_bar(mag, sr, downbeats)
    curve = energy.energy_curve(bars)

    meta = library.tags(path)
    result = {
        "file": str(path),
        "artist": meta["artist"],
        "title": meta["title"],
        "duration": round(duration, 4),
        **beat,
        **key.detect(a, sr),
        **energy.summarise(bars, curve, energy.loudness(a, sr)),
        **energy.tonal_balance(mag, sr),
        **energy.intro_character(bars, structure.cue_points(curve)["first_full_bar"]),
        "peak_db": round(float(20 * np.log10(np.max(np.abs(a)) + 1e-12)), 2),
        "year": meta.get("year"),
        "bar_period": round(4 * 60.0 / beat["bpm"], 6),
        "energy_curve": [round(float(x), 4) for x in curve],
        "cues": corrections.apply_cues(structure.analyse(bars, curve), over,
                                       beat["n_bars"]),
    }
    if over:
        result["manual"] = sorted(over)
    return result


def analyse(path, force=False):
    """Cached analysis for one file."""
    return library.cached(path, analyse_file, force=force)


def downbeat_times(meta, duration=None):
    """Regenerate downbeat positions in seconds from the cached model."""
    duration = duration if duration is not None else meta["duration"]
    bt = beats.beat_times(meta["bpm"], meta["beat_offset"], duration)
    return bt[meta["downbeat_phase"]::4]


def bar_time(meta, bar):
    """Time in seconds of a given bar index. Extrapolates past the end."""
    return meta["first_downbeat"] + bar * (4 * 60.0 / meta["bpm"])


def bar_of(meta, seconds):
    """Which bar a given time falls in (may be negative before the first)."""
    return (seconds - meta["first_downbeat"]) / (4 * 60.0 / meta["bpm"])


def describe(meta):
    """One-line summary for logs and the library table."""
    c = meta["cues"]
    return (f"{meta['artist'][:20]:<20} {meta['title'][:26]:<26} "
            f"{meta['bpm']:7.2f} {meta['camelot']:>4} {meta['key']:>4} "
            f"e={meta['energy']:.2f} bars={meta['n_bars']:3d} "
            f"intro={c['intro_bars']:3d} outro={c['outro_start_bar']:3d} "
            f"drops={len(c['drop_bars'])}")
