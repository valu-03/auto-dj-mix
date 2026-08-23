"""Pedalboard-backed alternatives to parts of the master chain.

Pedalboard is Spotify's Python wrapper around JUCE's DSP: the same C++ blocks
that ship inside commercial plugins, running out-of-process-fast on numpy
arrays. It is worth having here for one reason above all -- its limiter is a
properly written program-dependent limiter, and ours is a look-ahead gain
computer written to be *understood*, which is not the same as being written to
sound best under heavy correction.

It is an option, not a replacement, and it is off by default. Three reasons:

1. The existing chain is measured. `PROGRESS.md` records what it does to LUFS,
   crest and peak on real material, and a swap that is not measured the same
   way is a change of unknown sign.
2. Every stage in `master.py` exists because a specific fault was heard and
   then fixed. Replacing the chain wholesale discards that, including the
   things that were counter-intuitive -- like a loudness target you cannot
   reach making the result *quieter*.
3. `pedalboard` is an optional dependency. Nothing here may make the program
   fail to import when it is absent.

`AVAILABLE` tells you whether any of this can run. `compare()` measures both
chains on the same audio so the choice is made on numbers rather than on which
library sounds more impressive in a sentence.
"""

import numpy as np

try:
    import pedalboard as _pb
    AVAILABLE = True
except ImportError:                       # pragma: no cover - optional dep
    _pb = None
    AVAILABLE = False

from . import master as master_mod


def _require():
    if not AVAILABLE:
        raise RuntimeError(
            "pedalboard is not installed; `pip install pedalboard`")


def _as_pb(audio):
    """Our (channels, n) float32 -> pedalboard's expected layout."""
    return np.ascontiguousarray(np.atleast_2d(audio).astype(np.float32))


def limit(audio, sample_rate, ceiling_db=-0.3, release_ms=120.0):
    """JUCE's limiter, in place of `master.limit`.

    Threshold is given in dB rather than as linear ceiling because that is the
    unit the plugin works in, and converting at the call site is where an
    off-by-a-factor-of-two hides.
    """
    _require()
    board = _pb.Pedalboard([
        _pb.Limiter(threshold_db=float(ceiling_db),
                    release_ms=float(release_ms)),
    ])
    return board(_as_pb(audio), sample_rate)


def glue(audio, sample_rate, threshold_db=-8.0, ratio=2.0,
         attack_ms=15.0, release_ms=180.0):
    """A gentle bus compressor over the whole mix.

    Deliberately mild. A mix that has already been level-matched per track and
    multiband-compressed does not want another 4:1 on top; this is for the
    half-decibel of movement that makes a long set feel like one record.
    """
    _require()
    board = _pb.Pedalboard([
        _pb.Compressor(threshold_db=float(threshold_db), ratio=float(ratio),
                       attack_ms=float(attack_ms),
                       release_ms=float(release_ms)),
    ])
    return board(_as_pb(audio), sample_rate)


def transition_fx(audio, sample_rate, kind, amount=1.0):
    """Effects for transitions that our own DSP does not do as well.

    Only the two that are genuinely hard to write by hand. Reverb and a real
    resonant ladder filter are both places where a JUCE implementation beats a
    readable scipy one, and neither is on the critical path of a normal join --
    so an optional dependency is an acceptable price for them.
    """
    _require()
    a = float(np.clip(amount, 0.0, 1.0))
    if kind == "reverb_tail":
        chain = [_pb.Reverb(room_size=0.55 * a, wet_level=0.35 * a,
                            dry_level=1.0 - 0.25 * a, width=1.0)]
    elif kind == "ladder_lowpass":
        chain = [_pb.LadderFilter(mode=_pb.LadderFilter.Mode.LPF24,
                                  cutoff_hz=200.0 + 8000.0 * (1.0 - a),
                                  resonance=0.25 + 0.35 * a)]
    elif kind == "ladder_highpass":
        chain = [_pb.LadderFilter(mode=_pb.LadderFilter.Mode.HPF24,
                                  cutoff_hz=40.0 + 1200.0 * a,
                                  resonance=0.25 + 0.35 * a)]
    else:
        raise ValueError(f"unknown effect: {kind!r}")
    out = _pb.Pedalboard(chain)(_as_pb(audio), sample_rate)

    # Reverb adds energy: measured at amount=0.8 on real material it took a
    # peak of 0.97 up to 1.566. On the render path the master chain would
    # catch that, but a join preview is not mastered, and a float sink handed
    # 1.5 does not wrap politely -- it produces full-scale noise. Scaled back
    # as a whole so the effect's own dynamics are untouched.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def master(audio, sample_rate, target=None, ceiling=None):
    """The project's chain, with pedalboard doing the last two stages.

    Everything that encodes a decision about *this* program -- per-track
    loudness matching, the multiband stage, the tonal tilt -- stays. Only the
    glue compression and the final limiting, which are generic problems that
    JUCE solves better than we do, are handed over.
    """
    _require()
    target = master_mod.TARGET_LUFS if target is None else target
    ceiling = master_mod.CEILING if ceiling is None else ceiling

    # Every stage in master.py returns (audio, something-for-the-report) --
    # the gain applied, the reduction achieved, the tilt correction. The
    # second half belongs in the report, not in the signal, and feeding the
    # whole tuple onward produces a ragged object array rather than an error
    # anyone can read.
    out, _gain = master_mod.match_loudness(audio, sample_rate, target=target)
    out, _bands = master_mod.multiband_compress(out, sample_rate)
    out, _tilt = master_mod.tilt_to(out, sample_rate)
    out = glue(out, sample_rate)
    ceiling_db = 20.0 * np.log10(max(1e-6, ceiling))
    return limit(out, sample_rate, ceiling_db=ceiling_db)


def compare(audio, sample_rate, target=None):
    """Measure both chains on the same audio. Returns a dict of dicts.

    The point of this function is that "which mastering chain is better" is
    not a matter of opinion at this level: one of them lands closer to the
    loudness target, one of them keeps more crest factor, and one of them
    clips. Those are numbers.
    """
    target = master_mod.TARGET_LUFS if target is None else target
    rows = {}
    ours, _ = master_mod.master(audio, sample_rate, target=target)
    rows["built-in"] = _measure(ours, sample_rate, target)
    if AVAILABLE:
        theirs = master(audio, sample_rate, target=target)
        rows["pedalboard"] = _measure(theirs, sample_rate, target)
    return rows


def _measure(audio, sample_rate, target):
    a = np.atleast_2d(audio)
    lufs = master_mod.measure_lufs(a, sample_rate)
    peak = float(np.max(np.abs(a)))
    return {
        "lufs": round(float(lufs), 2),
        "lufs_error": round(float(lufs - target), 2),
        "peak": round(peak, 4),
        "clipped": bool(peak > 1.0),
        "crest_db": round(float(master_mod._crest(a)), 2),
    }
