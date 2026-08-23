"""Choosing a transition by rendering it and measuring the result.

Every other decision in this project is made from analysis: the planner reads
two tracks' keys, tempos and energies and reasons about which move suits them.
That reasoning is good, and it is still wrong sometimes, because it is a model
of what the join will sound like rather than the join itself. A bass handover
that looks clean on paper can still leave two sustained low synths overlapping;
a dissolve between compatible keys can still sag in the middle; a cut can land
on a bar where both singers happen to be mid-phrase.

A realtime DJ application cannot do anything about that -- it has to commit to
a transition before it has heard it. This one renders offline, which means the
transition can simply be *tried*, on the actual audio, and measured. Four
candidates over a sixteen-bar region is a couple of seconds of work per join,
against a mix that takes a minute.

So this module is the difference between "this should work" and "this measured
best". The planner still proposes -- its reasoning is what generates a sensible
shortlist -- and the measurements dispose.

What is measured, and why each one:

hole        the summed region going quieter than both of its inputs. The
            single most common way a transition fails: it does not sound
            wrong, it sounds like the energy briefly left the room.

bass        both tracks' low bands audible at the same moment. The rule the
            whole project is built around, verified rather than assumed.

mud         200-400 Hz build-up in the sum beyond what either side has alone.
            Two full arrangements playing together pile up exactly there, and
            it is what "muddy" means when someone says a mix sounds muddy.

clash       harmonic disagreement while both tracks are audible, measured on
            chroma rather than on the Camelot label. Two tracks in compatible
            keys can still be playing clashing chords at this particular
            moment, which no key-level analysis can see.

vocals      measured lead-voice overlap across the region, when stems are
            available.
"""

import numpy as np

from . import spectral, transitions
from .analysis import instruments
from .dsp import filters

MUD_BAND = (180.0, 420.0)

# Weights. Ordered by how badly each fault damages a mix, not by how easy it is
# to measure: a hole is fatal, mud is unpleasant, a brief clash is survivable.
WEIGHTS = {
    "hole": 3.2,
    "bass": 2.6,
    "mud": 1.5,
    "clash": 1.1,
    "vocals": 1.4,
}


def _envelope(a, sample_rate, window_s=0.12):
    """Short-term RMS envelope, one value per window."""
    x = spectral.as_mono_1d(a).astype(np.float64)
    n = max(1, int(window_s * sample_rate))
    trim = (len(x) // n) * n
    if trim < n:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-18)])
    return np.sqrt(np.mean(x[:trim].reshape(-1, n) ** 2, axis=1) + 1e-18)


def hole_score(mixed, out_side, in_side, sample_rate):
    """How far the sum drops below the louder of its two inputs.

    Measured in dB and only counted when it is negative: a transition that
    gets *louder* in the middle is a different problem, and the limiter deals
    with it. A correctly built equal-power crossfade scores near zero here; a
    linear one scores about 3 dB; one with a genuine gap scores far worse.
    """
    m = _envelope(mixed, sample_rate)
    a = _envelope(out_side, sample_rate)
    b = _envelope(in_side, sample_rate)
    k = min(len(m), len(a), len(b))
    if k == 0:
        return 0.0
    loudest = np.maximum(a[:k], b[:k])
    # Only where there was something to lose -- silence at the very edges of a
    # region is not a hole, it is the region ending.
    live = loudest > loudest.max() * 0.15
    if not live.any():
        return 0.0
    drop_db = 20.0 * np.log10((m[:k] + 1e-9) / (loudest + 1e-9))
    return float(np.mean(np.clip(-drop_db[live], 0.0, 24.0)) / 6.0)


def bass_score(out_side, in_side, sample_rate):
    """Fraction of the region with both low bands simultaneously audible.

    The overlap is the *minimum* of the two envelopes, normalised by the
    louder: a quiet rumble under a loud bassline is not two basslines. A clean
    handover measures near zero even though both tracks are playing.
    """
    lo_a = filters.split(np.atleast_2d(out_side), sample_rate)[0]
    lo_b = filters.split(np.atleast_2d(in_side), sample_rate)[0]
    a = _envelope(lo_a, sample_rate)
    b = _envelope(lo_b, sample_rate)
    k = min(len(a), len(b))
    if k == 0:
        return 0.0
    both = np.minimum(a[:k], b[:k])
    ref = np.maximum(a[:k], b[:k]).max() + 1e-9
    return float(np.mean(both / ref))


def mud_score(mixed, out_side, in_side, sample_rate):
    """Low-mid build-up in the sum beyond what the two sides carry alone."""
    def share(x):
        mag = spectral.magnitude(x)
        total = spectral.band_power(mag, sample_rate, (20.0, 11000.0))
        return spectral.band_power(mag, sample_rate, MUD_BAND) / (total + 1e-12)

    got = share(mixed)
    expect = max(share(out_side), share(in_side))
    return float(max(0.0, got - expect) * 6.0)


def clash_score(out_side, in_side, sample_rate, both_audible=None):
    """Harmonic disagreement between the two sides while both are audible.

    Chroma correlation, not key comparison. Two tracks in the same Camelot
    slot are compatible *on average*; this asks whether the specific bars
    about to be played together agree, which is the question that actually
    matters at a join and the one a key label cannot answer.
    """
    import librosa
    a = spectral.as_mono_1d(out_side).astype(np.float32)
    b = spectral.as_mono_1d(in_side).astype(np.float32)
    n = min(len(a), len(b))
    if n < sample_rate // 2:
        return 0.0
    ca = librosa.feature.chroma_cqt(y=a[:n], sr=sample_rate).mean(axis=1)
    cb = librosa.feature.chroma_cqt(y=b[:n], sr=sample_rate).mean(axis=1)
    ca = ca / (np.linalg.norm(ca) + 1e-9)
    cb = cb / (np.linalg.norm(cb) + 1e-9)
    agreement = float(np.dot(ca, cb))
    weight = 1.0 if both_audible is None else float(both_audible)
    return float(max(0.0, 1.0 - agreement) * weight)


def overlap_fraction(tr):
    """How much of the region genuinely has both tracks up, from the curves.

    A hard cut and a dissolve should not be judged by the same standard: a
    clash that lasts four milliseconds is not a clash. Weighting the harmonic
    measurement by the actual overlap is what lets one scale rank both.
    """
    o = np.asarray(tr.out_mid, dtype=float)
    i = np.asarray(tr.in_mid, dtype=float)
    k = min(o.size, i.size)
    if k == 0:
        return 1.0
    return float(np.mean(np.minimum(o[:k], i[:k])))


def evaluate(name, out_tail, in_head, sample_rate, bpm, bars,
             act_out=None, act_in=None, out_bar=0, in_bar=0):
    """Render one candidate over the region and measure it.

    Returns (score, metrics). Lower is better.
    """
    spb = sample_rate * (4 * 60.0 / bpm)
    beat_s = 60.0 / bpm
    tr = transitions.build(name, bars, spb, sample_rate, beat_s)
    mixed = transitions.render(tr, out_tail, in_head, sample_rate)

    n = min(np.atleast_2d(out_tail).shape[1], np.atleast_2d(in_head).shape[1])
    o = transitions.apply_side(np.atleast_2d(out_tail)[:, :n], sample_rate,
                               tr.out_low, tr.out_mid, tr.out_high,
                               tr.out_sweep)
    i = transitions.apply_side(np.atleast_2d(in_head)[:, :n], sample_rate,
                               tr.in_low, tr.in_mid, tr.in_high, tr.in_sweep)

    both = overlap_fraction(tr)
    metrics = {
        "hole": hole_score(mixed, o, i, sample_rate),
        "bass": bass_score(o, i, sample_rate),
        "mud": mud_score(mixed, o, i, sample_rate),
        "clash": clash_score(o, i, sample_rate, both),
        "vocals": 0.0,
    }
    if act_out is not None and act_in is not None \
            and len(act_out) and len(act_in):
        metrics["vocals"] = instruments.collision(act_out, out_bar,
                                                  act_in, in_bar, bars) * both

    score = sum(WEIGHTS[k] * v for k, v in metrics.items())
    return float(score), metrics


def shortlist(seed, style="smooth", limit=4):
    """Candidates worth trying for this join: the planner's pick, plus peers.

    Deliberately small. The point is not to search the space -- most of these
    transitions are wrong for most joins, and the planner already knows that --
    but to check the reasoning against the audio for the handful of moves that
    were plausible.
    """
    peers = {
        "smooth": ["dissolve", "smooth_swap", "fade", "stem_blend",
                   "filter_sweep"],
        "cut": ["hard_cut", "cut_with_echo", "loop_roll", "riser_cut",
                "tremolo"],
        "blend": ["bass_swap", "eq_blend", "filter_sweep", "stem_blend",
                  "echo_out"],
    }.get(style, ["dissolve", "bass_swap", "filter_sweep"])

    out = [seed] + [p for p in peers if p != seed]
    # A double drop is never a substitute for something else: it needs two
    # specific bars to coincide, which the segment plan arranged for this join
    # or did not. Auditioning it against a plan built for a dissolve would
    # measure a move that is not the one that would be rendered.
    if seed == "double_drop":
        return [seed]
    return out[:limit]


def report_line(name, score, metrics):
    bits = " ".join(f"{k}={metrics[k]:.2f}" for k in
                    ("hole", "bass", "mud", "clash", "vocals"))
    return f"{name:<16} {score:6.3f}   {bits}"
