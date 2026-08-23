"""Assembling the mix: place every track on one timeline and sum.

Each track is prepared once -- normalised, retimed to the mix tempo, sliced to
the bars it will actually play -- and then given a single set of EQ automation
curves covering its whole segment: incoming curves at the head, unity through
the middle, outgoing curves at the tail. Placing is then just addition.
"""

import numpy as np

from . import audio as audio_mod
from . import planner, spectral, transitions
from .analysis import instruments
from .analysis import track as track_mod
from .dsp import automation, effects, filters, master, stretch

DEFAULT_BARS = 16
MIN_PLAY_BARS = 24


def mix_tempo(metas, order):
    """One tempo for the whole set: the median, so nothing stretches far."""
    return float(np.median([metas[i]["bpm"] for i in order]))


def master_deck(metas, order):
    """Which track sets the tempo everything else syncs to.

    On a CDJ you pick a master deck and the others lock to it. The master then
    plays at its *own* tempo -- no pitch adjustment, no time-stretching, and so
    no artefacts at all.

    Taking the median BPM as the target instead means every track gets
    stretched, including the one already sitting at the target, and the phase
    vocoder smears transients on all of them for no reason. Choosing the track
    closest to the median as master gives the same set-wide tempo while leaving
    at least one deck completely untouched -- and it is usually the one playing
    longest at the start.

    Returns (index_into_order, bpm).
    """
    bpms = np.array([metas[i]["bpm"] for i in order], dtype=float)
    if bpms.size == 0:
        return 0, 120.0
    med = float(np.median(bpms))
    pos = int(np.argmin(np.abs(bpms - med)))
    return pos, float(bpms[pos])


TEMPO_MODES = ("off", "sync", "blend", "auto")
AUTO_BLEND_THRESHOLD = 0.05      # blend only past a 5% tempo difference
MIN_GLIDE_BARS = 8               # a tempo change needs room to be a glide
GLIDE_FLOOR = 0.002              # below 0.2%, do not bother gliding at all


def glide_lengths(lengths, tempos, minimum=MIN_GLIDE_BARS, floor=GLIDE_FLOOR):
    """Widen any join that has to carry a real tempo change.

    A glide is a rate of change, and the rate is the tempo difference divided
    by the length of the region. Four bars is a good length for a quick cross
    -- it is a poor length to move 4.7 BPM through, because that is the same
    tempo change happening four times faster than over sixteen bars, and both
    decks have to track it exactly.

    Measured on this library, the two 16-bar joins locked to 11.6 ms and 5.8 ms
    while the 4-bar one did not measure cleanly at all. Rather than leave a
    setting that is right most of the time, joins that actually change tempo
    are given at least `minimum` bars; joins whose tempo barely moves keep
    whatever length the duration fit chose, because there is nothing to glide.
    """
    out = list(lengths)
    for p in range(len(out)):
        if p + 1 >= len(tempos):
            break
        a, b = float(tempos[p]), float(tempos[p + 1])
        if abs(b - a) / max(1e-9, a) < floor:
            continue
        out[p] = max(out[p], minimum)
    return out


def deck_tempos(metas, order, mode="sync", master_bpm=None,
                threshold=AUTO_BLEND_THRESHOLD,
                max_stretch=stretch.SAFE_STRETCH):
    """The tempo each track plays at through its own body.

    Four modes, and the difference between them is *where the tempo is allowed
    to change*, not how well anything is beatmatched:

    off     no beatmatching. Every track plays at its native tempo and nothing
            is stretched at all, which is the highest possible audio quality
            and the lowest possible mix quality. Useful as a reference for
            hearing what the phase vocoder is costing.

    sync    one tempo for the whole set, taken from the master deck. Every
            other track is stretched onto it and stays there. This is what the
            mix has always done.

    blend   every track plays at its own tempo, and the tempo *moves* during
            each transition, gliding from the outgoing track's to the
            incoming one's. Only the joins are stretched; the bodies are
            untouched.

    auto    hold a steady tempo while the tracks are close, and glide only
            when the next one is more than `threshold` away. Which is what a
            person does: you do not re-pitch the whole night for half a BPM.

    In `auto` the running tempo is also released once holding it would stretch
    a track past the safe window -- otherwise a long run of slightly-faster
    tracks accumulates into a stretch nobody asked for.
    """
    bpms = np.array([metas[i]["bpm"] for i in order], dtype=float)
    if bpms.size == 0:
        return bpms
    if mode in ("off", "blend"):
        return bpms.copy()
    if mode == "sync":
        bpm = master_bpm if master_bpm else float(np.median(bpms))
        return np.full(bpms.size, float(bpm))

    # Anchored on the master deck, not on whatever happens to be first. Holding
    # track 1's tempo through a set means every other track is stretched onto
    # an arbitrary reference: measured here, anchoring on the opener put all
    # four tracks under the vocoder, where anchoring on the master leaves the
    # master untouched and moves the rest less.
    hold = float(master_bpm) if master_bpm else float(np.median(bpms))
    out = []
    for i in range(bpms.size):
        own = float(bpms[i])
        drift = abs(own - hold) / max(1e-9, hold)
        if drift > min(threshold, max_stretch):
            hold = own                    # too far to hold: move to meet it
        out.append(hold)
    return np.asarray(out, dtype=float)


def _smoothstep(x):
    """3x^2 - 2x^3: zero slope at both ends.

    This is the difference between a tempo blend you notice and one you do
    not. A linear glide changes tempo at a constant rate, which means the rate
    itself jumps from zero to full at the moment the glide starts and back to
    zero when it ends. Those two corners are the audible part -- the ear tracks
    tempo *change*, not tempo, so a kink in the ramp reads as a lurch even when
    the ramp is only 2 BPM tall. Smoothstep has zero derivative at both ends,
    so the glide begins and finishes invisibly and all the movement happens in
    the middle where a phrase is already carrying attention.
    """
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def tempo_profile(seg, prev_bpm, own_bpm, next_bpm, steps=512, ease=True):
    """Tempo as a function of position within one segment, in its own bars.

    Flat through the body, gliding across the head and tail regions. Because a
    join's head length equals the previous segment's tail length, and both are
    given the same pair of endpoint tempos and the same easing, the two decks
    glide along the *identical* curve while they overlap -- which is the whole
    point. A tempo blend where only the outgoing deck moves is not a blend, it
    is two decks drifting apart in public.
    """
    bars = float(seg["bars"])
    head, tail = float(seg["head"]), float(seg["tail"])
    shape = _smoothstep if ease else (lambda x: np.clip(x, 0.0, 1.0))
    b = np.linspace(0.0, bars, steps + 1)
    t = np.full_like(b, float(own_bpm))
    if head > 0 and prev_bpm is not None:
        m = b < head
        t[m] = prev_bpm + (own_bpm - prev_bpm) * shape(b[m] / head)
    if tail > 0 and next_bpm is not None:
        s = bars - tail
        m = b >= s
        t[m] = own_bpm + (next_bpm - own_bpm) * shape((b[m] - s) /
                                                      max(1e-9, tail))
    return b, t


def glide_cents(a_bpm, b_bpm):
    """How far a resampled tempo glide moves the pitch, in cents.

    Worth knowing rather than assuming. Resampling changes tempo and pitch
    together: a 5% tempo glide is 84 cents, most of a semitone, which is
    enough to break the harmonic match that the whole Camelot layer exists to
    protect. Below about 30 cents nobody notices and resampling is the better
    tool -- it is transparent where a phase vocoder smears. Above it, the glide
    has to be done as a real time-stretch instead, and this is the number that
    decides which.
    """
    if a_bpm <= 0 or b_bpm <= 0:
        return 0.0
    return float(abs(1200.0 * np.log2(b_bpm / a_bpm)))


PITCH_TOLERANCE_CENTS = 30.0


def _cumulative_seconds(b, t):
    """Elapsed time at each bar position, given a tempo curve over bars.

    A bar at tempo T lasts 4*60/T seconds, so time is the integral of the
    reciprocal of tempo -- not of tempo. Integrating tempo directly is the
    obvious mistake and it makes an accelerating passage get *longer*.
    """
    dur = 4.0 * 60.0 / np.maximum(t, 1e-6)
    return np.concatenate([[0.0], np.cumsum(np.diff(b) * (dur[:-1] + dur[1:]) / 2.0)])


def warp_to_profile(seg_audio, sample_rate, own_bpm, b, t,
                    preserve_pitch=None, blocks=12):
    """Resample a flat-tempo segment onto a gliding tempo curve.

    The segment arrives cut to an exact number of bars at one constant tempo,
    so bar `x` sits at sample `x * spb`. The output wants bar `x` at whatever
    time the tempo curve says, so the job is: build the output time base,
    invert the curve to get a bar position per output sample, and read the
    source there.

    Two ways to do that, and the right one depends on how big the glide is:

    resample       one interpolation, transient-perfect, and it moves the
                   pitch with the tempo -- which is what a real record does
                   when you nudge the platter, and below ~30 cents nobody
                   hears it as anything but tempo.

    stretch        pitch-invariant, via the phase vocoder applied block by
                   block at each block's local rate. Necessary once the glide
                   is big enough to break the harmonic match -- a 5% blend is
                   84 cents, most of a semitone, and no amount of Camelot
                   planning survives that. It costs transient smearing, which
                   is the trade being made deliberately.

    `preserve_pitch=None` decides per segment from the size of the glide.
    """
    cum = _cumulative_seconds(b, t)
    n_out = int(round(cum[-1] * sample_rate))
    n_src = seg_audio.shape[1]
    if n_out < 2 or n_src < 2:
        return seg_audio

    if preserve_pitch is None:
        preserve_pitch = glide_cents(float(np.min(t)),
                                     float(np.max(t))) > PITCH_TOLERANCE_CENTS

    spb = sample_rate * (4 * 60.0 / own_bpm)
    out_t = np.arange(n_out, dtype=np.float64) / sample_rate
    src = np.clip(np.interp(out_t, cum, b) * spb, 0.0, n_src - 1.0)

    if not preserve_pitch:
        base = np.arange(n_src, dtype=np.float64)
        return np.ascontiguousarray(
            np.vstack([np.interp(src, base, ch) for ch in seg_audio]
                      ).astype(np.float32))

    # Block-wise stretch. The rate varies slowly compared with a block, so a
    # constant rate per block is a good approximation; the blocks are then
    # butted with a short equal-power crossfade, because a phase vocoder run
    # twice on adjacent audio does not produce phase-continuous output at the
    # seam and the join would click.
    edges = np.linspace(0, n_src, blocks + 1).astype(int)
    xfade = min(256, max(32, n_src // (blocks * 8)))
    pieces = []
    for i in range(blocks):
        s, e = edges[i], edges[i + 1]
        if e - s < 64:
            continue
        # How much output this block of source is asked to fill.
        want = float(np.interp(e, src, out_t, left=out_t[0], right=out_t[-1])
                     - np.interp(s, src, out_t, left=out_t[0],
                                 right=out_t[-1])) * sample_rate
        rate = (e - s) / max(1.0, want)
        pieces.append(stretch.time_stretch(seg_audio[:, s:e], rate))

    if not pieces:
        return seg_audio
    out = pieces[0]
    for nxt in pieces[1:]:
        k = min(xfade, out.shape[1], nxt.shape[1])
        if k > 1:
            fo, fi = automation.equal_power(k)
            out[:, -k:] = out[:, -k:] * fo + nxt[:, :k] * fi
            out = np.concatenate([out, nxt[:, k:]], axis=1)
        else:
            out = np.concatenate([out, nxt], axis=1)

    # Force the exact target length, same reasoning as `_retime_slice`.
    if out.shape[1] != n_out:
        x = np.linspace(0.0, out.shape[1] - 1.0, n_out)
        base = np.arange(out.shape[1], dtype=np.float64)
        out = np.vstack([np.interp(x, base, ch) for ch in out])
    return np.ascontiguousarray(out.astype(np.float32))


def tempo_ramp(metas, order, spread=None, max_stretch=stretch.SAFE_STRETCH):
    """A tempo per track that climbs across the set, instead of one flat tempo.

    Real sets drift upward -- open at 128, finish at 138 -- and that climb is
    itself a source of energy. Forcing one median tempo throws that away *and*
    costs stretch: a 130 BPM opener and a 138 BPM peak track cannot both sit at
    134 without one of them being pulled 3% out of shape.

    Letting the target follow the running order fixes both at once. Each track
    is stretched toward a tempo near its own, so the stretch shrinks, and the
    set gains a rise that no EQ move can fake.

    The ramp is fitted to the tracks actually chosen, not imposed: it runs from
    the slower end of the set to the faster end, so nothing is asked to move
    further than it already would have. Every target is then clamped to within
    `max_stretch` of the track's own tempo, which means the ramp can never make
    the stretching worse than the flat tempo it replaces.
    """
    bpms = np.array([metas[i]["bpm"] for i in order], dtype=float)
    n = len(bpms)
    if n < 2:
        return bpms.copy()

    lo, hi = np.percentile(bpms, 15), np.percentile(bpms, 85)
    if spread is not None:
        mid = float(np.median(bpms))
        lo, hi = mid - spread / 2.0, mid + spread / 2.0
    if hi < lo:
        lo, hi = hi, lo

    targets = np.linspace(lo, hi, n)
    # Never ask a track to stretch further than it would have at a flat tempo.
    limit = bpms * max_stretch
    return np.clip(targets, bpms - limit, bpms + limit)


def bar_list(bars, n_tracks):
    """One region length per join. Accepts a single int or a per-join list.

    Per-join lengths exist because "how long is a transition" is not one
    answer for a whole set. A 4-bar cut into the peak and a 16-bar blend out of
    it belong in the same mix, and forcing both to the same number makes one of
    them wrong. A scalar still works and still means what it always meant.
    """
    n = max(0, n_tracks - 1)
    if np.isscalar(bars):
        return [int(bars)] * n
    out = [int(b) for b in list(bars)[:n]]
    while len(out) < n:
        out.append(int(out[-1]) if out else DEFAULT_BARS)
    return out


def segment_plan(metas, order, bars=DEFAULT_BARS, activity=None,
                 join_names=None, tempos=None, sample_rate=None):
    """Where each track enters and leaves, in its own bars and in mix bars.

    With `activity` (per-track lead-voice-per-bar arrays, keyed by track index)
    the entry point is nudged among phrase-aligned candidates to the one where
    the two tracks are least likely to be singing at once. Beatmatching decides
    *when* B enters; this decides *which* phrase of B enters, so the overlap is
    A's groove under B's groove rather than two lead vocals arguing.

    Candidates are always whole phrases away from the structurally chosen entry
    -- the alternative is a musically wrong entry that happens to dodge a
    vocal, which is worse than the collision.

    A double drop overrides both ends of its join. Every other transition asks
    "where should this track come in"; a double drop asks "which two bars must
    coincide", and answers it by placing A's exit and B's entry so that the two
    drops fall on the same mix bar. It is the only move here whose timing is
    dictated by the music rather than by the region.
    """
    lengths = bar_list(bars, len(order))
    at = transitions.DOUBLE_DROP_AT
    segs, cursor = [], 0.0
    last = len(order) - 1
    for pos, ti in enumerate(order):
        m = metas[ti]
        c = m["cues"]
        n_bars = max(1, m["n_bars"] - 1)
        head = lengths[pos - 1] if pos > 0 else 0
        tail = lengths[pos] if pos < last else 0
        name_in = join_names[pos - 1] if join_names and pos > 0 else None
        name_out = join_names[pos] if join_names and pos < last else None

        if pos == 0:
            enter = max(0, c["first_full_bar"] - 4)
        elif name_in == "double_drop" and \
                planner.usable_drop(m, "in") is not None and \
                planner.usable_drop(m, "in") - int(round(at * head)) >= 0:
            # B's drop has to land at the region's drop point, which sits
            # `at` of the way through the head region. Not clamped to zero: a
            # clamp would move the drop off the alignment point while still
            # rendering a double drop, which is the worst of both -- two full
            # tracks over each other with their drops in different bars. If it
            # does not fit, the normal entry rule runs instead.
            enter = planner.usable_drop(m, "in") - int(round(at * head))
        else:
            # How early the track must start so that it becomes AUDIBLE exactly
            # on its first full bar. For a blend that is the whole region; for a
            # hard cut it is only the part before the cut fires. Using the whole
            # region for a cut put every incoming track two bars early -- the
            # tempo was perfect and it still sounded wrong, because the cut
            # landed mid-phrase.
            lead = transitions.entry_lead(name_in, head) if name_in else head
            enter = max(0, c["first_full_bar"] - lead)
            if activity is not None:
                prev = segs[-1]
                cands = [enter + k * head for k in (0, 1, -1, 2)
                         if 0 <= enter + k * head <= max(0, n_bars - head)]
                if cands:
                    enter, _ = instruments.best_entry(
                        activity.get(order[pos - 1], np.zeros(0)),
                        prev["exit"] - head,
                        activity.get(ti, np.zeros(0)), cands, head)

        if pos == last:
            exit_bar = min(n_bars, c["last_full_bar"] + 4)
        else:
            exit_bar = min(c["outro_start_bar"], c["last_full_bar"] + 1) - tail
            exit_bar = max(exit_bar, enter + MIN_PLAY_BARS)
            exit_bar = min(exit_bar, n_bars - tail)
            exit_bar = max(exit_bar, enter + tail + 4)

            if name_out == "double_drop":
                drop = planner.usable_drop(m, "out")
                # A's own drop bar must land on the same mix bar as B's. If
                # that would leave too little of A playing, the alignment is
                # abandoned rather than forced: a double drop that truncates
                # its own outgoing track to eight bars is worse than an
                # ordinary aligned overlap, which is what the transition
                # degrades to.
                if drop is not None:
                    want = drop - int(round(at * tail))
                    if enter + tail + 4 <= want <= n_bars - tail:
                        exit_bar = want

        play = max(tail + 4, exit_bar - enter)
        segs.append({
            "track": ti, "pos": pos, "enter": enter, "exit": enter + play,
            "mix_start": cursor,
            "bars": play + tail,
            "head": head,
            "tail": tail,
            "bpm": float(tempos[pos]) if tempos is not None else None,
        })
        cursor += play
    return _place(segs, sample_rate)


def _place(segs, sample_rate):
    """Turn bar positions into sample positions.

    With one tempo for the whole set a bar is a fixed number of samples and
    `mix_start * spb` is enough. With a tempo ramp it is not: each segment has
    its own bar length, so positions have to accumulate in samples or every
    track after the first lands progressively further from where it belongs.
    """
    if sample_rate is None or segs and segs[0].get("bpm") is None:
        return segs
    at = 0
    for i, s in enumerate(segs):
        spb = sample_rate * (4 * 60.0 / s["bpm"])
        s["spb"] = spb
        s["start_sample"] = at
        s["len_samples"] = int(round(s["bars"] * spb))
        # The next track begins when this one's *play* portion ends; its head
        # then overlaps this one's tail, which is what a transition is.
        at += int(round((s["bars"] - s["tail"]) * spb))
    return segs


def _retime_slice(a, meta, enter_bar, n_bars, target_bpm, sample_rate):
    """Retime to the mix tempo and cut exactly n_bars, starting on a downbeat.

    Factored out so the vocal stem can be put through the *identical*
    transform as the full track. If the two ever diverge by even a few
    samples, subtracting one from the other stops cancelling and starts
    phasing.
    """
    rate = stretch.rate_for(meta["bpm"], target_bpm)
    t0 = track_mod.bar_time(meta, enter_bar)
    want = int(round(n_bars * (4 * 60.0 / target_bpm) * sample_rate))

    # Cut EXACTLY the bars we want, not a padded window. The old version took
    # 5% extra and then trimmed, which meant the segment's true length was
    # whatever the phase vocoder happened to produce and the trim decided where
    # the beats fell. Measured, that left the two sides of a join 2-7 ms apart.
    # Inaudible across a hard cut, but in an overlapping dissolve two kicks
    # 7 ms apart flam -- which is exactly what "not smooth" sounds like.
    src_len = n_bars * (4 * 60.0 / meta["bpm"])
    seg = audio_mod.clip(a, sample_rate, t0, t0 + src_len)
    if seg.shape[1] < 16:
        return np.zeros((2, want), dtype=np.float32)

    seg = stretch.time_stretch(seg, rate)

    # Then force the result to the exact target length by resampling. The
    # phase vocoder is only approximately rate-accurate, so padding or
    # truncating leaves the internal beat grid slightly wrong; resampling
    # scales the whole segment so every beat lands where the grid says. The
    # correction is a fraction of a percent -- under a cent of pitch shift,
    # which is inaudible, and it buys sample-exact phrase alignment.
    n = seg.shape[1]
    if n != want and n > 1:
        src_x = np.linspace(0.0, n - 1.0, want)
        idx = np.arange(n, dtype=np.float64)
        seg = np.vstack([np.interp(src_x, idx, ch) for ch in seg])
    elif n < want:
        seg = np.pad(seg, ((0, 0), (0, want - n)))
    return np.ascontiguousarray(seg[:, :want].astype(np.float32))


def beat_phase_error(seg, sample_rate, target_bpm, search=0.5):
    """How far a prepared segment's beats sit from the mix grid, in seconds.

    This is the closed-loop half of Sync. Everything upstream is open-loop --
    fit a grid, stretch by a ratio, cut on a bar -- and each step is
    *approximately* right. The phase vocoder in particular is only
    approximately rate-accurate: forcing the segment to the exact target length
    fixes the average, but local timing still drifts, and on the track taking
    the largest stretch that measured 16.9 ms of error at the join.

    16.9 ms is 3.75% of a beat. Across a hard cut nobody hears it, because
    nothing overlaps. In a dissolve two kicks 16.9 ms apart flam -- and that
    flam is exactly what "not smooth" sounds like.

    So rather than trusting the chain, measure what actually came out.
    """
    from .analysis import onsets

    a = np.atleast_2d(seg).mean(axis=0)
    if a.size < sample_rate // 2:
        return 0.0
    # Envelope at the analysis rate, same STFT settings as everywhere else.
    step = max(1, int(round(sample_rate / audio_mod.ANALYSIS_RATE)))
    low = np.ascontiguousarray(a[::step].astype(np.float32))
    # A quarter of the usual hop. The standard 512-sample hop is 23 ms per
    # frame at the analysis rate, which caps how precisely a phase can be
    # located no matter how finely the search is stepped -- and 23 ms is three
    # times the error we are trying to remove. 128 gives 5.8 ms frames, and
    # interpolating between them resolves well under a millisecond.
    hop = spectral.HOP // 4
    env = onsets.envelope(spectral.magnitude(low, hop=hop),
                          audio_mod.ANALYSIS_RATE)
    if env.size < 16:
        return 0.0

    hop_s = hop / audio_mod.ANALYSIS_RATE
    beat = 60.0 / target_bpm
    t = np.arange(len(env)) * hop_s
    best, best_v = 0.0, -1.0
    for ph in np.linspace(-search * beat, search * beat, 721):
        grid = np.arange(ph % beat, t[-1], beat)
        if grid.size < 4:
            continue
        v = float(np.interp(grid, t, env).mean())
        if v > best_v:
            best_v, best = v, ph % beat
    # Fold into +/- half a beat: a whole beat late is the same phase.
    if best > beat / 2:
        best -= beat
    return float(best)


def shift_samples(seg, shift):
    """Delay or advance by a fractional number of samples, without resampling.

    Linear interpolation between neighbours. The shift here is only a few
    milliseconds, so this is far cheaper than a resample and does not touch
    pitch or length.
    """
    if abs(shift) < 1e-6:
        return seg
    n = seg.shape[1]
    idx = np.arange(n, dtype=np.float64) + shift
    idx = np.clip(idx, 0, n - 1)
    base = np.arange(n, dtype=np.float64)
    return np.ascontiguousarray(
        np.vstack([np.interp(idx, base, ch) for ch in seg]).astype(np.float32))


def _wrap(x, period):
    """Fold a phase difference into +/- half a period."""
    return (x + period / 2.0) % period - period / 2.0


def sync_to(seg, sample_rate, target_bpm, reference_phase, tolerance_ms=1.0,
            window_bars=8):
    """Lock a segment's beatgrid to the MASTER deck's phase. The Sync button.

    The phase this measures is not absolute. `beat_phase_error` locates beats
    relative to the segment's first sample, and the onset envelope carries a
    systematic offset from STFT framing and smoothing -- measured, every
    segment reads about +200 ms, roughly half a beat, whether or not it is
    aligned. An earlier version compared that figure against zero, decided
    every track was 200 ms out, and refused to correct anything because the
    error exceeded its own sanity limit. It reported success while doing
    nothing.

    What matters is not where a deck sits against an arbitrary origin, but
    where it sits against the *master*. So the master's phase is the reference
    and everything else is moved onto it, exactly like a Sync button locking to
    a master deck. A constant bias shared by every measurement then cancels.

    Correction is a curve, not a constant: the vocoder's drift varies along a
    track (measured spread within one track: 48 ms), so a single shift would
    fix the average and leave the ends wrong.
    """
    n = seg.shape[1]
    spb = sample_rate * (4 * 60.0 / target_bpm)
    beat = 60.0 / target_bpm
    win = int(round(window_bars * spb))

    if n < 2 * win:
        err = _wrap(beat_phase_error(seg, sample_rate, target_bpm)
                    - reference_phase, beat)
        if abs(err) * 1000.0 < tolerance_ms:
            return seg, 0.0
        return shift_samples(seg, err * sample_rate), err * 1000.0

    centres, errs = [], []
    for start in range(0, n - win + 1, win):
        e = _wrap(beat_phase_error(seg[:, start:start + win], sample_rate,
                                   target_bpm) - reference_phase, beat)
        centres.append(start + win / 2.0)
        errs.append(e * sample_rate)
    if not errs:
        return seg, 0.0

    centres = np.asarray(centres, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    if errs.size >= 3:
        from scipy.ndimage import median_filter
        errs = median_filter(errs, size=3, mode="nearest")

    x = np.arange(n, dtype=np.float64)
    shift = np.interp(x, centres, errs, left=errs[0], right=errs[-1])
    worst = float(np.max(np.abs(errs)) / sample_rate * 1000.0)
    if worst < tolerance_ms:
        return seg, 0.0
    idx = np.clip(x + shift, 0, n - 1)
    out = np.vstack([np.interp(idx, x, ch) for ch in seg])
    return np.ascontiguousarray(out.astype(np.float32)), worst


def sync(seg, sample_rate, target_bpm, tolerance_ms=1.0, max_ms=90.0,
         window_bars=8):
    """Lock a prepared segment's beatgrid onto the mix grid. The Sync button.

    Corrects with a *curve*, not a constant. A single global shift assumes the
    segment drifts by the same amount throughout, and it does not: the phase
    vocoder's error varies along the track, so fixing the average left the
    worst join still ~7 ms out while other parts were over-corrected. Measuring
    every `window_bars` and interpolating between those measurements follows
    the drift instead of averaging it away.

    The shift curve is smoothed before it is applied. An abrupt change in
    timing offset is a pitch glitch -- the same reason a tape splice clicks --
    so the correction has to arrive gradually even though the measurement is
    per window.

    Returns (audio, worst_correction_ms).
    """
    n = seg.shape[1]
    spb = sample_rate * (4 * 60.0 / target_bpm)
    win = int(round(window_bars * spb))
    if n < 2 * win:
        err = beat_phase_error(seg, sample_rate, target_bpm)
        ms = err * 1000.0
        if abs(ms) < tolerance_ms or abs(ms) > max_ms:
            return seg, 0.0
        return shift_samples(seg, err * sample_rate), ms

    centres, errs = [], []
    for start in range(0, n - win + 1, win):
        chunk = seg[:, start:start + win]
        e = beat_phase_error(chunk, sample_rate, target_bpm)
        if abs(e * 1000.0) > max_ms:
            e = 0.0
        centres.append(start + win / 2.0)
        errs.append(e * sample_rate)
    if not errs:
        return seg, 0.0

    centres = np.asarray(centres, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    # A window that lands on a quiet passage can return a wild estimate; the
    # median filter keeps one bad window from bending the whole curve.
    if errs.size >= 3:
        from scipy.ndimage import median_filter
        errs = median_filter(errs, size=3, mode="nearest")

    x = np.arange(n, dtype=np.float64)
    shift = np.interp(x, centres, errs, left=errs[0], right=errs[-1])
    if abs(np.max(np.abs(shift))) < tolerance_ms * sample_rate / 1000.0:
        return seg, 0.0

    idx = np.clip(x + shift, 0, n - 1)
    out = np.vstack([np.interp(idx, x, ch) for ch in seg])
    return (np.ascontiguousarray(out.astype(np.float32)),
            float(np.max(np.abs(errs)) / sample_rate * 1000.0))


def prepare(path, meta, enter_bar, n_bars, target_bpm, sample_rate):
    """Load, loudness-match, retime to the mix tempo, slice to n_bars.

    Returns (audio, gain) -- the gain matters because anything that has to line
    up with this signal later must be scaled by the same amount.
    """
    a, _ = audio_mod.load(path, sample_rate, mono=False)
    if a.shape[0] == 1:
        a = np.vstack([a[0], a[0]])
    a, g = master.match_loudness(a, sample_rate, target=master.BUS_LUFS,
                                 max_peak=1.0)
    return _retime_slice(a, meta, enter_bar, n_bars, target_bpm,
                         sample_rate), g


STEM_NAMES = ("vocals", "drums", "bass", "other")


def prepare_stems(path, meta, enter_bar, n_bars, target_bpm, sample_rate,
                  gain):
    """All four stems through the identical transform, or None.

    Identical is not approximately identical. The stems are recombined by
    addition, and Demucs guarantees they sum back to the source -- but only
    sample for sample. Put one stem through a slightly different stretch and
    the sum stops being the track and starts being the track plus a comb
    filter, which is heard as a hollow, phasey version of what went in.
    """
    from .stems import separate as sep
    cached = sep.load_cached(path)
    if not cached:
        return None
    out = {}
    for name in STEM_NAMES:
        p = cached.get(name)
        if not p:
            return None
        v, _ = audio_mod.load(p, sample_rate)
        if v.shape[0] == 1:
            v = np.vstack([v[0], v[0]])
        out[name] = _retime_slice(v * gain, meta, enter_bar, n_bars,
                                  target_bpm, sample_rate)
    return out


def stem_mix(stems, n, head_n, tail_n, head_curves=None, tail_curves=None):
    """Rebuild a segment from its stems, with per-stem gain in the regions.

    Unity everywhere except the head and tail: because the stems sum to the
    source, a segment with no stem automation in either region reconstructs
    the original exactly, so this is safe to use on any segment that
    participates in a stem transition at either end.
    """
    out = None
    for name in STEM_NAMES:
        s = stems[name]
        g = np.ones(n, dtype=np.float64)
        if head_curves is not None and head_n:
            g[:head_n] = automation.fit(head_curves[name], head_n)
        if tail_curves is not None and tail_n:
            g[n - tail_n:] = automation.fit(tail_curves[name], tail_n)
        piece = s[:, :n] * g
        out = piece if out is None else out + piece
    return np.ascontiguousarray(out.astype(np.float32))


def prepare_vocal(path, meta, enter_bar, n_bars, target_bpm, sample_rate, gain):
    """The vocal stem alone, through the identical path at the identical gain.

    Returns None when the track has not been separated -- ducking is then
    simply skipped rather than approximated with a filter, because a 300 Hz-
    3.5 kHz cut would gut the whole track, which is the exact problem stems
    exist to avoid.
    """
    from .stems import separate as sep
    cached = sep.load_cached(path)
    if not cached:
        return None
    v, _ = audio_mod.load(cached["vocals"], sample_rate)
    if v.shape[0] == 1:
        v = np.vstack([v[0], v[0]])
    return _retime_slice(v * gain, meta, enter_bar, n_bars, target_bpm,
                         sample_rate)


def duck_curve(in_act, in_enter, bars, spb, depth=0.85, smooth_bars=0.5):
    """Per-sample gain for the OUTGOING vocal across an overlap.

    Driven by the incoming track's vocal activity bar by bar: where B sings,
    A's vocal steps aside; where B is instrumental, A is left alone. So this
    is not a fade-out -- a track whose replacement has an instrumental intro
    keeps its vocal at full level right to the end.

    Interpolated between bar centres rather than stepped. A hard change on the
    bar line is audible as a lurch, and the thing being ducked is a voice,
    where that is especially obvious.
    """
    in_act = np.asarray(in_act, dtype=np.float64)
    if in_act.size == 0:
        return np.ones(int(round(bars * spb)))
    idx = np.clip(np.arange(int(in_enter), int(in_enter) + int(bars)),
                  0, len(in_act) - 1)
    per_bar = 1.0 - depth * in_act[idx]
    n = int(round(bars * spb))
    x = np.linspace(0.0, bars, n, endpoint=False)
    g = np.interp(x, np.arange(bars) + 0.5, per_bar,
                  left=per_bar[0], right=per_bar[-1])
    w = max(1, int(smooth_bars * spb))
    if w > 1:
        from scipy.ndimage import uniform_filter1d
        g = uniform_filter1d(g, size=w, mode="nearest")
    return g


def _eq_region(chunk, sample_rate, low, mid, high, sweep=None):
    """EQ one sub-region. Only head and tail regions ever need this."""
    if chunk.shape[1] == 0:
        return chunk
    a = chunk
    if sweep is not None:
        btype, cutoffs = sweep
        a = np.atleast_2d(filters.sweep(a, sample_rate, cutoffs, btype=btype))
    n = a.shape[1]
    bands = filters.split(a, sample_rate)
    return filters.apply_eq(a, sample_rate, automation.fit(low, n),
                            automation.fit(mid, n), automation.fit(high, n),
                            bands=bands)


def _profiles(segs, tempos, sample_rate, ease=True, glide=True):
    """Per-segment tempo curve, length in samples, and placement.

    One code path for every tempo mode. With a flat set of tempos the profile
    is a constant, the warp below is skipped, and the arithmetic reduces to
    `mix_start * spb` -- exactly what it was before tempo blending existed. So
    the gliding case is not a special case bolted on beside the simple one; the
    simple one is the gliding case with a flat curve.
    """
    at = 0
    for i, seg in enumerate(segs):
        # `glide=False` is what separates "off" from "blend". Both give every
        # track its own tempo, and without this they produced byte-identical
        # output: the profile interpolated across every join regardless, so
        # "no beatmatching" was quietly performing a tempo blend. Off means the
        # tempo *jumps* at the join, because nothing is being matched.
        prev_bpm = float(tempos[i - 1]) if (glide and i > 0) else None
        own = float(tempos[i])
        nxt = (float(tempos[i + 1])
               if (glide and i + 1 < len(tempos)) else None)
        b, t = tempo_profile(seg, prev_bpm, own, nxt, ease=ease)
        cum = _cumulative_seconds(b, t)

        seg["profile"] = (b, t)
        seg["bpm"] = own
        seg["start_sample"] = at
        seg["len_samples"] = int(round(cum[-1] * sample_rate))
        seg["head_samples"] = int(round(
            float(np.interp(seg["head"], b, cum)) * sample_rate))
        body_end = float(np.interp(seg["bars"] - seg["tail"], b, cum))
        seg["tail_samples"] = seg["len_samples"] - int(round(body_end *
                                                             sample_rate))
        seg["flat"] = bool(np.ptp(t) < 1e-9)
        at += int(round(body_end * sample_rate))
    return segs


def render(paths, metas, order, bars=DEFAULT_BARS, sample_rate=None,
           progress=None, style="blend", activity=None, segs=None,
           join_names=None, tempo_mode="sync", cache=None, audition=False,
           ease=True):
    """Render the whole mix. Returns (audio, report, segs, joins).

    `join_names` overrides the planner's choice per join -- what the timeline
    inspector edits. `bars` may be a single number or one per join. `cache` is
    a dict the caller keeps between renders: prepared, synced segments are
    stored in it, so changing one transition re-renders in seconds instead of
    re-decoding and re-stretching every track. `audition` renders each join's
    candidates and keeps whichever measures best.
    """
    sample_rate = sample_rate or audio_mod.RENDER_RATE
    master_pos, master_bpm = master_deck(metas, order)
    inten = planner.intensity(metas)
    supplied = segs is not None
    lengths = bar_list(bars, len(order))

    # Which transition joins each consecutive pair. Decided BEFORE the segment
    # plan, because the transition type determines how early the incoming track
    # has to start in order to become audible on its own downbeat.
    if join_names is not None:
        names = list(join_names)[:max(0, len(order) - 1)]
        while len(names) < len(order) - 1:
            names.append("dissolve")
    else:
        names = [planner.choose_transition(metas[order[p]], metas[order[p + 1]],
                                           inten[order[p]], inten[order[p + 1]],
                                           style)
                 for p in range(len(order) - 1)]

    tempos = deck_tempos(metas, order, tempo_mode, master_bpm)
    glide = bool(np.ptp(tempos) > 1e-9)
    if glide and tempo_mode in ("blend", "auto"):
        lengths = glide_lengths(lengths, tempos)
    bpm = float(tempos[master_pos]) if glide else float(master_bpm)
    spb = sample_rate * (4 * 60.0 / bpm)          # nominal, for curve building
    beat_s = 60.0 / bpm

    # `segs` may be supplied by the arranger, which builds the set out of
    # sections rather than whole tracks. Its blocks have the same shape as a
    # segment plan, so everything below is identical either way.
    if segs is None:
        segs = segment_plan(metas, order, lengths, activity, names,
                            tempos if glide else None, sample_rate)

    tried = []
    if audition and join_names is None:
        names, tried = audition_joins(paths, metas, order, segs, names, lengths,
                                      tempos, sample_rate, style, activity,
                                      progress)
        if not supplied:
            # The winning moves may want different entry points from the seeds,
            # so the plan is rebuilt around them rather than kept.
            segs = segment_plan(metas, order, lengths, activity, names,
                                tempos if glide else None, sample_rate)

    if supplied:
        tempos = np.full(len(segs), bpm, dtype=float)
    glides = tempo_mode in ("blend", "auto")
    _profiles(segs, tempos, sample_rate, ease=ease, glide=glides)
    joins = [transitions.build(n, lengths[p], spb, sample_rate, beat_s)
             for p, n in enumerate(names)]

    total = (segs[-1]["start_sample"] + segs[-1]["len_samples"]) + sample_rate
    out = np.zeros((2, total), dtype=np.float32)

    # Measure the master deck's phase first: everything else locks to it, so
    # it has to exist before the loop that uses it. Costs one extra prepare of
    # a single track, and buys a reference that a shared measurement bias
    # cancels out of.
    cache = {} if cache is None else cache
    _mseg = segs[master_pos]
    _mti = order[master_pos]
    ref_key = ("ref", _mti, _mseg["enter"], _mseg["bars"], round(bpm, 6))
    if ref_key in cache:
        ref_phase = cache[ref_key]
    else:
        _ma, _ = prepare(paths[_mti], metas[_mti], _mseg["enter"],
                         _mseg["bars"], bpm, sample_rate)
        ref_phase = beat_phase_error(_ma, sample_rate, bpm)
        cache[ref_key] = ref_phase
        del _ma

    stretches, ducks, syncs, stem_joins = [], [], [], []
    for seg in segs:
        ti, pos = seg["track"], seg["pos"]
        m = metas[ti]
        seg_bpm = float(seg["bpm"])
        if progress:
            progress(pos, len(segs), m["title"])

        head_tr = joins[pos - 1] if pos > 0 else None
        tail_tr = joins[pos] if pos < len(joins) else None
        want_stems = bool((head_tr and head_tr.stems) or
                          (tail_tr and tail_tr.stems))

        key = (ti, seg["enter"], seg["bars"], round(seg_bpm, 6),
               round(float(np.ptp(seg["profile"][1])), 6), want_stems)
        hit = cache.get(key)
        if hit is not None:
            # Copy: ducking subtracts a vocal in place, and a cached array
            # mutated once is silently wrong on every later render.
            a, bus_gain, used_stems = hit[0].copy(), hit[1], hit[2]
        else:
            a, bus_gain = prepare(paths[ti], m, seg["enter"], seg["bars"],
                                  seg_bpm, sample_rate)
            used_stems = False

            if want_stems:
                st = prepare_stems(paths[ti], m, seg["enter"], seg["bars"],
                                   seg_bpm, sample_rate, bus_gain)
                if st is not None:
                    nn = a.shape[1]
                    hn = int(round(seg["head"] * sample_rate *
                                   (4 * 60.0 / seg_bpm)))
                    tn = int(round(seg["tail"] * sample_rate *
                                   (4 * 60.0 / seg_bpm)))
                    a = stem_mix(st, nn, min(hn, nn),
                                 min(tn, max(0, nn - min(hn, nn))),
                                 head_tr.stems["in"] if (head_tr and
                                                         head_tr.stems) else None,
                                 tail_tr.stems["out"] if (tail_tr and
                                                          tail_tr.stems) else None)
                    used_stems = True

            # Sync: lock this deck's beatgrid to the master grid. The master
            # deck itself is already at the mix tempo and untouched by the
            # vocoder, so it needs no correction and must not be nudged.
            if pos != master_pos or glide:
                a, corr = sync_to(a, sample_rate, seg_bpm, ref_phase)
                if corr:
                    syncs.append({"track": m["title"],
                                  "shift_ms": round(corr, 2)})

            # The tempo glide happens LAST, after the segment is beat-locked
            # at a constant tempo. Warping first would move the beats this
            # measurement is trying to find.
            if not seg["flat"]:
                b, t = seg["profile"]
                a = warp_to_profile(a, sample_rate, seg_bpm, b, t)

            cache[key] = (a.copy(), bus_gain, used_stems)

        stretches.append(stretch.rate_for(m["bpm"], seg_bpm))
        if used_stems:
            stem_joins.append(m["title"])

        n = a.shape[1]
        head_n = min(seg["head_samples"], n)
        tail_n = min(seg["tail_samples"], max(0, n - head_n))
        head_stems = used_stems and head_tr is not None and bool(head_tr.stems)
        tail_stems = used_stems and tail_tr is not None and bool(tail_tr.stems)

        # Where the incoming track sings over this one's outro, step this
        # track's vocal aside. Subtracting a scaled vocal stem attenuates only
        # the voice: the kick, bass and synths under it are untouched, which no
        # EQ can do. Demucs stems sum back to the source, so
        # `a - (1-g)*vocals` is exactly `a` with the vocal at gain g.
        # Skipped when the stem layer already handled this tail: `stem_mix`
        # takes A's vocal out entirely rather than ducking it, and subtracting
        # a vocal that is no longer there would carve a hole in the shape of
        # the singer.
        if tail_n and activity is not None and pos < len(order) - 1 \
                and not tail_stems:
            nxt = order[pos + 1]
            in_act = activity.get(nxt)
            if in_act is not None and len(in_act):
                voc = prepare_vocal(paths[ti], m, seg["enter"], seg["bars"],
                                    seg_bpm, sample_rate, bus_gain)
                if voc is not None:
                    spb_seg = sample_rate * (4 * 60.0 / seg_bpm)
                    g = duck_curve(in_act, segs[pos + 1]["enter"], seg["tail"],
                                   spb_seg)
                    k = min(tail_n, voc.shape[1], len(g))
                    if k > 0:
                        a[:, n - k:] -= (1.0 - g[:k]) * voc[:, voc.shape[1] - k:]
                        ducks.append({
                            "join": pos, "track": m["title"],
                            "max_duck_db": round(float(
                                20 * np.log10(max(g.min(), 1e-6))), 2),
                        })

        pieces = []
        if head_n:
            tr = joins[pos - 1]
            # A stem-built head already carries its own per-stem gains. Running
            # the band automation over it as well would fade the incoming track
            # twice -- once by instrument and once by frequency -- and the
            # result arrives late and thin.
            pieces.append(a[:, :head_n] if head_stems else
                          _eq_region(a[:, :head_n], sample_rate, tr.in_low,
                                     tr.in_mid, tr.in_high, tr.in_sweep))
        mid_lo, mid_hi = head_n, n - tail_n
        if mid_hi > mid_lo:
            pieces.append(a[:, mid_lo:mid_hi])
        if tail_n:
            tr = joins[pos]
            tail_raw = a[:, n - tail_n:]
            tail = (tail_raw.copy() if tail_stems else
                    _eq_region(tail_raw, sample_rate, tr.out_low, tr.out_mid,
                               tr.out_high, tr.out_sweep))
            if tr.echo:
                s = int(tr.echo["start"] * tail_n)
                src = tail_raw[:, max(0, s - int(2 * beat_s * sample_rate)):s]
                wet = effects.echo_tail(src, sample_rate, tr.echo["delay_s"],
                                        tr.echo["feedback"])
                room = tail_n - s
                k = min(room, wet.shape[1])
                if k > 0:
                    fade = np.linspace(1.0, 0.0, k) ** 1.5
                    tail[:, s:s + k] += wet[:, :k] * fade * 0.7

            if tr.fx:
                # A JUCE effect over the outgoing tail, from `dsp/pedal.py`.
                # Skipped silently when pedalboard is absent -- the transition
                # is still built from its curves, so the mix is quieter than
                # intended rather than broken, and `REQUIRES` already tells the
                # user in the inspector what the move wants.
                from .dsp import pedal
                if pedal.AVAILABLE:
                    s = int(tr.fx.get("start", 0.0) * tail_n)
                    s = max(0, min(s, tail_n - 1))
                    region = tail[:, s:]
                    if region.shape[1] > 0:
                        wet = pedal.transition_fx(
                            region, sample_rate, tr.fx["kind"],
                            tr.fx.get("amount", 1.0))
                        k = min(region.shape[1], wet.shape[1])
                        tail[:, s:s + k] = wet[:, :k]

            if tr.roll:
                # REPLACE the run-up rather than layering over it: a roll is
                # the track eating its own tail, so the original audio in that
                # span must go, or the loop plays against what it is looping.
                s = int(tr.roll["start"] * tail_n)
                span = tail_n - s
                if span > 64:
                    loop_s = sample_rate * (4 * 60.0 / seg_bpm) / sample_rate
                    src = tail_raw[:, max(0, s - int(loop_s * sample_rate)):s]
                    if src.shape[1] > 64:
                        rolled = effects.loop_roll(
                            src, sample_rate, loop_s, span / sample_rate,
                            halve_every=tr.roll.get("halve_every", 2))
                        k = min(span, rolled.shape[1])
                        tail[:, s:s + k] = rolled[:, :k]

            if tr.riser:
                s = int(tr.riser["start"] * tail_n)
                span = tail_n - s
                if span > 64:
                    swell = effects.riser(sample_rate, span / sample_rate)
                    k = min(span, swell.shape[1])
                    tail[:, s:s + k] += swell[:, :k] * tr.riser.get("gain", 0.3)
            pieces.append(tail)

        placed = np.concatenate(pieces, axis=1) if pieces else a
        start = int(seg["start_sample"])
        stop = min(total, start + placed.shape[1])
        out[:, start:stop] += placed[:, :stop - start].astype(np.float32)

    trimmed = np.max(np.abs(out), axis=0) > 1e-5
    if trimmed.any():
        last = int(np.nonzero(trimmed)[0][-1])
        out = out[:, :min(total, last + sample_rate // 2)]

    out, rep = master.master(out, sample_rate)
    rep["vocal_ducks"] = ducks
    rep["sync_corrections"] = syncs
    rep["master_deck"] = metas[order[master_pos]]["title"]
    rep.update({
        "mix_bpm": round(bpm, 3),
        "duration_s": round(out.shape[1] / sample_rate, 1),
        "tracks": len(order),
        "max_stretch_pct": round(100 * max(abs(s - 1) for s in stretches), 3),
        "transitions": [j.name for j in joins],
        "transition_bars": list(lengths),
        "tempo_mode": tempo_mode,
        "deck_bpms": [round(float(t), 3) for t in tempos],
        "tempo_glide": bool(glide and glides),
        "max_glide_cents": round(max(
            [glide_cents(float(tempos[i]), float(tempos[i + 1]))
             for i in range(len(tempos) - 1)] or [0.0]), 1) if glides else 0.0,
        "stem_transitions": stem_joins,
        "auditioned": tried,
    })
    return out, rep, segs, joins


def preview_join(paths, metas, order, segs, join, name, bars, sample_rate=None,
                 lead_bars=8, trail_bars=8, bpm=None, cache=None):
    """Render just one transition, with a little music either side.

    The reason this exists is a measurement. Caching prepared segments took a
    re-render from 54 s to 34 s -- but 29 s of what remained was the master
    chain running over the whole fifteen-minute mix, which no amount of caching
    upstream can avoid. So for the edit loop, do not render the mix: render the
    join. Two slices, one transition, and a master pass over about half a
    minute of audio instead of fifteen.

    That turns "change the transition and listen" from a coffee break into
    something you can do repeatedly while deciding, which is the difference
    between a program that generates a mix and one you can actually work with.
    """
    sample_rate = sample_rate or audio_mod.RENDER_RATE
    bpm = bpm or float(segs[join].get("bpm") or master_deck(metas, order)[1])
    spb = sample_rate * (4 * 60.0 / bpm)
    beat_s = 60.0 / bpm
    cache = {} if cache is None else cache

    ti, tj = order[join], order[join + 1]
    exit_bar = max(0, segs[join]["exit"] - bars)
    enter_bar = segs[join + 1]["enter"]

    def slice_of(ti, meta, start_bar, n_bars):
        key = ("prev", ti, start_bar, n_bars, round(bpm, 6))
        if key not in cache:
            a, _ = audio_mod.load(paths[ti], sample_rate, mono=False)
            if a.shape[0] == 1:
                a = np.vstack([a[0], a[0]])
            a, _ = master.match_loudness(a, sample_rate, target=master.BUS_LUFS,
                                         max_peak=1.0)
            cache[key] = _retime_slice(a, meta, start_bar, n_bars, bpm,
                                       sample_rate)
        return cache[key]

    lead = max(0, min(lead_bars, exit_bar))
    a_part = slice_of(ti, metas[ti], exit_bar - lead, lead + bars)
    b_part = slice_of(tj, metas[tj], enter_bar, bars + trail_bars)

    n_lead = int(round(lead * spb))
    n_region = int(round(bars * spb))
    total = n_lead + n_region + int(round(trail_bars * spb))
    out = np.zeros((2, total), dtype=np.float32)

    out[:, :n_lead] += a_part[:, :n_lead]
    tr = transitions.build(name, bars, spb, sample_rate, beat_s)
    mixed = transitions.render(tr, a_part[:, n_lead:n_lead + n_region],
                               b_part[:, :n_region], sample_rate)
    k = min(n_region, mixed.shape[1])
    out[:, n_lead:n_lead + k] += mixed[:, :k]
    rest = b_part[:, n_region:]
    k = min(total - n_lead - n_region, rest.shape[1])
    if k > 0:
        out[:, n_lead + n_region:n_lead + n_region + k] += rest[:, :k]

    out, rep = master.master(out, sample_rate)
    rep.update({"join": join, "transition": name, "bars": bars,
                "region_start_s": round(n_lead / sample_rate, 3),
                "region_end_s": round((n_lead + n_region) / sample_rate, 3),
                "mix_bpm": round(bpm, 3)})
    return out, rep


def audition_joins(paths, metas, order, segs, seeds, lengths, tempos,
                   sample_rate, style="smooth", activity=None, progress=None):
    """Try each join's plausible transitions on the real audio, keep the best.

    Cheap because it only ever touches the region: a join is a few bars of two
    tracks, not two tracks. The expensive part is decoding, so each track's
    loudness-matched audio is held for the length of the pass and every join it
    takes part in reuses it.

    Returns (names, report) where report lists what was tried and what each
    candidate measured, so the choice can be shown rather than asserted.
    """
    from . import audition as aud

    decoded, names, report = {}, list(seeds), []

    def full(ti):
        if ti not in decoded:
            a, _ = audio_mod.load(paths[ti], sample_rate, mono=False)
            if a.shape[0] == 1:
                a = np.vstack([a[0], a[0]])
            decoded[ti] = master.match_loudness(a, sample_rate,
                                                target=master.BUS_LUFS,
                                                max_peak=1.0)[0]
        return decoded[ti]

    for p in range(len(order) - 1):
        seed = seeds[p]
        cands = aud.shortlist(seed, style)
        if len(cands) < 2:
            continue
        if progress:
            progress(p, len(order) - 1, f"auditioning join {p + 1}")

        L = lengths[p]
        bpm = float(tempos[p])
        ti, tj = order[p], order[p + 1]
        exit_bar = max(0, segs[p]["exit"] - L)
        enter_bar = segs[p + 1]["enter"]

        tail = _retime_slice(full(ti), metas[ti], exit_bar, L, bpm, sample_rate)
        head = _retime_slice(full(tj), metas[tj], enter_bar, L, bpm,
                             sample_rate)

        act = activity or {}
        rows, best, best_score = [], seed, None
        for name in cands:
            try:
                score, metrics = aud.evaluate(
                    name, tail, head, sample_rate, bpm, L,
                    act.get(ti), act.get(tj), exit_bar, enter_bar)
            except Exception:
                # A transition that cannot be built for this pair is simply not
                # a candidate. It must not take the whole render down with it.
                continue
            rows.append({"name": name, "score": round(score, 4),
                         "metrics": {k: round(v, 4)
                                     for k, v in metrics.items()}})
            if best_score is None or score < best_score:
                best, best_score = name, score

        if rows:
            names[p] = best
            report.append({"join": p, "seed": seed, "chose": best,
                           "candidates": rows})
    return names, report


def seg_seconds(seg, spb, sample_rate):
    """Where a segment starts, in seconds.

    Prefers the sample position computed during placement. With a tempo glide a
    bar is not a fixed number of samples, so `mix_start * spb` -- which was
    exact for years -- silently drifts once tempos vary, and every timestamp
    after the first join is wrong by a growing amount.
    """
    if "start_sample" in seg:
        return seg["start_sample"] / sample_rate
    return seg["mix_start"] * spb / sample_rate


def tracklist(metas, order, segs, joins, spb, sample_rate):
    """Timestamped tracklist."""
    lines = []
    for seg, ti in zip(segs, order):
        m = metas[ti]
        t = seg_seconds(seg, spb, sample_rate)
        nxt = joins[seg["pos"]].name if seg["pos"] < len(joins) else "-"
        lines.append(f"{int(t) // 60:02d}:{int(t) % 60:02d}  "
                     f"{m['artist']} - {m['title']}  "
                     f"[{m['bpm']:.2f} {m['camelot']}]  ->{nxt}")
    return "\n".join(lines)


def cue_sheet(metas, order, segs, spb, sample_rate, wav_name):
    """A .cue file so players show track boundaries."""
    out = [f'FILE "{wav_name}" WAVE']
    for i, (seg, ti) in enumerate(zip(segs, order), 1):
        m = metas[ti]
        t = seg_seconds(seg, spb, sample_rate)
        mm, ss = divmod(t, 60)
        frames = int((ss - int(ss)) * 75)
        out += [f"  TRACK {i:02d} AUDIO",
                f'    TITLE "{m["title"]}"',
                f'    PERFORMER "{m["artist"]}"',
                f"    INDEX 01 {int(mm):02d}:{int(ss):02d}:{frames:02d}"]
    return "\n".join(out)
