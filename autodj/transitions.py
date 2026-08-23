"""Transitions, expressed as automation over a region of N bars.

A transition is not a special signal path -- it is a set of gain curves applied
to the two tracks' EQ bands, plus optionally a moving filter or an echo tail.
Writing them declaratively, as breakpoints in bars, means a new transition is a
table of numbers rather than new DSP.
"""

from dataclasses import dataclass, field

import numpy as np

from .dsp import automation, effects, filters


@dataclass
class Transition:
    name: str
    bars: int
    out_low: np.ndarray
    out_mid: np.ndarray
    out_high: np.ndarray
    in_low: np.ndarray
    in_mid: np.ndarray
    in_high: np.ndarray
    out_sweep: tuple = None     # (btype, cutoff curve)
    in_sweep: tuple = None
    echo: dict = field(default_factory=dict)
    # Effects that need the actual audio, not just a gain curve. The renderer
    # reads these; a transition only describes what should happen.
    roll: dict = field(default_factory=dict)     # loop the outgoing track
    riser: dict = field(default_factory=dict)    # synthesised noise sweep
    # Per-stem gain curves, {"in": {stem: curve}, "out": {stem: curve}}. Set
    # only by transitions that mix by instrument instead of by frequency; the
    # renderer falls back to the band curves when the stems are not cached.
    stems: dict = field(default_factory=dict)
    # Pedalboard (JUCE) effects on the outgoing tail: {"kind", "amount",
    # "start"}. Optional in the strict sense -- if pedalboard is not installed
    # the renderer skips it and the transition still works from its curves
    # alone, quieter than intended but never broken.
    fx: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _c(bars, spb, points, shape="s_curve"):
    """Breakpoints given as (fraction-of-region, value)."""
    return automation.by_bar(bars, [(f * bars, v) for f, v in points], spb, shape)


def _bass_handover(bars, spb, at=0.5, width=0.02):
    """Low-band curves that hand over instead of crossfading.

    Crossfading the low band is the mistake. Any pair of fade shapes that meet
    in the middle leaves both basslines audible at once -- an ease-in/ease-out
    pair sits at 0.75 each halfway through -- and two basslines is exactly the
    mud we are trying to avoid. So the outgoing bass falls to zero *before* the
    incoming one starts to rise. The gap is a couple of hundredths of the
    region, which on a downbeat reads as a switch, not a hole.
    """
    out = _c(bars, spb, [(0, 1), (at - width, 1), (at, 0)], "ease_out")
    inn = _c(bars, spb, [(0, 0), (at, 0), (at + width, 1)], "ease_in")
    return out, inn


def bass_swap(bars, spb):
    """The workhorse. Never let two basslines play at once.

    Two competing basslines are the single loudest tell of an amateur mix: the
    low end turns to mud and the groove stops being readable. So the low band is
    handled as a *switch*, not a fade -- outgoing bass cuts and incoming bass
    enters within the same bar, on a downbeat. Mids and highs cross gradually
    around it, which is what makes the switch feel like a transition rather than
    an edit.
    """
    out_low, in_low = _bass_handover(bars, spb, at=0.5)
    return Transition(
        "bass_swap", bars,
        out_low=out_low,
        out_mid=_c(bars, spb, [(0, 1), (0.25, 1), (0.85, 0.12), (1, 0)]),
        out_high=_c(bars, spb, [(0, 1), (0.15, 0.95), (0.75, 0.25), (1, 0)]),
        in_low=in_low,
        in_mid=_c(bars, spb, [(0, 0), (0.15, 0.15), (0.65, 1), (1, 1)]),
        in_high=_c(bars, spb, [(0, 0.2), (0.4, 0.6), (0.8, 1), (1, 1)]),
    )


def eq_blend(bars, spb):
    """Gentle three-band crossfade for harmonically compatible tracks."""
    out_g, in_g = automation.equal_power(int(round(bars * spb)))
    out_low, in_low = _bass_handover(bars, spb, at=0.55)
    return Transition(
        "eq_blend", bars,
        out_low=out_low, out_mid=out_g, out_high=out_g,
        in_low=in_low, in_mid=in_g, in_high=in_g,
    )


def filter_sweep(bars, spb, sample_rate):
    """Outgoing track climbs away under a highpass; incoming opens up.

    The right choice when the two keys clash: a highpass strips the outgoing
    track's harmonic content, so there is nothing left to argue with the
    incoming key.
    """
    n = int(round(bars * spb))
    out_cut = automation.ramp(64, 20.0, 4000.0, "ease_in", power=3.0)
    in_cut = automation.ramp(64, 300.0, 12000.0, "ease_out", power=2.0)
    out_low, in_low = _bass_handover(bars, spb, at=0.5)
    return Transition(
        "filter_sweep", bars,
        out_low=out_low,
        out_mid=_c(bars, spb, [(0, 1), (0.7, 0.7), (1, 0)]),
        out_high=_c(bars, spb, [(0, 1), (0.7, 0.7), (1, 0)]),
        in_low=in_low,
        in_mid=_c(bars, spb, [(0, 0.1), (0.5, 0.7), (1, 1)]),
        in_high=_c(bars, spb, [(0, 0.1), (0.5, 0.7), (1, 1)]),
        out_sweep=("high", out_cut),
        in_sweep=("low", in_cut),
    )


def echo_out(bars, spb, beat_s):
    """Outgoing track is cut on a downbeat and left ringing in a delay."""
    out_low, in_low = _bass_handover(bars, spb, at=0.45)
    return Transition(
        "echo_out", bars,
        out_low=out_low,
        out_mid=_c(bars, spb, [(0, 1), (0.38, 1), (0.45, 0)], "ease_out"),
        out_high=_c(bars, spb, [(0, 1), (0.38, 1), (0.45, 0)], "ease_out"),
        in_low=in_low,
        in_mid=_c(bars, spb, [(0, 0), (0.2, 0.2), (0.6, 1), (1, 1)]),
        in_high=_c(bars, spb, [(0, 0.15), (0.5, 0.8), (1, 1)]),
        echo={"delay_s": beat_s * 0.75, "feedback": 0.6,
              "start": 0.35, "length": 0.25},
    )


def vocal_slam_drop(bars, spb, slam_at=0.62):
    """Cut the outgoing track dead on beat 1 of the incoming vocal.

    Only works when the incoming track opens on an isolated vocal: with no kick
    underneath, the outgoing track can run at full level right up to the slam
    instead of being politely faded out. The silence where track A was is the
    effect -- fade it and you lose the whole point.
    """
    out_all = _c(bars, spb, [(0, 1), (slam_at - 0.01, 1), (slam_at, 0)], "ease_out")
    return Transition(
        "vocal_slam_drop", bars,
        out_low=out_all, out_mid=out_all, out_high=out_all,
        # B's vocal is audible before the slam; its bass arrives with it.
        in_low=_c(bars, spb, [(0, 0), (slam_at, 0), (slam_at + 0.02, 1)], "ease_in"),
        in_mid=_c(bars, spb, [(0, 0.55), (slam_at - 0.02, 0.75), (slam_at, 1)]),
        in_high=_c(bars, spb, [(0, 0.5), (slam_at, 1)]),
    )


def euro_rap_breakout(bars, spb, sample_rate, hold_bars=8):
    """Rap vocal from B, highpassed over A's instrumental, then the bass drops.

    B runs through a strict 400 Hz highpass for exactly `hold_bars` bars, which
    strips its kick and bassline so the rap sits cleanly over A's groove with
    nothing competing underneath. When the filter opens, B's low end arrives all
    at once -- that is the drop.
    """
    frac = min(0.85, hold_bars / max(1, bars))
    steps = 64
    hold = int(round(frac * steps))
    cut = np.concatenate([np.full(max(1, hold), 400.0),
                          np.linspace(400.0, 20.0, max(1, steps - hold))])
    return Transition(
        "euro_rap_breakout", bars,
        out_low=_c(bars, spb, [(0, 1), (frac, 1), (frac + 0.03, 0)], "ease_out"),
        # Duck A's mids so the rap has room, without losing A's groove.
        out_mid=_c(bars, spb, [(0, 1), (0.1, 0.45), (frac, 0.45), (1, 0)]),
        out_high=_c(bars, spb, [(0, 1), (frac, 0.8), (1, 0)]),
        in_low=_c(bars, spb, [(0, 0), (frac, 0), (frac + 0.03, 1)], "ease_in"),
        in_mid=_c(bars, spb, [(0, 1), (1, 1)]),
        in_high=_c(bars, spb, [(0, 0.9), (1, 1)]),
        in_sweep=("high", cut),
        meta={"hold_bars": hold_bars, "hpf_hz": 400.0},
    )


CUT_AT = 0.5    # where in the region a hard cut fires, as a fraction

# How many bars BEFORE its first full bar an incoming track must be placed, so
# that it becomes audible exactly on that bar.
#
# For a blend the answer is the whole region: the incoming track fades up
# underneath the outgoing one for the entire transition, and its first full bar
# lands as the transition completes.
#
# For a cut it is only the part of the region before the cut fires. Getting
# this wrong is silent and sounds like bad beatmatching even though the tempo
# is perfect: with a 4-bar region the incoming track was becoming audible 2
# bars before its first full bar, so every cut landed mid-phrase.
def entry_lead(name, bars):
    if name in ("hard_cut", "cut_with_echo", "loop_roll", "riser_cut"):
        return int(round(bars * CUT_AT))
    if name == "vocal_slam_drop":
        return int(round(bars * 0.62))
    if name == "double_drop":
        # Only a fallback. A real double drop sets both entry and exit from the
        # two drop bars (see `render.segment_plan`); this is what happens when
        # one of the tracks has no usable drop and the move degrades to an
        # ordinary aligned overlap.
        return int(round(bars * DOUBLE_DROP_AT))
    return bars


def hard_cut(bars, spb, cut_at=CUT_AT, click_ms=4.0, sample_rate=44100):
    """A and B swap on a downbeat, both at full level. No blend at all.

    Every other transition here answers "how do we get from A to B without it
    being noticeable". This one answers the opposite question. The join is the
    effect: A is playing at full, then on the 1 it is simply gone and B is
    there. It is what megamixes are made of, and it is why they feel fast --
    nothing is ever politely faded.

    Two details keep it from sounding like a mistake rather than an edit:

    click   A true sample-accurate jump from a non-zero sample to another one
            is a step discontinuity, and a step is a click -- broadband energy
            across the whole spectrum. A few milliseconds of fade removes it
            and is far too short to hear as a fade. `click_ms` is that ramp,
            deliberately in milliseconds rather than bars: it must not scale
            with tempo or region length, because it is not musical, it is a
            de-clicker.

            The two ramps must *overlap*, centred on the cut. Running them
            back to back -- A down, then B up -- puts a hole where the join is:
            measured, summed gain hit exactly 0.0 for the width of the ramp.
            Crossfading them equal-power over the same few milliseconds keeps
            out^2 + in^2 == 1 through the join, so there is no dropout and no
            click. Both tracks are technically audible together for ~4 ms,
            which is far too short for two basslines to fight.

    phase   The cut has to land exactly on a downbeat or it sounds like a
            dropout. That is not this function's job -- it comes from the
            fitted beatgrid and the phrase snapping in the planner -- but it
            is the reason a hard cut is only safe in a project that got the
            beatgrid right first.

    Nothing crossfades, so there is no double-bassline problem to solve and no
    `_bass_handover` here: the low band switches with everything else.
    """
    n = int(round(bars * spb))
    ramp = max(2, int(click_ms * 1e-3 * sample_rate))
    half = ramp // 2
    cut = int(np.clip(round(cut_at * n), ramp, n - ramp))
    lo, hi = cut - half, cut - half + ramp

    fade_out, fade_in = automation.equal_power(ramp)

    out = np.ones(n, dtype=np.float64)
    out[hi:] = 0.0
    out[lo:hi] = fade_out

    inn = np.zeros(n, dtype=np.float64)
    inn[hi:] = 1.0
    inn[lo:hi] = fade_in

    return Transition(
        "hard_cut", bars,
        out_low=out, out_mid=out, out_high=out,
        in_low=inn, in_mid=inn, in_high=inn,
        meta={"cut_at": cut_at, "click_ms": click_ms},
    )


def cut_with_echo(bars, spb, beat_s, cut_at=0.5, sample_rate=44100):
    """A hard cut with A's last moment thrown into a delay.

    The bare cut can leave a hole when A's final bar ends on a sustained note
    that stops dead. Feeding the cut point into a feedback delay gives A a tail
    that decays across B's first bars -- the join stays hard, but the space it
    leaves is filled by A rather than by nothing.
    """
    tr = hard_cut(bars, spb, cut_at, sample_rate=sample_rate)
    tr.name = "cut_with_echo"
    tr.echo = {"delay_s": beat_s * 0.75, "feedback": 0.55,
               "start": cut_at - 0.02, "length": 0.22}
    return tr


def dissolve(bars, spb, at=0.5, width=0.03):
    """A cross-dissolve: both tracks sounding, one becoming the other.

    The Premiere cross-dissolve, done properly for music. Three things make it
    smooth rather than muddy:

    equal power   the mid and high bands cross on a sin/cos pair, so summed
                  power stays constant. A linear crossfade dips ~3 dB in the
                  middle, and that dip is heard as the mix sagging exactly
                  where it should feel strongest.

    bass handover the low band still switches rather than crossing. This is
                  the one place a dissolve must NOT dissolve: two basslines
                  overlapping is mud, no matter how gently they are mixed.

    no EQ carving unlike `bass_swap`, the mids and highs are not scooped on
                  the way out. A dissolve should sound like one track becoming
                  another, not like two tracks being processed.

    Short by default. The point is a smooth *join*, not a long blend -- at 133
    BPM four bars is about seven seconds, which is enough to feel seamless and
    short enough that the outgoing track never seems to be fading away.
    """
    n = int(round(bars * spb))
    out_g, in_g = automation.equal_power(n)
    out_low, in_low = _bass_handover(bars, spb, at=at, width=width)
    return Transition(
        "dissolve", bars,
        out_low=out_low, out_mid=out_g, out_high=out_g,
        in_low=in_low, in_mid=in_g, in_high=in_g,
        meta={"at": at},
    )


def smooth_swap(bars, spb, at=0.5):
    """A dissolve with a gentle high-band lead-in for the incoming track.

    Letting B's highs arrive slightly before its body is the oldest trick for
    making an entrance feel inevitable: the ear registers the new hats and air
    first, so by the time the full track lands it has already been introduced.
    Everything else is the dissolve, including the bass handover.
    """
    tr = dissolve(bars, spb, at=at)
    tr.name = "smooth_swap"
    tr.in_high = _c(bars, spb, [(0, 0.30), (0.35, 0.72), (0.7, 1), (1, 1)])
    tr.out_high = _c(bars, spb, [(0, 1), (0.3, 0.92), (0.75, 0.35), (1, 0)])
    return tr


def loop_roll(bars, spb, beat_s, cut_at=CUT_AT, roll_bars=2,
              sample_rate=44100):
    """The outgoing track eats its own tail, accelerating into the cut.

    Its last bar is looped, then a half bar, then a beat, then half a beat --
    the repeats speed up while the pitch stays exactly where it was, which is
    what separates a roll from a tape stop. Because the loop is taken from the
    outgoing track itself, it always fits the key and the groove.

    The cut is unchanged underneath: the roll is tension, and the hard cut is
    the release. Fading out after a roll wastes it entirely.
    """
    tr = hard_cut(bars, spb, cut_at, sample_rate=sample_rate)
    tr.name = "loop_roll"
    tr.roll = {"bars": roll_bars,
               "start": max(0.0, cut_at - roll_bars / max(bars, 1)),
               "halve_every": 2}
    return tr


def riser_cut(bars, spb, beat_s, cut_at=CUT_AT, rise_bars=4, gain=0.30,
              sample_rate=44100):
    """A noise sweep climbs under the outgoing track and lands on the cut.

    Synthesised to the exact length of the run-up, so it always resolves on the
    downbeat rather than near it. The point of a riser is that it makes a hard
    cut feel *prepared*: the ear is told something is coming, and the cut then
    answers it.
    """
    tr = hard_cut(bars, spb, cut_at, sample_rate=sample_rate)
    tr.name = "riser_cut"
    tr.riser = {"bars": rise_bars,
                "start": max(0.0, cut_at - rise_bars / max(bars, 1)),
                "gain": gain}
    return tr


def fade(bars, spb, at=0.5):
    """The plain crossfader move: one track down, the other up. No EQ at all.

    Deliberately the least clever transition here, and the only one that treats
    the two tracks as single objects rather than as three bands each. It is
    what a crossfader does, and there are joins where that is the right answer
    -- two tracks with little low end, an ambient passage, a transition where
    any audible processing would draw attention to itself.

    Equal power, not linear. A linear crossfade sums to about -3 dB in the
    middle, heard as the mix sagging exactly where it should feel strongest;
    the sin/cos pair holds summed power flat across the join.

    The low band still hands over rather than crossing. That is not an EQ move
    -- it is the one rule this file does not break for anybody, because two
    basslines at once is mud regardless of how gently they were mixed.
    """
    n = int(round(bars * spb))
    out_g, in_g = automation.equal_power(n)
    out_low, in_low = _bass_handover(bars, spb, at=at)
    return Transition(
        "fade", bars,
        out_low=out_low, out_mid=out_g, out_high=out_g,
        in_low=in_low, in_mid=in_g, in_high=in_g,
        meta={"at": at},
    )


def tremolo(bars, spb, beat_s, sample_rate=44100, depth=0.9, start=0.30,
            switch=0.82, octaves=2.0):
    """The outgoing track is chopped in time, faster and faster, then swaps.

    A gate opening and closing on the beat, accelerating from one cycle per
    beat to four. It is tension built out of nothing but gain: no filter, no
    pitch, no added sound, just the track being interrupted at an increasing
    rate until the interruptions stop meaning "track" and start meaning
    "something is about to happen".

    Two details decide whether it reads as an effect or as a fault:

    shape   a square gate is a click every time it opens. The gate here is a
            raised cosine, so each cycle is a swell rather than a switch, and
            it stays a tremolo instead of becoming a stutter edit.

    phase   the rate is swept by accumulating instantaneous frequency, not by
            picking a rate per bar. Stepping the rate makes the gate restart
            mid-cycle at every change, and the discontinuity is audible as a
            tick exactly on the bar lines where the ear is listening hardest.

    The modulation is on the outgoing track only. Gating both would just be a
    gated mix, and the point is that A is being taken apart while B arrives
    intact.
    """
    n = int(round(bars * spb))
    t = np.arange(n, dtype=np.float64) / sample_rate
    span = max(1e-6, n / sample_rate)
    progress = t / span

    f0 = 1.0 / max(1e-6, beat_s)                 # one cycle per beat
    freq = f0 * 2.0 ** (progress * octaves)      # ... rising to four
    phase = np.cumsum(freq) / sample_rate
    gate = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase))

    # The effect arrives gradually: full-depth gating from the first bar sounds
    # like a dropout, not like a build.
    ramp = np.clip((progress - start) / max(1e-6, switch - start), 0.0, 1.0)
    env = 1.0 - depth * ramp * (1.0 - gate)
    env[progress >= switch] = 0.0

    inn = _c(bars, spb, [(0, 0), (switch - 0.02, 0.25), (switch, 1), (1, 1)],
             "ease_in")
    out_low, in_low = _bass_handover(bars, spb, at=switch, width=0.015)
    return Transition(
        "tremolo", bars,
        out_low=np.minimum(env, out_low), out_mid=env, out_high=env,
        in_low=in_low, in_mid=inn, in_high=inn,
        meta={"depth": depth, "octaves": octaves, "switch": switch},
    )


# Which stem enters when, as breakpoints over the region. Drums first: the ear
# accepts a new rhythm section over an existing track almost immediately,
# because it locks to the grid it is already following. Harmony second, once
# the groove is established. Vocals last and latest -- a lead vocal is the one
# element that cannot share the space, so it arrives only when the track it is
# joining has already handed over.
STEM_ENTRY = {
    "drums":  [(0.0, 0.0), (0.08, 0.55), (0.35, 1.0), (1.0, 1.0)],
    "bass":   [(0.0, 0.0), (0.5, 0.0), (0.52, 1.0), (1.0, 1.0)],
    "other":  [(0.0, 0.0), (0.30, 0.35), (0.65, 1.0), (1.0, 1.0)],
    "vocals": [(0.0, 0.0), (0.55, 0.0), (0.80, 0.85), (1.0, 1.0)],
}
STEM_EXIT = {
    "drums":  [(0.0, 1.0), (0.45, 0.85), (0.75, 0.25), (1.0, 0.0)],
    "bass":   [(0.0, 1.0), (0.48, 1.0), (0.50, 0.0), (1.0, 0.0)],
    "other":  [(0.0, 1.0), (0.55, 0.7), (1.0, 0.0)],
    # A's vocal leaves early and completely. Two lead vocals is the collision
    # the whole stem layer exists to prevent, and here we can simply not play
    # one rather than duck it.
    "vocals": [(0.0, 1.0), (0.25, 0.9), (0.5, 0.0), (1.0, 0.0)],
}


def stem_blend(bars, spb):
    """Mix the two tracks stem by stem instead of band by band.

    Every other transition in this file works on frequency: split into low,
    mid and high, and automate the three. That is what a DJ mixer can do, and
    it is a proxy for what a DJ actually wants, which is control over
    *instruments*. A bass swap is a swap of the low band because the bass
    happens to live there -- along with the kick, the low end of the pads, and
    whatever else.

    With separated stems the proxy is unnecessary. The drums can hand over
    while the vocal stays, the harmony can cross while the bass switches
    cleanly on the downbeat, and A's vocal can be *removed* rather than ducked.
    None of that is achievable with an EQ at any setting.

    Falls back to a dissolve when either track has no cached stems, because
    separation is a 60-second job per track and a transition is not the place
    to discover it has not been run.
    """
    tr = dissolve(bars, spb)
    tr.name = "stem_blend"
    tr.stems = {
        "in": {k: _c(bars, spb, v) for k, v in STEM_ENTRY.items()},
        "out": {k: _c(bars, spb, v) for k, v in STEM_EXIT.items()},
    }
    return tr


DOUBLE_DROP_AT = 0.5


def double_drop(bars, spb, at=DOUBLE_DROP_AT, hold=0.30):
    """Both tracks' drops land on the same downbeat, and both play through it.

    The most exposed move in the book, and the only one here whose success is
    decided entirely outside this function. Everything else in this module is a
    join between two tracks; a double drop is a *coincidence* between two
    specific bars, and if those bars are half a beat apart it is not a slightly
    worse double drop, it is a train wreck. The alignment happens in
    `render.segment_plan`, which sets A's exit and B's entry so the two drop
    bars fall on the same mix bar. This only shapes what happens around it.

    Three rules keep two full-energy tracks from turning into noise:

    bass      still a handover, and here it matters more than anywhere else.
              Two basslines is mud in any transition; two basslines *at a
              drop*, where both are at their loudest, is the single worst
              sound this program can make. A's low end is gone before B's
              arrives, on the drop itself.

    hold      after the drop both tracks stay up together for a few bars --
              `hold` as a fraction of the region -- and that overlap is the
              whole effect. Fade A immediately and you have an ordinary cut
              that happened to be well timed.

    lead-in   B's mids and highs come up over the bars *before* the drop, so
              its build is audible as a build. The ear needs to hear the run-up
              belonging to the new track, or the second drop arrives from
              nowhere.
    """
    end = min(1.0, at + hold)
    out_low, in_low = _bass_handover(bars, spb, at=at, width=0.015)
    return Transition(
        "double_drop", bars,
        out_low=out_low,
        # A holds full through the drop, sits back slightly while both play,
        # then leaves. It never ducks *at* the drop -- that is the moment.
        out_mid=_c(bars, spb, [(0, 1), (at, 1), (end, 0.72), (1, 0)]),
        out_high=_c(bars, spb, [(0, 1), (at, 1), (end, 0.7), (1, 0)]),
        in_low=in_low,
        # B's build is audible before the drop, and it is at full on it.
        in_mid=_c(bars, spb, [(0, 0.28), (at - 0.12, 0.62), (at, 1), (1, 1)]),
        in_high=_c(bars, spb, [(0, 0.35), (at - 0.12, 0.7), (at, 1), (1, 1)]),
        meta={"at": at, "hold": hold},
    )


def reverb_wash(bars, spb):
    """The outgoing track dissolves into its own reverb; the incoming enters dry.

    The move an echo-out cannot make. An echo repeats what was just played, so
    the outgoing track stays recognisable all the way down and keeps competing
    with the incoming one. A reverb tail smears it into unpitched wash instead
    -- it stops being a melody and becomes texture, which is why this is the
    safest thing to do across a hard key clash where even a highpass leaves
    too much.

    Needs pedalboard. A convolution-quality reverb is genuinely hard to write
    well, and the scipy one this project would otherwise have is not worth
    hearing next to JUCE's.
    """
    out_low, in_low = _bass_handover(bars, spb, at=0.42)
    return Transition(
        "reverb_wash", bars,
        out_low=out_low,
        # The dry outgoing signal leaves early, at 0.55, while the wet tail
        # keeps ringing underneath. Holding the dry signal to the end would
        # put the reverb on top of the track it is supposed to replace.
        out_mid=_c(bars, spb, [(0, 1), (0.35, 0.9), (0.55, 0.35), (1, 0)]),
        out_high=_c(bars, spb, [(0, 1), (0.3, 0.75), (0.55, 0.2), (1, 0)]),
        in_low=in_low,
        in_mid=_c(bars, spb, [(0, 0), (0.45, 0.25), (0.75, 0.85), (1, 1)]),
        in_high=_c(bars, spb, [(0, 0.05), (0.4, 0.4), (0.75, 0.9), (1, 1)]),
        fx={"kind": "reverb_tail", "amount": 0.85, "start": 0.30},
    )


def ladder_sweep(bars, spb, sample_rate):
    """`filter_sweep`, but the filter is a resonant 24 dB/oct ladder.

    Our own sweep is a Butterworth: clean, flat, and characterless, which is
    right when the filter is doing a job and wrong when it *is* the move. A
    ladder filter self-emphasises at the cutoff, so the sweep sings as it
    climbs -- the sound a live DJ gets from a mixer's filter knob and the
    reason people reach for it.

    The band curves stay identical to `filter_sweep`. The difference is
    entirely in the filter, which keeps the two comparable: if this sounds
    better it is the ladder, not a different fade.
    """
    n = int(round(bars * spb))
    out_cut = automation.ramp(64, 20.0, 4000.0, "ease_in", power=3.0)
    in_cut = automation.ramp(64, 300.0, 12000.0, "ease_out", power=2.0)
    out_low, in_low = _bass_handover(bars, spb, at=0.5)
    return Transition(
        "ladder_sweep", bars,
        out_low=out_low,
        out_mid=_c(bars, spb, [(0, 1), (0.7, 0.7), (1, 0)]),
        out_high=_c(bars, spb, [(0, 1), (0.7, 0.7), (1, 0)]),
        in_low=in_low,
        in_mid=_c(bars, spb, [(0, 0.1), (0.5, 0.7), (1, 1)]),
        in_high=_c(bars, spb, [(0, 0.1), (0.5, 0.7), (1, 1)]),
        out_sweep=("high", out_cut),
        in_sweep=("low", in_cut),
        fx={"kind": "ladder_lowpass", "amount": 0.7, "start": 0.15},
    )


def build(name, bars, spb, sample_rate, beat_s):
    """Factory."""
    if name == "reverb_wash":
        return reverb_wash(bars, spb)
    if name == "ladder_sweep":
        return ladder_sweep(bars, spb, sample_rate)
    if name == "double_drop":
        return double_drop(bars, spb)
    if name == "fade":
        return fade(bars, spb)
    if name == "tremolo":
        return tremolo(bars, spb, beat_s, sample_rate=sample_rate)
    if name == "stem_blend":
        return stem_blend(bars, spb)
    if name == "dissolve":
        return dissolve(bars, spb)
    if name == "smooth_swap":
        return smooth_swap(bars, spb)
    if name == "loop_roll":
        return loop_roll(bars, spb, beat_s, sample_rate=sample_rate)
    if name == "riser_cut":
        return riser_cut(bars, spb, beat_s, sample_rate=sample_rate)
    if name == "hard_cut":
        return hard_cut(bars, spb, sample_rate=sample_rate)
    if name == "cut_with_echo":
        return cut_with_echo(bars, spb, beat_s, sample_rate=sample_rate)
    if name == "bass_swap":
        return bass_swap(bars, spb)
    if name == "eq_blend":
        return eq_blend(bars, spb)
    if name == "filter_sweep":
        return filter_sweep(bars, spb, sample_rate)
    if name == "echo_out":
        return echo_out(bars, spb, beat_s)
    if name == "vocal_slam_drop":
        return vocal_slam_drop(bars, spb)
    if name == "euro_rap_breakout":
        return euro_rap_breakout(bars, spb, sample_rate)
    raise ValueError(f"unknown transition: {name}")


NAMES = ("bass_swap", "eq_blend", "filter_sweep", "echo_out",
         "vocal_slam_drop", "euro_rap_breakout", "hard_cut", "cut_with_echo",
         "loop_roll", "riser_cut", "dissolve", "smooth_swap", "double_drop",
         "fade", "tremolo", "stem_blend", "reverb_wash", "ladder_sweep")

# Moves that need something specific from the tracks and cannot simply be
# forced onto any join. The GUI still offers them -- overriding a transition by
# hand is the point of the inspector -- but it says what they need, and the
# renderer degrades them gracefully rather than producing a mess.
REQUIRES = {
    "double_drop": "both tracks need a detected drop",
    "vocal_slam_drop": "the incoming track should open on a vocal",
    "euro_rap_breakout": "suits a rap or spoken intro on the incoming track",
    "reverb_wash": "needs pedalboard installed for the reverb tail",
    "ladder_sweep": "needs pedalboard installed for the resonant filter",
}

def overlap_of(name, bars=8, sample_rate=44100, bpm=133.0):
    """How much of the region has both tracks genuinely up, from the curves.

    The mean of `min(out_mid, in_mid)`: zero when only one track is ever
    audible, rising as the two spend more time sounding together at level.
    """
    spb = sample_rate * (4 * 60.0 / bpm)
    tr = build(name, bars, spb, sample_rate, 60.0 / bpm)
    o = np.asarray(tr.out_mid, dtype=float)
    i = np.asarray(tr.in_mid, dtype=float)
    k = min(o.size, i.size)
    return float(np.mean(np.minimum(o[:k], i[:k]))) if k else 0.0


OVERLAP_FLOOR = 0.05

# Transitions that never have both tracks meaningfully audible at once. A key
# clash cannot be heard across one of these, so harmonic distance matters far
# less when picking one -- which is why megamixes get away with sets a
# long-blend set could not use.
#
# Computed from the curves rather than listed by hand. The hand-written version
# had drifted into being wrong: it named `dissolve` and `smooth_swap`, which
# measure 0.37 overlap -- both tracks are plainly sounding together for a third
# of the region. It was never read by anything, so the error was invisible;
# deriving it means it cannot go stale when a transition is added or reshaped.
NON_OVERLAPPING = tuple(n for n in NAMES if overlap_of(n) < OVERLAP_FLOOR)


def apply_side(audio, sample_rate, low, mid, high, sweep=None):
    """Apply one side's EQ automation (and optional filter sweep)."""
    a = np.atleast_2d(audio).astype(np.float64)
    n = a.shape[1]
    if sweep is not None:
        btype, cutoffs = sweep
        a = filters.sweep(a, sample_rate, cutoffs, btype=btype)
        a = np.atleast_2d(a).astype(np.float64)
    bands = filters.split(a, sample_rate)
    return filters.apply_eq(a, sample_rate,
                            automation.fit(low, n), automation.fit(mid, n),
                            automation.fit(high, n), bands=bands)


def render(tr, out_audio, in_audio, sample_rate):
    """Mix the two tracks through the transition. Both arrays must be the same
    length as the region."""
    n = min(np.atleast_2d(out_audio).shape[1], np.atleast_2d(in_audio).shape[1])
    o = np.atleast_2d(out_audio)[:, :n]
    i = np.atleast_2d(in_audio)[:, :n]

    o_wet = apply_side(o, sample_rate, tr.out_low, tr.out_mid, tr.out_high,
                       tr.out_sweep)
    i_wet = apply_side(i, sample_rate, tr.in_low, tr.in_mid, tr.in_high,
                       tr.in_sweep)

    mixed = o_wet[:, :n] + i_wet[:, :n]

    if tr.echo:
        s = int(tr.echo["start"] * n)
        e = min(n, s + int(tr.echo["length"] * n))
        tail = effects.echo_tail(o[:, max(0, s - int(0.5 * (e - s))):s],
                                 sample_rate, tr.echo["delay_s"],
                                 tr.echo["feedback"])
        room = n - s
        if room > 0 and tail.shape[1] > 0:
            k = min(room, tail.shape[1])
            fade = np.linspace(1.0, 0.0, k) ** 1.5
            mixed[:, s:s + k] += tail[:, :k] * fade * 0.8
    return mixed.astype(np.float32)
