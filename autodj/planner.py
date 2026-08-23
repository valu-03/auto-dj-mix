"""Choosing which tracks to play, in what order, and how to join them.

This is the difference between a playlist and a set. A playlist is tracks you
like; a set is tracks arranged so each one makes sense after the last, along an
energy arc that goes somewhere.
"""

import numpy as np

from .analysis import key as key_mod
from .dsp import stretch

W_KEY = 1.0
W_BPM = 0.8
W_ENERGY = 0.6
W_TIMBRE = 0.25
W_ARC = 1.2

MAX_BPM_SPREAD = 0.06     # matches stretch.SAFE_STRETCH

# --- hard rules, not preferences -------------------------------------------
# Camelot: same key, +/-1, or relative major/minor are always allowed. A +2
# jump ("energy flash") is a real technique but only lands as a lift near the
# peak; anywhere else it reads as a mistake.
LEGAL_KEY_DISTANCE = 1.0
FLASH_KEY_DISTANCE = 2.0
FLASH_WINDOW = (0.55, 0.85)      # fraction of the set where a flash is allowed

MAX_BPM_JUMP = 3.0               # absolute BPM between consecutive tracks
VIOLATION_PENALTY = 50.0         # dwarfs any legal cost, without hard-failing

PHRASE_LENGTHS = (4, 8, 16, 32)  # legal transition lengths, in bars

ARCS = {
    "warmup_to_peak": dict(peak_at=0.78, low=-1.1, high=1.15, flatten=0.7),
    "linear_rise": dict(peak_at=1.0, low=-1.2, high=1.2, flatten=1.0),
    "waveform": dict(peak_at=0.5, low=-0.9, high=1.1, flatten=0.5, waves=2),
    "steady_peak": dict(peak_at=0.35, low=-0.8, high=1.0, flatten=0.35),
}


def key_move_legal(a_cam, b_cam, position=None, n=None):
    """Is this key change allowed? `position`/`n` gate the +2 energy flash."""
    d = key_mod.camelot_distance(a_cam, b_cam)
    if d <= LEGAL_KEY_DISTANCE:
        return True
    if abs(d - FLASH_KEY_DISTANCE) < 1e-9 and position is not None and n:
        frac = position / max(1, n - 1)
        return FLASH_WINDOW[0] <= frac <= FLASH_WINDOW[1]
    return False


def snap_phrase(bars):
    """Round a transition length to the nearest legal phrase."""
    return min(PHRASE_LENGTHS, key=lambda p: (abs(p - bars), p))


def intensity(metas):
    """One cross-track energy number per track, z-scored over the actual set.

    The per-track `energy_curve` cannot be used here -- it is normalised inside
    each track, so every track peaks at 1.0. These are the absolute measures,
    standardised across whatever set is being mixed, which is the only way
    "harder than" means anything.
    """
    def z(x):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-9)

    loud = z([m["loudness"] for m in metas])
    dens = z([m["density"] for m in metas])
    low = z([m["low_level"] for m in metas])
    return 0.40 * loud + 0.35 * dens + 0.25 * low


def arc(n, shape="warmup_to_peak", peak_at=None):
    """Target intensity for each position in the set."""
    cfg = dict(ARCS.get(shape, ARCS["warmup_to_peak"]))
    if peak_at is not None:
        cfg["peak_at"] = peak_at
    if n <= 1:
        return np.array([cfg["high"]])

    t = np.linspace(0.0, 1.0, n)
    p = float(np.clip(cfg["peak_at"], 0.05, 1.0))
    if p >= 0.999:
        base = t                                   # linear rise, no come-down
    else:
        base = np.minimum(t / p, (1.0 - t) / (1.0 - p))
    base = np.clip(base, 0.0, 1.0) ** cfg.get("flatten", 0.7)

    waves = cfg.get("waves")
    if waves:
        # Peaks-and-troughs: ride the arc but dip between peaks, which is what
        # a long set actually does rather than one relentless climb.
        base = np.clip(base - 0.22 * (1 - np.cos(2 * np.pi * waves * t)) / 2,
                       0.0, 1.0)
    return cfg["low"] + (cfg["high"] - cfg["low"]) * base


def violations(a, b, position=None, n=None):
    """Which hard rules this join breaks. Empty list means it is legal."""
    bad = []
    if not key_move_legal(a["camelot"], b["camelot"], position, n):
        bad.append(f"key {a['camelot']}->{b['camelot']} "
                   f"(distance {key_mod.camelot_distance(a['camelot'], b['camelot']):.1f})")
    jump = abs(b["bpm"] - a["bpm"])
    if jump > MAX_BPM_JUMP:
        bad.append(f"bpm jump {jump:.2f} > {MAX_BPM_JUMP}")
    return bad


def transition_cost(a, b, inten_a, inten_b, position=None, n=None,
                    style="blend"):
    """How awkward is it to go from track a to track b?

    In "cut" style the harmonic terms are scaled right down. That is not a
    loosened standard, it is a different physical situation: a hard cut never
    has both tracks sounding together, so there is no interval for two keys to
    form and nothing for the Camelot distance to measure. Tempo still matters
    (the grid has to carry across the join) and energy still matters, so those
    keep their full weight.
    """
    kd = key_mod.camelot_distance(a["camelot"], b["camelot"])
    bpm_pct = abs(b["bpm"] - a["bpm"]) / a["bpm"]
    # Past the safe stretch window the cost climbs steeply, not linearly:
    # a 3% jump is a non-event, a 9% jump is a different genre.
    bpm_pen = bpm_pct / MAX_BPM_SPREAD
    if bpm_pct > MAX_BPM_SPREAD:
        bpm_pen += 6.0 * (bpm_pct - MAX_BPM_SPREAD) / MAX_BPM_SPREAD

    drop = max(0.0, inten_a - inten_b)        # falling energy costs more
    rise = max(0.0, inten_b - inten_a)
    energy_pen = 0.8 * drop + 0.3 * rise

    timbre = abs(a["brightness"] - b["brightness"]) / 1500.0

    key_w = W_KEY * (0.15 if style == "cut" else 1.0)
    cost = (key_w * kd + W_BPM * bpm_pen + W_ENERGY * energy_pen
            + W_TIMBRE * timbre)
    # Hard rules enter as a penalty rather than an exclusion: with a small or
    # awkward pool, refusing outright would leave us with no set at all. A
    # penalty this large means any legal ordering beats any illegal one, while
    # still producing the least-bad answer when nothing legal exists.
    bad = violations(a, b, position, n)
    if style == "cut":
        bad = [v for v in bad if not v.startswith("key")]
    cost += VIOLATION_PENALTY * len(bad)
    return cost


def total_cost(order, metas, inten, target, style="blend"):
    """Transition cost along the set, plus how far it strays from the arc."""
    n = len(order)
    c = 0.0
    for pos, (x, y) in enumerate(zip(order, order[1:])):
        c += transition_cost(metas[x], metas[y], inten[x], inten[y], pos, n,
                             style)
    c += W_ARC * float(np.abs(inten[list(order)] - target).sum())
    return c


def _greedy(metas, inten, target, start, style="blend"):
    """Nearest-neighbour seed: always take the cheapest legal next track."""
    n = len(metas)
    order = [start]
    left = set(range(n)) - {start}
    while left:
        cur = order[-1]
        pos = len(order)
        best = min(left, key=lambda j: (
            transition_cost(metas[cur], metas[j], inten[cur], inten[j],
                            pos - 1, n, style)
            + W_ARC * abs(inten[j] - target[pos])))
        order.append(best)
        left.discard(best)
    return order


def two_opt(order, metas, inten, target, rounds=40, style="blend"):
    """Reverse any segment that lowers the total cost, until nothing helps.

    Greedy ordering paints itself into corners -- it takes a cheap step early
    and pays for it at the end. 2-opt repairs exactly that: reversing a run of
    tracks re-links two joins while leaving everything else intact.
    """
    order = list(order)
    best = total_cost(order, metas, inten, target, style)
    for _ in range(rounds):
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                cand = order[:i + 1] + order[i + 1:j + 1][::-1] + order[j + 1:]
                c = total_cost(cand, metas, inten, target, style)
                if c < best - 1e-9:
                    order, best, improved = cand, c, True
        if not improved:
            break
    return order, best


def filter_pool(metas, bpm_window=MAX_BPM_SPREAD, anchor=None):
    """Keep only tracks that can actually be mixed together.

    With 3,000 tracks you do not order the library, you pick a pool first. A set
    is 15-30 tracks; ordering the whole library would be both slow and pointless
    because most pairs are unmixable anyway.
    """
    if not metas:
        return []
    bpms = np.array([m["bpm"] for m in metas])
    centre = anchor if anchor is not None else float(np.median(bpms))
    keep = [m for m in metas
            if abs(m["bpm"] - centre) / centre <= bpm_window]
    return keep or list(metas)


def plan(metas, shape="warmup_to_peak", peak_at=None, start=None,
         style="blend"):
    """Order a set. Returns (order, cost, intensity, target)."""
    n = len(metas)
    if n == 0:
        return [], 0.0, np.zeros(0), np.zeros(0)
    if n == 1:
        return [0], 0.0, np.zeros(1), np.zeros(1)

    inten = intensity(metas)
    target = arc(n, shape, peak_at)

    starts = [start] if start is not None else list(range(n))
    best = None
    for s in starts:
        order = _greedy(metas, inten, target, s, style)
        order, cost = two_opt(order, metas, inten, target, style=style)
        if best is None or cost < best[1]:
            best = (order, cost)
    return best[0], best[1], inten, target


EURODANCE_BPM = (128.0, 146.0)


def is_eurodance(m):
    """Tag heuristic. ID3 genre when present, tempo and density otherwise."""
    genre = (m.get("genre") or "").lower()
    if any(k in genre for k in ("eurodance", "euro-dance", "euro dance",
                                "hi-nrg", "italo")):
        return True
    return (EURODANCE_BPM[0] <= m["bpm"] <= EURODANCE_BPM[1]
            and m.get("density", 0) > 0.7)


DOUBLE_DROP_KEY = 1.0        # both tracks sound at full: harmony is audible
DOUBLE_DROP_ENERGY = 0.55    # how closely matched the two must be
DOUBLE_DROP_FLOOR = -0.15    # neither may be a low-energy track


def usable_drop(m, side):
    """The drop bar this track could contribute, or None.

    Not every detected drop can be used. The outgoing track needs a drop with
    enough music left after it to cover the second half of the region, and the
    incoming track needs one far enough in that there are bars to lead up to
    it. A drop in the last eight bars of a track is real and useless.
    """
    cues = m.get("cues") or {}
    drops = [int(d) for d in (cues.get("drop_bars") or [])]
    if not drops:
        return None
    n = int(m.get("n_bars") or 0)
    first = int(cues.get("first_full_bar", 0))
    if side == "out":
        ok = [d for d in drops if first + 8 <= d <= n - 12]
        return max(ok) if ok else None
    # The incoming side only needs enough bars before the drop to hold half a
    # region. A track whose drop is at bar 8 is not disqualified -- it simply
    # enters at bar 0 and uses its own intro as the build, which is what a DJ
    # would do with it anyway. An earlier floor of 12 bars rejected exactly
    # that case and left the move unusable on most of the library.
    ok = [d for d in drops if d >= 4 and d <= n - 16]
    return min(ok) if ok else None


def can_double_drop(a, b, inten_a, inten_b, kd=None):
    """Whether this pair can actually land a double drop."""
    if kd is None:
        kd = key_mod.camelot_distance(a["camelot"], b["camelot"])
    if kd > DOUBLE_DROP_KEY:
        return False
    if min(inten_a, inten_b) < DOUBLE_DROP_FLOOR:
        return False
    if abs(inten_a - inten_b) > DOUBLE_DROP_ENERGY:
        return False
    return (usable_drop(a, "out") is not None
            and usable_drop(b, "in") is not None)


def choose_transition(a, b, inten_a, inten_b, style="blend"):
    """Which transition suits this particular join.

    `style` picks the vocabulary, not the taste:

    "blend"  the default. Long overlapping transitions, so harmonic distance
             decides almost everything -- two keys really are sounding at once
             for sixteen bars.

    "cut"    megamix vocabulary. Joins are hard cuts on the 1, with A and B
             never audible together for more than a few milliseconds. Nothing
             overlaps, so nothing can clash, and the key rules that dominate
             "blend" simply stop applying. This is how a professional megamix
             gets away with a set like 8A/3A/3B/10A, which the blend planner
             refuses outright.
    """
    kd = key_mod.camelot_distance(a["camelot"], b["camelot"])
    euro = is_eurodance(a) and is_eurodance(b)

    # A double drop is checked before the per-style vocabulary, because it is
    # not a style preference -- it is a rare opportunity. Both tracks must have
    # a detected drop, be harmonically close enough to sound together at full
    # level, and be at comparable energy: dropping a quiet track against a loud
    # one is not a double drop, it is one track with a ghost behind it.
    if can_double_drop(a, b, inten_a, inten_b, kd):
        return "double_drop"

    if style == "smooth":
        # Everything overlaps and nothing is carved. Key still matters here --
        # unlike a cut, two tracks really are sounding together -- so a clash
        # falls back to the filter sweep, which strips the outgoing harmonics
        # rather than letting them argue.
        if kd > 1.0:
            return "filter_sweep"
        if euro and b.get("vocal_forward_intro"):
            return "smooth_swap"
        if abs(inten_a - inten_b) > 0.7:
            return "smooth_swap"
        return "dissolve"

    if style == "cut":
        # Vocabulary, chosen by what the join actually needs. Variety matters
        # here for its own sake -- he has twice preferred a mix with more
        # different moves over a more "correct" but uniform one -- but each of
        # these is still picked for a reason, not at random.
        if euro and b.get("vocal_forward_intro"):
            # B opens on a bare vocal: nothing to compete with, so cut dead.
            return "vocal_slam_drop"
        if inten_b - inten_a > 0.5:
            # Stepping UP in energy: earn it. A roll builds tension across the
            # last bars so the cut lands as a release rather than a surprise.
            return "loop_roll"
        if inten_b > 0.2 and abs(inten_a - inten_b) <= 0.5:
            # Level or nearly level at a decent energy: a riser announces the
            # change that the tracks themselves are not signalling.
            return "riser_cut"
        if inten_a - inten_b > 0.5:
            # A sustained outro left hanging by a bare cut leaves a hole; an
            # echo tail fills it with A instead of with silence.
            return "cut_with_echo"
        return "hard_cut"

    # Eurodance rules take priority over smooth blends: in this genre the
    # high-impact move is the point, and a polite crossfade wastes the drop.
    if euro and b.get("vocal_forward_intro") and kd <= 1.0:
        return "vocal_slam_drop"
    if euro and b.get("intro_vocal_ratio", 0) > 1.15 and kd <= 1.0 \
            and inten_b >= inten_a - 0.2:
        return "euro_rap_breakout"

    if kd > 1.0:
        # Keys clash: a highpass strips the outgoing harmonics so there is
        # nothing left to argue with the incoming key.
        return "filter_sweep"
    if inten_a - inten_b > 0.8:
        # Big energy drop: punctuate it rather than sliding down.
        return "echo_out"
    if kd == 0.0 and abs(inten_a - inten_b) < 0.35:
        return "eq_blend"
    return "bass_swap"


def bars_for_seconds(seconds, bpm):
    """A duration in seconds, expressed as a legal phrase length in bars.

    Seconds are how long a transition feels; bars are what it has to be. A
    transition that is not a whole number of bars starts or ends mid-phrase,
    and mid-phrase is audible as a mistake no matter how good the beatmatching
    is -- so a request in seconds is honoured as closely as a phrase allows
    rather than literally. At 133 BPM the available lengths are roughly 7, 14,
    29 and 58 seconds, which is coarser than a seconds control implies; the
    caller should show what was actually chosen.
    """
    bar_s = 4 * 60.0 / max(1e-6, bpm)
    return snap_phrase(max(1.0, float(seconds) / bar_s))


def auto_bars(a, b, inten_a, inten_b, name, floor=4, ceiling=32):
    """How long this particular join should be, in bars.

    One length for a whole set is the wrong shape of answer. A hard cut is an
    event and wants the shortest legal region; a blend between two similar
    tracks wants long enough to actually blend; a big energy drop wants time to
    land rather than being rushed. Choosing per join is what "automatic
    duration" means, and it is only possible because the region length is now
    per join rather than global.
    """
    if name in ("hard_cut", "cut_with_echo", "loop_roll", "riser_cut",
                "vocal_slam_drop", "tremolo"):
        # Nothing overlaps for long, so a longer region only delays the event.
        bars = 4 if name in ("hard_cut", "cut_with_echo") else 8
    elif name == "double_drop":
        # Long enough for B's build to be heard as a build, short enough that
        # the two tracks are not playing together for half a minute.
        bars = 16
    elif name in ("stem_blend", "euro_rap_breakout"):
        bars = 16
    elif name in ("dissolve", "smooth_swap", "fade"):
        bars = 8
    else:
        bars = 16

    gap = abs(inten_a - inten_b)
    kd = key_mod.camelot_distance(a["camelot"], b["camelot"])
    if gap > 1.0 and bars >= 8:
        # Two tracks far apart in energy sound wrong together; get it over with.
        bars //= 2
    elif gap < 0.3 and kd == 0.0 and bars >= 8:
        # Same key, same energy: there is nothing to hide, so take the time.
        bars *= 2
    return int(np.clip(snap_phrase(bars), floor, ceiling))


def _boundaries(m):
    """Structural boundary bars for a track, as a sorted array."""
    segs = (m.get("cues") or {}).get("segments") or []
    edges = {int(s.get("start_bar", 0)) for s in segs}
    edges |= {int(s.get("end_bar", 0)) for s in segs}
    cues = m.get("cues") or {}
    edges |= {int(cues.get("first_full_bar", 0)),
              int(cues.get("outro_start_bar", 0))}
    edges |= {int(d) for d in (cues.get("drop_bars") or [])}
    return np.array(sorted(e for e in edges if e >= 0), dtype=float)


def _edge_miss(m, bar, cap=8.0):
    """How far a bar sits from the nearest structural boundary, in bars."""
    edges = _boundaries(m)
    if edges.size == 0:
        return cap
    return float(min(cap, np.min(np.abs(edges - float(bar)))))


def fit_bars(a, b, inten_a, inten_b, name, act_a=None, act_b=None,
             lengths=PHRASE_LENGTHS, floor=4, ceiling=32):
    """Choose a join's length by measuring both tracks, not by a lookup table.

    "Automatic duration" as a table of genre rules is a guess made without
    looking at the music. Offline, there is no reason to guess: both tracks are
    fully analysed before a single sample is rendered, so the length can be
    *fitted* to what is actually there.

    Four things are scored for every legal length:

    structure   where the region starts and ends in each track, against that
                track's own section boundaries. A 16-bar blend that begins
                four bars into A's breakdown is worse than an 8-bar one that
                starts exactly on it, and the difference is audible as the
                transition seeming to begin at a random moment.

    vocals      the measured lead-voice overlap across the region, when stems
                or the spectral proxy are available. A longer region is not
                safer -- it is more time for two singers to collide.

    energy      how far apart the two tracks sit across the region. Two tracks
                a long way apart in energy should be joined quickly; two that
                match can afford to sit together.

    character   a prior from the transition type, so a hard cut stays an event
                and a stem blend stays a blend. The measurements adjust the
                type's natural length; they do not overrule what it is.

    Returns the best length in bars.
    """
    prior = auto_bars(a, b, inten_a, inten_b, name, floor, ceiling)
    curve_a = np.asarray(a.get("energy_curve") or [], dtype=float)
    curve_b = np.asarray(b.get("energy_curve") or [], dtype=float)
    ca, cb = a.get("cues") or {}, b.get("cues") or {}
    n_a = int(a.get("n_bars") or 0)

    best, best_cost = prior, None
    for L in lengths:
        if not floor <= L <= ceiling:
            continue
        from . import transitions            # local: transitions imports us
        lead = transitions.entry_lead(name, L)
        enter_b = max(0, int(cb.get("first_full_bar", 0)) - lead)
        exit_a = min(int(ca.get("outro_start_bar", n_a)),
                     int(ca.get("last_full_bar", n_a)) + 1) - L
        exit_a = max(0, exit_a)

        cost = 0.0
        # Structure: both ends of the region should land on section lines.
        cost += 0.55 * (_edge_miss(a, exit_a) + _edge_miss(b, enter_b)) / L ** 0.5
        # Character: stay near the type's natural length, in octaves of bars.
        cost += 1.1 * abs(np.log2(L / max(1, prior)))

        if act_a is not None and act_b is not None and len(act_a) and len(act_b):
            from .analysis import instruments
            cost += 2.4 * instruments.collision(act_a, exit_a, act_b, enter_b, L)

        if curve_a.size and curve_b.size:
            ea = float(curve_a[max(0, min(curve_a.size - 1, exit_a)):
                               min(curve_a.size, exit_a + L)].mean() or 0.0)
            eb = float(curve_b[max(0, min(curve_b.size - 1, enter_b)):
                               min(curve_b.size, enter_b + L)].mean() or 0.0)
            # Longer regions pay more for an energy mismatch, not a flat rate.
            cost += 1.6 * abs(ea - eb) * (L / 8.0) ** 0.5

        if best_cost is None or cost < best_cost:
            best, best_cost = L, cost
    return int(best)


def plan_bars(order, metas, inten, names, mode="bars", value=16, bpm=None,
              activity=None):
    """The per-join region lengths for a whole set, in bars.

    `mode` mirrors the three ways a user thinks about transition length:
    "bars" (the native unit), "seconds" (converted, then snapped to a phrase),
    and "auto" -- which here means fitted to the two tracks rather than looked
    up from their genre.
    """
    n = max(0, len(order) - 1)
    if mode == "auto":
        act = activity or {}
        return [fit_bars(metas[order[p]], metas[order[p + 1]],
                         inten[order[p]], inten[order[p + 1]],
                         names[p] if names else None,
                         act.get(order[p]), act.get(order[p + 1]))
                for p in range(n)]
    if mode == "seconds":
        bpm = bpm or float(np.median([metas[i]["bpm"] for i in order]))
        return [bars_for_seconds(value, bpm)] * n
    return [snap_phrase(int(value))] * n


def describe(order, metas, inten, target, style="blend"):
    """Human-readable plan, with rule violations called out."""
    n = len(order)
    lines = [f"{'#':>2} {'bpm':>7} {'dBPM':>6} {'cam':>4} {'int':>6} {'arc':>6}  "
             f"{'transition':<13} track"]
    bad_total = 0
    for pos, ti in enumerate(order):
        m = metas[ti]
        nxt, dbpm, flag = "", 0.0, ""
        if pos < n - 1:
            j = order[pos + 1]
            nxt = choose_transition(m, metas[j], inten[ti], inten[j], style)
            dbpm = metas[j]["bpm"] - m["bpm"]
            v = violations(m, metas[j], pos, n)
            if style == "cut":
                # Nothing overlaps, so a key clash cannot be heard.
                v = [x for x in v if not x.startswith("key")]
            bad_total += len(v)
            flag = "  <-- " + "; ".join(v) if v else ""
        lines.append(f"{pos + 1:2d} {m['bpm']:7.2f} {dbpm:+6.2f} "
                     f"{m['camelot']:>4} {inten[ti]:6.2f} {target[pos]:6.2f}  "
                     f"{nxt:<13} {m['artist'][:16]} - {m['title'][:26]}{flag}")
    lines.append(f"\nrule violations: {bad_total}")
    return "\n".join(lines)
