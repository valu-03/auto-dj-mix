"""Building a set out of SECTIONS rather than whole tracks.

Playing each track once, start to finish, is a playlist with crossfades. A DJ
does something else: takes the sixteen bars that matter, drops them where they
belong in the arc, and may come back to the same record twice. A megamix is
that taken to its limit -- the hook, the drop, the breakdown, in an order the
original tracks never had.

This module turns each analysed track into a set of candidate *blocks*, scores
them, and picks a sequence. The renderer then treats those blocks exactly as it
treated whole tracks: the block list has the same shape as a segment plan, so
transitions, sync, ducking and mastering all work unchanged.

Two rules keep it musical rather than merely rearranged:

phrase   every block starts and ends on a phrase boundary, in the track's own
         bars. A block that starts mid-phrase is wrong no matter how good the
         eight bars inside it are.

variety  the same track never plays twice in a row, and a block is never
         reused. A megamix revisits a record; it does not stutter on it.
"""

import numpy as np

from . import planner
from .analysis import key as key_mod

BLOCK_BARS = 32          # ~57 s at 133 BPM; a full verse-and-chorus
HOP_BARS = 8             # candidates start every 8 bars
MIN_BLOCK = 16

# What makes a block worth playing. Weights are relative, not absolute.
W_ENERGY = 1.0           # how hard it hits
W_DROP = 0.9             # contains a labelled drop
W_CHORUS = 0.5           # contains a labelled chorus
W_EDGE = -1.4            # penalty for straying into intro or outro


def candidates(meta, index, block_bars=BLOCK_BARS, hop_bars=HOP_BARS):
    """Every phrase-aligned block of one track, with a score.

    Scored on what is *inside* the block, not on the track as a whole. A weak
    track with one great drop should contribute that drop; a strong track's
    filler should not get in on reputation.
    """
    cues = meta.get("cues") or {}
    curve = np.asarray(meta.get("energy_curve") or [], dtype=float)
    n_bars = int(meta.get("n_bars") or len(curve))
    if n_bars < block_bars or curve.size == 0:
        return []

    first = int(cues.get("first_full_bar", 0))
    outro = int(cues.get("outro_start_bar", n_bars))
    segs = cues.get("segments") or []

    out = []
    start = max(0, first)
    while start + block_bars <= min(n_bars, outro + block_bars // 2):
        end = start + block_bars
        lo, hi = min(start, curve.size), min(end, curve.size)
        if hi - lo < block_bars // 2:
            break
        energy = float(curve[lo:hi].mean())

        # Which labelled sections this block overlaps, and by how much.
        drop = chorus = 0.0
        for s in segs:
            a, b = int(s.get("start_bar", 0)), int(s.get("end_bar", 0))
            overlap = max(0, min(end, b) - max(start, a)) / block_bars
            name = (s.get("name") or "").lower()
            if name == "drop":
                drop = max(drop, overlap)
            elif name in ("chorus", "build"):
                chorus = max(chorus, overlap)

        # How much of the block sits outside the track's usable body.
        before = max(0, first - start) / block_bars
        after = max(0, end - outro) / block_bars
        edge = before + after

        score = (W_ENERGY * energy + W_DROP * drop + W_CHORUS * chorus
                 + W_EDGE * edge)
        out.append({"track": index, "start": start, "bars": block_bars,
                    "score": float(score), "energy": energy,
                    "drop": drop > 0.25, "edge": edge})
        start += hop_bars
    return out


W_REUSE = 3.2            # cost of playing a record again before others have


def _slot_cost(cand, prev, metas, target, key_weight=1.0, uses=None):
    """Cost of putting `cand` in a slot whose target intensity is `target`.

    `uses` is what stops the degenerate answer. Without it the greedy search
    finds the cheapest *pair* of harmonically adjacent records and alternates
    between them forever: on a four-track set it produced Run To Me / Be My
    Lover eight times and never played the other two at all. Every step was
    locally optimal and the result was musically absurd.

    Charging for each previous use makes a fresh record competitive with a
    harmonically perfect repeat, so the set covers the crate before it starts
    revisiting. It is the same fix the whole-track selector needed when it
    chose ten tracks in the same key.
    """
    # How far this block is from the intensity the arc wants here. The energy
    # curve is normalised within its own track, so it is comparable across
    # tracks only as "how hard, for this record" -- which is the right question
    # for a block, unlike for a whole track.
    cost = 2.0 * abs(cand["energy"] - target)
    cost -= 0.8 * cand["score"]
    if uses:
        cost += W_REUSE * uses.get(cand["track"], 0)
    if prev is None:
        return cost

    a, b = metas[prev["track"]], metas[cand["track"]]
    kd = key_mod.camelot_distance(a["camelot"], b["camelot"])
    cost += key_weight * kd
    if kd > planner.LEGAL_KEY_DISTANCE:
        cost += 6.0
    cost += 1.2 * abs(b["bpm"] - a["bpm"]) / max(1.0, a["bpm"]) / 0.06
    return cost


def build(metas, minutes=8.0, block_bars=BLOCK_BARS, shape="warmup_to_peak",
          bpm=None, key_weight=1.0, seed=0):
    """Choose a sequence of blocks that fills `minutes` along an energy arc.

    Greedy along the arc rather than a global search: the constraint that
    matters most (no repeated track back to back, no reused block) is local,
    and a slot's best choice barely depends on slots two steps away. 2-opt,
    which pays off for whole-track ordering, mostly reshuffles here because
    every block is individually replaceable.

    Returns a list of blocks, each {track, start, bars}.
    """
    bpm = bpm or float(np.median([m["bpm"] for m in metas]))
    bar_s = 4 * 60.0 / bpm
    slots = max(2, int(round(minutes * 60.0 / (block_bars * bar_s))))

    pool = []
    for i, m in enumerate(metas):
        pool.extend(candidates(m, i, block_bars))
    if not pool:
        return []

    # Normalise energy across the whole pool so the arc means something.
    e = np.array([c["energy"] for c in pool])
    lo, hi = e.min(), e.max()
    span = (hi - lo) or 1.0
    for c in pool:
        c["energy"] = (c["energy"] - lo) / span

    target = planner.arc(slots, shape)
    t_lo, t_hi = target.min(), target.max()
    target = (target - t_lo) / ((t_hi - t_lo) or 1.0)

    used, chosen, prev = set(), [], None
    uses = {}
    n_tracks = len(metas)
    for s in range(slots):
        # Coverage as a CONSTRAINT, not a preference. A cost term alone loses
        # to harmony: the two distant records here cost +6 every time they are
        # considered, so even a stiff reuse charge left one of them never
        # played at all. Until every record has been heard once, only unplayed
        # records are eligible -- after that, harmony decides again.
        unplayed = {i for i in range(n_tracks) if i not in uses}
        remaining = slots - s
        must_cover = unplayed and len(unplayed) >= remaining - len(unplayed)

        best, best_cost = None, None
        for c in pool:
            tag = (c["track"], c["start"])
            if tag in used:
                continue
            if prev is not None and c["track"] == prev["track"]:
                continue        # never the same record twice in a row
            if must_cover and unplayed and c["track"] not in unplayed:
                continue
            cost = _slot_cost(c, prev, metas, float(target[s]), key_weight,
                              uses)
            if best_cost is None or cost < best_cost:
                best, best_cost = c, cost

        if best is None and must_cover:
            # Coverage was impossible this slot (e.g. the only unplayed record
            # is the one just played). Fall back rather than emit nothing.
            for c in pool:
                tag = (c["track"], c["start"])
                if tag in used or (prev and c["track"] == prev["track"]):
                    continue
                cost = _slot_cost(c, prev, metas, float(target[s]), key_weight,
                                  uses)
                if best_cost is None or cost < best_cost:
                    best, best_cost = c, cost
        if best is None:
            # Pool exhausted under the constraints: allow blocks to be reused
            # before allowing the same track twice running.
            used.clear()
            continue
        used.add((best["track"], best["start"]))
        uses[best["track"]] = uses.get(best["track"], 0) + 1
        chosen.append(best)
        prev = best
    return chosen


def to_segments(blocks, bars, sample_rate=None, tail_bars=None):
    """Turn chosen blocks into the segment dicts the renderer expects.

    Same keys as `render.segment_plan` produces, so everything downstream --
    transitions, sync, vocal ducking, the timeline widget -- works without
    knowing whether it is looking at whole tracks or blocks.
    """
    tail = bars if tail_bars is None else tail_bars
    segs, cursor = [], 0.0
    last = len(blocks) - 1
    for pos, b in enumerate(blocks):
        play = int(b["bars"])
        segs.append({
            "track": int(b["track"]), "pos": pos,
            "enter": int(b["start"]), "exit": int(b["start"]) + play,
            "mix_start": cursor,
            "bars": play + (0 if pos == last else tail),
            "head": 0 if pos == 0 else tail,
            "tail": 0 if pos == last else tail,
            "bpm": None,
        })
        cursor += play
    return segs


def describe(blocks, metas, bpm):
    """Readable plan: what plays, from where, for how long."""
    bar_s = 4 * 60.0 / bpm
    lines = [f"{'#':>2} {'at':>6} {'bars':>10} {'cam':>4} {'e':>5}  track"]
    t = 0.0
    for i, b in enumerate(blocks, 1):
        m = metas[b["track"]]
        lines.append(
            f"{i:2d} {int(t)//60:3d}:{int(t)%60:02d} "
            f"{b['start']:4d}-{b['start']+b['bars']:<5d} "
            f"{m['camelot']:>4} {b['energy']:5.2f}  "
            f"{m['artist'][:16]} - {m['title'][:26]}"
            f"{'  [drop]' if b.get('drop') else ''}")
        t += b["bars"] * bar_s
    lines.append(f"\n{len(blocks)} blocks, {t/60:.1f} min")
    return "\n".join(lines)
