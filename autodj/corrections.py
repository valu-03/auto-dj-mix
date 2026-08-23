"""Manual overrides for anything the analysis got wrong.

Beat detection is the one failure that ruins a whole mix. Every downstream
decision -- where a phrase starts, where a cut fires, how a segment is cut to an
exact number of bars -- is measured in bars off the fitted grid, so a track
whose downbeat is one beat out is not slightly wrong, it is wrong everywhere. On
electronic music the fit is reliable; on a live intro, a track with a bar of
3/4, or one that fades in under a held chord, it is not.

Until now there was no recourse. `--force` re-ran the same algorithm and got the
same answer. This is the recourse: a small file of per-track overrides that the
analyser applies as *inputs*, not as patches afterwards.

That distinction is the whole design. Patching a corrected BPM onto a finished
analysis leaves every per-bar array -- the energy curve, the segments, the cue
points -- still indexed against the grid that was wrong, so the numbers agree
with each other and disagree with the music. Feeding the correction in before
the per-bar stage means everything downstream is recomputed against the grid
you actually chose.

The cache follows automatically: the fingerprint includes a digest of a file's
corrections, so saving one invalidates exactly that track's cached analysis and
reverting one restores it.
"""

import hashlib
import json
from pathlib import Path

STORE = Path("cache/corrections.json")

# What may be overridden. Split by stage, because they enter the analysis at
# different points: the grid fields replace the beat-tracking result before any
# per-bar work happens, the cue fields replace the structural result after it.
GRID_FIELDS = ("bpm", "beat_offset", "downbeat_phase")
CUE_FIELDS = ("first_full_bar", "outro_start_bar", "drop_bars")
FIELDS = GRID_FIELDS + CUE_FIELDS


def _key(path):
    """Absolute path as the identity. Two folders may hold the same filename."""
    return str(Path(path).resolve())


def load(store=STORE):
    try:
        return json.loads(Path(store).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write(data, store=STORE):
    p = Path(store)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def get(path, store=STORE):
    """Every override held for one file. Empty dict when there are none."""
    return load(store).get(_key(path), {})


def set_fields(path, store=STORE, **values):
    """Record overrides. `None` removes a field rather than storing a null."""
    data = load(store)
    entry = dict(data.get(_key(path), {}))
    for k, v in values.items():
        if k not in FIELDS:
            raise KeyError(f"not a correctable field: {k}")
        if v is None:
            entry.pop(k, None)
        else:
            entry[k] = v
    if entry:
        data[_key(path)] = entry
    else:
        data.pop(_key(path), None)
    _write(data, store)
    return entry


def clear(path, store=STORE):
    """Drop every override for one file, restoring the analyser's own answer."""
    data = load(store)
    if data.pop(_key(path), None) is not None:
        _write(data, store)


def digest(path, store=STORE):
    """Short hash of one file's overrides, for the analysis cache key.

    Empty when there are none, so an uncorrected library keeps the fingerprints
    it already had and nothing is needlessly re-analysed.
    """
    entry = get(path, store)
    if not entry:
        return ""
    raw = json.dumps(entry, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ apply ---

def apply_grid(beat, duration, over):
    """Replace the fitted beatgrid with the corrected one, and re-derive.

    `first_downbeat`, `n_bars` and the beat period are all *consequences* of
    (bpm, offset, phase). Overriding one of the three and leaving the
    consequences alone is how a correction turns into a subtler bug than the
    thing it fixed, so they are recomputed here rather than trusted.
    """
    if not any(k in over for k in GRID_FIELDS):
        return beat
    out = dict(beat)
    for k in GRID_FIELDS:
        if k in over:
            out[k] = float(over[k]) if k != "downbeat_phase" else int(over[k])

    bpm = float(out["bpm"])
    beat_s = 60.0 / bpm
    phase = int(out["downbeat_phase"]) % 4
    offset = float(out["beat_offset"])

    first = offset + phase * beat_s
    # Keep the first downbeat inside the track: a negative offset from a
    # hand-placed marker near the start would otherwise put bar 0 before the
    # file begins, and every bar index after it would be shifted.
    bar_s = 4 * beat_s
    while first < 0:
        first += bar_s

    out["beat_period"] = round(beat_s, 6)
    out["first_downbeat"] = round(first, 6)
    out["n_beats"] = int(max(0, (duration - offset) / beat_s)) + 1
    out["n_bars"] = int(max(0, (duration - first) / bar_s)) + 1
    out["corrected"] = sorted(k for k in GRID_FIELDS if k in over)
    return out


def apply_cues(cues, over, n_bars):
    """Replace structural cue points, clamped to the track's real length."""
    if not any(k in over for k in CUE_FIELDS):
        return cues
    out = dict(cues)
    limit = max(0, int(n_bars) - 1)
    if "first_full_bar" in over:
        out["first_full_bar"] = int(min(max(0, int(over["first_full_bar"])),
                                        limit))
        out["intro_bars"] = out["first_full_bar"]
    if "outro_start_bar" in over:
        out["outro_start_bar"] = int(min(max(0, int(over["outro_start_bar"])),
                                         limit))
    if "drop_bars" in over:
        out["drop_bars"] = sorted({int(min(max(0, int(b)), limit))
                                   for b in over["drop_bars"]})
    # first_full must precede outro_start or the segment planner has a track
    # with a negative playable body and silently falls back to a minimum.
    if out.get("first_full_bar", 0) >= out.get("outro_start_bar", limit):
        out["outro_start_bar"] = min(limit, out.get("first_full_bar", 0) + 8)
    out["corrected"] = sorted(k for k in CUE_FIELDS if k in over)
    return out


def summary(path, store=STORE):
    """One line describing what has been overridden, for the UI."""
    entry = get(path, store)
    if not entry:
        return ""
    bits = []
    for k in FIELDS:
        if k not in entry:
            continue
        v = entry[k]
        if k == "drop_bars":
            bits.append(f"{len(v)} drops")
        elif isinstance(v, float):
            bits.append(f"{k} {v:.3f}")
        else:
            bits.append(f"{k} {v}")
    return "manual: " + ", ".join(bits)
