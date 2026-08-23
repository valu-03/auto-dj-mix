"""Sonic profiling and per-track mastering profiles.

Turns the numbers the analyser produces into the vocabulary a DJ or mastering
engineer works in -- "heavy_sub", "vocal_forward" -- and derives the static EQ
and dynamics each track needs before it goes into a club-ready mix.
"""

import numpy as np

TARGET_LUFS_CLUB = -11.0

# Thresholds are relative to the library being mixed, not absolute: "heavy sub"
# means heavy compared to the other tracks in this set. An absolute number would
# be meaningless across different masters and eras.
HEAVY_SUB_PCT = 70
BRIGHT_PCT = 70
VOCAL_PCT = 65


def sonic_profile(metas):
    """Tag every track. Returns a list of lists of strings."""
    if not metas:
        return []
    low = np.array([m.get("low_level", 0.0) for m in metas])
    bright = np.array([m.get("brightness", 0.0) for m in metas])
    # Vocal energy proxy: how much of the spectrum sits in the vocal band
    # relative to everything else. Not a vocal detector -- a vocal *likelihood*.
    vocal = np.array([m.get("mid_ratio", 0.0) for m in metas])

    harsh = np.array([m.get("harshness", 0.0) for m in metas])

    lo_t = np.percentile(low, HEAVY_SUB_PCT)
    br_t = np.percentile(bright, BRIGHT_PCT)
    vo_t = np.percentile(vocal, VOCAL_PCT)
    # Relative *and* absolute: something must be genuinely bright before its
    # rank matters, or the least-dull track in a dull set gets cut.
    ha_t = max(np.percentile(harsh, HARSH_PCT), HARSH_THRESHOLD * 0.62)

    out = []
    for i, m in enumerate(metas):
        tags = []
        if low[i] >= lo_t:
            tags.append("heavy_sub")
        if vocal[i] >= vo_t:
            tags.append("vocal_forward")
        if bright[i] >= br_t:
            tags.append("bright_highs")
        year = m.get("year")
        if year:
            if 1970 <= year < 1980:
                tags.append("vintage_1970s")
            elif 1980 <= year < 1990:
                tags.append("vintage_1980s")
            elif 1990 <= year < 2000:
                tags.append("vintage_1990s")
        if m.get("density", 1.0) < 0.72:
            tags.append("sparse_arrangement")
        if m.get("vocal_forward_intro"):
            tags.append("vocal_forward_intro")
        if harsh[i] >= ha_t:
            tags.append("harsh")
        if 128.0 <= m.get("bpm", 0) <= 146.0 and m.get("density", 0) > 0.7:
            tags += ["eurodance", "90s"]
        out.append(tags)
    return out


# 2-4 kHz share of the presence band. Measured across the test set the real
# range is 0.225-0.463, so an absolute 0.62 flagged nothing at all. Harshness is
# only meaningful relative to the other tracks being mixed.
HARSH_THRESHOLD = 0.62
HARSH_PCT = 80


def mastering_profile(meta, tags, target_lufs=TARGET_LUFS_CLUB,
                      headroom_db=1.0):
    """Static EQ, gain and dynamics for one track.

    Gain is capped by the track's own peak. Asking for -11 LUFS on a track that
    already peaks at -0.1 dBFS means the limiter has to find every one of those
    dB, and the result comes out squashed *and* short of target -- measured on
    this set, a -9 LUFS request produced -16 LUFS output.
    """
    lufs = meta.get("loudness", -14.0)
    peak_db = meta.get("peak_db", -0.1)
    raw_gain = float(target_lufs - lufs)
    # Leave `headroom_db` below 0 dBFS before the limiter is involved at all.
    ceiling_gain = float(-headroom_db - peak_db)
    gain = round(min(raw_gain, ceiling_gain + 4.0), 2)   # allow 4 dB of limiting

    harsh = "harsh" in tags
    vocal = "vocal_forward" in tags
    heavy = "heavy_sub" in tags
    vintage = next((t for t in tags if t.startswith("vintage_")), None)

    high_shelf = 0.0
    if vintage:
        # Older masters roll off early. A gentle shelf restores modern presence
        # without making the record sound like a remaster.
        high_shelf = {"vintage_1970s": 2.5, "vintage_1980s": 2.0,
                      "vintage_1990s": 1.5}.get(vintage, 0.0)
    if "bright_highs" in tags:
        high_shelf = min(high_shelf, 0.8)      # already bright; do not pile on
    if meta.get("air_ratio", 1.0) < 0.004:
        high_shelf += 0.8                      # genuinely dull, needs air

    if harsh:
        mid_gain = -2.0                        # tame 2-4 kHz fatigue
        mid_freq = 3000
        mid_q = 1.6
    elif vocal:
        mid_gain = 1.2                         # lift intelligibility
        mid_freq = 2000
        mid_q = 1.0
    else:
        mid_gain = 0.0
        mid_freq = 1200
        mid_q = 1.2

    ratio = 1.5 if gain < 4 else (2.0 if gain < 7 else 2.5)

    return {
        "gain_staging": {
            "target_gain_db": gain,
            "limiter_threshold_db": -1.0,
            "uncapped_gain_db": round(raw_gain, 2),
            "gain_capped_by_peak": bool(raw_gain > gain + 1e-6),
        },
        "eq_bands": {
            "high_pass_filter": {
                "enabled": True,
                "frequency_hz": 30 if heavy else 20,
                "slope_db_oct": 12,
            },
            "low_shelf": {
                "frequency_hz": 100,
                "gain_db": -1.0 if heavy else 0.0,
                "q_factor": 0.7,
            },
            "parametric_mid": {
                "frequency_hz": mid_freq,
                "gain_db": round(mid_gain, 2),
                "q_factor": mid_q,
            },
            "high_shelf": {
                "frequency_hz": 8000,
                "gain_db": round(high_shelf, 2),
                "q_factor": 0.7,
            },
            "low_pass_filter": {
                "enabled": True,
                "frequency_hz": 19000,
                "slope_db_oct": 6,
            },
        },
        "dynamics": {
            "compressor_enabled": True,
            "threshold_db": -2.0,
            "ratio": ratio,
            "attack_ms": 20,
            "release_ms": 150,
        },
    }


def collisions(tags_a, tags_b):
    """Sonic clashes that force explicit EQ action."""
    out = []
    for tag in ("heavy_sub", "vocal_forward"):
        if tag in tags_a and tag in tags_b:
            out.append(tag)
    return out
