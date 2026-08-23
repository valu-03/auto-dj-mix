"""Export the mix as a structured plan a human (or another tool) can act on."""

import json

import numpy as np

from . import planner
from .analysis import key as key_mod
from .analysis import track as track_mod

# Our internal transition names -> the vocabulary a DJ would use.
STYLE = {
    "bass_swap": "EQ Swap",
    "eq_blend": "Fade",
    "filter_sweep": "Filter Sweep",
    "echo_out": "Cut",
    "stem_blend": "Stem Separation Blend",
}


def mmss(seconds):
    seconds = max(0.0, float(seconds))
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _cross_bar(curve, bars, falling=True, level=0.5):
    """Which bar a gain curve crosses `level`."""
    c = np.asarray(curve)
    idx = np.nonzero(c < level)[0] if falling else np.nonzero(c > level)[0]
    if len(idx) == 0:
        return None
    return round(float(idx[0]) / len(c) * bars, 2)


def eq_instructions(tr, bars, a_time, beat_s):
    """Describe the automation in words, derived from the actual curves."""
    swap = _cross_bar(tr.out_low, bars, falling=True)
    in_bass = _cross_bar(tr.in_low, bars, falling=False)
    mid_out = _cross_bar(tr.out_mid, bars, falling=True)
    parts = []
    if swap is not None:
        t = a_time + swap * 4 * beat_s
        parts.append(f"Kill Low EQ on A at {mmss(t)} (bar {swap:g} of {bars})")
    if in_bass is not None:
        t = a_time + in_bass * 4 * beat_s
        parts.append(f"bring B's Low in at {mmss(t)} (bar {in_bass:g}) "
                     f"- never both basslines")
    if mid_out is not None:
        parts.append(f"A's Mid crosses under at bar {mid_out:g}, "
                     f"B's Mid up over {bars} bars")
    if tr.out_sweep:
        parts.append(f"sweep A under a {tr.out_sweep[0]}-pass rising "
                     f"{tr.out_sweep[1][0]:.0f}->{tr.out_sweep[1][-1]:.0f} Hz")
    if tr.echo:
        parts.append(f"cut A into a {tr.echo['delay_s']*1000:.0f} ms echo "
                     f"(feedback {tr.echo['feedback']:.2f}) and let it ring out")
    return "; ".join(parts)


def mix_plan(metas, order, segs, joins, rep, bars, sample_rate, arc_shape):
    """Full structured plan: one entry per transition."""
    inten = planner.intensity(metas)
    n = len(order)
    spb = sample_rate * (4 * 60.0 / rep["mix_bpm"])
    beat_s = 60.0 / rep["mix_bpm"]

    tracks = []
    for pos, ti in enumerate(order):
        m = metas[ti]
        t = segs[pos]["mix_start"] * spb / sample_rate
        tracks.append({
            "id": pos + 1,
            "file": m["file"],
            "artist": m["artist"],
            "title": m["title"],
            "bpm": round(m["bpm"], 2),
            "camelot": m["camelot"],
            "key": m["key"],
            "key_confidence": m["key_confidence"],
            "energy_score": round(float(inten[ti]) * 2 + 5, 2),   # 0..10 scale
            "loudness_lufs": m["loudness"],
            "set_position_time": mmss(t),
            "intro_ends_bar": m["cues"]["intro_bars"],
            "outro_starts_bar": m["cues"]["outro_start_bar"],
        })

    joins_out = []
    for pos in range(n - 1):
        ai, bi = order[pos], order[pos + 1]
        a, b = metas[ai], metas[bi]
        tr = joins[pos]

        a_start = track_mod.bar_time(a, segs[pos]["exit"])
        a_end = track_mod.bar_time(a, segs[pos]["exit"] + bars)
        b_start = track_mod.bar_time(b, segs[pos + 1]["enter"])
        mix_t = segs[pos + 1]["mix_start"] * spb / sample_rate

        delta = float(inten[bi] - inten[ai])
        joins_out.append({
            "track_a_id": pos + 1,
            "track_b_id": pos + 2,
            "transition_style": STYLE.get(tr.name, tr.name),
            "transition_bars": bars,
            "mix_start_time_a": mmss(a_start),
            "mix_end_time_a": mmss(a_end),
            "mix_start_time_b": mmss(b_start),
            "mix_timeline_start": mmss(mix_t),
            "mix_timeline_end": mmss(mix_t + bars * 4 * beat_s),
            "eq_instructions": eq_instructions(tr, bars, a_start, beat_s),
            "energy_delta": int(np.clip(round(delta * 2.5), -5, 5)),
            "bpm_delta": round(b["bpm"] - a["bpm"], 2),
            "key_move": f"{a['camelot']} -> {b['camelot']}",
            "key_distance": key_mod.camelot_distance(a["camelot"], b["camelot"]),
            "rule_violations": planner.violations(a, b, pos, n),
        })

    return {
        "set": {
            "arc": arc_shape,
            "mix_bpm": rep["mix_bpm"],
            "duration": mmss(rep["duration_s"]),
            "tracks": n,
            "transition_bars": bars,
            "max_stretch_pct": rep["max_stretch_pct"],
            "lufs": rep["lufs_out"],
            "peak": rep["peak_out"],
            "total_violations": sum(len(j["rule_violations"]) for j in joins_out),
        },
        "tracks": tracks,
        "transitions": joins_out,
    }


def write(path, plan_dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan_dict, fh, indent=2, ensure_ascii=False)


# --- DSP-engine schema -----------------------------------------------------

DSP_STYLE = {
    "bass_swap": "eq_swap",
    "eq_blend": "eq_swap",
    "filter_sweep": "filter_sweep",
    "echo_out": "cut_drop",
    "stem_blend": "stem_blend",
    "vocal_slam_drop": "vocal_slam_drop",
    "euro_rap_breakout": "euro_rap_breakout",
}


def _automation_steps(tr, bars, beat_s, collide):
    """Timestamped EQ actions, read off the actual automation curves."""
    bar_s = 4 * beat_s
    steps = [{"timestamp_offset_seconds": 0.0, "action": "kill_incoming_low_eq"}]

    mid_out = _cross_bar(tr.out_mid, bars, falling=True, level=0.85)
    if mid_out:
        steps.append({"timestamp_offset_seconds": round(mid_out * bar_s, 2),
                      "action": "gradual_fade_outgoing_mid"})
    swap = _cross_bar(tr.out_low, bars, falling=True)
    if swap is not None:
        steps.append({"timestamp_offset_seconds": round(swap * bar_s, 2),
                      "action": "instant_swap_low_eq"})
    if tr.out_sweep:
        steps.append({"timestamp_offset_seconds": 0.0,
                      "action": "engage_outgoing_highpass_sweep"})
    if tr.echo:
        steps.append({"timestamp_offset_seconds":
                      round(tr.echo["start"] * bars * bar_s, 2),
                      "action": "cut_outgoing_into_echo"})
    if tr.name == "vocal_slam_drop":
        slam = _cross_bar(tr.out_mid, bars, falling=True, level=0.5)
        steps.append({
            "timestamp_offset_seconds": round((slam or 0) * bar_s, 2),
            "action": "cut_outgoing_to_zero_on_beat_1",
            "reason": "incoming track opens on an isolated vocal",
        })
    if tr.name == "euro_rap_breakout":
        hold = tr.meta.get("hold_bars", 8)
        steps.append({"timestamp_offset_seconds": 0.0,
                      "action": "apply_400hz_hpf_to_incoming",
                      "hpf_hz": tr.meta.get("hpf_hz", 400.0),
                      "hold_bars": hold})
        steps.append({"timestamp_offset_seconds": round(hold * bar_s, 2),
                      "action": "release_incoming_bass_filter"})
    # Sonic collision prevention: both tracks carrying the same dominant
    # element need an explicit instruction, not a gradual blend.
    for tag in collide:
        steps.append({
            "timestamp_offset_seconds": round(bars * bar_s * 0.5, 2),
            "action": ("stem_cut_incoming_bass" if tag == "heavy_sub"
                       else "duck_outgoing_vocal_band"),
            "reason": f"both tracks are {tag}",
        })
    steps.sort(key=lambda s: s["timestamp_offset_seconds"])
    return steps


def dsp_plan(metas, order, segs, joins, rep, bars, sample_rate, arc_shape,
             target_lufs=None):
    """The deterministic DSP-engine schema: mastering profiles + timeline."""
    from . import profile as prof

    tags = prof.sonic_profile(metas)
    beat_s = 60.0 / rep["mix_bpm"]
    spb = sample_rate * 4 * beat_s
    target = target_lufs if target_lufs is not None else prof.TARGET_LUFS_CLUB

    profiles = {}
    for pos, ti in enumerate(order):
        m = metas[ti]
        tid = f"{pos + 1:02d}_{m['artist'][:14]}_{m['title'][:18]}".replace(" ", "_")
        p = prof.mastering_profile(m, tags[ti], target)
        p["sonic_profile"] = tags[ti]
        p["measured"] = {
            "lufs_average": m.get("loudness"),
            "peak_db": m.get("peak_db"),
            "harshness": m.get("harshness"),
            "air_ratio": m.get("air_ratio"),
            "year": m.get("year"),
        }
        profiles[tid] = p

    ids = list(profiles)
    timeline = []
    for pos in range(len(order) - 1):
        ai, bi = order[pos], order[pos + 1]
        a, b = metas[ai], metas[bi]
        tr = joins[pos]
        collide = prof.collisions(tags[ai], tags[bi])

        style = DSP_STYLE.get(tr.name, "eq_swap")
        if abs(b["bpm"] - a["bpm"]) > planner.MAX_BPM_JUMP:
            style = "cut_drop"           # spec: flag oversized tempo gaps
        if collide and style == "eq_swap":
            style = "stem_blend"

        a_start = track_mod.bar_time(a, segs[pos]["exit"])
        a_end = track_mod.bar_time(a, segs[pos]["exit"] + bars)
        b_start = track_mod.bar_time(b, segs[pos + 1]["enter"])
        timeline.append({
            "sequence_index": pos,
            "track_outgoing_id": ids[pos],
            "track_incoming_id": ids[pos + 1],
            "transition_style": style,
            "trigger_timestamps": {
                "outgoing_mix_start": mmss(a_start),
                "outgoing_mix_end": mmss(a_end),
                "incoming_mix_start": mmss(b_start),
                "incoming_mix_end": mmss(b_start + bars * 4 * beat_s),
            },
            "phrase_bars": bars,
            "bpm_delta": round(b["bpm"] - a["bpm"], 2),
            "key_move": f"{a['camelot']} -> {b['camelot']}",
            "sonic_collisions": collide,
            "rule_violations": planner.violations(a, b, pos, len(order)),
            "eq_automation_steps": _automation_steps(tr, bars, beat_s, collide),
        })

    return {
        "mix_metadata": {
            "total_tracks": len(order),
            "calculated_energy_arc": arc_shape,
            "mix_bpm": rep["mix_bpm"],
            "duration": mmss(rep["duration_s"]),
            "target_lufs": target,
            "achieved_lufs": rep["lufs_out"],
            "total_violations": sum(len(t["rule_violations"]) for t in timeline),
        },
        "mastering_profiles": profiles,
        "transition_timeline": timeline,
    }


# --- node-graph automation schema ------------------------------------------

def _nodes_from_curve(curve, bars, node, kind="eq", tol=0.06):
    """Turn a per-sample gain curve into sparse automation nodes.

    Emits a node only where the value actually moves, so a curve that holds at
    unity for 8 bars produces one node rather than 8. `kind="eq"` maps linear
    gain to the percentage form the schema uses: 1.0 -> 0.0 (unity),
    0.0 -> -100.0 (killed).
    """
    c = np.asarray(curve, dtype=float)
    if len(c) == 0:
        return []
    n = max(8, int(bars) * 4)
    idx = np.linspace(0, len(c) - 1, n).astype(int)
    vals = c[idx]
    offs = np.linspace(0.0, float(bars), n)

    def conv(v):
        return round(float(v), 4) if kind == "gain" else round((float(v) - 1.0) * 100.0, 2)

    out = [{"bar_offset": 0.0, "node": node, "target_value": conv(vals[0])}]
    last = vals[0]
    for o, v in zip(offs[1:], vals[1:]):
        if abs(v - last) >= tol:
            out.append({"bar_offset": round(float(o), 2), "node": node,
                        "target_value": conv(v)})
            last = v
    if abs(vals[-1] - last) > 1e-6:
        out.append({"bar_offset": round(float(bars), 2), "node": node,
                    "target_value": conv(vals[-1])})
    return out


def playbook_plan(metas, order, segs, joins, rep, bars, sample_rate, strategy,
                  reason=""):
    """The genre-playbook schema: strategy + a node-graph automation timeline."""
    from . import playbook as pb

    ids = [f"{p + 1:02d}_{metas[t]['artist'][:14]}_{metas[t]['title'][:18]}"
           .replace(" ", "_") for p, t in enumerate(order)]

    timeline = []
    for pos in range(len(order) - 1):
        tr = joins[pos]
        steps = []
        for curve, node, kind in (
            (tr.out_low, "eq_low_a", "eq"), (tr.out_mid, "eq_mid_a", "eq"),
            (tr.out_high, "eq_high_a", "eq"), (tr.in_low, "eq_low_b", "eq"),
            (tr.in_mid, "eq_mid_b", "eq"), (tr.in_high, "eq_high_b", "eq"),
        ):
            steps += _nodes_from_curve(curve, bars, node, kind)
        # A slam is a gain event, not an EQ event -- record it as one.
        if tr.name == "vocal_slam_drop":
            slam = _cross_bar(tr.out_mid, bars, falling=True, level=0.5) or 0.0
            steps += [{"bar_offset": 0.0, "node": "gain_a", "target_value": 1.0},
                      {"bar_offset": round(slam, 2), "node": "gain_a",
                       "target_value": 0.0}]
        if tr.out_sweep:
            steps.append({"bar_offset": 0.0, "node": "hpf_a_hz",
                          "target_value": float(tr.out_sweep[1][0])})
            steps.append({"bar_offset": float(bars), "node": "hpf_a_hz",
                          "target_value": float(tr.out_sweep[1][-1])})
        if tr.in_sweep:
            hold = tr.meta.get("hold_bars", bars / 2)
            steps.append({"bar_offset": 0.0, "node": "hpf_b_hz",
                          "target_value": float(tr.in_sweep[1][0])})
            steps.append({"bar_offset": float(hold), "node": "hpf_b_hz",
                          "target_value": float(tr.in_sweep[1][-1])})
        steps.sort(key=lambda s: (s["bar_offset"], s["node"]))

        timeline.append({
            "track_outgoing_id": ids[pos],
            "track_incoming_id": ids[pos + 1],
            "transition_style": tr.name,
            "steps": steps,
        })

    return {
        "selected_playbook_strategy": strategy,
        "strategy_reason": reason,
        "strategy_note": pb.STRATEGIES[strategy]["note"],
        "mix_alignment": {
            "target_mix_bpm": rep["mix_bpm"],
            "required_phrase_bars": bars,
        },
        "dsp_automation_timeline": timeline,
    }
