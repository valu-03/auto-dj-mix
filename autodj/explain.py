"""Saying why the mix is the way it is.

Every choice this program makes is already a consequence of numbers it
computed: the order comes from an energy arc and a cost matrix, the transition
from a short chain of rules about key and intensity, the length from a fit
against both tracks' structure, and increasingly the final say comes from
measurements taken on the rendered audio.

None of that was visible. The application produced a mix and the reasoning
stayed in the source code, which makes the output impossible to argue with --
you either like it or you do not, and if you do not, there is nothing to
adjust because there is nothing to disagree with. Showing the reason turns a
verdict into a proposal.

It also keeps the program honest. A reason that reads badly usually is bad:
writing these out is what exposed that the "cut" style was dropping key
violations silently rather than deciding they did not apply.
"""

from .analysis import key as key_mod
from . import planner, transitions

# Which rule in `planner.choose_transition` fired, in the order it is tested.
# Kept as text here rather than returned from the planner so the decision path
# stays a single readable function there.
TRANSITION_WHY = {
    "double_drop": "both tracks have a usable drop, they are close in key and "
                   "energy, so the two drops are aligned to the same bar",
    "filter_sweep": "the keys are too far apart to overlap, so a highpass "
                    "strips the outgoing harmonics rather than letting them "
                    "argue",
    "smooth_swap": "a noticeable energy step, so the incoming highs lead in "
                   "before its body arrives",
    "dissolve": "compatible keys at a similar energy: nothing needs hiding",
    "fade": "little low end in play, so a plain equal-power crossfade is the "
            "least intrusive answer",
    "stem_blend": "stems are cached for both tracks, so the two can be mixed "
                  "instrument by instrument instead of band by band",
    "vocal_slam_drop": "the incoming track opens on a bare vocal, so the "
                       "outgoing one can run full and cut dead",
    "euro_rap_breakout": "the incoming intro is vocal-forward, so it rides "
                         "highpassed over the outgoing groove before the bass "
                         "lands",
    "loop_roll": "stepping up in energy, so the outgoing track rolls into the "
                 "cut and the change is earned",
    "riser_cut": "level energy either side, so a riser announces a change the "
                 "tracks are not signalling themselves",
    "cut_with_echo": "a big drop in energy: the echo tail fills the space the "
                     "cut leaves",
    "hard_cut": "nothing overlaps, so the join is the effect",
    "tremolo": "the outgoing track is chopped at an accelerating rate to build "
               "tension into the swap",
    "echo_out": "a large energy drop, punctuated rather than slid down",
    "bass_swap": "the workhorse: bass hands over on the downbeat while mids "
                 "and highs cross around it",
    "eq_blend": "same key and matched energy, so a gentle three-band cross is "
                "enough",
}


def key_note(a, b):
    """One line on the harmonic relationship between two tracks."""
    d = key_mod.camelot_distance(a["camelot"], b["camelot"])
    pair = f"{a['camelot']} → {b['camelot']}"
    if d == 0.0:
        return f"{pair}: same key"
    if d <= 1.0:
        return f"{pair}: adjacent on the wheel (distance {d:.1f})"
    if abs(d - planner.FLASH_KEY_DISTANCE) < 1e-9:
        return f"{pair}: a +2 energy flash (distance {d:.1f})"
    return f"{pair}: clash, distance {d:.1f}"


def tempo_note(a, b, from_bpm=None, to_bpm=None):
    jump = b["bpm"] - a["bpm"]
    line = f"{a['bpm']:.2f} → {b['bpm']:.2f} BPM ({jump:+.2f})"
    if from_bpm and to_bpm and abs(to_bpm - from_bpm) > 1e-6:
        from .render import glide_cents
        line += (f", played {from_bpm:.2f} → {to_bpm:.2f} "
                 f"({glide_cents(from_bpm, to_bpm):.0f} cents of glide)")
    elif to_bpm:
        stretch_pct = 100.0 * (to_bpm / b["bpm"] - 1.0)
        line += f", incoming stretched {stretch_pct:+.2f}%"
    return line


def join(pos, metas, order, segs, joins, rep=None, bars=None):
    """Every reason behind one join, as a list of (label, text) pairs."""
    a, b = metas[order[pos]], metas[order[pos + 1]]
    tr = joins[pos]
    out = [("Transition",
            f"{tr.name} — {TRANSITION_WHY.get(tr.name, 'chosen by the planner')}")]

    length = bars[pos] if bars else tr.bars
    beats = length * 4
    bpm = (segs[pos].get("bpm") or a["bpm"])
    out.append(("Length",
                f"{length} bars ({beats} beats, {length * 4 * 60.0 / bpm:.1f} s "
                f"at {bpm:.1f} BPM), starting and ending on a phrase line"))
    out.append(("Key", key_note(a, b)))
    out.append(("Tempo", tempo_note(a, b, segs[pos].get("bpm"),
                                    segs[pos + 1].get("bpm"))))
    out.append(("Entry",
                f"incoming enters at its bar {segs[pos + 1]['enter']}; "
                f"outgoing leaves at its bar {segs[pos]['exit']}"))

    need = transitions.REQUIRES.get(tr.name)
    if need:
        out.append(("Needs", need))

    if rep:
        for row in rep.get("auditioned") or []:
            if row.get("join") != pos:
                continue
            best = row["candidates"][0]["name"] if row["candidates"] else ""
            ranked = sorted(row["candidates"], key=lambda c: c["score"])
            listing = ", ".join(f"{c['name']} {c['score']:.2f}"
                                for c in ranked[:4])
            out.append(("Measured",
                        f"chose {row['chose']} over {row['seed']} — "
                        f"scores (lower is better): {listing}"))
            m = ranked[0]["metrics"] if ranked else {}
            if m:
                out.append(("Faults",
                            "  ".join(f"{k} {v:.2f}" for k, v in m.items())))
        for d in rep.get("vocal_ducks") or []:
            if d.get("join") == pos:
                out.append(("Vocals",
                            f"outgoing vocal ducked up to "
                            f"{d['max_duck_db']:.1f} dB by stem subtraction"))
    return out


def track_note(pos, metas, order, inten, target):
    """Why this track is in this slot of the set."""
    ti = order[pos]
    m = metas[ti]
    want, got = float(target[pos]), float(inten[ti])
    fit = "matches" if abs(want - got) < 0.35 else (
        "above" if got > want else "below")
    return (f"slot {pos + 1}: arc wants {want:+.2f}, this track measures "
            f"{got:+.2f} ({fit}) — {m['bpm']:.2f} BPM, {m['camelot']}")


def summary(rep):
    """A few lines describing the render as a whole."""
    lines = [
        f"master deck {rep.get('master_deck', '?')} at "
        f"{rep.get('mix_bpm', 0):.2f} BPM",
        f"tempo mode '{rep.get('tempo_mode', 'sync')}'"
        + (f", glide up to {rep.get('max_glide_cents', 0):.0f} cents"
           if rep.get("tempo_glide") else ", flat across the set"),
        f"loudness {rep.get('lufs_out', '?')} LUFS, peak "
        f"{rep.get('peak_out', '?')}",
    ]
    syncs = rep.get("sync_corrections") or []
    if syncs:
        worst = max(abs(s["shift_ms"]) for s in syncs)
        lines.append(f"sync corrected {len(syncs)} deck(s), worst "
                     f"{worst:.1f} ms")
    if rep.get("stem_transitions"):
        lines.append("stem-mixed: " + ", ".join(rep["stem_transitions"]))
    tried = rep.get("auditioned") or []
    if tried:
        changed = sum(1 for r in tried if r["chose"] != r["seed"])
        lines.append(f"auditioned {len(tried)} join(s) on the rendered audio; "
                     f"{changed} changed from the planner's first choice")
    return lines
