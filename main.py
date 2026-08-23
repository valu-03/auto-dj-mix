"""Auto DJ Mix - point it at a folder, get a mixed set.

    python main.py --input musica --output mix.wav
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from autodj import (arrange, audio, explain, export, library, planner, render)
from autodj.analysis import instruments, track


def build(folder, out_path, bars=4, arc="warmup_to_peak", limit=None,
          force=False, style="smooth", align=False, sections=False,
          minutes=8.0, block_bars=arrange.BLOCK_BARS, duration_mode="bars",
          duration_value=None, tempo_mode="sync", audition=False):
    files = library.scan(folder)
    if limit:
        files = files[:limit]
    if not files:
        print(f"no audio files in {folder}")
        return 1
    print(f"found {len(files)} files in {folder}")

    t0 = time.time()
    metas = []
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] analysing {f.stem[:52]}", flush=True)
        metas.append(track.analyse(f, force=force))
    print(f"analysis: {time.time() - t0:.1f}s\n")

    bars = planner.snap_phrase(bars)
    segs = None
    if sections:
        # Build the set out of the best blocks of each record rather than
        # playing anything start to finish. The block list has the same shape
        # as a segment plan, so the renderer needs no special case.
        bpm = float(np.median([m["bpm"] for m in metas]))
        blocks = arrange.build(metas, minutes=minutes, block_bars=block_bars,
                               shape=arc, bpm=bpm)
        if not blocks:
            print("no usable sections found (tracks shorter than a block?)")
            return 1
        order = [b["track"] for b in blocks]
        segs = arrange.to_segments(blocks, bars)
        inten = planner.intensity(metas)
        target = planner.arc(len(order), arc)
        cost = 0.0
        print(arrange.describe(blocks, metas, bpm))
        print(f"\narc '{arc}'   style '{style}'   transition {bars} bars   "
              f"blocks of {block_bars} bars\n")
    else:
        order, cost, inten, target = planner.plan(metas, shape=arc, style=style)
        print(planner.describe(order, metas, inten, target, style))
        print(f"plan cost {cost:.2f}   arc '{arc}'   style '{style}'\n")

    activity = None
    if align:
        activity, srcs = {}, []
        for i, f in enumerate(files):
            act, src = instruments.vocal_activity(f, metas[i])
            activity[i] = act
            srcs.append(src)
        n_stem = srcs.count("stems")
        print(f"lead-voice map: {n_stem}/{len(files)} from stems, "
              f"{len(files) - n_stem} from the spectral proxy\n")

    names = [planner.choose_transition(metas[order[p]], metas[order[p + 1]],
                                       inten[order[p]], inten[order[p + 1]],
                                       style)
             for p in range(len(order) - 1)]
    lengths = planner.plan_bars(order, metas, inten, names, duration_mode,
                                duration_value if duration_value is not None
                                else bars, activity=activity)
    bar_s = 4 * 60.0 / float(np.median([m["bpm"] for m in metas]))
    print(f"duration '{duration_mode}': "
          + ", ".join(f"{b} bars ({b * bar_s:.1f}s)" for b in lengths)
          + f"\ntempo '{tempo_mode}'\n")

    t0 = time.time()
    mixed, rep, segs, joins = render.render(
        files, metas, order, bars=lengths, style=style, activity=activity,
        segs=segs, tempo_mode=tempo_mode, audition=audition,
        progress=lambda i, n, t: print(f"  [{i + 1}/{n}] rendering {t[:52]}",
                                       flush=True))
    print(f"render: {time.time() - t0:.1f}s")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.save(out_path, mixed, audio.RENDER_RATE)
    spb = audio.RENDER_RATE * (4 * 60.0 / rep["mix_bpm"])
    out_path.with_suffix(".txt").write_text(
        render.tracklist(metas, order, segs, joins, spb, audio.RENDER_RATE),
        encoding="utf-8")
    out_path.with_suffix(".cue").write_text(
        render.cue_sheet(metas, order, segs, spb, audio.RENDER_RATE,
                         out_path.name), encoding="utf-8")
    plan_doc = export.mix_plan(metas, order, segs, joins, rep, bars,
                               audio.RENDER_RATE, arc)
    export.write(out_path.with_suffix(".json"), plan_doc)
    dsp_doc = export.dsp_plan(metas, order, segs, joins, rep, bars,
                              audio.RENDER_RATE, arc)
    export.write(out_path.with_name(out_path.stem + "_dsp.json"), dsp_doc)

    print(f"\n{out_path}  ({rep['duration_s'] / 60:.1f} min)")
    for k in ("mix_bpm", "max_stretch_pct", "lufs_out", "peak_out",
              "max_gain_reduction"):
        print(f"  {k:<20} {rep[k]}")
    print(f"  transitions          "
          + ", ".join(f"{n}/{b}" for n, b in zip(rep["transitions"],
                                                 rep["transition_bars"])))
    for line in explain.summary(rep):
        print(f"  {line}")
    return 0


def convert(src, dst=None, kbps=audio.MP3_KBPS):
    """Convert any rendered mix to MP3, without re-rendering it.

    Decoding and re-encoding is cheap next to a 70-second render, so keeping
    this separate means you can produce a car-sized file from a WAV you already
    have -- and keep the WAV as the master.
    """
    src = Path(src)
    if not src.exists():
        print(f"not found: {src}")
        return 1
    dst = Path(dst) if dst else src.with_suffix(".mp3")

    a, sr = audio.load(src)
    audio.save(dst, a, sr, kbps=kbps)
    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    print(f"{src.name} ({src_mb:.1f} MB) -> {dst.name} ({dst_mb:.1f} MB), "
          f"{src_mb / max(dst_mb, 1e-9):.1f}x smaller, "
          f"{audio.duration(a, sr) / 60:.1f} min")

    # Carry the tracklist and cue sheet across so the MP3 is self-contained.
    for suffix in (".txt", ".cue"):
        side = src.with_suffix(suffix)
        if side.exists() and side != dst.with_suffix(suffix):
            dst.with_suffix(suffix).write_text(
                side.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Automatic DJ mixing")
    p.add_argument("--gui", action="store_true",
                   help="open the desktop interface")
    p.add_argument("--input", default="musica", help="folder of audio files")
    p.add_argument("--output", default="output/mix.wav", help="output file")
    p.add_argument("--bars", type=int, default=4,
                   help="transition length in bars, snapped to 4/8/16/32. "
                        "At 133 BPM: 4 bars = 7.2 s, 16 bars = 28.8 s")
    p.add_argument("--arc", default="warmup_to_peak", choices=list(planner.ARCS),
                   help="energy arc for the set")
    p.add_argument("--style", default="smooth",
                   choices=("smooth", "cut", "blend"),
                   help="smooth (default): short equal-power cross-dissolve "
                        "with a bass handover -- overlapping and seamless, but "
                        "quick. cut: hard cuts on the 1, nothing overlaps. "
                        "blend: the long EQ-carved transition")
    p.add_argument("--align-instruments", dest="align", action="store_true",
                   help="nudge each entry to the phrase where the two tracks "
                        "are least likely to be singing at once (uses cached "
                        "stems when present, a spectral proxy otherwise)")
    p.add_argument("--sections", action="store_true",
                   help="build the set from SECTIONS instead of whole tracks: "
                        "picks the best blocks of each record, may revisit a "
                        "track, and never plays one start to finish")
    p.add_argument("--minutes", type=float, default=8.0,
                   help="target length when using --sections (default 8)")
    p.add_argument("--block-bars", dest="block_bars", type=int,
                   default=arrange.BLOCK_BARS,
                   help="bars per section block (default 32, ~57 s at 133 BPM)")
    p.add_argument("--duration", dest="duration_mode", default="bars",
                   choices=("auto", "bars", "seconds"),
                   help="how transition length is chosen. auto: fitted per "
                        "join to both tracks' structure, vocals and energy. "
                        "bars/seconds: a fixed value, snapped to a phrase")
    p.add_argument("--duration-value", dest="duration_value", type=float,
                   help="the value for --duration bars|seconds "
                        "(defaults to --bars)")
    p.add_argument("--tempo", dest="tempo_mode", default="sync",
                   choices=render.TEMPO_MODES,
                   help="off: no beatmatching, native tempos. sync: one tempo "
                        "for the set, from the master deck. blend: each track "
                        "at its own tempo, gliding at every join. auto: steady "
                        "until the next track is more than 5%% away")
    p.add_argument("--audition", action="store_true",
                   help="render each join's plausible transitions and keep "
                        "whichever measures best (hole, double bass, mud, "
                        "harmonic clash, vocal collision)")
    p.add_argument("--limit", type=int, help="use only the first N files")
    p.add_argument("--force", action="store_true", help="ignore the cache")
    p.add_argument("--convert", metavar="FILE",
                   help="convert an existing mix to MP3 and exit")
    p.add_argument("--mp3", action="store_true",
                   help="also write an MP3 next to the rendered WAV")
    p.add_argument("--kbps", type=int, default=audio.MP3_KBPS,
                   help=f"MP3 bitrate (default {audio.MP3_KBPS}, LAME's max)")
    a = p.parse_args(argv)

    if a.gui:
        from autodj.gui import run
        return run(a.input)

    if a.convert:
        return convert(a.convert, kbps=a.kbps)

    rc = build(a.input, a.output, a.bars, a.arc, a.limit, a.force,
               a.style, a.align, a.sections, a.minutes, a.block_bars,
               a.duration_mode, a.duration_value, a.tempo_mode, a.audition)
    if rc == 0 and a.mp3 and Path(a.output).suffix.lower() != ".mp3":
        convert(a.output, kbps=a.kbps)
    return rc


if __name__ == "__main__":
    sys.exit(main())
