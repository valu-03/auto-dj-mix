"""Background jobs on QThreads, so the window never freezes.

Analysis is ~3 s per track and a render is over a minute. Both would lock the
event loop solid if run on the GUI thread -- no repaints, no cancel, and
Windows painting the window grey and offering to kill it. Every long job
therefore lives on a QThread and reports back through signals.

One rule holds all of this together: **workers never touch widgets.** They emit
plain data; the GUI thread decides what to draw with it. Qt's object affinity
makes touching a widget from another thread undefined behaviour that usually
looks like it works, right up until it corrupts the paint state.
"""

import time
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .. import audio as audio_mod
from .. import export, library, planner, render
from ..analysis import instruments, track


class Worker(QObject):
    """Base: progress, error and finished, with a cooperative cancel flag."""

    progress = pyqtSignal(int, int, str)      # done, total, message
    message = pyqtSignal(str, str)            # text, tone
    failed = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        """Cooperative, not forced. `QThread.terminate` can leave numpy and
        libsndfile mid-write, so jobs check this between items instead."""
        self._cancel = True

    def _guard(self, fn, *a, **kw):
        try:
            result = fn(*a, **kw)
        except Exception:
            self.failed.emit(traceback.format_exc(limit=6))
            return
        if not self._cancel:
            self.done.emit(result)


class AnalyseWorker(Worker):
    """Analyse every audio file in a folder, one at a time, cache-aware."""

    track_done = pyqtSignal(object)           # meta dict, as each lands

    def __init__(self, folder, force=False):
        super().__init__()
        self.folder, self.force = folder, force

    def run(self):
        self._guard(self._run)

    def _run(self):
        files = library.scan(self.folder)
        if not files:
            # An empty or missing folder is not an error, it is the state the
            # app opens in before you have chosen anything. Raising here threw
            # a red toast over a window whose empty state already says exactly
            # what to do next, and buried the "Choose folder" button under it.
            return [], []
        metas = []
        for i, f in enumerate(files):
            if self._cancel:
                break
            self.progress.emit(i, len(files), f.stem[:60])
            m = track.analyse(f, force=self.force)
            metas.append(m)
            # Emit per track so the table fills in as it goes rather than
            # appearing all at once at the end -- the same work feels far
            # faster when you can watch it happen.
            self.track_done.emit(m)
        self.progress.emit(len(files), len(files), "done")
        return files, metas


class RenderWorker(Worker):
    """Plan and render a full mix, optionally writing every sidecar file.

    `out_path=None` renders for playback only. That is the normal case now:
    you listen first, and writing a WAV, an MP3, a cue sheet and two JSON
    documents is something you ask for once you like what you hear, not a toll
    paid on every attempt.
    """

    def __init__(self, files, metas, out_path=None, bars=None,
                 arc="warmup_to_peak", style="smooth", align=False,
                 want_mp3=True, duration_mode="auto", duration_value=8,
                 tempo_mode="sync", order=None, join_names=None, cache=None,
                 audition=False, sections=False, minutes=8.0,
                 block_bars=None):
        super().__init__()
        self.files, self.metas = files, metas
        self.out_path = Path(out_path) if out_path else None
        self.bars, self.arc, self.style = bars, arc, style
        self.align, self.want_mp3 = align, want_mp3
        self.duration_mode, self.duration_value = duration_mode, duration_value
        self.tempo_mode = tempo_mode
        self.order, self.join_names = order, join_names
        self.cache = cache if cache is not None else {}
        self.audition = audition
        self.sections, self.minutes = sections, minutes
        self.block_bars = block_bars

    def run(self):
        self._guard(self._run)

    def _run(self):
        t0 = time.time()
        self.progress.emit(0, 100, "Planning the set")
        inten = planner.intensity(self.metas)
        target = planner.arc(len(self.metas), self.arc)
        cost = 0.0

        if self.order is not None:
            order = list(self.order)
        else:
            order, cost, inten, target = planner.plan(
                self.metas, shape=self.arc, style=self.style)

        # The lead-voice map is what lets entries dodge a vocal collision and
        # what the duration fit measures against, so it is worth the cost even
        # when no stems exist -- the spectral proxy still discriminates.
        activity = None
        if self.align:
            activity = {}
            for i, f in enumerate(self.files):
                if self._cancel:
                    return None
                self.progress.emit(int(3 + 12 * i / max(1, len(self.files))),
                                   100, f"Mapping lead voice {i + 1}")
                activity[i], _ = instruments.vocal_activity(f, self.metas[i])

        names = self.join_names
        if names is None:
            names = [planner.choose_transition(
                self.metas[order[p]], self.metas[order[p + 1]],
                inten[order[p]], inten[order[p + 1]], self.style)
                for p in range(len(order) - 1)]

        bars = self.bars
        if bars is None:
            bars = planner.plan_bars(order, self.metas, inten, names,
                                     self.duration_mode, self.duration_value,
                                     activity=activity)

        segs = None
        if self.sections:
            from .. import arrange
            bpm = float(np.median([m["bpm"] for m in self.metas]))
            blocks = arrange.build(self.metas, minutes=self.minutes,
                                   block_bars=self.block_bars
                                   or arrange.BLOCK_BARS,
                                   shape=self.arc, bpm=bpm)
            if blocks:
                order = [b["track"] for b in blocks]
                segs = arrange.to_segments(blocks, bars[0] if bars else 8)

        def on_track(i, total, title):
            self.progress.emit(int(18 + 68 * i / max(1, total)), 100,
                               f"Rendering {title[:44]}")

        mixed, rep, segs, joins = render.render(
            self.files, self.metas, order, bars=bars, style=self.style,
            activity=activity, progress=on_track, segs=segs,
            join_names=self.join_names, tempo_mode=self.tempo_mode,
            cache=self.cache, audition=self.audition)

        spb = audio_mod.RENDER_RATE * (4 * 60.0 / rep["mix_bpm"])
        mp3 = None
        if self.out_path is not None:
            self.progress.emit(90, 100, "Writing files")
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            audio_mod.save(self.out_path, mixed, audio_mod.RENDER_RATE)
            self.out_path.with_suffix(".txt").write_text(
                render.tracklist(self.metas, order, segs, joins, spb,
                                 audio_mod.RENDER_RATE), encoding="utf-8")
            self.out_path.with_suffix(".cue").write_text(
                render.cue_sheet(self.metas, order, segs, spb,
                                 audio_mod.RENDER_RATE, self.out_path.name),
                encoding="utf-8")
            export.write(self.out_path.with_suffix(".json"),
                         export.mix_plan(self.metas, order, segs, joins, rep,
                                         bars[0] if bars else 8,
                                         audio_mod.RENDER_RATE, self.arc))
            if self.want_mp3 and self.out_path.suffix.lower() != ".mp3":
                self.progress.emit(96, 100, "Encoding MP3")
                mp3 = self.out_path.with_suffix(".mp3")
                audio_mod.save(mp3, mixed, audio_mod.RENDER_RATE)

        self.progress.emit(100, 100, "Done")
        rep["elapsed_s"] = round(time.time() - t0, 1)
        return {"audio": mixed, "report": rep, "order": order, "segs": segs,
                "joins": joins, "spb": spb, "cost": cost, "bars": bars,
                "intensity": inten, "target": target,
                "path": self.out_path, "mp3": mp3}


class PreviewWorker(Worker):
    """Render one join with a little music either side, for immediate playback.

    Separate from `RenderWorker` because it answers a different question. A
    render asks "what does the set sound like"; a preview asks "does this
    transition work", and the second question is asked twenty times for every
    time the first one is. Mastering half a minute instead of fifteen is what
    makes that affordable.
    """

    def __init__(self, files, metas, order, segs, join, name, bars,
                 cache=None):
        super().__init__()
        self.files, self.metas, self.order = files, metas, order
        self.segs, self.join, self.name, self.bars = segs, join, name, bars
        self.cache = cache if cache is not None else {}

    def run(self):
        self._guard(self._run)

    def _run(self):
        t0 = time.time()
        self.progress.emit(30, 100, f"Previewing join {self.join + 1}")
        audio, rep = render.preview_join(
            self.files, self.metas, self.order, self.segs, self.join,
            self.name, self.bars, cache=self.cache)
        rep["elapsed_s"] = round(time.time() - t0, 1)
        self.progress.emit(100, 100, "Done")
        return {"audio": audio, "report": rep, "join": self.join}


class SeparateWorker(Worker):
    """Stem separation for the selected tracks. GPU-checked before it starts."""

    def __init__(self, files):
        super().__init__()
        self.files = list(files)

    def run(self):
        self._guard(self._run)

    def _run(self):
        from ..stems import separate as sep
        # Fail loudly and immediately rather than running 40x slower in
        # silence, which is what both backends do when the GPU is missing.
        sep.check_device()
        out = {}
        for i, f in enumerate(self.files):
            if self._cancel:
                break
            self.progress.emit(i, len(self.files), f"Separating {f.stem[:44]}")
            stems, elapsed = sep.separate(f)
            out[str(f)] = {"stems": stems, "seconds": elapsed}
        self.progress.emit(len(self.files), len(self.files), "done")
        return out


def start(worker, on_done=None, on_progress=None, on_failed=None,
          on_message=None):
    """Move a worker onto a fresh thread, wire it up and start it.

    Returns (thread, worker). The caller must keep a reference to both: if
    either is garbage collected while running, Qt destroys the thread
    mid-execution and the app dies with no traceback.
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_done:
        worker.done.connect(on_done)
    if on_progress:
        worker.progress.connect(on_progress)
    if on_failed:
        worker.failed.connect(on_failed)
    if on_message:
        worker.message.connect(on_message)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.start()
    return thread, worker
