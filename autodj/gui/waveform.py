"""Waveform, beatgrid, structure segments and cue markers, drawn with QPainter.

Not matplotlib. A DJ waveform is redrawn on every resize, hover, scrub and
playhead tick, and matplotlib is far too slow for that -- but the real reason
is control: the beatgrid, the phrase lines and the segment bands all have to
agree with the design system and with each other, and that is easier to
guarantee when we own every pixel.

The one performance rule that matters: never hand Qt a million points. The peak
envelope is reduced to one min/max pair per screen column, once per resize or
zoom, and drawing is then linear in the width of the widget rather than the
length of the track.
"""

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QLinearGradient, QPainter,
                         QPainterPath, QPen)
from PyQt6.QtWidgets import QMenu, QWidget

from . import theme as T

# Structure segment colours are `theme.SEGMENTS`, keyed by role rather than by
# index so the same kind of section is the same colour in every track. They
# used to be a hex table right here, which is precisely the drift the theme
# module's docstring claims does not happen -- and being outside the palette,
# they were never in the contrast audit, where three of the seven turned out
# to be effectively invisible.

# What a click does while a correction mode is active. `None` is the normal
# state, where clicking only moves the playhead.
EDIT_MODES = (None, "downbeat", "first_full_bar", "outro_start_bar", "drop")

EDIT_LABELS = {
    "downbeat": "Click a kick to set the downbeat",
    "first_full_bar": "Click where the track really starts",
    "outro_start_bar": "Click where the outro begins",
    "drop": "Click a drop  ·  right-click one to remove it",
}


class WaveformView(QWidget):
    """Peak envelope + beatgrid + segments + cues for one analysed track.

    Also the correction surface. Analysis gets the grid right on almost
    everything and catastrophically wrong on the rest, and until there was a
    way to say "the downbeat is *here*" the only recourse was to re-run the
    same algorithm and get the same answer. Clicking the waveform with an edit
    mode active is that way: the click is a statement about the music, and
    `corrections` turns it into an input the analyser has to respect.
    """

    scrubbed = pyqtSignal(float)          # position in seconds
    edited = pyqtSignal(str, float)       # field, seconds
    removed = pyqtSignal(str, float)      # field, seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.audio = None
        self.rate = 44100
        self.meta = None
        self._peaks = None                # (2, width) min/max per column
        self._cols = 0
        self._span = None                 # (t0, t1) visible window
        self.playhead = None
        self.hover_x = None
        self.show_grid = True
        self.show_segments = True
        self.edit_mode = None
        self.follow = True
        self.label = ""
        self.accent = T.ACCENT_HI
        # Set here and not only in mousePressEvent. It is assigned there in
        # one branch -- the one where the press lands on a draggable cue --
        # but mouseReleaseEvent reads it on every release, so any ordinary
        # click on the waveform raised AttributeError when the button came
        # back up.
        self._drag_cue = None

    # ------------------------------------------------------------- data ----
    def set_track(self, audio, rate, meta, label=""):
        self.audio = None if audio is None else np.atleast_2d(audio).mean(0)
        self.rate = rate
        self.meta = meta
        self.label = label
        self._peaks = None
        self._span = None
        self.update()

    def clear(self):
        self.set_track(None, self.rate, None)

    def duration(self):
        if self.audio is None:
            return 0.0
        return len(self.audio) / self.rate

    def set_playhead(self, seconds):
        """Move the playhead, scrolling the view if it would leave it."""
        self.playhead = None if seconds is None else float(seconds)
        if self.follow and self.playhead is not None and self._span:
            t0, t1 = self._span
            if not (t0 <= self.playhead <= t1):
                w = t1 - t0
                start = max(0.0, self.playhead - w * 0.25)
                self._span = (start, start + w)
                self._peaks = None
        self.update()

    # ------------------------------------------------------------- view ----
    def span(self):
        """The visible window in seconds, defaulting to the whole track."""
        dur = self.duration()
        if self._span is None or dur <= 0:
            return 0.0, dur
        t0, t1 = self._span
        return max(0.0, t0), min(dur, max(t0 + 0.05, t1))

    def set_span(self, t0, t1):
        dur = self.duration()
        if dur <= 0:
            return
        width = max(0.5, min(t1 - t0, dur))
        t0 = float(np.clip(t0, 0.0, max(0.0, dur - width)))
        self._span = None if width >= dur - 1e-6 else (t0, t0 + width)
        self._peaks = None
        self.update()

    def zoom(self, factor, centre=None):
        t0, t1 = self.span()
        if t1 <= t0:
            return
        w = (t1 - t0) / factor
        c = centre if centre is not None else (
            self.playhead if self.playhead is not None else (t0 + t1) / 2)
        self.set_span(c - w / 2, c + w / 2)

    def wheelEvent(self, e):
        if self.audio is None:
            return
        t0, t1 = self.span()
        at = t0 + e.position().x() / max(1, self.width()) * (t1 - t0)
        self.zoom(1.25 if e.angleDelta().y() > 0 else 1 / 1.25, at)
        e.accept()

    def _x(self, t):
        t0, t1 = self.span()
        if t1 <= t0:
            return 0.0
        return (t - t0) / (t1 - t0) * self.width()

    def _t(self, x):
        t0, t1 = self.span()
        return t0 + x / max(1, self.width()) * (t1 - t0)

    def _build_peaks(self, cols):
        """Reduce the visible window to one (min, max) per column.

        Reservoir-style decimation, not sub-sampling: taking every Nth sample
        would miss transients entirely and draw a waveform that is quietly
        wrong -- a kick could vanish between two sample points.
        """
        if self.audio is None or cols < 2:
            self._peaks, self._cols = None, cols
            return
        t0, t1 = self.span()
        a = int(np.clip(t0 * self.rate, 0, len(self.audio) - 1))
        b = int(np.clip(t1 * self.rate, a + 1, len(self.audio)))
        window = self.audio[a:b]
        edges = np.linspace(0, len(window), cols + 1).astype(int)
        lo = np.empty(cols, dtype=np.float32)
        hi = np.empty(cols, dtype=np.float32)
        for i in range(cols):
            s, e = edges[i], max(edges[i] + 1, edges[i + 1])
            chunk = window[s:e]
            if chunk.size:
                lo[i], hi[i] = chunk.min(), chunk.max()
            else:
                lo[i] = hi[i] = 0.0
        self._peaks = np.vstack([lo, hi])
        self._cols = cols

    # ------------------------------------------------------------ paint ----
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(T.BG))

        if self.audio is None:
            p.setPen(QColor(T.TEXT_MUTED))
            p.setFont(T.qfont("body"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Select a track to see its waveform")
            return

        band_h = 14 if (self.show_segments and self.meta) else 0
        wave_top = band_h + 4
        wave_h = h - wave_top - 17
        mid = wave_top + wave_h / 2

        if self.show_segments and self.meta:
            self._paint_segments(p, w, band_h)
        if self.show_grid and self.meta:
            self._paint_grid(p, w, wave_top, wave_h)

        if self._peaks is None or self._cols != w:
            self._build_peaks(w)

        if self._peaks is not None:
            grad = QLinearGradient(0, wave_top, 0, wave_top + wave_h)
            grad.setColorAt(0.0, QColor(T.SERIES[1]))
            grad.setColorAt(0.5, QColor(self.accent))
            grad.setColorAt(1.0, QColor(T.SERIES[1]))
            p.setPen(QPen(QBrush(grad), 1.0))
            lo, hi = self._peaks
            scale = wave_h / 2 * 0.94
            for x in range(self._peaks.shape[1]):
                y0 = mid - hi[x] * scale
                y1 = mid - lo[x] * scale
                if y1 - y0 < 1:
                    y0, y1 = mid - 0.5, mid + 0.5
                p.drawLine(QPointF(x, y0), QPointF(x, y1))

        if self.meta:
            self._paint_cues(p, w, wave_top, wave_h)

        if self.hover_x is not None:
            p.setPen(QPen(QColor(T.TEXT_MUTED), 1))
            p.drawLine(QPointF(self.hover_x, wave_top),
                       QPointF(self.hover_x, wave_top + wave_h))
        if self.playhead is not None:
            x = self._x(self.playhead)
            if -2 <= x <= w + 2:
                p.setPen(QPen(QColor(T.TEXT), 2))
                p.drawLine(QPointF(x, wave_top), QPointF(x, wave_top + wave_h))
                head = QPainterPath()
                head.moveTo(x - 5, wave_top)
                head.lineTo(x + 5, wave_top)
                head.lineTo(x, wave_top + 7)
                p.fillPath(head, QBrush(QColor(T.TEXT)))

        self._paint_ruler(p, w, h)
        if self.edit_mode:
            self._paint_edit_banner(p, w)
        if self.label:
            # Top-left on a chip, not along the bottom. The bottom is where
            # the time ruler lives, and a track title drawn over "3:10" is
            # unreadable twice over.
            p.setFont(T.qfont("caption"))
            fm = p.fontMetrics()
            text = fm.elidedText(self.label, Qt.TextElideMode.ElideRight,
                                 int(w * 0.55))
            box = QRectF(7, wave_top + 4, fm.horizontalAdvance(text) + 14, 18)
            path = QPainterPath()
            path.addRoundedRect(box, 6, 6)
            bg = QColor(T.BG)
            bg.setAlpha(215)
            p.fillPath(path, QBrush(bg))
            p.setPen(QColor(T.TEXT_DIM))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_edit_banner(self, p, w):
        text = EDIT_LABELS.get(self.edit_mode, "")
        p.setFont(T.qfont("caption"))
        box = QRectF(w / 2 - 150, 3, 300, 19)
        path = QPainterPath()
        path.addRoundedRect(box, 9, 9)
        c = QColor(T.WARNING)
        c.setAlpha(38)
        p.fillPath(path, QBrush(c))
        p.setPen(QPen(QColor(T.WARNING), 1))
        p.drawPath(path)
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _bar(self):
        """Seconds per bar, or None when this view has no beatgrid.

        Returns None rather than raising because two callers are legitimately
        gridless. The mix view holds the rendered set: real audio, but no
        single track's `meta`, because it is not one track. And a BPM of zero
        arrives from a correction field the moment it is cleared, mid-typing.
        Neither is an error, and neither should reach a division.
        """
        bpm = (self.meta or {}).get("bpm") or 0.0
        return 4 * 60.0 / bpm if bpm > 0 else None

    def _paint_segments(self, p, w, band_h):
        cues = (self.meta or {}).get("cues") or {}
        bar = self._bar()
        if bar is None:
            return
        first = self.meta.get("first_downbeat", 0.0)
        for s in cues.get("segments") or []:
            a = first + s.get("start_bar", 0) * bar
            b = first + s.get("end_bar", s.get("start_bar", 0) + 8) * bar
            x0, x1 = self._x(a), self._x(b)
            if x1 < 0 or x0 > w:
                continue
            colour = QColor(T.SEGMENTS.get(s.get("label", "verse"),
                                        T.HAIRLINE_HI))
            colour.setAlpha(200)
            path = QPainterPath()
            path.addRoundedRect(QRectF(x0 + 1, 0, max(2.0, x1 - x0 - 2),
                                       band_h), 4, 4)
            p.fillPath(path, QBrush(colour))

    def _paint_grid(self, p, w, top, hgt):
        """Beat lines, with every 4th and every 16th progressively stronger.

        The hierarchy is the point: a flat grid of identical lines tells you
        the tempo but not the phrasing, and phrasing is what you actually mix
        on. Beats are barely visible, bars readable, phrases obvious.
        """
        t0, t1 = self.span()
        bpm = (self.meta or {}).get("bpm") or 0.0
        if t1 <= t0 or bpm <= 0:
            return
        beat = 60.0 / bpm
        off = self.meta.get("beat_offset", 0.0)
        phase = int(self.meta.get("downbeat_phase", 0))
        px_per_beat = beat / (t1 - t0) * w
        if px_per_beat < 1.2:
            return
        i0 = int(np.floor((t0 - off) / beat)) - 1
        i1 = int(np.ceil((t1 - off) / beat)) + 1
        for i in range(i0, i1 + 1):
            x = self._x(off + i * beat)
            if x < -2 or x > w + 2:
                continue
            k = (i - phase) % 4
            bar_index = (i - phase) // 4
            if k != 0:
                if px_per_beat < 4:
                    continue
                p.setPen(QPen(QColor(T.HAIRLINE), 1))
                p.drawLine(QPointF(x, top + hgt * 0.42),
                           QPointF(x, top + hgt * 0.58))
            elif bar_index % 4 == 0:
                c = QColor(self.accent)
                c.setAlpha(110)
                p.setPen(QPen(c, 1))
                p.drawLine(QPointF(x, top), QPointF(x, top + hgt))
            else:
                c = QColor(T.HAIRLINE_HI)
                c.setAlpha(150)
                p.setPen(QPen(c, 1))
                p.drawLine(QPointF(x, top + hgt * 0.22),
                           QPointF(x, top + hgt * 0.78))

    def cue_marks(self):
        """(seconds, colour, label, field) for every cue on this track."""
        cues = (self.meta or {}).get("cues") or {}
        bar = self._bar()
        if bar is None:
            return []
        first = self.meta.get("first_downbeat", 0.0)
        marks = []
        for key, label in (("first_full_bar", "IN"),
                           ("outro_start_bar", "OUT")):
            if key in cues:
                colour = T.SUCCESS if label == "IN" else T.WARNING
                marks.append((first + cues[key] * bar, colour, label, key))
        for b in (cues.get("drop_bars") or []):
            marks.append((first + b * bar, T.SERIES[1], "DROP", "drop"))
        return marks

    def _paint_cues(self, p, w, top, hgt):
        p.setFont(T.qfont("caption"))
        # Below the title chip, not beside it: at the top of the wave area a
        # cue tag lands squarely on the track name.
        tag_y = top + (24 if self.label else 2)
        for t, colour, label, _ in self.cue_marks():
            x = self._x(t)
            if x < -20 or x > w + 20:
                continue
            p.setPen(QPen(QColor(colour), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, top), QPointF(x, top + hgt))
            # Flip to the left of the line near the right edge, so a cue in
            # the last seconds of a track does not have its label sliced off
            # by the card -- which is exactly where an OUT marker lives.
            width = 42.0
            left = x + 2 if x + 2 + width <= w - 3 else x - 2 - width
            tag = QRectF(max(2.0, left), tag_y, width, 15)
            path = QPainterPath()
            path.addRoundedRect(tag, 4, 4)
            c = QColor(colour)
            c.setAlpha(45)
            p.fillPath(path, QBrush(c))
            p.setPen(QColor(colour))
            p.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_ruler(self, p, w, h):
        t0, t1 = self.span()
        if t1 <= t0:
            return
        p.setFont(T.qfont("caption"))
        p.setPen(QColor(T.TEXT_MUTED))
        window = t1 - t0
        step = next(s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300)
                    if window / s <= 12) if window > 0 else 30
        t = np.ceil(t0 / step) * step
        while t <= t1:
            p.drawText(QRectF(self._x(t) + 3, h - 15, 60, 13),
                       Qt.AlignmentFlag.AlignLeft,
                       f"{int(t)//60}:{int(t)%60:02d}")
            t += step

    # ----------------------------------------------------------- input ----
    def _cue_at(self, x, tol=6):
        for t, _, _, field in self.cue_marks():
            if abs(self._x(t) - x) <= tol:
                return field, t
        return None, None

    def mouseMoveEvent(self, e):
        self.hover_x = e.position().x()
        if self.meta and not self.edit_mode:
            field, _ = self._cue_at(self.hover_x)
            self.setCursor(Qt.CursorShape.SizeHorCursor if field
                           else Qt.CursorShape.ArrowCursor)
        elif self.edit_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def leaveEvent(self, _):
        self.hover_x = None
        self.update()

    def mousePressEvent(self, e):
        if self.audio is None or self.width() < 2:
            return
        t = float(np.clip(self._t(e.position().x()), 0.0, self.duration()))

        if e.button() == Qt.MouseButton.RightButton:
            field, at = self._cue_at(e.position().x())
            if field:
                self.removed.emit(field, at)
            return

        if self.edit_mode:
            self.edited.emit(self.edit_mode, t)
            return

        # Dragging an existing marker is the fast path -- no mode to arm, just
        # grab the thing that is visibly in the wrong place.
        field, _ = self._cue_at(e.position().x())
        if field and field != "drop":
            self._drag_cue = field
            return

        self.playhead = t
        self.scrubbed.emit(t)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag_cue:
            self.edited.emit(self._drag_cue,
                             float(np.clip(self._t(e.position().x()), 0.0,
                                           self.duration())))
            self._drag_cue = None


class TimelineView(QWidget):
    """The planned set: one block per track, transitions marked at the joins.

    Editable, which is the whole difference between a picture of a plan and a
    plan. Blocks drag to reorder, the marker at each join opens the transition
    inspector, and the playhead runs across the whole thing so you can see
    which part of the set you are listening to.
    """

    selected = pyqtSignal(int)            # position in the set
    reordered = pyqtSignal(int, int)      # from position, to position
    join_clicked = pyqtSignal(int)        # index of the join
    seeked = pyqtSignal(float)            # seconds into the mix

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(112)
        self.setMouseTracking(True)
        self.segs = []
        self.metas = []
        self.order = []
        self.joins = []
        self.spb = 1.0
        self.rate = 44100
        self.hover = -1
        self.current = -1
        self.playhead = None
        self._drag_from = -1
        self._drag_x = None

    def set_plan(self, metas, order, segs, joins, spb, rate):
        self.metas, self.order = metas, list(order)
        self.segs, self.joins = segs, joins
        self.spb, self.rate = spb, rate
        self.update()

    def set_playhead(self, seconds):
        self.playhead = None if seconds is None else float(seconds)
        self.update()

    def _start(self, seg):
        if "start_sample" in seg:
            return seg["start_sample"] / self.rate
        return seg["mix_start"] * self.spb / self.rate

    def _end(self, seg):
        if "len_samples" in seg:
            return self._start(seg) + seg["len_samples"] / self.rate
        return (seg["mix_start"] + seg["bars"]) * self.spb / self.rate

    def _total(self):
        if not self.segs:
            return 1.0
        return max(1e-6, self._end(self.segs[-1]))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(T.BG))
        w, h = self.width(), self.height()
        if not self.segs:
            p.setPen(QColor(T.TEXT_MUTED))
            p.setFont(T.qfont("body"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Plan a set to see the timeline")
            return

        total = self._total()
        top, block_h = 12, h - 46
        p.setFont(T.qfont("caption"))

        for i, seg in enumerate(self.segs):
            x0 = self._start(seg) / total * w
            x1 = self._end(seg) / total * w
            r = QRectF(x0 + 2, top, max(6.0, x1 - x0 - 4), block_h)

            base = QColor(T.SERIES[i % len(T.SERIES)])
            if i == self._drag_from:
                base = base.darker(140)
            elif i == self.current:
                base = base.lighter(125)
            elif i == self.hover:
                base = base.lighter(112)
            grad = QLinearGradient(0, r.top(), 0, r.bottom())
            c1, c2 = QColor(base), QColor(base)
            # Opaque at the top, where the title sits. At alpha 210 the title
            # was reading against a washed mid-blue that no foreground clears
            # 4.5 against -- neither black nor white got past 4.33. The fade
            # still happens, just below the text.
            c1.setAlpha(255)
            c2.setAlpha(95)
            grad.setColorAt(0.0, c1)
            grad.setColorAt(1.0, c2)
            path = QPainterPath()
            path.addRoundedRect(r, 8, 8)
            p.fillPath(path, QBrush(grad))
            if i == self.current:
                p.setPen(QPen(QColor(T.TEXT), 2))
                p.drawPath(path)

            m = self.metas[self.order[i]]
            # Each label is coloured against the colour actually under it, not
            # against a token. The block is a data colour drawn translucent
            # over the card, so the top of it and the bottom of it are two
            # different backgrounds -- and in light mode a fixed white label
            # was sitting on near-white at the bottom edge.
            base_hex = base.name()
            p.setPen(QColor(T.on_colour(T.blend(base_hex, 255, T.SURFACE))))
            p.drawText(r.adjusted(9, 7, -9, 0),
                       Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                       f"{m['artist']} - {m['title']}"[:44])
            sub = QColor(T.on_colour(T.blend(base_hex, 95, T.SURFACE)))
            sub.setAlpha(205)
            p.setPen(sub)
            bpm = seg.get("bpm") or m["bpm"]
            p.drawText(r.adjusted(9, 0, -9, -7),
                       Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                       f"{bpm:.1f} · {m['camelot']} · bars {seg['enter']}"
                       f"-{seg['exit']}")

            if i < len(self.joins):
                self._paint_join(p, i, seg, x1, total, w, h, top, block_h)

        if self.playhead is not None:
            x = self.playhead / total * w
            p.setPen(QPen(QColor(T.TEXT), 2))
            p.drawLine(QPointF(x, top - 4), QPointF(x, top + block_h + 4))

        if self._drag_from >= 0 and self._drag_x is not None:
            p.setPen(QPen(QColor(T.ACCENT_HI), 3))
            slot = self._slot_at(self._drag_x)
            x = (self._start(self.segs[slot]) / total * w if slot < len(self.segs)
                 else w)
            p.drawLine(QPointF(x, top - 6), QPointF(x, top + block_h + 6))

        p.setPen(QColor(T.TEXT_MUTED))
        p.drawText(QRectF(0, h - 15, w - 4, 14), Qt.AlignmentFlag.AlignRight,
                   f"{total/60:.1f} min total")

    def _paint_join(self, p, i, seg, x1, total, w, h, top, block_h):
        tail_s = (seg.get("tail_samples", 0) / self.rate
                  if "tail_samples" in seg
                  else seg["tail"] * self.spb / self.rate)
        jx = x1 - tail_s / total * w
        p.setPen(QPen(QColor(T.TEXT), 1, Qt.PenStyle.DotLine))
        p.drawLine(QPointF(jx, top), QPointF(jx, top + block_h))

        j = self.joins[i]
        name = getattr(j, "name", str(j))
        bars = getattr(j, "bars", "")
        chip = QRectF(jx - 58, h - 33, 116, 21)
        path = QPainterPath()
        path.addRoundedRect(chip, 10, 10)
        p.fillPath(path, QBrush(QColor(T.SURFACE_2)))
        p.setPen(QPen(QColor(T.CONTROL_EDGE), 1))
        p.drawPath(path)
        # The transition's own mark, drawn from its curves, so the chip says
        # what the move does and not only what it is called.
        from . import icons
        p.drawPixmap(QRectF(chip.left() + 6, chip.top() + 2.5, 16, 16).toRect(),
                     icons.transition_pixmap(name, 16))
        p.setPen(QColor(T.TEXT_DIM))
        p.drawText(chip.adjusted(24, 0, -4, 0),
                   Qt.AlignmentFlag.AlignCenter,
                   f"{LABELS.get(name, name)} · {bars}")

    def _hit(self, x):
        total = self._total()
        for i, seg in enumerate(self.segs):
            if (self._start(seg) / total * self.width() <= x
                    <= self._end(seg) / total * self.width()):
                return i
        return -1

    def _join_hit(self, pos):
        """Which transition chip, if any, is under this point."""
        total, w, h = self._total(), self.width(), self.height()
        if pos.y() < h - 34:
            return -1
        for i, seg in enumerate(self.segs[:len(self.joins)]):
            tail_s = (seg.get("tail_samples", 0) / self.rate
                      if "tail_samples" in seg
                      else seg["tail"] * self.spb / self.rate)
            jx = self._end(seg) / total * w - tail_s / total * w
            if abs(pos.x() - jx) <= 54:
                return i
        return -1

    def _slot_at(self, x):
        """Insertion index for a drag ending at this x."""
        total = self._total()
        for i, seg in enumerate(self.segs):
            mid = (self._start(seg) + self._end(seg)) / 2 / total * self.width()
            if x < mid:
                return i
        return len(self.segs)

    def mouseMoveEvent(self, e):
        if self._drag_from >= 0:
            self._drag_x = e.position().x()
            self.update()
            return
        hit = self._hit(e.position().x())
        if hit != self.hover:
            self.hover = hit
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit >= 0
                           else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.update()

    def mousePressEvent(self, e):
        join = self._join_hit(e.position())
        if join >= 0:
            self.join_clicked.emit(join)
            return
        hit = self._hit(e.position().x())
        if hit < 0:
            return
        if e.button() == Qt.MouseButton.RightButton:
            self.seeked.emit(self._start(self.segs[hit]))
            return
        self._drag_from = hit
        self._drag_x = e.position().x()
        self.current = hit
        self.selected.emit(hit)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag_from < 0:
            return
        start, self._drag_from = self._drag_from, -1
        x, self._drag_x = self._drag_x, None
        if x is None or abs(x - e.position().x()) < 12:
            self.update()
            return
        slot = self._slot_at(e.position().x())
        if slot > start:
            slot -= 1
        if slot != start:
            self.reordered.emit(start, slot)
        self.update()

    def mouseDoubleClickEvent(self, e):
        hit = self._hit(e.position().x())
        if hit >= 0:
            self.seeked.emit(self._start(self.segs[hit]))


# djay's vocabulary, which is what people know these moves by. The internal
# names stay as they are -- they describe what the DSP does -- but nothing is
# gained by making a user learn "vocal_slam_drop" when the rest of the world
# calls it something else.
LABELS = {
    "dissolve": "Dissolve",
    "smooth_swap": "Smooth",
    "fade": "Fade",
    "filter_sweep": "Filter",
    "ladder_sweep": "Ladder filter",
    "bass_swap": "EQ",
    "eq_blend": "EQ blend",
    "echo_out": "Echo",
    "reverb_wash": "Reverb wash",
    "cut_with_echo": "Cut + echo",
    "hard_cut": "Cut",
    "loop_roll": "Loop roll",
    "riser_cut": "Riser",
    "tremolo": "Tremolo",
    "stem_blend": "Neural",
    "double_drop": "Double drop",
    "vocal_slam_drop": "Vocal slam",
    "euro_rap_breakout": "Rap breakout",
}
