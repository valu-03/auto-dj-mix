"""Reusable pieces: cards, stat tiles, charts, toasts, skeletons, empty states.

Every widget here is painted or styled from `theme`, never with inline colours.
The charts are drawn with QPainter rather than pulled from a charting library:
QtCharts is not installed, and more importantly a hand-drawn chart can obey the
design system exactly -- rounded caps, our own grid weight, our own palette --
instead of being talked out of it by someone else's defaults.
"""

import numpy as np
from PyQt6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF,
                          QSize, Qt, QTimer, pyqtProperty)
from PyQt6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter,
                         QPainterPath, QPen)
from PyQt6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout,
                             QLabel, QSizePolicy, QVBoxLayout, QWidget)

from . import theme as T


def _f(name, family=None):
    return T.qfont(name, family)


class Card(QFrame):
    """A bento cell. Title, optional hint, and a body you fill in."""

    def __init__(self, title="", hint="", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(T.PAD_CARD, T.PAD_CARD, T.PAD_CARD, T.PAD_CARD)
        outer.setSpacing(T.UNIT * 2)

        if title:
            head = QHBoxLayout()
            head.setSpacing(T.UNIT)
            lbl = QLabel(title)
            lbl.setObjectName("CardTitle")
            head.addWidget(lbl)
            head.addStretch(1)
            if hint:
                h = QLabel(hint)
                h.setObjectName("CardHint")
                head.addWidget(h)
            outer.addLayout(head)
            self.head = head

        self.body = QVBoxLayout()
        self.body.setSpacing(T.UNIT * 2)
        outer.addLayout(self.body, 1)


class StatTile(QFrame):
    """One number, its label, and an optional delta.

    The number is the widget. Label above in muted uppercase, delta below in a
    semantic colour -- so the eye lands on the value first and only then asks
    what it is, which is the order you actually read a dashboard in.
    """

    def __init__(self, label, value="--", delta=None, tone="neutral",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(T.PAD_CARD, int(T.UNIT * 2.5), T.PAD_CARD,
                               int(T.UNIT * 2.5))
        lay.setSpacing(int(T.UNIT // 2))

        self.label = QLabel(label.upper())
        self.label.setObjectName("MetricLabel")
        self.value = QLabel(value)
        self.value.setObjectName("Metric")
        self.delta = QLabel(delta or "")
        self.delta.setObjectName("CardHint")

        lay.addWidget(self.label)
        lay.addWidget(self.value)
        lay.addWidget(self.delta)
        self.set_tone(tone)

    def set_tone(self, tone):
        colour = {"good": T.SUCCESS, "bad": T.DANGER, "warn": T.WARNING,
                  "neutral": T.TEXT_MUTED}.get(tone, T.TEXT_MUTED)
        self.delta.setStyleSheet(f"color:{colour}; {T.font_css('caption')}")

    def set(self, value, delta=None, tone=None):
        self.value.setText(str(value))
        if delta is not None:
            self.delta.setText(delta)
        if tone:
            self.set_tone(tone)


class Sparkline(QWidget):
    """A small filled line chart. Rounded caps, soft gradient, no axes.

    No grid and no labels on purpose: at this size they would cost more
    legibility than they add. It answers "what shape is this" -- the exact
    numbers live in the tile beside it.
    """

    def __init__(self, values=None, colour=None, parent=None):
        super().__init__(parent)
        self.values = np.asarray(values if values is not None else [],
                                 dtype=float)
        self.colour = QColor(colour or T.SERIES[0])
        self.setMinimumHeight(56)

    def set_values(self, values):
        self.values = np.asarray(values, dtype=float)
        self.update()

    def paintEvent(self, _):
        if self.values.size < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 4
        v = self.values
        lo, hi = float(v.min()), float(v.max())
        rng = (hi - lo) or 1.0
        xs = np.linspace(pad, w - pad, len(v))
        ys = h - pad - (v - lo) / rng * (h - 2 * pad)

        line = QPainterPath(QPointF(xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            line.lineTo(QPointF(x, y))

        fill = QPainterPath(line)
        fill.lineTo(QPointF(xs[-1], h))
        fill.lineTo(QPointF(xs[0], h))
        fill.closeSubpath()
        grad = QLinearGradient(0, 0, 0, h)
        c = QColor(self.colour)
        c.setAlpha(80)
        grad.setColorAt(0.0, c)
        c2 = QColor(self.colour)
        c2.setAlpha(0)
        grad.setColorAt(1.0, c2)
        p.fillPath(fill, QBrush(grad))

        pen = QPen(self.colour, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(line)


class BarChart(QWidget):
    """Rounded vertical bars with a value axis. Used for per-track metrics."""

    def __init__(self, values=None, labels=None, colour=None, unit="",
                 zero=True, parent=None):
        super().__init__(parent)
        self.values = list(values or [])
        self.labels = list(labels or [])
        self.colour = QColor(colour or T.SERIES[0])
        self.unit = unit
        # `zero=False` scales to the data instead of to zero. For a quantity
        # that never goes near zero -- tempo being the obvious one -- a zero
        # baseline spends the whole chart on the empty space below the data and
        # renders 130 and 135 BPM as four identical bars.
        self.zero = zero
        self.setMinimumHeight(180)

    def set_data(self, values, labels=None):
        self.values = list(values)
        if labels is not None:
            self.labels = list(labels)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if not self.values:
            p.setPen(QColor(T.TEXT_MUTED))
            p.setFont(_f("caption"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "No data yet")
            return

        left, bottom, top = 44, 26, 12
        plot = QRectF(left, top, w - left - 8, h - top - bottom)
        lo, hi = min(self.values), max(self.values)
        if self.zero:
            vmax = hi * 1.15 or 1.0
            vmin = min(0.0, lo)
        else:
            pad = (hi - lo) * 0.35 or max(abs(hi) * 0.02, 1.0)
            vmin, vmax = lo - pad, hi + pad

        # Grid: three lines only. More would compete with the bars.
        p.setFont(_f("caption"))
        for i in range(4):
            frac = i / 3
            y = plot.bottom() - frac * plot.height()
            p.setPen(QPen(QColor(T.HAIRLINE), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(T.TEXT_MUTED))
            p.drawText(QRectF(0, y - 9, left - 8, 18),
                       Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter,
                       f"{vmin + frac * (vmax - vmin):.1f}"
                       if not self.zero else
                       f"{vmin + frac * (vmax - vmin):.0f}")

        n = len(self.values)
        slot = plot.width() / n
        bw = min(38.0, slot * 0.6)
        for i, v in enumerate(self.values):
            frac = (v - vmin) / (vmax - vmin or 1.0)
            bh = frac * plot.height()
            x = plot.left() + i * slot + (slot - bw) / 2
            r = QRectF(x, plot.bottom() - bh, bw, bh)
            grad = QLinearGradient(0, r.top(), 0, r.bottom())
            grad.setColorAt(0.0, self.colour)
            c = QColor(self.colour)
            c.setAlpha(120)
            grad.setColorAt(1.0, c)
            path = QPainterPath()
            path.addRoundedRect(r, 6, 6)
            p.fillPath(path, QBrush(grad))
            if i < len(self.labels):
                p.setPen(QColor(T.TEXT_MUTED))
                # Elided to the slot, not truncated by the caller. A fixed
                # character count cuts "Tonight Is The Night" to "Tonigh" no
                # matter how much room the chart has, and cuts nothing at all
                # when the chart is narrow and the label still does not fit.
                box = QRectF(plot.left() + i * slot, plot.bottom() + 5,
                             slot, 18)
                p.drawText(box, Qt.AlignmentFlag.AlignCenter,
                           p.fontMetrics().elidedText(
                               str(self.labels[i]),
                               Qt.TextElideMode.ElideRight,
                               int(slot - 6)))


class Skeleton(QWidget):
    """Loading placeholder with a sweeping highlight.

    A skeleton beats a spinner because it says *what* is coming and roughly how
    much of it, so the layout does not jump when the content lands.
    """

    def __init__(self, rows=4, parent=None):
        super().__init__(parent)
        self.rows = rows
        self._phase = 0.0
        self.setMinimumHeight(rows * 30)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self):
        self._phase = (self._phase + 0.022) % 1.6
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        for i in range(self.rows):
            y = i * 30
            rw = w * (0.95 if i % 3 else 0.66)
            r = QRectF(0, y, rw, 16)
            path = QPainterPath()
            path.addRoundedRect(r, 6, 6)
            grad = QLinearGradient(self._phase * w - w * 0.3, 0,
                                   self._phase * w + w * 0.3, 0)
            grad.setColorAt(0.0, QColor(T.SURFACE_2))
            grad.setColorAt(0.5, QColor(T.HAIRLINE_HI))
            grad.setColorAt(1.0, QColor(T.SURFACE_2))
            p.fillPath(path, QBrush(grad))

    def stop(self):
        self._timer.stop()


class EmptyState(QWidget):
    """What the user sees before anything exists. Icon, one line, one action."""

    def __init__(self, glyph="◎", title="Nothing here yet",
                 body="", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(T.UNIT)
        lay.addStretch(1)
        for text, size, colour, weight in (
                (glyph, 40, T.TEXT_MUTED, 400),
                (title, T.TYPE["heading"][0], T.TEXT, T.TYPE["heading"][1]),
                (body, T.TYPE["body"][0], T.TEXT_MUTED, 450)):
            if not text:
                continue
            l = QLabel(text)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setStyleSheet(f"color:{colour}; font-size:{size}px; "
                            f"font-weight:{weight};")
            l.setWordWrap(True)
            lay.addWidget(l)
        self.actions = QHBoxLayout()
        self.actions.addStretch(1)
        lay.addSpacing(T.UNIT)
        lay.addLayout(self.actions)
        lay.addStretch(1)

    def add_action(self, button):
        self.actions.insertWidget(self.actions.count() - 1, button)
        self.actions.addStretch(1)


class Toast(QFrame):
    """A transient banner that fades in, waits, and fades out.

    Parented to the window and positioned manually rather than docked, so it
    floats above the layout without reserving space -- feedback should never
    move the thing the user is looking at.
    """

    def __init__(self, parent, text, tone="info", ms=3200):
        super().__init__(parent)
        colour = {"good": T.SUCCESS, "bad": T.DANGER,
                  "warn": T.WARNING, "info": T.ACCENT}.get(tone, T.ACCENT)
        self.setStyleSheet(
            f"background:{T.SURFACE_2}; border:1px solid {T.CONTROL_EDGE};"
            f"border-left:3px solid {colour}; border-radius:{T.RADIUS_SM}px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 13, 18, 13)
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{T.TEXT}; border:none; background:transparent;")
        lay.addWidget(lbl)

        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._fx.setOpacity(0.0)
        self.adjustSize()
        self._place()
        self.show()

        self._in = self._fade(0.0, 1.0)
        self._in.start()
        QTimer.singleShot(ms, self._out)

    def _place(self):
        """Stack above any toast already on screen, rather than on top of it.

        Every toast used to move to the same corner, so a second message
        landed exactly over the first while the first was still fading -- two
        overlapping translucent panels, which reads as a rendering fault
        rather than as two messages.
        """
        p = self.parent()
        if not p:
            return
        others = [t for t in p.findChildren(Toast)
                  if t is not self and t.isVisible()]
        lift = sum(t.height() + T.UNIT for t in others)
        self.move(p.width() - self.width() - T.PAD_VIEW,
                  p.height() - self.height() - T.PAD_VIEW - lift)

    def _fade(self, a, b):
        anim = QPropertyAnimation(self._fx, b"opacity", self)
        anim.setDuration(T.ANIM_MS)
        anim.setStartValue(a)
        anim.setEndValue(b)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        return anim

    def _out(self):
        self._o = self._fade(1.0, 0.0)
        self._o.finished.connect(self.deleteLater)
        self._o.start()


class Badge(QLabel):
    """Small pill for keys, states, counts."""

    def __init__(self, text="", tone="neutral", parent=None):
        super().__init__(text, parent)
        self.set_tone(tone)

    def set_tone(self, tone):
        self._tone = tone
        colour = {"good": T.SUCCESS, "bad": T.DANGER, "warn": T.WARNING,
                  "accent": T.ACCENT_TEXT,
                  "neutral": T.TEXT_DIM}.get(tone, T.TEXT_DIM)
        # A surface token, not a white tint. `rgba(255,255,255,0.045)` is a
        # lighten-on-dark assumption: on a light ground it is white on white,
        # and the pill loses its fill entirely while keeping its border.
        self.setStyleSheet(
            f"color:{colour}; background:{T.SURFACE_2};"
            f"border:1px solid {T.HAIRLINE}; border-radius:{T.RADIUS_XS}px;"
            f"padding:3px 9px; {T.font_css('caption')}")

    def _restyle(self):
        self.set_tone(getattr(self, "_tone", "neutral"))


class ElidedLabel(QLabel):
    """A label that shortens its own text to fit, at paint time.

    Eliding by calling `setText(fontMetrics().elidedText(...))` from a resize
    handler looks correct and is a timing bug: the width you measure against is
    whatever the label had *before* the layout settled. Set the text too early
    and it elides against the wrong number; set it in a queued callback and it
    is right until something resizes in a way that does not trigger the
    handler. Either way the failure is silent -- the text simply runs off the
    edge of the card with no ellipsis to say it was cut.

    Doing it in `paintEvent` removes the question. The width at paint time is
    the width the label actually has, every time, so there is no ordering to
    get wrong.
    """

    def __init__(self, text="", mode=Qt.TextElideMode.ElideRight, parent=None):
        super().__init__(text, parent)
        self._full = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Preferred)

    def setText(self, text):
        self._full = text or ""
        self.setToolTip(self._full)
        super().setText(self._full)
        self.update()

    def full_text(self):
        return self._full

    def paintEvent(self, _):
        p = QPainter(self)
        p.setFont(self.font())
        p.setPen(self.palette().color(self.foregroundRole()))
        text = p.fontMetrics().elidedText(self._full, self._mode,
                                          max(0, self.width() - 2))
        p.drawText(self.rect(), int(self.alignment()), text)

    def minimumSizeHint(self):
        # Otherwise the layout reserves room for the whole untruncated string
        # and the label pushes everything beside it out of the window.
        h = self.fontMetrics().height()
        return QSize(24, h)
