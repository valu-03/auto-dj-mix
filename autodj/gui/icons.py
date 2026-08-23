"""Icons, drawn rather than shipped.

Two families, and the second one is the interesting half.

**UI icons** are ordinary vector marks -- a list, a target, a stack of layers --
painted with QPainter at whatever size is asked for. Drawn rather than loaded
because a handful of geometric paths is smaller than an icon font, has no
licence attached, follows the theme tokens without a second palette to keep in
sync, and cannot silently fall back to a tofu box on a machine missing a glyph.
The previous version used Unicode symbols and looked, at body size, like a
column of identical grey squares.

**Transition icons are generated from the transitions themselves.** Each one
builds the real `Transition` and plots its actual `out_mid` and `in_mid` gain
curves. So a hard cut's icon is a step because the curve is a step; a dissolve's
is two crossing arcs because that is what equal power looks like; a tremolo's
shows the gate oscillating and accelerating because that is literally the
automation being drawn.

That is worth more than a prettier hand-drawn set. An icon someone draws is a
claim about what a transition does, and claims drift -- this file already found
one, a hand-maintained list of "non-overlapping" transitions that had come to
include two that plainly overlap. A picture computed from the curve cannot
drift: change the DSP and the icon changes with it, and if the icon looks wrong
the transition *is* wrong.

Colour follows the deck convention used everywhere else in the interface:
SERIES[3] is the outgoing track, SERIES[1] the incoming one. They are read at
draw time rather than bound at import, because a module-level `OUT_COLOUR =
T.SERIES[3]` freezes whichever palette happened to be live when this module was
first imported -- and then every deck marking in the app keeps the dark theme's
amber after a switch to light.
"""

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import (QBrush, QColor, QIcon, QLinearGradient, QPainter,
                         QPainterPath, QPen, QPixmap)

from .. import transitions as tr_mod
from . import theme as T


def out_colour():
    """The outgoing deck's colour, in the palette that is live right now."""
    return T.SERIES[3]


def in_colour():
    return T.SERIES[1]

_cache = {}


def _pixmap(size, ratio=2):
    """A transparent pixmap at twice the logical size, for crisp edges."""
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(ratio)
    return pm


def _painter(pm):
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    return p


# ------------------------------------------------------------ transitions ---

def _curve_points(curve, box, invert=False):
    """Sample a gain curve into points inside `box`, top = 1.0."""
    a = np.asarray(curve, dtype=float)
    if a.size < 2:
        return []
    n = 48
    idx = np.linspace(0, a.size - 1, n)
    v = np.clip(np.interp(idx, np.arange(a.size), a), 0.0, 1.0)
    xs = box.left() + np.linspace(0.0, 1.0, n) * box.width()
    ys = box.bottom() - v * box.height()
    return [QPointF(float(x), float(y)) for x, y in zip(xs, ys)]


def _polyline(p, pts, colour, width=1.6):
    if len(pts) < 2:
        return
    path = QPainterPath(pts[0])
    for q in pts[1:]:
        path.lineTo(q)
    p.setPen(QPen(QColor(colour), width, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.drawPath(path)


def _accent(p, name, tr, box):
    """The one extra mark that separates transitions with similar curves.

    `hard_cut` and `cut_with_echo` have identical gain automation -- the echo
    is a separate signal path, not a curve -- so without this they would draw
    the same icon for two different moves. Same for the roll and the riser,
    which are both a hard cut plus something the renderer adds.
    """
    if tr.echo:
        # A decaying tail after the cut.
        x = box.left() + box.width() * float(tr.echo.get("start", 0.5))
        for i in range(3):
            r = 1.7 - i * 0.4
            c = QColor(out_colour())
            c.setAlpha(200 - i * 55)
            p.setBrush(QBrush(c))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x + 3.2 + i * 3.4, box.bottom() - 2.0),
                          r, r)
    if tr.roll:
        # Repeats speeding up into the cut.
        x0 = box.left() + box.width() * float(tr.roll.get("start", 0.3))
        x1 = box.left() + box.width() * tr_mod.CUT_AT
        c = QColor(out_colour())
        p.setPen(QPen(c, 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        gaps, x = [], x1
        step = (x1 - x0) * 0.42
        while x > x0 and step > 0.8:
            gaps.append(x)
            x -= step
            step *= 0.62
        for gx in gaps:
            p.drawLine(QPointF(gx, box.top() + 1.5),
                       QPointF(gx, box.top() + 5.5))
    if tr.riser:
        # A sweep climbing to the cut.
        x0 = box.left() + box.width() * float(tr.riser.get("start", 0.2))
        x1 = box.left() + box.width() * tr_mod.CUT_AT
        path = QPainterPath(QPointF(x0, box.bottom() - 1.0))
        path.quadTo(QPointF((x0 + x1) / 2, box.bottom() - 2.0),
                    QPointF(x1, box.top() + 1.0))
        c = QColor(T.SERIES[5])
        p.setPen(QPen(c, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    if tr.stems:
        # Four stem lanes entering in their own order -- the whole point of a
        # stem transition is that the instruments do not arrive together.
        lanes = ("drums", "other", "bass", "vocals")
        h = box.height() / len(lanes)
        for i, stem in enumerate(lanes):
            curve = np.asarray(tr.stems["in"].get(stem, []), dtype=float)
            if curve.size < 2:
                continue
            enter = float(np.argmax(curve > 0.5)) / curve.size
            y = box.top() + h * (i + 0.5)
            c = QColor(in_colour())
            c.setAlpha(235 - i * 35)
            p.setPen(QPen(c, 1.5, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(box.left() + box.width() * enter, y),
                       QPointF(box.right(), y))
    if name == "double_drop":
        # The bar where both drops land.
        x = box.left() + box.width() * tr_mod.DOUBLE_DROP_AT
        p.setPen(QPen(QColor(T.SUCCESS), 1.4))
        p.drawLine(QPointF(x, box.top() - 1.0), QPointF(x, box.bottom() + 1.0))
    if tr.out_sweep or tr.in_sweep:
        # A filter opening: a widening wedge along the bottom.
        c = QColor(T.SERIES[5])
        c.setAlpha(150)
        p.setPen(QPen(c, 1.2))
        p.drawLine(QPointF(box.left(), box.bottom() + 2.0),
                   QPointF(box.right(), box.bottom() - 1.0))


def transition(name, size=18, bars=8, bpm=133.0):
    """A QIcon showing what this transition actually does to the two decks."""
    key = ("tr", name, size)
    if key in _cache:
        return _cache[key]

    pm = _pixmap(size)
    p = _painter(pm)
    box = QRectF(1.5, 3.0, size - 3.0, size - 7.0)
    try:
        sample_rate = 44100
        spb = sample_rate * (4 * 60.0 / bpm)
        tr = tr_mod.build(name, bars, spb, sample_rate, 60.0 / bpm)
    except Exception:
        p.end()
        icon = QIcon(pm)
        _cache[key] = icon
        return icon

    # Baseline, so a curve that sits at zero is still visibly *at* zero.
    base = QColor(T.TEXT_DIM)
    base.setAlpha(120)
    p.setPen(QPen(base, 1.0))
    p.drawLine(QPointF(box.left(), box.bottom()),
               QPointF(box.right(), box.bottom()))

    _polyline(p, _curve_points(tr.out_mid, box), out_colour())
    _polyline(p, _curve_points(tr.in_mid, box), in_colour())
    _accent(p, name, tr, box)
    p.end()

    icon = QIcon(pm)
    _cache[key] = icon
    return icon


def transition_pixmap(name, size=18):
    """The same mark as a pixmap, for painting into a custom widget."""
    return transition(name, size).pixmap(QSize(size, size))


# --------------------------------------------------------------- UI icons ---

def _draw_ui(p, name, s, colour):
    """Every UI mark, on a nominal `s` x `s` box starting at (0, 0)."""
    pen = QPen(QColor(colour), 1.6, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    m = s * 0.18                       # margin
    a, b = m, s - m                    # inner box

    if name == "mix":
        # Two decks meeting: a falling line and a rising one.
        p.drawLine(QPointF(a, a + (b - a) * 0.15),
                   QPointF(b, b - (b - a) * 0.15))
        p.setPen(QPen(QColor(colour), 1.6, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(a, b - (b - a) * 0.15),
                   QPointF(b, a + (b - a) * 0.15))
    elif name == "library":
        for i in range(3):
            y = a + (b - a) * (0.1 + i * 0.4)
            p.drawLine(QPointF(a, y), QPointF(b, y))
    elif name == "correct":
        # Crosshair on a circle: placing something precisely.
        p.drawEllipse(QRectF(a + 1, a + 1, (b - a) - 2, (b - a) - 2))
        c = (a + b) / 2
        p.drawLine(QPointF(c, a - 1), QPointF(c, a + 2.5))
        p.drawLine(QPointF(c, b - 2.5), QPointF(c, b + 1))
        p.drawLine(QPointF(a - 1, c), QPointF(a + 2.5, c))
        p.drawLine(QPointF(b - 2.5, c), QPointF(b + 1, c))
    elif name == "stems":
        # Layers, pulled apart.
        for i in range(3):
            y = a + (b - a) * (0.12 + i * 0.38)
            path = QPainterPath(QPointF(a, y))
            path.lineTo(QPointF((a + b) / 2, y + (b - a) * 0.16))
            path.lineTo(QPointF(b, y))
            p.drawPath(path)
    elif name == "report":
        heights = (0.35, 0.75, 0.5)
        for i, hf in enumerate(heights):
            x = a + (b - a) * (0.16 + i * 0.34)
            p.drawLine(QPointF(x, b), QPointF(x, b - (b - a) * hf))
    elif name == "play":
        path = QPainterPath(QPointF(a + 1, a))
        path.lineTo(QPointF(b, (a + b) / 2))
        path.lineTo(QPointF(a + 1, b))
        path.closeSubpath()
        p.fillPath(path, QBrush(QColor(colour)))
    elif name == "pause":
        w = (b - a) * 0.28
        for x in (a + 1, b - w - 1):
            p.fillRect(QRectF(x, a, w, b - a), QBrush(QColor(colour)))
    elif name == "stop":
        p.fillRect(QRectF(a + 1, a + 1, (b - a) - 2, (b - a) - 2),
                   QBrush(QColor(colour)))
    elif name in ("back", "forward"):
        sign = -1 if name == "back" else 1
        c = (a + b) / 2
        for k in (0, 1):
            x = c + sign * ((b - a) * (0.05 + k * 0.3))
            # Apex on the side we are travelling towards. Subtracting here
            # instead of adding pointed "back" to the right and "forward" to
            # the left -- a mirror-image pair, both wrong, which is the sort
            # of error that survives review because the two look consistent.
            path = QPainterPath(QPointF(x, a + 1))
            path.lineTo(QPointF(x + sign * (b - a) * 0.28, c))
            path.lineTo(QPointF(x, b - 1))
            path.closeSubpath()
            p.fillPath(path, QBrush(QColor(colour)))
    elif name == "audition":
        # An ear on the join: a bracket around a centre line.
        c = (a + b) / 2
        p.drawLine(QPointF(c, a), QPointF(c, b))
        p.drawArc(QRectF(a, a + 1, (b - a) * 0.9, (b - a) - 2), 60 * 16,
                  240 * 16)
    elif name == "folder":
        path = QPainterPath(QPointF(a, b))
        path.lineTo(QPointF(a, a + (b - a) * 0.22))
        path.lineTo(QPointF(a + (b - a) * 0.42, a + (b - a) * 0.22))
        path.lineTo(QPointF(a + (b - a) * 0.55, a + (b - a) * 0.42))
        path.lineTo(QPointF(b, a + (b - a) * 0.42))
        path.lineTo(QPointF(b, b))
        path.closeSubpath()
        p.drawPath(path)
    elif name == "render":
        # A bolt: the one action that costs real time.
        path = QPainterPath(QPointF(a + (b - a) * 0.62, a))
        path.lineTo(QPointF(a + (b - a) * 0.24, (a + b) / 2 + 1))
        path.lineTo(QPointF(a + (b - a) * 0.52, (a + b) / 2 + 1))
        path.lineTo(QPointF(a + (b - a) * 0.38, b))
        path.lineTo(QPointF(a + (b - a) * 0.78, (a + b) / 2 - 1))
        path.lineTo(QPointF(a + (b - a) * 0.50, (a + b) / 2 - 1))
        path.closeSubpath()
        p.fillPath(path, QBrush(QColor(colour)))
    elif name == "split":
        # One line becoming four: separation.
        c = (a + b) / 2
        p.drawLine(QPointF(a, c), QPointF((a + b) / 2, c))
        for i in range(3):
            y = a + (b - a) * (0.15 + i * 0.35)
            p.drawLine(QPointF((a + b) / 2, c), QPointF(b, y))
    elif name == "downbeat":
        p.drawLine(QPointF((a + b) / 2, a), QPointF((a + b) / 2, b))
        for x in (a, b):
            c2 = QColor(colour)
            c2.setAlpha(110)
            p.setPen(QPen(c2, 1.3))
            p.drawLine(QPointF(x, a + (b - a) * 0.25),
                       QPointF(x, b - (b - a) * 0.25))
    elif name == "cue_in":
        p.drawLine(QPointF(a + 1, a), QPointF(a + 1, b))
        path = QPainterPath(QPointF(a + 3, a + (b - a) * 0.2))
        path.lineTo(QPointF(b, (a + b) / 2))
        path.lineTo(QPointF(a + 3, b - (b - a) * 0.2))
        path.closeSubpath()
        p.fillPath(path, QBrush(QColor(colour)))
    elif name == "cue_out":
        p.drawLine(QPointF(b - 1, a), QPointF(b - 1, b))
        path = QPainterPath(QPointF(b - 3, a + (b - a) * 0.2))
        path.lineTo(QPointF(a, (a + b) / 2))
        path.lineTo(QPointF(b - 3, b - (b - a) * 0.2))
        path.closeSubpath()
        p.fillPath(path, QBrush(QColor(colour)))
    elif name == "drop":
        c = (a + b) / 2
        p.drawLine(QPointF(c, a), QPointF(c, b - (b - a) * 0.3))
        path = QPainterPath(QPointF(c - (b - a) * 0.22, b - (b - a) * 0.34))
        path.lineTo(QPointF(c + (b - a) * 0.22, b - (b - a) * 0.34))
        path.lineTo(QPointF(c, b))
        path.closeSubpath()
        p.fillPath(path, QBrush(QColor(colour)))
    elif name == "reset":
        p.drawArc(QRectF(a, a, b - a, b - a), 40 * 16, 280 * 16)
        path = QPainterPath(QPointF(a + (b - a) * 0.62, a - 1))
        path.lineTo(QPointF(b + 1, a + (b - a) * 0.22))
        path.lineTo(QPointF(a + (b - a) * 0.66, a + (b - a) * 0.36))
        path.closeSubpath()
        p.fillPath(path, QBrush(QColor(colour)))
    elif name == "save":
        # Arrow down into a tray: the file leaving the app, not a floppy disk.
        c = s / 2.0
        p.drawLine(QPointF(c, a), QPointF(c, b - (b - a) * 0.34))
        path = QPainterPath(QPointF(c - (b - a) * 0.2, b - (b - a) * 0.46))
        path.lineTo(QPointF(c + (b - a) * 0.2, b - (b - a) * 0.46))
        path.lineTo(QPointF(c, b - (b - a) * 0.16))
        path.closeSubpath()
        p.fillPath(path, QBrush(QColor(colour)))
        p.drawLine(QPointF(a, b), QPointF(b, b))
    elif name == "theme":
        # Half-filled disc: the same shape reads for either direction, so the
        # button does not have to redraw itself to mean the opposite thing.
        c, r = s / 2.0, (b - a) / 2.0
        p.drawEllipse(QPointF(c, c), r, r)
        half = QPainterPath()
        half.moveTo(QPointF(c, c - r))
        half.arcTo(QRectF(c - r, c - r, r * 2, r * 2), 90, -180)
        half.closeSubpath()
        p.fillPath(half, QBrush(QColor(colour)))


def ui(name, size=16, colour=None):
    """A QIcon for a named interface mark."""
    colour = colour or T.TEXT_DIM
    key = ("ui", name, size, colour)
    if key in _cache:
        return _cache[key]
    pm = _pixmap(size)
    p = _painter(pm)
    _draw_ui(p, name, size, colour)
    p.end()
    icon = QIcon(pm)
    _cache[key] = icon
    return icon


def ui_pixmap(name, size=16, colour=None):
    return ui(name, size, colour).pixmap(QSize(size, size))


# --------------------------------------------------------------- app mark ---
# The application icon is the equal-power crossfade: two curves, one falling
# and one rising, crossing at the centre. That is not decoration -- it is the
# single rule the whole program is built on, and it is the same picture the
# Dissolve transition icon draws from its own automation.
#
# It is fixed in the brand blues rather than following the live palette. An app
# icon sits in a taskbar next to other applications' icons, where "the theme"
# is not this app's to decide, and an identity that changes colour is not an
# identity.
APP_BG_TOP = "#1565C0"
APP_BG_BOTTOM = "#0D47A1"
# The two curves are deliberately NOT the same value. Drawn at equal
# lightness they average into a plain X -- a shape a hundred apps use --
# and the whole idea is that one deck is leaving while the other arrives.
APP_OUT = "#64B5F6"     # outgoing: recedes
APP_IN = "#FFFFFF"      # incoming: arrives, and reads first


def app_icon(size=256):
    """The Auto DJ Mix mark, at any size, drawn from the crossfade curves."""
    key = ("app", size)
    if key in _cache:
        return _cache[key]

    pm = _pixmap(size, ratio=1)
    p = _painter(pm)
    s = float(size)

    # Rounded-square tile. The radius is a fraction of the size rather than a
    # constant, so the silhouette is the same shape at 16px and at 512.
    tile = QRectF(s * 0.045, s * 0.045, s * 0.91, s * 0.91)
    grad = QLinearGradient(tile.topLeft(), tile.bottomRight())
    grad.setColorAt(0.0, QColor(APP_BG_TOP))
    grad.setColorAt(1.0, QColor(APP_BG_BOTTOM))
    path = QPainterPath()
    path.addRoundedRect(tile, s * 0.22, s * 0.22)
    p.fillPath(path, QBrush(grad))

    # The two curves, from the real equal-power law rather than from eye.
    # Below about 20px the curves collapse into a smudge, so small sizes drop
    # to the crossing itself: a single X, which is the same idea at the only
    # fidelity that survives.
    box = QRectF(s * 0.235, s * 0.285, s * 0.53, s * 0.43)
    n = 64
    t = np.linspace(0.0, 1.0, n)
    fall = np.cos(t * np.pi / 2) ** 2
    rise = np.sin(t * np.pi / 2) ** 2
    width = max(1.35, s * 0.052)

    if size >= 20:
        for curve, colour in ((fall, APP_OUT), (rise, APP_IN)):
            pts = [QPointF(box.left() + x * box.width(),
                           box.bottom() - v * box.height())
                   for x, v in zip(t, curve)]
            _polyline(p, pts, colour, width)
    else:
        p.setPen(QPen(QColor(APP_OUT), width, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(box.topLeft(), box.bottomRight())
        p.setPen(QPen(QColor(APP_IN), width, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(box.bottomLeft(), box.topRight())

    # The crossing point, marked. Equal power's defining property is that the
    # two curves meet at 0.5 rather than at 0.707, and the dot is where the
    # handover happens.
    if size >= 32:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawEllipse(QPointF(box.center().x(), box.bottom() - box.height() / 2),
                      s * 0.045, s * 0.045)

    p.end()
    icon = QIcon(pm)
    _cache[key] = icon
    return icon


def app_pixmap(size=256):
    return app_icon(size).pixmap(QSize(size, size))


def app_qicon():
    """A multi-resolution QIcon, so Windows picks the right one per context.

    Handing Qt one large pixmap and letting it downscale produces a blurred
    16px taskbar icon; each size is drawn at its own scale instead, which is
    also why the small sizes can simplify the artwork.
    """
    icon = QIcon()
    for s in (16, 20, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(app_pixmap(s))
    return icon


UI_NAMES = ("mix", "library", "correct", "stems", "report", "play", "pause",
            "stop", "back", "forward", "audition", "folder", "render", "split",
            "downbeat", "cue_in", "cue_out", "drop", "reset", "theme",
            "save")


def clear_cache():
    """Drop every cached icon.

    Icons bake their colour in at draw time, so they survive a palette switch
    as the *old* palette's colour -- a set of dark-mode marks scattered across
    a light interface. The cache has to go when the mode does.
    """
    _cache.clear()


# ------------------------------------------------------- restyle protocol ---
# Clearing the cache is not enough on its own. A widget holds the QIcon it was
# given, so it keeps painting the old palette's mark no matter what the cache
# does. The fix is to remember how each icon was requested and ask again.
#
# The colour is stored as a *token name* rather than a colour, so it resolves
# against whichever palette is live at replay time. Storing "#E3F2FD" here
# would just re-apply the dark palette in light mode, one switch later.

def apply(widget, name, size=16, token=None, kind="ui"):
    """Set a widget's icon, and record how, so a palette switch can redo it."""
    widget._icon_spec = (kind, name, size, token)
    _apply_spec(widget, kind, name, size, token)


def _apply_spec(widget, kind, name, size, token):
    colour = getattr(T, token) if token else None
    if kind == "transition":
        icon = transition(name, size)
    else:
        icon = ui(name, size, colour)
    if hasattr(widget, "setIcon"):
        widget.setIcon(icon)
        if hasattr(widget, "setIconSize"):
            widget.setIconSize(QSize(size, size))
    else:
        widget.setPixmap(icon.pixmap(QSize(size, size)))


def restyle(root):
    """Re-draw every recorded icon and re-run every recorded style callback.

    Returns how many widgets were touched, which is the only cheap way to
    notice that a widget stopped registering itself -- a silent zero here is
    what a half-converted window looks like from the outside.
    """
    from PyQt6.QtWidgets import QWidget
    clear_cache()
    n = 0
    for w in [root] + root.findChildren(QWidget):
        spec = getattr(w, "_icon_spec", None)
        if spec is not None:
            _apply_spec(w, *spec)
            n += 1
        restyler = getattr(w, "_restyle", None)
        if restyler is not None:
            restyler()
            n += 1
    return n
