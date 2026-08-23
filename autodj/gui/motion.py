"""Motion: the durations and easing curves from the design system, as code.

Qt's stylesheet engine has no `transition` property. A design spec that says
"150ms cubic-bezier(0.2, 0, 0, 1)" therefore cannot be handed to the
stylesheet -- it has to be run by QPropertyAnimation, and the curve has to be
rebuilt as a QEasingCurve. That is the whole job of this module: keep the
motion vocabulary in one place, expressed in the same cubic-bezier terms the
rest of the design world uses, so it stays portable and comparable.

The named curves map to CSS control points exactly, via `BezierSpline`, which
takes the same two control points a CSS `cubic-bezier()` does. The alternative
-- picking the closest of Qt's ~40 preset enums -- means the documented curve
and the shipped curve quietly differ, which is the same class of drift the
colour tokens exist to prevent.

Restraint is the actual design position here. Nothing in this file animates
colour on hover: the stylesheet's :hover already repaints instantly, and an
animated hover on a dense tool like this one reads as lag. Movement is spent
only where it carries meaning -- something arriving, something leaving,
something changing size -- and never on decoration.
"""

from PyQt6.QtCore import (QAbstractAnimation, QEasingCurve, QPointF,
                          QParallelAnimationGroup, QPropertyAnimation, Qt)
from PyQt6.QtWidgets import QGraphicsOpacityEffect

from . import theme as T


def curve(points=None):
    """A QEasingCurve from CSS cubic-bezier control points.

    QEasingCurve.Type.BezierSpline expects the curve as cubic segments ending
    at (1, 1), which is exactly the CSS form: two control points plus the
    implicit endpoints.
    """
    p = points or T.EASE_STANDARD
    c = QEasingCurve(QEasingCurve.Type.BezierSpline)
    c.addCubicBezierSegment(QPointF(p[0], p[1]), QPointF(p[2], p[3]),
                            QPointF(1.0, 1.0))
    return c


def animate(target, prop, end, ms=None, points=None, start=None,
            on_done=None):
    """Animate one Qt property, and keep the animation alive.

    The animation is parented to the target. Without that it is garbage
    collected the moment this function returns and nothing visibly moves --
    the single most common way Qt animations "silently do nothing".
    """
    anim = QPropertyAnimation(target, prop.encode("ascii"), target)
    anim.setDuration(T.DUR_BASE if ms is None else ms)
    anim.setEasingCurve(curve(points))
    if start is not None:
        anim.setStartValue(start)
    anim.setEndValue(end)
    if on_done is not None:
        anim.finished.connect(on_done)
    anim.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return anim


def fade_in(widget, ms=None, points=None):
    """Fade a widget up from transparent.

    A QGraphicsOpacityEffect is applied rather than a window opacity because
    this has to work on child widgets, and `setWindowOpacity` only affects
    top-level windows -- on a child it is accepted and ignored, which looks
    exactly like a broken animation.
    """
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    widget.show()
    # The effect is dropped at the end: leaving one installed forces every
    # later repaint of that widget through an offscreen buffer, and on the
    # waveform views that is the difference between a smooth playhead and a
    # stuttering one.
    return animate(eff, "opacity", 1.0, ms or T.DUR_BASE, points or T.EASE_OUT,
                   start=0.0, on_done=lambda: widget.setGraphicsEffect(None))


def fade_out(widget, ms=None, points=None, then=None):
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)

    def done():
        widget.setGraphicsEffect(None)
        widget.hide()
        if then is not None:
            then()

    return animate(eff, "opacity", 0.0, ms or T.DUR_FAST,
                   points or T.EASE_IN, start=1.0, on_done=done)


def slide_width(widget, to, ms=None, points=None, on_done=None):
    """Animate a fixed width -- the sidebar collapse, the queue panel."""
    return animate(widget, "minimumWidth", to, ms or T.DUR_SLOW,
                   points or T.EASE_STANDARD, start=widget.width(),
                   on_done=lambda: (widget.setFixedWidth(to),
                                    on_done and on_done()))


def slide_in(widget, from_offset=(0, 12), ms=None, points=None):
    """Arrive: fade up while travelling a short distance to rest.

    The distance is deliberately small. A toast that flies in from off-screen
    is a slideshow effect; twelve pixels reads as the thing settling into
    place, and settles inside one frame budget at any refresh rate.
    """
    end = widget.pos()
    start = end + QPointF(*from_offset).toPoint()
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    widget.show()
    group = QParallelAnimationGroup(widget)

    pos = QPropertyAnimation(widget, b"pos", widget)
    pos.setDuration(ms or T.DUR_BASE)
    pos.setEasingCurve(curve(points or T.EASE_OUT))
    pos.setStartValue(start)
    pos.setEndValue(end)

    fade = QPropertyAnimation(eff, b"opacity", widget)
    fade.setDuration(ms or T.DUR_BASE)
    fade.setEasingCurve(curve(points or T.EASE_OUT))
    fade.setStartValue(0.0)
    fade.setEndValue(1.0)

    group.addAnimation(pos)
    group.addAnimation(fade)
    group.finished.connect(lambda: widget.setGraphicsEffect(None))
    group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return group


def pulse(widget, ms=None):
    """A single confirmation beat -- used when a render finishes.

    The only place the spring curve is allowed. Overshoot on anything the user
    is about to click makes the target move under the cursor.
    """
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(ms or T.DUR_SLOW)
    anim.setEasingCurve(curve(T.EASE_SPRING))
    anim.setKeyValueAt(0.0, 1.0)
    anim.setKeyValueAt(0.4, 0.45)
    anim.setKeyValueAt(1.0, 1.0)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return anim
