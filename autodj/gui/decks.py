"""The deck area: transport, the mix waveform, and the two tracks at a join.

Left-hand side of the djay layout. Three stacked things, in the order you need
them:

transport   because until this existed the application could render a mix and
            not play it, and every judgement about whether the mix was any good
            had to happen in another program.

mix         the whole rendered set as one waveform, with the playhead running
            across it. This is the object the app makes, so it gets the space.

decks       the outgoing and incoming tracks around the selected join, drawn
            against each other. Two tracks stacked with their beatgrids
            visible is how you see whether a transition is going to work, and
            it is the one view that makes the phrase alignment obvious rather
            than a claim in a log file.
"""

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QSlider, QVBoxLayout, QWidget)

from . import icons
from . import theme as T
from .waveform import LABELS, WaveformView
from .widgets import ElidedLabel


def _time(seconds):
    seconds = max(0.0, float(seconds))
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class Transport(QFrame):
    """Play, seek, and the two buttons that only make sense here.

    "Audition" jumps to fifteen seconds before the selected join, which is the
    single most-used action in the app: almost every question you have about a
    mix is a question about one transition, and the alternative is scrubbing
    for it by hand every time.
    """

    play_pause = pyqtSignal()
    stopped = pyqtSignal()
    seek_delta = pyqtSignal(float)
    audition = pyqtSignal()
    volume = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Transport")
        self.setFixedHeight(60)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(T.UNIT * 2, T.UNIT, T.UNIT * 2, T.UNIT)
        lay.setSpacing(T.UNIT)

        # Drawn marks, not text glyphs. The Unicode transport symbols all
        # exist in the font and all render as narrow grey ticks at body size,
        # which is a worse failure than a missing glyph because it looks
        # deliberate.
        self.play = QPushButton()
        self.play.setObjectName("Primary")
        self.play.setFixedSize(46, 36)
        icons.apply(self.play, "play", 15, "ON_ACCENT")
        self.play.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play.setToolTip("Play / pause  (Space)")
        self.play.clicked.connect(self.play_pause)
        lay.addWidget(self.play)

        for glyph, text, tip, delta in (("back", "15", "Back 15 s", -15.0),
                                        ("forward", "15", "Forward 15 s",
                                         15.0)):
            b = QPushButton(f" {text}")
            b.setObjectName("Ghost")
            # Wide enough for the icon AND the number. At 56 the two together
            # overflowed and Qt cropped the label to "1", which reads as a
            # broken glyph rather than a truncated word.
            b.setFixedSize(70, 36)
            icons.apply(b, glyph, 13)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(tip)
            b.clicked.connect(lambda _, d=delta: self.seek_delta.emit(d))
            lay.addWidget(b)

        self.stop = QPushButton()
        self.stop.setObjectName("Ghost")
        self.stop.setFixedSize(40, 36)
        icons.apply(self.stop, "stop", 12)
        self.stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop.setToolTip("Stop")
        self.stop.clicked.connect(self.stopped)
        lay.addWidget(self.stop)

        self.clock = QLabel("0:00 / 0:00")
        def _clock_style():
            self.clock.setStyleSheet(f"color:{T.TEXT}; background:transparent; "
                                     f"font-family:{T.MONO_STACK}; "
                                     f"{T.font_css('label')}")
        _clock_style()
        self.clock._restyle = _clock_style
        self.clock.setFixedWidth(112)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.clock)

        self.audition_btn = QPushButton("  Audition join")
        self.audition_btn.setObjectName("Ghost")
        icons.apply(self.audition_btn, "audition", 14)
        self.audition_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.audition_btn.setToolTip(
            "Jump to 15 s before the selected transition  (A)")
        self.audition_btn.clicked.connect(self.audition)
        lay.addWidget(self.audition_btn)

        lay.addStretch(1)

        self.here = QLabel("")
        self.here.setObjectName("Muted")
        lay.addWidget(self.here)
        # The row spacing is right between related controls and too tight
        # between two unrelated labels: the now-playing title butted straight
        # up against "VOL" and the two read as one string.
        lay.addSpacing(T.UNIT * 2)

        vol = QLabel("VOL")
        vol.setObjectName("MetricLabel")
        lay.addWidget(vol)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFixedWidth(104)
        self.slider.setRange(0, 100)
        self.slider.setValue(90)
        self.slider.valueChanged.connect(
            lambda v: self.volume.emit(v / 100.0))
        lay.addWidget(self.slider)

    def set_playing(self, playing):
        icons.apply(self.play, "pause" if playing else "play", 15, "ON_ACCENT")

    def set_clock(self, at, total):
        self.clock.setText(f"{_time(at)} / {_time(total)}")

    def set_here(self, text):
        self.here.setText(text)

    def set_enabled(self, on):
        for w in (self.play, self.stop, self.audition_btn, self.slider):
            w.setEnabled(on)


class DeckPair(QFrame):
    """Outgoing and incoming tracks at one join, stacked and aligned.

    Deliberately two separate waveforms rather than one overlay. Overlaid
    waveforms look impressive and tell you nothing -- the sum of two signals
    does not show you either of them. Stacked, with both beatgrids drawn, you
    can see the phrase lines agree, which is the actual question.
    """

    scrubbed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(T.PAD_CARD, T.UNIT * 2, T.PAD_CARD, T.UNIT * 2)
        lay.setSpacing(T.UNIT)

        head = QHBoxLayout()
        head.setSpacing(T.UNIT)
        self.mark = QLabel()
        self.mark.setFixedSize(20, 20)
        head.addWidget(self.mark)
        self.title = QLabel("Join")
        self.title.setObjectName("CardTitle")
        head.addWidget(self.title)
        self.detail = ElidedLabel("select a transition in the queue")
        self.detail.setObjectName("CardHint")
        # Elided rather than wrapped: this is a one-line note beside a
        # title, and letting it wrap pushes the waveforms down every time the
        # reason happens to be a long one.
        self.detail.setAlignment(Qt.AlignmentFlag.AlignRight |
                                 Qt.AlignmentFlag.AlignVCenter)
        # No stretch spacer beside it: with an Ignored size policy the label
        # reports no width, so a stretch(1) next to a widget(1) split the row
        # in half and the text was elided against twice the space it actually
        # had -- which is why it ran off the card edge with no ellipsis.
        head.addWidget(self.detail, 1)
        lay.addLayout(head)

        self.out = WaveformView()
        self.out.accent = T.SERIES[3]
        self.out.setMinimumHeight(96)
        self.inn = WaveformView()
        self.inn.accent = T.SERIES[1]
        self.inn.setMinimumHeight(96)
        lay.addWidget(self.out, 1)
        lay.addWidget(self.inn, 1)

    def set_join(self, name, bars, out_meta, in_meta, reason=""):
        self.title.setText(f"{LABELS.get(name, name)} · {bars} bars")
        icons.apply(self.mark, name, 20, kind="transition")
        self.detail.setText(reason)
        self.out.label = (f"OUT  {out_meta['artist']} — {out_meta['title']}"
                          if out_meta else "")
        self.inn.label = (f"IN   {in_meta['artist']} — {in_meta['title']}"
                          if in_meta else "")
        self.out.update()
        self.inn.update()

    def clear(self):
        self.out.clear()
        self.inn.clear()
        self.title.setText("Join")
        self.mark.clear()
        self.detail.setText("select a transition in the queue")
