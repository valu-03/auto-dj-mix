"""The Automix panel: settings above, the play queue below.

The layout is borrowed from djay, and borrowed deliberately. A DJ application
has two things on screen at once -- what is playing, and what is coming -- and
putting the queue in a fixed right-hand column beside the decks is the
arrangement that makes both legible without either moving. A bento grid of
equal cells cannot express that, because it treats every panel as equally
important, and here they are not: the decks are where you look, the queue is
where you decide.

What is different from djay is what the rows do. In a realtime app the queue is
a list of what will be played; here the whole mix already exists, so each row
carries the join that follows it -- its transition, its length, and the reason
the planner chose it. Clicking that changes the mix, which is the thing this
application can do that a live one cannot.
"""

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMenu, QPushButton, QSizePolicy, QSpinBox,
                             QVBoxLayout, QWidget)

from .. import transitions
from . import icons
from . import theme as T
from .waveform import LABELS
from .widgets import ElidedLabel

# Presented in the order a person reaches for them, not alphabetically:
# automatic first, then the moves that suit most joins, then the specialised
# ones. "Automatic" is a real entry rather than a null -- it means the planner
# decides this join and keeps deciding it as the set changes around it.
TRANSITION_CHOICES = [
    ("auto", "Automatic"),
    ("dissolve", "Dissolve"),
    ("smooth_swap", "Smooth"),
    ("fade", "Fade"),
    ("filter_sweep", "Filter"),
    ("ladder_sweep", "Ladder filter"),
    ("bass_swap", "EQ"),
    ("eq_blend", "EQ blend"),
    ("echo_out", "Echo"),
    ("reverb_wash", "Reverb wash"),
    ("stem_blend", "Neural Mix"),
    ("hard_cut", "Cut"),
    ("cut_with_echo", "Cut + echo"),
    ("loop_roll", "Loop roll"),
    ("riser_cut", "Riser"),
    ("tremolo", "Tremolo"),
    ("double_drop", "Double drop"),
    ("vocal_slam_drop", "Vocal slam"),
    ("euro_rap_breakout", "Rap breakout"),
]

DURATION_MODES = [("auto", "Automatic"), ("bars", "Bars"),
                  ("seconds", "Seconds")]

TEMPO_MODES = [
    ("sync", "Sync"),
    ("blend", "Sync + tempo blend"),
    ("auto", "Automatic"),
    ("off", "Off"),
]

TEMPO_HELP = {
    "sync": "One tempo for the whole set, taken from the master deck.",
    "blend": "Each track keeps its own tempo; the tempo glides at each join.",
    "auto": "Holds a steady tempo, gliding only past a 5% difference.",
    "off": "No beatmatching at all. Every track plays at its native tempo.",
}


class Field(QWidget):
    """A labelled control, laid out so a column of them aligns."""

    def __init__(self, label, widget, hint="", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = QLabel(label.upper())
        cap.setObjectName("MetricLabel")
        lay.addWidget(cap)
        lay.addWidget(widget)
        self.hint = QLabel(hint)
        self.hint.setObjectName("CardHint")
        self.hint.setWordWrap(True)
        self.hint.setVisible(bool(hint))
        lay.addWidget(self.hint)
        self.widget = widget

    def set_hint(self, text):
        self.hint.setText(text)
        self.hint.setVisible(bool(text))


class QueueRow(QFrame):
    """One track in the set, and the join that follows it.

    Two lines of track, one line of join. The join belongs to the row above it
    rather than sitting between rows because a transition is a property of
    leaving *this* track -- and because a control that floats between two rows
    is never obviously attached to either.
    """

    transition_changed = pyqtSignal(int, str)     # join index, name
    bars_changed = pyqtSignal(int, int)           # join index, bars
    clicked = pyqtSignal(int)                     # position

    def __init__(self, pos, meta, seg, join_name=None, join_bars=None,
                 is_last=False, parent=None):
        super().__init__(parent)
        self.setObjectName("QueueRow")
        self.pos = pos
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(T.UNIT + 4, T.UNIT + 2, T.UNIT + 4, T.UNIT)
        lay.setSpacing(5)

        top = QHBoxLayout()
        top.setSpacing(T.UNIT)
        num = QLabel(f"{pos + 1}")
        num.setObjectName("Muted")
        num.setFixedWidth(16)
        top.addWidget(num)
        title = ElidedLabel(f"{meta['artist']} — {meta['title']}")
        title._restyle = lambda: title.setStyleSheet(
            f"color:{T.TEXT}; background:transparent; {T.font_css('label')}")
        title.setStyleSheet(f"color:{T.TEXT}; background:transparent; "
                            f"{T.font_css('label')}")
        top.addWidget(title, 1)
        lay.addLayout(top)

        sub = QLabel(f"{seg.get('bpm') or meta['bpm']:.1f} BPM · "
                     f"{meta['camelot']} · bars {seg['enter']}–{seg['exit']}")
        sub.setObjectName("Muted")
        sub.setContentsMargins(20, 0, 0, 0)
        lay.addWidget(sub)

        if not is_last and join_name is not None:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.setContentsMargins(20, 2, 2, 0)
            arrow = QLabel("↳")
            arrow.setObjectName("Muted")
            row.addWidget(arrow)

            self.combo = QComboBox()
            self.combo.setObjectName("Compact")
            self.combo.setIconSize(QSize(16, 16))
            for key, text in TRANSITION_CHOICES:
                # "Automatic" has no single shape to draw, so it gets the
                # generic mix mark rather than a made-up curve.
                icon = (icons.ui("mix", 16) if key == "auto"
                        else icons.transition(key, 16))
                self.combo.addItem(icon, text, key)
            idx = self.combo.findData(join_name)
            self.combo.setCurrentIndex(max(0, idx))
            self.combo.setMinimumHeight(28)
            need = transitions.REQUIRES.get(join_name)
            self.combo.setToolTip(need or "Transition used at this join")
            self.combo.currentIndexChanged.connect(
                lambda _: self.transition_changed.emit(
                    pos, self.combo.currentData()))
            row.addWidget(self.combo, 1)

            self.bars = QSpinBox()
            self.bars.setObjectName("Compact")
            self.bars.setRange(4, 32)
            self.bars.setSingleStep(4)
            self.bars.setValue(int(join_bars or 8))
            self.bars.setSuffix(" bars")
            self.bars.setMinimumHeight(28)
            self.bars.setFixedWidth(92)
            self.bars.setToolTip("Length of this transition")
            self.bars.valueChanged.connect(
                lambda v: self.bars_changed.emit(pos, v))
            row.addWidget(self.bars)
            lay.addLayout(row)

        # Lay out now, so `sizeHint` is the height this row actually needs.
        # Asking a freshly constructed widget for its size hint returns the
        # hint of an empty layout, and the list then allocates a row about
        # eight pixels short -- which clips the bottom of the two controls
        # that are the entire point of the row.
        self.adjustSize()

    def mousePressEvent(self, e):
        self.clicked.emit(self.pos)
        super().mousePressEvent(e)

    def set_active(self, active):
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class QueuePanel(QFrame):
    """Automix settings and the play queue, in one fixed column."""

    settings_changed = pyqtSignal()
    transition_changed = pyqtSignal(int, str)
    bars_changed = pyqtSignal(int, int)
    row_clicked = pyqtSignal(int)
    render_requested = pyqtSignal()
    folder_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("QueuePanel")
        self.setFixedWidth(392)
        self._rows = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(T.PAD_CARD, T.PAD_CARD, T.PAD_CARD,
                                 T.PAD_CARD)
        outer.setSpacing(T.UNIT * 2)

        head = QHBoxLayout()
        title = QLabel("Automix")
        title.setObjectName("CardTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.source = QPushButton("  Choose folder…")
        self.source.setObjectName("Ghost")
        icons.apply(self.source, "folder", 14)
        self.source.setIconSize(QSize(14, 14))
        self.source.setCursor(Qt.CursorShape.PointingHandCursor)
        self.source.clicked.connect(self.folder_requested)
        head.addWidget(self.source)
        outer.addLayout(head)

        grid = QGridLayout()
        grid.setHorizontalSpacing(T.UNIT + 2)
        grid.setVerticalSpacing(T.UNIT + 4)

        # The set-wide transition, with the same sixteen moves the per-join
        # rows offer. "Automatic" hands the choice back to the planner, which
        # picks per join; anything else applies to every join at once, and an
        # individual row can still override it afterwards.
        #
        # This control used to be the three-entry planner policy below, under
        # the name "Transition style" -- djay's name for its transition-type
        # menu. Three options where sixteen were expected reads as fourteen
        # missing ones rather than as a different control, and asking for a
        # transition across the whole set is a reasonable thing to want.
        self.transition = QComboBox()
        for key, text in TRANSITION_CHOICES:
            icon = (icons.ui("mix", 16) if key == "auto"
                    else icons.transition(key, 16))
            self.transition.addItem(icon, text, key)
        self.transition.setIconSize(QSize(16, 16))
        self.transition_field = Field(
            "Transition", self.transition,
            "the planner chooses each join · override any row below")
        grid.addWidget(self.transition_field, 0, 0, 1, 2)

        # Only meaningful while Transition is Automatic: it biases *which*
        # move the planner reaches for. Pin a transition and there is nothing
        # left for it to decide, so it is disabled rather than left looking
        # like it still does something.
        self.style_box = QComboBox()
        self.style_box.addItem("Smooth — overlapping", "smooth")
        self.style_box.addItem("Cut — megamix", "cut")
        self.style_box.addItem("Blend — long EQ", "blend")
        self.style_field = Field("Mixing style", self.style_box,
                                 "how the planner picks, and how hard it "
                                 "weights key")
        grid.addWidget(self.style_field, 1, 0, 1, 2)

        self.duration_mode = QComboBox()
        for key, text in DURATION_MODES:
            self.duration_mode.addItem(text, key)
        self.duration_value = QSpinBox()
        self.duration_value.setRange(1, 64)
        self.duration_value.setValue(8)
        self.duration_field = Field("Duration", self.duration_mode)
        self.value_field = Field("Length", self.duration_value)
        grid.addWidget(self.duration_field, 2, 0)
        grid.addWidget(self.value_field, 2, 1)

        self.tempo_mode = QComboBox()
        for key, text in TEMPO_MODES:
            self.tempo_mode.addItem(text, key)
        self.tempo_field = Field("Tempo", self.tempo_mode, TEMPO_HELP["sync"])
        grid.addWidget(self.tempo_field, 3, 0, 1, 2)

        self.arc = QComboBox()
        from .. import planner
        for name in planner.ARCS:
            self.arc.addItem(name.replace("_", " ").title(), name)
        grid.addWidget(Field("Energy arc", self.arc), 4, 0, 1, 2)
        outer.addLayout(grid)

        self.list = QListWidget()
        self.list.setObjectName("Queue")
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setFrameShape(QFrame.Shape.NoFrame)
        self.list.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        outer.addWidget(self.list, 1)

        self.empty = QLabel("No set yet.\nAnalyse a folder, then render.")
        self.empty.setObjectName("Muted")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.empty)

        for w in (self.transition, self.style_box, self.duration_mode,
                  self.tempo_mode, self.arc):
            w.currentIndexChanged.connect(self._changed)
        self.duration_value.valueChanged.connect(self._changed)
        self._sync_duration_fields()
        self._sync_transition_fields()

    # ------------------------------------------------------------ state ----
    def _sync_transition_fields(self):
        """Mixing style only means something while Transition is Automatic."""
        auto = self.transition.currentData() == "auto"
        self.style_box.setEnabled(auto)
        self.style_field.set_hint(
            "how the planner picks, and how hard it weights key" if auto
            else "the planner is not choosing — Transition is pinned")
        self.transition_field.set_hint(
            "the planner chooses each join · override any row below" if auto
            else "used at every join · override any row below")

    def _changed(self):
        self._sync_transition_fields()
        self._sync_duration_fields()
        self.tempo_field.set_hint(TEMPO_HELP.get(self.tempo_mode.currentData(),
                                                 ""))
        self.settings_changed.emit()

    def _sync_duration_fields(self):
        mode = self.duration_mode.currentData()
        self.value_field.setVisible(mode != "auto")
        if mode == "seconds":
            self.duration_value.setSuffix(" s")
            self.duration_value.setRange(2, 120)
        else:
            self.duration_value.setSuffix(" bars")
            self.duration_value.setRange(4, 32)
            self.duration_value.setSingleStep(4)

    def settings(self):
        return {
            "transition": self.transition.currentData(),
            "style": self.style_box.currentData(),
            "duration_mode": self.duration_mode.currentData(),
            "duration_value": self.duration_value.value(),
            "tempo_mode": self.tempo_mode.currentData(),
            "arc": self.arc.currentData(),
        }

    def set_plan(self, metas, order, segs, joins, bars):
        self.list.clear()
        self._rows = []
        self.empty.setVisible(not segs)
        if not segs:
            return
        last = len(segs) - 1
        for i, seg in enumerate(segs):
            # Accept a built Transition or a bare name, matching
            # what TimelineView already tolerates.
            j = joins[i] if i < len(joins) else None
            name = getattr(j, "name", j)
            length = bars[i] if i < len(bars) else None
            row = QueueRow(i, metas[order[i]], seg, name, length, i == last)
            row.transition_changed.connect(self.transition_changed)
            row.bars_changed.connect(self.bars_changed)
            row.clicked.connect(self.row_clicked)
            item = QListWidgetItem(self.list)
            # +8 for the row's own top and bottom margin in the stylesheet,
            # which the layout does not know about.
            item.setSizeHint(QSize(0, row.sizeHint().height() + 8))
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            self._rows.append(row)

    def set_active(self, pos):
        for i, row in enumerate(self._rows):
            row.set_active(i == pos)
        if 0 <= pos < len(self._rows):
            self.list.scrollToItem(self.list.item(pos))
