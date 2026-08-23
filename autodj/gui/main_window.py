"""The application window: sidebar, decks on the left, Automix queue on the right.

The layout follows djay's, and the reason is structural rather than cosmetic. A
DJ application has two things on screen at once -- what is playing and what is
coming -- and a fixed right-hand queue beside a deck area is the arrangement
that keeps both readable without either moving. An equal-weight bento grid
cannot say that, because it treats every panel as equally important, and here
the decks are where you look while the queue is where you decide.

The sidebar switches views for real. It used to call `setFocus()` on a card in a
grid where everything was already visible, which is a no-op wearing the costume
of navigation.
"""

import numpy as np
from PyQt6.QtCore import (QEasingCurve, QPropertyAnimation, QSize, Qt, QTimer,
                          pyqtSignal)
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                             QDoubleSpinBox, QLabel, QLineEdit, QMainWindow,
                             QProgressBar,
                             QPushButton, QScrollArea, QSizePolicy, QSpinBox,
                             QStackedLayout, QStackedWidget, QTableView,
                             QVBoxLayout, QWidget)

from .. import audio as audio_mod
from .. import corrections, explain, planner
from . import icons
from . import theme as T
from . import workers
from .decks import DeckPair, Transport
from .library_view import COLUMNS, LibraryFilter, LibraryModel
from .player import AVAILABLE as AUDIO_AVAILABLE
from .player import Player
from .queue_view import QueuePanel
from .waveform import LABELS, TimelineView, WaveformView
from .widgets import (BarChart, Badge, Card, ElidedLabel, EmptyState, Skeleton,
                      Sparkline,
                      StatTile, Toast)

NARROW = 1280
# The status slot. It expands to whatever the row has spare, down to a floor
# that still shows a useful fragment of a message before eliding.
STATUS_MIN = 150
PROGRESS_W = 170
PROGRESS_NARROW = 120
# (label, icon name). Drawn marks rather than Unicode symbols: at body size
# the glyphs all resolved to near-identical small squares, so the sidebar read
# as five grey boxes with words beside them.
NAV = [("Mix", "mix"), ("Library", "library"), ("Correct", "correct"),
       ("Stems", "stems"), ("Report", "report")]


class Sidebar(QFrame):
    navigated = pyqtSignal(int)
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(T.SIDEBAR_W)
        self.collapsed = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(int(T.UNIT * 1.5), int(T.UNIT * 2.5),
                               int(T.UNIT * 1.5), T.UNIT * 2)
        lay.setSpacing(int(T.UNIT // 2))

        brand = QHBoxLayout()
        brand.setSpacing(T.UNIT + 2)
        # The app mark itself, not a glyph. "◐" was standing in for it, and
        # like the rest of the Unicode set it rendered as an anonymous shape
        # that meant nothing and matched nothing else in the interface.
        self.mark = QLabel()
        self.mark.setFixedSize(26, 26)
        self.mark.setPixmap(icons.app_pixmap(26))
        self.wordmark = QLabel("Auto DJ")
        self.wordmark.setStyleSheet(f"color:{T.TEXT}; background:transparent; "
                                    f"{T.font_css('title')}")
        self.wordmark._restyle = lambda: self.wordmark.setStyleSheet(
            f"color:{T.TEXT}; background:transparent; {T.font_css('title')}")
        brand.addWidget(self.mark)
        brand.addWidget(self.wordmark)
        brand.addStretch(1)
        lay.addLayout(brand)
        lay.addSpacing(T.UNIT * 3)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        for i, (name, glyph) in enumerate(NAV):
            b = QPushButton(f"   {name}")
            b.setObjectName("NavItem")
            b.setCheckable(True)
            icons.apply(b, glyph, 17)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, k=i: self.navigated.emit(k))
            self.group.addButton(b, i)
            self.buttons.append((b, name, glyph))
            lay.addWidget(b)
        self.buttons[0][0].setChecked(True)

        lay.addStretch(1)
        self.collapse_btn = QPushButton("  ‹   Collapse")
        self.collapse_btn.setObjectName("NavItem")
        self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_btn.clicked.connect(self.toggle)
        lay.addWidget(self.collapse_btn)

    def select(self, index):
        self.buttons[index][0].setChecked(True)

    def toggle(self):
        self.collapsed = not self.collapsed
        target = T.SIDEBAR_MIN if self.collapsed else T.SIDEBAR_W
        self._anim = QPropertyAnimation(self, b"minimumWidth", self)
        self._anim.setDuration(T.ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setStartValue(self.width())
        self._anim.setEndValue(target)
        self._anim.valueChanged.connect(lambda v: self.setFixedWidth(int(v)))
        self._anim.start()

        self.wordmark.setVisible(not self.collapsed)
        for b, name, glyph in self.buttons:
            b.setText("" if self.collapsed else f"   {name}")
            b.setToolTip(name if self.collapsed else "")
        self.collapse_btn.setText("  ›" if self.collapsed else "  ‹   Collapse")
        self.toggled.emit(self.collapsed)

    def set_collapsed(self, value):
        if value != self.collapsed:
            self.toggle()


class TopBar(QFrame):
    search_changed = pyqtSignal(str)
    render_clicked = pyqtSignal()
    theme_toggled = pyqtSignal()
    save_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(T.TOPBAR_H)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(T.PAD_VIEW, 0, T.PAD_VIEW, 0)
        lay.setSpacing(T.UNIT * 2)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search artist, title or key…")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(240)
        self.search.textChanged.connect(self.search_changed)
        lay.addWidget(self.search)

        # The status region is the one flexible thing in this row, and
        # everything else is fixed. That is what makes the bar fit at all.
        #
        # It did not fit before. The contents needed about 1200px of a 1080px
        # row at full width, so items simply overlapped -- the progress bar sat
        # across the theme button and the render button, and at 1280 the search
        # field ran 153px into the status text. A status label sized to its own
        # text made it worse, because "Analysing 1/4 · <title>" is a different
        # width for every track, so the whole row re-laid out on every progress
        # tick while a job ran.
        #
        # Now: fixed widths on the right, one expanding slot in the middle, and
        # a label that elides rather than overflowing whatever it is given.
        progress_box = QWidget()
        progress_box.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Preferred)
        progress_box.setMinimumWidth(STATUS_MIN)
        pl = QHBoxLayout(progress_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(T.UNIT)

        self.status = ElidedLabel("Ready")
        self.status.setObjectName("Muted")
        # ElidedLabel defaults to an Ignored horizontal policy so it can shrink
        # below its text. In a box layout that also means the layout allocates
        # it zero width when positioning the next item while still honouring
        # any fixed width when it sets the geometry -- the two disagree, and
        # the progress bar lands on top of the text. Expanding is both correct
        # here and self-consistent.
        self.status.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Preferred)
        pl.addWidget(self.status, 1)

        self.bar = QProgressBar()
        self.bar.setFixedWidth(PROGRESS_W)
        self.bar.setTextVisible(False)
        self.bar.hide()
        pl.addWidget(self.bar)
        lay.addWidget(progress_box, 1)

        self.dirty = Badge("unsaved changes", "warn")
        self.dirty.hide()
        lay.addWidget(self.dirty)

        self.save_btn = QPushButton()
        self.save_btn.setObjectName("IconBtn")
        self.save_btn.setFixedSize(34, 34)
        icons.apply(self.save_btn, "save", 15)
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setToolTip("Export the mix to a file  (Ctrl+S)")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_clicked)
        lay.addWidget(self.save_btn)

        self.theme_btn = QPushButton()
        self.theme_btn.setObjectName("IconBtn")
        self.theme_btn.setFixedSize(34, 34)
        icons.apply(self.theme_btn, "theme", 15)
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.setToolTip("Switch between dark and light  (Ctrl+D)")
        self.theme_btn.clicked.connect(self.theme_toggled)
        lay.addWidget(self.theme_btn)

        self.render = QPushButton("Render mix")
        self.render.setObjectName("Primary")
        self.render.setCursor(Qt.CursorShape.PointingHandCursor)
        icons.apply(self.render, "render", 15, "ON_ACCENT")
        self.render.setToolTip("Render the set  (Ctrl+R)")
        self.render.clicked.connect(self.render_clicked)
        lay.addWidget(self.render)

        self.gpu = Badge("GPU —", "neutral")
        lay.addWidget(self.gpu)

    def fit(self, width):
        """Drop the least important controls when the row cannot hold them.

        Below about 1300 the top bar genuinely does not fit: search, status,
        progress, theme, render and the GPU badge need roughly 800px of a
        640px row. Something has to go, and choosing nothing means they
        overlap -- which is what happened. Order of sacrifice is by how much
        each one is worth mid-task: the GPU badge is a static fact, the search
        box can be narrower and still usable, and the render button and the
        progress it reports never go.
        """
        self.gpu.setVisible(width >= 1500)
        # Ctrl+S still saves when the button is not on screen.
        self.save_btn.setVisible(width >= 1360)
        self.search.setFixedWidth(240 if width >= 1400 else 180)
        # The bar itself is the last thing to give ground, and it only gives a
        # little: a progress bar reads fine at 120px, and the alternative at
        # 1280 was six pixels of it under the theme button.
        self.bar.setFixedWidth(PROGRESS_W if width >= 1400 else PROGRESS_NARROW)

    def set_busy(self, text, pct=None):
        self.status.setText(text)
        if pct is None:
            self.bar.hide()
        else:
            self.bar.show()
            self.bar.setValue(int(pct))


class CorrectPanel(QFrame):
    """Fixing what the analysis got wrong, on the track you are looking at.

    Beat detection is the one failure that ruins a whole mix, and until this
    existed there was no recourse but re-running the same algorithm. Everything
    here writes a correction and re-analyses that single file, so the energy
    curve, the segments and the cue points are all rebuilt against the grid you
    chose rather than patched on top of the one you rejected.
    """

    mode_changed = pyqtSignal(object)
    bpm_changed = pyqtSignal(float)
    reset = pyqtSignal()
    nudged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(T.PAD_CARD, T.UNIT * 2, T.PAD_CARD, T.UNIT * 2)
        lay.setSpacing(T.UNIT + 2)

        title = QLabel("Correct the analysis")
        title.setObjectName("CardTitle")
        lay.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(T.UNIT)
        row.addWidget(QLabel("BPM"))
        # A double spin box, not an integer one scaled by 100. Showing 13362
        # and expecting the user to read it as 133.62 is the kind of thing that
        # is obvious to whoever wrote it and to nobody else.
        self.bpm = QDoubleSpinBox()
        self.bpm.setDecimals(2)
        self.bpm.setRange(40.0, 200.0)
        self.bpm.setSingleStep(0.01)
        self.bpm.setFixedWidth(116)
        self.bpm.setToolTip("Tempo, to a hundredth of a BPM")
        self.bpm.valueChanged.connect(self.bpm_changed)
        row.addWidget(self.bpm)
        for text, factor in (("÷2", 0.5), ("×2", 2.0)):
            b = QPushButton(text)
            b.setObjectName("Ghost")
            b.setFixedWidth(52)
            b.setToolTip("Halve / double the detected tempo — the most common "
                         "beat-tracking failure by far")
            b.clicked.connect(lambda _, d=factor: self.bpm_changed.emit(
                self.bpm.value() * d))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        nudge = QHBoxLayout()
        nudge.setSpacing(T.UNIT)
        nudge.addWidget(QLabel("Nudge grid"))
        for text, ms in (("-10 ms", -0.010), ("-1 ms", -0.001),
                         ("+1 ms", 0.001), ("+10 ms", 0.010)):
            b = QPushButton(text)
            b.setObjectName("Ghost")
            b.setMinimumWidth(78)
            b.setToolTip("Shift the whole beatgrid without changing the tempo")
            b.clicked.connect(lambda _, d=ms: self.nudged.emit(d))
            nudge.addWidget(b)
        nudge.addStretch(1)
        lay.addLayout(nudge)

        self.buttons = {}
        modes = QHBoxLayout()
        modes.setSpacing(T.UNIT)
        for key, text, glyph in (("downbeat", "Set downbeat", "downbeat"),
                                 ("first_full_bar", "Set IN", "cue_in"),
                                 ("outro_start_bar", "Set OUT", "cue_out"),
                                 ("drop", "Add drop", "drop")):
            b = QPushButton(f" {text}")
            b.setObjectName("Ghost")
            icons.apply(b, glyph, 14)
            b.setIconSize(QSize(14, 14))
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, k=key: self._pick(k))
            self.buttons[key] = b
            modes.addWidget(b)
        modes.addStretch(1)
        lay.addLayout(modes)

        foot = QHBoxLayout()
        self.state = QLabel("no manual corrections")
        self.state.setObjectName("CardHint")
        foot.addWidget(self.state, 1)
        clear = QPushButton(" Reset to analysis")
        clear.setObjectName("Ghost")
        icons.apply(clear, "reset", 14)
        clear.setIconSize(QSize(14, 14))
        clear.clicked.connect(self.reset)
        foot.addWidget(clear)
        lay.addLayout(foot)

    def _pick(self, key):
        active = self.buttons[key].isChecked()
        for k, b in self.buttons.items():
            b.setChecked(active and k == key)
        self.mode_changed.emit(key if active else None)

    def clear_mode(self):
        for b in self.buttons.values():
            b.setChecked(False)
        self.mode_changed.emit(None)

    def show_meta(self, meta):
        self.bpm.blockSignals(True)
        self.bpm.setValue(float(meta["bpm"]))
        self.bpm.blockSignals(False)
        note = corrections.summary(meta["file"])
        self.state.setText(note or "no manual corrections")


class ReasonPanel(QScrollArea):
    """Why the mix is the way it is, for the selected join."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # A scroll area is three widgets deep -- the area, its viewport and the
        # widget inside -- and each of them inherits the app ground from the
        # root QWidget rule. On a card that is a large dark rectangle behind
        # the text, which is the same fault as the label backgrounds one level
        # further down the tree.
        self.setObjectName("Reasons")
        self.viewport().setAutoFillBackground(False)
        self.body = QWidget()
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(0, 0, T.UNIT, 0)
        self.lay.setSpacing(T.UNIT + 2)
        self.lay.addStretch(1)
        self.setWidget(self.body)

    def show_rows(self, rows):
        while self.lay.count() > 1:
            item = self.lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for label, text in rows:
            block = QWidget()
            # One more level of the same fault: these row containers are
            # children of the scroll body, so a selector naming two levels of
            # child does not reach them and each one paints its own dark bar.
            block.setStyleSheet("background: transparent;")
            v = QVBoxLayout(block)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(3)
            cap = QLabel(label.upper())
            cap.setObjectName("MetricLabel")
            body = QLabel(text)
            body.setObjectName("Dim")
            body.setWordWrap(True)
            v.addWidget(cap)
            v.addWidget(body)
            self.lay.insertWidget(self.lay.count() - 1, block)


class MainWindow(QMainWindow):
    def __init__(self, folder="musica"):
        super().__init__()
        self.setWindowTitle("Auto DJ Mix")
        self.resize(1640, 1000)
        self.setMinimumSize(1020, 700)

        self.folder = folder
        self.files, self.metas = [], []
        self.result = None
        self.cache = {}
        self.order = None
        self.join_names = None
        self.bars = None
        self._segs = None
        self._spb = None
        self._joins = []
        self.dirty = False
        self.current_join = 0
        self._threads = []
        self._meta = None

        self.player = Player(self)
        self.player.position.connect(self._on_position)
        self.player.state_changed.connect(self._on_play_state)

        root = QWidget()
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.navigated.connect(self._navigate)
        shell.addWidget(self.sidebar)

        middle = QVBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        self.topbar = TopBar()
        self.topbar.search_changed.connect(self._search)
        self.topbar.render_clicked.connect(self.render_mix)
        self.topbar.theme_toggled.connect(self.toggle_theme)
        self.topbar.save_clicked.connect(self.save_mix)
        middle.addWidget(self.topbar)

        self.pages = QStackedWidget()
        self.pages.setObjectName("Canvas")
        middle.addWidget(self.pages, 1)
        shell.addLayout(middle, 1)

        self.queue = QueuePanel()
        self.queue.folder_requested.connect(self.choose_folder)
        self.queue.settings_changed.connect(self._settings_changed)
        self.queue.transition_changed.connect(self._set_transition)
        self.queue.bars_changed.connect(self._set_bars)
        self.queue.row_clicked.connect(self._select_position)
        shell.addWidget(self.queue)

        self._build_mix_page()
        self._build_library_page()
        self._build_correct_page()
        self._build_stems_page()
        self._build_report_page()
        self._shortcuts()
        self._check_gpu()
        if not AUDIO_AVAILABLE:
            self.transport.set_enabled(False)
        QTimer.singleShot(120, self.analyse)

    # ------------------------------------------------------------- pages ---
    def _page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(T.PAD_VIEW, T.PAD_VIEW, T.PAD_VIEW, T.PAD_VIEW)
        lay.setSpacing(T.GAP)
        self.pages.addWidget(w)
        return w, lay

    def _build_mix_page(self):
        page, lay = self._page()

        self.transport = Transport()
        self.transport.play_pause.connect(self.player.toggle)
        self.transport.stopped.connect(self.player.stop)
        self.transport.seek_delta.connect(
            lambda d: self.player.seek(self.player.seconds() + d))
        self.transport.audition.connect(self.audition_join)
        self.transport.volume.connect(self.player.set_volume)
        lay.addWidget(self.transport)

        mix_card = Card("The mix", "click to seek · scroll to zoom")
        self.mix_wave = WaveformView()
        self.mix_wave.show_grid = False
        self.mix_wave.show_segments = False
        self.mix_wave.setMinimumHeight(128)
        self.mix_wave.scrubbed.connect(self.player.seek)
        mix_card.body.addWidget(self.mix_wave, 1)
        lay.addWidget(mix_card, 3)

        self.decks = DeckPair()
        lay.addWidget(self.decks, 4)

        time_card = Card("Set timeline",
                         "drag to reorder · click a chip to change the move")
        self.timeline = TimelineView()
        self.timeline.selected.connect(self._select_position)
        self.timeline.reordered.connect(self._reorder)
        self.timeline.join_clicked.connect(self._select_join)
        self.timeline.seeked.connect(self.player.seek)
        time_card.body.addWidget(self.timeline, 1)
        lay.addWidget(time_card, 2)

    def _build_library_page(self):
        page, lay = self._page()

        tiles = QHBoxLayout()
        tiles.setSpacing(T.GAP)
        self.tile_tracks = StatTile("Tracks", "—", "in library")
        self.tile_bpm = StatTile("Mix tempo", "—", "median BPM")
        self.tile_keys = StatTile("Keys", "—", "distinct Camelot")
        self.tile_len = StatTile("Set length", "—", "estimated")
        for t in (self.tile_tracks, self.tile_bpm, self.tile_keys,
                  self.tile_len):
            tiles.addWidget(t)
        lay.addLayout(tiles)

        card = Card("Library", "click a row to inspect")
        self.model = LibraryModel()
        self.proxy = LibraryFilter()
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(False)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, (_, _, w) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, w)
        self.table.selectionModel().selectionChanged.connect(self._row_picked)

        self.lib_stack = QStackedLayout()
        self.lib_skeleton = Skeleton(rows=8)
        self.lib_empty = EmptyState("▤", "No tracks analysed",
                                    "Point Auto DJ at a folder of music "
                                    "to begin.")
        pick = QPushButton("  Choose folder…")
        pick.setObjectName("Primary")
        icons.apply(pick, "folder", 15, "ON_ACCENT")
        pick.setIconSize(QSize(15, 15))
        pick.setCursor(Qt.CursorShape.PointingHandCursor)
        pick.clicked.connect(self.choose_folder)
        self.lib_empty.add_action(pick)
        for w in (self.lib_skeleton, self.table, self.lib_empty):
            holder = QWidget()
            v = QVBoxLayout(holder)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(w)
            self.lib_stack.addWidget(holder)
        card.body.addLayout(self.lib_stack, 1)
        lay.addWidget(card, 3)

        chart = Card("Tempo across the library", "BPM per track")
        self.chart = BarChart(colour=T.SERIES[0], unit=" BPM", zero=False)
        chart.body.addWidget(self.chart, 1)
        lay.addWidget(chart, 1)

    def _build_correct_page(self):
        page, lay = self._page()

        card = Card("Waveform", "beatgrid · segments · cues")
        self.wave_title = QLabel("—")
        self.wave_title.setObjectName("Dim")
        badges = QHBoxLayout()
        badges.setSpacing(T.UNIT)
        badges.addWidget(self.wave_title)
        badges.addStretch(1)
        self.badge_bpm = Badge("—", "accent")
        self.badge_key = Badge("—", "neutral")
        self.badge_energy = Badge("—", "neutral")
        for b in (self.badge_bpm, self.badge_key, self.badge_energy):
            badges.addWidget(b)
        card.body.addLayout(badges)

        self.wave = WaveformView()
        self.wave.edited.connect(self._apply_correction)
        self.wave.removed.connect(self._remove_correction)
        card.body.addWidget(self.wave, 1)
        lay.addWidget(card, 3)

        self.correct = CorrectPanel()
        self.correct.mode_changed.connect(self._set_edit_mode)
        self.correct.bpm_changed.connect(
            lambda v: self._save_correction(bpm=round(v, 4)))
        self.correct.nudged.connect(self._nudge_grid)
        self.correct.reset.connect(self._reset_corrections)
        lay.addWidget(self.correct, 1)

    def _build_stems_page(self):
        page, lay = self._page()
        card = Card("Stem separation",
                    "vocals · drums · bass · other, cached to disk")
        body = QLabel(
            "Separating a track unlocks the Neural Mix transition, vocal "
            "collision avoidance, and clean bass swaps. Roughly a minute per "
            "track on the GPU; results are cached, so it is a one-off cost.")
        body.setObjectName("Dim")
        body.setWordWrap(True)
        card.body.addWidget(body)

        row = QHBoxLayout()
        row.setSpacing(T.UNIT + 4)
        self.btn_stems = QPushButton("  Separate selected track")
        self.btn_stems.setObjectName("Primary")
        icons.apply(self.btn_stems, "split", 15, "ON_ACCENT")
        self.btn_stems.setIconSize(QSize(15, 15))
        self.btn_stems.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stems.clicked.connect(lambda: self.separate(False))
        self.btn_stems_all = QPushButton("  Separate whole library")
        self.btn_stems_all.setObjectName("Ghost")
        icons.apply(self.btn_stems_all, "split", 15)
        self.btn_stems_all.setIconSize(QSize(15, 15))
        self.btn_stems_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stems_all.clicked.connect(lambda: self.separate(True))
        row.addWidget(self.btn_stems)
        row.addWidget(self.btn_stems_all)
        row.addStretch(1)
        card.body.addLayout(row)

        self.stem_state = QLabel("")
        self.stem_state.setObjectName("CardHint")
        self.stem_state.setWordWrap(True)
        card.body.addWidget(self.stem_state)
        card.body.addStretch(1)
        lay.addWidget(card, 1)

    def _build_report_page(self):
        page, lay = self._page()

        top = QHBoxLayout()
        top.setSpacing(T.GAP)

        master = Card("Master", "after the last render")
        self.spark = Sparkline([], T.SERIES[2])
        master.body.addWidget(self.spark)
        rows = QGridLayout()
        rows.setVerticalSpacing(T.UNIT)
        self._master_labels = {}
        for r, key in enumerate(["Loudness", "Crest", "Peak", "Stretch",
                                 "Glide"]):
            k = QLabel(key)
            k.setObjectName("Muted")
            v = QLabel("—")
            v.setObjectName("Dim")
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            rows.addWidget(k, r, 0)
            rows.addWidget(v, r, 1)
            self._master_labels[key] = v
        master.body.addLayout(rows)
        master.body.addStretch(1)
        top.addWidget(master, 1)

        summary = Card("This render", "what the engine did")
        self.summary_body = QLabel("Render a mix to see the report.")
        self.summary_body.setObjectName("Dim")
        self.summary_body.setWordWrap(True)
        self.summary_body.setAlignment(Qt.AlignmentFlag.AlignTop)
        summary.body.addWidget(self.summary_body, 1)
        top.addWidget(summary, 2)
        lay.addLayout(top, 2)

        why = Card("Why this join", "the reasoning behind the selected "
                                    "transition")
        self.reasons = ReasonPanel()
        why.body.addWidget(self.reasons, 1)
        lay.addWidget(why, 3)

    def _shortcuts(self):
        for keys, fn in ((Qt.Key.Key_Space, self.player.toggle),
                         ("A", self.audition_join),
                         ("Ctrl+R", self.render_mix),
                         ("Ctrl+S", self.save_mix),
                         ("Ctrl+D", self.toggle_theme),
                         ("Ctrl+O", self.choose_folder)):
            s = QShortcut(QKeySequence(keys), self)
            s.activated.connect(fn)

    # ------------------------------------------------------------- logic ---
    def toggle_theme(self):
        """Swap palette live.

        Three things have to happen together, and leaving any one out leaves
        the window visibly half-converted: the stylesheet is regenerated from
        the new tokens, the icon cache is dropped because every mark baked its
        colour in when it was drawn, and every custom-painted widget is told
        to repaint -- the waveforms, timeline and charts read `T.*` directly
        inside paintEvent and Qt has no reason to know those values moved.
        """
        mode = "light" if T.MODE == "dark" else "dark"
        sheet = T.set_mode(mode)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(sheet)
        touched = icons.restyle(self)
        self._repaint_all()
        self.toast(f"{mode.capitalize()} mode  ·  {touched} marks redrawn",
                   "info")

    def _repaint_all(self):
        for w in self.findChildren(QWidget):
            w.update()
        self.update()

    def toast(self, text, tone="info"):
        Toast(self, text, tone)

    def _navigate(self, index):
        self.pages.setCurrentIndex(index)
        self.queue.setVisible(index == 0 and self.width() >= NARROW)
        if index != 2:
            self.correct.clear_mode()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.topbar.fit(self.width())
        wide = self.width() >= NARROW
        if getattr(self, "_wide", None) != wide:
            self._wide = wide
            self.sidebar.set_collapsed(not wide)
            self.queue.setVisible(wide and self.pages.currentIndex() == 0)

    def _check_gpu(self):
        try:
            from ..stems import cuda
            st = cuda.torch_status()
            ok = st.get("available")
            self.topbar.gpu.setText("GPU ready" if ok else "GPU off")
            self.topbar.gpu.set_tone("good" if ok else "warn")
            self.topbar.gpu.setToolTip(st.get("device") or
                                       st.get("reason", ""))
        except Exception:
            self.topbar.gpu.setText("GPU —")

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a music folder",
                                             str(self.folder))
        if d:
            self.folder = d
            self.analyse()

    # ---------------------------------------------------------- analysis ---
    def analyse(self, force=False):
        self.lib_stack.setCurrentIndex(0)
        self.model.clear()
        self.metas, self.files = [], []
        self.cache.clear()
        w = workers.AnalyseWorker(self.folder, force=force)
        w.track_done.connect(self._track_analysed)
        t, w = workers.start(
            w, on_done=self._analysis_done,
            on_progress=lambda i, n, m: self.topbar.set_busy(
                f"Analysing {i}/{n} · {m}", 100 * i / max(1, n)),
            on_failed=self._failed)
        self._threads.append((t, w))

    def _track_analysed(self, meta):
        self.model.append(meta)
        if self.lib_stack.currentIndex() != 1:
            self.lib_stack.setCurrentIndex(1)

    def _analysis_done(self, result):
        if not result:
            return
        self.files, self.metas = result
        self.lib_stack.setCurrentIndex(1 if self.metas else 2)
        self.topbar.set_busy("Ready")
        self._refresh_stats()
        if not self.metas:
            # Land on Library, where the empty state and its "Choose folder"
            # button are. Opening on Mix with nothing to mix just looks broken.
            self.sidebar.select(1)
            self._navigate(1)
            self.toast(f"No audio in {self.folder} — choose a folder", "warn")
            return
        self.toast(f"Analysed {len(self.metas)} tracks", "good")
        self.table.selectRow(0)

    def _refresh_stats(self):
        if not self.metas:
            return
        bpms = [m["bpm"] for m in self.metas]
        keys = {m["camelot"] for m in self.metas}
        total = sum(m["duration"] for m in self.metas)
        self.tile_tracks.set(len(self.metas), "in library")
        self.tile_bpm.set(f"{np.median(bpms):.1f}",
                          f"{min(bpms):.0f}–{max(bpms):.0f} range")
        self.tile_keys.set(len(keys), "of 24 Camelot",
                           "good" if len(keys) <= 6 else "warn")
        self.tile_len.set(f"{total/60:.0f}m", "sum of tracks")
        self.chart.set_data([round(b, 1) for b in bpms],
                            [m["title"] for m in self.metas])

    def _search(self, text):
        self.proxy.set_query(text)

    def _row_picked(self, *_):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        meta = self.model.meta_at(self.proxy.mapToSource(rows[0]).row())
        if meta:
            self._show_track(meta)

    def _show_track(self, meta):
        self._meta = meta
        self.wave_title.setText(f"{meta['artist']} — {meta['title']}")
        self.badge_bpm.setText(f"{meta['bpm']:.2f} BPM")
        self.badge_key.setText(meta["camelot"])
        self.badge_energy.setText(f"energy {meta['energy']:.2f}")
        self.correct.show_meta(meta)
        try:
            a, sr = audio_mod.load(meta["file"], audio_mod.ANALYSIS_RATE,
                                   mono=True)
            self.wave.set_track(a, sr, meta)
        except Exception as e:
            self.toast(f"Could not load audio: {e}", "bad")

    # -------------------------------------------------------- corrections --
    def _set_edit_mode(self, mode):
        self.wave.edit_mode = mode
        self.wave.update()

    def _bar_of(self, seconds):
        m = self._meta
        bar = 4 * 60.0 / m["bpm"]
        return max(0, int(round((seconds - m.get("first_downbeat", 0.0))
                                / bar)))

    def _apply_correction(self, field, seconds):
        if not self._meta:
            return
        if field == "downbeat":
            # Place the nearest beat exactly here. Storing the offset rather
            # than a bar index means the correction survives a later tempo
            # change instead of silently pointing somewhere else.
            beat = 60.0 / self._meta["bpm"]
            phase = int(self._meta.get("downbeat_phase", 0))
            self._save_correction(beat_offset=round(seconds - phase * beat, 6))
        elif field == "drop":
            drops = sorted(set((self._meta["cues"].get("drop_bars") or [])
                               + [self._bar_of(seconds)]))
            self._save_correction(drop_bars=drops)
        else:
            self._save_correction(**{field: self._bar_of(seconds)})

    def _remove_correction(self, field, seconds):
        if not self._meta or field != "drop":
            return
        target = self._bar_of(seconds)
        drops = [d for d in (self._meta["cues"].get("drop_bars") or [])
                 if abs(d - target) > 1]
        self._save_correction(drop_bars=drops)

    def _nudge_grid(self, delta):
        if self._meta:
            self._save_correction(
                beat_offset=round(self._meta.get("beat_offset", 0.0) + delta,
                                  6))

    def _save_correction(self, **fields):
        if not self._meta:
            return
        corrections.set_fields(self._meta["file"], **fields)
        self._reanalyse_current()

    def _reset_corrections(self):
        if not self._meta:
            return
        corrections.clear(self._meta["file"])
        self._reanalyse_current()

    def _reanalyse_current(self):
        """Re-run analysis for the corrected file only, then refresh.

        Only this file: the correction changed one track's grid, and
        re-analysing a library because one downbeat moved is the sort of thing
        that makes people stop correcting downbeats.
        """
        from ..analysis import track as track_mod
        path = self._meta["file"]
        try:
            meta = track_mod.analyse(path)
        except Exception as e:
            self.toast(f"Re-analysis failed: {e}", "bad")
            return
        for i, m in enumerate(self.metas):
            if m["file"] == path:
                self.metas[i] = meta
                self.model.replace(i, meta)
                break
        self.cache.clear()
        self._mark_dirty()
        self._show_track(meta)
        self.toast("Re-analysed with your correction", "good")

    # ------------------------------------------------------------- render --
    def _mark_dirty(self, *_):
        self.dirty = True
        self.topbar.dirty.show()

    def save_mix(self):
        """Write the rendered mix to disk, with its tracklist and cue sheet.

        Writes the audio already in memory rather than re-rendering. The
        render worker has always been able to save -- it takes an `out_path`
        and writes WAV, tracklist, cue sheet and JSON -- but the window only
        ever called it with `out_path=None`, so the only way to get a file out
        of this program was the command line. Re-rendering to save would also
        be a minute of work to produce audio we are already holding.
        """
        if not getattr(self, "result", None):
            self.toast("Render a mix first", "warn")
            return
        from pathlib import Path

        from .. import export
        from .. import render as render_mod

        path, _ = QFileDialog.getSaveFileName(
            self, "Export mix", str(Path("output") / "mix.wav"),
            "Wave audio (*.wav);;MP3 audio (*.mp3);;FLAC audio (*.flac)")
        if not path:
            return

        out = Path(path)
        rep = self.result["report"]
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            audio_mod.save(out, self.result["audio"], audio_mod.RENDER_RATE)
            # `tracklist` and `mix_plan` take built joins, not names -- the
            # same distinction that took the render path down.
            out.with_suffix(".txt").write_text(
                render_mod.tracklist(self.metas, self.order, self._segs,
                                     self._joins, self._spb,
                                     audio_mod.RENDER_RATE),
                encoding="utf-8")
            out.with_suffix(".cue").write_text(
                render_mod.cue_sheet(self.metas, self.order, self._segs,
                                     self._spb, audio_mod.RENDER_RATE,
                                     out.name),
                encoding="utf-8")
            export.write(out.with_suffix(".json"),
                         export.mix_plan(self.metas, self.order, self._segs,
                                         self._joins, rep,
                                         self.bars[0] if self.bars else 8,
                                         audio_mod.RENDER_RATE,
                                         self.queue.settings()["arc"]))
        except Exception as exc:
            self.toast(f"Could not save: {exc}", "bad")
            return
        self.toast(f"Saved {out.name} · tracklist, cue and JSON alongside",
                   "good")

    def _settings_changed(self):
        """A settings change may rewrite every join, not just mark us dirty.

        The Transition control is set-wide: choosing a specific move means
        "use this at every join", and choosing Automatic hands each join back
        to the planner. Either way the per-row combos have to be re-shown, or
        the panel would claim one thing while the render used another.
        """
        self._mark_dirty()
        if self.join_names is None or self.order is None:
            return
        pick = self.queue.settings()["transition"]
        names = [self._auto_name(j) if pick == "auto" else pick
                 for j in range(len(self.join_names))]
        if names == self.join_names:
            return
        self.join_names = names
        self._refresh_queue()

    def _refresh_queue(self):
        """Re-show the queue rows and timeline from the current plan.

        Both panels take built `Transition` objects, not names -- the queue
        reads `join.name` and the timeline draws the chip from the join's own
        curves. Handing them the name strings raises inside a signal handler,
        which in PyQt takes the whole process down rather than printing
        something; that is exactly what it did.
        """
        from .. import transitions as tr_mod
        if self.order is None or not getattr(self, "_segs", None):
            return
        beat_s = self._spb / 4.0 / audio_mod.RENDER_RATE
        joins = []
        for i, name in enumerate(self.join_names):
            bars = self.bars[i] if i < len(self.bars) else 8
            try:
                joins.append(tr_mod.build(name, bars, self._spb,
                                          audio_mod.RENDER_RATE, beat_s))
            except Exception:
                joins.append(self._joins[i] if i < len(self._joins) else None)
        self._joins = joins
        self.queue.set_plan(self.metas, self.order, self._segs, joins,
                            self.bars)
        self.timeline.set_plan(self.metas, self.order, self._segs, joins,
                               self._spb, audio_mod.RENDER_RATE)
        self._select_join(min(self.current_join,
                              max(0, len(self.join_names) - 1)))

    def _set_transition(self, join, name):
        if self.join_names is None:
            return
        self.join_names[join] = (self._auto_name(join) if name == "auto"
                                 else name)
        self._mark_dirty()
        self.preview(join)

    def preview(self, join):
        """Render and play just this join, so a change can be heard at once.

        Not a full render. The whole mix takes about a minute, and most of
        that is the master chain over fifteen minutes of audio; one join is
        half a minute of audio and comes back in a couple of seconds. Waiting
        a minute to hear a four-bar decision is what stops people from trying
        things.
        """
        if not self.result or not AUDIO_AVAILABLE:
            return
        if getattr(self, "_previewing", False):
            return
        self._previewing = True
        self.topbar.set_busy(f"Previewing join {join + 1}…", 20)
        w = workers.PreviewWorker(self.files, self.metas, self.order,
                                  self.result["segs"], join,
                                  self.join_names[join], self.bars[join],
                                  cache=self.cache)
        t, w = workers.start(
            w, on_done=self._preview_done,
            on_progress=lambda i, n, m: self.topbar.set_busy(m, i),
            on_failed=self._preview_failed)
        self._threads.append((t, w))

    def _preview_done(self, result):
        self._previewing = False
        self.topbar.set_busy("Ready")
        if not result:
            return
        rep = result["report"]
        self.player.set_audio(result["audio"], audio_mod.RENDER_RATE)
        # Start a bar or so before the transition rather than at the very top
        # of the lead-in: the point is the join, and the run-up is context.
        self.player.play(max(0.0, rep["region_start_s"] - 4.0))
        self._preview_mode = True
        self.toast(f"Preview: {LABELS.get(rep['transition'], rep['transition'])}"
                   f" · {rep['bars']} bars · {rep['elapsed_s']}s", "info")

    def _preview_failed(self, text):
        self._previewing = False
        self.topbar.set_busy("Ready")
        self.toast(text.strip().splitlines()[-1][:130], "bad")

    def _auto_name(self, join):
        inten = planner.intensity(self.metas)
        a, b = self.order[join], self.order[join + 1]
        return planner.choose_transition(self.metas[a], self.metas[b],
                                         inten[a], inten[b],
                                         self.queue.settings()["style"])

    def _set_bars(self, join, value):
        if self.bars is not None and join < len(self.bars):
            self.bars[join] = planner.snap_phrase(value)
            self._mark_dirty()
            self.preview(join)

    def _reorder(self, src, dst):
        if self.order is None:
            return
        order = list(self.order)
        order.insert(dst, order.pop(src))
        self.order = order
        # The joins moved with the tracks, so every automatic choice around
        # the move is now about a different pair and has to be re-decided.
        self.join_names = None
        self._mark_dirty()
        self.toast(f"Moved to position {dst + 1} — render to hear it", "info")
        self.render_mix()

    def render_mix(self):
        if not self.metas:
            self.toast("Analyse a folder first", "warn")
            return
        s = self.queue.settings()
        self.topbar.render.setEnabled(False)
        self.topbar.render.setText("Rendering…")
        w = workers.RenderWorker(
            self.files, self.metas, out_path=None,
            style=s["style"], arc=s["arc"],
            duration_mode=s["duration_mode"], duration_value=s["duration_value"],
            tempo_mode=s["tempo_mode"], order=self.order,
            join_names=self.join_names, bars=self.bars,
            align=True, cache=self.cache)
        t, w = workers.start(
            w, on_done=self._render_done,
            on_progress=lambda i, n, m: self.topbar.set_busy(m, i),
            on_failed=self._failed)
        self._threads.append((t, w))

    def _render_done(self, result):
        self.topbar.render.setEnabled(True)
        self.topbar.render.setText("Render mix")
        self.topbar.set_busy("Ready")
        self.topbar.dirty.hide()
        self.dirty = False
        if not result:
            return
        self.result = result
        rep = result["report"]
        self.order = list(result["order"])
        self.join_names = list(rep["transitions"])
        self.bars = list(rep["transition_bars"])

        # Kept so the plan can be re-shown without re-rendering -- picking a
        # set-wide transition rewrites every join, and the panel has to show
        # that before the render that makes it true.
        self._segs = result["segs"]
        self._spb = result["spb"]
        self._joins = result["joins"]
        self.topbar.save_btn.setEnabled(True)
        self.timeline.set_plan(self.metas, self.order, self._segs,
                               self._joins, self._spb,
                               audio_mod.RENDER_RATE)
        self.queue.set_plan(self.metas, self.order, self._segs,
                            self._joins, self.bars)

        a = result["audio"]
        self._preview_mode = False
        self.player.set_audio(a, audio_mod.RENDER_RATE)
        self.mix_wave.set_track(a, audio_mod.RENDER_RATE, None)
        self.transport.set_clock(0, self.player.duration())
        self.transport.set_enabled(AUDIO_AVAILABLE)

        self._master_labels["Loudness"].setText(f"{rep['lufs_out']} LUFS")
        self._master_labels["Crest"].setText(f"{rep.get('crest_out','—')} dB")
        self._master_labels["Peak"].setText(f"{rep['peak_out']}")
        self._master_labels["Stretch"].setText(f"{rep['max_stretch_pct']}%")
        self._master_labels["Glide"].setText(
            f"{rep.get('max_glide_cents', 0):.0f} cents"
            if rep.get("tempo_glide") else "flat")
        self.summary_body.setText("\n".join(explain.summary(rep)))

        mono = a.mean(0)
        step = max(1, len(mono) // 700)
        env = np.sqrt(np.convolve(mono[::step] ** 2, np.ones(9) / 9, "same"))
        self.spark.set_values(env)
        self._select_join(min(self.current_join, max(0, len(self.join_names) - 1)))
        self.toast(f"Rendered {rep['duration_s']/60:.1f} min in "
                   f"{rep['elapsed_s']}s", "good")

    # ------------------------------------------------------------ playing --
    def _on_position(self, seconds):
        self.mix_wave.set_playhead(seconds)
        self.transport.set_clock(seconds, self.player.duration())
        # While a preview is loaded the player holds half a minute of one join,
        # not the set. Mapping that position onto the set's timeline would put
        # the playhead and the active queue row somewhere arbitrary.
        if getattr(self, "_preview_mode", False):
            return
        self.timeline.set_playhead(seconds)
        pos = self._position_at(seconds)
        if pos >= 0:
            self.queue.set_active(pos)
            self.transport.set_here(
                f"{self.metas[self.order[pos]]['title'][:34]}")

    def _on_play_state(self, playing):
        self.transport.set_playing(playing)

    def _position_at(self, seconds):
        if not self.result:
            return -1
        rate = audio_mod.RENDER_RATE
        for i, seg in enumerate(self.result["segs"]):
            start = seg.get("start_sample", 0) / rate
            end = start + seg.get("len_samples", 0) / rate
            if start <= seconds < end:
                return i
        return -1

    def join_time(self, join):
        """When a transition begins, in seconds into the mix."""
        if not self.result or join >= len(self.result["segs"]) - 1:
            return 0.0
        rate = audio_mod.RENDER_RATE
        seg = self.result["segs"][join]
        end = (seg.get("start_sample", 0) + seg.get("len_samples", 0)) / rate
        return max(0.0, end - seg.get("tail_samples", 0) / rate)

    def audition_join(self, lead=15.0):
        """Jump to just before the selected transition and play it."""
        if not self.result or not AUDIO_AVAILABLE:
            self.toast("Render a mix first", "warn")
            return
        if getattr(self, "_preview_mode", False):
            # The player currently holds a preview, not the set. Put the mix
            # back before seeking into it, or the position is meaningless.
            self._preview_mode = False
            self.player.set_audio(self.result["audio"], audio_mod.RENDER_RATE)
        at = max(0.0, self.join_time(self.current_join) - lead)
        self.player.play(at)
        self.sidebar.select(0)
        self.pages.setCurrentIndex(0)

    # ---------------------------------------------------------- selection --
    def _select_position(self, pos):
        if not self.result:
            return
        self.timeline.current = pos
        self.timeline.update()
        self.queue.set_active(pos)
        self._select_join(min(pos, max(0, len(self.result["joins"]) - 1)))

    def _select_join(self, join):
        if not self.result or not self.result["joins"]:
            return
        join = int(np.clip(join, 0, len(self.result["joins"]) - 1))
        self.current_join = join
        segs = self.result["segs"]
        a_meta = self.metas[self.order[join]]
        b_meta = self.metas[self.order[join + 1]]
        name = self.result["joins"][join].name
        bars = self.bars[join] if self.bars else self.result["joins"][join].bars

        rows = explain.join(join, self.metas, self.order, segs,
                            self.result["joins"], self.result["report"],
                            self.bars)
        self.reasons.show_rows(rows)
        reason = next((t for l, t in rows if l == "Transition"), "")
        # Whole string, not a 96-character slice. The slice chopped mid-word
        # with no ellipsis, so the reason looked clipped by the card when it
        # had actually been cut before it ever got there. The label elides at
        # paint time against its real width, which is the only place that
        # decision can be made correctly.
        self.decks.set_join(name, bars, a_meta, b_meta,
                            reason.split("— ", 1)[-1])
        self._load_deck(self.decks.out, a_meta, segs[join], tail=True,
                        bars=bars)
        self._load_deck(self.decks.inn, b_meta, segs[join + 1], tail=False,
                        bars=bars)

    def _load_deck(self, view, meta, seg, tail, bars):
        """Show the part of a track that takes part in the join."""
        try:
            a, sr = audio_mod.load(meta["file"], audio_mod.ANALYSIS_RATE,
                                   mono=True)
        except Exception:
            view.clear()
            return
        view.set_track(a, sr, meta, view.label)
        bar_s = 4 * 60.0 / meta["bpm"]
        first = meta.get("first_downbeat", 0.0)
        edge = seg["exit"] if tail else seg["enter"]
        centre = first + edge * bar_s
        window = max(12.0, bars * bar_s * 2.4)
        view.set_span(centre - window / 2, centre + window / 2)

    # ------------------------------------------------------------- stems ---
    def separate(self, whole_library):
        from pathlib import Path
        if whole_library:
            files = [Path(m["file"]) for m in self.metas]
        else:
            rows = self.table.selectionModel().selectedRows()
            if not rows:
                self.toast("Select a track in the Library first", "warn")
                return
            meta = self.model.meta_at(self.proxy.mapToSource(rows[0]).row())
            files = [Path(meta["file"])]
        if not files:
            return
        self.btn_stems.setEnabled(False)
        self.btn_stems_all.setEnabled(False)
        w = workers.SeparateWorker(files)
        t, w = workers.start(
            w, on_done=self._stems_done,
            on_progress=lambda i, n, m: self.topbar.set_busy(
                m, 100 * i / max(1, n)),
            on_failed=self._failed)
        self._threads.append((t, w))

    def _stems_done(self, result):
        self.btn_stems.setEnabled(True)
        self.btn_stems_all.setEnabled(True)
        self.topbar.set_busy("Ready")
        if result:
            secs = sum(v["seconds"] for v in result.values())
            self.stem_state.setText(
                f"Separated {len(result)} track(s) in {secs:.0f}s. "
                f"Neural Mix transitions are now available for them.")
            self.cache.clear()
            self.toast(f"Separated {len(result)} track(s)", "good")

    def _failed(self, text):
        self.topbar.render.setEnabled(True)
        self.topbar.render.setText("Render mix")
        self.btn_stems.setEnabled(True)
        self.btn_stems_all.setEnabled(True)
        self.topbar.set_busy("Ready")
        self.toast(text.strip().splitlines()[-1][:130], "bad")

    def closeEvent(self, e):
        self.player.stop()
        for t, w in self._threads:
            w.cancel()
            t.quit()
            t.wait(1500)
        super().closeEvent(e)


def run(folder="musica"):
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName("Auto DJ Mix")
    app.setWindowIcon(icons.app_qicon())
    # Family only. A QFont built with setPixelSize reports pointSize() == -1,
    # and Qt internals that round-trip the application font through
    # setPointSize then emit "QFont::setPointSize: Point size <= 0 (-1)" on
    # startup. Sizes come from the stylesheet, which sets font-size in px on
    # QWidget, so nothing is lost by leaving the point size alone here.
    app.setFont(T.app_font())
    app.setStyleSheet(T.stylesheet())
    win = MainWindow(folder)
    win.setWindowIcon(icons.app_qicon())
    win.show()
    return app.exec()
