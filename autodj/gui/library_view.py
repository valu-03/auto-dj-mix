"""The library table: a model over analysed tracks, plus search and sorting.

A QAbstractTableModel rather than QTableWidget. With a few hundred tracks the
item-widget approach allocates a widget per cell and the view stutters on every
scroll; a model hands Qt only the cells that are visible.
"""

from PyQt6.QtCore import (QAbstractTableModel, QModelIndex,
                          QSortFilterProxyModel, Qt)
from PyQt6.QtGui import QColor

from . import theme as T

COLUMNS = [
    ("Artist", "artist", 190),
    ("Title", "title", 230),
    ("BPM", "bpm", 78),
    ("Key", "camelot", 62),
    ("Energy", "energy", 78),
    ("Bars", "n_bars", 62),
    ("Length", "duration", 82),
]


class LibraryModel(QAbstractTableModel):
    def __init__(self, metas=None, parent=None):
        super().__init__(parent)
        self.metas = list(metas or [])

    # -- required -----------------------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.metas)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and \
                role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section][0]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        m = self.metas[index.row()]
        key = COLUMNS[index.column()][1]
        value = m.get(key)

        if role == Qt.ItemDataRole.DisplayRole:
            if key == "bpm":
                return f"{value:.2f}"
            if key == "energy":
                return f"{value:.2f}"
            if key == "duration":
                return f"{int(value)//60}:{int(value)%60:02d}"
            return str(value)

        # Sort on the raw number, not the formatted string -- otherwise "9.50"
        # sorts after "10.20" and the column is quietly useless.
        if role == Qt.ItemDataRole.UserRole:
            return value

        if role == Qt.ItemDataRole.TextAlignmentRole and key in (
                "bpm", "energy", "n_bars", "duration", "camelot"):
            return int(Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "camelot":
                return QColor(T.SERIES[1])
            if key in ("bpm", "energy"):
                return QColor(T.TEXT)
            if key in ("n_bars", "duration"):
                return QColor(T.TEXT_MUTED)
        return None

    # -- helpers ------------------------------------------------------------
    def set_metas(self, metas):
        self.beginResetModel()
        self.metas = list(metas)
        self.endResetModel()

    def append(self, meta):
        self.beginInsertRows(QModelIndex(), len(self.metas), len(self.metas))
        self.metas.append(meta)
        self.endInsertRows()

    def replace(self, row, meta):
        """Swap one row in place, after that track has been re-analysed.

        A full model reset would work and would also drop the selection, the
        scroll position and the sort -- which, when the user has just corrected
        a downbeat and wants to see the result, throws away the context that
        made the correction meaningful.
        """
        if not 0 <= row < len(self.metas):
            return
        self.metas[row] = meta
        self.dataChanged.emit(self.index(row, 0),
                              self.index(row, len(COLUMNS) - 1))

    def row_of(self, path):
        for i, m in enumerate(self.metas):
            if m.get("file") == path:
                return i
        return -1

    def clear(self):
        self.set_metas([])

    def meta_at(self, row):
        return self.metas[row] if 0 <= row < len(self.metas) else None


class LibraryFilter(QSortFilterProxyModel):
    """Case-insensitive search across artist, title and key."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortRole(Qt.ItemDataRole.UserRole)
        self._needle = ""

    def set_query(self, text):
        self._needle = (text or "").strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        if not self._needle:
            return True
        m = self.sourceModel().meta_at(row) or {}
        hay = f"{m.get('artist','')} {m.get('title','')} {m.get('camelot','')}"
        return self._needle in hay.lower()
