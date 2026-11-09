"""English-language PySide6 interface for interactive scan color matching."""

from __future__ import annotations

import os

# The bundled OpenBLAS (built for at most 2 threads) crashes when several Python
# threads call into it at once. Work is parallel per scan instead; set before numpy
# loads, and inherited by every spawned worker process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import copy
import hashlib
import io
import json
import math
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import (Qt, QObject, QThread, Signal, QRectF, QPointF, QEvent, QTimer, QSize,
                            QStandardPaths, QSettings, QLocale)
from PySide6.QtGui import (QBrush, QColor, QCursor, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QButtonGroup, QDialog, QFrame,
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout,
    QGridLayout, QLineEdit, QListWidget, QMainWindow, QMenu,
    QProgressBar, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QStyledItemDelegate, QStyle, QVBoxLayout, QWidget,
)

try:  # Package execution: python -m ...
    from . import engine, i18n
    from .calibration_worker import run_calibration
    from .project import Project
except ImportError:  # Direct entry point: python scan_color_match.py
    import engine
    import i18n
    from calibration_worker import run_calibration
    from project import Project

tr = i18n.tr


# Interface text is written in English; these drop-in subclasses translate it when it is
# shown (a no-op in English), so every label, button, status message and dialog follows
# the chosen language without wrapping each call site.
def _tr_first(args):
    return (tr(args[0]), *args[1:]) if args and isinstance(args[0], str) else args


class _TextTranslation:
    def __init__(self, *args, **kwargs):
        super().__init__(*_tr_first(args), **kwargs)

    def setText(self, text):
        super().setText(tr(text))

    def setToolTip(self, text):
        super().setToolTip(tr(text))


class QLabel(_TextTranslation, QtWidgets.QLabel):
    pass


class QPushButton(_TextTranslation, QtWidgets.QPushButton):
    pass


class QCheckBox(_TextTranslation, QtWidgets.QCheckBox):
    pass


class QRadioButton(_TextTranslation, QtWidgets.QRadioButton):
    pass


class QListWidgetItem(_TextTranslation, QtWidgets.QListWidgetItem):
    pass


class QAction(_TextTranslation, QtGui.QAction):
    pass


class QComboBox(QtWidgets.QComboBox):
    def addItem(self, *args, **kwargs):
        super().addItem(*_tr_first(args), **kwargs)

    def add_described_item(self, text, data, description):
        """An option whose description shows while hovering it in the open list."""
        self.addItem(text, data)
        self.setItemData(self.count() - 1, tr(description), Qt.ItemDataRole.ToolTipRole)

    def setToolTip(self, text):
        super().setToolTip(tr(text))


class QMessageBox(QtWidgets.QMessageBox):
    def setText(self, text):
        super().setText(i18n.QUICK_START_ZH if text == QUICK_START_HTML and i18n.language() == "zh"
                        else tr(text))

    def setWindowTitle(self, text):
        super().setWindowTitle(tr(text))

    @staticmethod
    def warning(parent, title, text, *args, **kwargs):
        return QtWidgets.QMessageBox.warning(parent, tr(title), tr(text), *args, **kwargs)

    @staticmethod
    def question(parent, title, text, *args, **kwargs):
        return QtWidgets.QMessageBox.question(parent, tr(title), tr(text), *args, **kwargs)

    @staticmethod
    def information(parent, title, text, *args, **kwargs):
        return QtWidgets.QMessageBox.information(parent, tr(title), tr(text), *args, **kwargs)


class QFileDialog(QtWidgets.QFileDialog):
    @staticmethod
    def getOpenFileNames(parent=None, caption="", directory="", filter="", *args, **kwargs):
        return QtWidgets.QFileDialog.getOpenFileNames(parent, tr(caption), directory, tr(filter),
                                                      *args, **kwargs)

    @staticmethod
    def getExistingDirectory(parent=None, caption="", directory="", *args, **kwargs):
        return QtWidgets.QFileDialog.getExistingDirectory(parent, tr(caption), directory,
                                                          *args, **kwargs)


class QInputDialog(QtWidgets.QInputDialog):
    @staticmethod
    def getText(parent, title, label, *args, **kwargs):
        return QtWidgets.QInputDialog.getText(parent, tr(title), tr(label), *args, **kwargs)


def translate_widget_tree(root: QtWidgets.QWidget) -> None:
    """Translate text set on widgets that are not covered by the subclasses above."""
    if i18n.language() == "en":
        return
    for widget in [root, *root.findChildren(QtWidgets.QWidget)]:
        if widget.toolTip():
            widget.setToolTip(tr(widget.toolTip()))
        if widget.isWindow() and widget.windowTitle():
            widget.setWindowTitle(tr(widget.windowTitle()))
        if isinstance(widget, QtWidgets.QMenu):
            widget.setTitle(tr(widget.title()))
        if isinstance(widget, QtWidgets.QAbstractButton) and widget.text():
            widget.setText(tr(widget.text()))


IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.webp)"
QUICK_START_HTML = (
            "<h3>Scan Color Match</h3>"
            "<p><b>1.</b> The default project opens automatically. You can also create or open another project. "
            "Projects save automatically; source images are not changed.</p>"
            "<p><b>2.</b> Drop one or more scans onto the left list. Select a scan, then drop its "
            "reference image(s) onto the References list. File names do not need to match. "
            "Dropping one image on a preview replaces its selected image; if none is selected, it adds one.</p>"
            "<p><b>3.</b> Choose Shared or Own LUT, then click Run Color Match. Scans without "
            "references use the shared LUT. When only part of a reference matches the scan, "
            "use Edit Mask on the reference preview: exclude watermarks or unmatched content, or "
            "switch to Inclusion and paint the matching part. The mask limits training "
            "samples, but the resulting LUT still covers the full scan. If alignment fails, crop "
            "the reference to the matching content before adding it.</p>"
            "<p><b>4.</b> For a local fix, select a reference, paint or draw a "
            "polygon, then click Refine All. It fits changed layers from bottom to top and "
            "chooses their strengths automatically. Toggle each layer with its checkbox.</p>"
            "<p><b>5.</b> Export when ready. The export uses the current visible layers and mask edits.</p>"
            "<p><b>Shortcuts:</b> B brush · E eraser · X eraser/previous tool · L polygon · Esc exit tool · hold Space for "
            "original · middle-drag to pan · wheel to zoom. "
            "Alt + right-drag changes brush diameter horizontally or hardness vertically.</p>"
)
PREVIEW_MAX_SIDE = 4000
# Keep the interactive layer composite on exactly the same pixel grid as the
# normal image preview. Swapping in a smaller pixmap made every mask edit blur.
LOCAL_FAST_PREVIEW_SIDE = PREVIEW_MAX_SIDE
SCAN_HAS_REF_ROLE = int(Qt.ItemDataRole.UserRole) + 1
SCAN_COLOR_ROLE = int(Qt.ItemDataRole.UserRole) + 2
SCAN_THUMB_ROLE = int(Qt.ItemDataRole.UserRole) + 3
SCAN_NAME_ROLE = int(Qt.ItemDataRole.UserRole) + 4
SCAN_DETAIL_ROLE = int(Qt.ItemDataRole.UserRole) + 5
SCAN_REF_COUNT_ROLE = int(Qt.ItemDataRole.UserRole) + 6
SCAN_INDICATOR_ROLE = int(Qt.ItemDataRole.UserRole) + 7
SCAN_LINKED_ROLE = int(Qt.ItemDataRole.UserRole) + 8
# Left indicator of scans with references: their LUT mode.
LUT_MODE_COLORS = {"own+shared": "#2684d8", "own": "#7b4fd6", "shared": "#0f8f8a"}
SCAN_THUMB_SIZE = 40
LAYER_THUMB_ROLE = int(Qt.ItemDataRole.UserRole) + 1
LAYER_MASK_VIEW_ROLE = int(Qt.ItemDataRole.UserRole) + 2
LAYER_RESULT_THUMB_ROLE = int(Qt.ItemDataRole.UserRole) + 3
LAYER_OVERLAY_ROLE = int(Qt.ItemDataRole.UserRole) + 4
LAYER_RESULT_VIEW_ROLE = int(Qt.ItemDataRole.UserRole) + 5
LAYER_STRENGTH_ROLE = int(Qt.ItemDataRole.UserRole) + 6
LAYER_STRENGTH_MAX_ROLE = int(Qt.ItemDataRole.UserRole) + 7
# The strength box of a layer row: wide enough for "500%"; the editor covers it exactly.
STRENGTH_BOX_WIDTH = 50
STRENGTH_BOX_RIGHT = 29   # room for the delete cross


def _strength_box(inner: QRectF) -> QRectF:
    """The strength box inside a row's inner rectangle (option.rect minus 2 px)."""
    return QRectF(inner).adjusted(inner.width() - STRENGTH_BOX_RIGHT - STRENGTH_BOX_WIDTH, 4,
                                  -STRENGTH_BOX_RIGHT, -4)
GLOBAL_LAYER_ID = "__global__"
BRUSH_RADIUS_DRAG_PIXELS = 3
BRUSH_HARDNESS_DRAG_PIXELS = 6
BRUSH_ADJUST_AXIS_PIXELS = 6   # movement before the drag commits to diameter or hardness
# Polygon edges have one setting, their feather: the soft transition on each side of
# the outline in image pixels (the mask code gets it as radius=feather, hardness=0).
POLYGON_DEFAULT_FEATHER = 10
POLYGON_MAX_FEATHER = 1000


def _save_mask_snapshots(snapshots):
    """Persist immutable in-memory mask snapshots without blocking painting."""
    for path, mask in snapshots:
        temporary = path.with_name(f"{path.stem}-{uuid.uuid4().hex}.tmp.png")
        try:
            Image.fromarray(mask, "L").save(temporary)
            engine.replace_file(temporary, path)  # a preview may be reading it right now
        finally:
            if temporary.exists():
                temporary.unlink()


def _separator() -> QFrame:
    """A thin horizontal rule between regions of a side panel."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.NoFrame)
    line.setFixedHeight(1)
    line.setStyleSheet("background: #e4e7eb; border: none;")
    return line


class ScanItemDelegate(QStyledItemDelegate):
    """Status as background color, LUT mode as the left indicator, a thumbnail, the
    file name (shortened in the middle: the end is usually a page number) with the
    reference count, and one line of details."""

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(2, 2, -2, -2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(index.data(SCAN_COLOR_ROLE) or "#edf0f3"))
        painter.drawRoundedRect(rect, 6, 6)
        indicator = index.data(SCAN_INDICATOR_ROLE)
        if indicator:
            painter.setBrush(QColor(indicator))
            painter.drawRoundedRect(QRectF(rect.left() + 3, rect.top() + 5, 4, rect.height() - 10), 2, 2)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setPen(QPen(QColor("#486687"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 6, 6)
        elif index.data(SCAN_LINKED_ROLE):
            # The scan whose LUT the hovered scan uses.
            painter.setPen(QPen(QColor("#2f6db0"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 6, 6)
        size = SCAN_THUMB_SIZE
        thumb_rect = QRectF(rect.left() + 12, rect.center().y() - size / 2, size, size)
        thumb = index.data(SCAN_THUMB_ROLE)
        if isinstance(thumb, QPixmap) and not thumb.isNull():
            painter.drawPixmap(thumb_rect.toRect(), thumb)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 18))
            painter.drawRect(thumb_rect)
        left = int(thumb_rect.right()) + 9
        metrics = option.fontMetrics
        line = metrics.height()
        detail_text = index.data(SCAN_DETAIL_ROLE) or ""
        lines = 2 if detail_text else 1
        top = rect.top() + (rect.height() - lines * line - (2 if detail_text else 0)) // 2
        count = index.data(SCAN_REF_COUNT_ROLE) or 0
        count_text = f"({count})" if count else ""
        count_width = metrics.horizontalAdvance(count_text) + 4 if count_text else 0
        name_width = max(10, rect.right() - 8 - left - count_width)
        painter.setPen(QColor("#25313d"))
        name = metrics.elidedText(index.data(SCAN_NAME_ROLE) or "", Qt.TextElideMode.ElideMiddle,
                                  name_width)
        painter.drawText(QRectF(left, top, name_width, line), Qt.AlignmentFlag.AlignVCenter, name)
        if count_text:
            painter.setPen(QColor("#5f6b78"))
            painter.drawText(QRectF(rect.right() - 8 - count_width, top, count_width, line),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, count_text)
        painter.setPen(QColor("#5f6b78"))
        detail_width = max(10, rect.right() - 8 - left)
        detail = metrics.elidedText(detail_text, Qt.TextElideMode.ElideRight, detail_width)
        painter.drawText(QRectF(left, top + line + 2, detail_width, line), Qt.AlignmentFlag.AlignVCenter,
                         detail)
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(0, max(SCAN_THUMB_SIZE + 14, option.fontMetrics.height() * 2 + 18))


class LayerItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(2, 2, -2, -2)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        palette = option.palette
        painter.fillRect(rect, palette.highlight() if selected else palette.base())
        text_color = palette.highlightedText().color() if selected else palette.text().color()
        if index.data(Qt.ItemDataRole.UserRole) == "+":
            # Drawn as a button (bordered, like Photoshop's New Layer), not as a text row.
            painter.fillRect(option.rect, palette.base())
            button = QRectF(rect.adjusted(2, 3, -2, -3))
            hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
            painter.setPen(QPen(QColor("#9aa5b1" if hovered else "#b8c0c9"), 1))
            painter.setBrush(QColor("#e3eaf2" if hovered else "#f3f5f8"))
            painter.drawRoundedRect(button, 4, 4)
            icon = QRectF(button.left() + 8, button.center().y() - 8, 16, 16)
            painter.setPen(QPen(QColor("#2f6db0"), 1.2))
            painter.setBrush(QColor("#ffffff"))
            painter.drawRoundedRect(icon, 2, 2)
            painter.setPen(QPen(QColor("#2f6db0"), 1.8))
            painter.drawLine(QPointF(icon.center().x(), icon.top() + 4), QPointF(icon.center().x(), icon.bottom() - 4))
            painter.drawLine(QPointF(icon.left() + 4, icon.center().y()), QPointF(icon.right() - 4, icon.center().y()))
            painter.setPen(palette.text().color())
            label = (index.data(Qt.ItemDataRole.DisplayRole) or "").replace("＋", "").strip()
            painter.drawText(button.adjusted(32, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter, label)
            painter.restore()
            return
        # QModelIndex.data() returns the integer 2 here, not a CheckState enum.
        # Draw the state explicitly so the check mark remains visible on all themes.
        checked = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value
        check = QRectF(rect.left() + 6, rect.center().y() - 8, 16, 16)
        painter.setPen(QPen(QColor("#175b98" if checked else "#66717d"), 1))
        painter.setBrush(QColor("#2377c9" if checked else "#ffffff"))
        painter.drawRoundedRect(check, 2, 2)
        if checked:
            tick = QPainterPath()
            tick.moveTo(check.left() + 3, check.top() + 8)
            tick.lineTo(check.left() + 7, check.top() + 12)
            tick.lineTo(check.left() + 13, check.top() + 4)
            painter.setPen(QPen(QColor("#ffffff"), 2.2,
                                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(tick)
        is_global = index.data(Qt.ItemDataRole.UserRole) == GLOBAL_LAYER_ID
        if is_global:
            eye = QRectF(rect.left() + 29, rect.center().y() - 5, 17, 10)
            painter.setPen(QPen(text_color, 1.3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(eye)
            painter.setBrush(text_color)
            painter.drawEllipse(QRectF(eye.center().x() - 2, eye.center().y() - 2, 4, 4))
            painter.setPen(QPen(QColor("#c43838"), 2))
            painter.drawLine(eye.bottomLeft(), eye.topRight())
            thumb = index.data(LAYER_RESULT_THUMB_ROLE)
            thumb_rect = QRectF(rect.left() + 54, rect.center().y() - 16, 32, 32)
            if isinstance(thumb, QPixmap):
                painter.drawPixmap(thumb_rect.toRect(), thumb)
            painter.setPen(text_color)
            painter.drawText(rect.adjusted(97, 0, -(STRENGTH_BOX_RIGHT + STRENGTH_BOX_WIDTH + 4), 0),
                             Qt.AlignmentFlag.AlignVCenter,
                             index.data(Qt.ItemDataRole.DisplayRole) or "global")
            strength = index.data(LAYER_STRENGTH_ROLE)
            editing = getattr(self.parent(), "_strength_edit_layer_id", None) == GLOBAL_LAYER_ID
            if strength is not None and not editing:
                strength_rect = _strength_box(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(strength_rect, 3, 3)
                painter.drawText(strength_rect, Qt.AlignmentFlag.AlignCenter, f"{strength}%")
            painter.restore()
            return
        eye = QRectF(rect.left() + 29, rect.center().y() - 5, 17, 10)
        painter.setPen(QPen(text_color, 1.3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(eye)
        painter.setBrush(text_color)
        painter.drawEllipse(QRectF(eye.center().x() - 2, eye.center().y() - 2, 4, 4))
        if not index.data(LAYER_OVERLAY_ROLE):
            painter.setPen(QPen(QColor("#c43838"), 2))
            painter.drawLine(eye.bottomLeft(), eye.topRight())
        for x, role, active_role in ((54, LAYER_RESULT_THUMB_ROLE, LAYER_RESULT_VIEW_ROLE),
                                     (92, LAYER_THUMB_ROLE, LAYER_MASK_VIEW_ROLE)):
            thumb_rect = QRectF(rect.left() + x, rect.center().y() - 16, 32, 32)
            thumb = index.data(role)
            if isinstance(thumb, QPixmap):
                painter.drawPixmap(thumb_rect.toRect(), thumb)
            painter.setPen(QPen(QColor("#2275ba") if index.data(active_role)
                                else palette.mid().color(), 2 if index.data(active_role) else 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(thumb_rect)
        painter.setPen(text_color)
        name_rect = rect.adjusted(135, 0, -(STRENGTH_BOX_RIGHT + STRENGTH_BOX_WIDTH + 4), 0)
        text = option.fontMetrics.elidedText(index.data(Qt.ItemDataRole.DisplayRole) or "",
                                             Qt.TextElideMode.ElideRight, name_rect.width())
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter, text)
        if getattr(self.parent(), "_strength_edit_layer_id", None) != index.data(Qt.ItemDataRole.UserRole):
            strength_rect = _strength_box(rect)
            painter.setPen(QPen(text_color, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(strength_rect, 3, 3)
            painter.drawText(strength_rect, Qt.AlignmentFlag.AlignCenter,
                             f"{index.data(LAYER_STRENGTH_ROLE) or 0}%")
        painter.setPen(text_color)
        painter.drawText(rect.adjusted(rect.width() - 26, 0, -5, 0),
                         Qt.AlignmentFlag.AlignCenter, "×")
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(0, max(40, option.fontMetrics.height() + 12))


class CompactRowDelegate(QStyledItemDelegate):
    """Fixed single-line rows, so a list can be sized to an exact number of rows."""

    def __init__(self, parent):
        super().__init__(parent)
        self.row_height = parent.fontMetrics().height() + 6

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        hint.setHeight(self.row_height)
        return hint


class FileDropList(QListWidget):
    dropped = Signal(list)
    deleteRequested = Signal()
    undoRequested = Signal()

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.dropped.emit([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.deleteRequested.emit()
            event.accept()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undoRequested.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class LayerList(QListWidget):
    addRequested = Signal()
    renameRequested = Signal()
    deleteRequested = Signal(str)
    maskRequested = Signal(str)
    resultRequested = Signal(str)
    overlayRequested = Signal(str)
    strengthRequested = Signal(str)
    strengthEdited = Signal(str, int)
    orderChanged = Signal()
    menuRequested = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setItemDelegate(LayerItemDelegate(self))
        self._strength_editor = QSpinBox(self.viewport())
        self._strength_editor.setRange(0, 200)
        self._strength_editor.setSingleStep(10)
        self._strength_editor.setSuffix("%")
        self._strength_editor.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._strength_editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Opaque, frameless and unpadded: it replaces the box exactly while editing
        # (the box itself is not drawn then), so nothing shows through or shifts.
        self._strength_editor.setFrame(False)
        self._strength_editor.setStyleSheet(
            "QSpinBox { border: 1px solid #2f6db0; border-radius: 3px; background: #ffffff;"
            " color: #1f2933; padding: 0px; margin: 0px;"
            " selection-background-color: #2f6db0; selection-color: #ffffff; }")
        self._strength_editor.editingFinished.connect(self._finish_strength_edit)
        self._strength_editor.hide()
        self._strength_edit_layer_id = None
        self._strength_wheel_target = None
        self._strength_wheel_delta = 0

    def edit_strength(self, item, value):
        if self._strength_edit_layer_id is not None:
            self._finish_strength_edit()
        rect = self.visualItemRect(item)
        self._strength_edit_layer_id = item.data(Qt.ItemDataRole.UserRole)
        self._strength_editor.setRange(0, int(item.data(LAYER_STRENGTH_MAX_ROLE) or 200))
        self._strength_editor.setValue(int(value))
        box = _strength_box(QRectF(rect).adjusted(2, 2, -2, -2)).toRect()
        self._strength_editor.setGeometry(box)
        self.viewport().update()
        self._strength_editor.show()
        self._strength_editor.setFocus()
        self._strength_editor.lineEdit().selectAll()

    def _finish_strength_edit(self):
        layer_id = self._strength_edit_layer_id
        if layer_id is None:
            return
        self._strength_edit_layer_id = None
        value = self._strength_editor.value()
        self._strength_editor.hide()
        self.viewport().update()
        self.strengthEdited.emit(layer_id, value)

    def wheelEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item and self._hit_part(item, event.position().toPoint()) == "strength":
            layer_id = item.data(Qt.ItemDataRole.UserRole)
            if layer_id != self._strength_wheel_target:
                self._strength_wheel_target = layer_id
                self._strength_wheel_delta = 0
            self._strength_wheel_delta += event.angleDelta().y()
            steps = int(self._strength_wheel_delta / 120)
            self._strength_wheel_delta -= steps * 120
            if steps:
                value = int(item.data(LAYER_STRENGTH_ROLE) or 0)
                maximum = int(item.data(LAYER_STRENGTH_MAX_ROLE) or 200)
                self.strengthEdited.emit(layer_id, max(0, min(maximum, value + steps * 10)))
            event.accept()
            return
        self._strength_wheel_target = None
        self._strength_wheel_delta = 0
        super().wheelEvent(event)

    def _hit_part(self, item, position):
        rect = self.visualItemRect(item)
        x = position.x() - rect.left()
        if item.data(Qt.ItemDataRole.UserRole) == GLOBAL_LAYER_ID:
            if (x >= rect.width() - STRENGTH_BOX_RIGHT - STRENGTH_BOX_WIDTH - 4
                    and item.data(LAYER_STRENGTH_ROLE) is not None):
                return "strength"
            return "visible" if x < 29 else "fixed_eye" if x < 53 else "name"
        if x < 29:
            return "visible"
        if x < 53:
            return "overlay"
        if x < 91:
            return "result"
        if x < 129:
            return "mask"
        if x >= rect.width() - 30:
            return "delete"
        if x >= rect.width() - STRENGTH_BOX_RIGHT - STRENGTH_BOX_WIDTH - 4:
            return "strength"
        return "name"

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if (event.button() == Qt.MouseButton.LeftButton and item
                and item.data(Qt.ItemDataRole.UserRole) == "+"):
            self.addRequested.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and item:
            layer_id = item.data(Qt.ItemDataRole.UserRole)
            part = self._hit_part(item, event.position().toPoint())
            if part == "visible":
                item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                                   else Qt.CheckState.Checked)
                event.accept()
                return
            if part == "mask":
                self.maskRequested.emit(layer_id)
                event.accept()
                return
            if part == "result":
                self.resultRequested.emit(layer_id)
                event.accept()
                return
            if part == "overlay":
                self.overlayRequested.emit(layer_id)
                event.accept()
                return
            if part == "strength":
                self.strengthRequested.emit(layer_id)
                event.accept()
                return
            if part == "delete":
                self.deleteRequested.emit(layer_id)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if (item and item.data(Qt.ItemDataRole.UserRole) != "+"
                and self._hit_part(item, event.position().toPoint()) == "name"):
            self.setCurrentItem(item)
            self.renameRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        item = self.currentItem()
        if (event.key() == Qt.Key.Key_Delete and item
                and item.data(Qt.ItemDataRole.UserRole) not in ("+", GLOBAL_LAYER_ID)):
            self.deleteRequested.emit(item.data(Qt.ItemDataRole.UserRole))
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        self.menuRequested.emit(event.globalPos(), self.itemAt(event.pos()))
        event.accept()

    def dropEvent(self, event):
        super().dropEvent(event)
        for row in range(self.count()):
            if self.item(row).data(Qt.ItemDataRole.UserRole) == "+":
                self.insertItem(0, self.takeItem(row))
                break
        for row in range(self.count()):
            if self.item(row).data(Qt.ItemDataRole.UserRole) == GLOBAL_LAYER_ID:
                self.addItem(self.takeItem(row))
                break
        self.orderChanged.emit()


class BatchPairDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Batch Import"))
        self.resize(760, 520)
        layout = QVBoxLayout(self)
        label = QLabel("Drop files into each list. Rows are paired by order, not by filename. Drag rows to reorder. Extra scans have no reference.")
        label.setWordWrap(True)
        layout.addWidget(label)
        lists = QHBoxLayout()
        self.scans = FileDropList()
        self.refs = FileDropList()
        for title, widget, callback in (("Scans", self.scans, self.add_scans),
                                        ("References", self.refs, self.add_references)):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            column.addWidget(widget, 1)
            browse = QPushButton(f"Add {title}...")
            browse.clicked.connect(lambda checked=False, fn=callback: fn())
            column.addWidget(browse)
            remove = QPushButton("Remove selected")
            remove.clicked.connect(lambda checked=False, target=widget: target.takeItem(target.currentRow())
                                   if target.currentRow() >= 0 else None)
            column.addWidget(remove)
            lists.addLayout(column, 1)
        self.scans.dropped.connect(self.add_scans)
        self.refs.dropped.connect(self.add_references)
        layout.addLayout(lists, 1)
        row = QHBoxLayout()
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        import_button = QPushButton("Import Pairs")
        import_button.clicked.connect(self.accept)
        row.addWidget(import_button)
        layout.addLayout(row)

    def _add(self, widget, paths):
        for path in paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            widget.addItem(item)

    def add_scans(self, paths=None):
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Choose scans", "", IMAGE_FILTER)
        self._add(self.scans, paths)

    def add_references(self, paths=None):
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, "Choose references", "", IMAGE_FILTER)
        self._add(self.refs, paths)

    def paths(self):
        return ([self.scans.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.scans.count())],
                [self.refs.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.refs.count())])


def _qimage_from_image(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    raw = rgba.tobytes("raw", "RGBA")
    qimage = QImage(raw, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888)
    return qimage.copy()


def _pixmap_from_array(array: np.ndarray) -> QPixmap:
    """QPixmap from an RGBA (H, W, 4) or grayscale (H, W) uint8 array, with one copy."""
    array = np.ascontiguousarray(array)
    height, width = array.shape[:2]
    if array.ndim == 2:
        image = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
    else:
        image = QImage(array.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(image)  # converts (copies) before ``array`` is released


def _pixmap_from_image(image: Image.Image) -> QPixmap:
    return QPixmap.fromImage(_qimage_from_image(image))


class _MainThreadCall(QObject):
    """Runs callables posted from worker threads on the GUI thread."""
    call = Signal(object)

    def __init__(self):
        super().__init__()
        self.call.connect(lambda function: function())


def _thumbnail_canvas(image: Image.Image, mode: str, fill) -> QImage:
    image.thumbnail((32, 32), Image.Resampling.BILINEAR)
    canvas = Image.new(mode, (32, 32), fill)
    canvas.paste(image, ((32 - image.width) // 2, (32 - image.height) // 2))
    return _qimage_from_image(canvas.convert("RGB"))


def _mask_thumbnail_image(mask_path: Path, mask_array: np.ndarray | None = None):
    """A layer's mask thumbnail (QImage), cropped to its painted area, and that area.

    Returns ``(image, box, size)``: ``box`` is the crop rectangle (None while nothing
    is painted) and ``size`` the mask size. ``mask_array`` is the mask being edited
    (newer than the file). Safe off the GUI thread.
    """
    blank = _qimage_from_image(Image.new("RGB", (32, 32), 0))
    if mask_array is not None:
        mask = Image.fromarray(mask_array, "L")
    elif mask_path.exists():
        with Image.open(mask_path) as source:
            mask = source.convert("L")
    else:
        return blank, None, None
    box = mask.getbbox()
    if box is None:
        return blank, None, mask.size  # nothing painted yet
    margin = max(3, round(max(box[2] - box[0], box[3] - box[1]) * .05))
    box = (max(0, box[0] - margin), max(0, box[1] - margin),
           min(mask.width, box[2] + margin), min(mask.height, box[3] + margin))
    return _thumbnail_canvas(mask.crop(box), "L", 0), box, mask.size


def _result_thumbnail_images(result_path: Path, crops: dict) -> dict:
    """Result thumbnails of several layers from one decode of the result image.

    ``crops`` maps a cache key to ``(box, mask_size)``; returns key -> QImage.
    Safe off the GUI thread.
    """
    with Image.open(result_path) as source:
        result = source.convert("RGB")
    images = {}
    for key, (box, mask_size) in crops.items():
        region = result.crop(box) if tuple(mask_size) == result.size else result
        images[key] = _thumbnail_canvas(region, "RGB", "#c9c9c9")
    return images


def _scan_thumbnail_image(path: Path) -> QImage:
    """A small square thumbnail of a scan for the scan list. Safe off the GUI thread."""
    size = SCAN_THUMB_SIZE * 2  # sharp on high-DPI screens
    with Image.open(path) as source:
        source.draft("RGB", (size, size))
        image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), "#dfe3e8")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return _qimage_from_image(canvas)


def _global_thumbnail_image(path: Path) -> QImage:
    with Image.open(path) as source:
        return _thumbnail_canvas(source.convert("RGB"), "RGB", "#c9c9c9")


class MaskPreviewThread(QThread):
    """Decode a layer mask and reduce it to the preview size off the GUI thread."""
    ready = Signal(object, object)

    def __init__(self, path: Path, key, source_size):
        super().__init__()
        self.path, self.key, self.source_size = path, key, source_size

    def run(self):
        try:
            with Image.open(self.path) as image:
                mask = image.convert("L")
                if mask.size != self.source_size:
                    return
                mask.thumbnail((1800, 1800), Image.Resampling.BILINEAR)
            self.ready.emit(self.key, np.asarray(mask, dtype=np.uint8).copy())
        except Exception:
            pass  # the overlay simply stays empty


QUICK_PREVIEW_SIDE = 2000
# On Windows, paint and timer messages wait until the input queue is empty, so a mouse
# that never stops would starve them. Brush feedback is therefore drawn synchronously
# from the move handler itself, at most once per frame.
FRAME_SECONDS = 1 / 60
# The brush ring alone is cheap to redraw (only the area around it), so it may follow
# the mouse faster than full frames: the system cursor of other tools moves at any rate.
CURSOR_FRAME_SECONDS = 1 / 120
# Up to this size (device pixels) the brush ring *is* the mouse cursor, moved by the
# system like any cursor: it can never lag. Larger rings are drawn in the scene.
BRUSH_CURSOR_MAX_PIXELS = 1024


def _image_key(path: Path | None, virtual_size=None, alignment_path: Path | None = None):
    """Identity of a displayed image: its file, and for a reference its alignment."""
    metadata = path.stat() if path and path.exists() else None
    alignment_meta = alignment_path.stat() if alignment_path and alignment_path.exists() else None
    return (str(path) if path else None, virtual_size,
            metadata.st_mtime_ns if metadata else None, metadata.st_size if metadata else None,
            str(alignment_path) if alignment_path else None,
            alignment_meta.st_mtime_ns if alignment_meta else None)


def _quick_preview_file(directory: Path | None, key) -> Path | None:
    if directory is None or key[0] is None or key[2] is None:
        return None
    return Path(directory) / (hashlib.sha1(repr(key).encode("utf-8")).hexdigest() + ".img")


def _build_preview(path: Path, alignment_path: Path | None = None,
                   virtual_size: tuple[int, int] | None = None) -> np.ndarray:
    """RGBA preview (longest side PREVIEW_MAX_SIDE), aligned to the scan when asked."""
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if alignment_path and virtual_size:
        width, height = virtual_size
        scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        homography = np.load(alignment_path)
        output_scale = np.diag([target[0] / width, target[1] / height, 1.0])
        return cv2.warpPerspective(rgba, output_scale @ homography, target,
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
    height, width = rgba.shape[:2]
    scale = min(1.0, PREVIEW_MAX_SIDE / max(width, height))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    if target != (width, height):
        transform = np.diag([target[0] / width, target[1] / height, 1.0])
        rgba = cv2.warpPerspective(rgba, transform, target, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return rgba


def _save_quick_preview(rgba: np.ndarray, destination: Path) -> None:
    """A small, fast-to-decode stand-in shown while the full preview loads."""
    height, width = rgba.shape[:2]
    scale = min(1.0, QUICK_PREVIEW_SIDE / max(width, height))
    if scale < 1.0:
        rgba = cv2.resize(rgba, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if np.all(rgba[:, :, 3] == 255):
        Image.fromarray(rgba[:, :, :3], "RGB").save(temporary, "JPEG", quality=95, subsampling=0)
    else:
        Image.fromarray(rgba, "RGBA").save(temporary, "PNG", compress_level=1)
    engine.replace_file(temporary, destination)


def _load_quick_preview(file: Path | None) -> QPixmap | None:
    if file is None or not file.exists():
        return None
    image = QImage(str(file))
    return None if image.isNull() else QPixmap.fromImage(image)


class PreviewThread(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, alignment_path: Path | None = None,
                 virtual_size: tuple[int, int] | None = None, quick_file: Path | None = None):
        super().__init__()
        self.path = path
        self.alignment_path = alignment_path
        self.virtual_size = virtual_size
        self.quick_file = quick_file

    def run(self):
        try:
            rgba = _build_preview(self.path, self.alignment_path, self.virtual_size)
            self.ready.emit(_qimage_from_image(Image.fromarray(rgba, "RGBA")))
            if self.quick_file is not None and not self.quick_file.exists():
                try:
                    _save_quick_preview(rgba, self.quick_file)
                except OSError:
                    pass  # only a speed-up
        except Exception as exc:
            self.failed.emit(str(exc))


class QuickPreviewPrefetch(QThread):
    """Prepare quick previews of every scan in the background, once per image."""

    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs  # (path, alignment_path, virtual_size, quick_file)
        self.stop = False

    def run(self):
        for path, alignment_path, virtual_size, quick_file in self.jobs:
            if self.stop:
                return
            if quick_file is None or quick_file.exists() or not path.exists():
                continue
            try:
                _save_quick_preview(_build_preview(path, alignment_path, virtual_size), quick_file)
            except Exception:
                continue


class LocalPreviewThread(QThread):
    """Compose fitted layers at screen-preview size while the exact PNG renders."""
    ready = Signal(object)

    def __init__(self, base_path: Path, layers: list[dict], directory: Path, cache: dict,
                 preload_only: bool = False, mask_overrides: dict | None = None,
                 patch_cache: "engine.PatchCache | None" = None):
        super().__init__()
        self.base_path = base_path
        self.layers = layers
        self.directory = directory
        self.cache = cache
        self.preload_only = preload_only
        self.mask_overrides = mask_overrides or {}
        self.patch_cache = patch_cache

    def _image_key(self, path: Path, size: tuple[int, int] | None):
        override = self.mask_overrides.get(path) if size else None
        if override is not None:
            return (str(path), "resident", override[0], size)
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size, size)

    def _cached_image(self, path: Path, size: tuple[int, int] | None):
        override = self.mask_overrides.get(path) if size else None
        key = self._image_key(path, size)
        if key not in self.cache:
            if override is not None:
                source_mask = override[1]
                pixels = (cv2.resize(source_mask, size, interpolation=cv2.INTER_LINEAR)
                          if (source_mask.shape[1], source_mask.shape[0]) != size
                          else source_mask)
            else:
                with Image.open(path) as source:
                    image = source.convert("L" if size else "RGBA")
                    if size:
                        image = image.resize(size, Image.Resampling.BILINEAR)
                        pixels = np.asarray(image, dtype=np.uint8).copy()
                    else:
                        width, height = image.size
                        scale = min(1.0, LOCAL_FAST_PREVIEW_SIDE / max(width, height))
                        target = (max(1, round(width * scale)), max(1, round(height * scale)))
                        pixels = np.asarray(image, dtype=np.uint8)
                        if target != (width, height):
                            transform = np.diag([target[0] / width, target[1] / height, 1.0])
                            pixels = cv2.warpPerspective(pixels, transform, target,
                                                         flags=cv2.INTER_LINEAR,
                                                         borderMode=cv2.BORDER_REPLICATE)
                        pixels = pixels.copy()
            self.cache[key] = pixels
            while len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def run(self):
        try:
            base = self._cached_image(self.base_path, None)
            size = (base.shape[1], base.shape[0])
            if self.preload_only:
                for layer in self.layers:
                    self._cached_image(self.directory / layer["mask"], size)
            visible = [layer for layer in self.layers if layer.get("visible", True)]
            source_width, _ = engine.validate_image(str(self.base_path))
            for layer in visible:
                if not layer.get("soft_mask", False):
                    layer["feather"] = max(0, round(layer["feather"] * size[0] / source_width))

            def load_mask(layer):
                path = self.directory / layer["mask"]
                return self._cached_image(path, size), self._image_key(path, size)

            current = engine.compose_layers(base[:, :, :3], visible, load_mask,
                                            self._image_key(self.base_path, None),
                                            self.patch_cache)
            if self.preload_only:
                return  # warm-up: the per-layer patches are now cached
            rgba = np.dstack((current, base[:, :, 3]))
            self.ready.emit(_qimage_from_image(Image.fromarray(rgba, "RGBA")))
        except Exception:
            # A fast preview is optional; the exact background render remains authoritative.
            pass


LIVE_TILE = 256


class LiveStrokePreview:
    """Recompose the corrected result live, tile by tile, while a layer mask is painted.

    Tiles touched by new brush tips are recomputed at about the displayed resolution
    (never finer than the fast preview), and only while they are on screen: all visible
    layers are re-stacked there with the live mask of the edited layer.
    """

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.context = None
        self.dirty: set = set()
        self._flush_pending = False
        self._serial = 0

    @property
    def view(self) -> "ImageView":
        return self.window.result_view

    def begin(self) -> bool:
        self.context = None
        self.dirty.clear()
        w, view = self.window, self.view
        project, scan, layer = w.project, w._current_scan(), w._current_layer()
        if (project is None or scan is None or layer is None or not project.is_current()
                or w.current_scan_id != scan["id"] or view.source_size == (0, 0)):
            return False
        run = project.run_dir()
        if (run is None or not layer.get("model") or layer.get("needs_refit", False)
                or not layer.get("visible", True) or layer.get("model_run") != run.name):
            return False
        base_path = (run / "base" / f"{scan['id']}.png" if scan.get("global_visible", True)
                     else Path(scan["path"]))
        if not base_path.exists():
            return False
        stat = base_path.stat()
        base_key = (str(base_path), stat.st_mtime_ns, stat.st_size, None)
        preview = w._fast_preview_cache.get(base_key)
        if preview is None:
            # The preview-size base is normally warm; prepare it for the next stroke.
            w._queue_fast_local_preview(scan, warm=True)
            return False
        # Work at the displayed resolution, in power-of-two steps below the preview.
        preview_scale = preview.shape[1] / view.source_size[0]
        display_scale = abs(view.transform().m11()) * view.devicePixelRatioF()
        level = 0
        while level < 6 and preview_scale / 2 ** (level + 1) >= display_scale:
            level += 1
        size = (max(1, round(preview.shape[1] / 2 ** level)),
                max(1, round(preview.shape[0] / 2 ** level)))
        base = self._resized(base_key, preview, size, cv2.INTER_AREA)
        layers, masks = [], {}
        for entry in scan["layers"]:
            if not entry.get("visible", True):
                continue
            item = copy.deepcopy(entry)
            if item["id"] == layer["id"]:
                item["soft_mask"] = True  # the live mask preview already holds the soft edge
            else:
                if not item.get("model"):
                    continue  # not fitted yet: no effect on the result
                if item.get("needs_refit", False):
                    return False
                if not item.get("soft_mask", False):
                    item["feather"] = max(0, round(item["feather"] * size[0] / view.source_size[0]))
                masks[item["id"]] = self._layer_mask(project.directory / item["mask"], size)
                if masks[item["id"]] is None:
                    return False
            layers.append(item)
        self._serial += 1
        self.context = {"base": base, "size": size, "layers": layers, "masks": masks,
                        "current": layer["id"], "scale": size[0] / view.source_size[0],
                        "level": level, "z": 1 + (self._serial % 1000) * 1e-4}
        return True

    def _resized(self, key, array, size, interpolation):
        if (array.shape[1], array.shape[0]) == size:
            return array
        cache = self.window._fast_preview_cache
        resized_key = ("live", key, size)
        if resized_key not in cache:
            cache[resized_key] = cv2.resize(array, size, interpolation=interpolation)
        return cache[resized_key]

    def _layer_mask(self, path: Path, size):
        w = self.window
        if path in w._resident_masks:
            key = (str(path), "resident", w._mask_revisions[path])
            return self._resized(key, w._resident_masks[path], size, cv2.INTER_AREA)
        if not path.exists():
            return None
        stat = path.stat()
        key = ("live-mask", str(path), stat.st_mtime_ns, stat.st_size, size)
        cache = w._fast_preview_cache
        if key not in cache:
            with Image.open(path) as image:
                cache[key] = np.asarray(image.convert("L").resize(size, Image.Resampling.BOX),
                                        np.uint8).copy()
        return cache[key]

    def update(self, box):
        """Mark tiles under ``box`` (x0, y0, x1, y1 in source pixels) for recomposition."""
        if self.context is None:
            return
        for tile in self._tiles_in(box):
            self.dirty.add(tile)
        if not self._flush_pending:
            self._flush_pending = True
            QTimer.singleShot(0, self.flush)

    def _tiles_in(self, box):
        scale = self.context["scale"]
        width, height = self.context["size"]
        x0, y0 = max(0, int(box[0] * scale)), max(0, int(box[1] * scale))
        x1 = min(width, int(np.ceil(box[2] * scale)) + 1)
        y1 = min(height, int(np.ceil(box[3] * scale)) + 1)
        if x0 >= x1 or y0 >= y1:
            return set()
        return {(tx, ty) for ty in range(y0 // LIVE_TILE, (y1 - 1) // LIVE_TILE + 1)
                for tx in range(x0 // LIVE_TILE, (x1 - 1) // LIVE_TILE + 1)}

    def flush(self):
        """Render dirty tiles that are on screen; off-screen ones wait until they are."""
        self._flush_pending = False
        if self.context is None or self.view._mask_preview is None:
            self.dirty.clear()
            return
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        on_screen = self._tiles_in((visible.left(), visible.top(), visible.right(), visible.bottom()))
        tiles = self.dirty & on_screen
        self.dirty -= tiles
        for tile in sorted(tiles):
            self._render_tile(*tile)

    def _render_tile(self, tx, ty):
        context = self.context
        base, (width, height) = context["base"], context["size"]
        x0, y0 = tx * LIVE_TILE, ty * LIVE_TILE
        x1, y1 = min(width, x0 + LIVE_TILE), min(height, y0 + LIVE_TILE)
        pad = 4 + max((0 if layer.get("soft_mask", False) else int(layer["feather"]))
                      for layer in context["layers"])
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(width, x1 + pad), min(height, y1 + pad)
        live = self.view._mask_preview
        sx, sy = live.shape[1] / width, live.shape[0] / height
        to_live = np.array([[sx, 0, (cx0 + .5) * sx - .5], [0, sy, (cy0 + .5) * sy - .5]])
        current_mask = cv2.warpAffine(live, to_live, (cx1 - cx0, cy1 - cy0),
                                      flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE)

        def load_mask(layer):
            if layer["id"] == context["current"]:
                return current_mask, None
            return context["masks"][layer["id"]][cy0:cy1, cx0:cx1], None

        crop = base[cy0:cy1, cx0:cx1]
        composed = engine.compose_layers(crop[:, :, :3], context["layers"], load_mask, None)
        inner = (slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))
        rgba = np.ascontiguousarray(np.dstack((composed[inner], crop[:, :, 3][inner])))
        self.view.set_live_tile((context["level"], tx, ty),
                                _qimage_from_image(Image.fromarray(rgba, "RGBA")),
                                x0 / context["scale"], y0 / context["scale"],
                                1 / context["scale"], context["z"])

    def end(self, cancelled=False):
        if self.context is not None and not cancelled:
            self.flush()
        self.context = None
        self.dirty.clear()
        if cancelled:
            self.view.clear_live_tiles()


class ImageView(QGraphicsView):
    stroke = Signal(list, bool)
    polygonSelected = Signal(list, int, int)
    modeChanged = Signal(str)
    viewChanged = Signal(object, object, object)
    imageFailed = Signal(str)
    dropped = Signal(list)
    brushAdjusted = Signal(int, int)
    brushAdjustmentFinished = Signal()
    # Emitted after the view re-fits itself (for example when a preview arrives).
    refitted = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.image_item)
        self.original_item = QGraphicsPixmapItem()
        # Space comparison must use the same interpolation as the result;
        # Qt's default fast mode shifts the apparent edges of fine line art.
        self.original_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # Comparing must cover every correction and editing overlay, not only
        # the global image item beneath local masks and strokes.
        self.original_item.setZValue(20)
        self.original_item.hide()
        self.scene().addItem(self.original_item)
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setZValue(5)
        self.mask_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.mask_item)
        self.mask_display_item = QGraphicsPixmapItem()
        self.mask_display_item.setZValue(5.5)
        self.mask_display_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.mask_display_item.hide()
        self.scene().addItem(self.mask_display_item)
        self.path_item = QGraphicsPathItem()
        self.path_item.setZValue(7)
        self.path_item.setPen(QPen(QColor(255, 110, 70, 180), 60,
                                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                   Qt.PenJoinStyle.RoundJoin))
        self.scene().addItem(self.path_item)
        self.polygon_item = QGraphicsPathItem()
        polygon_pen = QPen(QColor("#47d6ff"), 2)
        polygon_pen.setCosmetic(True)
        self.polygon_item.setPen(polygon_pen)
        self.polygon_item.setZValue(8)
        self.scene().addItem(self.polygon_item)
        self.polygon_start = QGraphicsEllipseItem(-5, -5, 10, 10)
        self.polygon_start.setPen(QPen(QColor("#10232d"), 2))
        self.polygon_start.setBrush(QColor("#47d6ff"))
        self.polygon_start.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.polygon_start.setZValue(8.5)
        self.polygon_start.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene().addItem(self.polygon_start)
        self.polygon_start.hide()
        close_pixmap = QPixmap(24, 24)
        close_pixmap.fill(Qt.GlobalColor.transparent)
        close_painter = QPainter(close_pixmap)
        close_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        close_painter.setPen(QPen(QColor("#10232d"), 3))
        close_painter.drawEllipse(3, 3, 18, 18)
        close_painter.setPen(QPen(QColor("#ffffff"), 1.5))
        close_painter.drawEllipse(3, 3, 18, 18)
        close_painter.end()
        self._polygon_close_cursor = QCursor(close_pixmap, 12, 12)
        self.brush_cursor = QGraphicsEllipseItem()
        cursor_pen = QPen(QColor("#15191e"), 1, Qt.PenStyle.SolidLine)
        cursor_pen.setCosmetic(True)
        self.brush_cursor.setPen(cursor_pen)
        self.brush_cursor.setBrush(Qt.BrushStyle.NoBrush)
        self.brush_cursor.setZValue(9)
        self.brush_cursor.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene().addItem(self.brush_cursor)
        self.brush_cursor.hide()
        self._ring_cursor_active = False
        self._ring_cursor_key = None
        self.adjust_preview = QGraphicsEllipseItem()
        self.adjust_preview.setPen(Qt.PenStyle.NoPen)
        self.adjust_preview.setZValue(10)
        self.adjust_preview.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene().addItem(self.adjust_preview)
        self.adjust_preview.hide()
        self.adjust_ring = QGraphicsEllipseItem()
        ring_pen = QPen(QColor("#151515"), 1)
        ring_pen.setCosmetic(True)
        self.adjust_ring.setPen(ring_pen)
        self.adjust_ring.setBrush(Qt.BrushStyle.NoBrush)
        self.adjust_ring.setZValue(10.1)
        self.adjust_ring.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene().addItem(self.adjust_ring)
        self.adjust_ring.hide()
        self.adjust_label = QLabel(self.viewport())
        self.adjust_label.setStyleSheet(
            "QLabel { color: white; background: #111111; padding: 5px 8px; border-radius: 3px; }")
        self.adjust_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.adjust_label.hide()
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#25282e"))
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.source_size = (0, 0)
        self.mode = "pan"
        self.points = []
        self.polygon_points: list[QPointF] = []
        self._syncing = False
        self._middle_pan_at = None
        self._pan_dragging = False
        self._brush_adjust_start = None
        self._brush_adjust_scene = None
        self._brush_adjust_local = None
        self._brush_adjust_axis = None
        self._brush_adjust_pending = QPointF()
        self._brush_adjust_reference = QPointF()
        self._brush_adjust_rebase = False
        self._brush_adjust_radius = 0.0
        self._brush_adjust_hardness = 0.0
        self._current_image_key = None
        self._temporary_preview_active = False
        self._preview_cache = {}
        self._image_pending = {}
        self._mask_key = None
        self._original_key = None
        self._original_generation = 0
        self._compare = False
        self.show_mask = False
        self.overlay_color = (255, 100, 70)
        self.view_mask = False
        self._mask_preview = None
        self._brush_sample_image = None
        self._brush_cursor_light = None
        self._mask_preview_base = None
        # Live stroke state: accumulated tip coverage at mask-preview size, how many
        # spaced stamps it already holds, and the overlay flags to restore on release.
        self._stroke_coverage = None
        self._stroke_applied = 0
        self._stroke_overlay_saved = None
        self._stroke_tail_box = None
        self._stroke_preview = None
        # Optional LiveStrokePreview: recomposes the corrected result while painting.
        self.live_stroke = None
        # Optional callable run before a brush stroke starts; returning False cancels it.
        # The window uses it to create the layer first, so the stroke can be shown live.
        self.before_stroke = None
        self._live_active = False
        self._live_tiles: dict = {}
        self._last_frame = 0.0
        self._last_cursor_frame = 0.0
        self._cursor_viewport_rect = None
        self._stroke_timer = QTimer(self)
        self._stroke_timer.setSingleShot(True)
        self._stroke_timer.setInterval(15)
        self._stroke_timer.timeout.connect(self._flush_stroke)
        self._overlay_other = None
        self._overlay_selected = False
        self._overlay_cache = {}
        self._mask_preview_cache = {}
        self.quick_cache_dir: Path | None = None  # set by the window per project
        self.brush_radius = 50
        self.brush_hardness = 80
        self._preview_generation = 0
        self._preview_workers: list[PreviewThread] = []
        self._view_touched = False
        # True only after the user zoomed or panned (here or in the synced view);
        # keeping a view across refreshes does not count.
        self._user_view = False
        self._fit_serial = 0  # counts programmatic fits, so callers can tell the view moved
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.set_brush_radius(50)

    def set_image(self, path: Path | None, virtual_size: tuple[int, int] | None = None,
                  preserve_view: bool = False, alignment_path: Path | None = None):
        metadata = path.stat() if path and path.exists() else None
        key = _image_key(path, virtual_size, alignment_path)
        # A temporary preview or live tiles may cover the image recorded for this key;
        # asking for it again must bring the real pixels back.
        if (key == self._current_image_key and not self._temporary_preview_active
                and not self._live_tiles):
            return
        self._current_image_key = key
        self._temporary_preview_active = False
        self._preview_generation += 1
        if not self.points:
            self.clear_live_tiles()
        old_size = self.source_size
        previous_pixmap = self.image_item.pixmap()
        self.source_size = (0, 0)
        self._view_touched = False
        if metadata is None:
            self.image_item.setPixmap(QPixmap())
            self.mask_item.setPixmap(QPixmap())
            self.mask_display_item.setPixmap(QPixmap())
            self._mask_preview = None
            self._mask_key = None
            self._overlay_other = None
            self._overlay_selected = False
            self.scene().setSceneRect(QRectF(0, 0, 1, 1))
            return
        with Image.open(path) as image:
            self.source_size = virtual_size or image.size
        keep_view = preserve_view and old_size == self.source_size
        if old_size != self.source_size:
            self.mask_item.setPixmap(QPixmap())
            self.mask_display_item.setPixmap(QPixmap())
            self._mask_preview = None
            self._mask_key = None
            self._overlay_other = None
            self._overlay_selected = False
        if not keep_view or previous_pixmap.isNull():
            self.image_item.setPixmap(QPixmap())
        if keep_view:
            # Re-fitting and then re-centering rounds through integer scrollbars.
            # A repeated layer toggle would otherwise move the view by a pixel each time.
            self._view_touched = True
        else:
            self._user_view = False
            self._syncing = True
            self.scene().setSceneRect(QRectF(0, 0, *self.source_size))
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._syncing = False
            self._fit_serial += 1
            self._apply_brush_cursor()  # the ring's on-screen size follows the zoom
        cached = self._preview_cache.get(key)
        if cached is not None:
            self._set_preview_pixmap(cached)
            return
        quick_file = _quick_preview_file(self.quick_cache_dir, key)
        quick = _load_quick_preview(quick_file)
        if quick is not None:
            # Show the prepared small preview at once; the full one replaces it shortly.
            self._set_preview_pixmap(quick)
        if key in self._image_pending:
            return
        worker = PreviewThread(path, alignment_path, self.source_size if alignment_path else None,
                               quick_file)
        self._image_pending[key] = worker
        self._preview_workers.append(worker)
        worker.ready.connect(lambda preview, image_key=key:
                             self._image_preview_ready(preview, image_key))
        worker.failed.connect(self.imageFailed)
        worker.finished.connect(lambda target=worker, image_key=key:
                                self._image_preview_finished(target, image_key))
        worker.start()

    def _image_preview_ready(self, preview: QImage, key):
        pixmap = QPixmap.fromImage(preview)
        self._cache_preview(key, pixmap)
        if (key == self._current_image_key and self.source_size != (0, 0)
                and not self._temporary_preview_active):
            self._set_preview_pixmap(pixmap)

    def set_live_tile(self, key, image: QImage, x: float, y: float, scale: float,
                      z: float = 1):
        item = self._live_tiles.get(key)
        if item is None:
            item = QGraphicsPixmapItem()
            item.setZValue(z)  # above the result, below mask overlays and tools
            item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.scene().addItem(item)
            self._live_tiles[key] = item
        item.setPixmap(QPixmap.fromImage(image))
        item.setTransform(QTransform.fromScale(scale, scale))
        item.setPos(x, y)
        item.setVisible(not self._compare)

    def clear_live_tiles(self):
        for item in self._live_tiles.values():
            self.scene().removeItem(item)
        self._live_tiles.clear()

    def show_temporary_preview(self, preview: QImage):
        if self.source_size == (0, 0):
            return
        self._temporary_preview_active = True
        self._view_touched = True
        self._set_preview_pixmap(QPixmap.fromImage(preview))

    def adopt_current_preview(self, path: Path):
        """Associate the resident pixels with a saved result without reloading it."""
        if self.source_size == (0, 0) or self.image_item.pixmap().isNull() or not path.exists():
            return
        key = _image_key(path)
        self._current_image_key = key
        self._temporary_preview_active = False
        self._preview_generation += 1
        self._cache_preview(key, self.image_item.pixmap())

    def _image_preview_finished(self, worker: PreviewThread, key):
        if self._image_pending.get(key) is worker:
            self._image_pending.pop(key, None)
        self._preview_finished(worker)

    def _set_preview_pixmap(self, pixmap: QPixmap):
        if not self.points:
            self.clear_live_tiles()
        self.image_item.setPixmap(pixmap)
        self._brush_sample_image = pixmap.toImage()
        self.image_item.setScale(1)
        self.image_item.setTransform(QTransform.fromScale(
            self.source_size[0] / pixmap.width(), self.source_size[1] / pixmap.height()))
        if not self._view_touched:
            self._syncing = True
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._syncing = False
            self._fit_serial += 1
            self._apply_brush_cursor()
            self.refitted.emit()

    def _preview_ready(self, preview: QImage, generation: int, key=None):
        pixmap = QPixmap.fromImage(preview)
        if key is not None:
            self._cache_preview(key, pixmap)
        if generation != self._preview_generation or self.source_size == (0, 0):
            return
        self._set_preview_pixmap(pixmap)

    def _cache_preview(self, key, pixmap: QPixmap):
        self._preview_cache.pop(key, None)
        self._preview_cache[key] = pixmap
        while len(self._preview_cache) > 8:
            self._preview_cache.pop(next(iter(self._preview_cache)))

    def preload_image(self, path: Path):
        if not path.exists():
            return
        key = _image_key(path)
        if key in self._preview_cache or key in self._image_pending:
            return
        worker = PreviewThread(path, quick_file=_quick_preview_file(self.quick_cache_dir, key))
        self._image_pending[key] = worker
        self._preview_workers.append(worker)
        worker.ready.connect(lambda preview, image_key=key:
                             self._image_preview_ready(preview, image_key))
        worker.finished.connect(lambda target=worker, image_key=key:
                                self._image_preview_finished(target, image_key))
        worker.start()

    def set_original(self, path: Path | None):
        metadata = path.stat() if path and path.exists() else None
        key = (str(path) if path else None,
               metadata.st_mtime_ns if metadata else None, metadata.st_size if metadata else None)
        if key == self._original_key:
            return
        self._original_key = key
        self._original_generation += 1
        generation = self._original_generation
        self.original_item.setPixmap(QPixmap())
        self.original_item.hide()
        if metadata is None:
            return
        image_key = _image_key(path)
        cached = self._preview_cache.get(image_key)
        if cached is not None:
            self._original_ready(cached.toImage(), generation)
            return
        quick_file = _quick_preview_file(self.quick_cache_dir, image_key)
        quick = _load_quick_preview(quick_file)
        if quick is not None:
            self._original_ready(quick.toImage(), generation)
        worker = PreviewThread(path, quick_file=quick_file)
        worker.ready.connect(lambda preview, cache_key=image_key:
                             self._cache_preview(cache_key, QPixmap.fromImage(preview)))
        self._preview_workers.append(worker)
        worker.ready.connect(lambda preview, token=generation: self._original_ready(preview, token))
        worker.failed.connect(self.imageFailed)
        worker.finished.connect(lambda target=worker: self._preview_finished(target))
        worker.start()

    def _original_ready(self, preview: QImage, generation: int):
        if generation != self._original_generation or self.source_size == (0, 0):
            return
        self.original_item.setPixmap(QPixmap.fromImage(preview))
        self.original_item.setScale(1)
        self.original_item.setTransform(QTransform.fromScale(
            self.source_size[0] / preview.width(), self.source_size[1] / preview.height()))
        self.original_item.setVisible(self._compare)

    def set_compare(self, pressed: bool):
        self._compare = pressed
        self.original_item.setVisible(pressed and not self.original_item.pixmap().isNull())
        for item in self._live_tiles.values():
            item.setVisible(not pressed)
        self._update_mask_visibility()

    def _update_mask_visibility(self):
        self.mask_item.setVisible(not self._compare and self.show_mask and not self.view_mask)
        self.mask_display_item.setVisible(not self._compare and self.view_mask)

    def set_show_mask(self, visible: bool):
        self.show_mask = visible
        self._update_mask_visibility()

    def set_view_mask(self, visible: bool):
        if visible:
            self.ensure_mask_preview()
        self.view_mask = visible
        self._render_mask_preview()

    def set_brush_radius(self, radius: int):
        self.brush_radius = radius
        pen = self.path_item.pen()
        pen.setWidthF(radius * 2 * (2 - self.brush_hardness / 100))
        self.path_item.setPen(pen)
        self.brush_cursor.setRect(-radius, -radius, radius * 2, radius * 2)
        if self.mode in {"paint", "erase"} and self._middle_pan_at is None:
            self._apply_brush_cursor()
        self.set_brush_hardness(self.brush_hardness)

    def set_brush_hardness(self, hardness: int):
        self.brush_hardness = max(0, min(100, hardness))
        radius = self.brush_radius
        feather = radius * (100 - self.brush_hardness) / 100
        outer = radius + feather
        inner = radius - feather
        pen = self.path_item.pen()
        pen.setWidthF(outer * 2)
        self.path_item.setPen(pen)
        # The adjustment preview shows the brush's coverage itself: fully opaque where
        # the brush paints 100%, fading with the feather (painted masks stay translucent).
        gradient = QRadialGradient(QPointF(0, 0), outer)
        gradient.setColorAt(0, QColor(255, 20, 20, 255))
        if feather:
            if inner:
                gradient.setColorAt(inner / outer, QColor(255, 20, 20, 255))
            gradient.setColorAt(radius / outer, QColor(255, 20, 20, 128))
            gradient.setColorAt(1, QColor(255, 20, 20, 0))
        else:
            gradient.setColorAt(1, QColor(255, 20, 20, 255))
        self.adjust_preview.setRect(-outer, -outer, outer * 2, outer * 2)
        self.adjust_preview.setBrush(QBrush(gradient))
        self.adjust_ring.setRect(-radius, -radius, radius * 2, radius * 2)

    def _set_brush_cursor_position(self, point: QPointF):
        self.brush_cursor.setPos(point)
        if self.source_size == (0, 0):
            return
        x = min(max(int(point.x()), 0), self.source_size[0] - 1)
        y = min(max(int(point.y()), 0), self.source_size[1] - 1)
        if self.view_mask and self._mask_preview is not None:
            mask = self._mask_preview
            brightness = int(mask[min(mask.shape[0] - 1, y * mask.shape[0] // self.source_size[1]),
                                  min(mask.shape[1] - 1, x * mask.shape[1] // self.source_size[0])])
        elif self._brush_sample_image is not None and not self._brush_sample_image.isNull():
            sample = self._brush_sample_image
            color = sample.pixelColor(min(sample.width() - 1, x * sample.width() // self.source_size[0]),
                                      min(sample.height() - 1, y * sample.height() // self.source_size[1]))
            brightness = (color.red() * 299 + color.green() * 587 + color.blue() * 114) // 1000
        else:
            return
        light = brightness < 128
        if light != self._brush_cursor_light:
            self._brush_cursor_light = light
            pen = self.brush_cursor.pen()
            pen.setColor(QColor("#ffffff" if light else "#15191e"))
            self.brush_cursor.setPen(pen)

    def _show_brush_cursor(self, visible: bool):
        self.brush_cursor.setVisible(visible and not self._ring_cursor_active)

    def _apply_brush_cursor(self, force: bool = False):
        """Make the brush ring the system cursor when it is small enough.

        Rebuilt only when its on-screen size changes (brush size, zoom, screen scale).
        A two-tone ring (white halo, dark line) stays visible on any image.
        """
        if self.mode not in {"paint", "erase"} or self._brush_adjust_start is not None:
            return
        ratio = self.devicePixelRatioF()
        diameter = 2 * self.brush_radius * abs(self.transform().m11())
        key = (round(diameter, 1), ratio)
        if not force and key == self._ring_cursor_key:
            return
        self._ring_cursor_key = key
        size = int(math.ceil(diameter * ratio)) + 8
        if size > BRUSH_CURSOR_MAX_PIXELS or self.source_size == (0, 0):
            was_cursor = self._ring_cursor_active
            self._ring_cursor_active = False
            self.setCursor(Qt.CursorShape.BlankCursor)
            # Switching to the drawn ring (e.g. zooming in): show it where the mouse is
            # now, instead of only after the next mouse move.
            local = self.viewport().mapFromGlobal(QCursor.pos())
            if was_cursor and self.source_size != (0, 0) and self.viewport().rect().contains(local):
                self._set_brush_cursor_position(self.mapToScene(local))
                self._show_brush_cursor(True)
                self.viewport().update()
            return
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center, radius = size / 2, diameter * ratio / 2
        painter.setPen(QPen(QColor(255, 255, 255, 220), 3 * ratio))
        painter.drawEllipse(QPointF(center, center), radius, radius)
        painter.setPen(QPen(QColor(21, 25, 30), 1 * ratio))
        painter.drawEllipse(QPointF(center, center), radius, radius)
        if diameter < 10:  # a tiny brush: mark its centre as well
            arm = 5 * ratio
            painter.drawLine(QPointF(center - arm, center), QPointF(center + arm, center))
            painter.drawLine(QPointF(center, center - arm), QPointF(center, center + arm))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        hot = round(size / ratio / 2)
        self.setCursor(QCursor(pixmap, hot, hot))
        self._ring_cursor_active = True
        self.brush_cursor.hide()

    def _update_adjust_label(self):
        self.adjust_label.setText(
            f"Feather:  {self.brush_radius} px" if self.mode == "polygon" else
            f"Diameter:  {self.brush_radius * 2} px\n"
            f"Hardness:  {self.brush_hardness}%")
        self.adjust_label.adjustSize()
        origin = self._brush_adjust_local
        x = min(max(0, origin.x() + 20), max(0, self.viewport().width() - self.adjust_label.width()))
        y = min(max(0, origin.y() + 20), max(0, self.viewport().height() - self.adjust_label.height()))
        self.adjust_label.move(x, y)

    def _preview_finished(self, worker: PreviewThread):
        if worker in self._preview_workers:
            self._preview_workers.remove(worker)
        worker.deleteLater()

    def finish_previews(self):
        for worker in list(self._preview_workers):
            worker.wait()

    def set_mask(self, path: Path | None):
        metadata = path.stat() if path and path.exists() else None
        key = (str(path) if path else None, self.source_size,
               metadata.st_mtime_ns if metadata else None, metadata.st_size if metadata else None)
        if key == self._mask_key:
            return
        self._mask_key = key
        self.mask_item.setPixmap(QPixmap())
        self.mask_display_item.setPixmap(QPixmap())
        self._mask_preview = None
        if metadata is None or self.source_size == (0, 0):
            return
        cached = self._mask_preview_cache.get(key)
        if cached is not None:
            self._mask_preview = cached.copy()
            self._render_mask_preview()
            return
        # Decoding a full-size mask takes a while: do it off the GUI thread.
        self._pending_mask = (path, key)
        worker = MaskPreviewThread(path, key, self.source_size)
        worker.ready.connect(self._mask_preview_loaded)
        worker.finished.connect(lambda target=worker: self._preview_finished(target))
        self._preview_workers.append(worker)
        worker.start()

    def ensure_mask_preview(self):
        """Finish a pending background mask load now (the user is about to use it)."""
        pending = getattr(self, "_pending_mask", None)
        if self._mask_preview is not None or pending is None or pending[1] != self._mask_key:
            return
        path, key = pending
        try:
            with Image.open(path) as image:
                mask = image.convert("L")
                if mask.size != self.source_size:
                    return
                mask.thumbnail((1800, 1800), Image.Resampling.BILINEAR)
        except OSError:
            return
        self._mask_preview_loaded(key, np.asarray(mask, dtype=np.uint8).copy())

    def _mask_preview_loaded(self, key, preview):
        if self._mask_preview is not None and key == self._mask_key:
            return  # already finished on demand
        self._mask_preview_cache[key] = preview
        while len(self._mask_preview_cache) > 12:
            self._mask_preview_cache.pop(next(iter(self._mask_preview_cache)))
        if key == self._mask_key and not self.points:
            self._mask_preview = preview.copy()
            self._render_mask_preview()

    def set_mask_array(self, path: Path, mask: np.ndarray, revision: int):
        key = (str(path), self.source_size, "resident", revision)
        if key == self._mask_key:
            return
        self._mask_key = key
        if self.source_size == (0, 0) or mask.shape != (self.source_size[1], self.source_size[0]):
            return
        scale = min(1.0, 1800 / max(mask.shape))
        size = (max(1, round(mask.shape[1] * scale)),
                max(1, round(mask.shape[0] * scale)))
        self._mask_preview = (cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
                              if size != (mask.shape[1], mask.shape[0]) else mask.copy())
        self._render_mask_preview()

    def set_overlay_masks(self, paths: list[Path], selected_path: Path | None,
                          resident_masks: dict | None = None):
        """Show independently enabled layer masks without changing their effect on the image."""
        self._overlay_selected = bool(selected_path and selected_path in paths)
        others = []
        for path in paths:
            if path == selected_path or not path.exists():
                continue
            if resident_masks and path in resident_masks:
                source_mask, revision = resident_masks[path]
                key = (str(path), "resident", revision, self.source_size)
                if key not in self._overlay_cache:
                    scale = min(1.0, 1800 / max(source_mask.shape))
                    size = (max(1, round(source_mask.shape[1] * scale)),
                            max(1, round(source_mask.shape[0] * scale)))
                    self._overlay_cache[key] = cv2.resize(source_mask, size,
                                                           interpolation=cv2.INTER_LINEAR)
                others.append(self._overlay_cache[key])
                continue
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size, self.source_size)
            if key not in self._overlay_cache:
                with Image.open(path) as source:
                    mask = source.convert("L")
                    if mask.size != self.source_size:
                        continue
                    mask.thumbnail((1800, 1800), Image.Resampling.BILINEAR)
                self._overlay_cache[key] = np.asarray(mask, dtype=np.uint8).copy()
                if len(self._overlay_cache) > 16:
                    self._overlay_cache.pop(next(iter(self._overlay_cache)))
            others.append(self._overlay_cache[key])
        if others:
            target_shape = (self._mask_preview.shape if self._mask_preview is not None
                            else others[0].shape)
            others = [cv2.resize(item, (target_shape[1], target_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
                      if item.shape != target_shape else item for item in others]
        self._overlay_other = np.maximum.reduce(others) if others else None
        self.show_mask = bool(paths)
        self._render_mask_preview()
        self._update_mask_visibility()

    def _render_mask_preview(self):
        mask = self._mask_preview
        overlay = self._overlay_other
        if self._overlay_selected and mask is not None:
            overlay = np.maximum(overlay, mask) if overlay is not None else mask
        if self.show_mask and overlay is not None:
            rgba = np.zeros((overlay.shape[0], overlay.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = self.overlay_color
            rgba[:, :, 3] = overlay // 3
            self.mask_item.setPixmap(_pixmap_from_array(rgba))
            self.mask_item.setTransform(QTransform.fromScale(
                self.source_size[0] / overlay.shape[1], self.source_size[1] / overlay.shape[0]))
        else:
            self.mask_item.setPixmap(QPixmap())
        if self.view_mask and mask is not None:
            self.mask_display_item.setPixmap(_pixmap_from_array(mask))
            self.mask_display_item.setTransform(QTransform.fromScale(
                self.source_size[0] / mask.shape[1], self.source_size[1] / mask.shape[0]))
        else:
            self.mask_display_item.setPixmap(QPixmap())
        self._update_mask_visibility()

    def _flush_stroke(self):
        self._stroke_timer.stop()
        if self.points:
            self._update_stroke_preview()
            self._update_live_mask(self.mode == "erase")
            if self._live_active:
                self.live_stroke.flush()      # its own timer could be starved as well
            self._last_frame = time.monotonic()
            self.viewport().repaint()  # the whole viewport: one pixel grid, no wobble

    def _present_cursor(self):
        """Show the brush circle where the mouse is right away, redrawing only around it.

        Windows starves queued repaints while the mouse keeps moving, so this paints
        synchronously; just the ring's old and new area, up to 120 times a second.
        """
        now = time.monotonic()
        if now - self._last_cursor_frame < CURSOR_FRAME_SECONDS:
            return  # the regular (queued) repaint covers it once input pauses
        self._last_cursor_frame = now
        rect = self.mapFromScene(self.brush_cursor.sceneBoundingRect()).boundingRect().adjusted(-4, -4, 4, 4)
        dirty = rect if self._cursor_viewport_rect is None else rect.united(self._cursor_viewport_rect)
        self._cursor_viewport_rect = rect
        self.viewport().repaint(dirty.intersected(self.viewport().rect()))

    def _begin_stroke_overlay(self):
        """Show the stroke live: the corrected result when possible, else the mask."""
        self._stroke_coverage = None
        self._stroke_applied = 0
        self._stroke_overlay_saved = None
        self._stroke_tail_box = None
        self._stroke_preview = None
        self._live_active = bool(self._mask_preview_base is not None and not self.view_mask
                                 and self.live_stroke and self.live_stroke.begin())
        if (self._mask_preview_base is None or self.view_mask or self._overlay_selected
                or self._live_active):
            return
        self._stroke_overlay_saved = (self._overlay_selected, self.show_mask)
        self._overlay_selected = True
        self.show_mask = True

    def _end_stroke_overlay(self, cancelled=False):
        if self._live_active:
            self._live_active = False
            self.live_stroke.end(cancelled)
        self._stroke_coverage = None
        self._stroke_preview = None
        self._stroke_applied = 0
        if self._stroke_overlay_saved is not None:
            self._overlay_selected, self.show_mask = self._stroke_overlay_saved
            self._stroke_overlay_saved = None
            self._render_mask_preview()

    def _update_live_mask(self, erase: bool):
        if self._mask_preview_base is None or not self.points:
            return
        self._compose_mask_preview(self.points, erase)

    def _compose_mask_preview(self, points, erase: bool):
        """Paint or erase the stroke into the preview mask, touching only changed pixels."""
        base = self._mask_preview_base
        if base is None or not points:
            return
        scale_x = base.shape[1] / self.source_size[0]
        scale_y = base.shape[0] / self.source_size[1]
        points = [(x * scale_x, y * scale_y) for x, y in points]
        radius = max(1, round(self.brush_radius * min(scale_x, scale_y)))
        feather = round(radius * (100 - self.brush_hardness) / 100)
        first = self._stroke_coverage is None or self._stroke_coverage.shape != base.shape
        if first:
            self._stroke_coverage = np.zeros(base.shape, np.float32)
            self._stroke_applied = 0
            self._stroke_preview = base.copy()
        coverage, preview = self._stroke_coverage, self._stroke_preview
        stamps, tail = engine.brush_stamps(points, radius)
        new = stamps[self._stroke_applied:]
        engine.stamp_coverage(coverage, 0, 0, new, radius, feather)
        self._stroke_applied = len(stamps)
        # Area changed since the last update: new tips, plus the old and new stroke end.
        reach = radius + feather + 3
        changed = [(float(c[0]), float(c[1])) for c in new]
        if tail is not None:
            changed.append((float(tail[0]), float(tail[1])))
        boxes = [(x - reach, y - reach, x + reach + 1, y + reach + 1) for x, y in changed]
        if self._stroke_tail_box is not None:
            boxes.append(self._stroke_tail_box)
        self._stroke_tail_box = ((float(tail[0]) - reach, float(tail[1]) - reach,
                                  float(tail[0]) + reach + 1, float(tail[1]) + reach + 1)
                                 if tail is not None else None)
        if not boxes:
            return
        box = np.array(boxes)
        bx0, by0 = max(0, int(box[:, 0].min())), max(0, int(box[:, 1].min()))
        bx1 = min(base.shape[1], int(np.ceil(box[:, 2].max())))
        by1 = min(base.shape[0], int(np.ceil(box[:, 3].max())))
        if bx0 >= bx1 or by0 >= by1:
            return
        saved = None
        if tail is not None:
            # The stroke end moves with every new point: draw it without keeping it.
            tx0, ty0 = max(0, int(tail[0]) - reach), max(0, int(tail[1]) - reach)
            tx1 = min(base.shape[1], int(tail[0]) + reach + 1)
            ty1 = min(base.shape[0], int(tail[1]) + reach + 1)
            saved = (tx0, ty0, coverage[ty0:ty1, tx0:tx1].copy())
            engine.stamp_coverage(coverage, 0, 0, [tail], radius, feather)
        region = (slice(by0, by1), slice(bx0, bx1))
        source = base[region].astype(np.float32)
        if erase:
            preview[region] = np.round(source * (1 - coverage[region]))
        else:
            preview[region] = np.round(source + (255 - source) * coverage[region])
        if saved is not None:
            tx0, ty0, patch = saved
            coverage[ty0:ty0 + patch.shape[0], tx0:tx0 + patch.shape[1]] = patch
        self._mask_preview = preview
        self._render_mask_preview()
        if self._live_active:
            self.live_stroke.update((bx0 / scale_x, by0 / scale_y, bx1 / scale_x, by1 / scale_y))

    def mark_mask_saved(self, path: Path):
        """Keep the already edited small mask preview after its full-size PNG is saved."""
        if self._mask_preview is None or not path.exists():
            return
        metadata = path.stat()
        self._mask_key = (str(path), self.source_size, metadata.st_mtime_ns,
                          metadata.st_size)

    def preview_polygon_edit(self, vertices, radius: int, hardness: int):
        self.ensure_mask_preview()
        if self._mask_preview is None or self.source_size == (0, 0):
            return
        scale_x = self._mask_preview.shape[1] / self.source_size[0]
        scale_y = self._mask_preview.shape[0] / self.source_size[1]
        scaled = [(x * scale_x, y * scale_y) for x, y in vertices]
        feather = round(radius * min(scale_x, scale_y) * (100 - hardness) / 100)
        x0, y0, x1, y1, coverage = engine.polygon_coverage(
            self._mask_preview.shape, scaled, feather)
        preview = self._mask_preview.copy()
        region = preview[y0:y1, x0:x1]
        np.maximum(region, np.round(coverage * 255).astype(np.uint8), out=region)
        self._mask_preview = preview
        self._render_mask_preview()

    def set_mode(self, mode: str):
        if self.mode == "polygon" and mode != "polygon":
            self._clear_polygon()
        self.mode = mode
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if mode == "pan" else QGraphicsView.DragMode.NoDrag)
        # Moving the brush invalidates many small scene regions. With a scaled
        # pixmap, Qt can rasterize those clipped regions on a slightly different
        # pixel grid, making the art appear to wobble beneath the cursor.
        # Paint the entire viewport in brush mode so every frame uses one grid.
        # (A device-coordinate item cache avoids that too, but leaves stale strips
        # behind when a large image is panned.) Frames are paced by FRAME_SECONDS.
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
            if mode in {"paint", "erase"}
            else QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self._restore_tool_cursor()
        self._show_brush_cursor(mode in {"paint", "erase"} and self.underMouse())
        self._update_mask_visibility()
        self.modeChanged.emit(mode)

    def _restore_tool_cursor(self):
        if self.mode in {"paint", "erase"}:
            self._apply_brush_cursor(force=True)
        elif self.mode == "polygon":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()

    def _clear_polygon(self):
        self.polygon_points.clear()
        self.polygon_item.setPath(QPainterPath())
        self.polygon_start.hide()

    def remove_polygon_point(self):
        if self.polygon_points:
            self.polygon_points.pop()
            self._update_polygon_preview()

    def keyPressEvent(self, event):
        if self.mode == "polygon" and event.key() == Qt.Key.Key_Backspace:
            self.remove_polygon_point()
            event.accept()
            return
        super().keyPressEvent(event)

    def _polygon_near_start(self, viewport_point):
        if len(self.polygon_points) < 3:
            return False
        start = self.mapFromScene(self.polygon_points[0])
        delta = start - viewport_point
        return delta.x() * delta.x() + delta.y() * delta.y() <= 14 * 14

    def _update_polygon_preview(self, hover: QPointF | None = None, close: bool = False):
        if not self.polygon_points:
            self.polygon_item.setPath(QPainterPath())
            self.polygon_start.hide()
            self._restore_tool_cursor()
            return
        first = self.polygon_points[0]
        path = QPainterPath(first)
        for point in self.polygon_points[1:]:
            path.lineTo(point)
        if hover is not None:
            path.lineTo(first if close else hover)
        self.polygon_item.setPath(path)
        self.polygon_start.setPos(first)
        self.polygon_start.show()
        self.setCursor(self._polygon_close_cursor if close else Qt.CursorShape.CrossCursor)

    def cancel_current_action(self):
        self._stroke_timer.stop()
        self.points = []
        self.path_item.setPath(QPainterPath())
        self._clear_polygon()
        if self._mask_preview_base is not None:
            self._mask_preview = self._mask_preview_base
            self._mask_preview_base = None
            self._render_mask_preview()
        self._end_stroke_overlay(cancelled=True)
        self._brush_adjust_start = None
        self._brush_adjust_axis = None
        self.adjust_preview.hide()
        self.adjust_ring.hide()
        self.adjust_label.hide()
        self.unsetCursor()
        self.set_mode("pan")

    def _point(self, event) -> QPointF:
        point = self.mapToScene(event.position().toPoint())
        width, height = self.source_size
        return QPointF(max(0, min(width - 1, point.x())), max(0, min(height - 1, point.y())))

    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.RightButton
                and event.modifiers() & Qt.KeyboardModifier.AltModifier
                and self.mode in {"paint", "erase", "polygon"} and self.source_size != (0, 0)):
            self._brush_adjust_start = event.globalPosition().toPoint()
            self._brush_adjust_scene = self._point(event)
            self._brush_adjust_local = event.position().toPoint()
            self._brush_adjust_axis = None
            self._brush_adjust_pending = QPointF()
            self._brush_adjust_reference = QPointF(self._brush_adjust_start)
            self._brush_adjust_rebase = False
            self._brush_adjust_radius = float(self.brush_radius)
            self._brush_adjust_hardness = float(self.brush_hardness)
            self.adjust_preview.setPos(self._brush_adjust_scene)
            self.adjust_ring.setPos(self._brush_adjust_scene)
            self.adjust_preview.show()
            self.adjust_ring.show()
            self._show_brush_cursor(False)
            self._update_adjust_label()
            self.adjust_label.show()
            self.setCursor(Qt.CursorShape.BlankCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan_at = event.position().toPoint()
            self.setCursor(Qt.CursorShape.BlankCursor if self.mode in {"paint", "erase"}
                           else Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.source_size != (0, 0):
            if self.mode == "polygon":
                self.setFocus()
                if self._polygon_near_start(event.position().toPoint()):
                    vertices = [(point.x(), point.y()) for point in self.polygon_points]
                    self._clear_polygon()
                    self._restore_tool_cursor()
                    self.polygonSelected.emit(vertices, self.brush_radius, self.brush_hardness)
                else:
                    self.polygon_points.append(self._point(event))
                    self._update_polygon_preview()
                event.accept()
                return
            if self.mode in {"paint", "erase"}:
                if self.before_stroke is not None and not self.before_stroke():
                    event.accept()
                    return
                self.ensure_mask_preview()  # a stroke must show live from its first move
                point = self._point(event)
                self.points = [(point.x(), point.y())]
                self._mask_preview_base = self._mask_preview.copy() if self._mask_preview is not None else None
                self._begin_stroke_overlay()
                self._update_stroke_preview()
                self._update_live_mask(self.mode == "erase")
                event.accept()
                return
            self._pan_dragging = self.mode == "pan"
        super().mousePressEvent(event)

    def _update_stroke_preview(self):
        # The flat path is only a fallback when no mask preview can be shown live.
        if not self.points or self._mask_preview_base is not None or self.view_mask:
            self.path_item.setPath(QPainterPath())
            return
        first = QPointF(*self.points[0])
        path = QPainterPath(first)
        if len(self.points) == 1:
            path.addEllipse(first, .1, .1)
        else:
            for x, y in self.points[1:]:
                path.lineTo(x, y)
        self.path_item.setPath(path)

    def mouseMoveEvent(self, event):
        if self._brush_adjust_start is not None:
            position = event.globalPosition()
            if self._brush_adjust_rebase:
                # The cursor was just moved back to the start. At fractional display
                # scaling it lands up to a pixel off; that echo is not a movement.
                self._brush_adjust_rebase = False
                if (abs(position.x() - self._brush_adjust_start.x()) <= 1
                        and abs(position.y() - self._brush_adjust_start.y()) <= 1):
                    self._brush_adjust_reference = position
                    event.accept()
                    return
            offset = position - self._brush_adjust_reference
            if offset.isNull():
                event.accept()
                return
            if self._brush_adjust_axis is None:
                # Commit to an axis only after a clear movement, not the first pixel.
                self._brush_adjust_pending += offset
                pending = self._brush_adjust_pending
                if max(abs(pending.x()), abs(pending.y())) < BRUSH_ADJUST_AXIS_PIXELS:
                    QCursor.setPos(self._brush_adjust_start)
                    self._brush_adjust_reference = QPointF(self._brush_adjust_start)
                    self._brush_adjust_rebase = True
                    event.accept()
                    return
                self._brush_adjust_axis = ("horizontal" if self.mode == "polygon"  # feather only
                                           or abs(pending.x()) >= abs(pending.y()) else "vertical")
                offset = pending
            # Accumulate the value itself, clamped: reversing at 0% or 100% responds at once.
            if self._brush_adjust_axis == "horizontal":
                low, high = ((0.0, float(POLYGON_MAX_FEATHER)) if self.mode == "polygon"
                             else (1.0, 1500.0))
                self._brush_adjust_radius = max(low, min(high, self._brush_adjust_radius
                                                         + offset.x() / BRUSH_RADIUS_DRAG_PIXELS))
                self.set_brush_radius(round(self._brush_adjust_radius))
            else:
                self._brush_adjust_hardness = max(0.0, min(100.0, self._brush_adjust_hardness
                                                           - offset.y() / BRUSH_HARDNESS_DRAG_PIXELS))
                self.set_brush_hardness(round(self._brush_adjust_hardness))
            self._update_adjust_label()
            self.brushAdjusted.emit(self.brush_radius * 2, self.brush_hardness)
            QCursor.setPos(self._brush_adjust_start)
            self._brush_adjust_reference = QPointF(self._brush_adjust_start)
            self._brush_adjust_rebase = True
            event.accept()
            return
        if self.mode in {"paint", "erase"} and self.source_size != (0, 0):
            if self._middle_pan_at is None:
                self._apply_brush_cursor()  # follows zoom and size changes
            if not self._ring_cursor_active:
                self._set_brush_cursor_position(self._point(event))
                self._show_brush_cursor(True)
                if not self.points:
                    self._present_cursor()
        if self._middle_pan_at is not None:
            position = event.position().toPoint()
            delta = position - self._middle_pan_at
            self._middle_pan_at = position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._view_touched = True
            self._emit_view()
            event.accept()
            return
        if self.mode == "polygon" and self.source_size != (0, 0):
            self._update_polygon_preview(self._point(event),
                                         self._polygon_near_start(event.position().toPoint()))
            event.accept()
            return
        if self.points:
            point = self._point(event)
            self.points.append((point.x(), point.y()))
            # Mice report far more often than the screen refreshes: only collect points
            # here and draw them in one batch per frame, so the stroke never lags behind.
            if time.monotonic() - self._last_frame >= FRAME_SECONDS:
                self._flush_stroke()          # due: draw now, even while the mouse keeps moving
            elif not self._stroke_timer.isActive():
                self._stroke_timer.start()    # otherwise when the mouse pauses
            event.accept()
            return
        super().mouseMoveEvent(event)
        if self._pan_dragging:
            self._emit_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.mode == "polygon":
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._brush_adjust_start is not None:
            frozen_point = self._brush_adjust_scene
            self._brush_adjust_start = None
            self._brush_adjust_scene = None
            self._brush_adjust_local = None
            self._brush_adjust_axis = None
            self.adjust_preview.hide()
            self.adjust_ring.hide()
            self.adjust_label.hide()
            self._restore_tool_cursor()
            if self.mode in {"paint", "erase"}:
                self._set_brush_cursor_position(frozen_point)
                self._show_brush_cursor(True)
            self.brushAdjustmentFinished.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_at is not None:
            self._middle_pan_at = None
            self._restore_tool_cursor()
            event.accept()
            return
        if self.points:
            self._stroke_timer.stop()
            if self._mask_preview_base is not None:
                self._compose_mask_preview(self.points, self.mode == "erase")
            points, self.points = self.points, []
            self.path_item.setPath(QPainterPath())
            self._end_stroke_overlay()
            self.stroke.emit(points, self.mode == "erase")
            self._mask_preview_base = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._view_touched = True
        if self._pan_dragging:
            self._emit_view()
        self._pan_dragging = False

    def leaveEvent(self, event):
        if self._brush_adjust_start is None:
            self._show_brush_cursor(False)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        cursor = event.position().toPoint()
        anchor = self.mapToScene(cursor)
        previous_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.scale(factor, factor)
        self.setTransformationAnchor(previous_anchor)
        moved = cursor - self.mapFromScene(anchor)
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - moved.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - moved.y())
        # Resize the ring at once, not on the next mouse move; only now (after the
        # scroll that keeps the point under the mouse) do view and scene line up.
        self._apply_brush_cursor()
        self._view_touched = True
        self._emit_view()
        event.accept()

    def _emit_view(self):
        if not self._syncing:
            self._user_view = True
            self.viewChanged.emit(self.transform(), self.mapToScene(self.viewport().rect().center()), self.source_size)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            self.dropped.emit([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def sync_view(self, transform, center, source_size):
        if self.source_size == (0, 0) or source_size == (0, 0):
            return
        self._syncing = True
        ratio = source_size[0] / self.source_size[0]
        adjusted = QTransform(transform)
        adjusted.scale(ratio, ratio)
        mapped_center = QPointF(
            center.x() * self.source_size[0] / source_size[0],
            center.y() * self.source_size[1] / source_size[1],
        )
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setTransform(adjusted)
        self.centerOn(mapped_center)
        self.setTransformationAnchor(anchor)
        self._apply_brush_cursor()
        self._view_touched = True
        self._user_view = True
        self._syncing = False


class ExclusionView(ImageView):
    """Reference editor reusing navigation, polygon and hard-brush interactions."""
    rectangleSelected = Signal(list)

    def __init__(self):
        super().__init__()
        self._rectangle_start = None
        self._exclusion_previews = {}
        self.set_brush_hardness(100)

    def set_brush_hardness(self, hardness):
        super().set_brush_hardness(100)

    def set_exclusion(self, mask, transform=None, visible=True):
        if mask is None or self.source_size == (0, 0):
            self._mask_preview = None
            self._overlay_selected = False
            self.mask_item.setPixmap(QPixmap())
            return
        width, height = self.source_size
        scale = min(1.0, 1800 / max(width, height))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        scaling = np.diag([size[0] / width, size[1] / height, 1.0])
        transform = np.eye(3) if transform is None else transform
        # Reuse the overlay of a mask already shown (masks are shared, read-only arrays).
        cache_key = (id(mask), mask.shape, transform.tobytes(), size)
        cached = self._exclusion_previews.get(cache_key)
        if cached is not None and cached[0] is mask:
            self._mask_preview = cached[1].copy()
            self._overlay_selected = True
            self._overlay_other = None
            self.show_mask = visible
            self._render_mask_preview()
            return
        source_mask = mask
        # Display only: reduce the full-size mask first (any painted pixel keeps its
        # reduced pixel painted), then warp the small mask to the preview.
        factor = min(1.0, 2 * 1800 / max(mask.shape))
        if factor < 1.0:
            small = (max(1, round(mask.shape[1] * factor)), max(1, round(mask.shape[0] * factor)))
            reduced = cv2.resize(mask, small, interpolation=cv2.INTER_AREA)
            transform = transform @ np.diag([mask.shape[1] / small[0], mask.shape[0] / small[1], 1.0])
            mask = reduced
        self._mask_preview = engine.warp_exclusion(mask, scaling @ transform, size, margin=0).astype(np.uint8) * 255
        if not source_mask.flags.writeable:  # only shared, immutable masks are cached
            self._exclusion_previews[cache_key] = (source_mask, self._mask_preview.copy())
            while len(self._exclusion_previews) > 6:
                self._exclusion_previews.pop(next(iter(self._exclusion_previews)))
        self._overlay_selected = True
        self._overlay_other = None
        self.show_mask = visible
        self._render_mask_preview()

    def cancel_current_action(self):
        self._rectangle_start = None
        super().cancel_current_action()

    def mousePressEvent(self, event):
        if self.mode == "exclude_rectangle" and event.button() == Qt.MouseButton.LeftButton:
            self._rectangle_start = self._point(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._rectangle_start is not None:
            path = QPainterPath()
            path.addRect(QRectF(self._rectangle_start, self._point(event)).normalized())
            self.polygon_item.setPath(path)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rectangle_start is not None and event.button() == Qt.MouseButton.LeftButton:
            rect = QRectF(self._rectangle_start, self._point(event)).normalized()
            self._rectangle_start = None
            self.polygon_item.setPath(QPainterPath())
            if rect.width() >= 1 and rect.height() >= 1:
                self.rectangleSelected.emit([(rect.left(), rect.top()), (rect.right(), rect.top()),
                                             (rect.right(), rect.bottom()), (rect.left(), rect.bottom())])
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TaskThread(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = self.operation(self.cancel_event, self.progress.emit)
            self.completed.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            # Anything that breaks after Cancel was pressed is part of stopping.
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class LayerMetricThread(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, directory: Path, data: dict, scan_id: str):
        super().__init__()
        self.directory = directory
        self.data = data
        self.scan_id = scan_id

    def run(self):
        try:
            project = Project(self.directory, self.data)
            scan = project.scan(self.scan_id)
            result = {"layer_effects": engine.evaluate_layer_effects(project, scan)}
            try:
                output = scan.get("local_output") if not scan.get("local_dirty") else None
                result["final_delta_e76"] = engine.output_delta_e76(project, scan, output)
            except Exception:
                result["final_delta_e76"] = None
            self.ready.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Color Match")
        self.resize(1500, 900)
        self.project: Project | None = None
        self.report = {}
        self.current_scan_id: str | None = None
        self.task: TaskThread | None = None
        self.undo_history: list[tuple[str, str, bytes, bool]] = []
        self.redo_history: list[tuple[str, str, bytes, bool]] = []
        self.removal_history: list[tuple] = []
        self._displayed_scan_id = None
        self._pending_local_scan_id = None
        self._mask_thumb_cache = {}
        self._last_thumbnails = {}
        self._pending_result_thumbs = {}
        self._result_thumb_timer = QTimer(self)
        self._result_thumb_timer.setSingleShot(True)
        self._result_thumb_timer.timeout.connect(self._load_result_thumbnails)
        self._fast_preview_cache = {}
        self._composed_preview_cache = {}
        self._resident_masks: dict[Path, np.ndarray] = {}
        self._mask_revisions: dict[Path, int] = {}
        self._dirty_masks: set[Path] = set()
        self._mask_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mask-save")
        self._mask_save_future = None
        self._mask_save_batch = []
        self._mask_save_error = None
        self._mask_save_timer = QTimer(self)
        self._mask_save_timer.setSingleShot(True)
        self._mask_save_timer.timeout.connect(self._persist_masks)
        self._active_preview_signature = None
        self._fast_preview_warmed = set()
        self._fast_preview_worker: LocalPreviewThread | None = None
        # Per-layer patches: layers whose masks do not overlap switch without recomputation.
        self._preview_patch_cache = engine.PatchCache(256 * 1024 * 1024)
        self._render_patch_cache = engine.PatchCache(768 * 1024 * 1024)
        self._fast_preview_pending = None
        self._local_render_timer = QTimer(self)
        self._local_render_timer.setSingleShot(True)
        self._local_render_timer.timeout.connect(self._run_pending_local_render)
        self._metric_worker = None
        self._metric_scheduled = False
        self._metric_failures = {}
        self._close_when_idle = False
        self._building = False
        self._exclusion_histories = {}
        self._exclusion_view_key = None
        # Slow file decoding for thumbnails and masks runs here, never on the GUI thread.
        self._loader = ThreadPoolExecutor(max_workers=2, thread_name_prefix="scan-color-loader")
        self._loader_bridge = _MainThreadCall()
        self._loading = set()
        self._applying_lut = set()
        self._global_strength_timer = QTimer(self)
        self._global_strength_timer.setSingleShot(True)
        self._global_strength_timer.timeout.connect(self._apply_global_strength)
        self._global_strength_scan_id = None
        self._prefetch_worker: QuickPreviewPrefetch | None = None
        self._prefetch_again = False
        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.timeout.connect(self._start_quick_prefetch)
        self._reference_masks = {}
        self._reference_mask_failures = set()
        self._reference_transform = None
        self._alignment_view_inputs = {}
        self._build_ui()
        self._build_menus()
        translate_widget_tree(self)
        self._open_default_project()
        QApplication.instance().installEventFilter(self)
        self._refresh()

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.Leave and hasattr(self, "scan_list")
                and watched is self.scan_list.viewport()):
            self._scan_hovered(None)
        if (event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape
                and self.isVisible() and QApplication.activeModalWidget() is None):
            if self.edit_exclusions.isChecked():
                self.edit_exclusions.setChecked(False)
                return True
            self.result_view.cancel_current_action()
            return True
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and event.key() == Qt.Key.Key_Space:
            if self.isActiveWindow():
                self.result_view.set_compare(event.type() == QEvent.Type.KeyPress)
                return True
        if (event.type() == QEvent.Type.KeyPress and self.isActiveWindow()
                and QApplication.activeModalWidget() is None):
            if self.edit_exclusions.isChecked():
                if event.key() == Qt.Key.Key_Backspace and self.reference_view.mode == "polygon":
                    self.reference_view.remove_polygon_point()
                    return True
                shortcuts = {Qt.Key.Key_B: "paint", Qt.Key.Key_E: "erase",
                             Qt.Key.Key_L: "polygon"}
                if (event.modifiers() == Qt.KeyboardModifier.NoModifier
                        and event.key() in (*shortcuts, Qt.Key.Key_X)
                        and not isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox))):
                    if event.key() == Qt.Key.Key_X:
                        current = self._exclusion_mode
                        self._set_exclusion_tool(self._exclusion_tool_before_erase
                                                 if current == "erase" else "erase")
                    else:
                        self._set_exclusion_tool(shortcuts[event.key()])
                    return True
                return super().eventFilter(watched, event)
            if (event.key() == Qt.Key.Key_Backspace and self.result_view.mode == "polygon"
                    and watched in (self.result_view, self.result_view.viewport())):
                self.result_view.remove_polygon_point()
                return True
            shortcuts = {Qt.Key.Key_B: "paint", Qt.Key.Key_E: "erase", Qt.Key.Key_L: "polygon"}
            if (event.modifiers() == Qt.KeyboardModifier.NoModifier
                    and event.key() in (*shortcuts, Qt.Key.Key_X)
                    and not isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox))):
                if event.key() == Qt.Key.Key_X:
                    self.toggle_eraser()
                else:
                    self.result_view.set_mode(shortcuts[event.key()])
                return True
        return super().eventFilter(watched, event)

    def _build_menus(self):
        file_menu = self.menuBar().addMenu(tr("File"))
        def menu_action(menu, title, callback=None, shortcut=None):
            item = QAction(title, self)
            if shortcut:
                item.setShortcut(shortcut)
            if callback is not None:
                item.triggered.connect(callback)
            menu.addAction(item)
            return item

        menu_action(file_menu, "New Project...", self.new_project, "Ctrl+N")
        menu_action(file_menu, "Open Project...", self.open_project, "Ctrl+O")
        self.menu_clear_project = menu_action(file_menu, "Clear Project...", self.clear_project)
        file_menu.addSeparator()
        self.menu_add_scans = menu_action(file_menu, "Add Scans...", self.add_scans)
        self.menu_add_references = menu_action(file_menu, "Add References...", self._browse_references)
        self.menu_batch_import = menu_action(file_menu, "Batch Import (ordered pairs)...", self.batch_import)
        file_menu.addSeparator()
        self.menu_export_current = menu_action(file_menu, "Export Current...", self.export_current, "Ctrl+E")
        self.menu_export_all = menu_action(file_menu, "Export All...", self.export_all)
        self.menu_export_preview = menu_action(file_menu, "Include Comparison Previews")
        self.menu_export_preview.setCheckable(True)
        self.menu_export_report = menu_action(file_menu, "Include JSON Reports")
        self.menu_export_report.setCheckable(True)
        file_menu.addSeparator()
        menu_action(file_menu, "Exit", self.close, "Ctrl+Q")
        file_menu.aboutToShow.connect(self._update_file_menu)

        edit_menu = self.menuBar().addMenu(tr("Edit"))
        self.menu_undo = menu_action(edit_menu, "Undo", self.undo_context, "Ctrl+Z")
        self.menu_redo_mask = menu_action(edit_menu, "Redo Mask Edit", self.redo_stroke, "Ctrl+Y")
        self.menu_undo_remove = menu_action(edit_menu, "Restore Removed Scan/Reference",
                                            self.undo_remove, "Ctrl+Shift+Z")
        edit_menu.aboutToShow.connect(self._update_edit_menu)

        help_menu = self.menuBar().addMenu(tr("Help"))
        menu_action(help_menu, "Quick Start...", self.show_quick_start, "F1")
        # Shown in both languages so it can always be found.
        language_menu = self.menuBar().addMenu("Language / 语言")
        self.language_actions = {}
        for code, name in i18n.LANGUAGES.items():
            action = QtGui.QAction(name, self)
            action.setCheckable(True)
            action.setChecked(code == i18n.language())
            action.triggered.connect(lambda _checked=False, value=code: self._choose_language(value))
            language_menu.addAction(action)
            self.language_actions[code] = action

    def _choose_language(self, code):
        for value, action in self.language_actions.items():
            action.setChecked(value == code)
        QSettings("ScanColorMatch", "ScanColorMatch").setValue("language", code)
        previous = i18n.language()
        if code == previous:
            return
        i18n.set_language(code)  # phrase the notice in the language just chosen
        try:
            QMessageBox.information(self, "Language", "Restart Scan Color Match to use the new language.")
        finally:
            i18n.set_language(previous)

    def _update_file_menu(self):
        has_project = self.project is not None
        scan = self._current_scan()
        has_scan = scan is not None
        run = self.project.run_dir() if has_project and self.project.is_current() else None
        ready_scans = ([item for item in self.project.data["scans"]
                        if (run / "base" / f"{item['id']}.png").exists()]
                       if run else [])
        self.menu_add_scans.setEnabled(has_project)
        self.menu_clear_project.setEnabled(has_project and bool(self.project.data["scans"]))
        self.menu_add_references.setEnabled(has_scan)
        self.menu_batch_import.setEnabled(has_project)
        self.menu_export_current.setEnabled(scan in ready_scans)
        self.menu_export_all.setEnabled(bool(ready_scans))

    def _update_edit_menu(self):
        focused = QApplication.focusWidget()
        in_file_list = focused in (self.scan_list, self.scan_list.viewport(),
                                  self.ref_list, self.ref_list.viewport())
        undo, redo = self._exclusion_histories.get(self._exclusion_key(), ([], []))
        self.menu_undo.setEnabled(bool(self.removal_history) if in_file_list else
                                  bool(undo) if self._exclusion_undo_context() else
                                  bool(self.undo_history or self.removal_history))
        self.menu_redo_mask.setEnabled(bool(redo) if self._exclusion_undo_context() else bool(self.redo_history))
        self.menu_undo_remove.setEnabled(bool(self.removal_history))

    def show_quick_start(self):
        message = QMessageBox(self)
        message.setWindowTitle("Quick Start")
        message.setIcon(QMessageBox.Icon.Information)
        message.setTextFormat(Qt.TextFormat.RichText)
        message.setText(QUICK_START_HTML)
        message.exec()

    def _button(self, title, action, layout):
        button = QPushButton(title)
        if action is not None:
            button.clicked.connect(action)
        layout.addWidget(button)
        return button

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        splitter = QSplitter()
        root.addWidget(splitter, 1)

        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(360)
        left_layout = QVBoxLayout(left)
        self.project_label = QLabel("No project")
        font = self.project_label.font()
        font.setBold(True)
        self.project_label.setFont(font)
        left_layout.addWidget(self.project_label)
        self.scan_list = FileDropList()
        self.scan_list.setItemDelegate(ScanItemDelegate(self.scan_list))
        self.scan_list.setDragEnabled(False)
        self.scan_list.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.scan_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scan_list.dropped.connect(self._add_scan_paths)
        self.scan_list.deleteRequested.connect(self.remove_selected_scans)
        self.scan_list.undoRequested.connect(self.undo_remove)
        self.scan_list.currentItemChanged.connect(self._selected_scan_changed)
        self.scan_list.setMouseTracking(True)
        self.scan_list.itemEntered.connect(self._scan_hovered)
        self.scan_list.viewport().installEventFilter(self)
        left_layout.addWidget(self.scan_list, 1)
        left_layout.addWidget(_separator())
        settings = QGridLayout()
        settings.setHorizontalSpacing(6)
        fit_help = ("What the global LUT is fitted to. Validation and the ΔE readouts always use ΔE76.\n"
                    "ΔE76: fits the perceptual color difference directly. Better results, "
                    "a few seconds slower per pass.\n"
                    "RGB: fits the RGB error. Fastest, slightly less accurate.")
        fit_label = QLabel("Fit target")
        fit_label.setToolTip(fit_help)
        self.fit_de76 = QRadioButton("ΔE76")
        self.fit_de76.setToolTip(fit_help)
        self.fit_rgb = QRadioButton("RGB")
        self.fit_rgb.setToolTip(fit_help)
        self.fit_group = QButtonGroup(self)
        self.fit_group.addButton(self.fit_de76)
        self.fit_group.addButton(self.fit_rgb)
        self.fit_de76.setChecked(True)
        self.fit_de76.toggled.connect(self._fit_settings_changed)
        settings.addWidget(fit_label, 0, 0)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.fit_de76)
        fit_row.addWidget(self.fit_rgb)
        fit_row.addStretch(1)
        settings.addLayout(fit_row, 0, 1)
        passes_help = ("Global passes for scans with their own LUT: each pass corrects what the "
                       "previous one left. Every further pass gains less, costs a few seconds per "
                       "scan, and fits misalignment and noise in the reference more closely "
                       "(overfitting). A pass that fails held-out validation is not used.\n"
                       "The global layer's strength then picks the point with the lowest ΔE76; "
                       "you can change it afterwards.")
        passes_label = QLabel("Passes")
        passes_label.setToolTip(passes_help)
        self.passes_box = QSpinBox()
        self.passes_box.setRange(1, engine.MAX_PASSES)
        self.passes_box.setValue(engine.DEFAULT_PASSES)
        self.passes_box.setToolTip(passes_help)
        self.passes_box.valueChanged.connect(self._fit_settings_changed)
        settings.addWidget(passes_label, 1, 0)
        passes_row = QHBoxLayout()
        passes_row.addWidget(self.passes_box)
        passes_row.addStretch(1)
        settings.addLayout(passes_row, 1, 1)
        settings.setColumnStretch(1, 1)
        left_layout.addLayout(settings)
        left_layout.addWidget(_separator())
        # One large button: Run Color Match, and Cancel while any task runs.
        self.run_button = QPushButton("Run Color Match")
        self.run_button.clicked.connect(self._run_or_cancel)
        left_layout.addWidget(self.run_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(26)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The native Windows 11 bar is a thin line with its text beside it: draw a
        # full-height bar with the percentage and time left inside instead.
        self._set_progress_done(False)
        self.progress_bar.setFormat("")
        left_layout.addWidget(self.progress_bar)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)
        left_layout.addWidget(_separator())
        export_row = QHBoxLayout()
        self.export_button = self._button("Export Current...", self.export_current, export_row)
        self.export_all_button = self._button("Export All...", self.export_all, export_row)
        left_layout.addLayout(export_row)
        splitter.addWidget(left)
        self.left_panel = left

        center = QWidget()
        center_layout = QVBoxLayout(center)
        headings = QHBoxLayout()
        headings.addWidget(QLabel("Scan / Result — drop scan to replace"), 1)
        headings.addWidget(QLabel("Reference — drop image to replace"), 1)
        center_layout.addLayout(headings)
        views = QSplitter()
        self.result_view = ImageView()
        self.result_view.live_stroke = LiveStrokePreview(self)
        self.result_view.before_stroke = self._prepare_layer_for_stroke
        self.reference_view = ExclusionView()
        self.reference_view.stroke.connect(self._exclusion_stroke)
        self.reference_view.polygonSelected.connect(lambda points, *_: self._edit_exclusion(points, "polygon"))
        self.reference_view.rectangleSelected.connect(lambda points: self._edit_exclusion(points, "rectangle"))
        self.reference_view.brushAdjusted.connect(lambda diameter, _: self.exclusion_diameter.setValue(diameter))
        self.result_view.viewChanged.connect(self.reference_view.sync_view)
        self.result_view.refitted.connect(self._result_refitted)
        self.reference_view.viewChanged.connect(self.result_view.sync_view)
        self.result_view.dropped.connect(self._result_dropped)
        self.reference_view.dropped.connect(self._reference_view_dropped)
        self.result_view.stroke.connect(self._paint_stroke)
        self.result_view.polygonSelected.connect(self._paint_polygon)
        views.addWidget(self.result_view)
        views.addWidget(self.reference_view)
        views.setStretchFactor(0, 1)
        views.setStretchFactor(1, 1)
        views.setSizes([500, 500])
        center_layout.addWidget(views, 1)
        self.view_info = QLabel("Drop a scan into the left preview to begin. Hold Space to compare the original.")
        self.view_info.setWordWrap(False)
        self.view_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        info_row = QHBoxLayout()
        info_row.addWidget(self.view_info, 1)
        self.color_metrics = QLabel()
        self.color_metrics.setWordWrap(False)
        self.color_metrics.setToolTip(
            "Mean CIE76 Lab color difference on the selected layer's mask. Off and On "
            "keep all other layers unchanged and include the current mask, strength and line protection.")
        # All ΔE76 measurements sit together on the right, separated by " | ".
        info_row.addWidget(self.color_metrics)
        self.metric_separator = QLabel("|")
        self.metric_separator.hide()
        info_row.addWidget(self.metric_separator)
        self.global_metric = QLabel()
        self.global_metric.setToolTip(
            "Mean ΔE76 against the aligned reference, lower is better. "
            "Mask: the selected layer off → on, inside its mask. "
            "Global: original scan → global LUT and spatial correction. "
            "Final: original scan → the result shown, with the global switch and all local layers")
        info_row.addWidget(self.global_metric)
        center_layout.addLayout(info_row)
        splitter.addWidget(center)

        side_scroll = QScrollArea()
        side_scroll.setMinimumWidth(290)
        side_scroll.setMaximumWidth(320)
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side = QWidget()
        controls = QVBoxLayout(side)
        side_scroll.setWidget(side)
        controls.addWidget(QLabel("References"))
        self.ref_list = FileDropList()
        self.ref_list.setDragEnabled(False)
        self.ref_list.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.ref_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        row_delegate = CompactRowDelegate(self.ref_list)
        self.ref_list.setItemDelegate(row_delegate)
        self.ref_list.setFixedHeight(row_delegate.row_height * 4 + 2 * self.ref_list.frameWidth())
        self.ref_list.dropped.connect(self._references_dropped)
        self.ref_list.deleteRequested.connect(self.remove_selected_references)
        self.ref_list.undoRequested.connect(self.undo_remove)
        self.ref_list.currentItemChanged.connect(self._ref_changed)
        controls.addWidget(self.ref_list)
        self.ref_list.itemChanged.connect(self._reference_item_changed)
        mask_label = QLabel("Sampling Mask")
        mask_label.setToolTip("Paint on the reference preview which reference colors may be used as "
                              "evidence; saved in original reference coordinates. The LUT still "
                              "applies to the full scan.")
        controls.addWidget(mask_label)
        self.mask_mode_exclude = QRadioButton("Exclusion")
        self.mask_mode_exclude.setToolTip("Painted pixels (red) are never used: watermarks, text, "
                                          "content that does not match the scan")
        self.mask_mode_include = QRadioButton("Inclusion")
        self.mask_mode_include.setToolTip("Only painted pixels (green) are used: the part of the "
                                          "reference that matches the scan")
        self.mask_mode_exclude.setChecked(True)
        self.mask_mode_group = QButtonGroup(self)
        self.mask_mode_group.addButton(self.mask_mode_exclude)
        self.mask_mode_group.addButton(self.mask_mode_include)
        self.mask_mode_include.toggled.connect(self._mask_mode_changed)
        mask_mode_row = QHBoxLayout()
        mask_mode_row.addWidget(self.mask_mode_exclude)
        mask_mode_row.addWidget(self.mask_mode_include)
        mask_mode_row.addStretch(1)
        self.show_exclusions = QCheckBox("Show mask")
        self.show_exclusions.setChecked(True)
        self.show_exclusions.toggled.connect(self._refresh_exclusion_overlay)
        mask_mode_row.addWidget(self.show_exclusions)
        controls.addLayout(mask_mode_row)
        exclusion_row = QHBoxLayout()
        self.edit_exclusions = self._button("Edit Mask", None, exclusion_row)
        self.edit_exclusions.setCheckable(True)
        self.edit_exclusions.toggled.connect(self._toggle_exclusion_edit)
        self.clear_exclusions = self._button("Clear Mask", self._clear_exclusions, exclusion_row)
        controls.addLayout(exclusion_row)
        self.exclusion_toolbar = QWidget()
        exclusion_tools = QVBoxLayout(self.exclusion_toolbar)
        exclusion_tools.setContentsMargins(0, 0, 0, 0)
        # Same button style as the layer tools; one tool is always active while editing.
        self._exclusion_mode = "polygon"
        self._exclusion_last_tool = "polygon"
        self._exclusion_tool_before_erase = "paint"
        self._exclusion_tool_buttons = {}
        tool_row = QHBoxLayout()
        for mode, title, tip in (("polygon", "Polygon", "Polygon (L): click the first point to close"),
                                 ("paint", "Paint", "Brush (B)"),
                                 ("erase", "Erase", "Eraser (E) · X switches back")):
            button = self._button(title, lambda _=False, mode=mode: self._set_exclusion_tool(mode), tool_row)
            button.setCheckable(True)
            button.setToolTip(tip)
            self._exclusion_tool_buttons[mode] = button
        self._exclusion_tool_buttons["polygon"].setChecked(True)
        exclusion_tools.addLayout(tool_row)
        self.exclusion_diameter_row = QWidget()
        diameter_row = QHBoxLayout(self.exclusion_diameter_row)
        diameter_row.setContentsMargins(0, 0, 0, 0)
        diameter_row.addWidget(QLabel("Brush diameter"))
        self.exclusion_diameter = QSpinBox()
        self.exclusion_diameter.setRange(2, 3000)
        self.exclusion_diameter.setValue(100)
        self.exclusion_diameter.valueChanged.connect(self._exclusion_diameter_changed)
        diameter_row.addWidget(self.exclusion_diameter)
        exclusion_tools.addWidget(self.exclusion_diameter_row)
        self.exclusion_diameter_row.hide()  # only the brush and the eraser have a diameter
        controls.addWidget(self.exclusion_toolbar)
        self.exclusion_toolbar.hide()

        controls.addWidget(_separator())
        controls.addWidget(QLabel("LUT Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Shared", "shared")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        controls.addWidget(self.mode_combo)

        local_controls = controls
        local_controls.addWidget(_separator())
        paint_row = QHBoxLayout()
        # The active tool's button stays pressed; clicking it again leaves the tool.
        self._tool_buttons = {}
        for mode, title, tip in (
                ("polygon", "Polygon", "Polygon selection (L): click vertices, then click the first point to close"),
                ("paint", "Paint", "Brush (B)"),
                ("erase", "Erase", "Eraser (E) · X switches between the eraser and the previous tool")):
            button = self._button(title, lambda _=False, mode=mode: self._tool_button_clicked(mode), paint_row)
            button.setCheckable(True)
            button.setToolTip(tip + " · click again to leave the tool")
            self._tool_buttons[mode] = button
        local_controls.addLayout(paint_row)
        tool_parameters = QGridLayout()
        tool_parameters.setHorizontalSpacing(6)
        self.tool_size_label = QLabel("Diameter")
        tool_parameters.addWidget(self.tool_size_label, 0, 0)
        tool_parameters.addWidget(QLabel("Hardness"), 0, 1)
        self._tool_values = {
            "polygon": (POLYGON_DEFAULT_FEATHER, None), "paint": (100, 80), "erase": (100, 80),
        }
        self.tool_diameter = QSpinBox()
        self.tool_diameter.setRange(2, 3000)
        self.tool_diameter.setSingleStep(2)
        self.tool_diameter.setFixedWidth(100)
        self.tool_hardness = QSpinBox()
        self.tool_hardness.setRange(0, 100)
        self.tool_hardness.setFixedWidth(90)
        tool_parameters.addWidget(self.tool_diameter, 1, 0)
        tool_parameters.addWidget(self.tool_hardness, 1, 1)
        self.tool_diameter.valueChanged.connect(self._tool_parameter_changed)
        self.tool_hardness.valueChanged.connect(self._tool_parameter_changed)
        tool_parameters.setColumnStretch(2, 1)
        self._last_tool_mode = "pan"
        self._tool_before_erase = "paint"
        self.result_view.modeChanged.connect(self._tool_mode_changed)
        self.result_view.brushAdjusted.connect(self._brush_drag_changed)
        self._tool_mode_changed(self.result_view.mode)
        local_controls.addLayout(tool_parameters)
        local_controls.addWidget(_separator())
        local_controls.addWidget(QLabel("Layers"))
        local_help = QLabel("Select a reference, click +, paint or use polygon selection, then Refine All. Click a thumbnail to view the result or mask; drag layers to reorder.")
        local_help.setWordWrap(True)
        local_controls.addWidget(local_help)
        self.layer_list = LayerList()
        self.layer_list.currentItemChanged.connect(self._layer_changed)
        self.layer_list.itemChanged.connect(self._layer_item_changed)
        self.layer_list.addRequested.connect(self.add_layer)
        self.layer_list.renameRequested.connect(self.rename_layer)
        self.layer_list.deleteRequested.connect(self._delete_layer_by_id)
        self.layer_list.maskRequested.connect(self._toggle_layer_mask_by_id)
        self.layer_list.resultRequested.connect(self._show_layer_result_by_id)
        self.layer_list.overlayRequested.connect(self._toggle_layer_overlay_by_id)
        self.layer_list.strengthRequested.connect(self._set_layer_strength_by_id)
        self.layer_list.strengthEdited.connect(self._set_layer_strength_value)
        self.layer_list.orderChanged.connect(self._layer_order_changed)
        self.layer_list.menuRequested.connect(self._layer_context_menu)
        self.layer_list.setMinimumHeight(110)
        local_controls.addWidget(self.layer_list, 1)
        self.protect_lines = QCheckBox("Protect dark lines")
        self.protect_lines.setToolTip("Reduce the selected layer's color correction on very dark lines")
        self.protect_lines.setChecked(True)
        self.protect_lines.toggled.connect(self._layer_options_changed)
        local_controls.addWidget(self.protect_lines)
        self.view_mask_box = QCheckBox("Mask View", side)
        self.view_mask_box.hide()
        self.view_mask_box.setToolTip("Show the selected layer mask alone in black and white")
        self.view_mask_box.toggled.connect(self.result_view.set_view_mask)
        self.view_mask_box.toggled.connect(self._refresh_mask_view_markers)
        self.refine_button = self._button("Refine All", self.refine_selection, local_controls)
        self.refine_button.setToolTip(
            "Re-fit changed layers bottom-to-top, then automatically choose their strengths")
        splitter.addWidget(side_scroll)
        self.side_panel = side_scroll
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 950, 300])


    def _error(self, message: str):
        QMessageBox.warning(self, "Scan Color Match", message)
        self.status.setText(message)

    def _current_scan(self):
        if self.project is None or self.current_scan_id is None:
            return None
        return next((item for item in self.project.data["scans"] if item["id"] == self.current_scan_id), None)

    def _current_ref(self):
        scan = self._current_scan()
        current = self.ref_list.currentItem()
        ref_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        if scan is None or ref_id is None:
            return None
        return next((ref for ref in scan["refs"] if ref["id"] == ref_id), None)

    def _current_layer(self):
        scan = self._current_scan()
        item = self.layer_list.currentItem()
        if scan is None or item is None:
            return None
        return next((layer for layer in scan["layers"] if layer["id"] == item.data(Qt.ItemDataRole.UserRole)), None)

    @staticmethod
    def _default_project_directory() -> Path:
        app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        return Path(app_data) / "default-project"

    def _export_start_directory(self, scans) -> str:
        """Where the export folder dialog opens.

        The default project lives in the application data folder, which is no place to
        look for exports: start next to the first exported image instead. A project the
        user created opens in its own folder.
        """
        if self.project.directory.resolve() == self._default_project_directory().resolve():
            first = next((Path(scan["path"]) for scan in scans if scan["path"]), None)
            if first is not None and first.parent.exists():
                return str(first.parent)
        return str(self.project.directory)

    def _open_default_project(self):
        directory = self._default_project_directory()
        try:
            self.project = (Project.open(directory) if (directory / "project.json").exists()
                            else Project.create(directory))
            self._load_report()
        except Exception as exc:
            self._error(f"Could not open the default project: {exc}")

    def clear_project(self, *, confirm=True):
        if self.project is None or not self.project.data["scans"]:
            return
        if confirm and self.task and self.task.isRunning():
            self._error("Wait for the current operation to finish before clearing the project")
            return
        if confirm and QMessageBox.question(
                self, "Clear Project", "Clear all scans, references, layers and generated cache from this project?\n"
                "Original images and exported files will remain untouched.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._flush_masks_blocking()
            self._local_render_timer.stop()
            self._pending_local_scan_id = None
            self._fast_preview_pending = None
            if self._fast_preview_worker and self._fast_preview_worker.isRunning():
                self._fast_preview_worker.wait()
            if self._metric_worker and self._metric_worker.isRunning():
                self._metric_worker.wait()
            self.result_view.finish_previews()
            self.reference_view.finish_previews()
            self.project.data.update(active_run=None, scans=[],
                                     calibration_revision=self.project.data["calibration_revision"] + 1)
            self.project.save()
            cache = self.project.directory / "cache"
            if cache.exists():
                shutil.rmtree(cache)
            self.report = {}
            self.current_scan_id = None
            self.undo_history.clear()
            self.redo_history.clear()
            self.removal_history.clear()
            self._resident_masks.clear()
            self._mask_revisions.clear()
            self._dirty_masks.clear()
            self._mask_thumb_cache.clear()
            self._fast_preview_cache.clear()
            self._composed_preview_cache.clear()
            self._active_preview_signature = None
            self._refresh()
            self.status.setText("Project cleared")
        except Exception as exc:
            self._error(f"Could not fully clear project: {exc}")

    def new_project(self):
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        directory = QFileDialog.getExistingDirectory(self, "Choose an empty project folder", documents)
        if not directory:
            return
        try:
            self._flush_masks_blocking()
            self._resident_masks.clear()
            self._mask_revisions.clear()
            self._fast_preview_cache.clear()
            self._composed_preview_cache.clear()
            self.project = Project.create(Path(directory))
            self.report = {}
            self.current_scan_id = None
            self.removal_history.clear()
            self._refresh()
        except Exception as exc:
            self._error(str(exc))

    def open_project(self):
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        directory = QFileDialog.getExistingDirectory(self, "Choose a project folder", documents)
        if not directory:
            return
        try:
            self._flush_masks_blocking()
            self._resident_masks.clear()
            self._mask_revisions.clear()
            self._fast_preview_cache.clear()
            self._composed_preview_cache.clear()
            self.project = Project.open(Path(directory))
            self.current_scan_id = None
            self.removal_history.clear()
            self._load_report()
            self._refresh()
        except Exception as exc:
            self._error(str(exc))

    def _load_report(self):
        self.report = {}
        run = self.project.run_dir() if self.project else None
        if run and (run / "report.json").exists():
            self.report = json.loads((run / "report.json").read_text(encoding="utf-8"))

    def add_pair(self):
        if self.project is None:
            self._error("Create or open a project first")
            return
        scan = self.project.add_scan()
        self.current_scan_id = scan["id"]
        self._refresh()

    def add_scans(self):
        if self.project is None:
            self._error("Create or open a project first")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add scans", "", IMAGE_FILTER)
        self._add_scan_paths(paths)

    def _valid_paths(self, paths):
        valid, errors = [], []
        for path in paths:
            try:
                engine.validate_image(path)
                valid.append(str(Path(path).resolve()))
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        if errors:
            self._error("Could not add:\n" + "\n".join(errors[:8]))
        return valid

    def _add_scan_paths(self, paths):
        if not self.project:
            self._error("Create or open a project first")
            return
        self._insert_scans(self._valid_paths(paths))

    def _insert_scans(self, paths):
        if not paths:
            return
        added = []
        was_current = self.project.is_current()
        empty = self._current_scan()
        for path in paths:
            if empty is not None and not empty["path"]:
                empty["path"] = path
                added.append(empty)
                empty = None
            else:
                added.append(self.project.add_scan(path, save=False))
        self.current_scan_id = added[0]["id"]
        self.project.invalidate()
        self._update_without_rerun(was_current, added)
        self._refresh()
        self.status.setText(f"Added {len(paths)} scan(s)")

    def _scan_dropped(self, paths):
        self._add_scan_paths(paths)

    def _result_dropped(self, paths):
        if not self.project:
            self._error("Create or open a project first")
            return
        if len(paths) != 1 or self._current_scan() is None or not self._current_scan()["path"]:
            self._add_scan_paths(paths)
            return
        valid = self._valid_paths(paths)
        if not valid:
            return
        scan = self._current_scan()
        was_current = self.project.is_current()
        scan["path"] = valid[0]
        scan["local_output"] = None
        scan.pop("previous_result_run", None)
        scan["local_dirty"] = bool(scan["layers"])
        self.project.invalidate()
        updated = self._update_without_rerun(was_current, [scan])
        self._refresh()
        if not updated:
            self.status.setText("Scan replaced. Run Color Match to update the result")

    def _reference_view_dropped(self, paths):
        scan = self._current_scan()
        if scan is None:
            self._error("Select or add a scan card first")
            return
        if len(paths) != 1 or self._current_ref() is None:
            self._references_dropped(paths)
            return
        valid = self._valid_paths(paths)
        if not valid:
            return
        ref = self._current_ref()
        ref["path"] = valid[0]
        # A new image starts with a clean sampling mask: no paint and the default mode.
        ref["mask_mode"] = "exclude"
        ref.pop("exclude_mask", None)
        ref.pop("exclude_mask_digest", None)
        self._exclusion_histories.pop(self._exclusion_key(), None)
        self.edit_exclusions.setChecked(False)
        self._alignment_view_inputs.pop(self._exclusion_key(), None)
        scan["local_dirty"] = bool(scan["layers"])
        self.project.invalidate()
        self._refresh()
        self.status.setText("Reference replaced. Run Color Match to update the result")

    def clear_scan(self):
        scan = self._current_scan()
        if scan is None or not scan["path"]:
            return
        self.removal_history.append(("scan_path", scan["id"], scan["path"]))
        scan["path"] = ""
        scan["local_output"] = None
        self.project.invalidate()
        self._refresh()

    def _references_dropped(self, paths):
        scan = self._current_scan()
        if scan is None:
            self._error("Select a scan card first")
            return
        valid = self._valid_paths(paths)
        if valid:
            for path in valid:
                self.project.add_reference(scan, path, save=False)
            self.project.invalidate()
            self._refresh()
            self.status.setText(f"Added {len(valid)} reference(s)")

    def batch_import(self):
        if self.project is None:
            self._error("Create or open a project first")
            return
        dialog = BatchPairDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        scans, refs = dialog.paths()
        if not scans:
            self._error("Add at least one scan")
            return
        if len(refs) > len(scans):
            self._error("There are more references than scans. Remove extra references or add scans.")
            return
        valid_scans = self._valid_paths(scans)
        valid_refs = self._valid_paths(refs)
        if len(valid_scans) != len(scans) or len(valid_refs) != len(refs):
            return
        self._import_ordered_pairs(valid_scans, valid_refs)

    def _import_ordered_pairs(self, scans, refs):
        added = []
        for index, path in enumerate(scans):
            card = self.project.add_scan(path, save=False)
            added.append(card)
            if index < len(refs):
                self.project.add_reference(card, refs[index], save=False)
        if added:
            self.current_scan_id = added[0]["id"]
            self.project.invalidate()
            self._refresh()
            self.status.setText(f"Imported {len(added)} image pair(s)")

    def _browse_references(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose references", "", IMAGE_FILTER)
        if paths:
            self._references_dropped(paths)

    def remove_scan(self):
        self.remove_selected_scans()

    def remove_selected_scans(self):
        if not self.project:
            return
        selected = {item.data(Qt.ItemDataRole.UserRole) for item in self.scan_list.selectedItems()}
        if not selected and self.current_scan_id:
            selected = {self.current_scan_id}
        if not selected:
            return
        removed = [(index, copy.deepcopy(scan)) for index, scan in enumerate(self.project.data["scans"])
                   if scan["id"] in selected]
        was_current = self.project.is_current()
        self.removal_history.append(("cards", removed))
        self.project.data["scans"] = [scan for scan in self.project.data["scans"] if scan["id"] not in selected]
        self.current_scan_id = None
        self.project.invalidate()
        self._keep_results_current(was_current, [scan for _, scan in removed])
        self._refresh()

    def _update_without_rerun(self, was_current, scans):
        """Scans without references only reuse fitted LUTs: re-apply, don't recalibrate."""
        run = self.project.data.get("active_run")
        if (not was_current or not run or not scans
                or any(scan["refs"] or not scan["path"] for scan in scans)):
            return False
        run["input_signature"] = self.project.calibration_signature()
        self.project.save()
        directory = self.project.directory
        for scan in scans:
            self.report.setdefault("scans", {}).pop(scan["id"], None)
            self._applying_lut.add(scan["id"])
            snapshot = Project(directory, copy.deepcopy(self.project.data))
            target = snapshot.scan(scan["id"])

            def done(updates, scan_id=scan["id"], signature=run["input_signature"]):
                self._applying_lut.discard(scan_id)
                if (not self.project or self.project.directory != directory
                        or (self.project.data.get("active_run") or {}).get("input_signature") != signature):
                    return  # the project moved on; a new run will cover this scan
                self.report.setdefault("scans", {})[scan_id] = updates
                report_path = self.project.run_dir() / "report.json"
                report_path.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
                scan = next((item for item in self.project.data["scans"] if item["id"] == scan_id), None)
                if scan is not None:
                    self._update_scan_item(scan)
                    if self.current_scan_id == scan_id:
                        self._refresh_image()
            self._in_background(("apply-lut", scan["id"], run["input_signature"]),
                                lambda target=target, snapshot=snapshot: engine.apply_existing_lut(snapshot, target),
                                done)
        self.status.setText("Applying the existing LUT to the changed scan(s)...")
        return True

    def _keep_results_current(self, was_current, scans, restored=False):
        """Removing or restoring scans without references changes no other result.

        Such scans train nothing and nothing can follow them, so the last run stays valid
        for every remaining scan; a restored one needs its own output still in that run.
        """
        run = self.project.data.get("active_run")
        if not was_current or not run or any(scan["refs"] for scan in scans):
            return
        if restored:
            run_dir = self.project.run_dir()
            if not all((run_dir / "base" / f"{scan['id']}.png").exists() for scan in scans):
                return
        run["input_signature"] = self.project.calibration_signature()
        self.project.save()

    def remove_reference(self):
        self.remove_selected_references()

    def remove_selected_references(self):
        scan = self._current_scan()
        if scan is None:
            return
        selected = {item.data(Qt.ItemDataRole.UserRole) for item in self.ref_list.selectedItems()}
        if not selected and self._current_ref():
            selected = {self._current_ref()["id"]}
        if not selected:
            return
        removed_refs = [(index, copy.deepcopy(ref)) for index, ref in enumerate(scan["refs"])
                        if ref["id"] in selected]
        removed_layers = [(index, copy.deepcopy(layer)) for index, layer in enumerate(scan["layers"])
                          if layer["reference_id"] in selected]
        self.removal_history.append(("references", scan["id"], removed_refs, removed_layers))
        scan["refs"] = [ref for ref in scan["refs"] if ref["id"] not in selected]
        scan["layers"] = [layer for layer in scan["layers"] if layer["reference_id"] not in selected]
        scan["local_dirty"] = bool(scan["layers"])
        self.project.invalidate()
        self._refresh()

    def undo_remove(self):
        if not self.project or not self.removal_history:
            return
        record = self.removal_history.pop()
        was_current = self.project.is_current()
        if record[0] == "cards":
            for index, scan in record[1]:
                self.project.data["scans"].insert(index, scan)
            self.current_scan_id = record[1][0][1]["id"]
        elif record[0] == "references":
            scan = next((item for item in self.project.data["scans"] if item["id"] == record[1]), None)
            if scan is None:
                return
            for index, ref in record[2]:
                scan["refs"].insert(index, ref)
            for index, layer in record[3]:
                scan["layers"].insert(index, layer)
            self.current_scan_id = scan["id"]
        elif record[0] == "card":
            self.project.data["scans"].insert(record[1], record[2])
            self.current_scan_id = record[2]["id"]
        elif record[0] == "scan_path":
            scan = next((item for item in self.project.data["scans"] if item["id"] == record[1]), None)
            if scan is None:
                return
            scan["path"] = record[2]
            self.current_scan_id = scan["id"]
        else:
            scan = next((item for item in self.project.data["scans"] if item["id"] == record[1]), None)
            if scan is None:
                return
            scan["refs"].insert(record[2], record[3])
            scan["layers"].extend(record[4])
            self.current_scan_id = scan["id"]
        self.project.invalidate()
        if record[0] == "cards":
            self._keep_results_current(was_current, [scan for _, scan in record[1]], restored=True)
        self._refresh()

    def undo_context(self):
        focused = QApplication.focusWidget()
        if focused in (self.scan_list, self.scan_list.viewport(), self.ref_list, self.ref_list.viewport()):
            self.undo_remove()
        elif self._exclusion_undo_context():
            self._restore_exclusion(False)
        elif self.undo_history:
            self.undo_stroke()
        else:
            self.undo_remove()

    def _selected_scan_changed(self, current, previous):
        if self._building:
            return
        next_scan_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        if next_scan_id != self.current_scan_id:
            self._active_preview_signature = None
        if next_scan_id != self.current_scan_id and self.result_view.mode == "polygon":
            self.result_view._clear_polygon()
        self.current_scan_id = next_scan_id
        self._refresh_details()
        self._maybe_recompose_saved_local_layer()

    def _maybe_recompose_saved_local_layer(self):
        """Repair legacy pending settings only when their fitted masks are unchanged."""
        scan = self._current_scan()
        if not scan or not self.project.is_current() or not scan.get("local_dirty"):
            return
        output = self.project.directory / scan["local_output"] if scan.get("local_output") else None
        if output is None or not output.exists():
            return
        run_id = self.project.data["active_run"]["id"]
        for layer in scan["layers"]:
            mask = self.project.directory / layer["mask"]
            if (not mask.exists() or mask.stat().st_mtime_ns > output.stat().st_mtime_ns
                    or not layer.get("model") or layer.get("model_run") != run_id
                    or layer.get("needs_refit", False)):
                return
        self._pending_local_scan_id = scan["id"]
        self._local_render_timer.start(250)
        self.status.setText("Updating saved local-layer settings...")

    def _ref_changed(self):
        if not self._building:
            self._refresh_ref()

    def _reference_item_changed(self, item):
        """The row checkbox enables or disables that reference for color matching."""
        if self._building:
            return
        scan = self._current_scan()
        ref_id = item.data(Qt.ItemDataRole.UserRole)
        ref = next((entry for entry in scan["refs"] if entry["id"] == ref_id), None) if scan else None
        enabled = item.checkState() == Qt.CheckState.Checked
        if ref is None or ref.get("enabled", True) == enabled:
            return
        ref["enabled"] = enabled
        self.project.invalidate()
        # Rebuild after this signal returns; the list must not clear its own item here.
        QTimer.singleShot(0, self._refresh)

    def _mode_changed(self):
        if self._building:
            return
        scan = self._current_scan()
        choice = self.mode_combo.currentData()
        if scan and choice:
            was_current = self.project.is_current()
            if choice.startswith("follow:"):
                scan["mode"], scan["follow"] = "follow", choice.split(":", 1)[1]
            else:
                scan["mode"] = "own" if choice in ("own", "own+shared") else choice
                scan["include_shared"] = choice == "own+shared"
                scan.pop("follow", None)
            self.project.invalidate()
            self._update_without_rerun(was_current, [scan])
            self._refresh()

    def _exclusion_key(self):
        scan, ref = self._current_scan(), self._current_ref()
        return (str(self.project.directory), scan["id"], ref["id"]) if self.project and scan and ref else None

    def _exclusion_undo_context(self):
        return (self.edit_exclusions.isChecked() or
                getattr(self, "_last_mask_target", None) == "exclusion" or
                QApplication.focusWidget() in (self.reference_view, self.reference_view.viewport()))

    def _toggle_exclusion_edit(self, editing):
        if editing and (not self._current_ref() or (self.task and self.task.isRunning())):
            self.edit_exclusions.setChecked(False)
            return
        self.exclusion_toolbar.setVisible(editing)
        self.reference_view.cancel_current_action()
        if editing:
            self.result_view.cancel_current_action()
            self.show_exclusions.setChecked(True)
            self._exclusion_tool_changed()
            self.reference_view.setFocus()
            self.status.setText(
                "Paint the reference colors to use only (green). Edits save automatically; rerun Color Match when done"
                if self._current_mask_mode() == "include" else
                "Mark untrusted reference colors in red. Edits save automatically; rerun Color Match when done")

    def _set_exclusion_tool(self, mode):
        if mode not in self._exclusion_tool_buttons:
            return
        self._exclusion_mode = mode
        for button_mode, button in self._exclusion_tool_buttons.items():
            blocked = button.blockSignals(True)
            button.setChecked(button_mode == mode)
            button.blockSignals(blocked)
        self._exclusion_tool_changed()

    def _exclusion_tool_changed(self, *_):
        mode = self._exclusion_mode
        self.exclusion_diameter_row.setVisible(mode in {"paint", "erase"})
        if mode == "erase" and self._exclusion_last_tool != "erase":
            self._exclusion_tool_before_erase = self._exclusion_last_tool
        self._exclusion_last_tool = mode
        if not self.edit_exclusions.isChecked():
            return
        self.reference_view.cancel_current_action()
        self.reference_view.set_mode(mode)
        self.exclusion_diameter.setEnabled(mode in {"paint", "erase"})
        self._exclusion_diameter_changed(self.exclusion_diameter.value())

    def _exclusion_diameter_changed(self, value):
        self.reference_view.set_brush_radius(max(1, round(value / 2)))
        self.reference_view.set_brush_hardness(100)

    def _exclusion_stroke(self, points, erase):
        self._edit_exclusion(points, "erase" if erase else "paint")

    def _current_mask_mode(self):
        ref = self._current_ref()
        return ref.get("mask_mode", "exclude") if ref else "exclude"

    def _mask_mode_changed(self, include):
        if self._building:
            return
        ref = self._current_ref()
        if ref is None or (self.task and self.task.isRunning()):
            self._update_mask_mode_ui()
            return
        try:
            changed = self.project.set_reference_mask_mode(ref, "include" if include else "exclude")
        except Exception as exc:
            self._error(str(exc))
            self._update_mask_mode_ui()
            return
        if not changed:
            return
        if ref.get("exclude_mask"):
            self._after_exclusion_edit()
        else:
            self._update_reference_item(ref)
            self._refresh_exclusion_overlay()
            self.status.setText("Use only painted: paint the matching part of the reference"
                                if include else "Exclude painted: paint untrusted reference colors")

    def _update_mask_mode_ui(self):
        ref = self._current_ref()
        include = self._current_mask_mode() == "include"
        enabled = bool(ref) and not (self.task and self.task.isRunning())
        buttons = (self.mask_mode_exclude, self.mask_mode_include)
        if enabled:  # enable first: an animated style may skip state changes while disabled
            for button in buttons:
                button.setEnabled(True)
        building, self._building = self._building, True
        try:
            (self.mask_mode_include if include else self.mask_mode_exclude).setChecked(True)
        finally:
            self._building = building
        for button in buttons:
            button.setEnabled(enabled)
            button.update()
        self.reference_view.overlay_color = (60, 200, 90) if include else (255, 100, 70)

    def _reference_mask_key(self, ref):
        return (str(self.project.directory), ref.get("exclude_mask"), ref.get("exclude_mask_digest"),
                ref["path"])

    def _remember_reference_mask(self, ref, mask):
        if ref.get("exclude_mask"):
            mask = np.array(mask, dtype=np.uint8, copy=True)
            mask.flags.writeable = False
            self._reference_masks[self._reference_mask_key(ref)] = mask

    def _reference_mask(self, ref):
        if ref.get("exclude_mask"):
            key = self._reference_mask_key(ref)
            mask = self._reference_masks.get(key)
            if mask is None:
                mask = engine.load_reference_mask(self.project.directory, ref)
                mask.flags.writeable = False  # shared: edits always work on copies
                self._reference_masks[key] = mask
                while len(self._reference_masks) > 6:
                    self._reference_masks.pop(next(iter(self._reference_masks)))
            return mask
        width, height = engine.validate_image(ref["path"])
        return np.zeros((height, width), np.uint8)  # nothing painted yet

    @staticmethod
    def _exclusion_snapshot(ref):
        return {key: ref[key] for key in ("exclude_mask", "exclude_mask_digest") if key in ref}

    def _after_exclusion_edit(self):
        scan = self._current_scan()
        scan["evaluation"] = None
        scan["local_dirty"] = bool(scan["layers"])
        for layer in scan["layers"]:
            layer["needs_refit"] = True
        scan.pop("refine_signature", None)
        self._metric_failures.clear()
        self.project.save()
        self._last_mask_target = "exclusion"
        self._update_reference_item(self._current_ref())
        self._refresh_exclusion_overlay()
        for affected in self.project.data["scans"]:
            self._update_scan_item(affected)
        self._refresh_color_metrics()
        self._update_edit_menu()
        self.view_info.setText("Out of date · reference sampling mask changed · rerun needed")
        self.status.setText("Reference sampling mask saved. Run Color Match to update the LUT and corrections")

    def _edit_exclusion(self, points, kind):
        if not self.edit_exclusions.isChecked() or (self.task and self.task.isRunning()):
            return
        ref = self._current_ref()
        if ref is None:
            return
        try:
            previous = self._exclusion_snapshot(ref)
            mask = self._reference_mask(ref)
            changed = engine.edit_reference_exclusion(mask, points, kind,
                self.reference_view.brush_radius, self._reference_transform)
            if self.project.set_reference_exclusion(ref, changed):
                self._remember_reference_mask(ref, changed)
                undo, redo = self._exclusion_histories.setdefault(self._exclusion_key(), ([], []))
                undo.append(previous)
                redo.clear()
                self._after_exclusion_edit()
            else:
                self._refresh_exclusion_overlay()
        except Exception as exc:
            self.reference_view.cancel_current_action()
            self._refresh_exclusion_overlay()
            self._error(str(exc))

    def _clear_exclusions(self):
        if not self.project or not self._current_ref() or (self.task and self.task.isRunning()):
            return
        ref = self._current_ref()
        if not ref.get("exclude_mask"):
            return
        try:
            previous = self._exclusion_snapshot(ref)
            width, height = engine.validate_image(ref["path"])
            self.reference_view.cancel_current_action()
            if self.project.set_reference_exclusion(ref, np.zeros((height, width), np.uint8)):
                undo, redo = self._exclusion_histories.setdefault(self._exclusion_key(), ([], []))
                undo.append(previous)
                redo.clear()
                self._after_exclusion_edit()
            self._exclusion_tool_changed()
        except Exception as exc:
            self._error(str(exc))

    def _restore_exclusion(self, redo=False):
        if not self.project or not self._current_ref() or (self.task and self.task.isRunning()):
            return
        undo_stack, redo_stack = self._exclusion_histories.get(self._exclusion_key(), ([], []))
        source, destination = (redo_stack, undo_stack) if redo else (undo_stack, redo_stack)
        if not source:
            return
        ref = self._current_ref()
        restored = dict(ref)
        for key in ("exclude_mask", "exclude_mask_digest"):
            restored.pop(key, None)
        restored.update(source[-1])
        try:
            mask = self._reference_mask(restored)
            previous = self._exclusion_snapshot(ref)
            self.reference_view.cancel_current_action()
            self.project.set_reference_exclusion(ref, mask)
            self._remember_reference_mask(ref, mask)
            source.pop()
            destination.append(previous)
            self._after_exclusion_edit()
            self._exclusion_tool_changed()
        except Exception as exc:
            self._error(str(exc))

    def _refresh_exclusion_overlay(self, *_):
        key = self._exclusion_key()
        if key != self._exclusion_view_key:
            self.reference_view.cancel_current_action()
            self._exclusion_view_key = key
        ref = self._current_ref()
        enabled = bool(ref and Path(ref["path"]).exists() and not (self.task and self.task.isRunning()))
        self.edit_exclusions.setEnabled(enabled)
        self.clear_exclusions.setEnabled(enabled and bool(ref.get("exclude_mask")))
        self._update_mask_mode_ui()
        if not enabled and self.edit_exclusions.isChecked():
            self.edit_exclusions.setChecked(False)
        mask_key = self._reference_mask_key(ref) if enabled else None
        if (enabled and ref.get("exclude_mask") and mask_key not in self._reference_masks
                and mask_key not in self._reference_mask_failures):
            # First view of this mask: read and verify it off the GUI thread.
            directory, snapshot = self.project.directory, dict(ref)

            def load():
                try:
                    return engine.load_reference_mask(directory, snapshot), None
                except Exception as exc:
                    return None, str(exc)

            def loaded(outcome, key=mask_key):
                mask, error = outcome
                if error is not None:
                    # The synchronous path below then reports the problem as before.
                    self._reference_mask_failures.add(key)
                else:
                    mask.flags.writeable = False
                    self._reference_masks[key] = mask
                if self._exclusion_key() is not None and self._current_ref() is not None                         and self._reference_mask_key(self._current_ref()) == key:
                    self._refresh_exclusion_overlay()
            self.reference_view.set_exclusion(None)
            self._in_background(("reference-mask",) + mask_key, load, loaded)
            return
        try:
            mask = self._reference_mask(ref) if enabled else None
            self.reference_view.set_exclusion(mask, self._reference_transform,
                self.show_exclusions.isChecked() or self.edit_exclusions.isChecked())
        except Exception as exc:
            self.reference_view.set_exclusion(None)
            self.edit_exclusions.setEnabled(False)
            self.status.setText(str(exc) + "; Clear Mask to reset")
            if self.edit_exclusions.isChecked():
                self.edit_exclusions.setChecked(False)
        if enabled and self.edit_exclusions.isChecked() and self.reference_view.mode == "pan":
            self._exclusion_tool_changed()

    def _reference_alignment(self, scan, ref, alignment):
        if not ref or not alignment or not alignment.exists() or not Path(ref["path"]).exists():
            return None
        signature = engine.alignment_signature(scan, ref)
        key = self._exclusion_key()
        stored = self.report.get("references", {}).get(ref["id"], {}).get("alignment_signature")
        if stored == signature or (stored is None and self.project.is_current()):
            self._alignment_view_inputs[key] = signature
        if self._alignment_view_inputs.get(key) != signature:
            return None
        return np.load(alignment)

    def _reference_label(self, ref):
        if not Path(ref["path"]).exists():
            return f"{Path(ref['path']).name} (missing)"
        status = self.report.get("references", {}).get(ref["id"], {}).get("status", "pending")
        mask_text = ((" · use only mask" if ref.get("mask_mode") == "include" else " · exclusions")
                     if ref.get("exclude_mask") else "")
        return f"{Path(ref['path']).name} ({status}{mask_text})"

    def _update_reference_item(self, ref):
        if ref is None:
            return
        for index in range(self.ref_list.count()):
            item = self.ref_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == ref["id"]:
                item.setText(self._reference_label(ref))
                break

    def _scan_status(self, scan):
        if scan["id"] in self._applying_lut:
            return "Applying LUT…", "#edf0f3"
        state = self.report.get("scans", {}).get(scan["id"], {})
        if scan["path"] and not Path(scan["path"]).exists():
            return "Missing source", "#fbe4e4"
        if not self.project.is_current():
            run = self.project.run_dir()
            if run and (run / "base" / f"{scan['id']}.png").exists():
                return "Previous result · rerun needed", "#fff0d6"
            return "Pending", "#edf0f3"
        if state.get("error"):
            return state["error"], "#fbe4e4"
        if any(layer.get("visible", True) and not layer.get("model")
               and layer.get("mask_changed_since_refine") for layer in scan["layers"]):
            return "Needs Refine All", "#fff0d6"
        if scan.get("local_dirty"):
            if any(layer.get("visible", True) and
                   (not layer.get("model") or layer.get("needs_refit", False))
                   for layer in scan["layers"]):
                return "Needs Refine All", "#fff0d6"
            return "Updating local layers", "#e2f3e7"
        failed_refs = any(self.report.get("references", {}).get(ref["id"], {}).get("status") == "failed"
                          for ref in scan["refs"])
        if failed_refs:
            return "Reference failed", "#fbe4e4"
        if state.get("status") == "ready":
            return "Ready", "#e2f3e7"
        if scan["refs"]:
            return "Reference attached", "#edf0f3"
        if scan["mode"] == "follow":
            return "Pending", "#edf0f3"
        return "Needs shared LUT", "#edf0f3"

    def _scan_number(self, scan_id):
        ids = [item["id"] for item in self.project.data["scans"]]
        return ids.index(scan_id) + 1 if scan_id in ids else None

    def _lut_label(self, scan):
        if scan["mode"] == "follow":
            return f"Uses #{self._scan_number(scan.get('follow'))} LUT"
        if scan["mode"] == "own":
            return "Own + joint LUT" if scan.get("include_shared") else "Own LUT"
        return "Shared LUT"

    def _scan_detail(self, scan, status):
        """Second line of a scan row: only what needs attention (empty when ready)."""
        return "" if status == "Ready" else status

    def _fill_scan_item(self, item, scan, index):
        status, color = self._scan_status(scan)
        name = Path(scan["path"]).name if scan["path"] else "Empty image pair"
        item.setText(f"{index + 1}. {name}\n{self._lut_label(scan)} · {len(scan['refs'])} reference(s) · {status}")
        item.setData(Qt.ItemDataRole.UserRole, scan["id"])
        item.setData(SCAN_HAS_REF_ROLE, bool(scan["refs"]))
        item.setData(SCAN_COLOR_ROLE, color)
        item.setData(SCAN_NAME_ROLE, f"{index + 1}. {tr(name)}")
        item.setData(SCAN_DETAIL_ROLE, tr(self._scan_detail(scan, status)))
        item.setData(SCAN_REF_COUNT_ROLE, len(scan["refs"]))
        mode = "own+shared" if scan["mode"] == "own" and scan.get("include_shared") else scan["mode"]
        item.setData(SCAN_INDICATOR_ROLE, LUT_MODE_COLORS.get(mode) if scan["refs"] else None)
        item.setData(SCAN_THUMB_ROLE, self._scan_thumbnail(scan))
        item.setToolTip(f"{name}\n{self._lut_label(scan)} · {len(scan['refs'])} reference(s) · {status}\n"
                        "Left bar: blue Own + joint, purple Own LUT, teal Shared + joint")

    def _scan_thumbnail(self, scan):
        path = Path(scan["path"]) if scan["path"] else None
        if path is None or not path.exists():
            return QPixmap()
        stat = path.stat()
        key = ("scan", str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._mask_thumb_cache:
            def store(image, key=key, scan_id=scan["id"]):
                self._mask_thumb_cache[key] = QPixmap.fromImage(image)
                for row in range(self.scan_list.count()):
                    item = self.scan_list.item(row)
                    if item.data(Qt.ItemDataRole.UserRole) == scan_id:
                        item.setData(SCAN_THUMB_ROLE, self._mask_thumb_cache[key])
            self._in_background(key, lambda: _scan_thumbnail_image(path), store)
            return QPixmap()
        return self._mask_thumb_cache[key]

    def _scan_hovered(self, item):
        """Hovering a scan that uses another scan's LUT outlines that scan (no scrolling)."""
        source = None
        if item is not None and self.project:
            scan = next((entry for entry in self.project.data["scans"]
                         if entry["id"] == item.data(Qt.ItemDataRole.UserRole)), None)
            if scan and scan["mode"] == "follow":
                source = scan.get("follow")
        for row in range(self.scan_list.count()):
            other = self.scan_list.item(row)
            linked = source is not None and other.data(Qt.ItemDataRole.UserRole) == source
            if bool(other.data(SCAN_LINKED_ROLE)) != linked:
                other.setData(SCAN_LINKED_ROLE, linked)

    def _update_scan_item(self, scan):
        for index in range(self.scan_list.count()):
            item = self.scan_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == scan["id"]:
                self._fill_scan_item(item, scan, index)
                break
        if scan["id"] == self.current_scan_id:
            self._refresh_color_metrics()

    def _refresh(self):
        if self.project and self.project.normalize_lut_modes():
            self.project.invalidate()
        quick_dir = self.project.directory / "cache" / "previews" if self.project else None
        self.result_view.quick_cache_dir = self.reference_view.quick_cache_dir = quick_dir
        self._prefetch_timer.start(1500)  # prepare quick previews once things settle
        self._building = True
        metric, passes = engine.fit_settings(self.project.data if self.project else {})
        (self.fit_de76 if metric == "de76" else self.fit_rgb).setChecked(True)
        self.passes_box.setValue(passes)
        self.project_label.setText(self.project.directory.name if self.project else "No project")
        self.project_label.setToolTip(str(self.project.directory) if self.project else "")
        self.scan_list.clear()
        if self.project:
            for index, scan in enumerate(self.project.data["scans"]):
                item = QListWidgetItem()
                self._fill_scan_item(item, scan, index)
                self.scan_list.addItem(item)
                if scan["id"] == self.current_scan_id:
                    self.scan_list.setCurrentItem(item)
            if self.current_scan_id is None and self.project.data["scans"]:
                self.current_scan_id = self.project.data["scans"][0]["id"]
                self.scan_list.setCurrentRow(0)
        self._building = False
        self._refresh_details()

    def _refresh_details(self):
        self._building = True
        scan = self._current_scan()
        current_ref_item = self.ref_list.currentItem()
        ref_id = current_ref_item.data(Qt.ItemDataRole.UserRole) if current_ref_item else None
        layer_id = self.layer_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.layer_list.currentItem() else None
        self.ref_list.clear()
        self.layer_list.clear()
        if scan:
            self.mode_combo.clear()
            combo = self.mode_combo
            if scan["refs"]:
                combo.add_described_item(
                    "Own LUT", "own",
                    "Learn a LUT only from this scan's references. It does not help other scans")
                combo.add_described_item(
                    "Shared + joint training", "shared",
                    "Help train one shared LUT together with the other shared scans, and use "
                    "that shared LUT. Best when this scan's references are few or partial")
                combo.add_described_item(
                    "Own + joint training", "own+shared",
                    "Use a LUT learned from this scan's references, and also help train the "
                    "shared LUT for other scans. Recommended")
            else:
                combo.add_described_item(
                    "Shared", "shared",
                    "Use the shared LUT learned from the scans that have references")
                for number, other in enumerate(self.project.data["scans"], 1):
                    if other is not scan and other["mode"] == "own" and other["refs"]:
                        name = Path(other["path"]).name if other["path"] else ""
                        combo.add_described_item(
                            f"Use #{number} {name} LUT", f"follow:{other['id']}",
                            f"Use the Own LUT of scan #{number}, for example a back cover "
                            "following its front cover")
            current = (f"follow:{scan.get('follow')}" if scan["mode"] == "follow" else
                       "own+shared" if scan["mode"] == "own" and scan.get("include_shared") else
                       scan["mode"])
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(current)))
            self.mode_combo.setToolTip(self.mode_combo.itemData(
                self.mode_combo.currentIndex(), Qt.ItemDataRole.ToolTipRole) or "")
            for ref in scan["refs"]:
                state = self.report.get("references", {}).get(ref["id"], {})
                item = QListWidgetItem(self._reference_label(ref))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                enabled = ref.get("enabled", True)
                item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
                if not enabled:
                    item.setForeground(QColor("#9aa0a6"))
                    item.setToolTip("Disabled: not used for color matching. Check to use it again")
                elif state.get("sample_pixels") is not None:
                    item.setToolTip(f"{state['sample_pixels']:,} reliable sample pixels at working resolution")
                elif state.get("error"):
                    item.setToolTip(state["error"])
                item.setData(Qt.ItemDataRole.UserRole, ref["id"])
                self.ref_list.addItem(item)
                if ref["id"] == ref_id:
                    self.ref_list.setCurrentItem(item)
            if self.ref_list.currentItem() is None and self.ref_list.count():
                self.ref_list.setCurrentRow(0)
            add_item = QListWidgetItem("＋  New layer")
            add_item.setData(Qt.ItemDataRole.UserRole, "+")
            add_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.layer_list.addItem(add_item)
            for layer in reversed(scan["layers"]):
                item = QListWidgetItem(layer["name"])
                item.setData(Qt.ItemDataRole.UserRole, layer["id"])
                result_thumb, mask_thumb = self._layer_thumbnails(scan, layer)
                item.setData(LAYER_RESULT_THUMB_ROLE, result_thumb)
                item.setData(LAYER_THUMB_ROLE, mask_thumb)
                item.setData(LAYER_OVERLAY_ROLE, layer.get("overlay_visible", False))
                item.setData(LAYER_STRENGTH_ROLE, layer.get("strength", 100))
                item.setToolTip("Checkbox: apply layer · Eye: show overlay · Percentage: click to edit, wheel ±10%")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
                item.setCheckState(Qt.CheckState.Checked if layer["visible"] else Qt.CheckState.Unchecked)
                self.layer_list.addItem(item)
                if layer["id"] == layer_id:
                    self.layer_list.setCurrentItem(item)
            global_item = QListWidgetItem(scan.get("global_name", "global"))
            global_item.setData(Qt.ItemDataRole.UserRole, GLOBAL_LAYER_ID)
            global_item.setData(LAYER_RESULT_THUMB_ROLE, self._global_thumbnail(scan))
            global_item.setFlags((global_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                                 & ~Qt.ItemFlag.ItemIsDragEnabled)
            global_item.setCheckState(Qt.CheckState.Checked if scan.get("global_visible", True)
                                      else Qt.CheckState.Unchecked)
            global_item.setToolTip(
                "Checkbox: apply the full-scan LUT and spatial correction · double-click to rename\n"
                "Percentage: global strength. 100% is one pass, each further 100% one more pass, "
                "blended in between. Unless you set it, it is chosen automatically: the lowest "
                "ΔE76. Click to edit, wheel ±10%, right-click for automatic")
            state = self.report.get("scans", {}).get(scan["id"], {})
            if self.project.is_current() and state.get("status") == "ready":
                global_item.setData(LAYER_STRENGTH_ROLE, engine.global_strength(scan, state))
                global_item.setData(LAYER_STRENGTH_MAX_ROLE, self._global_strength_limit(scan, state))
            self.layer_list.addItem(global_item)
            if layer_id == GLOBAL_LAYER_ID:
                self.layer_list.setCurrentItem(global_item)
            if self.layer_list.currentItem() is None:
                self.layer_list.setCurrentRow(1 if scan["layers"] else self.layer_list.count() - 1)
        self._building = False
        self._refresh_ref()
        self._refresh_layer()

    def _in_background(self, key, job, done):
        """Run ``job`` on the loader pool once per key; call ``done(result)`` on the GUI thread."""
        if key in self._loading:
            return
        self._loading.add(key)

        def finished(future):
            def deliver():
                self._loading.discard(key)
                try:
                    result = future.result()
                except Exception:
                    return
                done(result)
            try:
                self._loader_bridge.call.emit(deliver)
            except RuntimeError:
                pass  # the window is gone
        self._loader.submit(job).add_done_callback(finished)

    def _display_source(self, scan):
        """The file the scan preview shows (as chosen by _refresh_image)."""
        run = self.project.run_dir()
        local = (self.project.directory / scan["local_output"]
                 if scan.get("local_output") and not scan.get("local_dirty") else None)
        calibrated = run / "base" / f"{scan['id']}.png" if run else None
        previous = scan.get("previous_result_run")
        previous_result = (self.project.directory / "cache" / "runs" / previous / "base" /
                           f"{scan['id']}.png" if previous else None)
        for candidate in (local, calibrated if scan.get("global_visible", True) else None,
                          previous_result if scan.get("global_visible", True) else None):
            if candidate and candidate.exists():
                return candidate
        return Path(scan["path"])

    def _quick_prefetch_jobs(self):
        """Every image a scan switch shows, nearest scans first."""
        scans = [scan for scan in self.project.data["scans"] if scan["path"]]
        ids = [scan["id"] for scan in scans]
        current = ids.index(self.current_scan_id) if self.current_scan_id in ids else 0
        ordered = sorted(scans, key=lambda scan: abs(ids.index(scan["id"]) - current))
        directory = self.result_view.quick_cache_dir
        jobs = []
        for scan in ordered:
            try:
                source = self._display_source(scan)
                original = Path(scan["path"])
                for path in (source, original):
                    jobs.append((path, None, None, _quick_preview_file(directory, _image_key(path))))
                run = self.project.run_dir()
                for ref in scan["refs"]:
                    ref_path = Path(ref["path"])
                    alignment = run / "align" / f"{ref['id']}.npy" if run else None
                    if self._reference_alignment(scan, ref, alignment) is not None:
                        size = engine.validate_image(str(source))
                        key = _image_key(ref_path, size, alignment)
                        jobs.append((ref_path, alignment, size, _quick_preview_file(directory, key)))
                    else:
                        jobs.append((ref_path, None, None,
                                     _quick_preview_file(directory, _image_key(ref_path))))
            except (OSError, ValueError):
                continue
        return jobs

    def _start_quick_prefetch(self):
        if not self.project or self.result_view.quick_cache_dir is None:
            return
        if self._prefetch_worker is not None:
            self._prefetch_again = True
            return
        worker = QuickPreviewPrefetch(self._quick_prefetch_jobs())
        self._prefetch_worker = worker

        def finished():
            self._prefetch_worker = None
            worker.deleteLater()
            if self._prefetch_again and not self._close_when_idle:
                self._prefetch_again = False
                self._prefetch_timer.start(500)
        worker.finished.connect(finished)
        worker.start(QThread.Priority.LowPriority)  # yields to the user's own work

    def _thumbnails_ready(self):
        """Fill in thumbnails that finished loading, without rebuilding the list."""
        scan = self._current_scan()
        if scan is None:
            return
        layers = {layer["id"]: layer for layer in scan["layers"]}
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            layer_id = item.data(Qt.ItemDataRole.UserRole)
            if layer_id == GLOBAL_LAYER_ID:
                item.setData(LAYER_RESULT_THUMB_ROLE, self._global_thumbnail(scan))
            elif layer_id in layers:
                result_thumb, mask_thumb = self._layer_thumbnails(scan, layers[layer_id])
                item.setData(LAYER_RESULT_THUMB_ROLE, result_thumb)
                item.setData(LAYER_THUMB_ROLE, mask_thumb)

    def _placeholder_thumbnail(self):
        if not hasattr(self, "_placeholder_pixmap"):
            self._placeholder_pixmap = _pixmap_from_image(Image.new("RGB", (32, 32), "#c9c9c9"))
        return self._placeholder_pixmap

    def _layer_thumbnails(self, scan, layer):
        """A layer's (result, mask) thumbnails.

        Cached apart: the mask thumbnail depends only on its mask, so editing one layer
        never rebuilds another's. Result thumbnails of all layers come from a single
        decode of the new result. Until a thumbnail is ready the previous one stays.
        """
        mask_path = self.project.directory / layer["mask"]
        run = self.project.run_dir()
        local = (self.project.directory / scan["local_output"]
                 if scan.get("local_output") and not scan.get("local_dirty") else None)
        base = (run / "base" / f"{scan['id']}.png"
                if run and scan.get("global_visible", True) else None)
        result_path = next((path for path in (local, base, Path(scan["path"]) if scan["path"] else None)
                            if path and path.exists()), None)
        resident = self._resident_masks.get(mask_path)
        if resident is not None:  # being edited: newer than the file on disk
            mask_version = ("resident", self._mask_revisions[mask_path])
        else:
            mask_stat = mask_path.stat() if mask_path.exists() else None
            mask_version = mask_stat.st_mtime_ns if mask_stat else None
        last_result, last_mask = self._last_thumbnails.get(
            layer["id"], (self._placeholder_thumbnail(), self._placeholder_thumbnail()))
        mask_key = ("mask", str(mask_path), mask_version)
        mask_entry = self._mask_thumb_cache.get(mask_key)
        if mask_entry is None:
            def store(result, key=mask_key):
                image, box, size = result
                self._mask_thumb_cache[key] = (QPixmap.fromImage(image), box, size)
                self._trim_thumbnail_cache()
                self._thumbnails_ready()
            self._in_background(mask_key, lambda: _mask_thumbnail_image(mask_path, resident), store)
            return last_result, last_mask
        mask_pixmap, box, size = mask_entry
        if box is None or result_path is None:
            result_pixmap = self._placeholder_thumbnail()  # nothing painted, or no image yet
        else:
            stat = result_path.stat()
            result_key = ("result", str(result_path), stat.st_mtime_ns, stat.st_size, box, size)
            result_pixmap = self._mask_thumb_cache.get(result_key)
            if result_pixmap is None:
                self._pending_result_thumbs.setdefault(
                    (str(result_path), stat.st_mtime_ns, stat.st_size), {})[result_key] = (box, size)
                self._result_thumb_timer.start(0)
                result_pixmap = last_result
        self._last_thumbnails[layer["id"]] = (result_pixmap, mask_pixmap)
        return result_pixmap, mask_pixmap

    def _load_result_thumbnails(self):
        """Crop every waiting layer's result thumbnail from one decode per result image."""
        pending, self._pending_result_thumbs = self._pending_result_thumbs, {}
        for (path, modified, size), crops in pending.items():
            def store(images):
                for key, image in images.items():
                    self._mask_thumb_cache[key] = QPixmap.fromImage(image)
                self._trim_thumbnail_cache()
                self._thumbnails_ready()
            self._in_background(("result-thumbs", path, modified, size, tuple(sorted(crops, key=str))),
                                lambda path=path, crops=crops: _result_thumbnail_images(Path(path), crops),
                                store)

    def _trim_thumbnail_cache(self):
        while len(self._mask_thumb_cache) > 160:
            self._mask_thumb_cache.pop(next(iter(self._mask_thumb_cache)))

    def _global_thumbnail(self, scan):
        run = self.project.run_dir()
        base = run / "base" / f"{scan['id']}.png" if run else None
        path = base if base and base.exists() else Path(scan["path"]) if scan["path"] else None
        if path is None or not path.exists():
            return QPixmap()
        stat = path.stat()
        key = ("global", str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._mask_thumb_cache:
            def store(image, key=key):
                self._mask_thumb_cache[key] = QPixmap.fromImage(image)
                self._thumbnails_ready()
            self._in_background(key, lambda: _global_thumbnail_image(path), store)
            return self._last_thumbnails.get(("global", scan["id"]), self._placeholder_thumbnail())
        self._last_thumbnails[("global", scan["id"])] = self._mask_thumb_cache[key]
        return self._mask_thumb_cache[key]

    def _refresh_ref(self):
        self._refresh_image()

    def _refresh_image(self):
        self._active_preview_signature = None
        scan = self._current_scan()
        if scan is None or not scan["path"]:
            self._reference_transform = None
            self._displayed_scan_id = None
            self.result_view.set_image(None)
            self.result_view.set_original(None)
            self.reference_view.set_image(None)
            self._refresh_exclusion_overlay()
            self.view_info.setText("Drop a scan into the left preview to begin.")
            self._refresh_color_metrics()
            return
        source = Path(scan["path"])
        preserve_view = self._displayed_scan_id == scan["id"]
        self._displayed_scan_id = scan["id"]
        if not source.exists():
            self._reference_transform = None
            self.result_view.set_image(None)
            self.result_view.set_original(None)
            self.reference_view.set_image(None)
            self._refresh_exclusion_overlay()
            self.view_info.setText("Scan file is missing. Drop its new location into the left preview.")
            self._refresh_color_metrics()
            return
        run = self.project.run_dir()
        calibrated = run / "base" / f"{scan['id']}.png" if run else None
        previous_run = scan.get("previous_result_run")
        previous_result = (self.project.directory / "cache" / "runs" / previous_run /
                           "base" / f"{scan['id']}.png" if previous_run else None)
        showing_previous = False
        local = (self.project.directory / scan["local_output"]
                 if scan.get("local_output") and not scan.get("local_dirty") else None)
        if local and local.exists():
            source = local
        elif scan.get("global_visible", True) and calibrated and calibrated.exists():
            source = calibrated
        elif (scan.get("global_visible", True) and previous_result
              and previous_result.exists()):
            source = previous_result
            showing_previous = True
        result_fits, reference_key = self.result_view._fit_serial, self.reference_view._current_image_key
        self.result_view.set_image(source, preserve_view=preserve_view)
        self.result_view.set_original(Path(scan["path"]))
        if (local and source == local and scan.get("global_visible", True)
                and calibrated and calibrated.exists()):
            self.result_view.preload_image(calibrated)
        ref = self._current_ref()
        aligned = run / "reference-preview" / f"{ref['id']}.png" if run and ref else None
        alignment = run / "align" / f"{ref['id']}.npy" if run and ref else None
        self._reference_transform = self._reference_alignment(scan, ref, alignment)
        if self._reference_transform is not None:
            self.reference_view.set_image(Path(ref["path"]),
                                          virtual_size=engine.validate_image(str(source)),
                                          preserve_view=preserve_view, alignment_path=alignment)
        else:
            self.reference_view.set_image(Path(ref["path"]) if ref else None,
                                          preserve_view=preserve_view)
        view_moved = (self.result_view._fit_serial != result_fits
                      or self.reference_view._current_image_key != reference_key)
        if view_moved and (self._reference_transform is not None or self.result_view._user_view):
            # An aligned reference shares the scan's coordinates: it always shows exactly
            # the scan's view. An unaligned one keeps its own fit until the user zooms or
            # pans. Nothing is re-synced when neither view changed, so refreshing (adding
            # or deleting a layer...) never nudges the reference by a rounding step.
            self._sync_reference_to_result()
        self._refresh_exclusion_overlay()
        state = self.report.get("references", {}).get(ref["id"], {}) if ref else {}
        self.view_info.setText(
            ("Previous result · new match failed · " if showing_previous else
             "Previous result · rerun needed · " if not self.project.is_current() and source != Path(scan["path"])
             else "Out of date · " if not self.project.is_current() else "")
            + (f"Ref: {state.get('inliers', 0)} inliers · {state.get('inlier_ratio', 0):.2f}"
               if state.get("status") == "aligned" else state.get("error", "Choose a reference or review the scan"))
        )
        self.view_info.setToolTip(self.view_info.text())
        self._refresh_mask()
        self._refresh_color_metrics()
        if scan["layers"] and self.project.is_current() and not scan.get("local_dirty"):
            self._queue_fast_local_preview(scan, warm=True)

    def _result_refitted(self):
        """The scan view fitted itself again (e.g. its preview arrived): follow it."""
        if self._reference_transform is not None:
            self._sync_reference_to_result()

    def _sync_reference_to_result(self):
        view = self.result_view
        self.reference_view.sync_view(view.transform(),
                                      view.mapToScene(view.viewport().rect().center()),
                                      view.source_size)

    def _refresh_color_metrics(self):
        """One line: 'ΔE76: Mask a → b | Global x → y | Final x → z'."""
        self.color_metrics.hide()
        self.metric_separator.hide()
        scan = self._current_scan()
        if scan is None:
            self.global_metric.hide()
            return
        if not self.project.is_current():
            self.global_metric.setText("ΔE76: out of date")
            self.global_metric.show()
            return
        parts = [part for part in (self._mask_delta(scan), *self._delta_summary(scan)) if part]
        if parts:
            self.global_metric.setText("ΔE76: " + " | ".join(parts))
            self.global_metric.show()
        else:
            self.global_metric.hide()

    def _mask_delta(self, scan):
        """'Mask off → on' for the selected fitted layer, measured inside its mask."""
        layer = self._current_layer()
        if layer is None or not layer.get("model"):
            return None
        if scan.get("local_dirty"):
            return "Mask updating…"
        evaluation = scan.get("evaluation") or {}
        effects = evaluation.get("layer_effects")
        if effects is None:
            key = (self.project.directory, scan["id"], self._local_signature(scan))
            if key in self._metric_failures:
                return None
            self._schedule_layer_metric()
            return "Mask updating…"
        score = effects.get(layer["id"])
        if not score:
            return None
        return f"Mask {score['off_delta_e76']:.2f} → {score['on_delta_e76']:.2f}"

    def _delta_summary(self, scan):
        """['Global original → global', 'Final original → shown result']."""
        state = self.report.get("scans", {}).get(scan["id"], {})
        spatial = state.get("spatial", {})
        if not spatial:
            return []
        key = "spatial_delta_e76" if spatial.get("accepted") else "global_delta_e76"
        global_value = state.get("base_delta_e76", spatial.get(key))
        raw = spatial.get("raw_delta_e76")
        if global_value is None:
            return []
        if raw is None:  # runs made before before/after values were recorded
            return [f"Global {global_value:.2f}"]
        parts = [f"Global {raw:.2f} → {global_value:.2f}"]
        visible = [layer for layer in scan["layers"] if layer.get("visible", True)]
        if not visible:
            final = global_value if scan.get("global_visible", True) else raw
        elif scan.get("local_dirty"):
            final = "updating"
        else:
            evaluation = scan.get("evaluation")
            final = evaluation.get("final_delta_e76", "missing") if evaluation else "measuring"
            if final == "measuring":
                self._schedule_layer_metric()
        if isinstance(final, float):
            parts.append(f"Final {raw:.2f} → {final:.2f}")
        elif final in ("updating", "measuring"):
            parts.append("Final updating…")
        return parts

    def _schedule_layer_metric(self):
        if self._metric_scheduled or self._metric_worker is not None:
            return
        if self.task and self.task.isRunning():
            return
        self._metric_scheduled = True
        QTimer.singleShot(0, self._start_layer_metric)

    def _start_layer_metric(self):
        self._metric_scheduled = False
        scan = self._current_scan()
        if (not scan or not self.project or not self.project.is_current()
                or scan.get("local_dirty") or self._dirty_masks
                or self._metric_worker is not None
                or (self.task and self.task.isRunning())):
            return
        directory, scan_id, signature = (self.project.directory, scan["id"],
                                         self._local_signature(scan))
        worker = LayerMetricThread(directory, copy.deepcopy(self.project.data), scan_id)
        self._metric_worker = worker
        worker.ready.connect(lambda scores: self._layer_metric_ready(
            directory, scan_id, signature, scores))
        worker.failed.connect(lambda message: self._layer_metric_failed(
            directory, scan_id, signature, message))
        worker.finished.connect(self._layer_metric_finished)
        worker.finished.connect(worker.deleteLater)
        # Metrics only replace a text when done; they must never slow the image down.
        worker.start(QThread.Priority.LowPriority)

    def _layer_metric_ready(self, directory, scan_id, signature, scores):
        if not self.project or self.project.directory != directory:
            return
        scan = next((item for item in self.project.data["scans"]
                     if item["id"] == scan_id), None)
        if scan is None or scan.get("local_dirty") or self._local_signature(scan) != signature:
            return
        scan["evaluation"] = scores
        if scan.get("local_output"):
            self._remember_local_variant(scan, scan["local_output"])
        self.project.save()
        if self.current_scan_id == scan_id:
            self._refresh_color_metrics()

    def _layer_metric_failed(self, directory, scan_id, signature, message):
        self._metric_failures[(directory, scan_id, signature)] = message
        if self.project and self.project.directory == directory and self.current_scan_id == scan_id:
            self._refresh_color_metrics()

    def _layer_metric_finished(self):
        self._metric_worker = None
        self._refresh_color_metrics()

    def _refresh_mask(self):
        layer, scan = self._current_layer(), self._current_scan()
        selected_path = self.project.directory / layer["mask"] if layer and self.project else None
        if selected_path in self._resident_masks:
            self.result_view.set_mask_array(selected_path, self._resident_masks[selected_path],
                                            self._mask_revisions[selected_path])
        else:
            self.result_view.set_mask(selected_path)
        overlay_paths = ([self.project.directory / entry["mask"] for entry in scan["layers"]
                          if entry.get("overlay_visible", False)] if scan and self.project else [])
        overrides = {path: (mask, self._mask_revisions[path])
                     for path, mask in self._resident_masks.items() if path in overlay_paths}
        self.result_view.set_overlay_masks(overlay_paths, selected_path, overrides)
        if layer and scan:
            self._thumbnails_ready()  # both thumbnails follow the edited mask
        self._refresh_mask_view_markers()

    def _refresh_mask_view_markers(self):
        current = self._current_layer()
        active_id = current["id"] if current and self.view_mask_box.isChecked() else None
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) not in ("+", GLOBAL_LAYER_ID):
                item.setData(LAYER_MASK_VIEW_ROLE,
                             item.data(Qt.ItemDataRole.UserRole) == active_id)
                item.setData(LAYER_RESULT_VIEW_ROLE,
                             bool(current and item.data(Qt.ItemDataRole.UserRole) == current["id"]
                                  and not self.view_mask_box.isChecked()))

    def _toggle_layer_mask_by_id(self, layer_id):
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == layer_id:
                self.layer_list.setCurrentItem(item)
                self.view_mask_box.setChecked(True)
                self._refresh_mask_view_markers()
                break

    def _show_layer_result_by_id(self, layer_id):
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == layer_id:
                self.layer_list.setCurrentItem(item)
                self.view_mask_box.setChecked(False)
                self._refresh_mask_view_markers()
                break

    def _toggle_layer_overlay_by_id(self, layer_id):
        scan = self._current_scan()
        if not scan:
            return
        layer = next((item for item in scan["layers"] if item["id"] == layer_id), None)
        if layer is None:
            return
        layer["overlay_visible"] = not layer.get("overlay_visible", False)
        self.project.save()
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == layer_id:
                item.setData(LAYER_OVERLAY_ROLE, layer["overlay_visible"])
                break
        self._refresh_mask()

    def _delete_layer_by_id(self, layer_id):
        if layer_id == GLOBAL_LAYER_ID:
            return
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == layer_id:
                self.layer_list.setCurrentItem(item)
                self.remove_layer()
                break

    def add_layer(self):
        scan, ref = self._current_scan(), self._current_ref()
        if scan is None or ref is None or not scan["path"]:
            self._error("Choose a scan and reference before adding a layer")
            return
        try:
            layer = engine.new_layer(self.project, scan, ref["id"])
            self._refresh_details()
            for row in range(self.layer_list.count()):
                if self.layer_list.item(row).data(Qt.ItemDataRole.UserRole) == layer["id"]:
                    self.layer_list.setCurrentRow(row)
            self.result_view.set_mode("paint")
        except Exception as exc:
            self._error(str(exc))

    def _prepare_layer_for_stroke(self):
        """Before painting on the global row (or nothing), create and select a layer."""
        if self._current_layer() is not None:
            return True
        if self.result_view.mode == "erase":
            return False  # nothing to erase without a layer
        layer = self._layer_for_mask_edit()
        if layer is None:
            return False
        self._refresh_mask()
        return True

    def _layer_for_mask_edit(self):
        layer = self._current_layer()
        if layer is not None:
            return layer
        scan = self._current_scan()
        if scan is None or not scan["path"]:
            self._error("Choose a scan before editing a layer")
            return None
        ref = self._current_ref() or next((item for item in scan["refs"] if item.get("enabled", True)), None)
        if ref is None:
            self._error("Add a reference before editing a layer")
            return None
        try:
            layer = engine.new_layer(self.project, scan, ref["id"])
            self._refresh_details()
            for row in range(self.layer_list.count()):
                if self.layer_list.item(row).data(Qt.ItemDataRole.UserRole) == layer["id"]:
                    self.layer_list.setCurrentRow(row)
                    break
            return layer
        except Exception as exc:
            self._error(str(exc))
            return None

    def _layer_changed(self):
        if not self._building:
            if self.result_view.mode == "polygon":
                self.result_view._clear_polygon()
            self._refresh_layer()

    def _layer_item_changed(self, item):
        if self._building or item.data(Qt.ItemDataRole.UserRole) == "+":
            return
        scan = self._current_scan()
        if scan is None:
            return
        if item.data(Qt.ItemDataRole.UserRole) == GLOBAL_LAYER_ID:
            self._set_global_visible(scan, item.checkState() == Qt.CheckState.Checked)
            return
        layer = next((entry for entry in scan["layers"]
                      if entry["id"] == item.data(Qt.ItemDataRole.UserRole)), None)
        if layer is None:
            return
        self._commit_layer_options(layer, scan, {"visible": item.checkState() == Qt.CheckState.Checked})

    def _set_global_visible(self, scan, visible):
        if scan.get("global_visible", True) == visible:
            return
        previous_output = scan.get("local_output")
        if (self.project.is_current() and not scan.get("local_dirty") and scan.get("local_output")
                and (self.project.directory / scan["local_output"]).exists()):
            self._remember_local_variant(scan, scan["local_output"])
        scan["global_visible"] = visible
        if any(layer.get("visible", True) for layer in scan["layers"]):
            cached = self._cached_local_variant(scan)
            scan["local_output"] = cached or previous_output
            scan["local_dirty"] = cached is None
            if cached is None:
                scan["evaluation"] = None
        else:
            scan["local_output"] = None
            scan["local_dirty"] = False
            scan["evaluation"] = None
        self.project.save()
        self._update_scan_item(scan)
        if (not scan["local_dirty"] or not previous_output
                or not (self.project.directory / previous_output).exists()):
            if not self._show_resident_result(scan):
                self._refresh_image()
        if scan["local_dirty"] and self.project.is_current():
            self._queue_local_render(scan)
        else:
            self.status.setText("Global correction enabled" if visible else "Global correction disabled")

    def _layer_order_changed(self):
        if self._building:
            return
        scan = self._current_scan()
        if scan is None:
            return
        by_id = {layer["id"]: layer for layer in scan["layers"]}
        ids = [self.layer_list.item(row).data(Qt.ItemDataRole.UserRole)
               for row in range(self.layer_list.count())]
        ordered = [by_id[layer_id] for layer_id in reversed(ids) if layer_id in by_id]
        if ordered == scan["layers"]:
            return
        scan["layers"] = ordered
        scan["local_dirty"] = True
        scan["evaluation"] = None
        self.project.save()
        self._update_scan_item(scan)
        if self.project.is_current():
            self._queue_local_render(scan)

    def _layer_context_menu(self, position, item):
        menu = QMenu(self)
        add_action = menu.addAction(tr("New layer"))
        auto_action = None
        layer = None
        global_selected = bool(item and item.data(Qt.ItemDataRole.UserRole) == GLOBAL_LAYER_ID)
        if item and item.data(Qt.ItemDataRole.UserRole) != "+":
            self.layer_list.setCurrentItem(item)
            layer = self._current_layer()
        if layer is not None:
            rename_action = menu.addAction(tr("Rename"))
            remove_action = menu.addAction(tr("Delete layer"))
            strength_action = menu.addAction(tr("Correction strength..."))
            menu.addSeparator()
            overlay_action = menu.addAction(tr("Show mask overlay"))
            overlay_action.setCheckable(True)
            overlay_action.setChecked(layer.get("overlay_visible", False))
            mask_action = menu.addAction(tr("View layer mask"))
            mask_action.setCheckable(True)
            mask_action.setChecked(self.view_mask_box.isChecked())
        elif global_selected:
            rename_action = menu.addAction(tr("Rename"))
            remove_action = overlay_action = mask_action = None
            strength_action = menu.addAction(tr("Correction strength..."))
            auto_action = menu.addAction(tr("Automatic strength (lowest ΔE76)"))
            scan = self._current_scan()
            auto_action.setCheckable(True)
            auto_action.setChecked(bool(scan) and scan.get("global_strength") is None)
        else:
            rename_action = remove_action = strength_action = overlay_action = mask_action = None
        selected = menu.exec(position)
        if global_selected and selected == auto_action:
            self._request_global_strength(None)
        elif global_selected and selected == strength_action:
            self._set_layer_strength_by_id(GLOBAL_LAYER_ID)
        elif selected == add_action:
            self.add_layer()
        elif selected == rename_action:
            self.rename_layer()
        elif selected == remove_action:
            self.remove_layer()
        elif selected == strength_action:
            self._set_layer_strength_by_id(layer["id"])
        elif selected == overlay_action:
            self._toggle_layer_overlay_by_id(layer["id"])
        elif selected == mask_action:
            self.view_mask_box.setChecked(mask_action.isChecked())

    def _refresh_layer(self):
        layer = self._current_layer()
        self._building = True
        self.protect_lines.setChecked(layer["protect_lines"] if layer else True)
        self.protect_lines.setEnabled(layer is not None)
        self._building = False
        self._refresh_mask()
        self._refresh_color_metrics()

    def toggle_eraser(self):
        """X: switch to the eraser, or back to the tool used before it."""
        if self.result_view.mode == "erase":
            self.result_view.set_mode(self._tool_before_erase)
        else:
            self.result_view.set_mode("erase")

    def _tool_mode_changed(self, mode: str):
        if mode == "erase" and self._last_tool_mode != "erase":
            self._tool_before_erase = self._last_tool_mode
        self._last_tool_mode = mode
        if mode != "pan" and self.edit_exclusions.isChecked():
            self.edit_exclusions.setChecked(False)
        values = self._tool_values.get(mode)
        polygon = mode == "polygon"
        for button_mode, button in self._tool_buttons.items():
            blocked = button.blockSignals(True)
            button.setChecked(button_mode == mode)
            button.blockSignals(blocked)
        self.tool_size_label.setText("Feather" if polygon else "Diameter")
        self.tool_diameter.setEnabled(mode in self._tool_values)
        self.tool_hardness.setEnabled(mode in {"paint", "erase"})
        blocked = self.tool_diameter.blockSignals(True)
        self.tool_diameter.setRange(0 if polygon else 2, POLYGON_MAX_FEATHER if polygon else 3000)
        self.tool_diameter.setSingleStep(1 if polygon else 2)
        self.tool_diameter.blockSignals(blocked)
        self.tool_diameter.setToolTip(
            "Polygon edge feather: the soft transition on each side of the outline, in image "
            "pixels. 0 is a hard edge" if polygon else "")
        if values is None:
            return
        size, hardness = values
        controls = [(self.tool_diameter, size)] + ([] if polygon else [(self.tool_hardness, hardness)])
        for control, value in controls:
            blocked = control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(blocked)
        self.result_view.set_brush_radius(size if polygon else max(1, round(size / 2)))
        self.result_view.set_brush_hardness(0 if polygon else hardness)

    def _tool_button_clicked(self, mode):
        self.result_view.set_mode("pan" if self.result_view.mode == mode else mode)
        # Keep the pressed state in step with the tool even if the mode did not change.
        self._tool_buttons[mode].setChecked(self.result_view.mode == mode)

    def _tool_parameter_changed(self, *_):
        mode = self.result_view.mode
        if mode not in self._tool_values:
            return
        if mode == "polygon":
            feather = self.tool_diameter.value()
            self._tool_values[mode] = (feather, None)
            self.result_view.set_brush_radius(feather)
            self.result_view.set_brush_hardness(0)
            return
        diameter = self.tool_diameter.value()
        hardness = self.tool_hardness.value()
        self._tool_values[mode] = (diameter, hardness)
        self.result_view.set_brush_radius(max(1, round(diameter / 2)))
        self.result_view.set_brush_hardness(hardness)

    def _brush_drag_changed(self, diameter: int, hardness: int):
        mode = self.result_view.mode
        if mode not in {"paint", "erase", "polygon"}:
            return
        if mode == "polygon":
            feather = diameter // 2  # the view keeps the feather as its brush radius
            self._tool_values[mode] = (feather, None)
            blocked = self.tool_diameter.blockSignals(True)
            self.tool_diameter.setValue(feather)
            self.tool_diameter.blockSignals(blocked)
            self.status.setText(f"Polygon edge feather {feather} px")
            return
        self._tool_values[mode] = (diameter, hardness)
        for box, value in ((self.tool_diameter, diameter), (self.tool_hardness, hardness)):
            blocked = box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(blocked)
        self.status.setText(f"Brush diameter {diameter} px · hardness {hardness}%")

    def _layer_options_changed(self, *_):
        if self._building:
            return
        layer, scan = self._current_layer(), self._current_scan()
        if layer is None:
            return
        options = {"protect_lines": self.protect_lines.isChecked()}
        self._commit_layer_options(layer, scan, options)

    def _set_layer_strength_by_id(self, layer_id):
        scan = self._current_scan()
        if layer_id == GLOBAL_LAYER_ID:
            item = next((self.layer_list.item(row) for row in range(self.layer_list.count())
                         if self.layer_list.item(row).data(Qt.ItemDataRole.UserRole) == layer_id), None)
            if item and item.data(LAYER_STRENGTH_ROLE) is not None:
                self.layer_list.edit_strength(item, item.data(LAYER_STRENGTH_ROLE))
            return
        layer = next((entry for entry in scan["layers"] if entry["id"] == layer_id), None) if scan else None
        if layer is None:
            return
        item = next((self.layer_list.item(row) for row in range(self.layer_list.count())
                     if self.layer_list.item(row).data(Qt.ItemDataRole.UserRole) == layer_id), None)
        if item:
            self.layer_list.edit_strength(item, layer.get("strength", 100))

    def _global_strength_limit(self, scan, state):
        if engine.can_extend_passes(scan) and not state.get("passes_stopped_at"):
            return engine.MAX_PASSES * 100
        return 100 * max(1, state.get("passes", 1))

    def _request_global_strength(self, value):
        """Change the global strength: instant small preview, full result after a pause."""
        scan = self._current_scan()
        state = self.report.get("scans", {}).get(scan["id"], {}) if scan else {}
        if scan is None or not self.project.is_current() or state.get("status") != "ready":
            self.status.setText("Run Color Match first")
            return
        limit = self._global_strength_limit(scan, state)
        if value is not None and value > limit and state.get("passes_stopped_at"):
            self.status.setText(f"Pass {state['passes_stopped_at']} failed held-out validation "
                                f"(used only to blend towards): at most {limit}%")
            if scan.get("global_strength") == limit or (
                    scan.get("global_strength") is None and engine.global_strength(scan, state) == limit):
                return
        if value is not None:
            value = max(0, min(limit, int(value)))
        if value == scan.get("global_strength") and value is not None:
            return
        scan["global_strength"] = value
        self.project.save()
        strength = engine.global_strength(scan, state)
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == GLOBAL_LAYER_ID:
                item.setData(LAYER_STRENGTH_ROLE, strength)
        self._global_strength_scan_id = scan["id"]
        self._global_strength_timer.start(450)
        needs_more = math.ceil(strength / 100) > state.get("passes", 1)
        shows_global = scan.get("global_visible", True) and not any(
            layer.get("visible", True) for layer in scan["layers"])
        if not needs_more and shows_global:
            snapshot = Project(self.project.directory, copy.deepcopy(self.project.data))
            target = snapshot.scan(scan["id"])
            scan_id = scan["id"]

            def show(array):
                current = self._current_scan()
                if current and current["id"] == scan_id and engine.global_strength(
                        current, self.report.get("scans", {}).get(scan_id, {})) == strength:
                    self.result_view.show_temporary_preview(_qimage_from_image(Image.fromarray(array)))
            self._in_background(("global-preview", scan_id, strength),
                                lambda: engine.global_preview(snapshot, target, strength), show)
        self.status.setText(f"Global strength {strength}%" + (" (automatic)" if value is None else ""))

    def _apply_global_strength(self):
        scan_id = self._global_strength_scan_id
        if not self.project or scan_id is None:
            return
        if self.task and self.task.isRunning():
            self._global_strength_timer.start(500)
            return
        try:
            self.project.scan(scan_id)
        except StopIteration:
            return
        state = dict(self.report.get("scans", {}).get(scan_id, {}))
        shadow = Project(self.project.directory, copy.deepcopy(self.project.data))
        target = shadow.scan(scan_id)

        def operation(cancel, progress):
            updates = {}
            wanted = engine.global_strength(target, state)
            if math.ceil(wanted / 100) > state.get("passes", 1) and engine.can_extend_passes(target):
                updates.update(engine.extend_passes(shadow, target, math.ceil(wanted / 100),
                                                    cancel, progress))
            current = {**state, **updates}
            strength = min(engine.global_strength(target, current), 100 * current.get("passes", 1))
            updates.update(engine.render_global(shadow, target, strength, cancel, progress))
            return updates

        self._start_task(operation, lambda updates: self._global_strength_ready(scan_id, updates),
                         lock_side=False)

    def _global_strength_ready(self, scan_id, updates):
        entry = self.report.setdefault("scans", {}).setdefault(scan_id, {})
        entry.update(updates)
        run = self.project.run_dir()
        if run:
            (run / "report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        scan = self.project.scan(scan_id)
        message = f"Global strength {entry.get('base_strength', 0)}%"
        explicit = scan.get("global_strength")
        if explicit is not None and explicit > 100 * entry.get("passes", 1):
            scan["global_strength"] = 100 * entry.get("passes", 1)
        if entry.get("passes_stopped_at"):
            message += (f" · pass {entry['passes_stopped_at']} failed held-out validation "
                        "and is used only to blend towards")
        scan["evaluation"] = None
        if any(layer.get("visible", True) for layer in scan["layers"]):
            scan["local_dirty"] = True
        else:
            scan["local_output"] = None
            scan["local_dirty"] = False
        self.project.save()
        self._refresh()
        if scan["local_dirty"]:
            self._queue_local_render(scan)
        self.status.setText(message)

    def _set_layer_strength_value(self, layer_id, value):
        if layer_id == GLOBAL_LAYER_ID:
            self._request_global_strength(value)
            return
        scan = self._current_scan()
        layer = next((entry for entry in scan["layers"] if entry["id"] == layer_id), None) if scan else None
        if layer is not None:
            self._commit_layer_options(layer, scan, {"strength": value})

    def _commit_layer_options(self, layer, scan, options):
        if all(layer.get(key) == value for key, value in options.items()):
            return
        render_changed = any(layer.get(key) != value and key != "hardness"
                             for key, value in options.items())
        keep_previous_preview = (bool(scan.get("local_output"))
                                 and (self.project.directory / scan["local_output"]).exists())
        if (render_changed and self.project.is_current() and not scan.get("local_dirty") and scan.get("local_output")
                and (self.project.directory / scan["local_output"]).exists()):
            self._remember_local_variant(scan, scan["local_output"])
        layer.update(options)
        if not render_changed:
            self.project.save()
            return
        scan["local_dirty"] = True
        if self.project.is_current():
            cached = self._cached_local_variant(scan)
            if cached:
                scan["local_output"] = cached
                scan["local_dirty"] = False
            elif not any(item.get("visible", True) for item in scan["layers"]):
                scan["evaluation"] = None
                base = self.project.run_dir() / "base" / f"{scan['id']}.png"
                if base.exists() or not scan.get("global_visible", True):
                    scan["local_output"] = (str(base.relative_to(self.project.directory))
                                            if scan.get("global_visible", True) else None)
                    scan["local_dirty"] = False
                    if scan["local_output"]:
                        self._remember_local_variant(scan, scan["local_output"])
            else:
                scan["evaluation"] = None
            if not scan["local_dirty"]:
                self._pending_local_scan_id = None
                self._local_render_timer.stop()
        self.project.save()
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == layer["id"]:
                item.setText(layer["name"] + (" (hidden)" if not layer["visible"] else ""))
                item.setData(LAYER_STRENGTH_ROLE, layer.get("strength", 100))
                break
        self._update_scan_item(scan)
        if not keep_previous_preview or not scan["local_dirty"]:
            if not self._show_resident_result(scan):
                self._refresh_image()
        if scan["local_dirty"] and self.project.is_current():
            self._queue_local_render(scan)
        else:
            self.status.setText("Local layer visibility updated")

    def _queue_local_render(self, scan):
        if any(item.get("visible", True) and item.get("model") and item.get("needs_refit", False)
               for item in scan["layers"]):
            self.status.setText("Painted layer needs Refine All to fit its color correction")
            return
        self._pending_local_scan_id = scan["id"]
        # Let consecutive brush strokes / visibility clicks settle before the
        # expensive full-resolution export cache competes with the preview.
        self._local_render_timer.start(700)
        self._queue_fast_local_preview(scan)
        self.status.setText("Updating local layer preview...")

    def _queue_fast_local_preview(self, scan, warm=False):
        run = self.project.run_dir()
        if (run is None or self.current_scan_id != scan["id"]
                or (not warm and not scan.get("local_dirty"))):
            return
        base = (run / "base" / f"{scan['id']}.png" if scan.get("global_visible", True)
                else Path(scan["path"]))
        if not base.exists():
            return
        signature = self._local_signature(scan)
        preview_key = (self.project.directory, scan["id"], signature)
        if not warm and preview_key in self._composed_preview_cache:
            if self.current_scan_id == scan["id"]:
                self.result_view.show_temporary_preview(self._composed_preview_cache[preview_key])
                self._active_preview_signature = signature
            return
        if warm:
            assets = [base] + [self.project.directory / layer["mask"] for layer in scan["layers"]]
            if any(not path.exists() for path in assets):
                return
            # Re-warm when layer models change (e.g. after Refine All), not only files.
            asset_key = (self._local_signature(scan),
                         tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                               for path in assets))
            if asset_key in self._fast_preview_warmed:
                return
            self._fast_preview_warmed.add(asset_key)
            if len(self._fast_preview_warmed) > 64:
                self._fast_preview_warmed.clear()
        self._fast_preview_pending = (scan["id"], signature,
                                       self.project.directory, base, copy.deepcopy(scan["layers"]), warm)
        if self._fast_preview_worker is None:
            self._start_fast_local_preview()

    def _start_fast_local_preview(self):
        pending = self._fast_preview_pending
        self._fast_preview_pending = None
        if pending is None:
            return
        scan_id, signature, directory, base, layers, warm = pending
        overrides = {}
        for layer in layers:
            path = directory / layer["mask"]
            if path in self._resident_masks:
                overrides[path] = (self._mask_revisions[path], self._resident_masks[path])
        worker = LocalPreviewThread(base, layers, directory, self._fast_preview_cache,
                                    preload_only=warm, mask_overrides=overrides,
                                    patch_cache=self._preview_patch_cache)
        self._fast_preview_worker = worker
        worker.ready.connect(lambda preview, sid=scan_id, sig=signature, root=directory:
                             self._fast_local_preview_ready(sid, sig, root, preview))
        worker.finished.connect(lambda target=worker: self._fast_local_preview_finished(target))
        worker.start()

    def _fast_local_preview_ready(self, scan_id, signature, directory, preview):
        if not self.project or self.project.directory != directory:
            return
        key = (directory, scan_id, signature)
        self._composed_preview_cache.pop(key, None)
        self._composed_preview_cache[key] = preview
        while len(self._composed_preview_cache) > 8:
            self._composed_preview_cache.pop(next(iter(self._composed_preview_cache)))
        if self.current_scan_id != scan_id:
            return
        scan = self._current_scan()
        if scan and scan.get("local_dirty") and self._local_signature(scan) == signature:
            self.result_view.show_temporary_preview(preview)
            self._active_preview_signature = signature
            self.status.setText("Preview updated; rendering full-resolution result...")

    def _fast_local_preview_finished(self, worker):
        if self._fast_preview_worker is worker:
            self._fast_preview_worker = None
        worker.deleteLater()
        if self._fast_preview_pending is not None and not self._close_when_idle:
            self._start_fast_local_preview()

    def _local_signature(self, scan):
        run = self.project.data.get("active_run") or {}
        layers = []
        for layer in scan["layers"]:
            mask = self.project.directory / layer["mask"]
            metadata = mask.stat() if mask.exists() and mask not in self._resident_masks else None
            mask_version = (("resident", self._mask_revisions[mask]) if mask in self._resident_masks
                            else (metadata.st_mtime_ns, metadata.st_size) if metadata else None)
            layers.append((layer["id"], layer.get("visible", True), layer.get("strength"),
                           None if layer.get("soft_mask", False) else layer.get("feather"),
                           layer.get("protect_lines"), layer.get("needs_refit"),
                           layer.get("model_run"), layer.get("model"),
                           mask_version))
        base_strength = self.report.get("scans", {}).get(scan["id"], {}).get("base_strength")
        payload = json.dumps((run.get("id"), scan.get("global_visible", True), base_strength, layers),
                             sort_keys=True)
        exclusions = [self.project.exclusion_signature(ref) for ref in scan["refs"] if ref.get("exclude_mask")]
        if exclusions:
            payload += json.dumps(exclusions, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cached_local_variant(self, scan):
        signature = self._local_signature(scan)
        relative = scan.get("local_variants", {}).get(signature)
        if relative and (self.project.directory / relative).exists():
            scan["evaluation"] = scan.get("evaluation_variants", {}).get(signature)
            return relative
        return None

    def _show_resident_result(self, scan):
        if (self.current_scan_id != scan["id"] or scan.get("local_dirty")
                or not scan.get("local_output")):
            return False
        signature = self._local_signature(scan)
        preview = self._composed_preview_cache.get((self.project.directory, scan["id"], signature))
        if preview is None:
            return False
        output = self.project.directory / scan["local_output"]
        if not output.exists():
            return False
        self.result_view.show_temporary_preview(preview)
        self.result_view.adopt_current_preview(output)
        self._active_preview_signature = signature
        return True

    def _remember_local_variant(self, scan, output):
        variants = scan.setdefault("local_variants", {})
        key = self._local_signature(scan)
        variants.pop(key, None)
        variants[key] = output
        while len(variants) > 8:
            variants.pop(next(iter(variants)))
        if scan.get("evaluation"):
            scores = scan.setdefault("evaluation_variants", {})
            scores.pop(key, None)
            scores[key] = scan["evaluation"]
            while len(scores) > 8:
                scores.pop(next(iter(scores)))

    def _run_pending_local_render(self):
        scan_id = self._pending_local_scan_id
        if not scan_id or not self.project:
            return
        if self.task and self.task.isRunning():
            self._local_render_timer.start(300)
            return
        if self._fast_preview_worker and self._fast_preview_worker.isRunning():
            self._local_render_timer.start(100)
            return
        self._persist_masks()
        if self._mask_save_error is not None:
            self.status.setText("Layer mask could not be saved; full-resolution result was not updated")
            return
        if self._dirty_masks or self._mask_save_future is not None:
            self._local_render_timer.start(100)
            return
        self._pending_local_scan_id = None
        scan = next((item for item in self.project.data["scans"] if item["id"] == scan_id), None)
        if scan is None or not self.project.is_current() or not scan.get("local_dirty"):
            return
        signature = self._local_signature(scan)
        revision = self.project.data["calibration_revision"]
        shadow = Project(self.project.directory, copy.deepcopy(self.project.data))
        patch_cache = self._render_patch_cache

        def operation(cancel, progress):
            target = shadow.scan(scan_id)
            # Only the image here: ΔE76 metrics are measured afterwards by the background
            # metric thread, so the interface is free again as soon as the pixels exist.
            return engine.render_layers(shadow, target, cancel, progress, patch_cache), None

        self._start_task(operation, lambda result: self._local_render_complete(scan_id, revision, signature, *result),
                         lock_side=False)

    def _local_render_complete(self, scan_id, revision, signature, output, evaluation=None):
        if not self.project or self.project.data["calibration_revision"] != revision:
            return
        scan = next((item for item in self.project.data["scans"] if item["id"] == scan_id), None)
        if scan is None or self._local_signature(scan) != signature:
            return
        scan["local_output"] = output
        scan["local_dirty"] = False
        scan["evaluation"] = evaluation
        if self._fast_preview_pending and self._fast_preview_pending[0] == scan_id:
            self._fast_preview_pending = None
        if output:
            self._remember_local_variant(scan, output)
        self.project.save()
        self._update_scan_item(scan)
        if self.current_scan_id == scan_id:
            if self._active_preview_signature == signature and output:
                self.result_view.adopt_current_preview(self.project.directory / output)
            else:
                self._refresh_image()
        self.status.setText("Local layer preview updated")

    def remove_layer(self):
        scan, layer = self._current_scan(), self._current_layer()
        if scan is None or layer is None:
            return
        scan["layers"].remove(layer)
        scan["local_dirty"] = bool(scan["layers"])
        scan["evaluation"] = None
        if not scan["layers"]:
            scan["local_output"] = None
            scan["local_dirty"] = False
        self.project.save()
        self._refresh()
        if scan["local_dirty"] and self.project.is_current():
            self._queue_local_render(scan)

    def rename_layer(self):
        scan = self._current_scan()
        item = self.layer_list.currentItem()
        if scan and item and item.data(Qt.ItemDataRole.UserRole) == GLOBAL_LAYER_ID:
            name, accepted = QInputDialog.getText(
                self, "Rename Layer", "Layer name", text=scan.get("global_name", "global"))
            if accepted and name.strip():
                scan["global_name"] = name.strip()
                self.project.save()
                item.setText(scan["global_name"])
            return
        layer = self._current_layer()
        if layer is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename Layer", "Layer name", text=layer["name"])
        if accepted and name.strip():
            layer["name"] = name.strip()
            self.project.save()
            self._refresh_details()

    def _resident_mask(self, path: Path) -> np.ndarray:
        if path not in self._resident_masks:
            with Image.open(path) as source:
                self._resident_masks[path] = np.asarray(source.convert("L"), dtype=np.uint8).copy()
            self._mask_revisions[path] = 0
        return self._resident_masks[path]

    def _record_mask_edit(self, path: Path, mask: np.ndarray):
        self._last_mask_target = "layer"
        self._resident_masks[path] = mask
        self._mask_revisions[path] = self._mask_revisions.get(path, 0) + 1
        self._dirty_masks.add(path)
        self._mask_save_error = None
        self._mask_thumb_cache.clear()
        self._mask_save_timer.start(350)
        self.project.save()

    def _persist_masks(self):
        if self._mask_save_error is not None:
            return
        if self._mask_save_future is not None:
            if not self._mask_save_future.done():
                self._mask_save_timer.start(100)
                return
            try:
                self._mask_save_future.result()
            except Exception as exc:
                self._dirty_masks.update(self._mask_save_batch)
                self._mask_save_future = None
                self._mask_save_batch = []
                self._mask_save_error = str(exc)
                self._error(f"Could not save layer mask: {exc}")
                return
            self._mask_save_future = None
            self._mask_save_batch = []
        if self._dirty_masks:
            paths = list(self._dirty_masks)
            snapshots = [(path, self._resident_masks[path]) for path in paths]
            self._dirty_masks.clear()
            self._mask_save_batch = paths
            self._mask_save_future = self._mask_save_executor.submit(_save_mask_snapshots, snapshots)
            self._mask_save_timer.start(100)

    def _flush_masks_blocking(self):
        self._mask_save_timer.stop()
        if self._mask_save_future is not None:
            try:
                self._mask_save_future.result()
            except Exception:
                self._dirty_masks.update(self._mask_save_batch)
                raise
            finally:
                self._mask_save_future = None
                self._mask_save_batch = []
        if self._dirty_masks:
            paths = list(self._dirty_masks)
            _save_mask_snapshots([(path, self._resident_masks[path]) for path in paths])
            self._dirty_masks.clear()
        self._mask_save_error = None

    def _paint_stroke(self, points, erase):
        scan, layer = self._current_scan(), self._layer_for_mask_edit()
        if scan is None or layer is None:
            return
        mask_path = self.project.directory / layer["mask"]
        try:
            previous = self._resident_mask(mask_path)
            was_soft = layer.get("soft_mask", False)
            mask = engine.paint_mask(self.project, scan, layer, points,
                                     self.result_view.brush_radius, erase,
                                     mask=previous, persist=False,
                                     hardness=self.result_view.brush_hardness)
            self.undo_history.append((scan["id"], layer["id"], previous,
                                      was_soft))
            self.undo_history = self.undo_history[-12:]
            self.redo_history.clear()
            layer["mask_changed_since_refine"] = True
            self._record_mask_edit(mask_path, mask)
            self._refresh_mask()
            self._update_scan_item(scan)
            self._refresh_color_metrics()
            if layer.get("model") and not layer.get("needs_refit", False) and self.project.is_current():
                self._queue_local_render(scan)
            else:
                self.status.setText("Mask changed. Click Refine All to fit its color correction")
        except Exception as exc:
            if "was_soft" in locals():
                layer["soft_mask"] = was_soft
            if self.result_view._mask_preview_base is not None:
                self.result_view._mask_preview = self.result_view._mask_preview_base
                self.result_view._render_mask_preview()
            self._error(str(exc))

    def _paint_polygon(self, vertices, radius, hardness):
        scan, layer = self._current_scan(), self._layer_for_mask_edit()
        if scan is None or layer is None:
            return
        mask_path = self.project.directory / layer["mask"]
        try:
            previous = self._resident_mask(mask_path)
            was_soft = layer.get("soft_mask", False)
            mask = engine.paint_polygon_mask(self.project, scan, layer, vertices, radius, hardness,
                                             mask=previous, persist=False)
            self.undo_history.append((scan["id"], layer["id"], previous, was_soft))
            self.undo_history = self.undo_history[-12:]
            self.redo_history.clear()
            layer["mask_changed_since_refine"] = True
            self._record_mask_edit(mask_path, mask)
            self._refresh_mask()
            self._update_scan_item(scan)
            self._refresh_color_metrics()
            if layer.get("model") and not layer.get("needs_refit", False) and self.project.is_current():
                self._queue_local_render(scan)
            else:
                self.status.setText("Mask changed. Click Refine All to fit its color correction")
        except Exception as exc:
            if "was_soft" in locals():
                layer["soft_mask"] = was_soft
            self._error(str(exc))

    def _restore_stroke(self, source, destination):
        if not source or not self.project:
            return
        scan_id, layer_id, contents, soft_mask = source.pop()
        scan = next((item for item in self.project.data["scans"] if item["id"] == scan_id), None)
        if scan is None:
            return
        layer = next((item for item in scan["layers"] if item["id"] == layer_id), None)
        if layer is None:
            return
        path = self.project.directory / layer["mask"]
        destination.append((scan_id, layer_id, self._resident_mask(path),
                            layer.get("soft_mask", False)))
        if isinstance(contents, bytes):
            with Image.open(io.BytesIO(contents)) as source_image:
                contents = np.asarray(source_image.convert("L"), dtype=np.uint8).copy()
        self._record_mask_edit(path, contents)
        layer["soft_mask"] = soft_mask
        layer["mask_changed_since_refine"] = True
        if layer.get("model"):
            scan["local_dirty"] = True
        self.project.save()
        if self.current_scan_id == scan_id:
            self._refresh_mask()
            self._update_scan_item(scan)
            self._refresh_color_metrics()
            if layer.get("model") and not layer.get("needs_refit", False) and self.project.is_current():
                self._queue_local_render(scan)

    def undo_stroke(self):
        self._restore_stroke(self.undo_history, self.redo_history)

    def redo_stroke(self):
        if self._exclusion_undo_context():
            self._restore_exclusion(True)
        else:
            self._restore_stroke(self.redo_history, self.undo_history)

    def _start_task(self, operation, on_success, lock_side=True):
        if self.task and self.task.isRunning():
            self._error("Another operation is already running")
            return
        self.edit_exclusions.setChecked(False)
        self.reference_view.cancel_current_action()
        self.edit_exclusions.setEnabled(False)
        self.clear_exclusions.setEnabled(False)
        self.mask_mode_exclude.setEnabled(False)
        self.mask_mode_include.setEnabled(False)
        self.task = TaskThread(operation)
        self.task.progress.connect(self._progress)
        self.task.completed.connect(lambda _result: self._mark_progress_done())
        self.task.completed.connect(on_success)
        self.task.failed.connect(self._task_failed)
        self.task.cancelled.connect(self._task_cancelled)
        self.task.finished.connect(self._task_finished)
        self.run_button.setText("Cancel")
        self.run_button.setEnabled(True)
        self.refine_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_all_button.setEnabled(False)
        self._lock_left(True)
        self._set_progress_done(False)
        self.progress_bar.setRange(0, 0)  # busy until the first progress report
        self.progress_bar.setFormat("")
        if lock_side:
            self.side_panel.setEnabled(False)
        self.menuBar().setEnabled(False)
        self.task.start()

    PROGRESS_STYLE = ("QProgressBar {{ border: 1px solid #b8c0c9; border-radius: 4px; background: #eef1f4;"
                      " color: #1f2933; text-align: center; }}"
                      "QProgressBar::chunk {{ background: {chunk}; border-radius: 3px; }}")

    def _set_progress_done(self, done):
        """Blue while working, green once finished."""
        self.progress_bar.setStyleSheet(self.PROGRESS_STYLE.format(chunk="#7fd19a" if done else "#8cc4f2"))

    def _mark_progress_done(self):
        maximum = max(1, self.progress_bar.maximum())
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(maximum)
        self.progress_bar.setFormat("%p%")
        self._set_progress_done(True)

    def _lock_left(self, locked):
        for widget in (self.scan_list, self.fit_de76, self.fit_rgb, self.passes_box):
            widget.setEnabled(not locked)

    def _fit_settings_changed(self, *_):
        if self._building or self.project is None:
            return
        metric = "de76" if self.fit_de76.isChecked() else "rgb"
        passes = self.passes_box.value()
        if engine.fit_settings(self.project.data) == (metric, passes):
            return
        self.project.data["fit_metric"], self.project.data["passes"] = metric, passes
        self.project.invalidate()
        self._refresh()
        self.status.setText("Fit settings changed. Run Color Match to apply them")

    def _progress(self, value, total, message):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat("%p%")
        self._set_progress_done(value >= total > 0)
        self.status.setText(message)

    def _task_failed(self, message):
        self._error(message)

    def _task_cancelled(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self._set_progress_done(False)
        self.status.setText("Cancelled; the previous result is unchanged")

    def _task_finished(self):
        self.run_button.setText("Run Color Match")
        self.run_button.setEnabled(True)
        self.refine_button.setEnabled(True)
        if self.progress_bar.maximum() == 0:  # stopped (failed) before reporting anything
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("")
        self.export_button.setEnabled(True)
        self.export_all_button.setEnabled(True)
        self._lock_left(False)
        self.side_panel.setEnabled(True)
        self.menuBar().setEnabled(True)
        if self._close_when_idle:
            self.close()
        else:
            self._refresh_color_metrics()
            self._refresh_exclusion_overlay()

    def _run_or_cancel(self):
        if self.task and self.task.isRunning():
            self.cancel_task()
        else:
            self.run_match()

    def cancel_task(self):
        if self.task:
            self.task.cancel_event.set()
            self.status.setText("Cancelling after the current step...")

    def run_match(self):
        if self.project is None:
            self._error("Create or open a project first")
            return
        try:
            self._flush_masks_blocking()
        except Exception as exc:
            self._error(f"Could not save layer mask: {exc}")
            return
        snapshot = copy.deepcopy(self.project.data)
        directory = self.project.directory

        def operation(cancel, progress):
            return run_calibration(snapshot, directory, cancel, progress)

        self._start_task(operation, self._run_complete)

    def _run_complete(self, result):
        record, report, updates = result
        if self.project.data["calibration_revision"] != record["revision"]:
            self._error("The project changed during processing. Run again")
            return
        record["input_signature"] = self.project.calibration_signature()
        previous_run = self.project.data.get("active_run")
        self.project.data["active_run"] = record
        self.report = report
        for scan in self.project.data["scans"]:
            if report["scans"].get(scan["id"], {}).get("status") == "ready":
                scan.pop("previous_result_run", None)
            elif previous_run:
                previous_base = (self.project.directory / "cache" / "runs" /
                                 previous_run["id"] / "base" / f"{scan['id']}.png")
                if previous_base.exists():
                    scan["previous_result_run"] = previous_run["id"]
            scan["evaluation"] = None
            scan["evaluation_variants"] = {}
            if scan["id"] in updates:
                scan["layers"], scan["local_output"] = updates[scan["id"]]
                scan["local_dirty"] = False
                self._remember_local_variant(scan, scan["local_output"])
            else:
                scan["local_output"] = None
                scan["local_dirty"] = bool(scan["layers"])
        self.project.save()
        report_path = self.project.run_dir() / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self._refresh()
        ready = sum(row.get("status") == "ready" for row in report["scans"].values())
        self.status.setText(f"Color matching complete: {ready} scan(s) ready")

    def refine_selection(self):
        scan = self._current_scan()
        if scan is None or not scan["layers"]:
            self._error("Paint at least one correction layer first")
            return
        try:
            self._flush_masks_blocking()
        except Exception as exc:
            self._error(f"Could not save layer mask: {exc}")
            return
        shadow = Project(self.project.directory, copy.deepcopy(self.project.data))
        scan_id = scan["id"]
        patch_cache = self._render_patch_cache

        def operation(cancel, progress):
            target = shadow.scan(scan_id)
            layers, output = engine.refine_layers(shadow, target, cancel, progress, patch_cache)
            target["layers"] = layers
            signature = engine.refine_quick_signature(shadow, target)
            # A new result is measured afterwards by the background metric thread.
            evaluation = (target.get("evaluation") if output == target.get("local_output")
                          and (target.get("evaluation") or {}).get("layer_effects") is not None
                          else None)
            return layers, output, evaluation, signature

        self._start_task(operation, lambda result: self._refine_complete(scan_id, result))

    def _refine_complete(self, scan_id, result):
        scan = self.project.scan(scan_id)
        reused_output = (result[1] == scan.get("local_output") and not scan.get("local_dirty"))
        scan["layers"], scan["local_output"] = result[:2]
        scan["evaluation"] = result[2] if len(result) > 2 else None
        scan["refine_signature"] = result[3] if len(result) > 3 else None
        scan["local_dirty"] = False
        self._remember_local_variant(scan, scan["local_output"])
        self.project.save()
        self.result_view.set_mode("pan")
        self._refresh()
        self.status.setText("Refine All: cached result reused" if reused_output
                            else "Refine All: corrections and strengths updated")

    def export_current(self):
        scan = self._current_scan()
        if scan is not None:
            self._export([scan], clear_after=False)

    def export_all(self):
        if self.project:
            self._export(self.project.data["scans"], clear_after=True)

    def _export(self, scans, *, clear_after=False):
        try:
            self._flush_masks_blocking()
        except Exception as exc:
            self._error(f"Could not save layer mask: {exc}")
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose export folder",
                                                     self._export_start_directory(scans))
        if not directory:
            return
        include_preview = self.menu_export_preview.isChecked()
        include_report = self.menu_export_report.isChecked()
        project = self.project
        scan_ids = [scan["id"] for scan in scans]
        signatures = {scan["id"]: self._local_signature(scan) for scan in scans}
        shadow = Project(project.directory, copy.deepcopy(project.data))
        patch_cache = self._render_patch_cache

        def operation(cancel, progress):
            saved, failed, updates = [], [], {}
            for index, scan_id in enumerate(scan_ids):
                engine._check_cancel(cancel)
                scan = shadow.scan(scan_id)
                try:
                    if scan.get("local_dirty"):
                        progress(index, len(scan_ids), f"Updating {Path(scan['path']).name} for export")
                        output = engine.render_layers(
                            shadow, scan, cancel,
                            lambda _value, _total, message: progress(index, len(scan_ids), message),
                            patch_cache)
                        scan["local_output"] = output
                        scan["local_dirty"] = False
                        updates[scan_id] = (signatures[scan_id], output, None)
                    progress(index + 1, len(scan_ids), f"Exporting {Path(scan['path']).name}")
                    saved.append(str(engine.export_scan(
                        shadow, scan, Path(directory), "png", include_report, include_preview
                    )))
                except Exception as exc:
                    failed.append(f"{Path(scan['path']).name}: {exc}")
            return saved, failed, updates, project.directory

        self._start_task(operation, lambda result: self._export_complete(result, clear_after=clear_after))

    def _export_complete(self, result, *, clear_after=False):
        saved, failed, updates, directory = result
        if self.project and self.project.directory == directory:
            changed = False
            for scan_id, (signature, output, evaluation) in updates.items():
                scan = self.project.scan(scan_id)
                if self._local_signature(scan) != signature:
                    continue
                scan["local_output"] = output
                scan["local_dirty"] = False
                scan["evaluation"] = evaluation
                if output:
                    self._remember_local_variant(scan, output)
                self._update_scan_item(scan)
                if self.current_scan_id == scan_id:
                    if self._active_preview_signature == signature and output:
                        self.result_view.adopt_current_preview(self.project.directory / output)
                    else:
                        self._refresh_image()
                changed = True
            if changed:
                self.project.save()
        self.status.setText(f"Exported {len(saved)} image(s); {len(failed)} skipped")
        if failed:
            self._error("\n".join(failed[:8]))
        elif clear_after and saved and QMessageBox.question(
                self, "Clear Project", f"Exported all {len(saved)} image(s). Clear this project now?\n"
                "Original images and exported PNGs will remain untouched.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.clear_project(confirm=False)

    def closeEvent(self, event):
        if self.task and self.task.isRunning():
            self._close_when_idle = True
            self.task.cancel_event.set()
            self.status.setText("Closing after the current operation stops...")
            event.ignore()
            return
        try:
            self._flush_masks_blocking()
        except Exception as exc:
            self._error(f"Could not save layer mask: {exc}")
            event.ignore()
            return
        self.result_view.finish_previews()
        self.reference_view.finish_previews()
        self._prefetch_timer.stop()
        if self._prefetch_worker is not None:
            self._prefetch_worker.stop = True
            self._prefetch_worker.wait()
        self._loader.shutdown(wait=False, cancel_futures=True)
        self._fast_preview_pending = None
        if self._fast_preview_worker and self._fast_preview_worker.isRunning():
            self._fast_preview_worker.wait()
        if self._metric_worker and self._metric_worker.isRunning():
            self._metric_worker.wait()
        self._mask_save_executor.shutdown(wait=True)
        QApplication.instance().removeEventFilter(self)
        event.accept()


def stored_language() -> str:
    """The saved interface language, defaulting to the system language."""
    value = QSettings("ScanColorMatch", "ScanColorMatch").value("language")
    if value in i18n.LANGUAGES:
        return value
    return "zh" if QLocale.system().language() == QLocale.Language.Chinese else "en"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Scan Color Match")
    font = app.font()
    font.setFamilies(["Microsoft YaHei UI", "Microsoft YaHei", font.family()])
    app.setFont(font)
    i18n.set_language(stored_language())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
