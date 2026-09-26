# 여러 페이지에서 공통으로 사용하는 PySide6 위젯을 제공
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox, QComboBox, QDoubleSpinBox, QFrame, QListView, QSpinBox,
)

from common.theme import COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_SECONDARY, SPACE_LG


class ChevronComboBox(QComboBox):
    # ComboBox의 Dropdown View를 구성
    def __init__(self):
        super().__init__()

        view = QListView()
        view.setFrameShape(QFrame.Shape.NoFrame)
        self.setView(view)

        palette = view.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_OFF_WHITE))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_OFF_WHITE))
        palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_OFF_BLACK))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_SECONDARY))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(COLOR_OFF_BLACK))
        view.setPalette(palette)

    # 기본 ComboBox를 그린 뒤 오른쪽에 아래 방향 꺾쇠를 표시
    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.isEnabled():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(COLOR_OFF_BLACK))
        pen.setWidthF(1.5)
        painter.setPen(pen)

        center_x = self.width() - 16
        center_y = self.height() / 2

        path = QPainterPath()
        path.moveTo(QPointF(center_x - 4, center_y - 2))
        path.lineTo(QPointF(center_x, center_y + 2))
        path.lineTo(QPointF(center_x + 4, center_y - 2))

        painter.drawPath(path)


class _ChevronSpinBoxMixin:
    # 기본 삼각형 버튼을 숨기고 클릭 가능한 위·아래 꺾쇠 영역을 구성
    def __init__(self):
        super().__init__()
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    # 숫자 입력창 오른쪽에 ComboBox와 같은 선형 꺾쇠를 표시
    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.isEnabled():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(COLOR_OFF_BLACK))
        pen.setWidthF(1.5)
        painter.setPen(pen)

        center_x = self.width() - SPACE_LG / 2
        upper_y = self.height() * 0.35
        lower_y = self.height() * 0.65

        upper_path = QPainterPath()
        upper_path.moveTo(QPointF(center_x - 4, upper_y + 2))
        upper_path.lineTo(QPointF(center_x, upper_y - 2))
        upper_path.lineTo(QPointF(center_x + 4, upper_y + 2))
        painter.drawPath(upper_path)

        lower_path = QPainterPath()
        lower_path.moveTo(QPointF(center_x - 4, lower_y - 2))
        lower_path.lineTo(QPointF(center_x, lower_y + 2))
        lower_path.lineTo(QPointF(center_x + 4, lower_y - 2))
        painter.drawPath(lower_path)

    # 오른쪽 위는 값을 높이고 오른쪽 아래는 값을 낮춤
    def mousePressEvent(self, event):
        is_chevron_area = event.position().x() >= self.width() - SPACE_LG

        if self.isEnabled() and is_chevron_area:
            if event.position().y() < self.height() / 2:
                self.stepUp()
            else:
                self.stepDown()

            event.accept()
            return

        super().mousePressEvent(event)


class ChevronSpinBox(_ChevronSpinBoxMixin, QSpinBox):
    pass


class ChevronDoubleSpinBox(_ChevronSpinBoxMixin, QDoubleSpinBox):
    pass
