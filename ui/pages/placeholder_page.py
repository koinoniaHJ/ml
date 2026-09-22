from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


# 아직 구현 전인 Stage의 임시 화면
class PlaceholderPage(QWidget):
    def __init__(self, title: str, subtitle: str):
        super().__init__()

        layout = QVBoxLayout(self)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addStretch()