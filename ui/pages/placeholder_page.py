from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

class PlaceholderPage(QWidget):
    def __init__(self, title: str, korean_title: str):
        super().__init__()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        korean_label = QLabel(korean_title)
        korean_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)

        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(korean_label)
        layout.addStretch()