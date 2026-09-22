from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    # 아직 구현하지 않은 Stage의 임시 화면을 구성
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("placeholderPage")

        layout = QVBoxLayout(self)

        title_label = QLabel(title)
        title_label.setObjectName("placeholderTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(title_label)
        layout.addStretch()