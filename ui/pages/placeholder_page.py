from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

# QWidget을 상속해서 프로젝트 전용 Widget을 생성한다.
class PlaceholderPage(QWidget):
    def __init__(self, title: str, korean_title: str):
        super().__init__()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        korean_label = QLabel(korean_title)
        korean_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self) # Widget을 위에서 아래 방향으로 배치한다.

        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(korean_label)
        layout.addStretch()