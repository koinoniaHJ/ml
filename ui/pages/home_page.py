from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from common.theme import HOME_CARD_HEIGHT, HOME_CARD_WIDTH, SPACE_MD, SPACE_XL


class HomePage(QWidget):
    data_source_selected = Signal(str)

    # Home 화면을 구성
    def __init__(self):
        super().__init__()
        self.setObjectName("homePage")
        self._setup_ui()

    # Dataset 선택 안내와 Card를 배치
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addStretch()

        guide_label = QLabel("실습할 데이터를 선택하면 학습이 시작됩니다.")
        guide_label.setObjectName("homeGuideLabel")
        guide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(guide_label)

        layout.addSpacing(SPACE_XL)

        card_layout = QHBoxLayout()
        card_layout.setSpacing(SPACE_MD)
        card_layout.addStretch()

        builtin_button = self._create_dataset_button(
            "Built-in Sample\n\n프로그램에 포함된\n학습용 예제 데이터\n\n[선택]", "builtin"
        )
        sklearn_button = self._create_dataset_button(
            "Scikit-learn Sample\n\nscikit-learn이 제공하는\n실습용 Dataset\n\n[선택]", "sklearn"
        )
        csv_button = self._create_dataset_button(
            "Local CSV\n\n직접 준비한 CSV 파일\n불러오기\n\n[선택]", "csv"
        )

        card_layout.addWidget(builtin_button)
        card_layout.addWidget(sklearn_button)
        card_layout.addWidget(csv_button)
        card_layout.addStretch()

        layout.addLayout(card_layout)
        layout.addStretch()

    # Dataset Source를 전달하는 Card Button을 생성
    def _create_dataset_button(self, text: str, source: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("datasetButton")
        button.setFixedSize(HOME_CARD_WIDTH, HOME_CARD_HEIGHT)
        button.clicked.connect(lambda checked=False: self.data_source_selected.emit(source))
        return button