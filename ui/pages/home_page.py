from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.theme import SPACE_24


class HomePage(QWidget):
    # 어떤 Dataset Source를 선택했는지 MainWindow에 전달
    data_source_selected = Signal(str)

    def __init__(self):
        super().__init__()

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        main_layout.setSpacing(SPACE_24)

        main_layout.addStretch()

        guide_label = QLabel(
            "실습할 데이터를 선택하면 학습이 시작됩니다."
        )

        guide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(guide_label)

        cards_layout = QHBoxLayout()

        cards_layout.setSpacing(SPACE_24)

        cards_layout.addWidget(
            self._create_dataset_card(
                title="Built-in Sample",
                description="프로그램에 포함된\n학습용 예제 데이터",
                source="builtin",
            )
        )

        cards_layout.addWidget(
            self._create_dataset_card(
                title="scikit-learn Sample",
                description="scikit-learn이 제공하는\n회귀·분류 실습용 Dataset",
                source="sklearn",
            )
        )

        cards_layout.addWidget(
            self._create_dataset_card(
                title="Local CSV",
                description="직접 준비한 CSV 파일\n불러오기",
                source="csv",
            )
        )

        main_layout.addLayout(cards_layout)

        main_layout.addStretch()

    def _create_dataset_card(
        self,
        title: str,
        description: str,
        source: str,
    ) -> QWidget:
        card = QWidget()

        layout = QVBoxLayout(card)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        description_label = QLabel(description)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        select_button = QPushButton("선택")
        select_button.setObjectName("datasetButton")

        select_button.clicked.connect(
            lambda: self.data_source_selected.emit(source)
        )

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(select_button)

        return card