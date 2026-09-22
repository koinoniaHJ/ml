from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.theme import SPACE_24


# QWidget: 화면을 구성하는 가장 기본적인 UI 객체
# QWidget을 상속해서 프로젝트 전용 Widget을 생성
class HomePage(QWidget):
    # 어떤 Dataset Source를 선택했는지 MainWindow에 전달
    data_source_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    # Home 화면의 Widget과 Layout을 구성
    def _setup_ui(self):
        # QVBoxLayout: 자식 Widget을 위에서 아래 방향으로 배치
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(SPACE_24)

        # addStretch(): Layout에서 남는 공간을 빈 여백으로 채워 Widget 위치를 조정하는 메서드
        main_layout.addStretch()

        # QLabel: 화면에 글자를 표시하는 Widget
        guide_label = QLabel("실습할 데이터를 선택하면 학습이 시작됩니다.")
        # Qt.AlignmentFlag: 내용 정렬 방향을 지정할 때 사용하는 Qt의 정렬 값
        guide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(guide_label)

        # QHBoxLayout: 자식 Widget을 왼쪽에서 오른쪽 방향으로 배치
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
                description="scikit-learn이 제공하는\n실습용 Dataset",
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

    # Dataset 선택 카드 하나를 생성
    def _create_dataset_card(self, title: str, description: str, source: str) -> QWidget:
        card = QWidget()
        layout = QVBoxLayout(card)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        description_label = QLabel(description)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # QPushButton: 클릭할 수 있는 Button Widget
        select_button = QPushButton("선택")
        select_button.setObjectName("datasetButton")

        # clicked: QPushButton이 기본 제공하는 Signal
        # emit(): 직접 정의한 Signal을 실제로 발생시키는 메서드
        select_button.clicked.connect(lambda: self.data_source_selected.emit(source))

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(select_button)

        return card