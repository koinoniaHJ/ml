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
# QWidget을 상속해서 프로젝트 전용 Widget을 만든다.
class HomePage(QWidget):

    # Signal: 특정 이벤트가 발생했음을 다른 객체에 알리는 신호
    # 선택한 Dataset Source를 MainWindow에 전달하는 Custom Signal
    data_source_selected = Signal(str)

    def __init__(self):
        super().__init__()

        self._setup_ui()

    def _setup_ui(self):
        # QVBoxLayout: 자식 Widget을 위에서 아래 방향으로 배치
        main_layout = QVBoxLayout(self)

        # setSpacing(): Layout 안의 Widget 사이 간격 지정
        main_layout.setSpacing(SPACE_24)

        # addStretch(): 남는 공간을 여백으로 만들어 Widget 배치를 조정
        main_layout.addStretch()

        # QLabel: 화면에 글자를 표시하는 Widget
        guide_label = QLabel(
            "실습할 데이터를 선택하면 학습이 시작됩니다."
        )

        # Qt.AlignmentFlag: Widget 내용의 정렬 방향을 지정하는 값
        guide_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

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
        title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        description_label = QLabel(description)
        description_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        # QPushButton: 사용자가 클릭할 수 있는 Button Widget
        select_button = QPushButton("선택")

        # setObjectName(): QSS에서 특정 Widget을 선택할 수 있도록 이름 지정
        select_button.setObjectName("datasetButton")

        # clicked: QPushButton이 기본 제공하는 Signal
        # connect(): Signal과 Slot을 연결하는 메서드
        # lambda 함수가 Slot 역할을 하고 Custom Signal을 발생시킨다.
        select_button.clicked.connect(
            # emit(): 직접 정의한 Signal을 실제로 발생시키는 메서드
            lambda: self.data_source_selected.emit(source)
        )

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(select_button)

        return card