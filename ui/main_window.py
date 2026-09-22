from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from common.theme import (
    GRID_PC_MAX_WIDTH,
    SPACE_16,
    SPACE_24,
    SPACE_32,
)
from ui.pages.home_page import HomePage
from ui.pages.placeholder_page import PlaceholderPage

# QMainWindow: 프로그램의 최상위 Main Window를 만드는 클래스
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Machine Learning Lab")

        self.resize(
            GRID_PC_MAX_WIDTH,
            760,
        )

        self.navigation_buttons = {}

        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)

        # setContentsMargins(): Layout 내부 콘텐츠와 바깥 경계 사이의 여백을 지정하는 메서드
        root_layout.setContentsMargins(
            SPACE_32,
            SPACE_24,
            SPACE_32,
            SPACE_24,
        )

        # setSpacing(): Layout 안에서 서로 인접한 Widget 사이의 간격을 지정하는 메서드
        root_layout.setSpacing(SPACE_24)

        # Header
        header_label = QLabel("Machine Learning Lab")
        # setObjectName(): Widget에 이름을 지정해 QSS에서 특정 Widget만 선택할 수 있게 하는 메서드
        header_label.setObjectName("headerLabel")

        root_layout.addWidget(header_label)

        # Main Content
        content_layout = QHBoxLayout()

        content_layout.setSpacing(SPACE_24)

        self.navigation_frame = self._create_navigation()

        # QFrame: Container Widget
        self.page_frame = QFrame()
        self.page_frame.setObjectName("pageFrame")

        page_layout = QVBoxLayout(self.page_frame)

        page_layout.setContentsMargins(
            SPACE_16,
            SPACE_16,
            SPACE_16,
            SPACE_16,
        )

        # QStackedWidget: 여러 Page를 담아두고 그중 하나의 Page만 화면에 표시하는 Widget
        self.stack = QStackedWidget()

        page_layout.addWidget(self.stack)

        self._create_pages()

        # 12 Column Grid 기준
        # Navigation 3 / Page 9
        content_layout.addWidget(
            self.navigation_frame,
            3,
        )

        content_layout.addWidget(
            self.page_frame,
            9,
        )

        root_layout.addLayout(content_layout)

        self.switch_page("home")

    def _create_navigation(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("navigationFrame")

        layout = QVBoxLayout(frame)

        layout.setContentsMargins(
            SPACE_16,
            SPACE_16,
            SPACE_16,
            SPACE_16,
        )

        navigation_items = [
            ("home", "Home"),
            ("data", "Data Lab        (데이터 탐색)"),
            ("preprocessing", "Preprocessing   (데이터 전처리)"),
            ("regression", "Regression      (회귀)"),
            ("classification", "Classification  (분류)"),
            ("evaluation", "Evaluation      (모델 평가)"),
            ("selection", "Model Selection  (모델 선택)"),
            ("unsupervised", "Unsupervised    (비지도 학습)"),
            ("final", "Final Experiment (종합 실습)"),
        ]

        # QPushButton: 클릭할 수 있는 Button Widget
        for page_name, text in navigation_items:
            button = QPushButton(text)
        
            button.setObjectName("navigationButton")
            # setCheckable(): QPushButton이 선택됨/선택 해제됨 상태를 가질 수 있게 만드는 메서드
            button.setCheckable(True)

            # clicked: QPushButton이 기본 제공하는 Signal
            # connect(): Signal과 실행할 Slot을 연결하는 메서드
            button.clicked.connect(
                lambda checked=False, name=page_name:
                # Signal을 받았을 때 실행하는 함수
                self.switch_page(name)
            )

            self.navigation_buttons[page_name] = button

            layout.addWidget(button)

        # addStretch(): Layout에서 남는 공간을 빈 여백으로 채워 Widget 위치를 조정하는 메서드
        layout.addStretch()

        return frame

    def _create_pages(self):
        self.pages = {}

        home_page = HomePage()

        # home_page의 data_source_selected = Signal(str)를 받는 Slot
        home_page.data_source_selected.connect(
            self._handle_data_source_selected
        )

        self._add_page(
            "home",
            home_page,
        )

        self._add_page(
            "data",
            PlaceholderPage(
                "Data Lab",
                "데이터 탐색",
            ),
        )

        self._add_page(
            "preprocessing",
            PlaceholderPage(
                "Preprocessing",
                "데이터 전처리",
            ),
        )

        self._add_page(
            "regression",
            PlaceholderPage(
                "Regression",
                "회귀",
            ),
        )

        self._add_page(
            "classification",
            PlaceholderPage(
                "Classification",
                "분류",
            ),
        )

        self._add_page(
            "evaluation",
            PlaceholderPage(
                "Evaluation",
                "모델 평가",
            ),
        )

        self._add_page(
            "selection",
            PlaceholderPage(
                "Model Selection",
                "모델 선택",
            ),
        )

        self._add_page(
            "unsupervised",
            PlaceholderPage(
                "Unsupervised",
                "비지도 학습",
            ),
        )

        self._add_page(
            "final",
            PlaceholderPage(
                "Final Experiment",
                "종합 실습",
            ),
        )

    def _add_page(
        self,
        name: str,
        page: QWidget,
    ):
        self.pages[name] = page
        self.stack.addWidget(page)

    def switch_page(self, page_name: str):
        page = self.pages[page_name]

        self.stack.setCurrentWidget(page) # Page를 바꾼다.

        for name, button in self.navigation_buttons.items():
            button.setChecked(name == page_name)

    def _handle_data_source_selected(
        self,
        source: str,
    ):
        self.selected_data_source = source

        self.switch_page("data")