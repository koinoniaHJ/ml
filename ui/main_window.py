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
    WINDOW_HEIGHT,
)
from ui.pages.data_lab_page import DataLabPage
from ui.pages.home_page import HomePage
from ui.pages.placeholder_page import PlaceholderPage


# QMainWindow: 프로그램의 최상위 Main Window를 만드는 클래스
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Machine Learning Lab")

        self.resize(
            GRID_PC_MAX_WIDTH,
            WINDOW_HEIGHT,
        )

        self.navigation_buttons = {}

        self._setup_ui()

    # Main Window의 전체 UI를 구성
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

        root_layout.setSpacing(SPACE_24)

        header_label = QLabel("Machine Learning Lab")
        header_label.setObjectName("headerLabel")

        root_layout.addWidget(header_label)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(SPACE_24)

        self.navigation_frame = self._create_navigation()

        # QFrame: Widget들을 하나의 영역으로 묶을 때 사용하는 Container Widget
        self.page_frame = QFrame()
        self.page_frame.setObjectName("pageFrame")

        page_layout = QVBoxLayout(self.page_frame)

        page_layout.setContentsMargins(
            SPACE_16,
            SPACE_16,
            SPACE_16,
            SPACE_16,
        )

        # QStackedWidget: 여러 Page를 담고 그중 하나만 화면에 표시하는 Widget
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

    # 왼쪽 Navigation 영역을 생성
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

        for page_name, text in navigation_items:
            button = QPushButton(text)

            button.setObjectName("navigationButton")

            # setCheckable(): QPushButton이 선택됨/선택 해제됨 상태를 가질 수 있게 만드는 메서드
            button.setCheckable(True)

            button.clicked.connect(
                lambda checked=False, name=page_name:
                self.switch_page(name)
            )

            self.navigation_buttons[page_name] = button

            layout.addWidget(button)

        layout.addStretch()

        return frame

    # 사용할 Page들을 생성하고 QStackedWidget에 등록
    def _create_pages(self):
        self.pages = {}

        home_page = HomePage()

        home_page.data_source_selected.connect(
            self._handle_data_source_selected
        )

        self.data_lab_page = DataLabPage()

        self._add_page(
            "home",
            home_page,
        )

        self._add_page(
            "data",
            self.data_lab_page,
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

    # Page를 이름과 함께 저장하고 QStackedWidget에 추가
    def _add_page(
        self,
        name: str,
        page: QWidget,
    ):
        self.pages[name] = page
        self.stack.addWidget(page)

    # 선택한 Page를 화면에 표시
    def switch_page(
        self,
        page_name: str,
    ):
        page = self.pages[page_name]

        # setCurrentWidget(): QStackedWidget에서 현재 표시할 Page를 변경
        self.stack.setCurrentWidget(page)

        for name, button in self.navigation_buttons.items():
            button.setChecked(
                name == page_name
            )

    # 선택한 Dataset Source를 Data Lab에 전달하고 해당 Page로 이동
    def _handle_data_source_selected(
        self,
        source: str,
    ):
        self.data_lab_page.set_data_source(
            source
        )

        self.switch_page("data")