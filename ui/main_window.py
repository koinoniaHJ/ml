# 애플리케이션의 메인 창과 페이지 전환 구조를 구성
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QLayout, QMainWindow, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from common.theme import (
    PAGE_PADDING_X, PAGE_PADDING_Y, SIDEBAR_WIDTH, SPACE_MD, SPACE_XS,
    WINDOW_INITIAL_HEIGHT, WINDOW_INITIAL_WIDTH,
)
from ui.pages.data_lab_page import DataLabPage
from ui.pages.home_page import HomePage
from ui.pages.placeholder_page import PlaceholderPage
from ui.pages.preprocessing_page import PreprocessingPage


class MainWindow(QMainWindow):
    # 메인 창의 기본 UI와 페이지를 구성
    def __init__(self) -> None:
        super().__init__()

        self.setObjectName("mainWindow")
        self.setWindowTitle("Machine Learning Lab")
        self.resize(WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT)

        self.navigation_buttons: dict[str, QPushButton] = {}

        self._setup_ui()
        self.switch_page("home")

    # 헤더, 사이드 메뉴, 페이지 영역을 구성
    def _setup_ui(self) -> None:
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        layout.setContentsMargins(
            PAGE_PADDING_X,
            PAGE_PADDING_Y,
            PAGE_PADDING_X,
            PAGE_PADDING_Y,
        )

        # 헤더와 아래 콘텐츠 사이에 기본 간격을 적용
        layout.setSpacing(SPACE_MD)

        header_label = QLabel("Machine Learning Lab")
        header_label.setObjectName("headerLabel")
        layout.addWidget(header_label)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(SPACE_MD)

        navigation_frame = self._create_navigation()
        page_frame = self._create_pages()

        content_layout.addWidget(navigation_frame)
        content_layout.addWidget(page_frame, 1)

        layout.addLayout(content_layout, 1)

    # 왼쪽 사이드 메뉴 영역을 생성
    def _create_navigation(self) -> QFrame:
        navigation_frame = QFrame()
        navigation_frame.setObjectName("navigationFrame")
        navigation_frame.setFixedWidth(SIDEBAR_WIDTH)

        layout = QVBoxLayout(navigation_frame)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(0)

        # 여러 메뉴 중 하나만 선택된 상태로 유지
        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)

        navigation_items = [
            ("home", "Home", ""),
            ("data", "Data Lab", "(데이터 탐색)"),
            ("preprocessing", "Preprocessing", "(데이터 전처리)"),
            ("regression", "Regression", "(회귀)"),
            ("classification", "Classification", "(분류)"),
            ("evaluation", "Evaluation", "(모델 평가)"),
            ("selection", "Model Selection", "(모델 선택)"),
            ("unsupervised", "Unsupervised", "(비지도 학습)"),
            ("final", "Final Experiment", "(종합 실습)"),
        ]

        for page_name, english, korean in navigation_items:
            button = QPushButton()
            button.setObjectName("navigationButton")
            button.setCheckable(True)
            button.setAccessibleName(f"{english} {korean}".strip())
            button.clicked.connect(lambda checked=False, name=page_name: self.switch_page(name))

            text_layout = QHBoxLayout(button)
            text_layout.setContentsMargins(SPACE_XS, SPACE_XS, SPACE_XS, SPACE_XS)
            text_layout.setSpacing(SPACE_XS)
            text_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

            english_label = QLabel(english)
            english_label.setObjectName("navigationButtonText")
            english_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            english_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            text_layout.addWidget(english_label)
            text_layout.addStretch()

            if korean:
                korean_label = QLabel(korean)
                korean_label.setObjectName("navigationButtonText")
                korean_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                korean_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                text_layout.addWidget(korean_label)

            self.navigation_group.addButton(button)
            self.navigation_buttons[page_name] = button
            layout.addWidget(button)

        layout.addStretch()
        return navigation_frame

    # 오른쪽 페이지 영역과 단계별 페이지를 생성
    def _create_pages(self) -> QFrame:
        page_frame = QFrame()
        page_frame.setObjectName("pageFrame")

        layout = QVBoxLayout(page_frame)
        layout.setContentsMargins(0, 0, 0, 0)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")

        self.home_page = HomePage()
        self.data_lab_page = DataLabPage()
        self.preprocessing_page = PreprocessingPage()

        self.pages = {
            "home": self.home_page,
            "data": self.data_lab_page,
            "preprocessing": self.preprocessing_page,
            "regression": PlaceholderPage("Regression"),
            "classification": PlaceholderPage("Classification"),
            "evaluation": PlaceholderPage("Evaluation"),
            "selection": PlaceholderPage("Model Selection"),
            "unsupervised": PlaceholderPage("Unsupervised"),
            "final": PlaceholderPage("Final Experiment"),
        }

        for page in self.pages.values():
            self.page_stack.addWidget(page)

        self.home_page.data_source_selected.connect(self._handle_data_source_selected)
        self.data_lab_page.dataset_loaded.connect(self.preprocessing_page.set_dataset)
        layout.addWidget(self.page_stack)

        return page_frame

    # 선택한 사이드 메뉴 페이지로 전환
    def switch_page(self, page_name: str) -> None:
        self.page_stack.setCurrentWidget(self.pages[page_name])
        self.navigation_buttons[page_name].setChecked(True)

    # 홈에서 선택한 데이터셋 출처를 Data Lab으로 전달
    def _handle_data_source_selected(self, source: str) -> None:
        self.data_lab_page.set_data_source(source)
        self.switch_page("data")
