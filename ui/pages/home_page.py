# 머신러닝 개념 안내와 데이터셋 선택을 제공하는 홈 화면을 구성
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from common.theme import (
    FONT_WEIGHT_SEMIBOLD, SPACE_2XS, SPACE_LG, SPACE_MD, SPACE_SM,
    SPACE_XL, SPACE_XS,
)


# 제목이 색상 영역의 위쪽 경계에 겹치는 다이어그램 레이어를 구성
class DiagramLayer(QWidget):
    # 다이어그램 레이어의 제목과 색상 프레임 초기화
    def __init__(self, title: str, object_name: str) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.frame = QFrame(self)
        self.frame.setObjectName(object_name)
        self.content_layout = QVBoxLayout(self.frame)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("diagramTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # 내부 콘텐츠와 패딩을 기준으로 필요한 레이어 크기를 전달
    def sizeHint(self) -> QSize:
        title_size = self.title_label.sizeHint()
        frame_size = self.frame.sizeHint()
        return QSize(
            max(title_size.width(), frame_size.width()),
            title_size.height() // 2 + frame_size.height(),
        )

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # 제목의 중심이 색상 영역 위쪽 경계와 겹치도록 배치
    def resizeEvent(self, event: QResizeEvent) -> None:
        title_height = self.title_label.sizeHint().height()
        frame_top = title_height // 2

        self.frame.setGeometry(0, frame_top, self.width(), self.height() - frame_top)
        self.title_label.setGeometry(0, 0, self.width(), title_height)
        self.title_label.raise_()

        super().resizeEvent(event)


# 머신러닝 개념 안내와 데이터셋 선택 기능을 제공하는 홈 화면을 관리
class HomePage(QWidget):
    data_source_selected = Signal(str)

    # 홈 화면 기본 상태 초기화
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("homePage")
        self._setup_ui()

    # 머신러닝 개념과 데이터셋 선택 영역을 배치
    def _setup_ui(self) -> None:
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(0)

        layout.addWidget(self._create_machine_learning_section())
        layout.addSpacing(SPACE_XL)

        layout.addWidget(self._create_program_comparison_section())
        layout.addSpacing(SPACE_XL)

        layout.addWidget(self._create_learning_methods_section())
        layout.addSpacing(SPACE_XL)

        layout.addWidget(self._create_dataset_selection_section())

    # AI 안에 ML, ML 안에 DL이 포함되는 관계 구역을 생성
    def _create_machine_learning_section(self) -> QFrame:
        diagram = QFrame()
        diagram.setObjectName("machineLearningDiagram")

        diagram_layout = QVBoxLayout(diagram)
        diagram_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        diagram_layout.setSpacing(0)

        ai_layer = DiagramLayer("AI", "aiLayer")
        ai_layout = ai_layer.content_layout
        ai_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        ai_layout.setSpacing(SPACE_SM)
        ai_layout.addWidget(self._create_diagram_label(
            "<b>인공지능(AI)</b>은 컴퓨터와 기계가 인간의 학습, 이해, 문제 해결,<br>"
            "의사 결정, 창의성 및 자율성을 모방할 수 있도록 하는 기술 분야다.",
            "diagramText",
        ))
        ml_layer = DiagramLayer("ML", "mlLayer")
        ml_layout = ml_layer.content_layout
        ml_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        ml_layout.setSpacing(SPACE_SM)
        ml_layout.addWidget(self._create_diagram_label(
            "<b>머신러닝(Machine Learning)</b>은<br>"
            "데이터에서 패턴이나 관계를 학습하여 새로운 데이터에 대한 예측이나<br>"
            "판단을 수행할 수 있도록 모델을 만드는 방법이다.",
            "diagramText",
        ))
        dl_layer = DiagramLayer("DL", "dlLayer")
        dl_layout = dl_layer.content_layout
        dl_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        dl_layout.addWidget(self._create_diagram_label(
            "<b>딥러닝(Deep Learning)</b>은 여러 층으로 구성된 인공신경망을 이용해<br>"
            "데이터를 학습하는 머신러닝의 한 분야다.",
            "diagramText",
        ))

        ml_layout.addWidget(dl_layer)
        ai_layout.addWidget(ml_layer)
        diagram_layout.addWidget(ai_layer)
        diagram_layout.addSpacing(SPACE_MD)

        summary_layout = QVBoxLayout()
        summary_layout.setSpacing(0)
        summary_layout.addWidget(self._create_rich_text_label(
            "오늘날 사용되는 많은 <b>생성형 AI</b>는 딥러닝 기술을 기반으로 만들어진다.",
            "diagramSummary",
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        ))
        summary_layout.addSpacing(SPACE_XS)
        summary_layout.addWidget(self._create_rich_text_label(
            "<b>인공지능</b>은 데이터를 분석하거나 분류하고 예측하는 다양한 작업에 활용될 수 있다.",
            "diagramSummary",
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        ))
        summary_layout.addWidget(self._create_rich_text_label(
            "<b>생성형 AI</b>는 학습한 데이터의 패턴을 바탕으로 텍스트, 이미지, 음성, 코드와 같은 "
            "새로운 콘텐츠를 생성할 수 있다.",
            "diagramSummary",
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        ))
        diagram_layout.addLayout(summary_layout)

        return diagram

    # 일반 프로그램과 머신러닝의 처리 방식을 비교하는 구역을 생성
    def _create_program_comparison_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("programComparisonSection")

        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        section_layout.setSpacing(SPACE_MD)

        comparison_layout = QHBoxLayout()
        comparison_layout.setSpacing(SPACE_MD)
        comparison_layout.addWidget(self._create_program_comparison_card(
            "일반 프로그램",
            "사람이 규칙을 작성하고 결과를 출력한다.",
            "score = 80\n\n"
            "if score >= 60:\n"
            "    print(\"합격\")\n"
            "else:\n"
            "    print(\"불합격\")",
            "사람이 만든 규칙\n↓\n결과",
            Qt.AlignmentFlag.AlignLeft,
        ), 1)
        comparison_layout.addWidget(self._create_program_comparison_card(
            "머신러닝",
            "주어진 데이터에서 패턴과 관계를 학습하도록 한다.",
            "데이터 + 알고리즘\n"
            "↓\n"
            "(학습)\n"
            "↓\n"
            "모델\n"
            "↓\n"
            "(새로운 데이터 입력)\n"
            "↓\n"
            "예측·판단",
            "",
            Qt.AlignmentFlag.AlignCenter,
        ), 1)
        section_layout.addLayout(comparison_layout)

        definitions_layout = QVBoxLayout()
        definitions_layout.setContentsMargins(0, 0, 0, 0)
        definitions_layout.setSpacing(SPACE_2XS)

        definitions = (
            "<b>데이터(Data):</b> 모델이 학습하거나 예측할 때 사용하는 값",
            "<b>알고리즘(Algorithm):</b> 데이터에서 어떤 방법으로 패턴이나 관계를 찾을 것인지 "
            "정해 놓은 절차와 방법",
            "<b>학습(Training):</b> 데이터를 이용해 모델이 패턴이나 관계를 찾도록 만드는 과정",
            "<b>모델(Model):</b> 알고리즘을 데이터에 적용해 학습한 결과",
            "<b>예측(Prediction):</b> 학습한 관계를 이용해 새로운 데이터의 결과를 구하는 과정",
        )

        for definition in definitions:
            definitions_layout.addWidget(self._create_rich_text_label(
                definition,
                "programComparisonDefinitions",
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            ))

        section_layout.addLayout(definitions_layout)

        return section

    # 지도학습·비지도학습·강화학습 설명 구역을 생성
    def _create_learning_methods_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("learningMethodsSection")

        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        section_layout.setSpacing(SPACE_MD)

        intro_card = QFrame()
        intro_card.setObjectName("learningIntroCard")
        intro_layout = QVBoxLayout(intro_card)
        intro_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        intro_layout.setSpacing(SPACE_SM)

        intro_layout.addWidget(self._create_rich_text_label(
            "머신러닝에서는 <b>데이터</b>와 <b>알고리즘</b>을 이용해 모델을 만든다.",
            "learningIntroText",
            Qt.AlignmentFlag.AlignCenter,
        ))
        intro_layout.addWidget(self._create_rich_text_label(
            "이 프로그램에서는 머신러닝 알고리즘의 세부 원리와 직접 구현 방법을 다루지 않는다.<br>"
            "대신 Python 머신러닝 라이브러리가 제공하는 알고리즘을 활용해 각 학습 방식을 실습한다.",
            "learningIntroText",
            Qt.AlignmentFlag.AlignCenter,
        ))
        section_layout.addWidget(intro_card)

        section_title = QLabel("머신러닝의 학습 방식")
        section_title.setObjectName("learningSectionTitle")
        section_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        section_layout.addWidget(section_title)

        method_layout = QHBoxLayout()
        method_layout.setSpacing(SPACE_MD)
        method_layout.addWidget(self._create_learning_method_card(
            "지도학습",
            "입력 데이터와 정답을 함께 사용해\n입력과 정답의 관계를 학습하는 방식",
            "회귀 · 분류",
        ), 1)
        method_layout.addWidget(self._create_learning_method_card(
            "비지도학습",
            "정답이 주어지지 않은 데이터에서\n패턴이나 구조를 찾는 학습 방식",
            "군집화",
        ), 1)
        method_layout.addWidget(self._create_learning_method_card(
            "강화학습",
            "환경과 상호작용하면서\n장기적으로 더 높은 보상을 얻는\n행동을 학습하는 방식",
            "보상(Reward) 기반 학습",
        ), 1)
        section_layout.addLayout(method_layout)

        self.learning_details_toggle = QPushButton("[자세히 보기 ▼]")
        self.learning_details_toggle.setObjectName("learningDetailsToggle")
        self.learning_details_toggle.setCheckable(True)
        self.learning_details_toggle.toggled.connect(self._on_learning_details_toggled)
        section_layout.addWidget(self.learning_details_toggle)

        self.learning_details = QWidget()
        details_layout = QVBoxLayout(self.learning_details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(SPACE_MD)
        details_layout.addWidget(self._create_learning_detail_card(
            "<b>회귀(Regression):</b> 연속적인 숫자 값을 예측하는 문제<br>"
            "예) 공부 시간과 시험 점수, 집의 정보와 집값, 과거 매출과 다음 달 매출<br><br>"
            "<b>분류(Classification):</b> 입력 데이터가 어떤 범주에 속하는지를 예측하는 문제<br>"
            "예) 이메일 정상 또는 스팸, 시험 결과 합격 또는 불합격, 이미지 고양이 또는 강아지"
        ))
        details_layout.addWidget(self._create_learning_detail_card(
            "<b>군집화(Clustering):</b> 비슷한 데이터를 여러 그룹으로 묶는 방법<br>"
            "예) 구매 행동이 비슷한 고객을 찾아 여러 그룹으로 나누기"
        ))
        details_layout.addWidget(self._create_learning_detail_card(
            "<b>강화학습(Reinforcement Learning):</b> 행동을 수행하는 주체를 <b>Agent</b>, "
            "Agent가 상호작용하는 대상을 <b>Environment</b>, 행동의 결과로 받는 값을 "
            "<b>Reward</b>라고 한다."
        ))
        self.learning_details.setVisible(False)
        section_layout.addWidget(self.learning_details)

        return section

    # 데이터셋 선택 안내와 출처별 선택 카드를 포함하는 구역을 생성
    def _create_dataset_selection_section(self) -> QFrame:
        section = QFrame()
        section.setObjectName("datasetSelectionSection")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        section_layout.setSpacing(0)

        introduction_card = QFrame()
        introduction_card.setObjectName("datasetIntroductionCard")
        introduction_layout = QVBoxLayout(introduction_card)
        introduction_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        introduction_layout.setSpacing(SPACE_2XS)

        introduction_layout.addWidget(self._create_rich_text_label(
            "머신러닝의 목적은 모델이 학습할 때 사용하지 않은 새로운 데이터에서도 "
            "좋은 성능을 보이는 것이다.<br>"
            "이를 <b>일반화(Generalization)</b>라고 한다.",
            "datasetIntroductionText",
            Qt.AlignmentFlag.AlignCenter,
        ))
        introduction_layout.addWidget(self._create_rich_text_label(
            "같은 알고리즘을 사용하더라도 어떤 데이터를 사용했는지에 따라 모델의 결과는 "
            "달라질 수 있다.<br>"
            "머신러닝에서는 알고리즘뿐만 아니라 데이터를 확인하고 정리하고 분석하는 과정도 중요하다.",
            "datasetIntroductionText",
            Qt.AlignmentFlag.AlignCenter,
        ))
        section_layout.addWidget(introduction_card)
        section_layout.addSpacing(SPACE_LG)

        guide_label = QLabel("데이터를 선택하면 실습이 시작됩니다.")
        guide_label.setObjectName("homeGuideLabel")
        guide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        section_layout.addWidget(guide_label)
        section_layout.addSpacing(SPACE_XL)

        card_layout = QHBoxLayout()
        card_layout.setSpacing(SPACE_MD)

        builtin_card = self._create_dataset_card(
            "Built-in Sample",
            "프로그램에 포함된\n학습용 예제 데이터",
            "builtin",
        )
        sklearn_card = self._create_dataset_card(
            "Scikit-learn Sample",
            "scikit-learn이 제공하는\n실습용 Dataset",
            "sklearn",
        )
        csv_card = self._create_dataset_card(
            "Local CSV",
            "직접 준비한 CSV 파일\n불러오기",
            "csv",
        )

        card_layout.addWidget(builtin_card, 1)
        card_layout.addWidget(sklearn_card, 1)
        card_layout.addWidget(csv_card, 1)

        section_layout.addLayout(card_layout)
        return section

    # 다이어그램에 사용할 가운데 정렬 텍스트 라벨을 생성
    def _create_diagram_label(self, text: str, object_name: str) -> QLabel:
        return self._create_rich_text_label(
            text,
            object_name,
            Qt.AlignmentFlag.AlignCenter,
        )

    # 비교 제목·내용·결과를 하나의 카드로 구성
    def _create_program_comparison_card(
        self,
        title: str,
        description: str,
        content: str,
        footer: str,
        content_alignment: Qt.AlignmentFlag,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("programComparisonCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_SM)

        title_label = QLabel(title)
        title_label.setObjectName("programComparisonTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setObjectName("programComparisonText")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        layout.addSpacing(SPACE_XS)

        content_card = QFrame()
        content_card.setObjectName("programComparisonContentCard")
        content_layout = QVBoxLayout(content_card)
        content_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)

        content_label = QLabel(content)
        content_label.setObjectName("programComparisonText")
        content_label.setAlignment(content_alignment | Qt.AlignmentFlag.AlignVCenter)
        content_layout.addWidget(content_label)
        layout.addWidget(content_card, 1)

        if footer:
            footer_label = QLabel(footer)
            footer_label.setObjectName("programComparisonText")
            footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(footer_label)

        return card

    # 학습 방식의 제목·설명·개념을 보여주는 카드를 생성
    def _create_learning_method_card(
        self,
        title: str,
        description: str,
        key_concept: str,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("learningMethodCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_XS)

        title_label = QLabel(title)
        title_label.setObjectName("learningMethodTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setObjectName("learningMethodText")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        layout.addWidget(description_label, 1)

        key_label = QLabel(key_concept)
        key_label.setObjectName("learningMethodText")
        key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        key_label.setWordWrap(True)
        layout.addWidget(key_label)

        return card

    # 학습 방식의 세부 개념을 보여주는 상세 카드를 생성
    def _create_learning_detail_card(self, text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("learningDetailCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.addWidget(self._create_rich_text_label(
            text,
            "learningDetailText",
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        ))

        return card

    # 선택한 데이터셋 출처를 전달하는 카드를 생성
    def _create_dataset_card(
        self,
        title: str,
        description: str,
        source: str,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("datasetCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(0)
        layout.addStretch()

        title_label = QLabel(title)
        title_label.setObjectName("datasetTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        layout.addSpacing(SPACE_SM)

        description_label = QLabel(description)
        description_label.setObjectName("datasetText")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description_label)

        layout.addSpacing(SPACE_SM)

        select_button = QPushButton("[선택]")
        select_button.setObjectName("datasetSelectButton")
        select_button.setAccessibleName(f"{title} {description}")
        select_button.clicked.connect(
            lambda checked=False: self.data_source_selected.emit(source)
        )
        layout.addWidget(select_button)

        layout.addStretch()
        return card
    # 강조할 용어에 별도 글꼴을 적용한 서식 있는 텍스트 라벨을 생성
    def _create_rich_text_label(
        self,
        text: str,
        object_name: str,
        alignment: Qt.AlignmentFlag,
    ) -> QLabel:
        text = text.replace(
            "<b>",
            f'<span style="font-weight: {FONT_WEIGHT_SEMIBOLD};">',
        ).replace("</b>", "</span>")

        label = QLabel(text)
        label.setObjectName(object_name)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(alignment)
        label.setWordWrap(True)
        return label

    # 자세히 보기 영역의 표시 상태와 토글 문구를 함께 변경
    def _on_learning_details_toggled(self, expanded: bool) -> None:
        self.learning_details.setVisible(expanded)
        self.learning_details_toggle.setText(
            "[자세히 접기 ▲]" if expanded else "[자세히 보기 ▼]"
        )
