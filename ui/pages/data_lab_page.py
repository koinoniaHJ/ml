# 데이터를 불러와 탐색하고 시각화하는 Data Lab 화면을 구성
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY,
    PREVIEW_COLUMN_COUNT, PREVIEW_ROW_COUNT, SPACE_MD, SPACE_SM, SPACE_XS,
)
from ml.data_loader import (
    get_builtin_class_names, get_builtin_target, get_builtin_task_type,
    get_sklearn_target, get_sklearn_target_names, get_sklearn_task_type,
    load_builtin_dataset, load_local_csv, load_sklearn_dataset,
)
from ml.data_summary import (
    get_data_summary, get_feature_columns, get_identifier_columns, get_numeric_columns,
    get_target_columns, get_target_unique_count,
)
from ui.widgets import ChevronComboBox


CONCEPT_DESCRIPTIONS = {
    "Dataset": "서로 관련된 여러 Sample을 행과 열로 정리한 데이터 집합이다.",
    "Sample": "하나의 관측 대상 또는 기록으로, 표에서는 일반적으로 한 행에 해당한다.",
    "Feature": "모델이 데이터의 패턴을 학습하거나 예측할 때 입력으로 사용하는 정보다.",
    "Target": "모델이 예측하려는 결과다. 비지도학습에서는 Target이 없을 수 있다.",
    "Class": "분류 문제에서 모델이 구분하려는 각각의 범주다.",
    "Column · dtype": "Column은 하나의 정보를 나타내고 dtype은 그 Column에 저장된 값의 종류를 나타낸다.",
    "Missing Value (결측치)": (
        "값이 비어 있거나 존재하지 않는 데이터다. Python에서는 None, Pandas에서는 NaN 등으로 "
        "나타나며, 삭제하거나 다른 값으로 채울 때는 데이터의 의미와 모델 학습을 함께 고려해야 한다. "
        "구체적인 처리 방법은 데이터 전처리에서 다룬다."
    ),
    "scikit-learn": "머신러닝 알고리즘과 전처리·평가 기능, 예제 Dataset을 제공하는 Python 라이브러리다.",
}

GRAPH_DESCRIPTIONS = {
    "Line Plot": "<b>Line Plot(선 그래프)</b>: 순서가 있는 값의 변화나 흐름을 선으로 연결해 확인한다.",
    "Bar Plot": "<b>Bar Plot(막대 그래프)</b>: 여러 항목의 숫자 값을 막대 높이로 비교한다.",
    "Scatter Plot": "<b>Scatter Plot(산점도)</b>: 각 점을 하나의 Sample로 나타내 두 값 사이의 관계를 확인한다.",
    "Histogram": "<b>Histogram(히스토그램)</b>: 숫자 데이터를 여러 구간으로 나누어 값이 어느 범위에 얼마나 모여 있는지 확인한다.",
    "Box Plot": "<b>Box Plot(상자그림)</b>: 사분위수를 바탕으로 데이터의 중심과 퍼짐을 요약하고 이상치를 파악한다.",
}

TASK_TYPE_LABELS = {
    "classification": "Classification (분류)",
    "regression": "Regression (회귀)",
    "clustering": "Clustering (군집화)",
    "unknown": "미정",
}


class DataLabPage(QWidget):
    dataset_loaded = Signal(object, str, object, str)

    # Data Lab의 기본 상태를 만들고 UI를 구성
    def __init__(self):
        super().__init__()

        self.dataframe = pd.DataFrame()
        self.data_source = "builtin"
        self.dataset_name = ""
        self.selected_concept = ""
        self.class_names = None
        self.task_type = None

        self._setup_ui()

    # Data Lab 전체 화면을 Section 단위로 구성
    def _setup_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        # QScrollArea: Page Title을 포함한 전체 Data Lab을 Scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        page_layout.addWidget(self.scroll_area)

        content = QWidget()
        self.scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_MD)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(SPACE_SM)

        page_title = QLabel("Data Lab")
        page_title.setObjectName("pageTitle")
        page_title.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(page_title)

        page_description = QLabel(
            "데이터를 불러와 표의 구조와 결측값을 확인하고, 그래프로 데이터를 시각화한다."
        )
        page_description.setObjectName("bodyText")
        page_description.setWordWrap(True)
        header_layout.addWidget(page_description, 1)

        layout.addLayout(header_layout)

        self._create_dataset_source(layout)
        self._create_concept_section(layout)
        self._create_data_section(layout)
        self._create_visualization_section(layout)

        layout.addStretch()
        self.set_data_source("builtin")

        self.concept_buttons["Dataset"].setChecked(True)
        self._show_concept("Dataset")

    # 주요 단락 제목에 공통으로 사용할 라벨을 생성
    def _create_section_badge(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionBadge")
        label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        return label

    # Dataset Source 선택 영역을 구성
    def _create_dataset_source(self, layout: QVBoxLayout):
        section_title = self._create_section_badge("Dataset Source")
        layout.addWidget(section_title, alignment=Qt.AlignmentFlag.AlignLeft)

        source_layout = QHBoxLayout()
        source_layout.setSpacing(SPACE_SM)

        self.source_combo = ChevronComboBox()
        self.source_combo.setObjectName("dataControl")
        self.source_combo.addItem("Built-in", "builtin")
        self.source_combo.addItem("scikit-learn", "sklearn")
        self.source_combo.addItem("Local CSV", "csv")
        self.source_combo.currentIndexChanged.connect(self._change_data_source)
        source_layout.addWidget(self.source_combo, 2)

        self.dataset_combo = ChevronComboBox()
        self.dataset_combo.setObjectName("dataControl")
        source_layout.addWidget(self.dataset_combo, 6)

        self.load_button = QPushButton("불러오기")
        self.load_button.setObjectName("dataButton")
        self.load_button.clicked.connect(self._load_selected_dataset)
        source_layout.addWidget(self.load_button, 1)

        layout.addLayout(source_layout)

    # Data Preview와 Dataset Information을 2열로 구성
    def _create_data_section(self, layout: QVBoxLayout):
        data_layout = QGridLayout()
        data_layout.setSpacing(SPACE_MD)

        preview_header_layout = QHBoxLayout()
        preview_header_layout.setContentsMargins(0, 0, 0, 0)
        preview_header_layout.setSpacing(SPACE_SM)

        preview_title = self._create_section_badge("Data Preview")
        preview_header_layout.addWidget(preview_title)

        preview_guide_label = QLabel(
            f"전체 데이터 중 최대 {PREVIEW_ROW_COUNT}개 Sample과 "
            f"앞의 {PREVIEW_COLUMN_COUNT}개 Column만 미리 표시합니다."
        )
        preview_guide_label.setObjectName("smallText")
        preview_guide_label.setWordWrap(True)
        preview_header_layout.addWidget(preview_guide_label, 1)

        data_layout.addLayout(preview_header_layout, 0, 0)

        information_title = self._create_section_badge("Dataset Information")
        data_layout.addWidget(information_title, 0, 1, Qt.AlignmentFlag.AlignLeft)

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(SPACE_XS)

        self.preview_note_label = QLabel()
        self.preview_note_label.setObjectName("smallText")
        preview_layout.addWidget(self.preview_note_label)

        self.data_table = QTableWidget()
        self.data_table.setObjectName("dataPreviewTable")
        self.data_table.setMinimumHeight(320)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setVisible(False)

        table_palette = self.data_table.palette()
        table_palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_OFF_WHITE))
        table_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_OFF_WHITE))
        self.data_table.setPalette(table_palette)
        self.data_table.viewport().setAutoFillBackground(True)

        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setTextElideMode(Qt.TextElideMode.ElideNone)

        self.data_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        preview_layout.addWidget(self.data_table)
        data_layout.addLayout(preview_layout, 1, 0)

        information_frame = QFrame()
        information_layout = QVBoxLayout(information_frame)
        information_layout.setContentsMargins(0, 0, 0, 0)
        information_layout.setSpacing(SPACE_SM)

        self.summary_label = QLabel("Samples    -\nColumns    -\nMissing    -")
        self.summary_label.setObjectName("smallText")
        information_layout.addWidget(self.summary_label)

        target_title = QLabel("Target")
        target_title.setObjectName("sectionTitle")
        information_layout.addWidget(target_title)

        self.target_combo = ChevronComboBox()
        self.target_combo.setObjectName("dataControl")
        self.target_combo.currentIndexChanged.connect(self._change_target)
        information_layout.addWidget(self.target_combo)

        self.target_guide_label = QLabel()
        self.target_guide_label.setObjectName("smallText")
        self.target_guide_label.setWordWrap(True)
        information_layout.addWidget(self.target_guide_label)

        feature_title = QLabel("Feature 후보")
        feature_title.setObjectName("sectionTitle")
        information_layout.addWidget(feature_title)

        self.feature_label = QLabel("-")
        self.feature_label.setObjectName("smallText")
        self.feature_label.setWordWrap(True)
        information_layout.addWidget(self.feature_label)

        self.target_info_label = QLabel()
        self.target_info_label.setObjectName("smallText")
        self.target_info_label.setWordWrap(True)
        information_layout.addWidget(self.target_info_label)

        self.class_info_label = QLabel()
        self.class_info_label.setObjectName("smallText")
        self.class_info_label.setTextFormat(Qt.TextFormat.RichText)
        information_layout.addWidget(self.class_info_label)

        information_layout.addStretch()

        data_layout.addWidget(information_frame, 1, 1)
        data_layout.setColumnStretch(0, 7)
        data_layout.setColumnStretch(1, 3)

        layout.addLayout(data_layout)

    # Visualization 영역을 구성
    def _create_visualization_section(self, layout: QVBoxLayout):
        section_title = self._create_section_badge("Visualization")
        layout.addWidget(section_title, alignment=Qt.AlignmentFlag.AlignLeft)

        matplotlib_description = QLabel(
            "Python에서 데이터를 그래프로 표현할 때 사용하는 시각화 라이브러리인 "
            "Matplotlib을 이용해 데이터를 시각화한다."
        )
        matplotlib_description.setObjectName("bodyText")
        matplotlib_description.setWordWrap(True)
        layout.addWidget(matplotlib_description)

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)

        graph_label = QLabel("Graph")
        graph_label.setObjectName("smallText")
        controls.addWidget(graph_label)

        self.graph_type_combo = ChevronComboBox()
        self.graph_type_combo.setObjectName("dataControl")
        self.graph_type_combo.addItems(
            ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot"]
        )
        self.graph_type_combo.currentTextChanged.connect(self._update_graph_controls)
        controls.addWidget(self.graph_type_combo)

        self.x_label = QLabel("Column")
        self.x_label.setObjectName("smallText")
        controls.addWidget(self.x_label)

        self.x_column_combo = ChevronComboBox()
        self.x_column_combo.setObjectName("dataControl")
        controls.addWidget(self.x_column_combo, 1)

        self.y_label = QLabel("Y")
        self.y_label.setObjectName("smallText")
        controls.addWidget(self.y_label)

        self.y_column_combo = ChevronComboBox()
        self.y_column_combo.setObjectName("dataControl")
        controls.addWidget(self.y_column_combo, 1)

        self.draw_button = QPushButton("그래프 그리기")
        self.draw_button.setObjectName("dataButton")
        self.draw_button.clicked.connect(self._draw_graph)
        controls.addWidget(self.draw_button)

        layout.addLayout(controls)

        self.graph_description_label = QLabel()
        self.graph_description_label.setObjectName("graphDescription")
        self.graph_description_label.setWordWrap(True)
        layout.addWidget(self.graph_description_label)

        self.figure = Figure(facecolor=COLOR_OFF_WHITE)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(300)

        # installEventFilter(): Canvas에서 발생하는 Wheel Event를 Page Scroll로 전달하기 위해 등록
        self.canvas.installEventFilter(self)

        # 그래프를 그리기 전에는 빈 Canvas를 표시하지 않음
        self.canvas.hide()

        layout.addWidget(self.canvas)

        self._update_graph_controls()

    # 데이터 탐색에 필요한 개념 버튼과 설명 영역을 구성
    def _create_concept_section(self, layout: QVBoxLayout):
        section_title = self._create_section_badge("Concept")
        layout.addWidget(section_title, alignment=Qt.AlignmentFlag.AlignLeft)

        button_layout = QGridLayout()
        button_layout.setSpacing(SPACE_SM)

        concepts = [
            "Dataset", "Sample", "Feature", "Target",
            "Class", "Column · dtype", "Missing Value (결측치)", "scikit-learn",
        ]

        # 개념 버튼 중 하나만 선택된 상태로 유지
        self.concept_button_group = QButtonGroup(self)
        self.concept_button_group.setExclusive(True)
        self.concept_buttons = {}

        for index, concept in enumerate(concepts):
            button = QPushButton(concept)
            button.setObjectName("conceptButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, name=concept: self._show_concept(name))

            self.concept_button_group.addButton(button)
            self.concept_buttons[concept] = button
            button_layout.addWidget(button, index // 4, index % 4)

        for column in range(4):
            button_layout.setColumnStretch(column, 1)

        layout.addLayout(button_layout)

        detail_card = QFrame()
        detail_card.setObjectName("conceptDetailCard")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, SPACE_SM)
        detail_layout.setSpacing(SPACE_XS)

        self.concept_description_label = QLabel()
        self.concept_description_label.setWordWrap(True)
        self.concept_description_label.setTextFormat(Qt.TextFormat.RichText)
        self.concept_description_label.setObjectName("bodyText")
        detail_layout.addWidget(self.concept_description_label)

        layout.addWidget(detail_card)

    # Matplotlib Canvas 위에서도 Mouse Wheel로 Page를 Scroll
    def eventFilter(self, watched, event):
        if watched is self.canvas and event.type() == QEvent.Type.Wheel:
            pixel_delta = event.pixelDelta().y()
            angle_delta = event.angleDelta().y()
            delta = pixel_delta if pixel_delta else angle_delta

            scroll_bar = self.scroll_area.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.value() - delta)

            return True

        return super().eventFilter(watched, event)

    # Dataset Source ComboBox에서 선택한 Source를 적용
    def _change_data_source(self, _index: int):
        source = self.source_combo.currentData()

        if source:
            self.set_data_source(source)

    # 선택한 Dataset Source에 맞게 Dataset 선택 UI를 변경
    def set_data_source(self, source: str):
        self.data_source = source
        source_index = self.source_combo.findData(source)

        if source_index >= 0:
            self.source_combo.blockSignals(True)
            self.source_combo.setCurrentIndex(source_index)
            self.source_combo.blockSignals(False)

        self.dataset_combo.clear()

        if source == "builtin":
            self.dataset_combo.addItem("Student Basic (기본 데이터 구조)", "Student Basic")
            self.dataset_combo.addItem("Student Dirty (결측값 전처리)", "Student Dirty")
            self.dataset_combo.addItem("Preprocessing Sample (전처리 실습)", "Preprocessing Sample")
            self.dataset_combo.addItem("Regression Sample (연속값 예측 회귀)", "Regression Sample")
            self.dataset_combo.addItem("Classification Sample (범주 예측 분류)", "Classification Sample")
            self.dataset_combo.addItem("Clustering Sample (비지도 데이터 그룹화)", "Clustering Sample")
            self.dataset_combo.addItem("Final Challenge (종합 실습)", "Final Challenge")
            self.dataset_combo.show()
            self.load_button.setText("불러오기")

        elif source == "sklearn":
            self.dataset_combo.addItem("Iris (붓꽃 다중 분류)", "Iris")
            self.dataset_combo.addItem("Wine (와인 다중 분류)", "Wine")
            self.dataset_combo.addItem("Breast Cancer (유방암 이진 분류)", "Breast Cancer")
            self.dataset_combo.addItem("Diabetes (당뇨 진행도 회귀)", "Diabetes")
            self.dataset_combo.show()
            self.load_button.setText("불러오기")

        elif source == "csv":
            self.dataset_combo.hide()
            self.load_button.setText("CSV 선택")

    # 현재 선택된 Dataset을 불러와 DataFrame으로 저장
    def _load_selected_dataset(self):
        if self.data_source == "builtin":
            self.dataset_name = self.dataset_combo.currentData()
            self.dataframe = load_builtin_dataset(self.dataset_name)
            self.class_names = get_builtin_class_names(self.dataset_name)
            self.task_type = get_builtin_task_type(self.dataset_name)

        elif self.data_source == "sklearn":
            self.dataset_name = self.dataset_combo.currentData()
            self.dataframe = load_sklearn_dataset(self.dataset_name)
            self.class_names = get_sklearn_target_names(self.dataset_name)
            self.task_type = get_sklearn_task_type(self.dataset_name)

        else:
            self._select_csv()
            return

        self._show_load_result()

    # Local CSV 파일을 선택해 DataFrame으로 저장
    def _select_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "CSV 파일 선택", "", "CSV Files (*.csv)")

        if not file_path:
            return

        try:
            dataframe = load_local_csv(Path(file_path))
        except pd.errors.EmptyDataError:
            QMessageBox.warning(self, "불러오기 실패", "CSV 파일에 데이터가 없습니다.")
            return
        except (pd.errors.ParserError, UnicodeDecodeError):
            QMessageBox.warning(self, "불러오기 실패", "CSV 파일 형식을 확인하세요.")
            return
        except OSError:
            QMessageBox.warning(self, "불러오기 실패", "CSV 파일을 읽을 수 없습니다.")
            return

        self.dataset_name = Path(file_path).name
        self.class_names = None
        self.task_type = "unknown"
        self.dataframe = dataframe
        self._show_load_result()

    # 불러온 Dataset의 기본 정보와 관련 UI를 갱신
    def _show_load_result(self):
        summary = get_data_summary(self.dataframe)

        self.summary_label.setText(
            f"Samples    {summary['sample_count']}\n"
            f"Columns    {summary['column_count']}\n"
            f"Missing    {summary['missing_count']}"
        )

        # Dataset을 새로 불러오면 이전 그래프를 제거하고 다시 숨김
        self.figure.clear()
        self.canvas.hide()

        self._configure_target()
        self._update_graph_columns()
        self._show_dataframe()
        self._emit_dataset_loaded()

    # Dataset Source에 따라 Target을 고정하거나 선택 가능하게 구성
    def _configure_target(self):
        self.target_combo.blockSignals(True)
        self.target_combo.clear()

        if self.data_source == "builtin":
            target = get_builtin_target(self.dataset_name)

            if target is None:
                self.target_combo.addItem("Target 없음 (비지도 학습)", None)
                self.target_guide_label.setText(
                    "비지도 학습은 정답 없이 데이터의 패턴이나 그룹을 찾기 때문에 Target을 사용하지 않습니다."
                )
            else:
                self.target_combo.addItem(target, target)
                self.target_guide_label.setText("Dataset의 학습 목적에 맞게 Target이 고정되어 있습니다.")

            self.target_combo.setEnabled(False)

        elif self.data_source == "sklearn":
            target = get_sklearn_target(self.dataset_name)
            self.target_combo.addItem(target, target)
            self.target_combo.setEnabled(False)
            self.target_guide_label.setText("scikit-learn에서 정의한 원래 Target을 사용하며 변경하지 않습니다.")

        else:
            self.target_combo.addItem("Target 없음 (비지도 학습)", None)

            for column in get_target_columns(self.dataframe):
                self.target_combo.addItem(column, column)

            self.target_combo.setEnabled(True)
            self.target_guide_label.setText(
                "Local CSV는 Target을 직접 선택합니다. 비지도 학습이라면 Target 없음을 선택합니다."
            )

        self.target_combo.blockSignals(False)
        self._update_target_info()

    # 현재 선택된 Target Column을 반환
    def _get_target_column(self) -> str | None:
        return self.target_combo.currentData()

    # Target 변경 내용을 화면과 다음 Preprocessing 단계에 전달
    def _change_target(self, _index: int = -1) -> None:
        self._update_target_info()
        self._emit_dataset_loaded()

    # 현재 Dataset과 Target 정보를 Preprocessing 페이지에 전달
    def _emit_dataset_loaded(self) -> None:
        if self.dataframe.empty:
            return

        self.dataset_loaded.emit(
            self.dataframe.copy(deep=True),
            self.dataset_name,
            self._get_target_column(),
            self.task_type,
        )

    # 분류 Target의 Class 번호, 화살표, 이름을 일정한 열에 맞춰 표시
    def _get_class_text(self, target: str) -> str:
        if self.class_names:
            rows = "".join(
                f'<tr><td width="24">{index}</td><td width="24">→</td>'
                f"<td>{escape(str(name))}</td></tr>"
                for index, name in enumerate(self.class_names)
            )
            return f'<table cellspacing="0" cellpadding="0">{rows}</table>'

        class_values = self.dataframe[target].dropna().unique().tolist()
        rows = "".join(f"<tr><td>{escape(str(value))}</td></tr>" for value in class_values)
        return f'<table cellspacing="0" cellpadding="0">{rows}</table>'

    # 선택한 Target을 기준으로 Feature 후보와 Target 정보를 표시
    def _update_target_info(self, _index: int = -1):
        target = self._get_target_column()
        feature_columns = get_feature_columns(self.dataframe, target)
        identifier_columns = get_identifier_columns(self.dataframe)
        feature_text = ", ".join(map(str, feature_columns)) or "-"

        self.class_info_label.clear()
        self.class_info_label.hide()

        if identifier_columns:
            identifiers = ", ".join(map(str, identifier_columns))
            feature_text += f"\n\n식별자 Column 제외\n{identifiers}"

        self.feature_label.setText(feature_text)

        if self.dataframe.empty:
            self.target_info_label.setText("표시할 데이터가 없습니다.")
            self._update_concept_detail()
            return

        if target is None:
            task_type = "Clustering (군집화)" if self.task_type == "clustering" else "Unsupervised (비지도 학습)"
            self.target_info_label.setText(f"문제 유형    {task_type}\nTarget    없음")

        else:
            unique_count = get_target_unique_count(self.dataframe, target)

            if self.task_type == "classification":
                class_text = self._get_class_text(target)
                target_text = (
                    f"문제 유형    {TASK_TYPE_LABELS[self.task_type]}\n"
                    f"Target    {target}\n"
                    f"Class 수    {unique_count}\n\n"
                    "Class"
                )
                self.class_info_label.setText(class_text)
                self.class_info_label.show()
            else:
                task_type = TASK_TYPE_LABELS.get(self.task_type, TASK_TYPE_LABELS["unknown"])
                target_text = (
                    f"문제 유형    {task_type}\n"
                    f"Target    {target}\n"
                    f"Target 고유값 수    {unique_count}"
                )

            self.target_info_label.setText(target_text)

        self._update_concept_detail()

    # 선택한 그래프에서 사용할 수 있는 Column을 선택 목록에 표시
    def _update_graph_columns(self):
        numeric_columns = get_numeric_columns(self.dataframe)
        graph_type = self.graph_type_combo.currentText()
        x_columns = list(map(str, self.dataframe.columns)) if graph_type == "Bar Plot" else numeric_columns

        selected_x = self.x_column_combo.currentText()
        selected_y = self.y_column_combo.currentText()

        self.x_column_combo.clear()
        self.y_column_combo.clear()
        self.x_column_combo.addItems(x_columns)
        self.y_column_combo.addItems(numeric_columns)

        if selected_x in x_columns:
            self.x_column_combo.setCurrentText(selected_x)

        if selected_y in numeric_columns:
            self.y_column_combo.setCurrentText(selected_y)
        elif len(numeric_columns) > 1:
            self.y_column_combo.setCurrentIndex(1)

    # 선택한 그래프에 필요한 Column UI를 표시
    def _update_graph_controls(self, _graph_type: str = ""):
        graph_type = self.graph_type_combo.currentText()
        uses_x_and_y = graph_type in {"Line Plot", "Bar Plot", "Scatter Plot"}
        self.x_label.setText("X" if uses_x_and_y else "Column")
        self.y_label.setVisible(uses_x_and_y)
        self.y_column_combo.setVisible(uses_x_and_y)
        self.graph_description_label.setText(GRAPH_DESCRIPTIONS.get(graph_type, ""))
        self._update_graph_columns()

    # 선택한 그래프 종류와 Column으로 그래프를 그림
    def _draw_graph(self):
        if self.dataframe.empty:
            return

        x_column = self.x_column_combo.currentText()

        if not x_column:
            return

        graph_type = self.graph_type_combo.currentText()

        if graph_type in {"Line Plot", "Bar Plot", "Scatter Plot"}:
            y_column = self.y_column_combo.currentText()

            if not y_column:
                return

            plot_data = pd.DataFrame(
                {"x": self.dataframe[x_column], "y": self.dataframe[y_column]}
            ).dropna()

            if graph_type == "Bar Plot":
                plot_data = plot_data[np.isfinite(plot_data["y"])]
            else:
                plot_data = plot_data[np.isfinite(plot_data).all(axis=1)]
        else:
            plot_data = self.dataframe[x_column].dropna()
            plot_data = plot_data[np.isfinite(plot_data)]

        if plot_data.empty:
            self.figure.clear()
            self.canvas.hide()
            QMessageBox.warning(
                self,
                "그래프 생성 실패",
                "유효한 숫자 데이터가 없습니다.",
            )
            return

        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.set_facecolor(COLOR_OFF_WHITE)

        if graph_type == "Line Plot":
            axes.plot(
                plot_data["x"],
                plot_data["y"],
                marker="o",
                color=COLOR_SECONDARY,
            )
            axes.set_xlabel(x_column)
            axes.set_ylabel(y_column)

        elif graph_type == "Bar Plot":
            x_labels = plot_data["x"].astype(str)
            axes.bar(x_labels, plot_data["y"], color=COLOR_SECONDARY)
            axes.set_xlabel(x_column)
            axes.set_ylabel(y_column)

            has_many_labels = x_labels.nunique() > 10
            has_long_label = x_labels.str.len().max() > 8
            if has_many_labels or has_long_label:
                axes.tick_params(axis="x", labelrotation=45)

        elif graph_type == "Histogram":
            axes.hist(plot_data, color=COLOR_SECONDARY)
            axes.set_xlabel(x_column)
            axes.set_ylabel("Frequency")

        elif graph_type == "Scatter Plot":
            axes.scatter(plot_data["x"], plot_data["y"], color=COLOR_SECONDARY)
            axes.set_xlabel(x_column)
            axes.set_ylabel(y_column)

        elif graph_type == "Box Plot":
            axes.boxplot(
                plot_data,
                boxprops={"color": COLOR_SECONDARY},
                whiskerprops={"color": COLOR_SECONDARY},
                capprops={"color": COLOR_SECONDARY},
                medianprops={"color": COLOR_OFF_BLACK},
                flierprops={"markeredgecolor": COLOR_SECONDARY},
            )
            axes.set_ylabel(x_column)

        axes.set_title(graph_type, color=COLOR_OFF_BLACK)
        axes.xaxis.label.set_color(COLOR_OFF_BLACK)
        axes.yaxis.label.set_color(COLOR_OFF_BLACK)
        axes.tick_params(colors=COLOR_OFF_BLACK)

        for spine in axes.spines.values():
            spine.set_color(COLOR_PRIMARY)

        # 그래프가 만들어진 뒤 Canvas를 표시
        self.canvas.show()
        self.canvas.draw()

    # DataFrame 앞부분을 dtype과 함께 Table에 표시
    def _show_dataframe(self):
        preview = self.dataframe.iloc[:PREVIEW_ROW_COUNT, :PREVIEW_COLUMN_COUNT]

        self.data_table.clear()
        self.data_table.horizontalHeader().setVisible(True)
        self.data_table.setRowCount(len(preview))
        self.data_table.setColumnCount(len(preview.columns))

        headers = []

        for column in preview.columns:
            column_name = str(column).replace("_", "\n").replace(" ", "\n")
            headers.append(f"{column_name}\n({self.dataframe[column].dtype})")

        self.data_table.setHorizontalHeaderLabels(headers)

        header = self.data_table.horizontalHeader()
        header_line_count = max(text.count("\n") + 1 for text in headers)
        header.setMinimumHeight(
            header.fontMetrics().lineSpacing() * header_line_count + 2 * SPACE_XS
        )

        for row_index in range(len(preview)):
            for column_index in range(len(preview.columns)):
                value = preview.iloc[row_index, column_index]
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.data_table.setItem(row_index, column_index, item)

        total_columns = len(self.dataframe.columns)
        self.preview_note_label.setText(
            f"Preview: {len(preview)} Samples(Row) / "
            f"{len(preview.columns)} of {total_columns} Columns"
        )

    # 선택한 데이터 개념과 현재 Dataset의 예시를 함께 표시
    def _show_concept(self, concept: str):
        self.selected_concept = concept
        description = CONCEPT_DESCRIPTIONS[concept]
        current_data = []

        if self.dataframe.empty:
            detail = self._format_concept_detail(concept, description, current_data)
            self.concept_description_label.setText(detail)
            return

        target = self._get_target_column()
        feature_columns = get_feature_columns(self.dataframe, target)

        if concept == "Dataset":
            current_data.append(
                f"현재 Dataset\n{self.dataset_name}\n"
                f"{len(self.dataframe)} Samples · {len(self.dataframe.columns)} Columns"
            )

        elif concept == "Sample":
            current_data.append(f"현재 Dataset의 Sample 수\n{len(self.dataframe)}개")

        elif concept == "Feature":
            feature_text = ", ".join(map(str, feature_columns)) or "없음"
            current_data.append(f"현재 Feature 후보\n{feature_text}")
            identifier_columns = get_identifier_columns(self.dataframe)

            if identifier_columns:
                identifiers = ", ".join(map(str, identifier_columns))
                current_data.append(
                    f"식별자 Column 제외\n{identifiers}\n"
                    "식별자는 대상을 구분하기 위한 값이므로 기본 Feature 후보에서 제외한다."
                )

        elif concept == "Target":
            if target:
                current_data.append(f"현재 Target\n{target}")
            else:
                current_data.append("현재 Dataset은 비지도학습용이므로 Target을 사용하지 않는다.")

        elif concept == "Class":
            if target is None:
                current_data.append("Target이 없는 비지도학습에서는 미리 정해진 Class도 없다.")
            elif self.task_type == "classification":
                current_data.append(f"현재 Class\n{self._get_class_text(target)}")
            elif self.task_type == "regression":
                current_data.append("현재 Dataset은 연속값을 예측하는 회귀 문제이므로 Class를 사용하지 않는다.")
            else:
                unique_count = get_target_unique_count(self.dataframe, target)
                current_data.append(
                    f"현재 Target 고유값 수: {unique_count}\n"
                    "고유값 개수만으로 분류와 회귀를 결정하지 않고 학습 목적도 함께 확인한다."
                )

        elif concept == "Column · dtype":
            preview_columns = list(self.dataframe.columns[:PREVIEW_COLUMN_COUNT])
            column_types = "\n".join(
                f"{column}: {self.dataframe[column].dtype}"
                for column in preview_columns
            )
            remaining_count = len(self.dataframe.columns) - len(preview_columns)

            if remaining_count > 0:
                column_types += f"\n외 {remaining_count}개 Column"

            current_data.append(f"현재 Column과 dtype\n{column_types}")

        elif concept == "Missing Value (결측치)":
            missing_counts = self.dataframe.isna().sum()
            missing_columns = missing_counts[missing_counts > 0]
            total_missing = int(missing_counts.sum())

            if missing_columns.empty:
                current_data.append("현재 Dataset의 Missing Value (결측치)\n0개")
            else:
                column_text = "\n".join(
                    f"{column}: {count}개"
                    for column, count in missing_columns.items()
                )
                current_data.append(
                    f"현재 Dataset의 Missing Value (결측치)\n총 {total_missing}개\n{column_text}"
                )

        elif concept == "scikit-learn":
            if self.data_source == "sklearn":
                current_data.append(f"현재 scikit-learn Dataset\n{self.dataset_name}")

        detail = self._format_concept_detail(concept, description, current_data)
        self.concept_description_label.setText(detail)

    # 개념 이름은 굵게 표시하고 현재 Dataset 정보는 줄바꿈을 유지
    def _format_concept_detail(
        self,
        concept: str,
        description: str,
        current_data: list[str],
    ) -> str:
        formatted_description = escape(description)
        if concept not in {"Target", "Missing Value (결측치)"}:
            formatted_description = formatted_description.replace("다. ", "다.<br>")
        detail = f"<b>{escape(concept)}</b>: {formatted_description}"

        if current_data:
            current_text = "<br><br>".join(
                escape(text).replace("\n", "<br>")
                for text in current_data
            )
            detail += f"<br><br>{current_text}"

        return detail

    # Target 변경 시 현재 Concept Detail을 다시 표시
    def _update_concept_detail(self, _value: str = ""):
        if self.selected_concept:
            self._show_concept(self.selected_concept)
