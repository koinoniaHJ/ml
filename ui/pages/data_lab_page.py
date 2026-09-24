# 데이터를 불러와 탐색하고 시각화하는 Data Lab 화면을 구성
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QListView, QPlainTextEdit, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY,
    PREVIEW_COLUMN_COUNT, PREVIEW_ROW_COUNT, SPACE_SM, SPACE_MD, TABLE_HEADER_HEIGHT,
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


CONCEPT_DESCRIPTIONS = {
    "Sample": "Dataset을 구성하는 하나의 데이터 단위로, 일반적으로 한 Row에 해당한다.",
    "Feature": "Model이 데이터의 패턴을 학습할 때 사용하는 입력 데이터다.",
    "Target": "지도 학습에서 Model이 예측하려는 정답 데이터다.",
    "Class": "분류 문제에서 Target이 가질 수 있는 범주다.",
    "X / y": "X는 Model에 입력하는 Feature 데이터이고, y는 지도 학습에서 사용하는 Target 데이터다.",
    "Train / Test": "Train은 Model 학습에 사용하고 Test는 학습된 Model의 성능 확인에 사용한다.",
    "fit()": "Train 데이터를 이용해 Model을 학습시키는 메서드다.",
    "predict()": "학습된 Model을 이용해 결과를 생성하는 메서드로, 지원 여부와 결과의 의미는 Model에 따라 다르다.",
    "Histogram": "하나의 숫자형 Column 값이 어떤 범위에 얼마나 분포하는지 확인하는 그래프다.",
    "Scatter Plot": "두 숫자형 Column 사이의 관계를 점으로 확인하는 그래프다.",
    "Box Plot": "숫자형 Column의 중앙값, 분포, 이상치 등을 Box 형태로 확인하는 그래프다.",
    "1D Array": "한 방향으로 값이 나열된 NumPy 배열이다.",
    "2D Array": "행과 열로 구성된 NumPy 배열이다.",
    "shape": "NumPy Array의 각 차원 크기를 튜플로 나타내는 속성이다.",
    "indexing": "위치 번호를 이용해 NumPy Array의 특정 값을 선택하는 방법이다.",
}

TASK_TYPE_LABELS = {
    "classification": "Classification (분류)",
    "regression": "Regression (회귀)",
    "clustering": "Clustering (군집화)",
    "unknown": "미정",
}


# QPainter: Widget 위에 직접 선이나 도형을 그리는 Qt 객체
class ChevronComboBox(QComboBox):
    # ComboBox의 Dropdown View를 구성
    def __init__(self):
        super().__init__()

        view = QListView()
        view.setFrameShape(QFrame.Shape.NoFrame)
        self.setView(view)

        palette = view.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_OFF_WHITE))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_OFF_WHITE))
        palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_OFF_BLACK))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_SECONDARY))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(COLOR_OFF_BLACK))
        view.setPalette(palette)

    # 기본 ComboBox를 그린 뒤 오른쪽에 아래 방향 꺾쇠를 표시
    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.isEnabled():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor(COLOR_OFF_BLACK))
        pen.setWidthF(1.5)
        painter.setPen(pen)

        center_x = self.width() - 16
        center_y = self.height() / 2

        path = QPainterPath()
        path.moveTo(QPointF(center_x - 4, center_y - 2))
        path.lineTo(QPointF(center_x, center_y + 2))
        path.lineTo(QPointF(center_x + 4, center_y - 2))

        painter.drawPath(path)


class DataLabPage(QWidget):
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

        page_title = QLabel("Data Lab - 데이터 탐색")
        page_title.setObjectName("pageTitle")
        layout.addWidget(page_title)

        self._create_dataset_source(layout)
        self._create_data_section(layout)
        self._create_visualization_section(layout)
        self._create_concept_section(layout)

        layout.addStretch()
        self.set_data_source("builtin")

    # Dataset Source 선택 영역을 구성
    def _create_dataset_source(self, layout: QVBoxLayout):
        section_title = QLabel("Dataset Source")
        section_title.setObjectName("sectionTitle")
        layout.addWidget(section_title)

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

        self.status_label = QLabel()
        self.status_label.setObjectName("smallText")
        layout.addWidget(self.status_label)

    # Data Preview와 Dataset Information을 2열로 구성
    def _create_data_section(self, layout: QVBoxLayout):
        data_layout = QGridLayout()
        data_layout.setSpacing(SPACE_MD)

        preview_title = QLabel("Data Preview")
        preview_title.setObjectName("sectionTitle")
        data_layout.addWidget(preview_title, 0, 0)

        information_title = QLabel("Dataset Information")
        information_title.setObjectName("sectionTitle")
        data_layout.addWidget(information_title, 0, 1)

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

        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFixedHeight(TABLE_HEADER_HEIGHT)
        header.setTextElideMode(Qt.TextElideMode.ElideNone)

        self.data_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        data_layout.addWidget(self.data_table, 1, 0)

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
        self.target_combo.currentIndexChanged.connect(self._update_target_info)
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

        information_layout.addStretch()

        data_layout.addWidget(information_frame, 1, 1)
        data_layout.setColumnStretch(0, 7)
        data_layout.setColumnStretch(1, 3)

        layout.addLayout(data_layout)

        self.preview_note_label = QLabel()
        self.preview_note_label.setObjectName("smallText")
        layout.addWidget(self.preview_note_label)

    # Visualization 영역을 구성
    def _create_visualization_section(self, layout: QVBoxLayout):
        section_title = QLabel("Visualization")
        section_title.setObjectName("sectionTitle")
        layout.addWidget(section_title)

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)

        graph_label = QLabel("Graph")
        graph_label.setObjectName("smallText")
        controls.addWidget(graph_label)

        self.graph_type_combo = ChevronComboBox()
        self.graph_type_combo.setObjectName("dataControl")
        self.graph_type_combo.addItems(["Histogram", "Scatter Plot", "Box Plot"])
        self.graph_type_combo.currentTextChanged.connect(self._update_graph_controls)
        controls.addWidget(self.graph_type_combo)

        x_label = QLabel("X")
        x_label.setObjectName("smallText")
        controls.addWidget(x_label)

        self.x_column_combo = ChevronComboBox()
        self.x_column_combo.setObjectName("dataControl")
        self.x_column_combo.currentTextChanged.connect(self._update_concept_detail)
        controls.addWidget(self.x_column_combo, 1)

        self.y_label = QLabel("Y")
        self.y_label.setObjectName("smallText")
        controls.addWidget(self.y_label)

        self.y_column_combo = ChevronComboBox()
        self.y_column_combo.setObjectName("dataControl")
        self.y_column_combo.currentTextChanged.connect(self._update_concept_detail)
        controls.addWidget(self.y_column_combo, 1)

        self.draw_button = QPushButton("그래프 그리기")
        self.draw_button.setObjectName("dataButton")
        self.draw_button.clicked.connect(self._draw_graph)
        controls.addWidget(self.draw_button)

        layout.addLayout(controls)

        self.figure = Figure(facecolor=COLOR_OFF_WHITE)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(300)

        # installEventFilter(): Canvas에서 발생하는 Wheel Event를 Page Scroll로 전달하기 위해 등록
        self.canvas.installEventFilter(self)

        # 그래프를 그리기 전에는 빈 Canvas를 표시하지 않음
        self.canvas.hide()

        layout.addWidget(self.canvas)

        self._update_graph_controls()

    # Concept Button과 Detail 영역을 구성
    def _create_concept_section(self, layout: QVBoxLayout):
        section_title = QLabel("Concept")
        section_title.setObjectName("sectionTitle")
        layout.addWidget(section_title)

        button_layout = QGridLayout()
        button_layout.setSpacing(SPACE_SM)

        concepts = [
            "Sample", "Feature", "Target", "Class",
            "X / y", "Train / Test", "fit()", "predict()",
            "Histogram", "Scatter Plot", "Box Plot", "1D Array",
            "2D Array", "shape", "indexing",
        ]

        # QButtonGroup: Concept Button 중 하나만 선택된 상태로 유지
        self.concept_button_group = QButtonGroup(self)
        self.concept_button_group.setExclusive(True)

        for index, concept in enumerate(concepts):
            button = QPushButton(concept)
            button.setObjectName("conceptButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, name=concept: self._show_concept(name))

            self.concept_button_group.addButton(button)
            button_layout.addWidget(button, index // 4, index % 4)

        layout.addLayout(button_layout)

        detail_layout = QGridLayout()
        detail_layout.setSpacing(SPACE_MD)

        concept_detail_title = QLabel("Concept Detail")
        concept_detail_title.setObjectName("sectionTitle")
        detail_layout.addWidget(concept_detail_title, 0, 0)

        code_title = QLabel("Python Code")
        code_title.setObjectName("sectionTitle")
        detail_layout.addWidget(code_title, 0, 1)

        self.concept_description_label = QLabel("확인할 개념을 선택하세요.")
        self.concept_description_label.setWordWrap(True)
        self.concept_description_label.setObjectName("bodyText")
        self.concept_description_label.setMinimumHeight(150)
        detail_layout.addWidget(self.concept_description_label, 1, 0)

        self.code_view = QPlainTextEdit()
        self.code_view.setObjectName("codeView")
        self.code_view.setReadOnly(True)
        self.code_view.setMinimumHeight(150)
        self.code_view.setPlainText("# 개념을 선택하면 관련 Python Code가 표시됩니다.")
        detail_layout.addWidget(self.code_view, 1, 1)

        detail_layout.setColumnStretch(0, 1)
        detail_layout.setColumnStretch(1, 1)
        layout.addLayout(detail_layout)

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
            self.status_label.setText("불러오기 실패: CSV 파일에 데이터가 없습니다.")
            return
        except (pd.errors.ParserError, UnicodeDecodeError):
            self.status_label.setText("불러오기 실패: CSV 파일 형식을 확인하세요.")
            return
        except OSError:
            self.status_label.setText("불러오기 실패: CSV 파일을 읽을 수 없습니다.")
            return

        self.dataset_name = Path(file_path).name
        self.class_names = None
        self.task_type = "unknown"
        self.dataframe = dataframe
        self._show_load_result()

    # 불러온 Dataset의 기본 정보와 관련 UI를 갱신
    def _show_load_result(self):
        summary = get_data_summary(self.dataframe)

        self.status_label.setText(f"Load 완료: {summary['sample_count']} Samples / {summary['column_count']} Columns")
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
                self.target_guide_label.setText("이 Dataset의 학습 목적에 맞게 Target이 고정되어 있습니다.")

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

    # 분류 Target의 Class 값을 화면에 표시할 문자열로 변환
    def _get_class_text(self, target: str) -> str:
        if self.class_names:
            return "\n".join(f"{index} → {name}" for index, name in enumerate(self.class_names))

        class_values = self.dataframe[target].dropna().unique().tolist()
        return "\n".join(map(str, class_values))

    # 선택한 Target을 기준으로 Feature 후보와 Target 정보를 표시
    def _update_target_info(self, _index: int = -1):
        target = self._get_target_column()
        feature_columns = get_feature_columns(self.dataframe, target)
        identifier_columns = get_identifier_columns(self.dataframe)
        feature_text = ", ".join(map(str, feature_columns)) or "-"

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
                    f"Class\n{class_text}"
                )
            else:
                task_type = TASK_TYPE_LABELS.get(self.task_type, TASK_TYPE_LABELS["unknown"])
                target_text = (
                    f"문제 유형    {task_type}\n"
                    f"Target    {target}\n"
                    f"Target 고유값 수    {unique_count}"
                )

            self.target_info_label.setText(target_text)

        self._update_concept_detail()

    # 숫자형 Column을 그래프 Column 선택 목록에 표시
    def _update_graph_columns(self):
        numeric_columns = get_numeric_columns(self.dataframe)

        self.x_column_combo.clear()
        self.y_column_combo.clear()
        self.x_column_combo.addItems(numeric_columns)
        self.y_column_combo.addItems(numeric_columns)

    # 선택한 그래프에 필요한 Column UI를 표시
    def _update_graph_controls(self, _graph_type: str = ""):
        is_scatter = self.graph_type_combo.currentText() == "Scatter Plot"
        self.y_label.setVisible(is_scatter)
        self.y_column_combo.setVisible(is_scatter)

    # 선택한 그래프 종류와 Column으로 그래프를 그림
    def _draw_graph(self):
        if self.dataframe.empty:
            return

        x_column = self.x_column_combo.currentText()

        if not x_column:
            return

        graph_type = self.graph_type_combo.currentText()

        if graph_type == "Scatter Plot":
            y_column = self.y_column_combo.currentText()

            if not y_column:
                return

            plot_data = self.dataframe[[x_column, y_column]].dropna()
            plot_data = plot_data[np.isfinite(plot_data).all(axis=1)]
        else:
            plot_data = self.dataframe[x_column].dropna()
            plot_data = plot_data[np.isfinite(plot_data)]

        if plot_data.empty:
            self.figure.clear()
            self.canvas.hide()
            self.status_label.setText("그래프 생성 실패: 유효한 숫자 데이터가 없습니다.")
            return

        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.set_facecolor(COLOR_OFF_WHITE)

        if graph_type == "Histogram":
            axes.hist(plot_data, color=COLOR_SECONDARY)
            axes.set_xlabel(x_column)
            axes.set_ylabel("Frequency")

        elif graph_type == "Scatter Plot":
            axes.scatter(plot_data[x_column], plot_data[y_column], color=COLOR_SECONDARY)
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
        self.status_label.setText("그래프 생성 완료")

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

        for row_index in range(len(preview)):
            for column_index in range(len(preview.columns)):
                value = preview.iloc[row_index, column_index]
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.data_table.setItem(row_index, column_index, item)

        total_columns = len(self.dataframe.columns)
        self.preview_note_label.setText(
            f"Preview: {len(preview)} Samples(Row) / {len(preview.columns)} of {total_columns} Columns"
        )

    # 선택한 머신러닝 개념의 설명과 Python Code를 표시
    def _show_concept(self, concept: str):
        self.selected_concept = concept
        description = CONCEPT_DESCRIPTIONS[concept]
        code = ""

        if self.dataframe.empty:
            self.concept_description_label.setText(description)
            self.code_view.clear()
            return

        target = self._get_target_column()
        feature_columns = get_feature_columns(self.dataframe, target)
        x_column = self.x_column_combo.currentText()
        y_column = self.y_column_combo.currentText()
        feature_array = self.dataframe[feature_columns].to_numpy()
        target_array = self.dataframe[target].to_numpy() if target else None

        if concept == "Sample":
            description += f"\n\n현재 Dataset은 {len(self.dataframe)}개의 Sample을 가진다."
            code = "df.head()"

        elif concept == "Feature":
            description += "\n\n현재 Feature 후보\n" + "\n".join(feature_columns)
            identifier_columns = get_identifier_columns(self.dataframe)

            if identifier_columns:
                description += (
                    "\n\n식별자 Column 제외\n"
                    + "\n".join(identifier_columns)
                    + "\n\n식별자는 대상을 구분하기 위한 값이므로 이 Lab의 기본 Feature 후보에서 제외한다."
                )

            code = f"X = df[{feature_columns!r}]"

        elif concept == "Target":
            if target:
                description += f"\n\n현재 Target\n{target}"
                code = f"y = df[{target!r}]"
            else:
                description += "\n\n현재 Dataset은 비지도 학습이므로 Target을 사용하지 않는다."
                code = "# 비지도 학습에서는 Target y를 사용하지 않습니다."

        elif concept == "Class":
            if target is None:
                description += "\n\nTarget이 없는 비지도 학습에서는 미리 정해진 Class도 없다."
                code = "# Model이 데이터의 패턴이나 Cluster를 스스로 찾습니다."

            elif self.task_type == "classification":
                description += f"\n\n현재 Class\n{self._get_class_text(target)}"
                code = f"df[{target!r}].value_counts()"

            elif self.task_type == "regression":
                description += "\n\n현재 Dataset은 연속값을 예측하는 회귀 문제이므로 Class를 사용하지 않는다."
                code = f"df[{target!r}].describe()"

            else:
                unique_count = get_target_unique_count(self.dataframe, target)
                description += (
                    f"\n\n현재 Target 고유값 수: {unique_count}"
                    "\n고유값 개수만으로 분류와 회귀를 결정할 수 없다. 학습 목적에 따라 문제 유형을 결정한다."
                )
                code = f"df[{target!r}].nunique()"

        elif concept == "X / y":
            if target:
                code = f"X = df[{feature_columns!r}]\ny = df[{target!r}]"
            else:
                description += "\n\n비지도 학습에서는 X만 사용하고 정답 y는 사용하지 않는다."
                code = f"X = df[{feature_columns!r}]\n# y 없음"

        elif concept == "Train / Test":
            if target:
                code = "X_train, X_test, y_train, y_test = train_test_split(X, y)"
            else:
                description += "\n\n비지도 학습은 목적과 평가 방법에 따라 데이터 분할 여부가 달라진다."
                code = "# 비지도 학습에서는 평가 방법을 먼저 정한 뒤 데이터 분할 여부를 결정합니다."

        elif concept == "fit()":
            code = "model.fit(X_train, y_train)" if target else "model.fit(X)"

        elif concept == "predict()":
            if target:
                description += "\n\n지도 학습에서는 새로운 Feature에 대한 Target을 예측한다."
                code = "prediction = model.predict(X_test)"
            elif self.task_type == "clustering":
                description += "\n\n군집화에서는 fit_predict()을 지원하는 Model로 학습과 Cluster 할당을 함께 수행할 수 있다."
                code = "labels = model.fit_predict(X)"
            else:
                description += "\n\n비지도 학습에서 사용할 수 있는 예측 메서드는 Model마다 다르다."
                code = "# 선택한 Model이 지원하는 메서드를 확인합니다."

        elif concept == "Histogram":
            description += f"\n\n현재 X Column\n{x_column}"
            code = f"axes.hist(df[{x_column!r}].dropna())"

        elif concept == "Scatter Plot":
            description += f"\n\n현재 선택\nX: {x_column}\nY: {y_column}"
            code = f"axes.scatter(df[{x_column!r}], df[{y_column!r}])"

        elif concept == "Box Plot":
            description += f"\n\n현재 X Column\n{x_column}"
            code = f"axes.boxplot(df[{x_column!r}].dropna())"

        elif concept == "1D Array":
            if target_array is not None:
                description += f"\n\n현재 y Array shape\n{target_array.shape}"
                code = f"y_array = df[{target!r}].to_numpy()"

            elif feature_columns:
                first_feature = feature_columns[0]
                feature_1d = self.dataframe[first_feature].to_numpy()
                description += f"\n\n현재 {first_feature} Array shape\n{feature_1d.shape}"
                code = f"array_1d = df[{first_feature!r}].to_numpy()"

        elif concept == "2D Array":
            description += f"\n\n현재 X Array shape\n{feature_array.shape}"
            code = f"X_array = df[{feature_columns!r}].to_numpy()"

        elif concept == "shape":
            if target_array is not None:
                description += f"\n\n현재 X shape: {feature_array.shape}\n현재 y shape: {target_array.shape}"
                code = "X_array.shape\ny_array.shape"
            else:
                description += f"\n\n현재 X shape: {feature_array.shape}\n비지도 학습이므로 y는 없다."
                code = "X_array.shape"

        elif concept == "indexing":
            if feature_columns:
                first_value = feature_array[0, 0]
                description += f"\n\n2D Array에서는 [Row, Column] 위치로 값을 선택한다.\n현재 X_array[0, 0] 값: {first_value}"
                code = "value = X_array[0, 0]"
            else:
                description += "\n\n현재 Dataset에는 indexing할 Feature가 없다."
                code = "# 먼저 Feature Column이 필요합니다."

        self.concept_description_label.setText(description)
        self.code_view.setPlainText(code)

    # Target 또는 그래프 Column 변경 시 현재 Concept Detail을 다시 표시
    def _update_concept_detail(self, _value: str = ""):
        if self.selected_concept:
            self._show_concept(self.selected_concept)
