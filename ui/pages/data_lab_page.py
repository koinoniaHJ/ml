from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QPlainTextEdit, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from common.theme import PREVIEW_COLUMN_COUNT, PREVIEW_ROW_COUNT, SPACE_12, SPACE_16, SPACE_24, TABLE_HEADER_HEIGHT
from ml.data_loader import (
    get_builtin_target, get_sklearn_target, get_sklearn_target_names,
    load_builtin_dataset, load_local_csv, load_sklearn_dataset,
)
from ml.data_summary import get_data_summary, get_feature_columns, get_numeric_columns, get_target_columns, get_target_unique_count


CONCEPT_DESCRIPTIONS = {
    "Sample": "Dataset을 구성하는 하나의 데이터 단위로, 일반적으로 한 Row에 해당한다.",
    "Feature": "Model이 데이터의 패턴을 학습할 때 사용하는 입력 데이터다.",
    "Target": "지도 학습에서 Model이 예측하려는 정답 데이터다.",
    "Class": "분류 문제에서 Target이 가질 수 있는 범주다.",
    "X / y": "X는 Model에 입력하는 Feature 데이터이고, y는 지도 학습에서 사용하는 Target 데이터다.",
    "Train / Test": "Train은 Model 학습에 사용하고 Test는 학습된 Model의 성능 확인에 사용한다.",
    "fit()": "Train 데이터를 이용해 Model을 학습시키는 메서드다.",
    "predict()": "학습된 Model을 이용해 새로운 값을 예측하는 메서드다.",
    "Histogram": "하나의 숫자형 Column 값이 어떤 범위에 얼마나 분포하는지 확인하는 그래프다.",
    "Scatter Plot": "두 숫자형 Column 사이의 관계를 점으로 확인하는 그래프다.",
    "Box Plot": "숫자형 Column의 중앙값, 분포, 이상치 등을 Box 형태로 확인하는 그래프다.",
    "1D Array": "한 방향으로 값이 나열된 NumPy 배열이다.",
    "2D Array": "행과 열로 구성된 NumPy 배열이다.",
    "shape": "NumPy Array의 각 차원 크기를 튜플로 나타내는 속성이다.",
    "indexing": "위치 번호를 이용해 NumPy Array의 특정 값을 선택하는 방법이다.",
}


class DataLabPage(QWidget):
    # Data Lab의 기본 상태를 만들고 UI를 구성
    def __init__(self):
        super().__init__()

        self.dataframe = pd.DataFrame()

        # Data Lab에 직접 들어왔을 때 사용할 기본 Dataset Source
        self.data_source = "builtin"
        self.dataset_name = ""
        self.selected_concept = ""
        self.class_names = None

        self._setup_ui()

    # Data Lab 전체 화면을 Section 단위로 구성
    def _setup_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        # QScrollArea: 내용이 화면보다 길어질 때 Scroll로 확인할 수 있는 Widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        page_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACE_16, SPACE_16, SPACE_16, SPACE_16)
        layout.setSpacing(SPACE_24)

        layout.addWidget(QLabel("Data Lab — 머신러닝 기초"))

        self._create_dataset_source(layout)
        self._create_data_section(layout)
        self._create_visualization_section(layout)
        self._create_concept_section(layout)

        layout.addStretch()
        self.set_data_source("builtin")

    # Dataset Source 선택 영역을 구성
    def _create_dataset_source(self, layout: QVBoxLayout):
        layout.addWidget(QLabel("Dataset Source"))

        source_layout = QHBoxLayout()
        source_layout.setSpacing(SPACE_12)

        self.source_combo = QComboBox()
        self.source_combo.setObjectName("dataControl")
        self.source_combo.addItem("Built-in", "builtin")
        self.source_combo.addItem("scikit-learn", "sklearn")
        self.source_combo.addItem("Local CSV", "csv")
        self.source_combo.currentIndexChanged.connect(self._change_data_source)
        source_layout.addWidget(self.source_combo, 2)

        self.dataset_combo = QComboBox()
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
        data_layout.setSpacing(SPACE_24)

        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("Data Preview"))

        self.preview_guide_label = QLabel(
            "최대 10개 Sample(Row)과 5개 Column만 미리보기하며 실제 학습에는 전체 Dataset을 사용합니다."
        )
        self.preview_guide_label.setObjectName("smallText")
        self.preview_guide_label.setWordWrap(True)
        preview_header.addWidget(self.preview_guide_label, 1)

        data_layout.addLayout(preview_header, 0, 0)
        data_layout.addWidget(QLabel("Dataset Information"), 0, 1)

        # QTableWidget: 행과 열 형태의 데이터를 화면에 표시하는 Table Widget
        self.data_table = QTableWidget()
        self.data_table.setObjectName("dataPreviewTable")
        self.data_table.setMinimumHeight(320)
        self.data_table.verticalHeader().setVisible(False)

        # QAbstractItemView.EditTrigger.NoEditTriggers: Table의 Cell을 사용자가 직접 수정하지 못하게 설정
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # QHeaderView.Stretch: 현재 Table 너비에 맞춰 Column 너비를 균등하게 조절
        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # setHighlightSections(): Cell 선택 시 해당 Header가 강조되는 기능 설정
        header.setHighlightSections(False)

        # Header Text를 가운데 정렬
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        # 긴 Column 이름과 dtype을 여러 줄로 표시할 수 있도록 Header 높이를 확보
        header.setFixedHeight(TABLE_HEADER_HEIGHT)

        # ElideNone: Header 문자열을 ...으로 줄이지 않고 그대로 표시
        header.setTextElideMode(Qt.TextElideMode.ElideNone)

        self.data_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        data_layout.addWidget(self.data_table, 1, 0)

        information_frame = QFrame()
        information_layout = QVBoxLayout(information_frame)
        information_layout.setContentsMargins(0, 0, 0, 0)
        information_layout.setSpacing(SPACE_12)

        self.summary_label = QLabel("Samples    -\nColumns    -\nMissing    -")
        self.summary_label.setObjectName("smallText")
        information_layout.addWidget(self.summary_label)

        information_layout.addWidget(QLabel("Target"))

        self.target_combo = QComboBox()
        self.target_combo.setObjectName("dataControl")

        # currentIndexChanged: QComboBox의 선택 위치가 바뀌었을 때 발생하는 기본 Signal
        self.target_combo.currentIndexChanged.connect(self._update_target_info)
        information_layout.addWidget(self.target_combo)

        self.target_guide_label = QLabel()
        self.target_guide_label.setObjectName("smallText")
        self.target_guide_label.setWordWrap(True)
        information_layout.addWidget(self.target_guide_label)

        information_layout.addWidget(QLabel("Feature 후보"))

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
        layout.addWidget(QLabel("Visualization"))

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_12)

        graph_label = QLabel("Graph")
        graph_label.setObjectName("smallText")
        controls.addWidget(graph_label)

        self.graph_type_combo = QComboBox()
        self.graph_type_combo.setObjectName("dataControl")
        self.graph_type_combo.addItems(["Histogram", "Scatter Plot", "Box Plot"])
        self.graph_type_combo.currentTextChanged.connect(self._update_graph_controls)
        controls.addWidget(self.graph_type_combo)

        x_label = QLabel("X")
        x_label.setObjectName("smallText")
        controls.addWidget(x_label)

        self.x_column_combo = QComboBox()
        self.x_column_combo.setObjectName("dataControl")
        self.x_column_combo.currentTextChanged.connect(self._update_concept_detail)
        controls.addWidget(self.x_column_combo, 1)

        self.y_label = QLabel("Y")
        self.y_label.setObjectName("smallText")
        controls.addWidget(self.y_label)

        self.y_column_combo = QComboBox()
        self.y_column_combo.setObjectName("dataControl")
        self.y_column_combo.currentTextChanged.connect(self._update_concept_detail)
        controls.addWidget(self.y_column_combo, 1)

        self.draw_button = QPushButton("그래프 그리기")
        self.draw_button.setObjectName("dataButton")
        self.draw_button.clicked.connect(self._draw_graph)
        controls.addWidget(self.draw_button)

        layout.addLayout(controls)

        # Figure: Matplotlib에서 전체 그래프 영역을 나타내는 객체
        self.figure = Figure()

        # FigureCanvasQTAgg: Matplotlib Figure를 PySide6 Widget 안에 표시하는 Canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(300)
        layout.addWidget(self.canvas)

        self._update_graph_controls()

    # Concept Button과 Concept Detail, Python Code 영역을 구성
    def _create_concept_section(self, layout: QVBoxLayout):
        layout.addWidget(QLabel("Concept"))

        button_layout = QGridLayout()
        button_layout.setSpacing(SPACE_12)

        concepts = [
            "Sample", "Feature", "Target", "Class",
            "X / y", "Train / Test", "fit()", "predict()",
            "Histogram", "Scatter Plot", "Box Plot", "1D Array",
            "2D Array", "shape", "indexing",
        ]

        for index, concept in enumerate(concepts):
            button = QPushButton(concept)
            button.setObjectName("conceptButton")
            button.clicked.connect(lambda checked=False, name=concept: self._show_concept(name))

            row = index // 4
            column = index % 4
            button_layout.addWidget(button, row, column)

        layout.addLayout(button_layout)

        detail_layout = QGridLayout()
        detail_layout.setSpacing(SPACE_24)

        detail_layout.addWidget(QLabel("Concept Detail"), 0, 0)
        detail_layout.addWidget(QLabel("Python Code"), 0, 1)

        self.concept_description_label = QLabel("확인할 개념을 선택하세요.")

        # setWordWrap(): 긴 문장을 Widget 너비에 맞게 자동 줄바꿈
        self.concept_description_label.setWordWrap(True)
        self.concept_description_label.setObjectName("smallText")
        self.concept_description_label.setMinimumHeight(150)
        detail_layout.addWidget(self.concept_description_label, 1, 0)

        # QPlainTextEdit: 여러 줄의 Text를 표시하거나 입력하는 Widget
        self.code_view = QPlainTextEdit()
        self.code_view.setObjectName("codeView")
        self.code_view.setReadOnly(True)
        self.code_view.setMinimumHeight(150)
        self.code_view.setPlainText("# 개념을 선택하면 관련 Python Code가 표시됩니다.")
        detail_layout.addWidget(self.code_view, 1, 1)

        detail_layout.setColumnStretch(0, 1)
        detail_layout.setColumnStretch(1, 1)

        layout.addLayout(detail_layout)

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
            # addItem(): 첫 번째 값은 화면에 표시하고 두 번째 값은 실제 데이터로 저장
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
        self.class_names = None

        if self.data_source == "builtin":
            self.dataset_name = self.dataset_combo.currentData()
            self.dataframe = load_builtin_dataset(self.dataset_name)

        elif self.data_source == "sklearn":
            self.dataset_name = self.dataset_combo.currentData()
            self.dataframe = load_sklearn_dataset(self.dataset_name)
            self.class_names = get_sklearn_target_names(self.dataset_name)

        else:
            self._select_csv()
            return

        self._show_load_result()

    # Local CSV 파일을 선택해 DataFrame으로 저장
    def _select_csv(self):
        # QFileDialog: 사용자가 파일을 선택할 수 있는 Dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "CSV 파일 선택", "", "CSV Files (*.csv)")

        if not file_path:
            return

        self.dataset_name = Path(file_path).name
        self.class_names = None
        self.dataframe = load_local_csv(Path(file_path))
        self._show_load_result()

    # 불러온 Dataset의 기본 정보와 관련 UI를 갱신
    def _show_load_result(self):
        summary = get_data_summary(self.dataframe)

        self.status_label.setText(
            f"Load 완료: {summary['sample_count']} Samples / {summary['column_count']} Columns"
        )

        self.summary_label.setText(
            f"Samples    {summary['sample_count']}\n"
            f"Columns    {summary['column_count']}\n"
            f"Missing    {summary['missing_count']}"
        )

        self._configure_target()
        self._update_graph_columns()
        self._show_dataframe()

    # Dataset Source에 따라 Target을 고정하거나 사용자가 선택할 수 있게 구성
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

            # setEnabled(): Widget을 사용자가 조작할 수 있는지 설정
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

    # 선택한 Target을 기준으로 Feature 후보와 Target 정보를 표시
    def _update_target_info(self, _index: int = -1):
        if self.dataframe.empty:
            return

        target = self._get_target_column()
        feature_columns = get_feature_columns(self.dataframe, target)

        # join(): 문자열 목록을 지정한 구분자로 하나의 문자열로 연결
        self.feature_label.setText(", ".join(feature_columns))

        if target is None:
            self.target_info_label.setText("Target 없음 · 비지도 학습")

        else:
            unique_count = get_target_unique_count(self.dataframe, target)
            target_text = f"Target 고유값 수    {unique_count}"

            if self.class_names and target == "target":
                class_text = "\n".join(
                    f"{index} → {class_name}" for index, class_name in enumerate(self.class_names)
                )
                target_text += f"\n\nClass\n{class_text}"

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

        # clear(): Figure에 그려진 기존 그래프를 제거
        self.figure.clear()

        # add_subplot(): Figure 안에 실제 그래프를 그릴 Axes를 생성
        axes = self.figure.add_subplot(111)

        graph_type = self.graph_type_combo.currentText()

        if graph_type == "Histogram":
            # dropna(): 결측값을 제외한 데이터를 반환
            # hist(): 데이터 값의 분포를 Histogram으로 표시
            axes.hist(self.dataframe[x_column].dropna())
            axes.set_xlabel(x_column)
            axes.set_ylabel("Frequency")

        elif graph_type == "Scatter Plot":
            y_column = self.y_column_combo.currentText()

            if not y_column:
                return

            plot_data = self.dataframe[[x_column, y_column]].dropna()

            # scatter(): 두 변수의 관계를 점으로 표시
            axes.scatter(plot_data[x_column], plot_data[y_column])
            axes.set_xlabel(x_column)
            axes.set_ylabel(y_column)

        elif graph_type == "Box Plot":
            # boxplot(): 데이터의 분포와 중앙값 등을 Box 형태로 표시
            axes.boxplot(self.dataframe[x_column].dropna())
            axes.set_ylabel(x_column)

        axes.set_title(graph_type)

        # draw(): 변경된 Figure를 Canvas에 다시 표시
        self.canvas.draw()

    # DataFrame 앞부분을 dtype과 함께 Table에 표시
    def _show_dataframe(self):
        preview = self.dataframe.iloc[:PREVIEW_ROW_COUNT, :PREVIEW_COLUMN_COUNT]

        self.data_table.clear()
        self.data_table.setRowCount(len(preview))
        self.data_table.setColumnCount(len(preview.columns))

        # 긴 Column 이름은 공백과 _를 기준으로 줄바꿈하고 마지막에 dtype을 표시
        headers = []

        for column in preview.columns:
            column_name = str(column).replace("_", "\n").replace(" ", "\n")
            headers.append(f"{column_name}\n({self.dataframe[column].dtype})")

        self.data_table.setHorizontalHeaderLabels(headers)

        for row_index in range(len(preview)):
            for column_index in range(len(preview.columns)):
                # iloc: Row와 Column의 위치 번호로 DataFrame 값을 가져오는 방법
                value = preview.iloc[row_index, column_index]

                # QTableWidgetItem: QTableWidget의 한 Cell에 들어가는 항목
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
            code = f'X = df.drop(columns=["{target}"])' if target else "X = df.copy()"

        elif concept == "Target":
            if target:
                description += f"\n\n현재 Target\n{target}"
                code = f'y = df["{target}"]'
            else:
                description += "\n\n현재 Dataset은 비지도 학습이므로 Target을 사용하지 않는다."
                code = "# 비지도 학습에서는 Target y를 사용하지 않습니다."

        elif concept == "Class":
            if target is None:
                description += "\n\nTarget이 없는 비지도 학습에서는 미리 정해진 Class도 없다."
                code = "# Model이 데이터의 패턴이나 Cluster를 스스로 찾습니다."

            else:
                unique_count = get_target_unique_count(self.dataframe, target)

                if self.class_names and target == "target":
                    class_text = "\n".join(
                        f"{index} → {class_name}" for index, class_name in enumerate(self.class_names)
                    )
                    description += f"\n\n현재 Class\n{class_text}"

                elif unique_count <= 10:
                    class_values = self.dataframe[target].dropna().unique().tolist()
                    description += f"\n\n현재 Class 값\n{', '.join(map(str, class_values))}"

                else:
                    description += "\n\n현재 Target은 고유값이 많아 연속형 Target으로 볼 수 있다."

                code = f'df["{target}"].value_counts()'

        elif concept == "X / y":
            if target:
                code = f'X = df.drop(columns=["{target}"])\ny = df["{target}"]'
            else:
                description += "\n\n비지도 학습에서는 X만 사용하고 정답 y는 사용하지 않는다."
                code = "X = df.copy()\n# y 없음"

        elif concept == "Train / Test":
            code = (
                "X_train, X_test, y_train, y_test = train_test_split(X, y)"
                if target
                else "X_train, X_test = train_test_split(X)"
            )

        elif concept == "fit()":
            code = "model.fit(X_train, y_train)" if target else "model.fit(X)"

        elif concept == "predict()":
            code = "prediction = model.predict(X_test)" if target else "cluster = model.fit_predict(X)"

        elif concept == "Histogram":
            description += f"\n\n현재 X Column\n{x_column}"
            code = f'axes.hist(df["{x_column}"].dropna())'

        elif concept == "Scatter Plot":
            description += f"\n\n현재 선택\nX: {x_column}\nY: {y_column}"

            code = (
                "axes.scatter(\n"
                f'    df["{x_column}"],\n'
                f'    df["{y_column}"],\n'
                ")"
            )

        elif concept == "Box Plot":
            description += f"\n\n현재 X Column\n{x_column}"
            code = f'axes.boxplot(df["{x_column}"].dropna())'

        elif concept == "1D Array":
            if target_array is not None:
                description += f"\n\n현재 y Array shape\n{target_array.shape}"
                code = f'y_array = df["{target}"].to_numpy()'

            elif feature_columns:
                first_feature = feature_columns[0]
                feature_1d = self.dataframe[first_feature].to_numpy()

                description += f"\n\n현재 {first_feature} Array shape\n{feature_1d.shape}"
                code = f'array_1d = df["{first_feature}"].to_numpy()'

        elif concept == "2D Array":
            description += f"\n\n현재 X Array shape\n{feature_array.shape}"
            code = f'X_array = df.drop(columns=["{target}"]).to_numpy()' if target else "X_array = df.to_numpy()"

        elif concept == "shape":
            if target_array is not None:
                description += f"\n\n현재 X shape: {feature_array.shape}\n현재 y shape: {target_array.shape}"
                code = "X_array.shape\ny_array.shape"
            else:
                description += f"\n\n현재 X shape: {feature_array.shape}\n비지도 학습이므로 y는 없다."
                code = "X_array.shape"

        elif concept == "indexing":
            first_value = feature_array[0, 0]
            description += (
                "\n\n2D Array에서는 [Row, Column] 위치로 값을 선택한다.\n"
                f"현재 X_array[0, 0] 값: {first_value}"
            )
            code = "value = X_array[0, 0]"

        self.concept_description_label.setText(description)
        self.code_view.setPlainText(code)

    # Target 또는 그래프 Column 변경 시 현재 Concept Detail을 다시 표시
    def _update_concept_detail(self, _value: str = ""):
        if not self.selected_concept:
            return

        self._show_concept(self.selected_concept)