"""Target 없이 Feature의 군집 구조를 확인하는 Unsupervised Learning 화면을 구성한다."""

from html import escape

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from sklearn.preprocessing import StandardScaler

from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY,
    SPACE_MD, SPACE_SM, SPACE_XS,
)
from ml.clustering import (
    KMeansResult, create_clustered_frame, fit_kmeans, predict_cluster,
    validate_clustering_data,
)
from ml.data_loader import load_builtin_dataset
from ui.widgets import ChevronComboBox, ChevronDoubleSpinBox, ChevronSpinBox


CONCEPT_DESCRIPTIONS = {
    "Unsupervised Learning": "정답인 Target 없이 Feature만 이용해 데이터의 구조나 패턴을 찾는 학습 방식이다.",
    "Clustering": "비슷한 Sample을 하나의 그룹으로 묶는 비지도학습 작업이다.",
    "K-Means": "Sample 사이의 거리를 이용해 데이터를 K개의 Cluster로 나누는 군집화 알고리즘이다.",
    "Cluster": "비슷한 Sample들이 모인 하나의 그룹이다. Cluster 번호는 그룹을 구분하는 Label일 뿐 순서나 등급을 뜻하지 않는다.",
    "Target": "지도학습에서 예측할 정답이다. K-Means는 Target과 Train/Test 분리 없이 선택한 Feature 전체에서 군집 구조를 찾을 수 있다.",
    "Distance": "K-Means가 Sample과 Centroid가 얼마나 가까운지 판단할 때 사용하는 값이다. Feature의 값 범위에 영향을 받는다.",
    "Scaling": "Feature 값의 크기를 일정한 기준으로 변환하는 과정이다. 거리 기반 알고리즘에서 큰 단위의 Feature가 거리를 지나치게 좌우하는 것을 줄인다.",
    "StandardScaler": "각 Feature의 평균과 표준편차를 이용해 평균이 0, 표준편차가 1 정도가 되도록 변환한다.",
    "K / n_clusters": "데이터를 몇 개의 Cluster로 나눌지 정하는 Hyperparameter다. 너무 작거나 큰 K는 군집 구조를 지나치게 합치거나 나눌 수 있다.",
    "fit_predict()": "데이터로 K-Means를 학습하고 각 Sample이 속한 Cluster 번호를 반환한다.",
    "Centroid": "각 Cluster의 중심이다. K-Means는 가까운 Centroid에 Sample을 배정하고 Centroid를 다시 계산하는 과정을 반복한다.",
    "cluster_centers_": "학습된 각 Cluster의 Centroid를 반환한다. Scaling했다면 이 값도 Scaling된 단위다.",
    "inverse_transform()": "Scaling된 값을 원래 Feature 단위로 되돌린다. Centroid를 공부 시간이나 출석률 같은 원래 단위로 해석할 때 사용한다.",
    "n_init": "서로 다른 초기 Centroid로 K-Means를 반복 실행하는 횟수다. Sample과 Centroid 사이 제곱 거리의 합이 가장 작은 결과를 사용한다.",
    "random_state": "초기 Centroid 선택에 사용하는 난수 기준을 고정해 같은 조건에서 결과를 다시 확인할 수 있게 한다.",
    "predict()": "새 Sample을 학습된 Centroid 중 가장 가까운 Cluster에 배정한다. 반환값은 정답 Class가 아닌 Cluster 번호다.",
}

CONCEPT_LABELS = {
    "Unsupervised Learning": "Unsupervised Learning\n(비지도학습)",
    "Clustering": "Clustering\n(군집화)",
    "K-Means": "K-Means\n(K-평균)",
    "Cluster": "Cluster\n(군집)",
    "Target": "Target\n(정답)",
    "Distance": "Distance\n(거리)",
    "Scaling": "Scaling\n(크기 변환)",
    "StandardScaler": "StandardScaler\n(표준화)",
    "K / n_clusters": "K / n_clusters\n(군집 수)",
    "fit_predict()": "fit_predict()\n(학습과 배정)",
    "Centroid": "Centroid\n(군집 중심)",
    "cluster_centers_": "cluster_centers_\n(학습된 중심)",
    "inverse_transform()": "inverse_transform()\n(원래 단위 복원)",
    "n_init": "n_init\n(초기화 횟수)",
    "random_state": "random_state\n(난수 기준)",
    "predict()": "predict()\n(새 Sample 배정)",
}

STEP_NAMES = ["1. Data / Scaling", "2. K-Means", "3. Result", "4. New Sample"]


class UnsupervisedPage(QWidget):
    """K-Means의 데이터 준비, 학습, 결과 해석과 새 Sample 예측을 구성한다."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = ""
        self.dataframe = pd.DataFrame()
        self.target_column: str | None = None
        self.task_type = "unknown"
        self.feature_checkboxes: dict[str, QCheckBox] = {}
        self.new_value_controls: dict[str, ChevronDoubleSpinBox] = {}
        self.prepared_features = pd.DataFrame()
        self.prepared_values = pd.DataFrame()
        self.result: KMeansResult | None = None

        self._setup_ui()
        self._load_default_dataset()

    def _setup_ui(self) -> None:
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page_layout.addWidget(self.scroll_area)

        content = QWidget()
        self.scroll_area.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_MD)

        header = QHBoxLayout()
        header.setSpacing(SPACE_SM)
        title = QLabel("Unsupervised Learning")
        title.setObjectName("pageTitle")
        title.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        header.addWidget(title)
        description = QLabel(
            "Target 없이 Feature를 이용해 데이터의 구조를 찾는다.<br>"
            "K-Means로 비슷한 Sample을 Cluster로 묶고 Centroid를 확인한다."
        )
        description.setObjectName("bodyText")
        description.setWordWrap(True)
        header.addWidget(description, 1)
        layout.addLayout(header)

        self._create_dataset_section(layout)
        self._create_concept_section(layout)
        self._create_workflow_section(layout)
        layout.addStretch()

    def _create_dataset_section(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(SPACE_SM)
        row.addWidget(self._create_section_badge("Current Dataset"))
        self.current_dataset_label = QLabel()
        self.current_dataset_label.setObjectName("bodyText")
        self.current_dataset_label.setWordWrap(True)
        row.addWidget(self.current_dataset_label, 1)
        layout.addLayout(row)
        self.dataset_info_label = QLabel()
        self.dataset_info_label.setObjectName("preprocessingInfoCard")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)
        self.status_label = QLabel()
        self.status_label.setObjectName("smallText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _create_concept_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Concept"), alignment=Qt.AlignmentFlag.AlignLeft)
        grid = QGridLayout()
        grid.setSpacing(SPACE_SM)
        self.concept_group = QButtonGroup(self)
        self.concept_group.setExclusive(True)
        self.concept_buttons: dict[str, QPushButton] = {}
        for index, concept in enumerate(CONCEPT_DESCRIPTIONS):
            button = QPushButton(CONCEPT_LABELS[concept])
            button.setObjectName("conceptButton")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            button.clicked.connect(lambda checked=False, name=concept: self._show_concept(name))
            self.concept_group.addButton(button)
            self.concept_buttons[concept] = button
            grid.addWidget(button, index // 3, index % 3)
        for column in range(3):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)

        card = QFrame()
        card.setObjectName("conceptDetailCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        self.concept_detail_label = QLabel()
        self.concept_detail_label.setObjectName("bodyText")
        self.concept_detail_label.setWordWrap(True)
        card_layout.addWidget(self.concept_detail_label)
        layout.addWidget(card)
        self.concept_buttons["Unsupervised Learning"].setChecked(True)
        self._show_concept("Unsupervised Learning")

    def _create_workflow_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("K-Means Lab"), alignment=Qt.AlignmentFlag.AlignLeft)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE_SM)
        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.step_buttons: list[QPushButton] = []
        self.step_pages = [
            self._create_data_step(), self._create_model_step(),
            self._create_result_step(), self._create_prediction_step(),
        ]
        for index, name in enumerate(STEP_NAMES):
            button = QPushButton(name)
            button.setObjectName("stepButton")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            button.setEnabled(index == 0)
            button.clicked.connect(lambda checked=False, value=index: self._show_step(value))
            self.step_group.addButton(button)
            self.step_buttons.append(button)
            buttons.addWidget(button, 1)
        layout.addLayout(buttons)
        for page in self.step_pages:
            layout.addWidget(page)
        self.step_buttons[0].setChecked(True)
        self._show_step(0)

    def _create_data_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Feature와 Scaling",
            "K-Means는 Target 없이 선택한 Feature 전체를 사용한다. 거리 계산이 값의 크기에 좌우되지 않도록 StandardScaler를 적용할 수 있다.",
        )
        feature_card = QFrame()
        feature_card.setObjectName("preprocessingCard")
        feature_layout = QVBoxLayout(feature_card)
        feature_layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        title = QLabel("Feature 선택")
        title.setObjectName("sectionTitle")
        feature_layout.addWidget(title)
        self.feature_guide_label = QLabel()
        self.feature_guide_label.setObjectName("smallText")
        self.feature_guide_label.setWordWrap(True)
        feature_layout.addWidget(self.feature_guide_label)
        self.feature_grid = QGridLayout()
        for column in range(3):
            self.feature_grid.setColumnStretch(column, 1)
        feature_layout.addLayout(self.feature_grid)
        layout.addWidget(feature_card)

        controls = QHBoxLayout()
        self.scaling_checkbox = QCheckBox("StandardScaler 적용")
        self.scaling_checkbox.setObjectName("dataCheckBox")
        self.scaling_checkbox.setChecked(True)
        self.scaling_checkbox.toggled.connect(self._preparation_changed)
        controls.addWidget(self.scaling_checkbox)
        controls.addStretch()
        self.prepare_button = QPushButton("데이터 준비")
        self.prepare_button.setObjectName("dataButton")
        self.prepare_button.clicked.connect(self._prepare_data)
        controls.addWidget(self.prepare_button)
        layout.addLayout(controls)

        tables = QGridLayout()
        original_card, self.original_table = self._create_table_card("변환 전")
        scaled_card, self.scaled_table = self._create_table_card("변환 후")
        self.original_table.setMinimumHeight(300)
        self.scaled_table.setMinimumHeight(300)
        tables.addWidget(original_card, 0, 0)
        tables.addWidget(scaled_card, 0, 1)
        tables.setColumnStretch(0, 1)
        tables.setColumnStretch(1, 1)
        layout.addLayout(tables)
        code_card, self.preparation_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_model_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "K-Means 학습",
            "K는 Cluster의 개수이며 학습 전에 정하는 Hyperparameter다. fit_predict()는 모델을 학습하고 각 Sample의 Cluster 번호를 반환한다.",
        )
        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)
        controls.addWidget(self._create_control_label("K / n_clusters"))
        self.cluster_spin = ChevronSpinBox()
        self.cluster_spin.setObjectName("dataControl")
        self.cluster_spin.setRange(1, 3)
        self.cluster_spin.setValue(3)
        self.cluster_spin.valueChanged.connect(self._model_setting_changed)
        controls.addWidget(self.cluster_spin)
        controls.addWidget(self._create_control_label("n_init"))
        self.n_init_spin = ChevronSpinBox()
        self.n_init_spin.setObjectName("dataControl")
        self.n_init_spin.setRange(1, 100)
        self.n_init_spin.setValue(10)
        self.n_init_spin.valueChanged.connect(self._model_setting_changed)
        controls.addWidget(self.n_init_spin)
        controls.addWidget(self._create_control_label("random_state"))
        self.random_state_spin = ChevronSpinBox()
        self.random_state_spin.setObjectName("dataControl")
        self.random_state_spin.setRange(0, 999999)
        self.random_state_spin.setValue(42)
        self.random_state_spin.valueChanged.connect(self._model_setting_changed)
        controls.addWidget(self.random_state_spin)
        controls.addStretch()
        self.fit_button = QPushButton("K-Means 학습")
        self.fit_button.setObjectName("dataButton")
        self.fit_button.clicked.connect(self._fit_model)
        controls.addWidget(self.fit_button)
        layout.addLayout(controls)
        self.model_result_label = self._create_result_card()
        self.model_result_label.setText("K-Means를 학습하기 전입니다.")
        layout.addWidget(self.model_result_label)
        count_card, self.cluster_count_table = self._create_table_card("Cluster별 Sample 수")
        self.cluster_count_table.setMinimumHeight(200)
        layout.addWidget(count_card)
        code_card, self.model_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_result_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Cluster와 Centroid",
            "Cluster 번호는 그룹을 구분하는 Label이며 순서나 등급을 뜻하지 않는다. Centroid는 각 Cluster의 중심을 나타낸다.",
        )
        warning = self._create_result_card()
        warning.setText(
            "실행 조건이나 초기 Centroid가 달라지면 같은 형태의 군집도 Cluster 번호가 바뀔 수 있다.<br>"
            "Cluster의 의미는 군집 결과와 원래 데이터를 보고 사람이 해석한다."
        )
        layout.addWidget(warning)
        result_card, self.cluster_result_table = self._create_table_card("Sample별 Cluster")
        self.cluster_result_table.setMinimumHeight(330)
        layout.addWidget(result_card)
        centroid_tables = QGridLayout()
        original_card, self.original_centroid_table = self._create_table_card("원래 단위 Centroid")
        scaled_card, self.scaled_centroid_table = self._create_table_card("Scaling 단위 Centroid")
        self.original_centroid_table.setMinimumHeight(220)
        self.scaled_centroid_table.setMinimumHeight(220)
        centroid_tables.addWidget(original_card, 0, 0)
        centroid_tables.addWidget(scaled_card, 0, 1)
        centroid_tables.setColumnStretch(0, 1)
        centroid_tables.setColumnStretch(1, 1)
        layout.addLayout(centroid_tables)

        graph_controls = QHBoxLayout()
        graph_controls.addWidget(self._create_control_label("X"))
        self.x_combo = ChevronComboBox()
        self.x_combo.setObjectName("dataControl")
        graph_controls.addWidget(self.x_combo, 1)
        graph_controls.addWidget(self._create_control_label("Y"))
        self.y_combo = ChevronComboBox()
        self.y_combo.setObjectName("dataControl")
        graph_controls.addWidget(self.y_combo, 1)
        self.graph_button = QPushButton("Cluster 그래프")
        self.graph_button.setObjectName("dataButton")
        self.graph_button.clicked.connect(self._draw_cluster_graph)
        graph_controls.addWidget(self.graph_button)
        layout.addLayout(graph_controls)
        self.graph_guide_label = QLabel(
            "두 Feature를 선택하면 각 Sample의 Cluster와 원래 단위의 Centroid를 Scatter Plot으로 표시합니다."
        )
        self.graph_guide_label.setObjectName("smallText")
        self.graph_guide_label.setWordWrap(True)
        layout.addWidget(self.graph_guide_label)
        self.figure = Figure(facecolor=COLOR_OFF_WHITE)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(420)
        self.canvas.installEventFilter(self)
        self.canvas.hide()
        layout.addWidget(self.canvas)
        code_card, self.result_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_prediction_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "새로운 Sample의 Cluster 예측",
            "새 Sample에는 학습 때 사용한 Feature와 같은 Scaling 기준을 적용한다. predict()는 가장 가까운 Centroid의 Cluster 번호를 반환한다.",
        )
        form_card = QFrame()
        form_card.setObjectName("preprocessingCard")
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        title = QLabel("새 Sample")
        title.setObjectName("sectionTitle")
        form_layout.addWidget(title)
        self.new_value_grid = QGridLayout()
        self.new_value_grid.setSpacing(SPACE_SM)
        form_layout.addLayout(self.new_value_grid)
        layout.addWidget(form_card)
        row = QHBoxLayout()
        row.addStretch()
        self.predict_button = QPushButton("Cluster 예측")
        self.predict_button.setObjectName("dataButton")
        self.predict_button.clicked.connect(self._predict_new_sample)
        row.addWidget(self.predict_button)
        layout.addLayout(row)
        self.prediction_result_label = self._create_result_card()
        self.prediction_result_label.setText("새 Sample을 입력하고 Cluster 예측을 눌러주세요.")
        layout.addWidget(self.prediction_result_label)
        code_card, self.prediction_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def set_dataset(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        target_column: str | None,
        task_type: str,
        use_default_notice: bool = False,
    ) -> None:
        self.dataframe = dataframe.copy(deep=True)
        self.dataset_name = dataset_name
        self.target_column = target_column if target_column in self.dataframe.columns else None
        self.task_type = task_type
        self.current_dataset_label.setText(
            "선택된 Dataset이 없어 프로그램에 포함된 Clustering Sample을 사용합니다."
            if use_default_notice else dataset_name
        )
        candidate_features = [
            str(column) for column in self.dataframe.columns
            if str(column) != self.target_column
        ]
        numeric_count = sum(
            pd.api.types.is_numeric_dtype(self.dataframe[column])
            for column in candidate_features
        )
        target_text = self.target_column or "없음"
        self.dataset_info_label.setText(
            f"<b>{escape(dataset_name)}</b> · Target: {escape(target_text)} "
            f"(K-Means 학습에는 사용하지 않음)<br>"
            f"Samples: {len(self.dataframe)} · 사용할 수 있는 숫자형 Features: {numeric_count} · "
            "Train/Test 분리 없음"
        )
        self._populate_features()
        self._reset_workflow()
        if self.dataframe.empty:
            self.status_label.setText("Data Lab에서 Dataset을 불러와 주세요.")
        elif not self._selected_features():
            self.status_label.setText("군집화에 사용할 숫자형 Feature를 선택해 주세요.")
        elif self.target_column:
            self.status_label.setText(
                f"현재 Target인 {self.target_column}은 사용하지 않습니다. Feature를 선택하고 데이터를 준비해 주세요."
            )
        else:
            self.status_label.setText("Feature와 Scaling 적용 여부를 선택한 뒤 데이터를 준비해 주세요.")

    def _load_default_dataset(self) -> None:
        self.set_dataset(
            load_builtin_dataset("Clustering Sample"),
            "Clustering Sample",
            None,
            "clustering",
            use_default_notice=True,
        )

    def _populate_features(self) -> None:
        self._clear_layout(self.feature_grid)
        self.feature_checkboxes.clear()
        excluded = []
        row = 0
        for column in self.dataframe.columns:
            name = str(column)
            if name == self.target_column or not pd.api.types.is_numeric_dtype(self.dataframe[column]):
                continue
            identifier = name.lower() == "id" or name.lower().endswith("_id")
            checkbox = QCheckBox(name)
            checkbox.setObjectName("dataCheckBox")
            checkbox.setChecked(not identifier)
            checkbox.toggled.connect(self._preparation_changed)
            self.feature_checkboxes[name] = checkbox
            self.feature_grid.addWidget(checkbox, row // 3, row % 3)
            row += 1
            if identifier:
                excluded.append(name)
        if excluded:
            self.feature_guide_label.setText(
                "식별자 성격의 Feature는 기본 선택에서 제외했습니다: " + ", ".join(excluded)
            )
        else:
            self.feature_guide_label.setText("거리 계산에 사용할 숫자형 Feature를 선택하세요.")

    def _selected_features(self) -> list[str]:
        return [name for name, box in self.feature_checkboxes.items() if box.isChecked()]

    def _preparation_changed(self, _checked=False) -> None:
        self._reset_after_preparation()
        self.prepare_button.setEnabled(bool(self._selected_features()))
        self.status_label.setText("설정이 변경되었습니다. 데이터를 다시 준비해 주세요.")

    def _prepare_data(self) -> None:
        features = self._selected_features()
        if not features:
            self.status_label.setText("하나 이상의 숫자형 Feature를 선택해 주세요.")
            return
        source = self.dataframe.loc[:, features].copy()
        try:
            validate_clustering_data(source, 1)
        except (TypeError, ValueError) as error:
            self.status_label.setText(f"데이터 준비 실패: {error}")
            return
        if self.scaling_checkbox.isChecked():
            scaler = StandardScaler()
            values = scaler.fit_transform(source)
            prepared = pd.DataFrame(values, index=source.index, columns=features)
        else:
            prepared = source.astype(float)
        self.prepared_features = source
        self.prepared_values = prepared
        self._populate_table(self.original_table, source.reset_index(drop=True))
        self._populate_table(self.scaled_table, prepared.reset_index(drop=True))
        distinct_samples = len(source.drop_duplicates())
        self.cluster_spin.setRange(1, distinct_samples)
        self.cluster_spin.setValue(min(3, distinct_samples))
        self.preparation_code_label.setText(
            "X = df[" + repr(features) + "]\n\n"
            + (
                "from sklearn.preprocessing import StandardScaler\n\n"
                "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)"
                if self.scaling_checkbox.isChecked()
                else "# Scaling을 적용하지 않고 원래 Feature 값을 사용한다.\nX_scaled = X.copy()"
            )
        )
        self.status_label.setText(
            "데이터 준비를 완료했습니다. K-Means 단계에서 Cluster 수를 정하고 모델을 학습하세요."
        )
        self.step_buttons[1].setEnabled(True)
        self._show_step(1)
        self.step_buttons[1].setChecked(True)

    def _model_setting_changed(self, _value=0) -> None:
        self._reset_after_model()
        if not self.prepared_features.empty:
            self.status_label.setText("K-Means 설정이 변경되었습니다. 모델을 다시 학습해 주세요.")

    def _fit_model(self) -> None:
        if self.prepared_features.empty:
            self.status_label.setText("먼저 Feature와 Scaling 단계에서 데이터를 준비해 주세요.")
            return
        try:
            self.result = fit_kmeans(
                self.prepared_features,
                n_clusters=self.cluster_spin.value(),
                scale_features=self.scaling_checkbox.isChecked(),
                random_state=self.random_state_spin.value(),
                n_init=self.n_init_spin.value(),
            )
        except (TypeError, ValueError) as error:
            self.status_label.setText(f"K-Means 학습 실패: {error}")
            return
        counts = pd.Series(self.result.labels).value_counts().sort_index()
        count_frame = pd.DataFrame({
            "Cluster": counts.index,
            "Samples": counts.to_numpy(),
        })
        self._populate_table(self.cluster_count_table, count_frame)
        self.model_result_label.setText(
            f"{len(self.result.labels)}개 Sample을 {self.result.model.n_clusters}개 Cluster로 나눴습니다.<br>"
            "Cluster 번호는 크기나 순위를 의미하지 않습니다."
        )
        self.model_code_label.setText(
            "from sklearn.cluster import KMeans\n"
            f"\nmodel = KMeans(\n    n_clusters={self.cluster_spin.value()},\n"
            f"    random_state={self.random_state_spin.value()},\n"
            f"    n_init={self.n_init_spin.value()},\n)\n"
            f"cluster = model.fit_predict({'X_scaled' if self.scaling_checkbox.isChecked() else 'X'})"
        )
        self._populate_result()
        self._populate_new_sample_controls()
        self.step_buttons[2].setEnabled(True)
        self.step_buttons[3].setEnabled(True)
        self.status_label.setText("K-Means 학습을 완료했습니다. Cluster와 Centroid를 확인하세요.")
        self._show_step(2)
        self.step_buttons[2].setChecked(True)

    def _populate_result(self) -> None:
        if self.result is None:
            return
        clustered = create_clustered_frame(self.dataframe, self.result.labels)
        self._populate_table(self.cluster_result_table, clustered.reset_index(drop=True))
        original = self.result.original_centers.reset_index()
        scaled = self.result.scaled_centers.reset_index()
        self._populate_table(self.original_centroid_table, original)
        self._populate_table(self.scaled_centroid_table, scaled)
        features = list(self.result.feature_names)
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(features)
        self.y_combo.addItems(features)
        if len(features) > 1:
            self.y_combo.setCurrentIndex(1)
        self.graph_button.setEnabled(len(features) >= 2)
        self.canvas.hide()
        self.result_code_label.setText(
            "df['cluster'] = cluster\n\n"
            "scaled_centers = model.cluster_centers_\n"
            + (
                "centers = scaler.inverse_transform(scaled_centers)\n"
                if self.result.scaler is not None else "centers = scaled_centers.copy()\n"
            )
            + "print(df)\nprint(centers)"
        )

    def _draw_cluster_graph(self) -> None:
        if self.result is None or len(self.result.feature_names) < 2:
            return
        x_name = self.x_combo.currentText()
        y_name = self.y_combo.currentText()
        if x_name == y_name:
            self.graph_guide_label.setText("서로 다른 X와 Y Feature를 선택해 주세요.")
            return
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        for cluster in range(self.result.model.n_clusters):
            mask = self.result.labels == cluster
            axes.scatter(
                self.prepared_features.loc[mask, x_name],
                self.prepared_features.loc[mask, y_name],
                s=55,
                label=f"Cluster {cluster}",
            )
        centers = self.result.original_centers
        axes.scatter(
            centers[x_name], centers[y_name], marker="X", s=220,
            color=COLOR_OFF_BLACK, edgecolors=COLOR_OFF_WHITE, linewidths=1.2,
            label="Centroid",
        )
        axes.set_xlabel(x_name)
        axes.set_ylabel(y_name)
        axes.set_title("K-Means Cluster and Centroid")
        axes.legend()
        self.figure.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.16)
        self._style_axes(axes)
        self.canvas.show()
        self.canvas.draw()
        self.graph_guide_label.setText(
            "각 점은 하나의 Sample이고 X 표시는 원래 Feature 단위의 Centroid입니다."
        )

    def _populate_new_sample_controls(self) -> None:
        self._clear_layout(self.new_value_grid)
        self.new_value_controls.clear()
        if self.result is None:
            return
        for index, feature in enumerate(self.result.feature_names):
            label = self._create_control_label(feature)
            control = ChevronDoubleSpinBox()
            control.setObjectName("dataControl")
            control.setRange(-1_000_000_000.0, 1_000_000_000.0)
            control.setDecimals(3)
            control.setValue(float(self.prepared_features[feature].mean()))
            column = (index % 2) * 2
            row = index // 2
            self.new_value_grid.addWidget(label, row, column)
            self.new_value_grid.addWidget(control, row, column + 1)
            self.new_value_grid.setColumnStretch(column + 1, 1)
            self.new_value_controls[feature] = control
        self.prediction_result_label.setText("새 Sample을 입력하고 Cluster 예측을 눌러주세요.")
        self.prediction_code_label.clear()

    def _predict_new_sample(self) -> None:
        if self.result is None:
            self.status_label.setText("먼저 K-Means를 학습해 주세요.")
            return
        values = {
            feature: [control.value()]
            for feature, control in self.new_value_controls.items()
        }
        new_data = pd.DataFrame(values)
        try:
            cluster = int(predict_cluster(self.result, new_data)[0])
        except (TypeError, ValueError) as error:
            self.status_label.setText(f"새 Sample 예측 실패: {error}")
            return
        center = self.result.original_centers.loc[cluster]
        center_text = ", ".join(
            f"{feature}={float(center[feature]):.3f}"
            for feature in self.result.feature_names
        )
        self.prediction_result_label.setText(
            f"예측 Cluster: <b>{cluster}</b><br>해당 Cluster의 Centroid: {escape(center_text)}<br>"
            "이 번호는 K-Means가 만든 그룹 Label이며 정답 Class나 등급이 아닙니다."
        )
        scale_code = (
            "new_data_scaled = scaler.transform(new_data)\n"
            "new_cluster = model.predict(new_data_scaled)"
            if self.result.scaler is not None
            else "new_cluster = model.predict(new_data)"
        )
        self.prediction_code_label.setText(
            f"new_data = pd.DataFrame({values!r})\n\n{scale_code}\nprint(new_cluster)"
        )
        self.status_label.setText("새 Sample을 가장 가까운 Centroid의 Cluster에 배정했습니다.")

    def _reset_workflow(self) -> None:
        self.prepared_features = pd.DataFrame()
        self.prepared_values = pd.DataFrame()
        self.result = None
        for button in self.step_buttons[1:]:
            button.setEnabled(False)
        self._show_step(0)
        self.step_buttons[0].setChecked(True)
        self.prepare_button.setEnabled(bool(self._selected_features()))
        self._clear_table(self.original_table)
        self._clear_table(self.scaled_table)
        self.preparation_code_label.clear()
        self._reset_after_model()

    def _reset_after_preparation(self) -> None:
        self.prepared_features = pd.DataFrame()
        self.prepared_values = pd.DataFrame()
        self.step_buttons[1].setEnabled(False)
        self._clear_table(self.original_table)
        self._clear_table(self.scaled_table)
        self.preparation_code_label.clear()
        self._reset_after_model()

    def _reset_after_model(self) -> None:
        self.result = None
        for button in self.step_buttons[2:]:
            button.setEnabled(False)
        for table in (
            self.cluster_count_table, self.cluster_result_table,
            self.original_centroid_table, self.scaled_centroid_table,
        ):
            self._clear_table(table)
        self.model_result_label.setText("K-Means를 학습하기 전입니다.")
        self.model_code_label.clear()
        self.result_code_label.clear()
        self.canvas.hide()
        self._clear_layout(self.new_value_grid)
        self.new_value_controls.clear()
        self.prediction_result_label.setText("먼저 K-Means를 학습해 주세요.")
        self.prediction_code_label.clear()

    def _show_concept(self, concept: str) -> None:
        description = escape(CONCEPT_DESCRIPTIONS[concept]).replace("다. ", "다.<br>")
        title = escape(CONCEPT_LABELS[concept].replace("\n", " "))
        self.concept_detail_label.setText(f"<b>{title}</b>: {description}")

    def _show_step(self, index: int) -> None:
        for page_index, page in enumerate(self.step_pages):
            page.setHidden(page_index != index)

    def eventFilter(self, watched, event):
        if watched is self.canvas and event.type() == QEvent.Type.Wheel:
            delta = event.pixelDelta().y() or event.angleDelta().y()
            bar = self.scroll_area.verticalScrollBar()
            bar.setValue(bar.value() - delta)
            return True
        return super().eventFilter(watched, event)

    def _style_axes(self, axes) -> None:
        axes.set_facecolor(COLOR_OFF_WHITE)
        axes.tick_params(colors=COLOR_OFF_BLACK)
        axes.xaxis.label.set_color(COLOR_OFF_BLACK)
        axes.yaxis.label.set_color(COLOR_OFF_BLACK)
        for spine in axes.spines.values():
            spine.set_color(COLOR_PRIMARY)

    def _create_section_badge(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionBadge")
        label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        return label

    def _create_step_page(self, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, SPACE_SM, 0, 0)
        layout.setSpacing(SPACE_SM)
        title_label = QLabel(title)
        title_label.setObjectName("stepTitle")
        layout.addWidget(title_label)
        description_label = QLabel(description.replace("다. ", "다.<br>"))
        description_label.setObjectName("bodyText")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        return page, layout

    def _create_control_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("smallText")
        return label

    def _create_result_card(self) -> QLabel:
        label = QLabel()
        label.setObjectName("preprocessingResultCard")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        return label

    def _create_table_card(self, title: str) -> tuple[QFrame, QTableWidget]:
        card = QFrame()
        card.setObjectName("preprocessingCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)
        table = QTableWidget()
        table.setObjectName("dataPreviewTable")
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setHighlightSections(False)
        layout.addWidget(table)
        return card, table

    def _create_code_block(self) -> tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("codeBlockCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        title = QLabel("Python 코드")
        title.setObjectName("codeBlockTitle")
        layout.addWidget(title)
        label = QLabel()
        label.setObjectName("codeBlock")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)
        return card, label

    def _populate_table(self, table: QTableWidget, dataframe: pd.DataFrame) -> None:
        table.clear()
        table.setRowCount(len(dataframe))
        table.setColumnCount(len(dataframe.columns))
        table.setHorizontalHeaderLabels([str(column) for column in dataframe.columns])
        for row in range(len(dataframe)):
            for column in range(len(dataframe.columns)):
                value = dataframe.iloc[row, column]
                if isinstance(value, (float, np.floating)):
                    text = "-" if pd.isna(value) else f"{float(value):.3f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)

    def _clear_table(self, table: QTableWidget) -> None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)
