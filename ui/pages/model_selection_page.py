"""교차 검증으로 모델과 Hyperparameter를 비교하는 Model Selection 화면을 구성한다."""

from copy import deepcopy
from html import escape
from typing import Any

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
from sklearn.metrics import get_scorer as get_sklearn_scorer

from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY,
    SPACE_MD, SPACE_SM, SPACE_XS,
)
from ml.classification import (
    calculate_classification_metrics, create_confusion_frame,
    create_majority_baseline,
)
from ml.data_loader import load_builtin_dataset
from ml.model_selection import (
    SCORING_OPTIONS, CrossValidationResult, compare_candidates,
    create_selection_pipeline, create_validation_splits, display_scores,
    get_candidate_specs, get_scorer, parameter_combination_count,
    pipeline_parameter_grid, run_cross_validation, run_parameter_search,
)
from ml.preprocessing import (
    PreprocessingArtifacts, PreprocessingResult, one_hot_encode_feature,
    split_dataset,
)
from ml.regression import calculate_regression_metrics, create_mean_baseline
from ui.widgets import ChevronComboBox, ChevronSpinBox


CONCEPT_DESCRIPTIONS = {
    "Model Selection": "여러 모델과 Hyperparameter를 같은 검증 기준으로 비교하여 최종 모델을 정하는 과정이다.",
    "Validation Data": "Train Data 안에서 모델과 Hyperparameter를 비교할 때 사용하는 데이터다. 마지막 평가에 사용하는 Test Data와 역할이 다르다.",
    "Cross Validation": "Train Data를 여러 Fold로 나누고 학습과 검증을 반복하여 한 번의 분할에만 의존하는 것을 줄이는 방법이다.",
    "Fold": "Cross Validation을 위해 Train Data를 나눈 각 부분이다. 매 반복에서 한 Fold를 Validation으로 사용하고 나머지를 학습에 사용한다.",
    "K-Fold": "Train Data를 K개의 Fold로 나누고 Validation Fold를 바꾸며 K번 학습하고 평가하는 방법이다.",
    "Stratified K-Fold": "분류에서 각 Fold의 Class 비율이 전체 Train Data와 비슷하게 유지되도록 나누는 방법이다.",
    "cross_val_score()": "모델을 Cross Validation으로 평가하고 Fold별 Validation Score를 반환하는 함수다. 이 화면은 Train Score와 학습 시간도 함께 확인하기 위해 cross_validate()를 사용한다.",
    "Scoring": "Cross Validation과 탐색에서 후보를 비교할 평가 기준이다. 문제의 목적에 맞는 지표를 선택한다.",
    "Parameter": "모델이 Train Data를 학습하며 결정하는 값이다. 선형 모델의 계수와 절편 등이 해당한다.",
    "Hyperparameter": "학습을 시작하기 전에 사용자가 정하는 설정값이다. n_neighbors, max_depth, C, alpha 등이 해당한다.",
    "GridSearchCV": "지정한 Hyperparameter 후보의 모든 조합을 Cross Validation으로 비교하는 기능이다.",
    "RandomizedSearchCV": "후보 조합 중 지정한 수만큼 무작위로 선택하여 Cross Validation으로 비교하는 기능이다.",
    "Pipeline Parameter": "Pipeline 내부의 Hyperparameter는 단계 이름과 설정 이름을 두 개의 밑줄로 연결해 지정한다. model__n_neighbors가 그 예다.",
    "Search Space": "Hyperparameter 탐색에서 확인할 값과 조합의 범위다. 범위가 커지면 필요한 학습 횟수도 증가한다.",
    "n_iter": "RandomizedSearchCV가 무작위로 선택해 확인할 후보 조합의 수다.",
    "random_state": "데이터 분할이나 무작위 후보 선택을 같은 조건에서 다시 확인할 수 있도록 난수 생성 기준을 고정하는 설정값이다.",
    "best_params_": "탐색한 후보 중 가장 좋은 평균 Validation Score를 얻은 Hyperparameter 조합이다.",
    "best_score_": "선택된 Hyperparameter 조합이 Cross Validation에서 얻은 평균 Validation Score다.",
    "best_estimator_": "선택된 Hyperparameter로 전체 Train Data에 다시 학습된 최종 모델이다.",
    "Mean / Std": "Mean은 Fold 점수의 평균이고 Std는 Fold에 따라 점수가 얼마나 달라지는지 나타내는 표준편차다.",
    "Pipeline": "전처리와 모델을 순서대로 연결한다. Cross Validation에서는 각 Fold의 Train Data에만 fit하고 Validation Fold에는 transform한다.",
    "Data Leakage": "Validation이나 Test 정보가 전처리 기준 또는 모델 선택 과정에 미리 들어가는 문제다.",
    "Final Test": "모델과 Hyperparameter 선택을 마친 뒤 따로 보관한 Test Data로 최종 일반화 성능을 확인하는 단계다.",
}

CONCEPT_LABELS = {
    "Model Selection": "Model Selection\n(모델 선택)",
    "Validation Data": "Validation Data\n(검증 데이터)",
    "Cross Validation": "Cross Validation\n(교차 검증)",
    "Fold": "Fold\n(분할 단위)",
    "K-Fold": "K-Fold\n(K겹 교차 검증)",
    "Stratified K-Fold": "Stratified K-Fold\n(계층적 K겹)",
    "cross_val_score()": "cross_val_score()\n(교차 검증 점수)",
    "Scoring": "Scoring\n(평가 기준)",
    "Parameter": "Parameter\n(매개변수)",
    "Hyperparameter": "Hyperparameter\n(하이퍼파라미터)",
    "GridSearchCV": "GridSearchCV\n(전체 조합 탐색)",
    "RandomizedSearchCV": "RandomizedSearchCV\n(무작위 탐색)",
    "Pipeline Parameter": "Pipeline Parameter\n(내부 설정 이름)",
    "Search Space": "Search Space\n(탐색 범위)",
    "n_iter": "n_iter\n(탐색 횟수)",
    "random_state": "random_state\n(난수 기준)",
    "best_params_": "best_params_\n(최적 설정)",
    "best_score_": "best_score_\n(최고 검증 점수)",
    "best_estimator_": "best_estimator_\n(최종 모델)",
    "Mean / Std": "Mean / Std\n(평균 / 표준편차)",
    "Pipeline": "Pipeline\n(처리 과정)",
    "Data Leakage": "Data Leakage\n(데이터 누수)",
    "Final Test": "Final Test\n(최종 평가)",
}

STEP_NAMES = [
    "1. Validation", "2. Cross Validation", "3. Comparison",
    "4. Search", "5. Final Test",
]


class ModelSelectionPage(QWidget):
    """회귀와 분류의 공통 모델 검증·선택 과정을 구성한다."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = ""
        self.target_column: str | None = None
        self.task_type = "unknown"
        self.raw_x_train = pd.DataFrame()
        self.raw_x_test = pd.DataFrame()
        self.raw_y_train = pd.Series(dtype=object)
        self.raw_y_test = pd.Series(dtype=object)
        self.artifacts = PreprocessingArtifacts()
        self.feature_checkboxes: dict[str, QCheckBox] = {}
        self.model_checkboxes: dict[str, QCheckBox] = {}
        self.search_option_boxes: dict[str, list[tuple[Any, QCheckBox]]] = {}

        self.splits: list[tuple[np.ndarray, np.ndarray]] = []
        self.cv_result: CrossValidationResult | None = None
        self.comparison_results: dict[str, CrossValidationResult] = {}
        self.search = None
        self.search_result_table = pd.DataFrame()

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
        title = QLabel("Model Selection")
        title.setObjectName("pageTitle")
        title.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        header.addWidget(title)
        description = QLabel(
            "Train Data 안에서 Cross Validation으로 모델과 Hyperparameter를 비교한다.<br>"
            "선택이 끝난 최종 모델만 따로 보관한 Test Data로 평가한다."
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
        self.concept_buttons["Model Selection"].setChecked(True)
        self._show_concept("Model Selection")

    def _create_workflow_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Model Selection Workflow"), alignment=Qt.AlignmentFlag.AlignLeft)
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE_SM)
        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.step_buttons: list[QPushButton] = []
        self.step_pages = [
            self._create_validation_step(), self._create_cv_step(),
            self._create_comparison_step(), self._create_search_step(),
            self._create_final_step(),
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

    def _create_validation_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Validation Setup",
            "Train Data를 나눌 Fold와 Scoring을 정한다. Classification은 Stratified K-Fold를 사용하고 Regression은 K-Fold를 사용한다.",
        )
        method_controls = QHBoxLayout()
        method_controls.setSpacing(SPACE_SM)
        method_controls.addWidget(self._create_control_label("CV Method"))
        self.cv_method_label = QLabel()
        self.cv_method_label.setObjectName("smallText")
        method_controls.addWidget(self.cv_method_label)
        method_controls.addWidget(self._create_control_label("Folds"))
        self.fold_spin = ChevronSpinBox()
        self.fold_spin.setObjectName("dataControl")
        self.fold_spin.setRange(2, 10)
        self.fold_spin.setValue(5)
        method_controls.addWidget(self.fold_spin)
        method_controls.addStretch()
        layout.addLayout(method_controls)

        scoring_controls = QHBoxLayout()
        scoring_controls.setSpacing(SPACE_SM)
        scoring_controls.addWidget(self._create_control_label("Scoring"))
        self.scoring_combo = ChevronComboBox()
        self.scoring_combo.setObjectName("dataControl")
        scoring_controls.addWidget(self.scoring_combo, 1)
        self.positive_label = self._create_control_label("Positive Class")
        scoring_controls.addWidget(self.positive_label)
        self.positive_combo = ChevronComboBox()
        self.positive_combo.setObjectName("dataControl")
        scoring_controls.addWidget(self.positive_combo, 1)
        layout.addLayout(scoring_controls)

        option_controls = QHBoxLayout()
        option_controls.setSpacing(SPACE_SM)
        self.shuffle_checkbox = QCheckBox("Shuffle")
        self.shuffle_checkbox.setObjectName("dataCheckBox")
        self.shuffle_checkbox.setChecked(True)
        option_controls.addWidget(self.shuffle_checkbox)
        option_controls.addWidget(self._create_control_label("Random State"))
        self.random_state_spin = ChevronSpinBox()
        self.random_state_spin.setObjectName("dataControl")
        self.random_state_spin.setRange(0, 999999)
        self.random_state_spin.setValue(42)
        option_controls.addWidget(self.random_state_spin)
        option_controls.addStretch()
        self.validation_button = QPushButton("검증 설정 적용")
        self.validation_button.setObjectName("dataButton")
        self.validation_button.clicked.connect(self._apply_validation_settings)
        option_controls.addWidget(self.validation_button)
        layout.addLayout(option_controls)

        for control in (
            self.fold_spin, self.random_state_spin, self.scoring_combo,
            self.positive_combo,
        ):
            if hasattr(control, "valueChanged"):
                control.valueChanged.connect(self._validation_setting_changed)
            else:
                control.currentIndexChanged.connect(self._validation_setting_changed)
        self.shuffle_checkbox.toggled.connect(self._validation_setting_changed)
        self.scoring_combo.currentIndexChanged.connect(self._update_positive_control)

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
        self.feature_grid.setHorizontalSpacing(SPACE_XS)
        self.feature_grid.setVerticalSpacing(0)
        for column in range(3):
            self.feature_grid.setColumnStretch(column, 1)
        feature_layout.addLayout(self.feature_grid)
        layout.addWidget(feature_card)

        self.validation_summary_label = self._create_result_card()
        self.validation_summary_label.setText("검증 설정을 적용하기 전입니다.")
        layout.addWidget(self.validation_summary_label)
        card, self.fold_table = self._create_table_card("Fold 구성")
        self.fold_table.setMinimumHeight(200)
        layout.addWidget(card)
        code_card, self.validation_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_cv_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Cross Validation",
            "한 모델을 동일한 Fold에서 반복 학습하고 Fold별 Train Score와 Validation Score를 확인한다.",
        )
        row = QHBoxLayout()
        row.addWidget(self._create_control_label("Model"))
        self.cv_model_combo = ChevronComboBox()
        self.cv_model_combo.setObjectName("dataControl")
        self.cv_model_combo.currentIndexChanged.connect(self._cv_model_changed)
        row.addWidget(self.cv_model_combo, 1)
        self.cv_button = QPushButton("교차 검증 실행")
        self.cv_button.setObjectName("dataButton")
        self.cv_button.clicked.connect(self._run_cv)
        row.addWidget(self.cv_button)
        layout.addLayout(row)
        self.cv_guide_label = QLabel()
        self.cv_guide_label.setObjectName("smallText")
        self.cv_guide_label.setWordWrap(True)
        layout.addWidget(self.cv_guide_label)
        card, self.cv_table = self._create_table_card("Fold Scores")
        layout.addWidget(card)
        self.cv_interpretation_label = self._create_result_card()
        self.cv_interpretation_label.setText("교차 검증을 실행하면 평균과 Fold별 변동을 표시합니다.")
        layout.addWidget(self.cv_interpretation_label)
        self.cv_figure = Figure(facecolor=COLOR_OFF_WHITE)
        self.cv_canvas = FigureCanvasQTAgg(self.cv_figure)
        self.cv_canvas.setMinimumHeight(300)
        self.cv_canvas.installEventFilter(self)
        self.cv_canvas.hide()
        layout.addWidget(self.cv_canvas)
        code_card, self.cv_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_comparison_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Model Comparison",
            "여러 알고리즘을 같은 Train Data, Fold, Scoring으로 비교한다. 평균뿐 아니라 표준편차와 Train Score도 함께 확인한다.",
        )
        card = QFrame()
        card.setObjectName("preprocessingCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        title = QLabel("비교할 Model")
        title.setObjectName("sectionTitle")
        card_layout.addWidget(title)
        self.model_grid = QGridLayout()
        self.model_grid.setHorizontalSpacing(SPACE_XS)
        for column in range(3):
            self.model_grid.setColumnStretch(column, 1)
        card_layout.addLayout(self.model_grid)
        layout.addWidget(card)
        row = QHBoxLayout()
        self.comparison_guide_label = QLabel("두 개 이상의 모델을 선택하세요.")
        self.comparison_guide_label.setObjectName("smallText")
        row.addWidget(self.comparison_guide_label, 1)
        self.compare_button = QPushButton("모델 비교")
        self.compare_button.setObjectName("dataButton")
        self.compare_button.clicked.connect(self._run_comparison)
        row.addWidget(self.compare_button)
        layout.addLayout(row)
        table_card, self.comparison_table = self._create_table_card("Model Comparison")
        layout.addWidget(table_card)
        comparison_note = self._create_result_card()
        comparison_note.setText(
            "가장 높은 Accuracy가 모든 상황에서 가장 좋은 모델을 의미하지는 않는다.<br>"
            "Class 불균형이 있으면 Precision, Recall, F1도 확인한다.<br>"
            "문제의 목적에 따라 학습·예측 시간과 모델의 해석 가능성도 함께 고려한다."
        )
        layout.addWidget(comparison_note)
        self.comparison_figure = Figure(facecolor=COLOR_OFF_WHITE)
        self.comparison_canvas = FigureCanvasQTAgg(self.comparison_figure)
        self.comparison_canvas.setMinimumHeight(380)
        self.comparison_canvas.installEventFilter(self)
        self.comparison_canvas.hide()
        layout.addWidget(self.comparison_canvas)
        code_card, self.comparison_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_search_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Hyperparameter Search",
            "GridSearchCV는 선택한 모든 후보 조합을 확인하고 RandomizedSearchCV는 그중 일부 조합을 무작위로 확인한다.",
        )
        row = QHBoxLayout()
        row.addWidget(self._create_control_label("Model"))
        self.search_model_combo = ChevronComboBox()
        self.search_model_combo.setObjectName("dataControl")
        self.search_model_combo.currentIndexChanged.connect(self._populate_search_options)
        row.addWidget(self.search_model_combo, 1)
        row.addWidget(self._create_control_label("Search"))
        self.search_method_combo = ChevronComboBox()
        self.search_method_combo.setObjectName("dataControl")
        self.search_method_combo.addItem("GridSearchCV", "grid")
        self.search_method_combo.addItem("RandomizedSearchCV", "random")
        self.search_method_combo.currentIndexChanged.connect(self._update_search_scope)
        row.addWidget(self.search_method_combo, 1)
        self.n_iter_label = self._create_control_label("n_iter")
        row.addWidget(self.n_iter_label)
        self.n_iter_spin = ChevronSpinBox()
        self.n_iter_spin.setObjectName("dataControl")
        self.n_iter_spin.setRange(1, 1000)
        self.n_iter_spin.setValue(10)
        self.n_iter_spin.valueChanged.connect(self._update_search_scope)
        row.addWidget(self.n_iter_spin)
        self.search_button = QPushButton("탐색 실행")
        self.search_button.setObjectName("dataButton")
        self.search_button.clicked.connect(self._run_search)
        row.addWidget(self.search_button)
        layout.addLayout(row)

        option_card = QFrame()
        option_card.setObjectName("preprocessingCard")
        option_layout = QVBoxLayout(option_card)
        option_layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        option_title = QLabel("Hyperparameter 후보")
        option_title.setObjectName("sectionTitle")
        option_layout.addWidget(option_title)
        self.search_option_grid = QGridLayout()
        self.search_option_grid.setSpacing(SPACE_XS)
        option_layout.addLayout(self.search_option_grid)
        layout.addWidget(option_card)
        self.search_scope_label = QLabel()
        self.search_scope_label.setObjectName("smallText")
        self.search_scope_label.setWordWrap(True)
        layout.addWidget(self.search_scope_label)
        self.best_result_label = self._create_result_card()
        self.best_result_label.setText("Hyperparameter 탐색을 실행하기 전입니다.")
        layout.addWidget(self.best_result_label)
        table_card, self.search_table = self._create_table_card("Search Results")
        layout.addWidget(table_card)
        code_card, self.search_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_final_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Final Test",
            "선택된 best_estimator_에 처음부터 따로 보관한 Test Data를 한 번 전달하여 최종 일반화 성능을 확인한다.",
        )
        row = QHBoxLayout()
        self.final_model_label = QLabel("먼저 Hyperparameter Search를 완료하세요.")
        self.final_model_label.setObjectName("smallText")
        self.final_model_label.setWordWrap(True)
        row.addWidget(self.final_model_label, 1)
        self.final_button = QPushButton("최종 Test 평가")
        self.final_button.setObjectName("dataButton")
        self.final_button.setEnabled(False)
        self.final_button.clicked.connect(self._run_final_test)
        row.addWidget(self.final_button)
        layout.addLayout(row)
        self.final_result_label = self._create_result_card()
        self.final_result_label.setText("최종 Test 평가 전입니다.")
        layout.addWidget(self.final_result_label)
        table_card, self.final_metric_table = self._create_table_card("Final Metrics")
        self.final_metric_table.setMinimumHeight(210)
        layout.addWidget(table_card)
        self.final_confusion_card, self.final_confusion_table = self._create_table_card("Final Confusion Matrix")
        self.final_confusion_table.setMinimumHeight(180)
        layout.addWidget(self.final_confusion_card)
        code_card, self.final_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def set_preprocessing_result(self, result: PreprocessingResult, use_default_notice: bool = False) -> None:
        self.dataset_name = result.dataset_name
        self.target_column = result.target_column
        self.task_type = result.task_type
        has_raw = not result.raw_x_train.empty and not result.raw_x_test.empty
        self.raw_x_train = (result.raw_x_train if has_raw else result.x_train).copy(deep=True)
        self.raw_x_test = (result.raw_x_test if has_raw else result.x_test).copy(deep=True)
        self.raw_y_train = (result.raw_y_train if not result.raw_y_train.empty else result.y_train).copy(deep=True)
        self.raw_y_test = (result.raw_y_test if not result.raw_y_test.empty else result.y_test).copy(deep=True)
        self.artifacts = deepcopy(result.artifacts) if has_raw else PreprocessingArtifacts(
            input_features=tuple(map(str, self.raw_x_train.columns)),
            output_features=tuple(map(str, self.raw_x_train.columns)),
        )
        self.current_dataset_label.setText(
            "선택된 전처리 결과가 없어 프로그램에 포함된 Classification Sample을 사용합니다."
            if use_default_notice else result.dataset_name
        )
        self.dataset_info_label.setText(
            f"<b>{escape(self.dataset_name)}</b> · Task: {escape(self.task_type.title())} · "
            f"Target: {escape(str(self.target_column or '없음'))}<br>"
            f"Train: {len(self.raw_x_train)} Samples · Final Test: {len(self.raw_x_test)} Samples · "
            f"Raw Features: {len(self.raw_x_train.columns)}"
        )
        self.final_confusion_card.setVisible(self.task_type == "classification")
        self._populate_features()
        self._populate_task_controls()
        if self.task_type == "classification":
            class_counts = self.raw_y_train.value_counts(dropna=False)
            maximum_folds = int(class_counts.min()) if not class_counts.empty else 0
        else:
            maximum_folds = len(self.raw_y_train)
        maximum_folds = min(10, maximum_folds)
        if maximum_folds >= 2:
            self.fold_spin.setRange(2, maximum_folds)
            self.fold_spin.setValue(min(5, maximum_folds))
        self._reset_selection()
        can_validate = (
            self.task_type in {"classification", "regression"}
            and not self.raw_x_train.empty and not self.raw_x_test.empty
            and not self.raw_y_train.empty and bool(self._selected_features())
            and maximum_folds >= 2
        )
        self.validation_button.setEnabled(can_validate)
        if not can_validate:
            self.status_label.setText("Preprocessing에서 Train/Test 분리와 필요한 전처리 설정을 완료해 주세요.")
        elif not has_raw and not use_default_notice:
            self.status_label.setText("원본 Train/Test 정보가 없어 현재 Feature를 그대로 검증합니다. Preprocessing에서 다시 분리하면 Fold별 Pipeline을 사용할 수 있습니다.")
        else:
            self.status_label.setText("Fold와 Scoring을 정한 뒤 검증 설정 적용을 눌러주세요.")

    def _load_default_dataset(self) -> None:
        dataframe = load_builtin_dataset("Classification Sample")
        target = "purchased"
        features = [column for column in dataframe.columns if column != target]
        raw_x_train, raw_x_test, y_train, y_test = split_dataset(
            dataframe, features, target, 0.25, 42, True
        )
        _, _, _, encoder = one_hot_encode_feature(
            raw_x_train, raw_x_test, "member_type"
        )
        artifacts = PreprocessingArtifacts(
            input_features=tuple(features),
            feature_encoders={"member_type": encoder},
        )
        self.set_preprocessing_result(
            PreprocessingResult(
                dataset_name="Classification Sample", target_column=target,
                task_type="classification", x_train=raw_x_train, x_test=raw_x_test,
                y_train=y_train, y_test=y_test, artifacts=artifacts,
                raw_x_train=raw_x_train, raw_x_test=raw_x_test,
                raw_y_train=y_train, raw_y_test=y_test,
            ),
            use_default_notice=True,
        )

    def _populate_features(self) -> None:
        self._clear_layout(self.feature_grid)
        self.feature_checkboxes.clear()
        excluded = []
        for index, column in enumerate(self.raw_x_train.columns):
            name = str(column)
            checkbox = QCheckBox(name)
            checkbox.setObjectName("dataCheckBox")
            identifier = name.lower() == "id" or name.lower().endswith("_id")
            checkbox.setChecked(not identifier)
            checkbox.toggled.connect(self._validation_setting_changed)
            self.feature_checkboxes[name] = checkbox
            self.feature_grid.addWidget(checkbox, index // 3, index % 3)
            if identifier:
                excluded.append(name)
        self.feature_guide_label.setText(
            "식별자 성격의 Feature는 기본 선택에서 제외했습니다: " + ", ".join(excluded)
            if excluded else "모델 비교에 사용할 Feature를 선택하세요."
        )

    def _populate_task_controls(self) -> None:
        self.scoring_combo.blockSignals(True)
        self.scoring_combo.clear()
        for key, label in SCORING_OPTIONS.get(self.task_type, {}).items():
            self.scoring_combo.addItem(label, key)
        self.scoring_combo.blockSignals(False)
        self.cv_method_label.setText(
            "Stratified K-Fold" if self.task_type == "classification" else "K-Fold"
        )
        self.positive_combo.blockSignals(True)
        self.positive_combo.clear()
        labels = list(pd.unique(self.raw_y_train))
        try:
            labels = sorted(labels)
        except TypeError:
            pass
        for label in labels:
            self.positive_combo.addItem(str(label), label)
        if len(labels) == 2:
            self.positive_combo.setCurrentIndex(1)
        self.positive_combo.blockSignals(False)
        self._update_positive_control()

        specs = get_candidate_specs(self.task_type)
        for combo in (self.cv_model_combo, self.search_model_combo):
            combo.blockSignals(True)
            combo.clear()
            for key, spec in specs.items():
                combo.addItem(spec.label, key)
            if self.task_type == "classification":
                combo.setCurrentIndex(combo.findData("knn"))
            elif self.task_type == "regression":
                combo.setCurrentIndex(combo.findData("ridge"))
            combo.blockSignals(False)
        self._populate_model_checkboxes()
        self._populate_search_options()
        self._update_cv_guide()

    def _populate_model_checkboxes(self) -> None:
        self._clear_layout(self.model_grid)
        self.model_checkboxes.clear()
        for index, (key, spec) in enumerate(get_candidate_specs(self.task_type).items()):
            checkbox = QCheckBox(spec.label)
            checkbox.setObjectName("dataCheckBox")
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._comparison_selection_changed)
            self.model_checkboxes[key] = checkbox
            self.model_grid.addWidget(checkbox, index // 3, index % 3)
        self._comparison_selection_changed()

    def _selected_features(self) -> list[str]:
        return [
            str(column) for column in self.raw_x_train.columns
            if str(column) in self.feature_checkboxes and self.feature_checkboxes[str(column)].isChecked()
        ]

    def _selection_artifacts(self) -> PreprocessingArtifacts:
        features = set(self._selected_features())
        artifacts = deepcopy(self.artifacts)
        artifacts.input_features = tuple(
            column for column in map(str, self.raw_x_train.columns) if column in features
        )
        artifacts.output_features = ()
        artifacts.feature_imputers = {
            column: transformer for column, transformer in artifacts.feature_imputers.items()
            if column in features
        }
        artifacts.feature_encoders = {
            column: transformer for column, transformer in artifacts.feature_encoders.items()
            if column in features
        }
        artifacts.scaler_columns = tuple(
            column for column in artifacts.scaler_columns if column in features
        )
        artifacts.preprocessor = None
        return artifacts

    def _current_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        features = self._selected_features()
        return (
            self.raw_x_train.loc[:, features], self.raw_x_test.loc[:, features],
            self.raw_y_train, self.raw_y_test,
        )

    def _update_positive_control(self, _index=-1) -> None:
        binary_metric = self.scoring_combo.currentData() in {"precision", "recall", "f1"}
        visible = self.task_type == "classification" and self.raw_y_train.nunique() == 2 and binary_metric
        self.positive_label.setVisible(visible)
        self.positive_combo.setVisible(visible)

    def _positive_class(self):
        if self.task_type == "classification" and self.raw_y_train.nunique() == 2:
            return self.positive_combo.currentData()
        return None

    def _validation_setting_changed(self, _value=None) -> None:
        if hasattr(self, "validation_summary_label"):
            self._reset_selection("검증 설정이 변경되었습니다. 다시 적용해 주세요.")
        self.validation_button.setEnabled(bool(self._selected_features()))

    def _apply_validation_settings(self) -> None:
        if not self._selected_features():
            self.status_label.setText("하나 이상의 Feature를 선택해 주세요.")
            return
        try:
            self.splits = create_validation_splits(
                self.task_type,
                self.raw_y_train,
                self.fold_spin.value(),
                shuffle=self.shuffle_checkbox.isChecked(),
                random_state=self.random_state_spin.value(),
            )
        except ValueError as error:
            self.status_label.setText(f"검증 설정 실패: {error}")
            return
        rows = []
        for index, (train_index, validation_index) in enumerate(self.splits):
            rows.append({
                "Fold": f"Fold {index + 1}",
                "Train Samples": len(train_index),
                "Validation Samples": len(validation_index),
            })
        self._populate_table(self.fold_table, pd.DataFrame(rows))
        scoring = self.scoring_combo.currentText()
        self.validation_summary_label.setText(
            f"<b>{escape(self.cv_method_label.text())}</b><br>"
            f"Train: {len(self.raw_x_train)} Samples · Final Test: {len(self.raw_x_test)} Samples<br>"
            f"Folds: {len(self.splits)} · Scoring: {escape(scoring)} · "
            f"Features: {escape(', '.join(self._selected_features()))}"
        )
        splitter = "StratifiedKFold" if self.task_type == "classification" else "KFold"
        extra = ", stratify와 같은 Class 비율 유지" if self.task_type == "classification" else ""
        self.validation_code_label.setText(
            f"from sklearn.model_selection import {splitter}\n\n"
            f"cv = {splitter}(\n    n_splits={self.fold_spin.value()},\n"
            f"    shuffle={self.shuffle_checkbox.isChecked()},\n"
            f"    random_state={self.random_state_spin.value() if self.shuffle_checkbox.isChecked() else None},\n)"
            f"\n# Final Test Data는 모델 선택에 사용하지 않는다{extra}"
        )
        self.status_label.setText("검증 설정을 적용했습니다. Cross Validation에서 한 모델의 Fold 결과를 확인하세요.")
        self._populate_search_options()
        self.step_buttons[1].setEnabled(True)
        self._show_step(1)
        self.step_buttons[1].setChecked(True)

    def _cv_model_changed(self, _index=-1) -> None:
        self.cv_result = None
        self._clear_table(self.cv_table)
        self.cv_canvas.hide()
        self.cv_code_label.clear()
        self.cv_interpretation_label.setText("교차 검증을 실행하면 평균과 Fold별 변동을 표시합니다.")
        self._update_cv_guide()

    def _update_cv_guide(self) -> None:
        if not hasattr(self, "cv_guide_label") or self.cv_model_combo.currentData() is None:
            return
        spec = get_candidate_specs(self.task_type)[self.cv_model_combo.currentData()]
        pipeline_text = "전처리"
        if spec.needs_scaling and self.artifacts.scaler is None:
            pipeline_text += " → StandardScaler"
        pipeline_text += f" → {spec.label}"
        self.cv_guide_label.setText(
            f"각 Fold에서 {pipeline_text} 순서로 Train Fold에 fit하고 Validation Fold에 같은 기준을 적용합니다."
        )

    def _run_cv(self) -> None:
        if not self.splits:
            self.status_label.setText("먼저 Validation 단계에서 검증 설정을 적용해 주세요.")
            return
        train_x, _, train_y, _ = self._current_data()
        key = self.cv_model_combo.currentData()
        try:
            pipeline = create_selection_pipeline(
                self.task_type, key, self._selection_artifacts()
            )
            self.cv_result = run_cross_validation(
                pipeline, train_x, train_y, self.splits,
                task_type=self.task_type,
                scoring_key=self.scoring_combo.currentData(),
                positive_label=self._positive_class(),
            )
        except (TypeError, ValueError) as error:
            self.status_label.setText(f"교차 검증 실패: {error}")
            return
        self._populate_table(self.cv_table, self.cv_result.fold_table)
        self.cv_interpretation_label.setText(
            f"평균 Train Score는 {self.cv_result.train_mean:.3f}이고 평균 Validation Score는 "
            f"{self.cv_result.validation_mean:.3f}이다.<br>"
            f"Validation Score의 표준편차는 {self.cv_result.validation_std:.3f}이다.<br>"
            "Train과 Validation의 차이와 Fold별 변동을 함께 확인하며 이 결과만으로 과적합을 단정하지 않는다."
        )
        self._draw_cv_graph()
        self.cv_code_label.setText(
            "from sklearn.model_selection import cross_validate\n\n"
            "result = cross_validate(\n    pipeline, X_train, y_train,\n"
            f"    cv=cv, scoring={self._scoring_code()!r},\n"
            "    return_train_score=True,\n)\n\n"
            "print(result['test_score'])\nprint(result['test_score'].mean())"
        )
        self.status_label.setText("교차 검증을 완료했습니다. 같은 Fold로 여러 모델을 비교할 수 있습니다.")
        self.step_buttons[2].setEnabled(True)

    def _draw_cv_graph(self) -> None:
        if self.cv_result is None:
            return
        self.cv_figure.clear()
        axes = self.cv_figure.add_subplot(111)
        values = self.cv_result.validation_scores
        positions = np.arange(1, len(values) + 1)
        axes.bar(positions, values, color=COLOR_SECONDARY)
        axes.axhline(np.mean(values), color=COLOR_OFF_BLACK, linestyle="--", label="Mean")
        axes.set_xticks(positions, [f"Fold {value}" for value in positions])
        axes.set_ylabel(self.scoring_combo.currentText())
        axes.set_title("Validation Score by Fold")
        axes.legend()
        self._style_axes(axes)
        self.cv_canvas.show()
        self.cv_canvas.draw()

    def _comparison_selection_changed(self, _checked=False) -> None:
        count = sum(box.isChecked() for box in self.model_checkboxes.values())
        self.comparison_guide_label.setText(f"선택한 모델: {count}개")
        self.compare_button.setEnabled(count >= 2)
        self.search = None

    def _run_comparison(self) -> None:
        keys = [key for key, box in self.model_checkboxes.items() if box.isChecked()]
        if len(keys) < 2 or not self.splits:
            return
        train_x, _, train_y, _ = self._current_data()
        try:
            frame, self.comparison_results = compare_candidates(
                self.task_type, keys, self._selection_artifacts(),
                train_x, train_y, self.splits,
                scoring_key=self.scoring_combo.currentData(),
                positive_label=self._positive_class(),
            )
        except (TypeError, ValueError) as error:
            self.status_label.setText(f"모델 비교 실패: {error}")
            return
        self._populate_table(self.comparison_table, frame)
        self._draw_comparison_graph(keys)
        self.comparison_code_label.setText(
            "models = {'KNN': knn_pipeline, 'Random Forest': forest_pipeline}\n\n"
            "for name, candidate in models.items():\n"
            "    result = cross_validate(\n        candidate, X_train, y_train,\n"
            f"        cv=cv, scoring={self._scoring_code()!r},\n"
            "        return_train_score=True,\n    )\n"
            "    print(name, result['test_score'].mean())"
        )
        best_name = str(frame.iloc[0]["Model"])
        self.status_label.setText(
            f"모델 비교를 완료했습니다. 현재 검증 기준의 첫 번째 모델은 {best_name}입니다. 다른 조건도 함께 확인하세요."
        )
        best_key = next(
            key for key, spec in get_candidate_specs(self.task_type).items()
            if spec.label == best_name
        )
        self.search_model_combo.setCurrentIndex(self.search_model_combo.findData(best_key))
        self.step_buttons[3].setEnabled(True)

    def _draw_comparison_graph(self, keys: list[str]) -> None:
        specs = get_candidate_specs(self.task_type)
        self.comparison_figure.clear()
        axes = self.comparison_figure.add_subplot(111)
        data = [self.comparison_results[key].validation_scores for key in keys]
        labels = []
        for key in keys:
            words = specs[key].label.rsplit(" ", 1)
            labels.append("\n".join(words) if len(words) == 2 else words[0])
        axes.boxplot(data, tick_labels=labels, patch_artist=True)
        axes.set_ylabel(self.scoring_combo.currentText())
        axes.set_title("Validation Score Distribution")
        axes.tick_params(axis="x", pad=8)
        self.comparison_figure.subplots_adjust(
            left=0.10, right=0.98, top=0.88, bottom=0.24
        )
        self._style_axes(axes)
        self.comparison_canvas.show()
        self.comparison_canvas.draw()

    def _populate_search_options(self, _index=-1) -> None:
        self._clear_layout(self.search_option_grid)
        self.search_option_boxes.clear()
        key = self.search_model_combo.currentData()
        if key is None:
            return
        spec = get_candidate_specs(self.task_type)[key]
        if not spec.parameter_options:
            label = QLabel("별도로 탐색할 Hyperparameter가 없어 기본 설정 한 개를 검증합니다.")
            label.setObjectName("smallText")
            self.search_option_grid.addWidget(label, 0, 0)
        for row, (parameter, values) in enumerate(spec.parameter_options.items()):
            if spec.key == "knn" and parameter == "n_neighbors" and self.splits:
                max_neighbors = min(len(train_index) for train_index, _ in self.splits)
                values = [value for value in values if int(value) <= max_neighbors]
            name_label = QLabel(parameter)
            name_label.setObjectName("smallText")
            self.search_option_grid.addWidget(name_label, row, 0)
            self.search_option_boxes[parameter] = []
            for column, value in enumerate(values, start=1):
                checkbox = QCheckBox("None" if value is None else str(value))
                checkbox.setObjectName("dataCheckBox")
                checkbox.setChecked(True)
                checkbox.toggled.connect(self._update_search_scope)
                self.search_option_boxes[parameter].append((value, checkbox))
                self.search_option_grid.addWidget(checkbox, row, column)
        self.search = None
        self.best_result_label.setText("Hyperparameter 탐색을 실행하기 전입니다.")
        self._clear_table(self.search_table)
        self.search_code_label.clear()
        self.final_button.setEnabled(False)
        self._update_search_scope()

    def _selected_search_options(self) -> dict[str, list[Any]]:
        return {
            parameter: [value for value, box in boxes if box.isChecked()]
            for parameter, boxes in self.search_option_boxes.items()
        }

    def _update_search_scope(self, _value=None) -> None:
        random_search = self.search_method_combo.currentData() == "random"
        self.n_iter_label.setVisible(random_search)
        self.n_iter_spin.setVisible(random_search)
        options = self._selected_search_options()
        if any(not values for values in options.values()):
            self.search_scope_label.setText("각 Hyperparameter에서 하나 이상의 후보를 선택하세요.")
            self.search_button.setEnabled(False)
            return
        total = parameter_combination_count(options)
        selected = min(total, self.n_iter_spin.value()) if random_search else total
        fits = selected * max(len(self.splits), self.fold_spin.value())
        self.search_scope_label.setText(
            f"전체 후보 조합: {total}개 · 이번 탐색 조합: {selected}개 · "
            f"Fold별 학습: {fits}회 · 선택 후 전체 Train 재학습: 1회"
        )
        self.search_button.setEnabled(bool(self.splits))

    def _run_search(self) -> None:
        if not self.splits:
            return
        options = self._selected_search_options()
        if any(not values for values in options.values()):
            return
        key = self.search_model_combo.currentData()
        spec = get_candidate_specs(self.task_type)[key]
        train_x, _, train_y, _ = self._current_data()
        try:
            pipeline = create_selection_pipeline(
                self.task_type, key, self._selection_artifacts()
            )
            self.search, frame = run_parameter_search(
                pipeline, spec, train_x, train_y, self.splits,
                task_type=self.task_type,
                scoring_key=self.scoring_combo.currentData(),
                selected_options=options,
                search_method=self.search_method_combo.currentData(),
                n_iter=self.n_iter_spin.value(),
                positive_label=self._positive_class(),
                random_state=self.random_state_spin.value(),
            )
        except (TypeError, ValueError) as error:
            self.status_label.setText(f"Hyperparameter 탐색 실패: {error}")
            return
        self.search_result_table = frame
        self._populate_table(self.search_table, frame)
        best_score = display_scores(
            np.asarray([self.search.best_score_]), self.task_type,
            self.scoring_combo.currentData(),
        )[0]
        best_params = ", ".join(
            f"{name.split('__')[-1]}={value}" for name, value in self.search.best_params_.items()
        ) or "기본 설정"
        self.best_result_label.setText(
            f"<b>{escape(spec.label)}</b><br>Best Hyperparameter: {escape(best_params)}<br>"
            f"Best Validation {escape(self.scoring_combo.currentText())}: {best_score:.3f}"
        )
        search_class = "GridSearchCV" if self.search_method_combo.currentData() == "grid" else "RandomizedSearchCV"
        parameter_name = "param_grid" if search_class == "GridSearchCV" else "param_distributions"
        extra = "" if search_class == "GridSearchCV" else f",\n    n_iter={min(self.n_iter_spin.value(), parameter_combination_count(options))}, random_state={self.random_state_spin.value()}"
        code_grid = pipeline_parameter_grid(spec, options)
        self.search_code_label.setText(
            f"from sklearn.model_selection import {search_class}\n\n"
            f"search = {search_class}(\n    pipeline,\n    {parameter_name}={code_grid!r},\n"
            f"    cv=cv, scoring={self._scoring_code()!r}{extra},\n)\n"
            "search.fit(X_train, y_train)\n\n"
            "print(search.best_params_)\nprint(search.best_score_)\n"
            "final_model = search.best_estimator_"
        )
        self.final_model_label.setText(
            f"Selected Model: {spec.label}<br>Best Hyperparameter: {best_params}<br>"
            f"Mean Validation {self.scoring_combo.currentText()}: {best_score:.3f}"
        )
        self.final_button.setEnabled(True)
        self.step_buttons[4].setEnabled(True)
        self.status_label.setText("Hyperparameter 탐색을 완료했습니다. Final Test에서 보관한 Test Data로 최종 평가하세요.")

    def _run_final_test(self) -> None:
        if self.search is None:
            return
        _, test_x, train_y, test_y = self._current_data()
        prediction = self.search.best_estimator_.predict(test_x)
        best_validation = display_scores(
            np.asarray([self.search.best_score_]), self.task_type,
            self.scoring_combo.currentData(),
        )[0]
        scorer = get_scorer(
            self.task_type,
            self.scoring_combo.currentData(),
            self._positive_class(),
        )
        if isinstance(scorer, str):
            scorer = get_sklearn_scorer(scorer)
        raw_final_score = scorer(self.search.best_estimator_, test_x, test_y)
        final_score = display_scores(
            np.asarray([raw_final_score]),
            self.task_type,
            self.scoring_combo.currentData(),
        )[0]
        if self.task_type == "classification":
            positive = self._positive_class()
            model_metrics = calculate_classification_metrics(
                test_y, prediction, positive_label=positive
            )
            baseline_prediction, _ = create_majority_baseline(
                self.raw_x_train.loc[:, self._selected_features()],
                train_y,
                test_x,
            )
            baseline = calculate_classification_metrics(
                test_y, baseline_prediction, positive_label=positive
            )
            frame = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1"],
                "Final Model": [model_metrics.accuracy, model_metrics.precision, model_metrics.recall, model_metrics.f1],
                "Test Baseline": [baseline.accuracy, baseline.precision, baseline.recall, baseline.f1],
            })
            self._populate_table(self.final_metric_table, frame)
            labels = list(self.search.best_estimator_.classes_)
            confusion = create_confusion_frame(test_y, prediction, labels)
            self._populate_table(
                self.final_confusion_table,
                confusion.reset_index(names="Actual / Prediction"),
            )
        else:
            model_metrics = calculate_regression_metrics(test_y, prediction)
            baseline_prediction = create_mean_baseline(train_y, len(test_y))
            baseline = calculate_regression_metrics(test_y, baseline_prediction)
            frame = pd.DataFrame({
                "Metric": ["MAE", "MSE", "RMSE", "R²"],
                "Final Model": [model_metrics.mae, model_metrics.mse, model_metrics.rmse, model_metrics.r2],
                "Test Baseline": [baseline.mae, baseline.mse, baseline.rmse, baseline.r2],
            })
            self._populate_table(self.final_metric_table, frame)
            self._clear_table(self.final_confusion_table)
        self.final_result_label.setText(
            f"Mean Validation {escape(self.scoring_combo.currentText())}: {best_validation:.3f}"
            f"<br>Final Test {escape(self.scoring_combo.currentText())}: {final_score:.3f}"
            "<br>Validation과 Test 결과는 데이터 구성에 따라 다를 수 있다."
        )
        self.final_code_label.setText(
            "final_model = search.best_estimator_\n"
            "test_prediction = final_model.predict(X_test)\n\n"
            "# Test Data는 모델과 Hyperparameter 선택이 끝난 뒤 최종 평가에 사용한다."
        )
        self.final_button.setEnabled(False)
        self.status_label.setText("최종 Test 평가를 완료했습니다.")

    def _scoring_code(self) -> str:
        key = self.scoring_combo.currentData()
        if self.task_type == "regression":
            return {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}[key]
        return key

    def _reset_selection(self, message: str | None = None) -> None:
        self.splits = []
        self.cv_result = None
        self.comparison_results = {}
        self.search = None
        self.search_result_table = pd.DataFrame()
        if not hasattr(self, "step_buttons"):
            return
        for button in self.step_buttons[1:]:
            button.setEnabled(False)
        self._show_step(0)
        self.step_buttons[0].setChecked(True)
        self.validation_summary_label.setText("검증 설정을 적용하기 전입니다.")
        self.cv_interpretation_label.setText("교차 검증을 실행하면 평균과 Fold별 변동을 표시합니다.")
        self.best_result_label.setText("Hyperparameter 탐색을 실행하기 전입니다.")
        self.final_model_label.setText("먼저 Hyperparameter Search를 완료하세요.")
        self.final_result_label.setText("최종 Test 평가 전입니다.")
        self.final_button.setEnabled(False)
        for table in (
            self.fold_table, self.cv_table, self.comparison_table,
            self.search_table, self.final_metric_table, self.final_confusion_table,
        ):
            self._clear_table(table)
        for label in (
            self.validation_code_label, self.cv_code_label, self.comparison_code_label,
            self.search_code_label, self.final_code_label,
        ):
            label.clear()
        for canvas in (self.cv_canvas, self.comparison_canvas):
            canvas.hide()
        if message:
            self.status_label.setText(message)

    def _show_concept(self, concept: str) -> None:
        description = escape(CONCEPT_DESCRIPTIONS[concept]).replace("다. ", "다.<br>")
        label = escape(CONCEPT_LABELS[concept].replace("\n", " "))
        self.concept_detail_label.setText(f"<b>{label}</b>: {description}")

    def _show_step(self, index: int) -> None:
        for page_index, page in enumerate(self.step_pages):
            page.setHidden(page_index != index)

    def eventFilter(self, watched, event):
        canvases = {
            canvas for canvas in (
                getattr(self, "cv_canvas", None),
                getattr(self, "comparison_canvas", None),
            )
            if canvas is not None
        }
        if watched in canvases and event.type() == QEvent.Type.Wheel:
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
        table.setMinimumHeight(240)
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
                if pd.isna(value):
                    text = "계산 불가"
                elif isinstance(value, (float, np.floating)):
                    text = f"{float(value):.3f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)

    def _clear_table(self, table: QTableWidget) -> None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

