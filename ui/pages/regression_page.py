# 선형 회귀와 Ridge 모델을 학습하고 평가하는 Regression Lab 화면을 구성
from html import escape
from math import comb

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

from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY,
    SPACE_MD, SPACE_SM, SPACE_XS,
)
from ml.data_loader import load_builtin_dataset
from ml.preprocessing import PreprocessingArtifacts, PreprocessingResult, split_dataset
from ml.regression import (
    RegressionMetrics, calculate_regression_metrics, create_mean_baseline,
    create_regressor, validate_regression_data,
)
from ui.widgets import ChevronComboBox, ChevronDoubleSpinBox, ChevronSpinBox


CONCEPT_DESCRIPTIONS = {
    "Regression": "Feature를 이용해 연속적인 숫자 Target을 예측하는 머신러닝 문제다.",
    "Linear Regression": "Feature와 Target 사이의 관계를 선형식으로 표현하여 숫자 값을 예측하는 알고리즘이다.",
    "Linear Equation": "Feature가 하나인 선형 회귀는 예측값 = Feature × 계수 + 절편 형태로 값을 계산한다.",
    "Coefficient": "다른 Feature가 같다고 가정할 때 해당 Feature의 변화에 따라 예측값이 얼마나 변하는지를 나타내며 선형 회귀에서는 가중치라고도 한다. Scaling이나 다항 Feature를 사용하면 계수의 단위와 해석이 달라진다.",
    "Intercept": "Feature 값이 0일 때 선형식이 가지는 기본값이다. 데이터에 따라 실제 의미를 가지지 않을 수도 있다.",
    "Least Squares": "각 Sample의 오차를 제곱한 값들의 합이 작아지는 방향으로 계수와 절편을 찾는 방법이다.",
    "Multiple Regression": "여러 Feature를 사용하며 각 Feature마다 하나의 계수를 학습하는 선형 회귀다.",
    "Polynomial Regression": "PolynomialFeatures로 원래 Feature를 거듭제곱과 조합 형태로 확장한 뒤 LinearRegression으로 곡선 관계를 학습하는 방법이다.",
    "Ridge": "계수의 제곱 크기에 규제를 적용하여 계수가 지나치게 커지지 않도록 하는 선형 회귀다.",
    "Lasso": "계수의 절댓값 크기에 규제를 적용하며 규제가 강해지면 일부 Feature의 계수를 0으로 만들 수 있는 선형 회귀다.",
    "ElasticNet": "Ridge의 제곱 규제와 Lasso의 절댓값 규제를 함께 사용하는 선형 회귀다. alpha는 전체 규제 강도를 정하고 l1_ratio는 두 규제 방식의 비율을 정한다.",
    "Regularization": "모델의 계수 크기에 제한을 추가하여 학습 데이터에 지나치게 맞춰지는 것을 줄이는 방법이다.",
    "Hyperparameter": "모델이 학습으로 구하는 계수와 달리 학습을 시작하기 전에 사용자가 정하는 설정값이다. degree, alpha, l1_ratio 등이 있으며 Validation Data나 Cross Validation 결과를 기준으로 선택한다.",
    "alpha": "Ridge, Lasso, ElasticNet의 전체 규제 강도를 정하는 하이퍼파라미터다. 값이 커질수록 규제가 강해진다.",
    "Actual / Prediction": "Actual은 Test Target의 실제값이고 Prediction은 모델이 계산한 예측값이다.",
    "Error / Residual": "실제값에서 예측값을 뺀 차이다. 여러 오차를 단순히 더하면 양수와 음수가 상쇄될 수 있다.",
    "MAE": "실제값과 예측값 차이의 절댓값을 평균한 값이다. Target과 같은 단위이며 0에 가까울수록 오차가 작다.",
    "MSE": "실제값과 예측값의 차이를 제곱한 뒤 평균한 값이다. 큰 오차를 더 크게 반영한다.",
    "RMSE": "MSE에 제곱근을 적용한 값이다. Target과 같은 단위이며 큰 오차의 영향을 많이 받는다.",
    "R²": "모델이 Target 값의 변화를 어느 정도 설명하는지 나타내는 지표다. 정확도 백분율이 아니며 음수가 나올 수 있다. Train Target 평균을 사용하는 Test Baseline의 R²는 정확히 0이 아닐 수 있다.",
    "Baseline": "모델 성능과 비교하기 위한 간단한 기준이다. 회귀에서는 Train Target 평균을 모든 Test Sample에 예측할 수 있다.",
    "Validation / CV": "Validation Data나 Cross Validation은 모델과 Hyperparameter를 비교할 때 사용한다. Test Data는 선택이 끝난 최종 모델의 일반화 성능을 확인할 때 사용한다.",
    "Generalization": "모델이 학습하지 않은 새로운 데이터에서도 적절한 예측을 만드는 능력이다.",
    "Overfitting": "Train Data에서는 좋은 결과를 보이지만 Test Data에서는 성능이 크게 떨어지는 상태다.",
    "Underfitting": "모델이 데이터의 관계를 충분히 학습하지 못해 Train과 Test 모두에서 성능이 낮은 상태다.",
}

CONCEPT_LABELS = {
    "Regression": "Regression\n(회귀)",
    "Linear Regression": "Linear Regression\n(선형 회귀)",
    "Linear Equation": "Linear Equation\n(선형식)",
    "Coefficient": "Coefficient\n(계수)",
    "Intercept": "Intercept\n(절편)",
    "Least Squares": "Least Squares\n(최소제곱법)",
    "Multiple Regression": "Multiple Regression\n(다중 선형 회귀)",
    "Polynomial Regression": "Polynomial Regression\n(다항 회귀)",
    "Ridge": "Ridge\n(릿지 회귀)",
    "Lasso": "Lasso\n(라쏘 회귀)",
    "ElasticNet": "ElasticNet\n(엘라스틱넷)",
    "Regularization": "Regularization\n(규제)",
    "Hyperparameter": "Hyperparameter\n(하이퍼파라미터)",
    "alpha": "alpha\n(규제 강도)",
    "Actual / Prediction": "Actual / Prediction\n(실제값 / 예측값)",
    "Error / Residual": "Error / Residual\n(오차 / 잔차)",
    "MAE": "MAE\n(평균 절대 오차)",
    "MSE": "MSE\n(평균 제곱 오차)",
    "RMSE": "RMSE\n(평균 제곱근 오차)",
    "R²": "R²\n(결정계수)",
    "Baseline": "Baseline\n(기준 결과)",
    "Validation / CV": "Validation / CV\n(검증 / 교차 검증)",
    "Generalization": "Generalization\n(일반화)",
    "Overfitting": "Overfitting\n(과적합)",
    "Underfitting": "Underfitting\n(과소적합)",
}

STEP_NAMES = ["1. Model", "2. Prediction", "3. Evaluation", "4. Visualization"]

GRAPH_DESCRIPTIONS = {
    "Actual vs Prediction": (
        "<b>Actual vs Prediction(실제값과 예측값)</b>: 실제값과 예측값을 점으로 표시한다. "
        "점이 대각선에 가까울수록 두 값의 차이가 작다."
    ),
    "Regression Line": (
        "<b>Regression Line(회귀선)</b>: 하나의 Feature와 Target을 점으로 표시하고 "
        "모델이 학습한 선형 관계를 선으로 나타낸다."
    ),
    "Residual Plot": (
        "<b>Residual Plot(잔차 그래프)</b>: 예측값에 따른 잔차를 표시한다. "
        "잔차는 실제값에서 예측값을 뺀 값이다."
    ),
}


class RegressionPage(QWidget):
    # Regression Lab의 데이터, 모델, 평가 결과와 UI를 구성
    def __init__(self) -> None:
        super().__init__()

        self.dataset_name = ""
        self.target_column: str | None = None
        self.task_type = "unknown"
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series(dtype=float)
        self.y_test = pd.Series(dtype=float)
        self.preprocessing_artifacts = PreprocessingArtifacts()
        self.feature_checkboxes: dict[str, QCheckBox] = {}

        self.model = None
        self.train_prediction: np.ndarray | None = None
        self.test_prediction: np.ndarray | None = None
        self.train_metrics: RegressionMetrics | None = None
        self.test_metrics: RegressionMetrics | None = None
        self.baseline_metrics: RegressionMetrics | None = None

        self._setup_ui()
        self._load_default_dataset()

    # Regression 전체 화면을 Dataset, Concept, 단계별 실습으로 구성
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
        title = QLabel("Regression")
        title.setObjectName("pageTitle")
        title.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        header.addWidget(title)

        description = QLabel(
            "회귀 모델은 Feature를 이용해 연속적인 숫자 Target을 예측한다.<br>"
            "Train Data로 모델을 학습하고 Test Data의 예측 결과를 평가한다."
        )
        description.setObjectName("bodyText")
        description.setWordWrap(True)
        header.addWidget(description, 1)
        layout.addLayout(header)

        self._create_dataset_section(layout)
        self._create_concept_section(layout)
        self._create_workflow_section(layout)
        layout.addStretch()

    # 현재 Regression 실습에 사용하는 전처리 결과를 표시
    def _create_dataset_section(self, layout: QVBoxLayout) -> None:
        header = QHBoxLayout()
        header.setSpacing(SPACE_SM)
        header.addWidget(self._create_section_badge("Current Dataset"))
        self.current_dataset_label = QLabel()
        self.current_dataset_label.setObjectName("bodyText")
        self.current_dataset_label.setWordWrap(True)
        header.addWidget(self.current_dataset_label, 1)
        layout.addLayout(header)

        self.dataset_info_label = QLabel()
        self.dataset_info_label.setObjectName("preprocessingInfoCard")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)

        self.status_label = QLabel()
        self.status_label.setObjectName("smallText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # 회귀 모델과 평가에 필요한 개념 버튼을 구성
    def _create_concept_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Concept"), alignment=Qt.AlignmentFlag.AlignLeft)

        buttons = QGridLayout()
        buttons.setSpacing(SPACE_SM)
        self.concept_group = QButtonGroup(self)
        self.concept_group.setExclusive(True)
        self.concept_buttons: dict[str, QPushButton] = {}
        column_count = 4

        for index, concept in enumerate(CONCEPT_DESCRIPTIONS):
            button = QPushButton(CONCEPT_LABELS[concept])
            button.setObjectName("conceptButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, name=concept: self._show_concept(name))
            self.concept_group.addButton(button)
            self.concept_buttons[concept] = button
            buttons.addWidget(button, index // column_count, index % column_count)

        for column in range(column_count):
            buttons.setColumnStretch(column, 1)
        layout.addLayout(buttons)

        detail = QFrame()
        detail.setObjectName("conceptDetailCard")
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        self.concept_detail_label = QLabel()
        self.concept_detail_label.setObjectName("bodyText")
        self.concept_detail_label.setWordWrap(True)
        detail_layout.addWidget(self.concept_detail_label)
        layout.addWidget(detail)

        self.concept_buttons["Regression"].setChecked(True)
        self._show_concept("Regression")

    # Model, Prediction, Evaluation, Visualization 단계 화면을 구성
    def _create_workflow_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Regression Workflow"), alignment=Qt.AlignmentFlag.AlignLeft)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(SPACE_SM)
        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.step_buttons: list[QPushButton] = []
        self.step_pages = [
            self._create_model_step(),
            self._create_prediction_step(),
            self._create_evaluation_step(),
            self._create_visualization_step(),
        ]

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        for index, (name, page) in enumerate(zip(STEP_NAMES, self.step_pages)):
            button = QPushButton(name)
            button.setObjectName("stepButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page_index=index: self._show_step(page_index))
            self.step_group.addButton(button)
            self.step_buttons.append(button)
            button_layout.addWidget(button, 1)
            container_layout.addWidget(page)

        layout.addLayout(button_layout)
        layout.addWidget(container, alignment=Qt.AlignmentFlag.AlignTop)
        self.step_buttons[0].setChecked(True)
        self._show_step(0)

    # 회귀 모델과 Feature, Hyperparameter를 선택하고 학습하는 화면을 구성
    def _create_model_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Model Training",
            "사용할 Feature와 회귀 모델을 선택한 뒤 Train Data로 모델을 학습한다.",
        )

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)
        controls.addWidget(self._create_control_label("Model"))
        self.model_combo = ChevronComboBox()
        self.model_combo.setObjectName("dataControl")
        self.model_combo.addItem("Linear Regression", "linear")
        self.model_combo.addItem("Polynomial Regression", "polynomial")
        self.model_combo.addItem("Ridge", "ridge")
        self.model_combo.addItem("Lasso", "lasso")
        self.model_combo.addItem("ElasticNet", "elastic_net")
        self.model_combo.currentIndexChanged.connect(self._update_model_controls)
        controls.addWidget(self.model_combo, 1)

        self.degree_label = self._create_control_label("degree")
        controls.addWidget(self.degree_label)
        self.degree_spin = ChevronSpinBox()
        self.degree_spin.setObjectName("dataControl")
        self.degree_spin.setRange(2, 5)
        self.degree_spin.setValue(2)
        self.degree_spin.valueChanged.connect(self._model_setting_changed)
        controls.addWidget(self.degree_spin)

        self.alpha_label = self._create_control_label("alpha")
        controls.addWidget(self.alpha_label)
        self.alpha_spin = ChevronDoubleSpinBox()
        self.alpha_spin.setObjectName("dataControl")
        self.alpha_spin.setRange(0.0001, 1000000.0)
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.valueChanged.connect(self._model_setting_changed)
        controls.addWidget(self.alpha_spin)

        self.l1_ratio_label = self._create_control_label("l1_ratio")
        controls.addWidget(self.l1_ratio_label)
        self.l1_ratio_spin = ChevronDoubleSpinBox()
        self.l1_ratio_spin.setObjectName("dataControl")
        self.l1_ratio_spin.setRange(0.0, 1.0)
        self.l1_ratio_spin.setDecimals(2)
        self.l1_ratio_spin.setSingleStep(0.1)
        self.l1_ratio_spin.setValue(0.5)
        self.l1_ratio_spin.valueChanged.connect(self._model_setting_changed)
        controls.addWidget(self.l1_ratio_spin)

        self.train_button = QPushButton("모델 학습")
        self.train_button.setObjectName("dataButton")
        self.train_button.clicked.connect(self._train_model)
        controls.addWidget(self.train_button)
        layout.addLayout(controls)

        self.model_guide_label = QLabel()
        self.model_guide_label.setObjectName("smallText")
        self.model_guide_label.setWordWrap(True)
        layout.addWidget(self.model_guide_label)

        feature_card = QFrame()
        feature_card.setObjectName("preprocessingCard")
        feature_layout = QVBoxLayout(feature_card)
        feature_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        feature_layout.setSpacing(SPACE_XS)
        feature_title = QLabel("Feature 선택")
        feature_title.setObjectName("encodingTitle")
        feature_layout.addWidget(feature_title)
        self.feature_guide_label = QLabel()
        self.feature_guide_label.setObjectName("smallText")
        self.feature_guide_label.setWordWrap(True)
        feature_layout.addWidget(self.feature_guide_label)
        self.feature_grid = QGridLayout()
        self.feature_grid.setSpacing(SPACE_SM)
        feature_layout.addLayout(self.feature_grid)
        layout.addWidget(feature_card)

        self.model_result_label = self._create_result_card()
        self.model_result_label.setText("모델을 학습하기 전입니다.")
        layout.addWidget(self.model_result_label)

        coefficient_card, self.coefficient_table = self._create_table_card("Coefficient")
        self.coefficient_table.setMinimumHeight(180)
        layout.addWidget(coefficient_card)

        code_card, self.train_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        self._update_model_controls()
        return page

    # Test Data 예측값과 Sample별 오차를 표시하는 화면을 구성
    def _create_prediction_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Test Prediction",
            "학습된 모델에 Test Feature를 전달하여 Target을 예측한다.",
        )

        status_layout = QHBoxLayout()
        status_layout.setSpacing(SPACE_SM)
        self.prediction_status_label = QLabel("먼저 Model 단계에서 모델을 학습하세요.")
        self.prediction_status_label.setObjectName("smallText")
        self.prediction_status_label.setWordWrap(True)
        status_layout.addWidget(self.prediction_status_label, 1)

        self.predict_button = QPushButton("Test 예측")
        self.predict_button.setObjectName("dataButton")
        self.predict_button.clicked.connect(self._predict_test)
        status_layout.addWidget(self.predict_button)
        layout.addLayout(status_layout)

        prediction_card, self.prediction_table = self._create_table_card(
            "Actual / Prediction / Error"
        )
        layout.addWidget(prediction_card)

        code_card, self.prediction_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    # Train/Test 모델과 Baseline 평가 지표를 비교하는 화면을 구성
    def _create_evaluation_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Regression Evaluation",
            "MAE, MSE, RMSE, R²를 이용해 실제값과 예측값의 차이를 확인하고 Baseline과 비교한다. "
            "모델과 Hyperparameter는 Validation Data나 Cross Validation으로 비교하고 Test Data는 최종 평가에 사용한다.",
        )

        metric_card, self.metric_table = self._create_table_card("Metrics")
        self.metric_table.setMinimumHeight(220)
        layout.addWidget(metric_card)

        performance_card = QFrame()
        performance_card.setObjectName("preprocessingCard")
        performance_layout = QVBoxLayout(performance_card)
        performance_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        performance_title = QLabel("Performance Interpretation")
        performance_title.setObjectName("encodingTitle")
        performance_layout.addWidget(performance_title)
        self.performance_label = QLabel(
            "Test 예측을 실행하면 일반화 성능과 Train/Test 결과를 함께 확인할 수 있습니다."
        )
        self.performance_label.setObjectName("bodyText")
        self.performance_label.setWordWrap(True)
        performance_layout.addWidget(self.performance_label)
        layout.addWidget(performance_card)

        code_card, self.evaluation_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    # 실제값·예측값, 회귀선, 잔차 그래프를 구성
    def _create_visualization_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Regression Visualization",
            "Matplotlib을 이용해 모델의 예측 결과와 오차를 그래프로 확인한다.",
        )

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)
        controls.addWidget(self._create_control_label("Graph"))
        self.graph_combo = ChevronComboBox()
        self.graph_combo.setObjectName("dataControl")
        self.graph_combo.addItems(["Actual vs Prediction", "Regression Line", "Residual Plot"])
        self.graph_combo.currentTextChanged.connect(self._update_graph_controls)
        controls.addWidget(self.graph_combo, 1)

        self.graph_feature_label = self._create_control_label("Feature")
        controls.addWidget(self.graph_feature_label)
        self.graph_feature_combo = ChevronComboBox()
        self.graph_feature_combo.setObjectName("dataControl")
        controls.addWidget(self.graph_feature_combo, 1)

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
        self.canvas.setMinimumHeight(320)
        self.canvas.installEventFilter(self)
        self.canvas.hide()
        layout.addWidget(self.canvas)
        layout.addStretch()
        self._update_graph_controls()
        return page

    # Preprocessing에서 전달한 결과를 현재 Regression 데이터로 설정
    def set_preprocessing_result(
        self,
        result: PreprocessingResult,
        use_default_notice: bool = False,
    ) -> None:
        self.dataset_name = result.dataset_name
        self.target_column = result.target_column
        self.task_type = result.task_type
        self.x_train = result.x_train.copy(deep=True)
        self.x_test = result.x_test.copy(deep=True)
        self.y_train = result.y_train.copy(deep=True)
        self.y_test = result.y_test.copy(deep=True)
        self.preprocessing_artifacts = result.artifacts

        if use_default_notice:
            self.current_dataset_label.setText(
                "선택된 전처리 결과가 없어 프로그램에 포함된 Regression Sample을 사용합니다."
            )
        else:
            self.current_dataset_label.setText(result.dataset_name)

        feature_count = len(self.x_train.columns)
        target_text = self.target_column or "없음"
        self.dataset_info_label.setText(
            f"<b>{escape(self.dataset_name)}</b> · Target: {escape(str(target_text))}<br>"
            f"Train: {len(self.x_train)} Samples · Test: {len(self.x_test)} Samples · "
            f"Features: {feature_count}"
        )
        self._populate_feature_checkboxes()
        self._invalidate_model()
        self._update_model_guide()

        if self.task_type != "regression":
            self.status_label.setText(
                "현재 Dataset은 회귀 Dataset이 아닙니다. Data Lab에서 회귀 Dataset을 선택해 주세요."
            )
        elif self.x_train.empty or self.x_test.empty:
            self.status_label.setText(
                "Preprocessing의 1. Train/Test에서 데이터를 분리한 뒤 Regression으로 이동해 주세요."
            )
        elif not pd.api.types.is_numeric_dtype(self.y_train):
            self.status_label.setText("회귀 Target은 숫자형이어야 합니다.")
        else:
            self.status_label.setText(
                "사용할 Feature와 Model을 선택한 뒤 모델 학습을 눌러주세요."
            )

        self._update_train_button()

    # 프로그램에 포함된 Regression Sample을 초기 실습 데이터로 설정
    def _load_default_dataset(self) -> None:
        dataframe = load_builtin_dataset("Regression Sample")
        target = "price"
        features = [column for column in dataframe.columns if column != target]
        x_train, x_test, y_train, y_test = split_dataset(
            dataframe,
            features,
            target,
            test_size=0.25,
            random_state=42,
            use_stratify=False,
        )
        self.set_preprocessing_result(
            PreprocessingResult(
                dataset_name="Regression Sample",
                target_column=target,
                task_type="regression",
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
            ),
            use_default_notice=True,
        )

    # 현재 Feature Column을 선택 가능한 CheckBox로 표시
    def _populate_feature_checkboxes(self) -> None:
        while self.feature_grid.count():
            item = self.feature_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.feature_checkboxes.clear()
        excluded = []
        for index, column in enumerate(self.x_train.columns):
            checkbox = QCheckBox(str(column))
            checkbox.setObjectName("dataCheckBox")
            is_identifier = str(column).lower() == "id" or str(column).lower().endswith("_id")
            checkbox.setChecked(not is_identifier)
            checkbox.toggled.connect(self._feature_selection_changed)
            self.feature_checkboxes[str(column)] = checkbox
            self.feature_grid.addWidget(checkbox, index // 4, index % 4)
            if is_identifier:
                excluded.append(str(column))

        if excluded:
            self.feature_guide_label.setText(
                "식별자 성격의 Feature는 기본 선택에서 제외했습니다: " + ", ".join(excluded)
            )
        else:
            self.feature_guide_label.setText(
                "모델 학습에 사용할 Feature를 선택하세요. 하나 이상의 Feature가 필요합니다."
            )

    # 선택한 Feature 이름을 원래 Column 순서로 반환
    def _selected_features(self) -> list[str]:
        return [
            str(column)
            for column in self.x_train.columns
            if str(column) in self.feature_checkboxes
            and self.feature_checkboxes[str(column)].isChecked()
        ]

    # Model과 Hyperparameter 설정에 맞게 입력 영역을 갱신
    def _update_model_controls(self, _index: int = -1) -> None:
        model_name = self.model_combo.currentData()
        uses_degree = model_name == "polynomial"
        uses_alpha = model_name in {"ridge", "lasso", "elastic_net"}
        uses_l1_ratio = model_name == "elastic_net"
        self.degree_label.setVisible(uses_degree)
        self.degree_spin.setVisible(uses_degree)
        self.alpha_label.setVisible(uses_alpha)
        self.alpha_spin.setVisible(uses_alpha)
        self.l1_ratio_label.setVisible(uses_l1_ratio)
        self.l1_ratio_spin.setVisible(uses_l1_ratio)
        self._model_setting_changed()
        self._update_model_guide()

    def _model_setting_changed(self, _value=None) -> None:
        if hasattr(self, "prediction_status_label"):
            self._invalidate_model("Model 설정이 변경되었습니다. 다시 학습해 주세요.")
        self._update_model_guide()

    def _feature_selection_changed(self, _checked: bool = False) -> None:
        self._invalidate_model("Feature 선택이 변경되었습니다. 다시 학습해 주세요.")
        self._update_train_button()
        self._update_model_guide()

    # 선택한 모델에서 중요한 전처리와 복잡도 조건을 안내
    def _update_model_guide(self) -> None:
        if not hasattr(self, "model_guide_label"):
            return

        model_key = self.model_combo.currentData()
        messages = []
        if model_key in {"ridge", "lasso", "elastic_net"}:
            if self.preprocessing_artifacts.scaler is None:
                messages.append(
                    "Ridge, Lasso, ElasticNet은 Feature 크기의 영향을 받으므로 "
                    "Preprocessing에서 Scaling을 적용하는 것이 일반적입니다."
                )
            else:
                messages.append("Preprocessing에서 학습한 Scaling 기준이 함께 전달되었습니다.")

        if model_key == "polynomial":
            feature_count = len(self._selected_features())
            degree = self.degree_spin.value()
            expanded_count = comb(feature_count + degree, degree) - 1 if feature_count else 0
            messages.append(
                f"현재 설정은 {feature_count}개 Feature를 {expanded_count}개 다항 Feature로 확장합니다."
            )
            if expanded_count >= len(self.x_train) and len(self.x_train):
                messages.append(
                    "생성되는 Feature 수가 Train Sample 수 이상이므로 과적합 가능성을 확인해 주세요."
                )

        if not messages:
            messages.append(
                "모델과 Hyperparameter 선택은 이후 Model Selection에서 Cross Validation으로 비교합니다."
            )
        self.model_guide_label.setText("<br>".join(messages))

    # 현재 데이터와 Feature 선택으로 모델을 학습할 수 있는지 갱신
    def _update_train_button(self) -> None:
        can_train = (
            self.task_type == "regression"
            and not self.x_train.empty
            and not self.x_test.empty
            and bool(self._selected_features())
            and pd.api.types.is_numeric_dtype(self.y_train)
        )
        self.train_button.setEnabled(can_train)

    # 선택한 모델을 Train Data에 fit하고 계수와 절편을 표시
    def _train_model(self) -> None:
        features = self._selected_features()
        if not features:
            self.status_label.setText("하나 이상의 Feature를 선택해 주세요.")
            return

        self._invalidate_model()
        train_x = self.x_train.loc[:, features]
        test_x = self.x_test.loc[:, features]
        try:
            validate_regression_data(train_x, self.y_train)
            validate_regression_data(test_x, self.y_test)
            self.model = create_regressor(
                self.model_combo.currentData(),
                alpha=self.alpha_spin.value(),
                degree=self.degree_spin.value(),
                l1_ratio=self.l1_ratio_spin.value(),
            )
            self.model.fit(train_x, self.y_train)
        except (TypeError, ValueError) as error:
            self.model = None
            self.status_label.setText(f"모델 학습 실패: {error}")
            return

        self.train_prediction = self.model.predict(train_x)
        model_key = self.model_combo.currentData()
        model_name = self.model_combo.currentText()
        estimator = self.model
        coefficient_features = features
        if model_key == "polynomial":
            polynomial = self.model.named_steps["polynomial"]
            estimator = self.model.named_steps["linear"]
            coefficient_features = polynomial.get_feature_names_out(features).tolist()

        coefficient_data = pd.DataFrame({
            "Feature": coefficient_features,
            "Coefficient": np.asarray(estimator.coef_, dtype=float),
        })
        self._populate_table(self.coefficient_table, coefficient_data)
        result_lines = [
            f"<b>{escape(model_name)}</b><br>"
            f"Intercept: {float(estimator.intercept_):.3f}<br>"
            f"Features: {escape(', '.join(features))}"
        ]
        if model_key == "polynomial":
            result_lines.append(f"<br>생성된 다항 Feature: {len(coefficient_features)}개")
        self.model_result_label.setText("".join(result_lines))

        if model_key == "linear":
            model_code = "from sklearn.linear_model import LinearRegression\n\nmodel = LinearRegression()"
        elif model_key == "polynomial":
            model_code = (
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import PolynomialFeatures\n"
                "from sklearn.linear_model import LinearRegression\n\n"
                "model = Pipeline([\n"
                f"    ('polynomial', PolynomialFeatures(degree={self.degree_spin.value()}, include_bias=False)),\n"
                "    ('linear', LinearRegression()),\n"
                "])"
            )
        elif model_key == "ridge":
            model_code = (
                "from sklearn.linear_model import Ridge\n\n"
                f"model = Ridge(alpha={self.alpha_spin.value():g})"
            )
        elif model_key == "lasso":
            model_code = (
                "from sklearn.linear_model import Lasso\n\n"
                f"model = Lasso(alpha={self.alpha_spin.value():g}, max_iter=10000)"
            )
        else:
            model_code = (
                "from sklearn.linear_model import ElasticNet\n\n"
                "model = ElasticNet(\n"
                f"    alpha={self.alpha_spin.value():g}, l1_ratio={self.l1_ratio_spin.value():g},\n"
                "    max_iter=10000,\n"
                ")"
            )
        self.train_code_label.setText(
            f"features = {features!r}\n"
            f"{model_code}\n"
            "model.fit(X_train[features], y_train)\n\n"
            "coefficient = model.coef_\n"
            "intercept = model.intercept_"
        )
        self.status_label.setText(
            f"{model_name} 모델 학습을 완료했습니다. Prediction 단계에서 Test Data를 예측해 보세요."
        )
        self.prediction_status_label.setText("학습된 모델로 Test Data를 예측할 수 있습니다.")
        self.step_buttons[1].setEnabled(True)
        self.predict_button.setEnabled(True)

    # 학습된 모델로 Test Data를 예측하고 평가 결과를 생성
    def _predict_test(self) -> None:
        if self.model is None:
            self.prediction_status_label.setText("먼저 Model 단계에서 모델을 학습하세요.")
            return

        features = self._selected_features()
        self.test_prediction = self.model.predict(self.x_test.loc[:, features])
        actual_values = self.y_test.to_numpy(dtype=float)
        errors = actual_values - self.test_prediction
        prediction_data = pd.DataFrame({
            "Sample": self.y_test.index.astype(str),
            "Actual": actual_values,
            "Prediction": self.test_prediction,
            "Error": errors,
        })
        self._populate_table(self.prediction_table, prediction_data)

        self.train_metrics = calculate_regression_metrics(self.y_train, self.train_prediction)
        self.test_metrics = calculate_regression_metrics(self.y_test, self.test_prediction)
        baseline_prediction = create_mean_baseline(self.y_train, len(self.y_test))
        self.baseline_metrics = calculate_regression_metrics(self.y_test, baseline_prediction)
        self._show_metrics()
        self._update_performance_interpretation()

        self.prediction_code_label.setText(
            "prediction = model.predict(X_test[features])\n\n"
            "result = pd.DataFrame({\n"
            "    'Actual': y_test,\n"
            "    'Prediction': prediction,\n"
            "})\n"
            "result['Error'] = result['Actual'] - result['Prediction']"
        )
        self.evaluation_code_label.setText(
            "from sklearn.metrics import (\n"
            "    mean_absolute_error, mean_squared_error,\n"
            "    root_mean_squared_error, r2_score,\n"
            ")\n\n"
            "mae = mean_absolute_error(y_test, prediction)\n"
            "mse = mean_squared_error(y_test, prediction)\n"
            "rmse = root_mean_squared_error(y_test, prediction)\n"
            "r2 = r2_score(y_test, prediction)\n\n"
            "baseline_prediction = np.full(len(y_test), y_train.mean())"
        )
        self.prediction_status_label.setText(
            f"Test {len(self.y_test)} Samples의 예측을 완료했습니다."
        )
        self.status_label.setText("Test 예측과 회귀 모델 평가를 완료했습니다.")
        self.step_buttons[2].setEnabled(True)
        self.step_buttons[3].setEnabled(True)
        self._update_graph_controls()

    # Train/Test 모델과 Test Baseline 지표를 표에 표시
    def _show_metrics(self) -> None:
        if not self.train_metrics or not self.test_metrics or not self.baseline_metrics:
            return

        metric_data = pd.DataFrame({
            "Metric": ["MAE", "MSE", "RMSE", "R²"],
            "Train Model": self._metric_values(self.train_metrics),
            "Test Model": self._metric_values(self.test_metrics),
            "Test Baseline": self._metric_values(self.baseline_metrics),
        })
        self._populate_table(self.metric_table, metric_data)

    def _metric_values(self, metrics: RegressionMetrics) -> list[float]:
        return [metrics.mae, metrics.mse, metrics.rmse, metrics.r2]

    # Train/Test 차이와 Baseline 비교를 해석하되 고정 기준으로 진단하지 않음
    def _update_performance_interpretation(self) -> None:
        if not self.train_metrics or not self.test_metrics or not self.baseline_metrics:
            return

        messages = [
            "Test Data는 모델 학습에 사용하지 않은 데이터이므로 최종 일반화 성능을 확인하는 데 사용한다.",
            f"Train RMSE는 {self.train_metrics.rmse:.3f}이고 Test RMSE는 {self.test_metrics.rmse:.3f}이다.",
        ]
        if self.test_metrics.rmse > self.train_metrics.rmse:
            messages.append(
                "Test 오차가 Train 오차보다 크지만 이 차이만으로 과적합이라고 단정하지 않고 반복 분할이나 Cross Validation에서도 같은 경향이 나타나는지 확인한다."
            )
        else:
            messages.append(
                "한 번의 분할에서는 Test 오차가 Train 오차보다 크지 않지만 다른 데이터 분할에서도 같은 결과가 나오는지 확인한다."
            )

        if self.test_metrics.mae >= self.baseline_metrics.mae:
            messages.append(
                "Test MAE가 Baseline보다 작지 않으므로 모델이 충분한 관계를 학습했는지 확인해야 한다."
            )
        else:
            messages.append(
                "Test MAE가 Train Target 평균을 사용하는 Baseline보다 작다."
            )

        if len(self.y_test) < 2:
            messages.append(
                "Test Sample이 1개이므로 R²를 계산하지 않는다. R²를 확인하려면 Test Sample이 2개 이상 필요하다."
            )

        messages.append(
            "모델과 Hyperparameter를 반복 비교할 때는 Test 결과가 아니라 Validation Data나 Cross Validation을 사용한다."
        )
        self.performance_label.setText("<br>".join(messages))

    # 그래프 종류에 따라 Feature 입력과 안내 문구를 갱신
    def _update_graph_controls(self, _text: str = "") -> None:
        graph_type = self.graph_combo.currentText()
        uses_feature = graph_type == "Regression Line"
        self.graph_feature_label.setVisible(uses_feature)
        self.graph_feature_combo.setVisible(uses_feature)
        self.graph_feature_combo.clear()
        self.graph_feature_combo.addItems(self._selected_features())

        description = GRAPH_DESCRIPTIONS.get(graph_type, "")
        selected_features = self._selected_features()
        line_available = len(selected_features) == 1
        self.graph_feature_combo.setEnabled(not uses_feature or line_available)
        if uses_feature and not line_available:
            description += (
                "<br>Regression Line은 하나의 Feature로 학습한 모델에서 확인할 수 있습니다. "
                "<br>Model 단계에서 하나의 Feature만 선택한 뒤 다시 학습해 주세요."
            )

        self.graph_description_label.setText(description)
        self.draw_button.setEnabled(
            self.test_prediction is not None and (not uses_feature or line_available)
        )

    # 선택한 회귀 결과 그래프를 Matplotlib Canvas에 그림
    def _draw_graph(self) -> None:
        if self.model is None or self.test_prediction is None:
            return

        graph_type = self.graph_combo.currentText()
        actual = self.y_test.to_numpy(dtype=float)
        prediction = np.asarray(self.test_prediction, dtype=float)

        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.set_facecolor(COLOR_OFF_WHITE)

        if graph_type == "Actual vs Prediction":
            axes.scatter(actual, prediction, color=COLOR_SECONDARY)
            minimum = min(actual.min(), prediction.min())
            maximum = max(actual.max(), prediction.max())
            axes.plot([minimum, maximum], [minimum, maximum], color=COLOR_OFF_BLACK, linestyle="--")
            axes.set_xlabel("Actual")
            axes.set_ylabel("Prediction")
        elif graph_type == "Residual Plot":
            residual = actual - prediction
            axes.scatter(prediction, residual, color=COLOR_SECONDARY)
            axes.axhline(0, color=COLOR_OFF_BLACK, linestyle="--")
            axes.set_xlabel("Prediction")
            axes.set_ylabel("Residual")
        elif graph_type == "Regression Line" and len(self._selected_features()) == 1:
            features = self._selected_features()
            feature = self.graph_feature_combo.currentText()
            if not feature or feature not in features:
                return

            train_x = self.x_train[feature].astype(float)
            test_x = self.x_test[feature].astype(float)
            all_x = pd.concat([train_x, test_x])
            line_x = np.linspace(all_x.min(), all_x.max(), 200)
            line_frame = pd.DataFrame({feature: line_x})
            line_y = self.model.predict(line_frame)
            axes.scatter(train_x, self.y_train.astype(float), color=COLOR_SECONDARY, label="Train")
            axes.scatter(test_x, self.y_test.astype(float), color=COLOR_PRIMARY, label="Test")
            axes.plot(line_x, line_y, color=COLOR_OFF_BLACK, label="Regression Line")
            axes.set_xlabel(feature)
            axes.set_ylabel(str(self.target_column))
            axes.legend()

        axes.set_title(graph_type, color=COLOR_OFF_BLACK)
        axes.xaxis.label.set_color(COLOR_OFF_BLACK)
        axes.yaxis.label.set_color(COLOR_OFF_BLACK)
        axes.tick_params(colors=COLOR_OFF_BLACK)
        for spine in axes.spines.values():
            spine.set_color(COLOR_PRIMARY)

        self.canvas.show()
        self.canvas.draw()

    # 설정 변경 시 기존 모델과 이후 결과를 초기화
    def _invalidate_model(self, message: str | None = None) -> None:
        self.model = None
        self.train_prediction = None
        self.test_prediction = None
        self.train_metrics = None
        self.test_metrics = None
        self.baseline_metrics = None

        self.model_result_label.setText("모델을 학습하기 전입니다.")
        self.prediction_status_label.setText("먼저 Model 단계에서 모델을 학습하세요.")
        self.performance_label.setText(
            "Test 예측을 실행하면 일반화 성능과 Train/Test 결과를 함께 확인할 수 있습니다."
        )
        self.train_code_label.clear()
        self.prediction_code_label.clear()
        self.evaluation_code_label.clear()
        self._clear_table(self.coefficient_table)
        self._clear_table(self.prediction_table)
        self._clear_table(self.metric_table)
        self.figure.clear()
        self.canvas.hide()

        if hasattr(self, "step_buttons"):
            for button in self.step_buttons[1:]:
                button.setEnabled(False)
            self.predict_button.setEnabled(False)
            self._show_step(0)
            self.step_buttons[0].setChecked(True)

        if message:
            self.status_label.setText(message)

    # 선택한 Concept의 설명과 현재 설정을 표시
    def _show_concept(self, concept: str) -> None:
        description = escape(CONCEPT_DESCRIPTIONS[concept])
        if concept != "Intercept":
            description = description.replace("다. ", "다.<br>")
        concept_label = escape(CONCEPT_LABELS[concept].replace("\n", ""))
        current = ""
        if concept == "alpha" and hasattr(self, "alpha_spin"):
            current = f"<br><br>현재 alpha: {self.alpha_spin.value():g}"
        elif concept == "Actual / Prediction" and self.test_prediction is not None:
            current = f"<br><br>현재 Test Prediction: {len(self.test_prediction)}개"
        self.concept_detail_label.setText(
            f"<b>{concept_label}</b>: {description}{current}"
        )

    # 현재 단계 페이지만 표시
    def _show_step(self, index: int) -> None:
        for page_index, page in enumerate(self.step_pages):
            page.setHidden(page_index != index)

    # Canvas 위 Wheel Event를 페이지 Scroll로 전달
    def eventFilter(self, watched, event):
        if watched is self.canvas and event.type() == QEvent.Type.Wheel:
            delta = event.pixelDelta().y() or event.angleDelta().y()
            scroll_bar = self.scroll_area.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.value() - delta)
            return True
        return super().eventFilter(watched, event)

    # 공통 Section Badge를 생성
    def _create_section_badge(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionBadge")
        label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        return label

    # 단계 제목과 설명이 포함된 기본 페이지를 생성
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

    # 제목과 Table로 구성된 Card를 생성
    def _create_table_card(self, title: str) -> tuple[QFrame, QTableWidget]:
        card = QFrame()
        card.setObjectName("preprocessingCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_XS)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)

        table = QTableWidget()
        table.setObjectName("dataPreviewTable")
        table.setMinimumHeight(260)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setHighlightSections(False)
        layout.addWidget(table)
        return card, table

    # scikit-learn 사용 코드를 표시할 Block을 생성
    def _create_code_block(self) -> tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("codeBlockCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_XS)
        title = QLabel("Python 코드")
        title.setObjectName("codeBlockTitle")
        layout.addWidget(title)
        label = QLabel()
        label.setObjectName("codeBlock")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)
        return card, label

    # DataFrame의 모든 행과 열을 Table에 표시
    def _populate_table(self, table: QTableWidget, dataframe: pd.DataFrame) -> None:
        table.clear()
        table.setRowCount(len(dataframe))
        table.setColumnCount(len(dataframe.columns))
        table.setHorizontalHeaderLabels([str(column) for column in dataframe.columns])
        for row_index in range(len(dataframe)):
            for column_index in range(len(dataframe.columns)):
                value = dataframe.iloc[row_index, column_index]
                if pd.isna(value):
                    text = "계산 불가"
                elif isinstance(value, (float, np.floating)):
                    text = f"{float(value):.3f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, column_index, item)

    def _clear_table(self, table: QTableWidget) -> None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)
