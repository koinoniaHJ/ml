"""분류 모델을 학습하고 예측·평가·시각화하는 Classification Lab 화면을 구성한다."""

from html import escape
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY,
    SPACE_MD, SPACE_SM, SPACE_XS,
)
from ml.classification import (
    ClassificationMetrics, calculate_classification_metrics,
    create_classification_report, create_classifier, create_confusion_frame,
    create_majority_baseline, validate_classification_data,
)
from ml.data_loader import load_builtin_dataset
from ml.preprocessing import (
    PreprocessingArtifacts, PreprocessingResult, one_hot_encode_feature,
    split_dataset,
)
from ui.widgets import ChevronComboBox, ChevronDoubleSpinBox, ChevronSpinBox


# 글의 학습 순서를 유지하면서 공통 평가 개념과 모델별 개념을 함께 제공한다.
CONCEPT_DESCRIPTIONS = {
    "Classification": "정해진 범주 중 하나를 Target으로 예측하는 머신러닝 문제다. 이진 분류는 두 Class를 구분하고 다중 분류는 세 개 이상의 Class를 구분한다.",
    "Actual / Prediction": "Actual은 Test Target의 실제 Class이고 Prediction은 모델이 예측한 Class다. 분류 평가는 두 값을 비교하는 데서 시작한다.",
    "Accuracy": "전체 Sample 중 실제 Class를 올바르게 예측한 Sample의 비율이다. Accuracy는 올바르게 예측한 Sample 수를 전체 Sample 수로 나누어 계산한다. Class별 Sample 수가 크게 다르면 Accuracy만으로 성능을 판단하기 어렵다.",
    "Class Imbalance": "Class별 Sample 수가 크게 다른 상태다. 다수 Class만 예측해도 Accuracy가 높아질 수 있으므로 다른 평가 지표를 함께 확인한다.",
    "Confusion Matrix": "실제 Class와 예측 Class의 조합별 Sample 수를 나타낸 표다. 행은 실제 Class이고 열은 예측 Class다.",
    "TP / TN / FP / FN": "이진 분류에서 TP와 TN은 올바른 예측이고 FP와 FN은 서로 다른 종류의 오답이다. 어떤 Class를 Positive로 정했는지에 따라 값의 의미가 달라진다.",
    "Precision": "Positive로 예측한 Sample 중 실제로 Positive인 비율이며 TP / (TP + FP)로 계산한다. FP를 줄이는 것이 중요한 문제에서 주의 깊게 확인한다.",
    "Recall": "실제 Positive Sample 중 Positive로 올바르게 예측한 비율이며 TP / (TP + FN)으로 계산한다. FN을 줄이는 것이 중요한 문제에서 주의 깊게 확인한다.",
    "F1 Score": "Precision과 Recall의 조화평균이며 2 × Precision × Recall / (Precision + Recall)로 계산한다. 한쪽 값만 높고 다른 값이 낮으면 F1도 낮아진다.",
    "Classification Report": "각 Class의 Precision, Recall, F1 Score와 Support를 함께 표시하는 결과표다. 새로운 평가 기준이 아니라 기존 지표를 Class별로 모아 보여준다.",
    "Support": "평가 데이터에 실제로 포함된 각 Class의 Sample 수다.",
    "Macro / Weighted Avg": "Macro Avg는 각 Class 지표를 같은 비중으로 평균하고 Weighted Avg는 각 Class의 Support를 반영해 평균한다.",
    "Baseline": "모델 성능과 비교하기 위한 간단한 기준이다. 이 화면에서는 Train Data의 최빈 Class를 모든 Sample에 예측한다.",
    "Generalization": "모델이 학습하지 않은 새로운 데이터에서도 적절한 예측을 만드는 능력이다. 한 번의 작은 Test 결과만으로 항상 같은 성능이 나온다고 단정하지 않는다.",
    "Overfitting": "Train Data에서는 좋은 결과를 보이지만 새로운 데이터에서는 성능이 떨어지는 상태다. 깊은 Tree처럼 복잡한 모델에서 Train과 Test 결과 차이를 확인한다.",
    "Underfitting": "모델이 데이터의 관계를 충분히 학습하지 못해 Train과 Test 모두에서 성능이 낮은 상태다. 지나치게 얕은 Tree에서도 발생할 수 있다.",
    "Hyperparameter": "모델이 학습으로 구하는 값과 달리 학습 전에 사용자가 정하는 설정값이다. Test Data를 반복해서 보며 고르지 않고 Validation Data나 Cross Validation으로 비교한다.",
    "Logistic Regression": "Feature로 각 Class에 속할 가능성을 계산해 Class를 예측하는 분류 알고리즘이다. 이름에 Regression이 들어가지만 scikit-learn의 LogisticRegression은 Classifier다.",
    "Probability": "predict_proba()가 반환하는 각 Class의 예측 확률이다. 열 순서는 모델의 classes_와 같으며 예측 확률을 확정적인 사실로 해석하지 않는다. 확률을 직접 제공하지 않는 모델은 별도 확률 보정이 필요할 수 있다.",
    "Threshold": "이진 분류에서 Positive Class로 판단할 최소 점수나 확률 기준이다. 기준을 바꾸면 Precision, Recall, F1과 Confusion Matrix가 달라질 수 있다.",
    "Gaussian Naive Bayes": "각 Class에서 숫자형 Feature가 정규분포를 따른다고 가정하고, Class가 주어졌을 때 Feature들이 조건부 독립이라고 가정해 확률을 계산하는 분류 알고리즘이다.",
    "KNN": "새로운 Sample과 가까운 Train Sample들을 확인하고 분류에서는 가장 많이 나타난 Class로 예측하는 알고리즘이다. 회귀에는 KNeighborsRegressor를 사용한다.",
    "Distance": "KNN이 가까운 이웃을 찾을 때 사용하는 Sample 사이의 거리다. 기본 Minkowski 거리에서 p=2이면 Euclidean Distance에 해당한다.",
    "n_neighbors": "KNN이 예측할 때 확인할 이웃의 수를 정하는 Hyperparameter다. 너무 작으면 가까운 일부 Sample에 민감하고 너무 크면 지역적인 특징이 약해질 수 있다. 이진 분류에서는 동률을 줄이기 위해 홀수를 자주 사용하지만 모든 동률을 보장해서 없애는 것은 아니다.",
    "Scaling": "Feature 값의 크기를 일정한 기준으로 변환하는 과정이다. 거리나 경계에 값의 크기가 영향을 주는 Logistic Regression, KNN, SVM에서는 일반적으로 적용을 고려한다.",
    "Decision Tree": "Feature에 조건을 적용해 데이터를 반복해서 나누고 도착한 Leaf에서 Class를 예측하는 알고리즘이다. 회귀에는 DecisionTreeRegressor를 사용한다.",
    "Node / Branch / Leaf": "Node는 조건을 적용하는 지점이고 Branch는 조건 결과에 따른 경로이며 Leaf는 최종 예측을 결정하는 지점이다. Tree 그림의 samples는 Node에 도달한 Train Sample 수, value는 Class별 Sample 수, class는 해당 Node의 예측 Class다.",
    "Gini Impurity": "한 Node에 서로 다른 Class가 얼마나 섞여 있는지 나타내는 불순도다. 하나의 Class만 있으면 0이며 DecisionTreeClassifier의 기본 분할 기준이다.",
    "max_depth": "Tree가 조건을 나누며 내려갈 수 있는 최대 깊이다. 너무 깊으면 과적합되고 너무 얕으면 과소적합될 수 있다.",
    "Random Forest": "Train Data의 부분집합과 일부 Feature를 이용해 여러 Decision Tree를 학습하고 각 Tree의 Class 확률을 평균해 예측하는 Ensemble 알고리즘이다. 회귀에는 RandomForestRegressor를 사용한다.",
    "Ensemble": "여러 모델의 예측 결과를 결합해 하나의 최종 예측을 만드는 방법이다.",
    "Bootstrap": "복원 추출로 원래 Train Sample 수만큼 다시 뽑아 각 Tree의 학습 데이터를 만드는 방법이다. RandomForestClassifier는 기본적으로 Bootstrap Sampling을 사용한다.",
    "Feature Importance": "Tree 분할에서 각 Feature가 불순도 감소에 기여한 상대적 값을 나타낸다. 인과관계를 뜻하지 않으며 고유한 값이 많은 Feature에 편향될 수 있어 참고값으로 해석한다.",
    "Boosting": "약한 모델을 순서대로 추가하며 앞 단계에서 부족했던 부분을 보완하는 Ensemble 방식이다. Random Forest처럼 여러 Tree를 사용할 수 있지만 학습 순서가 다르다.",
    "Gradient Boosting": "작은 Tree를 순서대로 추가하면서 이전 모델의 손실을 줄이는 방향으로 학습한다. 분류에는 GradientBoostingClassifier, 회귀에는 GradientBoostingRegressor를 사용한다.",
    "XGBoost": "Gradient Boosting을 확장해 규제와 학습 최적화 기능을 제공한다. scikit-learn 기본 모델이 아니므로 xgboost 패키지를 별도로 설치하며 XGBClassifier와 XGBRegressor를 사용한다. n_estimators, learning_rate, max_depth 등을 조절한다.",
    "LightGBM": "Boosting 기반으로 대규모 데이터의 빠른 학습과 메모리 효율을 목표로 한다. lightgbm 패키지를 별도로 설치하며 LGBMClassifier와 LGBMRegressor를 사용한다. 작은 데이터에서는 속도 장점이 크게 드러나지 않을 수 있다.",
    "CatBoost": "Boosting 기반이며 cat_features로 범주형 Feature를 지정해 처리할 수 있다. catboost 패키지를 별도로 설치하며 CatBoostClassifier와 CatBoostRegressor를 사용한다. 숫자 Feature만 사용하면 일반적인 Boosting 모델과 비슷한 입력 형태로 학습할 수 있다.",
    "SVM": "서로 다른 Class를 나누는 Decision Boundary와 가까운 Sample 사이의 여유 공간을 고려하는 알고리즘이다. 분류에는 SVC, 회귀에는 SVR을 사용한다.",
    "Decision Boundary": "Feature 공간에서 서로 다른 Class를 구분하는 경계다. 두 Feature에서는 선이나 곡선으로 시각화할 수 있다.",
    "Margin": "Decision Boundary와 가까운 Train Sample 사이의 여유 공간이다. SVM은 오분류와 Margin을 함께 고려한다.",
    "Support Vector": "Decision Boundary 가까이에 있어 경계와 Margin을 정하는 데 직접 영향을 주는 Train Sample이다. Scaling 후 학습하면 support_vectors_에는 Scaling된 Feature 값이 저장된다.",
    "C": "SVC와 LogisticRegression의 규제 강도에 반비례하는 Hyperparameter다. SVC에서는 값이 클수록 Train 오분류에 더 큰 불이익을 주는 방향으로 작동한다.",
    "Kernel": "SVM이 선형으로 나누기 어려운 데이터에서 다양한 형태의 Decision Boundary를 만들도록 하는 방법이다. SVC는 linear, poly, rbf, sigmoid 등을 제공하고 기본값은 rbf다.",
}

CONCEPT_LABELS = {
    "Classification": "Classification\n(분류)",
    "Actual / Prediction": "Actual / Prediction\n(실제 Class / 예측 Class)",
    "Accuracy": "Accuracy\n(정확도)",
    "Class Imbalance": "Class Imbalance\n(Class 불균형)",
    "Confusion Matrix": "Confusion Matrix\n(혼동 행렬)",
    "TP / TN / FP / FN": "TP / TN / FP / FN\n(이진 분류 결과)",
    "Precision": "Precision\n(정밀도)",
    "Recall": "Recall\n(재현율)",
    "F1 Score": "F1 Score\n(F1 점수)",
    "Classification Report": "Classification Report\n(분류 평가표)",
    "Support": "Support\n(Sample 수)",
    "Macro / Weighted Avg": "Macro / Weighted Avg\n(평균 방식)",
    "Baseline": "Baseline\n(기준 결과)",
    "Generalization": "Generalization\n(일반화)",
    "Overfitting": "Overfitting\n(과적합)",
    "Underfitting": "Underfitting\n(과소적합)",
    "Hyperparameter": "Hyperparameter\n(하이퍼파라미터)",
    "Logistic Regression": "Logistic Regression\n(로지스틱 회귀)",
    "Probability": "Probability\n(예측 확률)",
    "Threshold": "Threshold\n(임계값)",
    "Gaussian Naive Bayes": "Gaussian Naive Bayes\n(가우시안 나이브 베이즈)",
    "KNN": "KNN\n(K-최근접 이웃)",
    "Distance": "Distance\n(거리)",
    "n_neighbors": "n_neighbors\n(이웃 수)",
    "Scaling": "Scaling\n(스케일링)",
    "Decision Tree": "Decision Tree\n(의사결정나무)",
    "Node / Branch / Leaf": "Node / Branch / Leaf\n(노드 / 가지 / 잎)",
    "Gini Impurity": "Gini Impurity\n(지니 불순도)",
    "max_depth": "max_depth\n(최대 깊이)",
    "Random Forest": "Random Forest\n(랜덤포레스트)",
    "Ensemble": "Ensemble\n(앙상블)",
    "Bootstrap": "Bootstrap\n(복원 추출)",
    "Feature Importance": "Feature Importance\n(Feature 중요도)",
    "Boosting": "Boosting\n(부스팅)",
    "Gradient Boosting": "Gradient Boosting\n(그래디언트 부스팅)",
    "XGBoost": "XGBoost\n(외부 Boosting)",
    "LightGBM": "LightGBM\n(외부 Boosting)",
    "CatBoost": "CatBoost\n(외부 Boosting)",
    "SVM": "SVM\n(서포트 벡터 머신)",
    "Decision Boundary": "Decision Boundary\n(결정 경계)",
    "Margin": "Margin\n(여유 공간)",
    "Support Vector": "Support Vector\n(서포트 벡터)",
    "C": "C\n(규제 Parameter)",
    "Kernel": "Kernel\n(커널)",
}

STEP_NAMES = [
    "1. Model", "2. Prediction", "3. Evaluation", "4. Threshold", "5. Visualization"
]

MODEL_GUIDES = {
    "logistic": "Class별 가능성을 계산해 예측한다. Feature Scaling을 적용하는 것이 일반적이며 C는 규제 강도에 반비례한다.",
    "gaussian_nb": "Class별 숫자 Feature가 정규분포를 따르고 Feature들이 조건부 독립이라고 가정해 확률을 계산한다.",
    "knn": "가까운 Train Sample들의 투표로 Class를 예측한다. 거리 계산을 사용하므로 Scaling과 n_neighbors를 함께 확인한다.",
    "decision_tree": "Feature 조건으로 데이터를 반복해서 나눈다. Scaling은 일반적으로 필요하지 않으며 max_depth로 Tree 복잡도를 조절한다.",
    "random_forest": "Bootstrap Sample과 일부 Feature로 여러 Tree를 학습해 Class 확률을 평균한다. 하나의 Tree보다 과적합을 줄이는 데 도움이 될 수 있지만 항상 방지하는 것은 아니다.",
    "gradient_boosting": "Tree를 순서대로 추가해 이전 단계의 부족한 부분을 보완한다. n_estimators와 learning_rate를 함께 조절한다.",
    "svm": "Decision Boundary와 Margin을 고려한다. Scaling이 중요하며 C와 Kernel에 따라 경계가 달라진다. Threshold 실습에 사용할 확률은 CalibratedClassifierCV로 보정한다.",
}

GRAPH_DESCRIPTIONS = {
    "Confusion Matrix": "<b>Confusion Matrix(혼동 행렬)</b>: 실제 Class를 행, 예측 Class를 열로 표시한다.",
    "Decision Boundary": "<b>Decision Boundary(결정 경계)</b>: 두 숫자 Feature에서 모델이 Class를 나누는 영역을 표시한다.",
    "Tree Structure": "<b>Tree Structure(Tree 구조)</b>: Decision Tree의 분할 조건과 Node, Leaf를 표시한다.",
    "Feature Importance": "<b>Feature Importance</b>: Tree 기반 모델이 분할에서 사용한 Feature의 상대적 중요도를 표시한다.",
    "Threshold Metrics": "<b>Threshold Metrics</b>: Threshold 변화에 따른 Precision, Recall, F1 변화를 표시한다.",
}


class ClassificationPage(QWidget):
    """Classification Lab의 데이터, 모델, 평가 결과와 UI를 구성한다."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset_name = ""
        self.target_column: str | None = None
        self.task_type = "unknown"
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series(dtype=object)
        self.y_test = pd.Series(dtype=object)
        self.preprocessing_artifacts = PreprocessingArtifacts()
        self.feature_checkboxes: dict[str, QCheckBox] = {}
        self.new_sample_inputs: dict[str, QLineEdit] = {}

        self.model = None
        self.train_prediction: np.ndarray | None = None
        self.test_prediction: np.ndarray | None = None
        self.test_probability: np.ndarray | None = None
        self.class_labels: list[Any] = []
        self.baseline_prediction: np.ndarray | None = None

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
        title = QLabel("Classification")
        title.setObjectName("pageTitle")
        title.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        header.addWidget(title)
        description = QLabel(
            "분류 모델은 Feature를 이용해 정해진 Class 중 하나를 예측한다.<br>"
            "Train Data로 모델을 학습하고 Test Data의 예측 Class와 실제 Class를 비교해 평가한다."
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

    def _create_concept_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Concept"), alignment=Qt.AlignmentFlag.AlignLeft)
        buttons = QGridLayout()
        buttons.setSpacing(SPACE_SM)
        self.concept_group = QButtonGroup(self)
        self.concept_group.setExclusive(True)
        self.concept_buttons: dict[str, QPushButton] = {}
        for index, concept in enumerate(CONCEPT_DESCRIPTIONS):
            button = QPushButton(CONCEPT_LABELS[concept])
            button.setObjectName("conceptButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, name=concept: self._show_concept(name))
            self.concept_group.addButton(button)
            self.concept_buttons[concept] = button
            buttons.addWidget(button, index // 4, index % 4)
        for column in range(4):
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
        self.concept_buttons["Classification"].setChecked(True)
        self._show_concept("Classification")

    def _create_workflow_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Classification Workflow"), alignment=Qt.AlignmentFlag.AlignLeft)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(SPACE_SM)
        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.step_buttons: list[QPushButton] = []
        self.step_pages = [
            self._create_model_step(), self._create_prediction_step(),
            self._create_evaluation_step(), self._create_threshold_step(),
            self._create_visualization_step(),
        ]
        for index, name in enumerate(STEP_NAMES):
            button = QPushButton(name)
            button.setObjectName("stepButton")
            button.setCheckable(True)
            button.setEnabled(index == 0)
            button.clicked.connect(lambda checked=False, page_index=index: self._show_step(page_index))
            self.step_group.addButton(button)
            self.step_buttons.append(button)
            button_layout.addWidget(button, 1)
        layout.addLayout(button_layout)
        for page in self.step_pages:
            layout.addWidget(page)
        self.step_buttons[0].setChecked(True)
        self._show_step(0)

    def _create_model_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Model Training",
            "사용할 Feature와 분류 모델을 선택한 뒤 Train Data로 모델을 학습한다. 모델에 따라 필요한 Hyperparameter와 Scaling 조건이 달라진다.",
        )
        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)
        controls.addWidget(self._create_control_label("Model"))
        self.model_combo = ChevronComboBox()
        self.model_combo.setObjectName("dataControl")
        for text, key in (
            ("Logistic Regression", "logistic"),
            ("Gaussian Naive Bayes", "gaussian_nb"),
            ("KNN", "knn"),
            ("Decision Tree", "decision_tree"),
            ("Random Forest", "random_forest"),
            ("Gradient Boosting", "gradient_boosting"),
            ("SVM", "svm"),
        ):
            self.model_combo.addItem(text, key)
        self.model_combo.currentIndexChanged.connect(self._update_model_controls)
        controls.addWidget(self.model_combo, 1)

        self.parameter_widgets: dict[str, tuple[QLabel, QWidget]] = {}
        self.c_spin = self._add_double_control(controls, "C", 0.0001, 1000000.0, 1.0, 4)
        self.var_smoothing_spin = self._add_double_control(controls, "var_smoothing", 1e-12, 0.1, 1e-9, 12)
        self.neighbor_spin = self._add_int_control(controls, "n_neighbors", 1, 999, 5)
        self.weights_combo = ChevronComboBox()
        self.weights_combo.setObjectName("dataControl")
        self.weights_combo.addItems(["uniform", "distance"])
        self._add_existing_control(controls, "weights", self.weights_combo)
        self.depth_spin = self._add_int_control(controls, "max_depth", 0, 50, 3)
        self.depth_spin.setSpecialValueText("None")
        self.split_spin = self._add_int_control(controls, "min_samples_split", 2, 100, 2)
        self.leaf_spin = self._add_int_control(controls, "min_samples_leaf", 1, 100, 1)
        self.estimator_spin = self._add_int_control(controls, "n_estimators", 10, 1000, 100)
        self.learning_rate_spin = self._add_double_control(controls, "learning_rate", 0.001, 1.0, 0.1, 3)
        self.kernel_combo = ChevronComboBox()
        self.kernel_combo.setObjectName("dataControl")
        self.kernel_combo.addItems(["linear", "rbf", "poly", "sigmoid"])
        self._add_existing_control(controls, "kernel", self.kernel_combo)

        for widget in (
            self.c_spin, self.var_smoothing_spin, self.neighbor_spin, self.weights_combo,
            self.depth_spin, self.split_spin, self.leaf_spin, self.estimator_spin,
            self.learning_rate_spin, self.kernel_combo,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._model_setting_changed)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._model_setting_changed)

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
        feature_layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        feature_layout.setSpacing(SPACE_XS)
        title = QLabel("Feature 선택")
        title.setObjectName("encodingTitle")
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

        self.model_result_label = self._create_result_card()
        self.model_result_label.setText("모델을 학습하기 전입니다.")
        layout.addWidget(self.model_result_label)
        model_table_card, self.model_detail_table = self._create_table_card("Model Detail")
        self.model_detail_table.setMinimumHeight(180)
        layout.addWidget(model_table_card)
        code_card, self.train_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        self._update_model_controls()
        return page

    def _create_prediction_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Class Prediction",
            "predict()로 Test Data의 Class를 예측한다. predict_proba()로 각 Class의 예측 확률과 classes_의 열 순서를 확인한다.",
        )
        row = QHBoxLayout()
        self.prediction_status_label = QLabel("먼저 Model 단계에서 모델을 학습하세요.")
        self.prediction_status_label.setObjectName("smallText")
        self.prediction_status_label.setWordWrap(True)
        row.addWidget(self.prediction_status_label, 1)
        self.predict_button = QPushButton("Test 예측")
        self.predict_button.setObjectName("dataButton")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self._predict_test)
        row.addWidget(self.predict_button)
        layout.addLayout(row)

        card, self.prediction_table = self._create_table_card("Actual / Prediction / Probability")
        layout.addWidget(card)

        new_card = QFrame()
        new_card.setObjectName("preprocessingCard")
        new_layout = QVBoxLayout(new_card)
        new_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        new_title = QLabel("New Sample Prediction")
        new_title.setObjectName("sectionTitle")
        new_layout.addWidget(new_title)
        self.new_sample_guide = QLabel("모델을 학습하면 선택한 Feature의 값을 입력할 수 있습니다.")
        self.new_sample_guide.setObjectName("smallText")
        self.new_sample_guide.setWordWrap(True)
        new_layout.addWidget(self.new_sample_guide)
        self.new_sample_grid = QGridLayout()
        new_layout.addLayout(self.new_sample_grid)
        self.new_predict_button = QPushButton("새 Sample 예측")
        self.new_predict_button.setObjectName("dataButton")
        self.new_predict_button.setEnabled(False)
        self.new_predict_button.clicked.connect(self._predict_new_sample)
        new_layout.addWidget(self.new_predict_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.new_prediction_label = QLabel()
        self.new_prediction_label.setObjectName("smallText")
        self.new_prediction_label.setWordWrap(True)
        new_layout.addWidget(self.new_prediction_label)
        layout.addWidget(new_card)

        code_card, self.prediction_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_evaluation_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Classification Evaluation",
            "Accuracy, Precision, Recall, F1과 Confusion Matrix를 함께 확인한다. Class Imbalance가 있으면 Accuracy만으로 성능을 판단하지 않는다.",
        )
        positive_row = QHBoxLayout()
        positive_row.addWidget(self._create_control_label("Positive Class"))
        self.positive_combo = ChevronComboBox()
        self.positive_combo.setObjectName("dataControl")
        self.positive_combo.currentIndexChanged.connect(self._positive_class_changed)
        positive_row.addWidget(self.positive_combo, 1)
        self.average_guide_label = QLabel()
        self.average_guide_label.setObjectName("smallText")
        self.average_guide_label.setWordWrap(True)
        positive_row.addWidget(self.average_guide_label, 2)
        layout.addLayout(positive_row)

        card, self.metric_table = self._create_table_card("Metrics / Baseline")
        self.metric_table.setMinimumHeight(210)
        layout.addWidget(card)
        cm_card, self.confusion_table = self._create_table_card("Confusion Matrix")
        self.confusion_table.setMinimumHeight(180)
        layout.addWidget(cm_card)
        self.binary_confusion_label = QLabel()
        self.binary_confusion_label.setObjectName("smallText")
        self.binary_confusion_label.setWordWrap(True)
        layout.addWidget(self.binary_confusion_label)
        report_card, self.report_table = self._create_table_card("Classification Report")
        self.report_table.setMinimumHeight(240)
        layout.addWidget(report_card)

        imbalance = QLabel(
            "<b>Class Imbalance Demo</b><br>Class 0이 5개이고 Class 1이 95개일 때 "
            "모든 Sample을 Class 1로 예측하면 Accuracy는 95%다.<br>"
            "하지만 Class 0은 하나도 찾지 못하므로 Precision, Recall, F1과 Confusion Matrix를 함께 확인한다."
        )
        imbalance.setObjectName("preprocessingInfoCard")
        imbalance.setWordWrap(True)
        layout.addWidget(imbalance)
        self.performance_label = self._create_result_card()
        self.performance_label.setText("Test 예측을 실행하면 Train/Test와 Baseline 결과를 해석합니다.")
        layout.addWidget(self.performance_label)
        code_card, self.evaluation_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_threshold_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Threshold Experiment",
            "이진 분류에서 Positive Class의 예측 확률이 Threshold 이상이면 Positive로 분류한다. Threshold를 바꾸며 Precision, Recall, F1과 오답 유형의 변화를 확인한다.",
        )
        row = QHBoxLayout()
        row.addWidget(self._create_control_label("Threshold"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 99)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self._update_threshold_result)
        row.addWidget(self.threshold_slider, 1)
        self.threshold_value_label = QLabel("0.50")
        self.threshold_value_label.setObjectName("smallText")
        row.addWidget(self.threshold_value_label)
        layout.addLayout(row)
        self.threshold_guide_label = QLabel("Test 예측 후 이진 분류 결과에서 사용할 수 있습니다.")
        self.threshold_guide_label.setObjectName("smallText")
        self.threshold_guide_label.setWordWrap(True)
        layout.addWidget(self.threshold_guide_label)
        card, self.threshold_metric_table = self._create_table_card("Threshold Metrics")
        self.threshold_metric_table.setMinimumHeight(170)
        layout.addWidget(card)
        cm_card, self.threshold_confusion_table = self._create_table_card("Threshold Confusion Matrix")
        self.threshold_confusion_table.setMinimumHeight(180)
        layout.addWidget(cm_card)
        code_card, self.threshold_code_label = self._create_code_block()
        layout.addWidget(code_card)
        layout.addStretch()
        return page

    def _create_visualization_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Classification Visualization",
            "Matplotlib을 이용해 분류 결과와 모델이 학습한 구조를 확인한다. 모델과 선택한 Feature 수에 따라 사용할 수 있는 그래프가 달라진다.",
        )
        controls = QHBoxLayout()
        controls.addWidget(self._create_control_label("Graph"))
        self.graph_combo = ChevronComboBox()
        self.graph_combo.setObjectName("dataControl")
        self.graph_combo.addItems(list(GRAPH_DESCRIPTIONS))
        self.graph_combo.currentTextChanged.connect(self._update_graph_controls)
        controls.addWidget(self.graph_combo, 1)
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
        self.canvas.setMinimumHeight(380)
        self.canvas.installEventFilter(self)
        self.canvas.hide()
        layout.addWidget(self.canvas)
        layout.addStretch()
        self._update_graph_controls()
        return page

    def set_preprocessing_result(self, result: PreprocessingResult, use_default_notice: bool = False) -> None:
        self.dataset_name = result.dataset_name
        self.target_column = result.target_column
        self.task_type = result.task_type
        self.x_train = result.x_train.copy(deep=True)
        self.x_test = result.x_test.copy(deep=True)
        self.y_train = result.y_train.copy(deep=True)
        self.y_test = result.y_test.copy(deep=True)
        self.preprocessing_artifacts = result.artifacts
        self.current_dataset_label.setText(
            "선택된 전처리 결과가 없어 프로그램에 포함된 Classification Sample을 사용합니다."
            if use_default_notice else result.dataset_name
        )
        class_count = self.y_train.nunique(dropna=False) if not self.y_train.empty else 0
        self.dataset_info_label.setText(
            f"<b>{escape(self.dataset_name)}</b> · Target: {escape(str(self.target_column or '없음'))}<br>"
            f"Train: {len(self.x_train)} Samples · Test: {len(self.x_test)} Samples · "
            f"Features: {len(self.x_train.columns)} · Classes: {class_count}"
        )
        self._populate_feature_checkboxes()
        self._invalidate_model()
        if self.task_type != "classification":
            self.status_label.setText("현재 Dataset은 분류 Dataset이 아닙니다. Data Lab에서 분류 Dataset을 선택해 주세요.")
        elif self.x_train.empty or self.x_test.empty:
            self.status_label.setText("Preprocessing의 1. Train/Test에서 데이터를 분리한 뒤 Classification으로 이동해 주세요.")
        else:
            self.status_label.setText("사용할 Feature와 Model을 선택한 뒤 모델 학습을 눌러주세요.")
        self._update_train_button()
        self._update_model_guide()

    def _load_default_dataset(self) -> None:
        dataframe = load_builtin_dataset("Classification Sample")
        target = "purchased"
        features = [column for column in dataframe.columns if column != target]
        x_train, x_test, y_train, y_test = split_dataset(
            dataframe, features, target, test_size=0.25, random_state=42, use_stratify=True,
        )
        x_train, x_test, encoded_columns, encoder = one_hot_encode_feature(
            x_train, x_test, "member_type"
        )
        artifacts = PreprocessingArtifacts(
            input_features=tuple(features),
            output_features=tuple(map(str, x_train.columns)),
            feature_encoders={"member_type": encoder},
        )
        self.set_preprocessing_result(
            PreprocessingResult(
                dataset_name="Classification Sample", target_column=target,
                task_type="classification", x_train=x_train, x_test=x_test,
                y_train=y_train, y_test=y_test, artifacts=artifacts,
            ),
            use_default_notice=True,
        )

    def _populate_feature_checkboxes(self) -> None:
        while self.feature_grid.count():
            item = self.feature_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.feature_checkboxes.clear()
        excluded = []
        for index, column in enumerate(self.x_train.columns):
            name = str(column)
            checkbox = QCheckBox(name)
            checkbox.setObjectName("dataCheckBox")
            is_identifier = name.lower() == "id" or name.lower().endswith("_id")
            checkbox.setChecked(not is_identifier)
            checkbox.toggled.connect(self._feature_selection_changed)
            self.feature_checkboxes[name] = checkbox
            self.feature_grid.addWidget(checkbox, index // 3, index % 3)
            if is_identifier:
                excluded.append(name)
        self.feature_guide_label.setText(
            "식별자 성격의 Feature는 기본 선택에서 제외했습니다: " + ", ".join(excluded)
            if excluded else "모델 학습에 사용할 Feature를 하나 이상 선택하세요."
        )

    def _selected_features(self) -> list[str]:
        return [
            str(column) for column in self.x_train.columns
            if str(column) in self.feature_checkboxes and self.feature_checkboxes[str(column)].isChecked()
        ]

    def _add_existing_control(self, layout: QHBoxLayout, name: str, widget: QWidget) -> None:
        label = self._create_control_label(name)
        layout.addWidget(label)
        layout.addWidget(widget)
        self.parameter_widgets[name] = (label, widget)

    def _add_int_control(self, layout, name, minimum, maximum, value):
        widget = ChevronSpinBox()
        widget.setObjectName("dataControl")
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        self._add_existing_control(layout, name, widget)
        return widget

    def _add_double_control(self, layout, name, minimum, maximum, value, decimals):
        widget = ChevronDoubleSpinBox()
        widget.setObjectName("dataControl")
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        self._add_existing_control(layout, name, widget)
        return widget

    def _update_model_controls(self, _index: int = -1) -> None:
        visible = {
            "logistic": {"C"},
            "gaussian_nb": {"var_smoothing"},
            "knn": {"n_neighbors", "weights"},
            "decision_tree": {"max_depth", "min_samples_split", "min_samples_leaf"},
            "random_forest": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"},
            "gradient_boosting": {"n_estimators", "learning_rate", "max_depth", "min_samples_leaf"},
            "svm": {"C", "kernel"},
        }[self.model_combo.currentData()]
        for name, (label, widget) in self.parameter_widgets.items():
            label.setVisible(name in visible)
            widget.setVisible(name in visible)
        self._model_setting_changed()

    def _model_setting_changed(self, _value=None) -> None:
        if hasattr(self, "prediction_status_label"):
            self._invalidate_model("Model 설정이 변경되었습니다. 다시 학습해 주세요.")
        self._update_model_guide()

    def _feature_selection_changed(self, _checked=False) -> None:
        self._invalidate_model("Feature 선택이 변경되었습니다. 다시 학습해 주세요.")
        self._update_train_button()
        self._update_model_guide()

    def _update_model_guide(self) -> None:
        if not hasattr(self, "model_guide_label"):
            return
        key = self.model_combo.currentData()
        messages = [MODEL_GUIDES[key]]
        if key in {"logistic", "knn", "svm"}:
            if self._needs_internal_scaling(self._selected_features()):
                messages.append("선택한 Feature 전체에 Scaling이 적용되어 있지 않아 모델 Pipeline 안에서 StandardScaler를 Train Data에 fit한다.")
            else:
                messages.append("선택한 Feature에 Preprocessing에서 Train Data로 학습한 Scaling 기준이 적용되어 있다.")
        messages.append("모델과 Hyperparameter의 최종 비교는 Validation Data나 Cross Validation으로 수행한다.")
        self.model_guide_label.setText("<br>".join(messages))

    def _update_train_button(self) -> None:
        self.train_button.setEnabled(
            self.task_type == "classification" and not self.x_train.empty
            and not self.x_test.empty and bool(self._selected_features())
        )

    def _train_model(self) -> None:
        features = self._selected_features()
        if not features:
            self.status_label.setText("하나 이상의 Feature를 선택해 주세요.")
            return
        self._invalidate_model()
        train_x = self.x_train.loc[:, features]
        test_x = self.x_test.loc[:, features]
        key = self.model_combo.currentData()
        depth = self.depth_spin.value() or None
        if key == "knn" and self.neighbor_spin.value() > len(train_x):
            self.status_label.setText(f"n_neighbors는 Train Sample 수 {len(train_x)} 이하로 설정해 주세요.")
            return
        if key == "svm" and int(self.y_train.value_counts().min()) < 2:
            self.status_label.setText("SVM 확률 보정에는 각 Class의 Train Sample이 두 개 이상 필요합니다.")
            return
        try:
            validate_classification_data(train_x, self.y_train)
            validate_classification_data(test_x, self.y_test, require_multiple_classes=False)
            self.model = create_classifier(
                key,
                scale_features=self._needs_internal_scaling(features),
                c=self.c_spin.value(),
                var_smoothing=self.var_smoothing_spin.value(),
                n_neighbors=self.neighbor_spin.value(),
                weights=self.weights_combo.currentText(),
                max_depth=depth,
                min_samples_split=self.split_spin.value(),
                min_samples_leaf=self.leaf_spin.value(),
                n_estimators=self.estimator_spin.value(),
                learning_rate=self.learning_rate_spin.value(),
                kernel=self.kernel_combo.currentText(),
            )
            self.model.fit(train_x, self.y_train)
        except (TypeError, ValueError) as error:
            self.model = None
            self.status_label.setText(f"모델 학습 실패: {error}")
            return

        self.train_prediction = np.asarray(self.model.predict(train_x))
        estimator = self._estimator()
        self.class_labels = list(estimator.classes_)
        details = self._model_detail_frame(self._core_estimator(), features)
        self._populate_table(self.model_detail_table, details)
        self.model_result_label.setText(
            f"<b>{escape(self.model_combo.currentText())}</b><br>"
            f"Train Samples: {len(train_x)} · Classes: {escape(', '.join(map(str, self.class_labels)))}<br>"
            f"Features: {escape(', '.join(features))}"
        )
        self.train_code_label.setText(self._training_code(features))
        self._populate_new_sample_inputs(features)
        self.status_label.setText(
            f"{self.model_combo.currentText()} 모델 학습을 완료했습니다. Prediction 단계에서 Test Data를 예측해 보세요."
        )
        self.prediction_status_label.setText("학습된 모델로 Test Data를 예측할 수 있습니다.")
        self.step_buttons[1].setEnabled(True)
        self.predict_button.setEnabled(True)
        self.new_predict_button.setEnabled(True)

    def _model_detail_frame(self, estimator, features: list[str]) -> pd.DataFrame:
        if hasattr(estimator, "feature_importances_"):
            return pd.DataFrame({"Feature": features, "Importance": estimator.feature_importances_})
        if hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_)
            if coefficients.ndim == 2 and coefficients.shape[0] == 1:
                coefficients = coefficients[0]
                return pd.DataFrame({"Feature": features, "Coefficient": coefficients})
            return pd.DataFrame({"Class": np.repeat(estimator.classes_, len(features)), "Feature": features * len(estimator.classes_), "Coefficient": coefficients.ravel()})
        if self.model_combo.currentData() == "knn":
            return pd.DataFrame({"Setting": ["n_neighbors", "weights", "metric"], "Value": [estimator.n_neighbors, estimator.weights, "Minkowski (p=2)"]})
        if self.model_combo.currentData() == "gaussian_nb":
            return pd.DataFrame({"Class": estimator.classes_, "Prior": estimator.class_prior_})
        if self.model_combo.currentData() == "svm":
            return pd.DataFrame({"Class": estimator.classes_, "Support Vectors": estimator.n_support_})
        return pd.DataFrame()

    def _training_code(self, features: list[str]) -> str:
        key = self.model_combo.currentData()
        params = {
            "logistic": ("from sklearn.linear_model import LogisticRegression", f"LogisticRegression(C={self.c_spin.value():g}, max_iter=1000)"),
            "gaussian_nb": ("from sklearn.naive_bayes import GaussianNB", f"GaussianNB(var_smoothing={self.var_smoothing_spin.value():g})"),
            "knn": ("from sklearn.neighbors import KNeighborsClassifier", f"KNeighborsClassifier(n_neighbors={self.neighbor_spin.value()}, weights='{self.weights_combo.currentText()}')"),
            "decision_tree": ("from sklearn.tree import DecisionTreeClassifier", f"DecisionTreeClassifier(max_depth={self.depth_spin.value() or None}, min_samples_split={self.split_spin.value()}, min_samples_leaf={self.leaf_spin.value()}, random_state=42)"),
            "random_forest": ("from sklearn.ensemble import RandomForestClassifier", f"RandomForestClassifier(n_estimators={self.estimator_spin.value()}, max_depth={self.depth_spin.value() or None}, min_samples_split={self.split_spin.value()}, min_samples_leaf={self.leaf_spin.value()}, random_state=42)"),
            "gradient_boosting": ("from sklearn.ensemble import GradientBoostingClassifier", f"GradientBoostingClassifier(n_estimators={self.estimator_spin.value()}, learning_rate={self.learning_rate_spin.value():g}, max_depth={self.depth_spin.value() or 3}, min_samples_leaf={self.leaf_spin.value()}, random_state=42)"),
            "svm": (
                "from sklearn.svm import SVC\nfrom sklearn.calibration import CalibratedClassifierCV",
                "CalibratedClassifierCV(\n"
                f"        SVC(C={self.c_spin.value():g}, kernel='{self.kernel_combo.currentText()}', random_state=42),\n"
                "        cv=2, ensemble=False,\n    )",
            ),
        }
        import_line, constructor = params[key]
        if self._needs_internal_scaling(features) and key in {"logistic", "knn", "svm"}:
            code = (
                f"{import_line}\nfrom sklearn.pipeline import Pipeline\n"
                "from sklearn.preprocessing import StandardScaler\n\n"
                "model = Pipeline([\n    ('scaler', StandardScaler()),\n"
                f"    ('classifier', {constructor}),\n])"
            )
        else:
            code = f"{import_line}\n\nmodel = {constructor}"
        return f"features = {features!r}\n{code}\nmodel.fit(X_train[features], y_train)"

    def _predict_test(self) -> None:
        if self.model is None:
            return
        features = self._selected_features()
        test_x = self.x_test.loc[:, features]
        self.test_prediction = np.asarray(self.model.predict(test_x))
        self.test_probability = np.asarray(self.model.predict_proba(test_x))
        self.class_labels = list(self._estimator().classes_)
        data: dict[str, Any] = {
            "Sample": self.y_test.index.astype(str),
            "Actual": self.y_test.to_numpy(),
            "Prediction": self.test_prediction,
        }
        for index, label in enumerate(self.class_labels):
            data[f"P(Class {label})"] = self.test_probability[:, index]
        self._populate_table(self.prediction_table, pd.DataFrame(data))
        self.baseline_prediction, _ = create_majority_baseline(
            self.x_train.loc[:, features], self.y_train, test_x
        )
        self._populate_positive_classes()
        self._update_evaluation()
        self._update_threshold_result()
        self.prediction_code_label.setText(
            "prediction = model.predict(X_test[features])\n"
            "probability = model.predict_proba(X_test[features])\n\n"
            "print(model.classes_)  # 확률 열의 Class 순서\n"
            "result = pd.DataFrame({'Actual': y_test, 'Prediction': prediction})"
        )
        self.evaluation_code_label.setText(
            "from sklearn.metrics import (\n    accuracy_score, confusion_matrix,\n"
            "    precision_score, recall_score, f1_score, classification_report,\n)\n"
            "from sklearn.dummy import DummyClassifier\n\n"
            "accuracy = accuracy_score(y_test, prediction)\n"
            "precision = precision_score(y_test, prediction, pos_label=positive_class)\n"
            "recall = recall_score(y_test, prediction, pos_label=positive_class)\n"
            "f1 = f1_score(y_test, prediction, pos_label=positive_class)\n"
            "cm = confusion_matrix(y_test, prediction, labels=model.classes_)\n"
            "report = classification_report(y_test, prediction)\n\n"
            "baseline = DummyClassifier(strategy='most_frequent')\n"
            "baseline.fit(X_train[features], y_train)"
        )
        self.prediction_status_label.setText(f"Test {len(self.y_test)} Samples의 예측을 완료했습니다.")
        self.status_label.setText("Test 예측과 분류 모델 평가를 완료했습니다.")
        for index in (2, 3, 4):
            self.step_buttons[index].setEnabled(True)
        self._update_graph_controls()

    def _populate_positive_classes(self) -> None:
        current = self.positive_combo.currentData()
        self.positive_combo.blockSignals(True)
        self.positive_combo.clear()
        for label in self.class_labels:
            self.positive_combo.addItem(str(label), label)
        if current in self.class_labels:
            self.positive_combo.setCurrentIndex(self.class_labels.index(current))
        elif len(self.class_labels) == 2:
            self.positive_combo.setCurrentIndex(1)
        self.positive_combo.blockSignals(False)
        is_binary = len(self.class_labels) == 2
        self.positive_combo.setEnabled(is_binary)
        self.average_guide_label.setText(
            "이진 분류는 선택한 Positive Class를 기준으로 계산합니다."
            if is_binary else "다중 분류는 각 Class 지표와 Macro Avg, Weighted Avg를 함께 확인합니다."
        )

    def _positive_class_changed(self, _index=-1) -> None:
        if self.test_prediction is not None:
            self._update_evaluation()
            self._update_threshold_result()

    def _positive_label(self):
        return self.positive_combo.currentData() if len(self.class_labels) == 2 else None

    def _update_evaluation(self) -> None:
        if self.test_prediction is None or self.baseline_prediction is None:
            return
        positive = self._positive_label()
        train = calculate_classification_metrics(self.y_train, self.train_prediction, positive_label=positive)
        test = calculate_classification_metrics(self.y_test, self.test_prediction, positive_label=positive)
        baseline = calculate_classification_metrics(self.y_test, self.baseline_prediction, positive_label=positive)
        self._populate_table(self.metric_table, pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Train Model": self._metric_values(train),
            "Test Model": self._metric_values(test),
            "Test Baseline": self._metric_values(baseline),
        }))
        cm = create_confusion_frame(self.y_test, self.test_prediction, self.class_labels)
        self._populate_table(self.confusion_table, cm.reset_index(names="Actual / Prediction"))
        if len(self.class_labels) == 2:
            positive = self._positive_label()
            negative = next(label for label in self.class_labels if label != positive)
            binary_matrix = confusion_matrix(
                self.y_test,
                self.test_prediction,
                labels=[negative, positive],
            )
            tn, fp, fn, tp = binary_matrix.ravel()
            self.binary_confusion_label.setText(
                f"Positive Class: {positive} · TN: {tn} · FP: {fp} · FN: {fn} · TP: {tp}"
            )
        else:
            self.binary_confusion_label.setText(
                "TP, TN, FP, FN은 한 Class를 Positive로 정해 해석하는 이진 분류 개념입니다."
            )
        self._populate_table(
            self.report_table,
            create_classification_report(self.y_test, self.test_prediction, self.class_labels),
        )
        ratios = self.y_train.value_counts(normalize=True)
        ratio_text = " · ".join(f"Class {label}: {ratio * 100:.1f}%" for label, ratio in ratios.items())
        messages = [
            f"Train Class 비율은 {ratio_text}다.",
            f"Test Accuracy는 {test.accuracy:.3f}이고 최빈 Class Baseline Accuracy는 {baseline.accuracy:.3f}이다.",
        ]
        if test.accuracy <= baseline.accuracy:
            messages.append("모델 Accuracy가 Baseline보다 높지 않으므로 단순한 최빈 Class 예측보다 나은지 다른 지표도 확인한다.")
        else:
            messages.append("모델 Accuracy가 최빈 Class Baseline보다 높다.")
        messages.append("작은 Test Data의 한 번 결과만으로 일반화 성능을 단정하지 않고 이후 Cross Validation에서도 확인한다.")
        self.performance_label.setText("<br>".join(messages))

    @staticmethod
    def _metric_values(metrics: ClassificationMetrics) -> list[float]:
        return [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1]

    def _update_threshold_result(self, _value=None) -> None:
        threshold = self.threshold_slider.value() / 100
        self.threshold_value_label.setText(f"{threshold:.2f}")
        available = self.test_probability is not None and len(self.class_labels) == 2
        self.threshold_slider.setEnabled(available)
        if not available:
            self.threshold_guide_label.setText("Threshold 실습은 예측 확률을 제공하는 이진 분류 결과에서 사용할 수 있습니다.")
            self._clear_table(self.threshold_metric_table)
            self._clear_table(self.threshold_confusion_table)
            return
        positive = self._positive_label()
        negative = next(label for label in self.class_labels if label != positive)
        positive_index = self.class_labels.index(positive)
        prediction = np.where(self.test_probability[:, positive_index] >= threshold, positive, negative)
        metrics = calculate_classification_metrics(self.y_test, prediction, positive_label=positive)
        self._populate_table(self.threshold_metric_table, pd.DataFrame({
            "Threshold": [threshold], "Positive Class": [positive],
            "Accuracy": [metrics.accuracy], "Precision": [metrics.precision],
            "Recall": [metrics.recall], "F1": [metrics.f1],
        }))
        cm = create_confusion_frame(self.y_test, prediction, self.class_labels)
        self._populate_table(self.threshold_confusion_table, cm.reset_index(names="Actual / Prediction"))
        self.threshold_guide_label.setText(
            f"Class {positive}의 예측 확률이 {threshold:.2f} 이상이면 Positive로 분류합니다. "
            "Threshold를 높이면 Positive 판정 조건이 엄격해지고 낮추면 더 많은 Sample을 Positive로 예측합니다."
        )
        self.threshold_code_label.setText(
            f"positive_class = {positive!r}\npositive_index = list(model.classes_).index(positive_class)\n"
            f"threshold = {threshold:.2f}\npositive_probability = probability[:, positive_index]\n"
            "custom_prediction = np.where(\n    positive_probability >= threshold,\n"
            f"    positive_class, {negative!r},\n)"
        )

    def _populate_new_sample_inputs(self, features: list[str]) -> None:
        while self.new_sample_grid.count():
            item = self.new_sample_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.new_sample_inputs.clear()
        for index, feature in enumerate(features):
            label = QLabel(feature)
            label.setObjectName("smallText")
            field = QLineEdit()
            field.setObjectName("dataControl")
            median = pd.to_numeric(self.x_train[feature], errors="coerce").median()
            field.setText(f"{median:g}" if pd.notna(median) else "0")
            self.new_sample_grid.addWidget(label, index // 4 * 2, index % 4)
            self.new_sample_grid.addWidget(field, index // 4 * 2 + 1, index % 4)
            self.new_sample_inputs[feature] = field
        self.new_sample_guide.setText(
            "현재 모델이 사용하는 전처리 후 Feature 값을 입력하세요. 모델 Pipeline의 Scaling은 Train Data에서 학습한 같은 기준을 적용합니다."
        )

    def _predict_new_sample(self) -> None:
        if self.model is None:
            return
        try:
            values = {name: [float(field.text())] for name, field in self.new_sample_inputs.items()}
        except ValueError:
            self.new_prediction_label.setText("모든 Feature에 숫자 값을 입력해 주세요.")
            return
        sample = pd.DataFrame(values)
        prediction = self.model.predict(sample)[0]
        probability = self.model.predict_proba(sample)[0]
        probability_text = " · ".join(
            f"Class {label}: {value:.3f}" for label, value in zip(self.class_labels, probability)
        )
        self.new_prediction_label.setText(f"예측 Class: {prediction}<br>Class별 예측 확률: {probability_text}")

    def _update_graph_controls(self, _text="") -> None:
        if not hasattr(self, "graph_combo"):
            return
        graph = self.graph_combo.currentText()
        key = self.model_combo.currentData() if hasattr(self, "model_combo") else ""
        features = self._selected_features() if hasattr(self, "feature_checkboxes") else []
        available = self.test_prediction is not None
        reason = ""
        if graph == "Decision Boundary" and len(features) != 2:
            available = False
            reason = "<br>Decision Boundary는 숫자 Feature를 정확히 두 개 선택해 학습한 경우 확인할 수 있습니다."
        elif graph == "Tree Structure" and key != "decision_tree":
            available = False
            reason = "<br>Tree Structure는 Decision Tree 모델에서 확인할 수 있습니다."
        elif graph == "Feature Importance" and key not in {"decision_tree", "random_forest", "gradient_boosting"}:
            available = False
            reason = "<br>Feature Importance는 Tree 기반 모델에서 확인할 수 있습니다."
        elif graph == "Threshold Metrics" and len(self.class_labels) != 2:
            available = False
            reason = "<br>Threshold Metrics는 이진 분류 결과에서 확인할 수 있습니다."
        self.graph_description_label.setText(GRAPH_DESCRIPTIONS.get(graph, "") + reason)
        self.draw_button.setEnabled(available)

    def _draw_graph(self) -> None:
        if self.model is None or self.test_prediction is None:
            return
        graph = self.graph_combo.currentText()
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.set_facecolor(COLOR_OFF_WHITE)
        if graph == "Confusion Matrix":
            matrix = confusion_matrix(self.y_test, self.test_prediction, labels=self.class_labels)
            image = axes.imshow(matrix, cmap="Purples")
            for row in range(len(self.class_labels)):
                for column in range(len(self.class_labels)):
                    axes.text(column, row, str(matrix[row, column]), ha="center", va="center", color=COLOR_OFF_BLACK)
            axes.set_xticks(range(len(self.class_labels)), [str(v) for v in self.class_labels])
            axes.set_yticks(range(len(self.class_labels)), [str(v) for v in self.class_labels])
            axes.set_xlabel("Predicted Class")
            axes.set_ylabel("Actual Class")
            self.figure.colorbar(image, ax=axes)
        elif graph == "Decision Boundary":
            self._draw_decision_boundary(axes)
        elif graph == "Tree Structure":
            plot_tree(
                self._core_estimator(), feature_names=self._selected_features(),
                class_names=[str(value) for value in self.class_labels], filled=True, ax=axes,
            )
        elif graph == "Feature Importance":
            importance = self._core_estimator().feature_importances_
            order = np.argsort(importance)
            axes.barh(np.asarray(self._selected_features())[order], importance[order], color=COLOR_SECONDARY)
            axes.set_xlabel("Relative Importance")
        elif graph == "Threshold Metrics":
            self._draw_threshold_metrics(axes)
        axes.set_title(graph, color=COLOR_OFF_BLACK)
        axes.tick_params(colors=COLOR_OFF_BLACK)
        for spine in axes.spines.values():
            spine.set_color(COLOR_PRIMARY)
        self.canvas.show()
        self.canvas.draw()

    def _draw_decision_boundary(self, axes) -> None:
        features = self._selected_features()
        train = self.x_train.loc[:, features].astype(float)
        test = self.x_test.loc[:, features].astype(float)
        combined = pd.concat([train, test])
        x_pad = max((combined.iloc[:, 0].max() - combined.iloc[:, 0].min()) * 0.08, 0.1)
        y_pad = max((combined.iloc[:, 1].max() - combined.iloc[:, 1].min()) * 0.08, 0.1)
        xx, yy = np.meshgrid(
            np.linspace(combined.iloc[:, 0].min() - x_pad, combined.iloc[:, 0].max() + x_pad, 220),
            np.linspace(combined.iloc[:, 1].min() - y_pad, combined.iloc[:, 1].max() + y_pad, 220),
        )
        grid = pd.DataFrame({features[0]: xx.ravel(), features[1]: yy.ravel()})
        grid_prediction = self.model.predict(grid)
        class_index = {label: index for index, label in enumerate(self.class_labels)}
        zz = np.asarray([class_index[value] for value in grid_prediction]).reshape(xx.shape)
        axes.contourf(xx, yy, zz, alpha=0.22, cmap="Purples")
        for label in self.class_labels:
            train_mask = self.y_train.to_numpy() == label
            test_mask = self.y_test.to_numpy() == label
            axes.scatter(train.loc[train_mask, features[0]], train.loc[train_mask, features[1]], label=f"Train {label}")
            axes.scatter(test.loc[test_mask, features[0]], test.loc[test_mask, features[1]], marker="x", s=70, label=f"Test {label}")
        if self.model_combo.currentData() == "svm":
            estimator = self._core_estimator()
            support = estimator.support_vectors_
            if isinstance(self.model, Pipeline):
                support = self.model.named_steps["scaler"].inverse_transform(support)
            axes.scatter(support[:, 0], support[:, 1], facecolors="none", edgecolors=COLOR_OFF_BLACK, s=120, label="Support Vector")
        axes.set_xlabel(features[0])
        axes.set_ylabel(features[1])
        axes.legend()

    def _draw_threshold_metrics(self, axes) -> None:
        positive = self._positive_label()
        negative = next(label for label in self.class_labels if label != positive)
        probability = self.test_probability[:, self.class_labels.index(positive)]
        thresholds = np.linspace(0.01, 0.99, 99)
        precision_values, recall_values, f1_values = [], [], []
        for threshold in thresholds:
            prediction = np.where(probability >= threshold, positive, negative)
            metrics = calculate_classification_metrics(self.y_test, prediction, positive_label=positive)
            precision_values.append(metrics.precision)
            recall_values.append(metrics.recall)
            f1_values.append(metrics.f1)
        axes.plot(thresholds, precision_values, label="Precision")
        axes.plot(thresholds, recall_values, label="Recall")
        axes.plot(thresholds, f1_values, label="F1")
        axes.axvline(self.threshold_slider.value() / 100, color=COLOR_OFF_BLACK, linestyle="--", label="Current")
        axes.set_xlabel("Threshold")
        axes.set_ylabel("Score")
        axes.set_ylim(-0.02, 1.02)
        axes.legend()

    def _estimator(self):
        if isinstance(self.model, Pipeline):
            return self.model.named_steps["classifier"]
        return self.model

    def _needs_internal_scaling(self, features: list[str]) -> bool:
        if self.preprocessing_artifacts.scaler is None:
            return True
        scaled = set(map(str, self.preprocessing_artifacts.scaler_columns))
        return not set(features).issubset(scaled)

    def _core_estimator(self):
        estimator = self._estimator()
        if self.model_combo.currentData() == "svm" and hasattr(estimator, "calibrated_classifiers_"):
            return estimator.calibrated_classifiers_[0].estimator
        return estimator

    def _invalidate_model(self, message: str | None = None) -> None:
        self.model = None
        self.train_prediction = None
        self.test_prediction = None
        self.test_probability = None
        self.baseline_prediction = None
        self.class_labels = []
        if hasattr(self, "model_result_label"):
            self.model_result_label.setText("모델을 학습하기 전입니다.")
            self.prediction_status_label.setText("먼저 Model 단계에서 모델을 학습하세요.")
            self.performance_label.setText("Test 예측을 실행하면 Train/Test와 Baseline 결과를 해석합니다.")
            self.binary_confusion_label.clear()
            self.new_prediction_label.clear()
            self.train_code_label.clear()
            self.prediction_code_label.clear()
            self.evaluation_code_label.clear()
            self.threshold_code_label.clear()
            for table in (
                self.model_detail_table, self.prediction_table, self.metric_table,
                self.confusion_table, self.report_table, self.threshold_metric_table,
                self.threshold_confusion_table,
            ):
                self._clear_table(table)
            self.figure.clear()
            self.canvas.hide()
            for button in self.step_buttons[1:]:
                button.setEnabled(False)
            self.predict_button.setEnabled(False)
            self.new_predict_button.setEnabled(False)
            self._show_step(0)
            self.step_buttons[0].setChecked(True)
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
        if watched is self.canvas and event.type() == QEvent.Type.Wheel:
            delta = event.pixelDelta().y() or event.angleDelta().y()
            bar = self.scroll_area.verticalScrollBar()
            bar.setValue(bar.value() - delta)
            return True
        return super().eventFilter(watched, event)

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
        layout.setSpacing(SPACE_XS)
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

