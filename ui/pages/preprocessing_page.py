# Train/Test 분리와 대표적인 데이터 전처리를 실습하는 화면을 구성
from html import escape

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from common.theme import SPACE_MD, SPACE_SM, SPACE_XS
from ml.data_loader import load_builtin_dataset
from ml.data_summary import get_feature_columns
from ml.preprocessing import (
    compare_scaler_fit, encode_target, get_class_ratios, impute_feature,
    one_hot_encode_feature, scale_features, split_dataset,
)
from ui.widgets import ChevronComboBox, ChevronDoubleSpinBox, ChevronSpinBox


CONCEPT_DESCRIPTIONS = {
    "Preprocessing": "원본 데이터를 머신러닝 모델이 학습하기 적절한 형태로 변환하는 과정이다.",
    "X / y": "X는 모델에 입력하는 Feature 데이터이고, y는 모델이 예측하려는 Target 데이터다.",
    "Train / Test": "Train Data는 학습에 사용하고 Test Data는 학습에 사용하지 않은 결과를 확인할 때 사용한다.",
    "Test Size": "전체 데이터 중 Test Data로 사용할 비율이다. 0.20이면 20%를 Test Data로 사용하고 나머지 80%를 Train Data로 사용한다.",
    "Random State": "데이터를 무작위로 나눌 때 같은 분할 결과를 다시 얻기 위해 사용하는 값이다. 같은 정수를 사용하면 분할 결과를 재현할 수 있으며 모델의 성능을 높이는 값은 아니다.",
    "Stratify": "분류 데이터를 나눌 때 Target의 Class 비율이 Train Data와 Test Data에서 비슷하게 유지되도록 한다. Sample 수와 분할 비율에 따라 Class 비율이 완전히 같지 않을 수 있으며 회귀에서는 일반적으로 사용하지 않는다.",
    "Encoding": "문자열로 된 범주형 Feature를 머신러닝 알고리즘이 사용할 수 있는 숫자 형태로 변환하는 과정이다. 순서가 없는 범주에 단순히 숫자를 대응시키면 실제로 존재하지 않는 크기 관계가 생길 수 있으므로 Feature의 의미에 맞는 Encoding 방법을 선택해야 한다.",
    "Scaling": "Feature마다 단위와 값의 크기가 다를 때 값의 크기를 일정한 기준으로 변환하는 과정이다. 일부 머신러닝 알고리즘은 Feature 값의 크기 차이에 영향을 받을 수 있으므로 알고리즘과 데이터에 따라 Scaling을 적용한다.",
    "fit()": "전처리에 필요한 평균, 최솟값, 범주 같은 변환 기준을 Train Data에서 구한다.",
    "transform()": "fit()으로 정한 기준을 실제 데이터에 적용한다.",
    "fit_transform()": "Train Data에서 변환 기준을 구한 뒤 바로 적용하는 fit()과 transform()의 결합이다.",
    "Data Leakage": "전처리 기준을 정할 때 Test Data처럼 알 수 없어야 하는 정보가 들어가는 문제다.",
}

STEP_NAMES = ["1. Train/Test", "2. Missing", "3. Encoding", "4. Scaling", "5. Data Leakage"]


class PreprocessingPage(QWidget):
    # Preprocessing Lab의 상태와 UI를 구성
    def __init__(self) -> None:
        super().__init__()

        self.dataframe = pd.DataFrame()
        self.dataset_name = ""
        self.target_column: str | None = None
        self.task_type = "unknown"
        self.feature_columns: list[str] = []

        self.x_train_raw = pd.DataFrame()
        self.x_test_raw = pd.DataFrame()
        self.y_train_raw = pd.Series(dtype=object)
        self.y_test_raw = pd.Series(dtype=object)
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series(dtype=object)
        self.y_test = pd.Series(dtype=object)
        self.target_mapping: dict[str, int] | None = None
        self.scaling_base_train = pd.DataFrame()
        self.scaling_base_test = pd.DataFrame()

        self._setup_ui()
        self.set_dataset(
            load_builtin_dataset("Student Basic"),
            "Student Basic",
            "passed",
            "classification",
            use_default_notice=True,
        )

    # Preprocessing 전체 화면을 Dataset, Concept, 단계별 실습으로 구성
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

        header_layout = QHBoxLayout()
        header_layout.setSpacing(SPACE_SM)

        title = QLabel("Preprocessing")
        title.setObjectName("pageTitle")
        title.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(title)

        description = QLabel(
            "원본 데이터를 나누고 결측치·범주형 값·숫자 범위를 모델이 학습할 수 있는 형태로 변환한다."
        )
        description.setObjectName("bodyText")
        description.setWordWrap(True)
        header_layout.addWidget(description, 1)
        layout.addLayout(header_layout)

        self._create_dataset_section(layout)
        self._create_concept_section(layout)
        self._create_workflow_section(layout)
        layout.addStretch()

    # 현재 실습에 사용하는 Dataset 정보를 표시
    def _create_dataset_section(self, layout: QVBoxLayout) -> None:
        dataset_header = QHBoxLayout()
        dataset_header.setSpacing(SPACE_SM)
        dataset_header.addWidget(self._create_section_badge("Current Dataset"))

        self.current_dataset_label = QLabel()
        self.current_dataset_label.setObjectName("bodyText")
        self.current_dataset_label.setWordWrap(True)
        dataset_header.addWidget(self.current_dataset_label, 1)
        layout.addLayout(dataset_header)

        self.dataset_info_label = QLabel()
        self.dataset_info_label.setObjectName("preprocessingInfoCard")
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label)

        self.status_label = QLabel()
        self.status_label.setObjectName("smallText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # 전처리에 필요한 개념 버튼과 설명을 구성
    def _create_concept_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Concept"), alignment=Qt.AlignmentFlag.AlignLeft)

        button_layout = QGridLayout()
        button_layout.setSpacing(SPACE_SM)
        self.concept_group = QButtonGroup(self)
        self.concept_group.setExclusive(True)
        self.concept_buttons: dict[str, QPushButton] = {}
        column_count = 6

        for index, concept in enumerate(CONCEPT_DESCRIPTIONS):
            button = QPushButton(concept)
            button.setObjectName("conceptButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, name=concept: self._show_concept(name))
            self.concept_group.addButton(button)
            self.concept_buttons[concept] = button
            button_layout.addWidget(button, index // column_count, index % column_count)

        for column in range(column_count):
            button_layout.setColumnStretch(column, 1)

        layout.addLayout(button_layout)

        detail_card = QFrame()
        detail_card.setObjectName("conceptDetailCard")
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)

        self.concept_detail_label = QLabel()
        self.concept_detail_label.setObjectName("bodyText")
        self.concept_detail_label.setWordWrap(True)
        detail_layout.addWidget(self.concept_detail_label)
        layout.addWidget(detail_card)

        self.concept_buttons["Preprocessing"].setChecked(True)
        self._show_concept("Preprocessing")

    # 전처리 단계 선택 버튼과 각 실습 화면을 구성
    def _create_workflow_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self._create_section_badge("Preprocessing Workflow"), alignment=Qt.AlignmentFlag.AlignLeft)

        step_layout = QHBoxLayout()
        step_layout.setSpacing(SPACE_SM)
        self.step_group = QButtonGroup(self)
        self.step_group.setExclusive(True)
        self.step_buttons: list[QPushButton] = []

        self.step_pages = [
            self._create_split_step(),
            self._create_missing_step(),
            self._create_encoding_step(),
            self._create_scaling_step(),
            self._create_leakage_step(),
        ]

        step_container = QWidget()
        step_container_layout = QVBoxLayout(step_container)
        step_container_layout.setContentsMargins(0, 0, 0, 0)

        for index, (name, page) in enumerate(zip(STEP_NAMES, self.step_pages)):
            button = QPushButton(name)
            button.setObjectName("stepButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page_index=index: self._show_step(page_index))
            self.step_group.addButton(button)
            self.step_buttons.append(button)
            step_layout.addWidget(button, 1)
            step_container_layout.addWidget(page)

        layout.addLayout(step_layout)
        layout.addWidget(step_container, alignment=Qt.AlignmentFlag.AlignTop)
        self.step_buttons[0].setChecked(True)
        self._show_step(0)

    # Train/Test 입력과 분리 결과를 구성
    def _create_split_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Train/Test Split",
            "X와 y를 같은 기준으로 학습용과 테스트용으로 나눈다.",
        )

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)

        controls.addWidget(self._create_control_label("Test Size"))
        self.test_size_spin = ChevronDoubleSpinBox()
        self.test_size_spin.setObjectName("dataControl")
        self.test_size_spin.setRange(0.10, 0.50)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setDecimals(2)
        self.test_size_spin.setValue(0.20)
        self.test_size_spin.valueChanged.connect(self._refresh_selected_concept)
        controls.addWidget(self.test_size_spin)

        controls.addWidget(self._create_control_label("Random State"))
        self.random_state_spin = ChevronSpinBox()
        self.random_state_spin.setObjectName("dataControl")
        self.random_state_spin.setRange(0, 9999)
        self.random_state_spin.setValue(42)
        self.random_state_spin.valueChanged.connect(self._refresh_selected_concept)
        controls.addWidget(self.random_state_spin)

        self.stratify_checkbox = QCheckBox("Stratify 사용")
        self.stratify_checkbox.setObjectName("dataCheckBox")
        self.stratify_checkbox.setChecked(True)
        self.stratify_checkbox.toggled.connect(self._refresh_selected_concept)
        controls.addWidget(self.stratify_checkbox, alignment=Qt.AlignmentFlag.AlignVCenter)
        controls.addStretch()

        self.split_button = QPushButton("분리하기")
        self.split_button.setObjectName("dataButton")
        self.split_button.clicked.connect(self._run_split)
        controls.addWidget(self.split_button)
        layout.addLayout(controls)

        result_layout = QHBoxLayout()
        result_layout.setSpacing(SPACE_SM)
        self.train_summary_label = self._create_result_card()
        self.test_summary_label = self._create_result_card()
        result_layout.addWidget(self.train_summary_label, 1)
        result_layout.addWidget(self.test_summary_label, 1)
        layout.addLayout(result_layout)

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(SPACE_SM)
        train_preview, self.train_table = self._create_table_card("Train Data Preview")
        test_preview, self.test_table = self._create_table_card("Test Data Preview")
        preview_layout.addWidget(train_preview, 1)
        preview_layout.addWidget(test_preview, 1)
        layout.addLayout(preview_layout)

        split_code_card, self.split_code_label = self._create_code_block()
        layout.addWidget(split_code_card)
        layout.addStretch()
        return page

    # SimpleImputer 입력과 처리 전후 비교를 구성
    def _create_missing_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Missing Value",
            "scikit-learn의 SimpleImputer를 사용하여 결측치를 평균, 중앙값, "
            "최빈값처럼 지정한 기준으로 채울 수 있다. 채울 기준은 Train Data에서 "
            "구하고, Test Data에는 같은 기준을 적용한다.",
        )

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)
        controls.addWidget(self._create_control_label("Feature"))
        self.missing_column_combo = ChevronComboBox()
        self.missing_column_combo.setObjectName("dataControl")
        self.missing_column_combo.currentIndexChanged.connect(self._update_imputer_strategies)
        controls.addWidget(self.missing_column_combo, 1)

        controls.addWidget(self._create_control_label("Strategy"))
        self.imputer_strategy_combo = ChevronComboBox()
        self.imputer_strategy_combo.setObjectName("dataControl")
        controls.addWidget(self.imputer_strategy_combo, 1)

        self.impute_button = QPushButton("적용하기")
        self.impute_button.setObjectName("dataButton")
        self.impute_button.clicked.connect(self._apply_imputer)
        controls.addWidget(self.impute_button)
        layout.addLayout(controls)

        self.imputer_result_label = QLabel("결측치가 있는 Feature를 선택하세요.")
        self.imputer_result_label.setObjectName("smallText")
        self.imputer_result_label.setWordWrap(True)
        layout.addWidget(self.imputer_result_label)

        comparison = QHBoxLayout()
        comparison.setSpacing(SPACE_SM)
        before_card, self.missing_before_table = self._create_table_card("처리 전")
        after_card, self.missing_after_table = self._create_table_card("처리 후")
        comparison.addWidget(before_card, 1)
        comparison.addWidget(after_card, 1)
        layout.addLayout(comparison)

        missing_code_card, self.missing_code_label = self._create_code_block()
        layout.addWidget(missing_code_card)
        layout.addStretch()
        return page

    # Target Label Encoding과 Feature One-Hot Encoding을 구성
    def _create_encoding_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Encoding",
            "문자열 Target은 Label Encoding하고 순서가 없는 범주형 Feature는 One-Hot Encoding한다.",
        )

        label_card = QFrame()
        label_card.setObjectName("preprocessingCard")
        label_layout = QVBoxLayout(label_card)
        label_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        label_layout.setSpacing(SPACE_SM)

        label_header = QHBoxLayout()
        label_title = QLabel("Target Label Encoding")
        label_title.setObjectName("encodingTitle")
        label_header.addWidget(label_title)
        label_header.addStretch()
        self.label_encode_button = QPushButton("Target 변환")
        self.label_encode_button.setObjectName("dataButton")
        self.label_encode_button.clicked.connect(self._apply_label_encoder)
        label_header.addWidget(self.label_encode_button)
        label_layout.addLayout(label_header)

        label_encoding_description = QLabel(
            "scikit-learn의 LabelEncoder로 문자열 Target y의 각 Class를 숫자로 변환한다. <br/>"
            "일반적으로 입력 Feature인 X보다 Target인 y의 Class를 변환하는 데 사용한다."
        )
        label_encoding_description.setObjectName("bodyText")
        label_encoding_description.setWordWrap(True)
        label_layout.addWidget(label_encoding_description)

        self.label_mapping_label = QLabel()
        self.label_mapping_label.setObjectName("smallText")
        label_layout.addWidget(self.label_mapping_label)

        label_tables = QHBoxLayout()
        before_card, self.label_before_table = self._create_table_card("변환 전")
        after_card, self.label_after_table = self._create_table_card("변환 후")
        label_tables.addWidget(before_card, 1)
        label_tables.addWidget(after_card, 1)
        label_layout.addLayout(label_tables)
        layout.addWidget(label_card)

        one_hot_card = QFrame()
        one_hot_card.setObjectName("preprocessingCard")
        one_hot_layout = QVBoxLayout(one_hot_card)
        one_hot_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        one_hot_layout.setSpacing(SPACE_SM)

        one_hot_header = QHBoxLayout()
        one_hot_title = QLabel("Feature One-Hot Encoding")
        one_hot_title.setObjectName("encodingTitle")
        one_hot_header.addWidget(one_hot_title)
        one_hot_header.addWidget(self._create_control_label("Feature"))
        self.one_hot_column_combo = ChevronComboBox()
        self.one_hot_column_combo.setObjectName("dataControl")
        one_hot_header.addWidget(self.one_hot_column_combo, 1)
        self.one_hot_button = QPushButton("Feature 변환")
        self.one_hot_button.setObjectName("dataButton")
        self.one_hot_button.clicked.connect(self._apply_one_hot_encoder)
        one_hot_header.addWidget(self.one_hot_button)
        one_hot_layout.addLayout(one_hot_header)

        one_hot_description = QLabel(
            "순서가 없는 범주형 Feature의 각 범주를 별도 Feature로 만들고, 해당 범주는 1, "
            "나머지는 0으로 변환한다.<br>"
            "범주 기준은 Train Data에서 정하고 Test Data에 동일하게 적용한다."
        )
        one_hot_description.setObjectName("bodyText")
        one_hot_description.setWordWrap(True)
        one_hot_layout.addWidget(one_hot_description)

        self.one_hot_result_label = QLabel()
        self.one_hot_result_label.setObjectName("smallText")
        self.one_hot_result_label.setWordWrap(True)
        one_hot_layout.addWidget(self.one_hot_result_label)

        one_hot_tables = QHBoxLayout()
        before_card, self.one_hot_before_table = self._create_table_card("변환 전")
        after_card, self.one_hot_after_table = self._create_table_card("변환 후")
        one_hot_tables.addWidget(before_card, 1)
        one_hot_tables.addWidget(after_card, 1)
        one_hot_layout.addLayout(one_hot_tables)
        layout.addWidget(one_hot_card)

        encoding_code_card, self.encoding_code_label = self._create_code_block()
        layout.addWidget(encoding_code_card)
        layout.addStretch()
        return page

    # StandardScaler와 MinMaxScaler 선택 및 처리 전후 비교를 구성
    def _create_scaling_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Scaling",
            "숫자형 Feature의 크기를 변환하며 모든 알고리즘에 Scaling이 필요한 것은 아니다.",
        )

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)
        controls.addWidget(self._create_control_label("Method"))
        self.scaler_combo = ChevronComboBox()
        self.scaler_combo.setObjectName("dataControl")
        self.scaler_combo.addItem("없음", "none")
        self.scaler_combo.addItem(
            "StandardScaler (학습 데이터의 평균과 표준편차를 기준으로 변환)",
            "standard",
        )
        self.scaler_combo.addItem(
            "MinMaxScaler (각 Feature의 최솟값과 최댓값을 기준으로 값의 범위 변환)",
            "minmax",
        )
        controls.addWidget(self.scaler_combo, 1)
        self.scale_button = QPushButton("적용하기")
        self.scale_button.setObjectName("dataButton")
        self.scale_button.clicked.connect(self._apply_scaler)
        controls.addWidget(self.scale_button)
        layout.addLayout(controls)

        self.scaling_columns_label = QLabel()
        self.scaling_columns_label.setObjectName("smallText")
        self.scaling_columns_label.setWordWrap(True)
        layout.addWidget(self.scaling_columns_label)

        comparison = QHBoxLayout()
        comparison.setSpacing(SPACE_SM)
        before_card, self.scaling_before_table = self._create_table_card("변환 전")
        after_card, self.scaling_after_table = self._create_table_card("변환 후")
        comparison.addWidget(before_card, 1)
        comparison.addWidget(after_card, 1)
        layout.addLayout(comparison)

        self.scaling_result_label = QLabel()
        self.scaling_result_label.setObjectName("smallText")
        self.scaling_result_label.setWordWrap(True)
        layout.addWidget(self.scaling_result_label)

        scaling_code_card, self.scaling_code_label = self._create_code_block()
        layout.addWidget(scaling_code_card)
        layout.addStretch()
        return page

    # Train 전용 fit과 전체 데이터 fit을 비교하는 Data Leakage 실습을 구성
    def _create_leakage_step(self) -> QWidget:
        page, layout = self._create_step_page(
            "Data Leakage",
            "전처리 기준은 Train Data에서만 정해야 한다.",
        )

        controls = QHBoxLayout()
        controls.setSpacing(SPACE_SM)
        controls.addWidget(self._create_control_label("Numeric Feature"))
        self.leakage_column_combo = ChevronComboBox()
        self.leakage_column_combo.setObjectName("dataControl")
        self.leakage_column_combo.currentIndexChanged.connect(self._update_leakage_comparison)
        controls.addWidget(self.leakage_column_combo, 1)
        controls.addStretch()
        layout.addLayout(controls)

        comparison_note = QLabel("비교 결과는 실제 전처리 데이터에 반영하지 않습니다.")
        comparison_note.setObjectName("smallText")
        layout.addWidget(comparison_note)

        comparison = QHBoxLayout()
        comparison.setSpacing(SPACE_SM)
        self.safe_fit_label = self._create_result_card()
        self.leaked_fit_label = self._create_result_card()
        comparison.addWidget(self.safe_fit_label, 1)
        comparison.addWidget(self.leaked_fit_label, 1)
        layout.addLayout(comparison)

        explanation = QLabel(
            "전체 데이터로 fit하면 Test Data의 평균과 표준편차가 변환 기준에 포함된다.<br>"
            "Test Data가 모델의 fit()에 직접 들어가지 않아도 학습 과정에 영향을 주므로 Data Leakage다."
        )
        explanation.setObjectName("preprocessingInfoCard")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        leakage_code_card, self.leakage_code_label = self._create_code_block()
        self.leakage_code_label.setText(
            "# 올바른 방법\nscaler.fit(X_train)\nX_test_scaled = scaler.transform(X_test)\n\n"
            "# Data Leakage\nscaler.fit(X 전체)"
        )
        layout.addWidget(leakage_code_card)
        layout.addStretch()
        return page

    # Data Lab에서 전달받은 Dataset을 복사해 현재 실습 데이터로 설정
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
        self.target_column = target_column
        self.task_type = task_type
        self.feature_columns = get_feature_columns(self.dataframe, target_column)

        if use_default_notice:
            self.current_dataset_label.setText(
                "선택된 Dataset이 없어 프로그램에 포함된 Student Basic을 사용합니다."
            )
        else:
            self.current_dataset_label.setText(dataset_name)

        target_text = target_column or "없음"
        missing_count = int(self.dataframe.isna().sum().sum())
        self.dataset_info_label.setText(
            f"<b>{escape(dataset_name)}</b> · {len(self.dataframe)} Samples · "
            f"{len(self.dataframe.columns)} Columns<br>"
            f"Target: {escape(str(target_text))} · Missing Values: {missing_count}"
        )

        can_split = target_column is not None and target_column in self.dataframe.columns
        self.stratify_checkbox.setEnabled(can_split and task_type == "classification")
        self.stratify_checkbox.setChecked(can_split and task_type == "classification")

        if can_split:
            self._reset_split_state()
            self.status_label.setText(
                "아래 Preprocessing Workflow의 1. Train/Test에서 "
                "Test Size와 Random State를 설정한 뒤 분리하기를 눌러주세요."
            )
        else:
            self._reset_split_state()
            self.status_label.setText("Target이 없는 Dataset은 현재 Train/Test 전처리 실습에서 사용할 수 없습니다.")

        self._show_concept(next((name for name, button in self.concept_buttons.items() if button.isChecked()), "Preprocessing"))

    # 선택한 비율과 난수 기준으로 Dataset을 Train/Test로 분리
    def _run_split(self) -> None:
        if self.target_column is None or self.dataframe.empty:
            return

        try:
            split_result = split_dataset(
                self.dataframe,
                self.feature_columns,
                self.target_column,
                self.test_size_spin.value(),
                self.random_state_spin.value(),
                self.stratify_checkbox.isChecked(),
            )
        except ValueError as error:
            self.status_label.setText(f"Train/Test 분리 실패: {error}")
            return

        self.x_train_raw, self.x_test_raw, self.y_train_raw, self.y_test_raw = split_result
        self.x_train = self.x_train_raw.copy()
        self.x_test = self.x_test_raw.copy()
        self.y_train = self.y_train_raw.copy()
        self.y_test = self.y_test_raw.copy()
        self.target_mapping = None
        self._set_scaling_base()

        train_ratios = self._format_class_ratios(self.y_train_raw)
        test_ratios = self._format_class_ratios(self.y_test_raw)
        self.train_summary_label.setText(
            f"<b>Train Data</b><br>{len(self.x_train_raw)} Samples<br>{train_ratios}"
        )
        self.test_summary_label.setText(
            f"<b>Test Data</b><br>{len(self.x_test_raw)} Samples<br>{test_ratios}"
        )

        self._populate_table(
            self.train_table,
            pd.concat([self.x_train_raw, self.y_train_raw], axis=1),
        )
        self._populate_table(
            self.test_table,
            pd.concat([self.x_test_raw, self.y_test_raw], axis=1),
        )
        self.split_code_label.setText(
            "from sklearn.model_selection import train_test_split\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            f"    X, y, test_size={self.test_size_spin.value():.2f},\n"
            f"    random_state={self.random_state_spin.value()}, "
            f"stratify={'y' if self.stratify_checkbox.isChecked() else 'None'}\n)"
        )

        self._reset_step_results()
        self._refresh_preprocessing_controls()
        for button in self.step_buttons[1:]:
            button.setEnabled(True)

        selected_concept = next(
            (name for name, button in self.concept_buttons.items() if button.isChecked()),
            "Preprocessing",
        )
        self._show_concept(selected_concept)
        self.status_label.setText(
            f"분리 완료: Train {len(self.x_train_raw)} Samples / Test {len(self.x_test_raw)} Samples"
        )

    # Dataset을 불러온 뒤 사용자가 분리 설정을 적용하기 전 상태로 초기화
    def _reset_split_state(self) -> None:
        self.x_train_raw = pd.DataFrame()
        self.x_test_raw = pd.DataFrame()
        self.y_train_raw = pd.Series(dtype=object)
        self.y_test_raw = pd.Series(dtype=object)
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series(dtype=object)
        self.y_test = pd.Series(dtype=object)
        self.target_mapping = None
        self.scaling_base_train = pd.DataFrame()
        self.scaling_base_test = pd.DataFrame()

        self.train_summary_label.setText("<b>Train Data</b><br>분리하기 전입니다.")
        self.test_summary_label.setText("<b>Test Data</b><br>분리하기 전입니다.")
        self._clear_table(self.train_table)
        self._clear_table(self.test_table)
        self._reset_step_results()
        self._refresh_preprocessing_controls()

        self._show_step(0)
        self.step_buttons[0].setChecked(True)
        for button in self.step_buttons[1:]:
            button.setEnabled(False)

    # 선택한 Feature의 결측치를 Train Data 기준으로 채움
    def _apply_imputer(self) -> None:
        column = self.missing_column_combo.currentData()
        strategy = self.imputer_strategy_combo.currentData()

        if not column or not strategy:
            return

        train_source = self.x_train.copy()
        test_source = self.x_test.copy()
        train_source[column] = self.x_train_raw[column]
        test_source[column] = self.x_test_raw[column]

        before = pd.concat(
            [
                pd.DataFrame({"Data": "Train", column: train_source[column]}),
                pd.DataFrame({"Data": "Test", column: test_source[column]}),
            ],
            ignore_index=True,
        )

        try:
            self.x_train, self.x_test, statistic = impute_feature(
                train_source,
                test_source,
                column,
                strategy,
            )
        except ValueError as error:
            self.imputer_result_label.setText(f"처리 실패: {error}")
            return

        self._populate_table(self.missing_before_table, before)
        after = pd.concat(
            [
                pd.DataFrame({"Data": "Train", column: self.x_train[column]}),
                pd.DataFrame({"Data": "Test", column: self.x_test[column]}),
            ],
            ignore_index=True,
        )
        self._populate_table(self.missing_after_table, after)
        self.imputer_result_label.setText(
            f"Train Data에서 구한 {column}의 처리 기준: {self._format_value(statistic)}"
        )
        self.missing_code_label.setText(
            "from sklearn.impute import SimpleImputer\n\n"
            f"imputer = SimpleImputer(strategy={strategy!r})\n"
            f"X_train[{column!r}] = imputer.fit_transform(X_train[[{column!r}]]).ravel()\n"
            f"X_test[{column!r}] = imputer.transform(X_test[[{column!r}]]).ravel()"
        )
        self._set_scaling_base()
        self._refresh_preprocessing_controls()
        self.status_label.setText(f"{column} 결측치 처리가 완료되었습니다.")

    # 문자열 Target을 숫자 Class로 변환
    def _apply_label_encoder(self) -> None:
        if self.y_train_raw.empty or pd.api.types.is_numeric_dtype(self.y_train_raw):
            self.label_mapping_label.setText("현재 Target은 이미 숫자형이므로 Label Encoding이 필요하지 않습니다.")
            return

        try:
            self.y_train, self.y_test, mapping = encode_target(self.y_train_raw, self.y_test_raw)
            self.target_mapping = mapping
        except ValueError as error:
            self.label_mapping_label.setText(f"변환 실패: {error}")
            return

        mapping_text = " · ".join(f"{name} → {value}" for name, value in mapping.items())
        self.label_mapping_label.setText(f"Class Mapping: {mapping_text}")
        self._populate_table(self.label_before_table, self.y_train_raw.to_frame())
        self._populate_table(self.label_after_table, self.y_train.to_frame())
        self._update_encoding_code()
        self.status_label.setText("Target Label Encoding이 완료되었습니다.")

    # 선택한 범주형 Feature를 One-Hot Encoding
    def _apply_one_hot_encoder(self) -> None:
        column = self.one_hot_column_combo.currentData()

        if not column:
            return

        if self.x_train[column].isna().any() or self.x_test[column].isna().any():
            self.one_hot_result_label.setText(
                "이 Feature에 결측치가 남아 있습니다. Missing 단계에서 먼저 처리하세요."
            )
            return

        before = self.x_train[[column]].copy()
        self.x_train, self.x_test, encoded_columns = one_hot_encode_feature(
            self.x_train,
            self.x_test,
            column,
        )
        self._populate_table(self.one_hot_before_table, before)
        self._populate_table(self.one_hot_after_table, self.x_train[encoded_columns])
        self._set_scaling_base()
        self._refresh_preprocessing_controls()
        self.one_hot_result_label.setText("생성된 Feature: " + ", ".join(encoded_columns))
        self._update_encoding_code(column)
        self.status_label.setText(f"{column} One-Hot Encoding이 완료되었습니다.")

    # 선택한 Scaling 방법을 숫자형 Feature에 적용
    def _apply_scaler(self) -> None:
        method = self.scaler_combo.currentData()
        columns = self._get_scaling_columns(self.scaling_base_train)

        if self.scaling_base_train[columns].isna().any().any():
            self.scaling_result_label.setText(
                "숫자형 Feature에 결측치가 남아 있습니다. Missing 단계에서 먼저 처리하세요."
            )
            return

        before = self.scaling_base_train[columns].copy()
        self.x_train, self.x_test, scaler = scale_features(
            self.scaling_base_train,
            self.scaling_base_test,
            columns,
            method,
        )
        self._populate_table(self.scaling_before_table, before)
        self._populate_table(self.scaling_after_table, self.x_train[columns].round(3))

        if method == "standard" and scaler is not None:
            self.scaling_result_label.setText(
                "Train 평균: " + self._format_array(scaler.mean_) +
                " / Train 표준편차: " + self._format_array(scaler.scale_)
            )
            scaler_name = "StandardScaler"
        elif method == "minmax" and scaler is not None:
            self.scaling_result_label.setText(
                "Train 최솟값: " + self._format_array(scaler.data_min_) +
                " / Train 최댓값: " + self._format_array(scaler.data_max_)
            )
            scaler_name = "MinMaxScaler"
        else:
            self.scaling_result_label.setText("Scaling을 적용하지 않은 원래 값으로 되돌렸습니다.")
            scaler_name = None

        if scaler_name:
            self.scaling_code_label.setText(
                f"from sklearn.preprocessing import {scaler_name}\n\n"
                f"numeric_features = {columns!r}\n"
                f"scaler = {scaler_name}()\n"
                "X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])\n"
                "X_test[numeric_features] = scaler.transform(X_test[numeric_features])"
            )
        else:
            self.scaling_code_label.setText("# Scaling을 적용하지 않는다.")

        self.status_label.setText("Scaling 설정이 적용되었습니다.")

    # 선택한 숫자형 Feature에서 안전한 fit과 누수가 있는 fit을 비교
    def _update_leakage_comparison(self, _index: int = -1) -> None:
        column = self.leakage_column_combo.currentData()

        if not column or self.x_train_raw.empty:
            return

        train_values = self.x_train_raw[column].dropna()
        test_values = self.x_test_raw[column].dropna()

        if train_values.empty or test_values.empty:
            return

        comparison = compare_scaler_fit(train_values, test_values)
        self.safe_fit_label.setText(
            "<b>올바른 전처리</b><br>scaler.fit(X_train)<br>"
            f"평균: {comparison['train_mean']:.3f}<br>"
            f"표준편차: {comparison['train_scale']:.3f}<br>"
            f"첫 Test 변환값: {comparison['safe_test_first']:.3f}"
        )
        self.leaked_fit_label.setText(
            "<b>Data Leakage 발생</b><br>scaler.fit(X 전체)<br>"
            f"평균: {comparison['full_mean']:.3f}<br>"
            f"표준편차: {comparison['full_scale']:.3f}<br>"
            f"첫 Test 변환값: {comparison['leaked_test_first']:.3f}"
        )

    # 현재 Dataset과 선택한 개념을 연결해 설명
    def _show_concept(self, concept: str) -> None:
        description = CONCEPT_DESCRIPTIONS[concept]
        current_text = ""

        if concept == "X / y" and self.target_column:
            features = ", ".join(map(str, self.feature_columns))
            current_text = f"<br><br>현재 X: {escape(features)}<br>현재 y: {escape(self.target_column)}"
        elif concept == "Train / Test" and not self.x_train_raw.empty:
            current_text = (
                f"<br><br>현재 Train: {len(self.x_train_raw)} Samples"
                f"<br>현재 Test: {len(self.x_test_raw)} Samples"
            )
        elif concept == "Test Size":
            current_text = f"<br><br>현재 설정: {self.test_size_spin.value():.2f}"
        elif concept == "Random State":
            current_text = f"<br><br>현재 설정: {self.random_state_spin.value()}"
        elif concept == "Stratify":
            status = "사용" if self.stratify_checkbox.isChecked() else "사용 안 함"
            current_text = f"<br><br>현재 설정: {status}"

        formatted_description = escape(description).replace("다. ", "다.<br>")
        self.concept_detail_label.setText(
            f"<b>{escape(concept)}</b>: {formatted_description}{current_text}"
        )

    # 숫자 설정이 바뀌면 현재 선택된 Concept의 값도 함께 갱신
    def _refresh_selected_concept(self, _value=None) -> None:
        selected_concept = next(
            (name for name, button in self.concept_buttons.items() if button.isChecked()),
            "Preprocessing",
        )
        self._show_concept(selected_concept)

    # 현재 단계 페이지만 표시
    def _show_step(self, index: int) -> None:
        for page_index, page in enumerate(self.step_pages):
            page.setHidden(page_index != index)

    # 결측치, Encoding, Scaling, Leakage 선택 목록을 현재 데이터에 맞게 갱신
    def _refresh_preprocessing_controls(self) -> None:
        self._refresh_missing_columns()
        self._refresh_encoding_controls()
        self._refresh_scaling_controls()
        self._refresh_leakage_columns()

    # 결측치가 남은 Feature만 Missing Value 목록에 표시
    def _refresh_missing_columns(self) -> None:
        selected = self.missing_column_combo.currentData()
        self.missing_column_combo.blockSignals(True)
        self.missing_column_combo.clear()

        original_missing_columns = [
            column
            for column in self.x_train.columns
            if column in self.x_train_raw.columns
            and (
                self.x_train_raw[column].isna().any()
                or self.x_test_raw[column].isna().any()
            )
        ]

        for column in original_missing_columns:
            missing_count = int(
                self.x_train[column].isna().sum()
                + self.x_test[column].isna().sum()
            )
            status = f"{missing_count}개" if missing_count else "처리 완료"
            self.missing_column_combo.addItem(f"{column} ({status})", column)

        if self.missing_column_combo.count() == 0:
            self.missing_column_combo.addItem("결측치 없음", None)
            self.impute_button.setEnabled(False)
        else:
            index = self.missing_column_combo.findData(selected)
            self.missing_column_combo.setCurrentIndex(max(index, 0))

        self.missing_column_combo.blockSignals(False)
        self._update_imputer_strategies()

    # 선택한 Feature 자료형에 맞는 SimpleImputer 전략을 표시
    def _update_imputer_strategies(self, _index: int = -1) -> None:
        column = self.missing_column_combo.currentData()
        self.imputer_strategy_combo.clear()

        if not column:
            self.impute_button.setEnabled(False)
            self.impute_button.setText("적용하기")
            return

        remaining_missing_count = int(
            self.x_train[column].isna().sum()
            + self.x_test[column].isna().sum()
        )
        original_missing_count = int(
            self.x_train_raw[column].isna().sum()
            + self.x_test_raw[column].isna().sum()
        )
        self.impute_button.setEnabled(original_missing_count > 0)
        self.impute_button.setText(
            "다시 적용" if remaining_missing_count == 0 else "적용하기"
        )

        if pd.api.types.is_numeric_dtype(self.x_train[column]):
            self.imputer_strategy_combo.addItem("Mean (평균)", "mean")
            self.imputer_strategy_combo.addItem("Median (중앙값)", "median")

        self.imputer_strategy_combo.addItem("Most Frequent (최빈값)", "most_frequent")

    # 현재 문자열 Target과 범주형 Feature를 Encoding 영역에 표시
    def _refresh_encoding_controls(self) -> None:
        split_completed = not self.y_train_raw.empty
        target_is_text = split_completed and not pd.api.types.is_numeric_dtype(self.y_train_raw)
        self.label_encode_button.setEnabled(target_is_text)
        if not split_completed:
            self.label_mapping_label.setText("Train/Test 분리 후 Target을 변환할 수 있습니다.")
        elif self.target_mapping:
            mapping_text = " · ".join(
                f"{name} → {value}" for name, value in self.target_mapping.items()
            )
            self.label_mapping_label.setText(f"Class Mapping: {mapping_text}")
        elif target_is_text:
            self.label_mapping_label.setText(f"Target: {self.target_column}")
        else:
            self.label_mapping_label.setText(
                "현재 Target은 이미 숫자형이므로 Label Encoding이 필요하지 않습니다."
            )

        selected = self.one_hot_column_combo.currentData()
        categorical_columns = self.x_train.select_dtypes(exclude="number").columns.tolist()
        self.one_hot_column_combo.clear()

        for column in categorical_columns:
            self.one_hot_column_combo.addItem(str(column), column)

        self.one_hot_button.setEnabled(bool(categorical_columns))
        index = self.one_hot_column_combo.findData(selected)
        if index >= 0:
            self.one_hot_column_combo.setCurrentIndex(index)

        if not categorical_columns:
            self.one_hot_column_combo.addItem("범주형 Feature 없음", None)
            self.one_hot_result_label.setText("모든 범주형 Feature가 숫자 형태로 변환되었습니다.")

    # 현재 Scaling 대상 숫자형 Feature와 미리보기를 표시
    def _refresh_scaling_controls(self) -> None:
        columns = self._get_scaling_columns(self.scaling_base_train)
        self.scaling_columns_label.setText(
            "Scaling 대상: " + (", ".join(columns) if columns else "없음")
        )
        self.scale_button.setEnabled(bool(columns))

        if columns:
            self._populate_table(self.scaling_before_table, self.scaling_base_train[columns])
            self._populate_table(self.scaling_after_table, self.scaling_base_train[columns])

    # Data Leakage 비교에 사용할 숫자형 Feature를 표시
    def _refresh_leakage_columns(self) -> None:
        selected = self.leakage_column_combo.currentData()
        numeric_columns = self.x_train_raw.select_dtypes(include="number").columns.tolist()
        self.leakage_column_combo.blockSignals(True)
        self.leakage_column_combo.clear()

        for column in numeric_columns:
            self.leakage_column_combo.addItem(str(column), column)

        preferred_column = "sleep_hours" if "sleep_hours" in numeric_columns else selected
        index = self.leakage_column_combo.findData(preferred_column)
        self.leakage_column_combo.setCurrentIndex(max(index, 0))
        self.leakage_column_combo.blockSignals(False)
        self._update_leakage_comparison()

    # Split 변경 시 이후 단계에서 표시하던 결과를 초기화
    def _reset_step_results(self) -> None:
        for table in (
            self.missing_before_table, self.missing_after_table,
            self.label_before_table, self.label_after_table,
            self.one_hot_before_table, self.one_hot_after_table,
            self.scaling_before_table, self.scaling_after_table,
        ):
            self._clear_table(table)

        self.imputer_result_label.setText("결측치가 있는 Feature를 선택하세요.")
        self.one_hot_result_label.clear()
        self.scaling_result_label.clear()
        self.missing_code_label.clear()
        self.encoding_code_label.clear()
        self.scaling_code_label.clear()
        self.safe_fit_label.clear()
        self.leaked_fit_label.clear()

    # 이후 Scaling을 반복 적용할 때 사용할 변환 직전 데이터를 저장
    def _set_scaling_base(self) -> None:
        self.scaling_base_train = self.x_train.copy()
        self.scaling_base_test = self.x_test.copy()

    # 원래 숫자형 Feature 중 현재 데이터에 남아 있는 Column을 반환
    def _get_scaling_columns(self, dataframe: pd.DataFrame) -> list[str]:
        return [
            column
            for column in self.feature_columns
            if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
        ]

    # Encoding 실습에서 사용하는 Python 코드를 현재 선택에 맞게 표시
    def _update_encoding_code(self, one_hot_column: str | None = None) -> None:
        code_parts = [
            "from sklearn.preprocessing import LabelEncoder, OneHotEncoder",
            "",
            "label_encoder = LabelEncoder()",
            "y_train_encoded = label_encoder.fit_transform(y_train)",
            "y_test_encoded = label_encoder.transform(y_test)",
        ]

        if one_hot_column:
            code_parts.extend([
                "",
                "encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
                f"X_train_encoded = encoder.fit_transform(X_train[[{one_hot_column!r}]])",
                f"X_test_encoded = encoder.transform(X_test[[{one_hot_column!r}]])",
            ])

        self.encoding_code_label.setText("\n".join(code_parts))

    # Class 비율을 한 줄 문자열로 변환
    def _format_class_ratios(self, target: pd.Series) -> str:
        if self.task_type != "classification":
            return "Class 비율: 해당 없음"

        ratios = get_class_ratios(target)
        ratio_text = " · ".join(
            f"Class {name}: {ratio:.1f}%"
            for name, ratio in ratios.items()
        )
        return "Class 비율: " + ratio_text

    # 배열 값을 짧은 문자열로 변환
    def _format_array(self, values) -> str:
        return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"

    # 숫자와 문자열 값을 UI에 표시하기 좋은 문자열로 변환
    def _format_value(self, value: object) -> str:
        if isinstance(value, (float, int)):
            return f"{value:.3f}"
        return str(value)

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

        formatted_description = description.replace("다. ", "다.<br>")
        description_label = QLabel(formatted_description)
        description_label.setObjectName("bodyText")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        return page, layout

    # 입력 Widget 앞에 표시할 Label을 생성
    def _create_control_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("smallText")
        return label

    # Train/Test 결과와 Data Leakage 비교에 사용하는 Card를 생성
    def _create_result_card(self) -> QLabel:
        label = QLabel()
        label.setObjectName("preprocessingResultCard")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        return label

    # 제목과 Table로 구성된 비교 Card를 생성
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
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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

    # 전처리 결과 DataFrame의 모든 행과 열을 Table에 표시
    def _populate_table(self, table: QTableWidget, dataframe: pd.DataFrame) -> None:
        table.clear()
        table.setRowCount(len(dataframe))
        table.setColumnCount(len(dataframe.columns))
        table.setHorizontalHeaderLabels([str(column) for column in dataframe.columns])

        for row_index in range(len(dataframe)):
            for column_index in range(len(dataframe.columns)):
                value = dataframe.iloc[row_index, column_index]
                text = "NaN" if pd.isna(value) else self._format_value(value)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, column_index, item)

    # Table의 이전 Header와 값을 모두 제거
    def _clear_table(self, table: QTableWidget) -> None:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)
