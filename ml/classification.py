"""분류 모델 학습과 평가에 사용하는 계산을 UI와 분리해 제공한다."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ClassificationMetrics:
    """한 분류 결과에서 계산한 대표 평가 지표다."""

    accuracy: float
    precision: float
    recall: float
    f1: float


def create_classifier(
    model_name: str,
    *,
    scale_features: bool = False,
    c: float = 1.0,
    max_iter: int = 1000,
    var_smoothing: float = 1e-9,
    n_neighbors: int = 5,
    weights: str = "uniform",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = 42,
) -> ClassifierMixin | Pipeline:
    """이름과 Hyperparameter에 맞는 scikit-learn 분류 모델을 생성한다."""
    if model_name == "logistic":
        estimator: ClassifierMixin = LogisticRegression(
            C=c,
            max_iter=max_iter,
            random_state=random_state,
        )
    elif model_name == "gaussian_nb":
        estimator = GaussianNB(var_smoothing=var_smoothing)
    elif model_name == "knn":
        estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
        )
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth if max_depth is not None else 3,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif model_name == "svm":
        estimator = CalibratedClassifierCV(
            SVC(
                C=c,
                kernel=kernel,
                gamma=gamma,
                random_state=random_state,
            ),
            cv=2,
            ensemble=False,
        )
    else:
        raise ValueError(f"지원하지 않는 분류 모델입니다: {model_name}")

    if scale_features and model_name in {"logistic", "knn", "svm"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])
    return estimator


def validate_classification_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    require_multiple_classes: bool = True,
) -> None:
    """분류 모델에 전달할 Feature와 Target의 기본 조건을 확인한다."""
    if features.empty:
        raise ValueError("학습에 사용할 Feature가 없습니다.")
    if target.empty:
        raise ValueError("학습에 사용할 Target이 없습니다.")
    if len(features) != len(target):
        raise ValueError("Feature와 Target의 Sample 수가 다릅니다.")
    if features.isna().any().any():
        raise ValueError("Feature에 결측치가 있습니다. Preprocessing에서 먼저 처리해 주세요.")
    non_numeric = [
        str(column)
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise TypeError(
            "문자열 Feature를 숫자로 Encoding해 주세요: " + ", ".join(non_numeric)
        )
    if target.isna().any():
        raise ValueError("Target에 결측치가 있습니다.")
    if require_multiple_classes and target.nunique(dropna=False) < 2:
        raise ValueError("분류 학습에는 두 개 이상의 Class가 필요합니다.")


def calculate_classification_metrics(
    actual: Sequence[Any],
    prediction: Sequence[Any],
    *,
    positive_label: Any | None = None,
) -> ClassificationMetrics:
    """이진 분류는 Positive Class, 다중 분류는 weighted 평균으로 계산한다."""
    labels = np.unique(np.concatenate([np.asarray(actual), np.asarray(prediction)]))
    if len(labels) == 2 and positive_label is not None:
        average = "binary"
        kwargs = {"pos_label": positive_label}
    else:
        average = "weighted"
        kwargs = {}

    return ClassificationMetrics(
        accuracy=float(accuracy_score(actual, prediction)),
        precision=float(
            precision_score(actual, prediction, average=average, zero_division=0, **kwargs)
        ),
        recall=float(
            recall_score(actual, prediction, average=average, zero_division=0, **kwargs)
        ),
        f1=float(f1_score(actual, prediction, average=average, zero_division=0, **kwargs)),
    )


def create_majority_baseline(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
) -> tuple[np.ndarray, DummyClassifier]:
    """Train Data의 최빈 Class를 모든 Test Sample에 예측하는 Baseline을 만든다."""
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train_features, train_target)
    return baseline.predict(test_features), baseline


def create_classification_report(
    actual: Sequence[Any],
    prediction: Sequence[Any],
    labels: Sequence[Any],
) -> pd.DataFrame:
    """Classification Report를 UI 표에 표시할 DataFrame으로 변환한다."""
    report = classification_report(
        actual,
        prediction,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    for label in labels:
        values = report[str(label)]
        rows.append({
            "Class": str(label),
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1-score"],
            "Support": int(values["support"]),
        })
    for key, label in (("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg")):
        values = report[key]
        rows.append({
            "Class": label,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1-score"],
            "Support": int(values["support"]),
        })
    return pd.DataFrame(rows)


def create_confusion_frame(
    actual: Sequence[Any],
    prediction: Sequence[Any],
    labels: Sequence[Any],
) -> pd.DataFrame:
    """실제 Class를 행, 예측 Class를 열로 갖는 Confusion Matrix를 만든다."""
    matrix = confusion_matrix(actual, prediction, labels=list(labels))
    return pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Pred {label}" for label in labels],
    )

