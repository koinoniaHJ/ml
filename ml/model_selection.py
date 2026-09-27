"""Cross Validation과 Hyperparameter 탐색 계산을 UI와 분리해 제공한다."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_validate,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.preprocessing import PreprocessingArtifacts, create_preprocessing_pipeline


@dataclass(frozen=True)
class CandidateSpec:
    """Model Selection에서 비교할 모델의 이름과 탐색 후보를 정의한다."""

    key: str
    label: str
    task_type: str
    needs_scaling: bool
    parameter_options: dict[str, list[Any]]


@dataclass(frozen=True)
class CrossValidationResult:
    """Fold별 Train/Validation 점수와 시간을 보관한다."""

    fold_table: pd.DataFrame
    train_scores: np.ndarray
    validation_scores: np.ndarray
    fit_times: np.ndarray

    @property
    def train_mean(self) -> float:
        return float(np.mean(self.train_scores))

    @property
    def validation_mean(self) -> float:
        return float(np.mean(self.validation_scores))

    @property
    def validation_std(self) -> float:
        return float(np.std(self.validation_scores))


CLASSIFICATION_SPECS = {
    spec.key: spec
    for spec in (
        CandidateSpec("logistic", "Logistic Regression", "classification", True, {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        }),
        CandidateSpec("gaussian_nb", "Gaussian Naive Bayes", "classification", False, {
            "var_smoothing": [1e-11, 1e-9, 1e-7],
        }),
        CandidateSpec("knn", "KNN", "classification", True, {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
        }),
        CandidateSpec("decision_tree", "Decision Tree", "classification", False, {
            "max_depth": [None, 2, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }),
        CandidateSpec("random_forest", "Random Forest", "classification", False, {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }),
        CandidateSpec("gradient_boosting", "Gradient Boosting", "classification", False, {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [1, 2, 3],
        }),
        CandidateSpec("svm", "SVM", "classification", True, {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }),
    )
}

REGRESSION_SPECS = {
    spec.key: spec
    for spec in (
        CandidateSpec("linear", "Linear Regression", "regression", False, {}),
        CandidateSpec("polynomial", "Polynomial Regression", "regression", False, {
            "degree": [2, 3, 4],
        }),
        CandidateSpec("ridge", "Ridge", "regression", True, {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }),
        CandidateSpec("lasso", "Lasso", "regression", True, {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }),
        CandidateSpec("elastic_net", "ElasticNet", "regression", True, {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        }),
    )
}


SCORING_OPTIONS = {
    "classification": {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "f1_macro": "F1 Macro",
        "f1_weighted": "F1 Weighted",
    },
    "regression": {
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
    },
}


def get_candidate_specs(task_type: str) -> dict[str, CandidateSpec]:
    """Task에 맞는 모델 후보를 반환한다."""
    if task_type == "classification":
        return CLASSIFICATION_SPECS
    if task_type == "regression":
        return REGRESSION_SPECS
    return {}


def create_validation_splits(
    task_type: str,
    target: pd.Series,
    folds: int,
    *,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """모든 후보가 재사용할 동일한 Fold Index를 생성한다."""
    if folds < 2:
        raise ValueError("Fold 수는 2 이상이어야 합니다.")
    if folds > len(target):
        raise ValueError("Fold 수는 Train Sample 수 이하여야 합니다.")

    splitter_random_state = random_state if shuffle else None
    if task_type == "classification":
        class_counts = target.value_counts(dropna=False)
        if class_counts.empty or int(class_counts.min()) < folds:
            raise ValueError(
                "Stratified K-Fold를 사용하려면 각 Class의 Train Sample 수가 Fold 수 이상이어야 합니다."
            )
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=shuffle,
            random_state=splitter_random_state,
        )
        return list(splitter.split(np.zeros(len(target)), target))

    splitter = KFold(
        n_splits=folds,
        shuffle=shuffle,
        random_state=splitter_random_state,
    )
    return list(splitter.split(np.zeros(len(target))))


def create_selection_pipeline(
    task_type: str,
    model_key: str,
    artifacts: PreprocessingArtifacts,
) -> Pipeline:
    """Fold 안에서 전처리와 모델이 함께 학습되는 Pipeline을 만든다."""
    spec = get_candidate_specs(task_type).get(model_key)
    if spec is None:
        raise ValueError(f"지원하지 않는 모델입니다: {model_key}")

    if task_type == "classification":
        estimators: dict[str, BaseEstimator] = {
            "logistic": LogisticRegression(max_iter=1000, random_state=42),
            "gaussian_nb": GaussianNB(),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "decision_tree": DecisionTreeClassifier(max_depth=3, random_state=42),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=3, random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42
            ),
            "svm": SVC(C=1.0, kernel="rbf", random_state=42),
        }
    else:
        estimators = {
            "linear": LinearRegression(),
            "polynomial": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=1.0, max_iter=10000),
            "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        }

    steps: list[tuple[str, BaseEstimator]] = [
        ("preprocessor", create_preprocessing_pipeline(artifacts)),
    ]
    already_scaled = artifacts.scaler is not None
    if spec.needs_scaling and not already_scaled:
        steps.append(("scaler", StandardScaler()))
    if model_key == "polynomial":
        steps.append(("polynomial", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("model", estimators[model_key]))
    return Pipeline(steps)


def get_scorer(task_type: str, scoring_key: str, positive_label: Any | None = None):
    """UI의 평가 기준을 scikit-learn Scorer로 변환한다."""
    if task_type == "regression":
        mapping = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
        return mapping[scoring_key]

    if scoring_key == "accuracy":
        return "accuracy"
    if scoring_key == "f1_macro":
        return "f1_macro"
    if scoring_key == "f1_weighted":
        return "f1_weighted"
    if positive_label is None:
        average = "weighted"
        metric = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[scoring_key]
        return make_scorer(metric, average=average, zero_division=0)
    metric = {"precision": precision_score, "recall": recall_score, "f1": f1_score}[scoring_key]
    return make_scorer(metric, pos_label=positive_label, zero_division=0)


def display_scores(values: np.ndarray, task_type: str, scoring_key: str) -> np.ndarray:
    """음수 오차 Scorer를 사용한 회귀 점수를 양수 오차로 표시한다."""
    array = np.asarray(values, dtype=float)
    if task_type == "regression" and scoring_key in {"mae", "rmse"}:
        return -array
    return array


def run_cross_validation(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    task_type: str,
    scoring_key: str,
    positive_label: Any | None = None,
) -> CrossValidationResult:
    """동일한 Fold에서 Train과 Validation 성능 및 학습 시간을 계산한다."""
    parameters = estimator.get_params()
    if "model__n_neighbors" in parameters:
        smallest_train_fold = min(len(train_index) for train_index, _ in splits)
        current_neighbors = int(parameters["model__n_neighbors"])
        if current_neighbors > smallest_train_fold:
            estimator = estimator.set_params(model__n_neighbors=smallest_train_fold)
    result = cross_validate(
        estimator,
        features,
        target,
        cv=splits,
        scoring=get_scorer(task_type, scoring_key, positive_label),
        return_train_score=True,
        error_score="raise",
    )
    train_scores = display_scores(result["train_score"], task_type, scoring_key)
    validation_scores = display_scores(result["test_score"], task_type, scoring_key)
    fit_times = np.asarray(result["fit_time"], dtype=float)
    table = pd.DataFrame({
        "Fold": [f"Fold {index + 1}" for index in range(len(splits))],
        "Train Score": train_scores,
        "Validation Score": validation_scores,
        "Fit Time": fit_times,
    })
    table.loc[len(table)] = [
        "Mean", float(np.mean(train_scores)), float(np.mean(validation_scores)),
        float(np.mean(fit_times)),
    ]
    table.loc[len(table)] = [
        "Std", float(np.std(train_scores)), float(np.std(validation_scores)),
        float(np.std(fit_times)),
    ]
    return CrossValidationResult(table, train_scores, validation_scores, fit_times)


def compare_candidates(
    task_type: str,
    model_keys: list[str],
    artifacts: PreprocessingArtifacts,
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    scoring_key: str,
    positive_label: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, CrossValidationResult]]:
    """여러 알고리즘을 같은 Fold와 Scoring으로 비교한다."""
    specs = get_candidate_specs(task_type)
    results: dict[str, CrossValidationResult] = {}
    rows = []
    for model_key in model_keys:
        started = perf_counter()
        cv_result = run_cross_validation(
            create_selection_pipeline(task_type, model_key, artifacts),
            features,
            target,
            splits,
            task_type=task_type,
            scoring_key=scoring_key,
            positive_label=positive_label,
        )
        results[model_key] = cv_result
        rows.append({
            "Model": specs[model_key].label,
            "Mean Validation": cv_result.validation_mean,
            "Std": cv_result.validation_std,
            "Mean Train": cv_result.train_mean,
            "Mean Fit Time": float(np.mean(cv_result.fit_times)),
            "Elapsed": perf_counter() - started,
        })
    ascending = task_type == "regression" and scoring_key in {"mae", "rmse"}
    frame = pd.DataFrame(rows).sort_values(
        "Mean Validation", ascending=ascending, ignore_index=True
    )
    return frame, results


def pipeline_parameter_grid(
    spec: CandidateSpec,
    selected_options: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    """표시용 Parameter 이름을 Pipeline 내부 이름으로 변환한다."""
    options = selected_options or spec.parameter_options
    prefix = "polynomial" if spec.key == "polynomial" else "model"
    return {f"{prefix}__{name}": list(values) for name, values in options.items()}


def parameter_combination_count(options: dict[str, list[Any]]) -> int:
    """Grid Search가 확인할 전체 후보 조합 수를 계산한다."""
    if not options:
        return 1
    count = 1
    for values in options.values():
        count *= len(values)
    return count


def run_parameter_search(
    estimator: Pipeline,
    spec: CandidateSpec,
    features: pd.DataFrame,
    target: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    task_type: str,
    scoring_key: str,
    selected_options: dict[str, list[Any]],
    search_method: str,
    n_iter: int = 10,
    positive_label: Any | None = None,
    random_state: int = 42,
) -> tuple[GridSearchCV | RandomizedSearchCV, pd.DataFrame]:
    """GridSearchCV 또는 RandomizedSearchCV를 실행하고 후보 결과표를 만든다."""
    grid = pipeline_parameter_grid(spec, selected_options)
    scorer = get_scorer(task_type, scoring_key, positive_label)
    if search_method == "grid":
        search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            estimator,
            param_grid=grid,
            cv=splits,
            scoring=scorer,
            return_train_score=True,
            refit=True,
            error_score="raise",
        )
    elif search_method == "random":
        total = parameter_combination_count(selected_options)
        search = RandomizedSearchCV(
            estimator,
            param_distributions=grid,
            n_iter=min(n_iter, total),
            cv=splits,
            scoring=scorer,
            return_train_score=True,
            refit=True,
            random_state=random_state,
            error_score="raise",
        )
    else:
        raise ValueError(f"지원하지 않는 탐색 방법입니다: {search_method}")

    search.fit(features, target)
    cv_results = search.cv_results_
    validation = display_scores(cv_results["mean_test_score"], task_type, scoring_key)
    train = display_scores(cv_results["mean_train_score"], task_type, scoring_key)
    std = np.asarray(cv_results["std_test_score"], dtype=float)
    params = [
        ", ".join(f"{key.split('__')[-1]}={value}" for key, value in values.items())
        or "기본 설정"
        for values in cv_results["params"]
    ]
    ascending = task_type == "regression" and scoring_key in {"mae", "rmse"}
    order = np.argsort(validation) if ascending else np.argsort(-validation)
    result_frame = pd.DataFrame({
        "Rank": np.arange(1, len(order) + 1),
        "Parameters": np.asarray(params, dtype=object)[order],
        "Mean Validation": validation[order],
        "Std": std[order],
        "Mean Train": train[order],
    })
    return search, result_frame

