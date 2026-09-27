# 회귀 모델 학습, 예측, 평가 계산을 UI와 분리해 제공
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


SUPPORTED_MODELS = {
    "linear": "Linear Regression",
    "polynomial": "Polynomial Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic_net": "ElasticNet",
}


@dataclass(frozen=True)
class RegressionMetrics:
    """회귀 평가 지표를 한 묶음으로 보관한다."""

    mae: float
    mse: float
    rmse: float
    r2: float


def create_regressor(
    model_name: str,
    alpha: float = 1.0,
    degree: int = 2,
    l1_ratio: float = 0.5,
) -> LinearRegression | Ridge | Lasso | ElasticNet | Pipeline:
    """선택한 이름과 하이퍼파라미터로 회귀 모델을 생성한다."""
    if model_name == "linear":
        return LinearRegression()

    if model_name == "polynomial":
        if degree < 2:
            raise ValueError("degree는 2 이상이어야 합니다.")
        return Pipeline([
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ])

    if model_name in {"ridge", "lasso", "elastic_net"}:
        if alpha < 0:
            raise ValueError("alpha는 0 이상이어야 합니다.")

    if model_name == "ridge":
        return Ridge(alpha=alpha)

    if model_name == "lasso":
        return Lasso(alpha=alpha, max_iter=10000)

    if model_name == "elastic_net":
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio는 0 이상 1 이하여야 합니다.")
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    raise ValueError(f"지원하지 않는 회귀 모델입니다: {model_name}")


def validate_regression_data(features: pd.DataFrame, target: pd.Series) -> None:
    """scikit-learn 회귀 모델에 전달할 데이터 형태를 확인한다."""
    if features.empty or target.empty:
        raise ValueError("학습할 데이터가 없습니다.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("모든 Feature를 숫자 형태로 변환해 주세요.")

    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError("회귀 Target은 숫자형이어야 합니다.")

    if features.isna().any().any() or target.isna().any():
        raise ValueError("결측치를 먼저 처리해 주세요.")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature에 유한하지 않은 값이 있습니다.")

    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target에 유한하지 않은 값이 있습니다.")


def calculate_regression_metrics(
    actual: pd.Series | np.ndarray,
    prediction: np.ndarray,
) -> RegressionMetrics:
    """실제값과 예측값으로 대표적인 회귀 평가 지표를 계산한다."""
    actual_values = np.asarray(actual, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)

    r2 = (
        float(r2_score(actual_values, prediction_values))
        if len(actual_values) >= 2
        else float("nan")
    )
    return RegressionMetrics(
        mae=float(mean_absolute_error(actual_values, prediction_values)),
        mse=float(mean_squared_error(actual_values, prediction_values)),
        rmse=float(root_mean_squared_error(actual_values, prediction_values)),
        r2=r2,
    )


def create_mean_baseline(train_target: pd.Series, sample_count: int) -> np.ndarray:
    """Train Target 평균을 모든 Sample에 예측하는 Baseline을 생성한다."""
    return np.full(sample_count, float(train_target.mean()), dtype=float)
