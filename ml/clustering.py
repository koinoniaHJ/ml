"""K-Means 군집화 계산을 UI와 분리해 제공한다."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class KMeansResult:
    """학습된 K-Means와 변환 데이터, Cluster 및 Centroid를 보관한다."""

    model: KMeans
    scaler: StandardScaler | None
    feature_names: tuple[str, ...]
    transformed_features: pd.DataFrame
    labels: np.ndarray
    scaled_centers: pd.DataFrame
    original_centers: pd.DataFrame


def validate_clustering_data(features: pd.DataFrame, n_clusters: int) -> None:
    """K-Means에 전달할 Feature와 Cluster 수의 기본 조건을 확인한다."""
    if features.empty:
        raise ValueError("군집화에 사용할 Feature가 없습니다.")
    if len(features) < 2:
        raise ValueError("군집화를 확인하려면 두 개 이상의 Sample이 필요합니다.")

    non_numeric = [
        str(column)
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise TypeError("숫자형 Feature를 선택해 주세요: " + ", ".join(non_numeric))
    if features.isna().any().any():
        raise ValueError("Feature에 결측치가 있습니다. 먼저 결측치를 처리해 주세요.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature에 유한하지 않은 값이 있습니다.")
    distinct_samples = len(features.drop_duplicates())
    if not 1 <= n_clusters <= distinct_samples:
        raise ValueError(
            f"K는 1 이상 서로 다른 Sample 수({distinct_samples}) 이하여야 합니다."
        )


def fit_kmeans(
    features: pd.DataFrame,
    *,
    n_clusters: int,
    scale_features: bool = True,
    random_state: int = 42,
    n_init: int = 10,
) -> KMeansResult:
    """선택한 Feature를 필요에 따라 Scaling하고 K-Means를 학습한다."""
    validate_clustering_data(features, n_clusters)
    if n_init < 1:
        raise ValueError("n_init은 1 이상이어야 합니다.")

    feature_names = tuple(map(str, features.columns))
    source = features.loc[:, list(feature_names)].astype(float)
    scaler: StandardScaler | None = StandardScaler() if scale_features else None
    transformed_values = scaler.fit_transform(source) if scaler is not None else source.to_numpy()
    transformed = pd.DataFrame(
        transformed_values,
        index=source.index,
        columns=feature_names,
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(transformed)
    scaled_centers = pd.DataFrame(model.cluster_centers_, columns=feature_names)
    original_values = (
        scaler.inverse_transform(model.cluster_centers_)
        if scaler is not None
        else model.cluster_centers_
    )
    original_centers = pd.DataFrame(original_values, columns=feature_names)
    scaled_centers.index.name = "Cluster"
    original_centers.index.name = "Cluster"
    return KMeansResult(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        transformed_features=transformed,
        labels=np.asarray(labels, dtype=int),
        scaled_centers=scaled_centers,
        original_centers=original_centers,
    )


def create_clustered_frame(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """원본 Feature에 각 Sample의 Cluster 번호를 추가한다."""
    if len(features) != len(labels):
        raise ValueError("Feature와 Cluster Label의 Sample 수가 다릅니다.")
    result = features.copy(deep=True)
    result["cluster"] = np.asarray(labels, dtype=int)
    return result


def predict_cluster(
    result: KMeansResult,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """학습 때와 같은 Scaling 기준을 적용해 새로운 Sample의 Cluster를 예측한다."""
    missing = [name for name in result.feature_names if name not in new_data.columns]
    if missing:
        raise ValueError("새 데이터에 Feature가 없습니다: " + ", ".join(missing))
    source = new_data.loc[:, list(result.feature_names)].astype(float)
    if source.isna().any().any() or not np.isfinite(source.to_numpy()).all():
        raise ValueError("새 데이터에는 결측치나 유한하지 않은 값을 사용할 수 없습니다.")
    transformed_values = (
        result.scaler.transform(source)
        if result.scaler is not None
        else source.to_numpy()
    )
    transformed = pd.DataFrame(
        transformed_values,
        index=source.index,
        columns=result.feature_names,
    )
    return np.asarray(result.model.predict(transformed), dtype=int)
