import numpy as np
import pandas as pd
import pytest

from ml.clustering import (
    create_clustered_frame, fit_kmeans, predict_cluster,
    validate_clustering_data,
)
from ml.data_loader import load_builtin_dataset


def _features() -> pd.DataFrame:
    dataframe = load_builtin_dataset("Clustering Sample")
    return dataframe[["study_time", "attendance"]]


def test_kmeans_scales_features_and_creates_three_clusters():
    features = _features()
    result = fit_kmeans(features, n_clusters=3, random_state=42, n_init=10)

    assert sorted(np.bincount(result.labels).tolist()) == [5, 5, 5]
    assert np.allclose(result.transformed_features.mean().to_numpy(), 0.0)
    assert np.allclose(
        result.transformed_features.std(ddof=0).to_numpy(),
        1.0,
    )


def test_original_centers_match_cluster_feature_means():
    features = _features()
    result = fit_kmeans(features, n_clusters=3)

    for cluster in range(3):
        expected = features.loc[result.labels == cluster].mean().to_numpy()
        assert np.allclose(result.original_centers.loc[cluster].to_numpy(), expected)


def test_cluster_frame_and_new_sample_prediction():
    features = _features()
    result = fit_kmeans(features, n_clusters=3)
    frame = create_clustered_frame(features, result.labels)
    prediction = predict_cluster(
        result,
        pd.DataFrame({"study_time": [9], "attendance": [84]}),
    )

    assert frame.columns.tolist() == ["study_time", "attendance", "cluster"]
    assert prediction.shape == (1,)
    assert int(prediction[0]) in {0, 1, 2}


@pytest.mark.parametrize(
    ("features", "clusters", "message"),
    [
        (pd.DataFrame({"x": [1.0, np.nan]}), 1, "결측치"),
        (pd.DataFrame({"x": ["a", "b"]}), 1, "숫자형"),
        (pd.DataFrame({"x": [1.0, 2.0]}), 3, "Sample 수"),
    ],
)
def test_invalid_clustering_data_is_rejected(features, clusters, message):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_clustering_data(features, clusters)
