import numpy as np
import pandas as pd
import pytest

from ml.data_loader import load_builtin_dataset
from ml.preprocessing import (
    PreprocessingArtifacts, apply_preprocessing_pipeline, compare_scaler_fit,
    encode_target, get_class_ratios, impute_feature, one_hot_encode_feature,
    scale_features, split_dataset,
)


FEATURE_COLUMNS = ["study_time", "attendance", "study_place", "sleep_hours"]
TARGET_COLUMN = "result"


def _split_preprocessing_sample():
    dataframe = load_builtin_dataset("Preprocessing Sample")
    return split_dataset(
        dataframe,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        test_size=0.2,
        random_state=42,
        use_stratify=True,
    )


def test_split_dataset_preserves_sample_count_and_class_ratio() -> None:
    train_data, test_data, train_target, test_target = _split_preprocessing_sample()

    assert len(train_data) == 16
    assert len(test_data) == 4
    assert get_class_ratios(train_target) == {"fail": 50.0, "pass": 50.0}
    assert get_class_ratios(test_target) == {"fail": 50.0, "pass": 50.0}


def test_imputer_removes_missing_values_using_train_statistic() -> None:
    train_data, test_data, _, _ = _split_preprocessing_sample()
    train_result, test_result, statistic, _ = impute_feature(
        train_data,
        test_data,
        "study_time",
        "mean",
    )

    assert train_result["study_time"].isna().sum() == 0
    assert test_result["study_time"].isna().sum() == 0
    assert np.isclose(statistic, train_data["study_time"].mean())


def test_label_and_one_hot_encoding_results() -> None:
    train_data, test_data, train_target, test_target = _split_preprocessing_sample()
    train_data, test_data, _, _ = impute_feature(
        train_data,
        test_data,
        "study_place",
        "most_frequent",
    )

    encoded_train_target, encoded_test_target, mapping, _ = encode_target(
        train_target,
        test_target,
    )
    encoded_train, encoded_test, encoded_columns, _ = one_hot_encode_feature(
        train_data,
        test_data,
        "study_place",
    )

    assert mapping == {"fail": 0, "pass": 1}
    assert set(encoded_train_target.unique()) == {0, 1}
    assert set(encoded_test_target.unique()) == {0, 1}
    assert encoded_columns == [
        "study_place_cafe",
        "study_place_home",
        "study_place_library",
    ]
    assert encoded_train.shape == (16, 6)
    assert encoded_test.shape == (4, 6)


def test_scaler_keeps_shape_and_fits_train_data_only() -> None:
    train_data, test_data, _, _ = _split_preprocessing_sample()
    train_result, test_result, scaler = scale_features(
        train_data,
        test_data,
        ["sleep_hours"],
        "standard",
    )

    assert train_result.shape == train_data.shape
    assert test_result.shape == test_data.shape
    assert scaler is not None
    assert np.isclose(scaler.mean_[0], train_data["sleep_hours"].mean())
    assert not np.isclose(scaler.mean_[0], load_builtin_dataset("Preprocessing Sample")["sleep_hours"].mean())


def test_data_leakage_demo_uses_different_fit_ranges() -> None:
    train_data, test_data, _, _ = _split_preprocessing_sample()
    comparison = compare_scaler_fit(
        train_data["sleep_hours"],
        test_data["sleep_hours"],
    )

    assert not np.isclose(comparison["train_mean"], comparison["full_mean"])
    assert not np.isclose(comparison["safe_test_first"], comparison["leaked_test_first"])


def test_split_dataset_rejects_missing_target() -> None:
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "target": [10, np.nan, 30]})

    with pytest.raises(ValueError, match="Target에 결측치가 1개"):
        split_dataset(
            dataframe,
            ["feature"],
            "target",
            test_size=0.33,
            random_state=42,
            use_stratify=False,
        )


def test_preprocessing_artifacts_reuse_train_transformers() -> None:
    raw_train, raw_test, _, _ = _split_preprocessing_sample()
    train_data = raw_train.copy()
    test_data = raw_test.copy()
    imputers = {}
    for column, strategy in (
        ("study_time", "mean"),
        ("attendance", "mean"),
        ("study_place", "most_frequent"),
    ):
        train_data, test_data, _, imputer = impute_feature(
            train_data,
            test_data,
            column,
            strategy,
        )
        imputers[column] = imputer
    encoded_train, encoded_test, _, encoder = one_hot_encode_feature(
        train_data,
        test_data,
        "study_place",
    )
    scaled_train, _, scaler = scale_features(
        encoded_train,
        encoded_test,
        ["study_time", "attendance", "sleep_hours"],
        "standard",
    )
    artifacts = PreprocessingArtifacts(
        input_features=tuple(FEATURE_COLUMNS),
        output_features=tuple(scaled_train.columns),
        feature_imputers=imputers,
        feature_encoders={"study_place": encoder},
        scaler=scaler,
        scaler_columns=("study_time", "attendance", "sleep_hours"),
    )
    new_data = pd.DataFrame({
        "study_time": [5.0],
        "attendance": [88.0],
        "study_place": ["new_place"],
        "sleep_hours": [7.0],
    })

    pipeline_train, _, preprocessor = apply_preprocessing_pipeline(
        raw_train,
        raw_test,
        artifacts,
    )
    artifacts.preprocessor = preprocessor
    artifacts.output_features = tuple(pipeline_train.columns)
    transformed = artifacts.transform_features(new_data)

    assert set(pipeline_train.columns) == set(scaled_train.columns)
    for column in scaled_train.columns:
        assert np.allclose(
            pipeline_train[column].to_numpy(dtype=float),
            scaled_train[column].to_numpy(dtype=float),
        )
    assert transformed.columns.tolist() == pipeline_train.columns.tolist()
    assert transformed.isna().sum().sum() == 0
    assert transformed.filter(like="study_place_").to_numpy().sum() == 0
