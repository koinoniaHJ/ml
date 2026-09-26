import numpy as np

from ml.data_loader import load_builtin_dataset
from ml.preprocessing import (
    compare_scaler_fit, encode_target, get_class_ratios, impute_feature,
    one_hot_encode_feature, scale_features, split_dataset,
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
    train_result, test_result, statistic = impute_feature(
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
    train_data, test_data, _ = impute_feature(
        train_data,
        test_data,
        "study_place",
        "most_frequent",
    )

    encoded_train_target, encoded_test_target, mapping = encode_target(
        train_target,
        test_target,
    )
    encoded_train, encoded_test, encoded_columns = one_hot_encode_feature(
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
