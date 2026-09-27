import numpy as np
import pandas as pd
import pytest

from ml.data_loader import load_builtin_dataset
from ml.model_selection import (
    CLASSIFICATION_SPECS, REGRESSION_SPECS, compare_candidates,
    create_selection_pipeline, create_validation_splits, display_scores,
    parameter_combination_count, run_cross_validation, run_parameter_search,
)
from ml.preprocessing import (
    PreprocessingArtifacts, one_hot_encode_feature, split_dataset,
)


def _classification_data():
    dataframe = load_builtin_dataset("Classification Sample")
    features = [column for column in dataframe.columns if column not in {"customer_id", "purchased"}]
    train_x, test_x, train_y, test_y = split_dataset(
        dataframe,
        features,
        "purchased",
        test_size=0.25,
        random_state=42,
        use_stratify=True,
    )
    _, _, _, encoder = one_hot_encode_feature(
        train_x, test_x, "member_type"
    )
    artifacts = PreprocessingArtifacts(
        input_features=tuple(features),
        feature_encoders={"member_type": encoder},
    )
    return train_x, test_x, train_y, test_y, artifacts


def test_stratified_splits_keep_every_sample_in_validation_once():
    train_x, _, train_y, _, _ = _classification_data()
    splits = create_validation_splits(
        "classification", train_y, 3, random_state=42
    )

    validation_indices = np.concatenate([validation for _, validation in splits])

    assert len(splits) == 3
    assert sorted(validation_indices.tolist()) == list(range(len(train_x)))
    assert all(set(train_y.iloc[validation].unique()) == {0, 1} for _, validation in splits)


def test_stratified_splits_reject_too_many_folds():
    _, _, train_y, _, _ = _classification_data()

    with pytest.raises(ValueError, match="각 Class"):
        create_validation_splits("classification", train_y, 5)


def test_cross_validation_fits_raw_categorical_data_inside_pipeline():
    train_x, _, train_y, _, artifacts = _classification_data()
    splits = create_validation_splits("classification", train_y, 3)
    pipeline = create_selection_pipeline(
        "classification", "logistic", artifacts
    )

    result = run_cross_validation(
        pipeline,
        train_x,
        train_y,
        splits,
        task_type="classification",
        scoring_key="accuracy",
    )

    assert len(result.validation_scores) == 3
    assert result.fold_table["Fold"].tolist()[-2:] == ["Mean", "Std"]
    assert np.all((0 <= result.validation_scores) & (result.validation_scores <= 1))


def test_model_comparison_reuses_same_folds():
    train_x, _, train_y, _, artifacts = _classification_data()
    splits = create_validation_splits("classification", train_y, 3)

    frame, results = compare_candidates(
        "classification",
        ["logistic", "decision_tree"],
        artifacts,
        train_x,
        train_y,
        splits,
        scoring_key="accuracy",
    )

    assert set(frame["Model"]) == {"Logistic Regression", "Decision Tree"}
    assert set(results) == {"logistic", "decision_tree"}
    assert all(len(result.validation_scores) == 3 for result in results.values())


def test_grid_search_refits_best_estimator_for_final_test():
    train_x, test_x, train_y, _, artifacts = _classification_data()
    splits = create_validation_splits("classification", train_y, 3)
    pipeline = create_selection_pipeline("classification", "knn", artifacts)

    search, frame = run_parameter_search(
        pipeline,
        CLASSIFICATION_SPECS["knn"],
        train_x,
        train_y,
        splits,
        task_type="classification",
        scoring_key="accuracy",
        selected_options={
            "n_neighbors": [1, 3],
            "weights": ["uniform", "distance"],
        },
        search_method="grid",
    )

    assert len(frame) == 4
    assert search.best_estimator_ is not None
    assert len(search.best_estimator_.predict(test_x)) == len(test_x)


def test_regression_error_scores_are_displayed_as_positive_values():
    values = np.array([-3.0, -4.0])

    assert display_scores(values, "regression", "mae").tolist() == [3.0, 4.0]
    assert display_scores(values, "regression", "rmse").tolist() == [3.0, 4.0]
    assert display_scores(values, "regression", "r2").tolist() == [-3.0, -4.0]


def test_parameter_combination_count():
    assert parameter_combination_count({"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}) == 6
    assert parameter_combination_count(REGRESSION_SPECS["linear"].parameter_options) == 1

