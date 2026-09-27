import numpy as np
import pandas as pd
import pytest

from ml.classification import (
    calculate_classification_metrics,
    create_classification_report,
    create_classifier,
    create_confusion_frame,
    create_majority_baseline,
    validate_classification_data,
)


@pytest.fixture
def classification_data():
    x = pd.DataFrame({
        "study_time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "attendance": [55, 60, 62, 68, 72, 76, 80, 84, 88, 92, 96, 99],
    })
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    return x, y


@pytest.mark.parametrize(
    "model_name,kwargs",
    [
        ("logistic", {"scale_features": True}),
        ("gaussian_nb", {}),
        ("knn", {"scale_features": True, "n_neighbors": 3}),
        ("decision_tree", {"max_depth": 3}),
        ("random_forest", {"n_estimators": 20, "max_depth": 3}),
        ("gradient_boosting", {"n_estimators": 20, "max_depth": 2}),
        ("svm", {"scale_features": True, "kernel": "linear"}),
    ],
)
def test_supported_classifiers_predict_class_and_probability(
    classification_data, model_name, kwargs
):
    x, y = classification_data
    model = create_classifier(model_name, **kwargs)
    model.fit(x, y)

    prediction = model.predict(x.iloc[:3])
    probability = model.predict_proba(x.iloc[:3])

    assert prediction.shape == (3,)
    assert probability.shape == (3, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_metrics_use_selected_positive_class():
    actual = [0, 0, 0, 1, 1]
    prediction = [0, 0, 1, 0, 1]

    metrics = calculate_classification_metrics(
        actual, prediction, positive_label=1
    )

    assert metrics.accuracy == pytest.approx(0.6)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


def test_confusion_matrix_uses_actual_rows_and_prediction_columns():
    frame = create_confusion_frame(
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 1],
    )

    assert frame.loc["Actual 0", "Pred 0"] == 2
    assert frame.loc["Actual 0", "Pred 1"] == 1
    assert frame.loc["Actual 1", "Pred 0"] == 1
    assert frame.loc["Actual 1", "Pred 1"] == 1


def test_report_contains_class_and_average_rows():
    report = create_classification_report(
        [0, 0, 1, 1], [0, 1, 1, 1], [0, 1]
    )

    assert report["Class"].tolist() == ["0", "1", "Macro Avg", "Weighted Avg"]
    assert report["Support"].tolist() == [2, 2, 4, 4]


def test_majority_baseline_learns_only_train_target(classification_data):
    x, _ = classification_data
    train_x = x.iloc[:5]
    test_x = x.iloc[5:8]
    train_y = pd.Series([0, 0, 0, 1, 1])

    prediction, baseline = create_majority_baseline(
        train_x, train_y, test_x
    )

    assert prediction.tolist() == [0, 0, 0]
    assert baseline.classes_.tolist() == [0, 1]


def test_validation_rejects_string_and_missing_features(classification_data):
    x, y = classification_data
    string_x = x.copy()
    string_x["place"] = "home"
    with pytest.raises(TypeError, match="Encoding"):
        validate_classification_data(string_x, y)

    missing_x = x.copy()
    missing_x.loc[0, "study_time"] = np.nan
    with pytest.raises(ValueError, match="결측치"):
        validate_classification_data(missing_x, y)

