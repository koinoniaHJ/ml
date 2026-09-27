import numpy as np
import pandas as pd

from ml.regression import (
    calculate_regression_metrics,
    create_mean_baseline,
    create_regressor,
    validate_regression_data,
)


def test_linear_regression_fits_exact_linear_relationship() -> None:
    train_x = pd.DataFrame({"study_time": [1, 2, 3, 4, 5, 6]})
    train_y = pd.Series([52, 57, 62, 67, 72, 77], name="score")
    test_x = pd.DataFrame({"study_time": [7, 8]})
    test_y = pd.Series([82, 87], name="score")

    model = create_regressor("linear")
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    metrics = calculate_regression_metrics(test_y, prediction)

    assert np.allclose(prediction, [82, 87])
    assert np.isclose(model.coef_[0], 5.0)
    assert np.isclose(model.intercept_, 47.0)
    assert np.isclose(metrics.mae, 0.0)
    assert np.isclose(metrics.rmse, 0.0)
    assert np.isclose(metrics.r2, 1.0)


def test_ridge_uses_alpha_and_shrinks_coefficient() -> None:
    train_x = pd.DataFrame({"feature": [-2, -1, 0, 1, 2]})
    train_y = pd.Series([-4, -2, 0, 2, 4])

    linear = create_regressor("linear").fit(train_x, train_y)
    ridge = create_regressor("ridge", alpha=10.0).fit(train_x, train_y)

    assert abs(ridge.coef_[0]) < abs(linear.coef_[0])


def test_polynomial_regression_fits_quadratic_relationship() -> None:
    train_x = pd.DataFrame({"feature": [-2, -1, 0, 1, 2]})
    train_y = pd.Series([4, 1, 0, 1, 4])

    model = create_regressor("polynomial", degree=2).fit(train_x, train_y)
    prediction = model.predict(pd.DataFrame({"feature": [3]}))

    assert np.allclose(prediction, [9.0])


def test_lasso_and_elastic_net_use_regularization_parameters() -> None:
    lasso = create_regressor("lasso", alpha=0.1)
    elastic_net = create_regressor("elastic_net", alpha=0.1, l1_ratio=0.25)

    assert np.isclose(lasso.alpha, 0.1)
    assert np.isclose(elastic_net.alpha, 0.1)
    assert np.isclose(elastic_net.l1_ratio, 0.25)


def test_mean_baseline_uses_train_target_mean() -> None:
    train_y = pd.Series([10.0, 20.0, 30.0])

    prediction = create_mean_baseline(train_y, sample_count=2)

    assert np.array_equal(prediction, np.array([20.0, 20.0]))


def test_r2_is_not_calculated_from_one_test_sample() -> None:
    metrics = calculate_regression_metrics(
        np.array([80.0]),
        np.array([78.0]),
    )

    assert np.isnan(metrics.r2)


def test_regression_data_requires_numeric_values_without_missing_data() -> None:
    features = pd.DataFrame({"study_place": ["home", "library"]})
    target = pd.Series([70.0, 80.0])

    try:
        validate_regression_data(features, target)
    except ValueError as error:
        assert "숫자 형태" in str(error)
    else:
        raise AssertionError("문자열 Feature를 허용했습니다.")
