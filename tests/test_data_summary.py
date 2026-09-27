import numpy as np
import pandas as pd

from ml.data_summary import (
    get_data_quality_summary,
    get_numeric_statistics,
    get_target_distribution,
)


def test_data_quality_summary_counts_missing_and_duplicate_samples() -> None:
    dataframe = pd.DataFrame({
        "score": [80.0, np.nan, 80.0],
        "place": ["home", "cafe", "home"],
    })

    summary = get_data_quality_summary(dataframe)

    assert summary == {
        "missing_rows": 1,
        "duplicate_rows": 1,
        "numeric_columns": 1,
        "categorical_columns": 1,
    }


def test_numeric_statistics_excludes_identifier_columns() -> None:
    dataframe = pd.DataFrame({
        "student_id": [1, 2, 3],
        "score": [70, 80, 90],
    })

    statistics = get_numeric_statistics(dataframe)

    assert statistics["Column"].tolist() == ["score"]
    assert statistics.loc[0, "Mean"] == 80
    assert statistics.loc[0, "Median"] == 80


def test_target_distribution_includes_missing_values() -> None:
    dataframe = pd.DataFrame({"result": ["pass", "fail", "pass", None]})

    distribution = get_target_distribution(dataframe, "result")

    counts = dict(zip(distribution["Target Value"], distribution["Count"]))
    assert counts == {"pass": 2, "fail": 1, "Missing": 1}
    assert np.isclose(distribution["Ratio (%)"].sum(), 100.0)
