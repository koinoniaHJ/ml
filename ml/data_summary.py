# 데이터의 기본 통계와 Target, Feature, Column 정보를 계산
import pandas as pd


# DataFrame의 기본 정보를 계산
def get_data_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    sample_count, column_count = dataframe.shape
    missing_count = int(dataframe.isna().sum().sum())
    dtypes = dataframe.dtypes

    return {
        "sample_count": sample_count,
        "column_count": column_count,
        "missing_count": missing_count,
        "dtypes": dtypes,
    }


# Column 이름을 기준으로 식별자 Column인지 확인
def is_identifier_column(column: str) -> bool:
    column = str(column).lower()
    return column == "id" or column.endswith("_id")


# 식별자 Column을 반환
def get_identifier_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in dataframe.columns if is_identifier_column(column)]


# 식별자 Column을 제외한 Target 후보를 반환
def get_target_columns(dataframe: pd.DataFrame) -> list[str]:
    identifier_columns = get_identifier_columns(dataframe)
    return [column for column in dataframe.columns if column not in identifier_columns]


# 선택한 Target을 제외한 Column을 Feature 후보로 반환
def get_feature_columns(dataframe: pd.DataFrame, target_column: str | None) -> list[str]:
    identifier_columns = get_identifier_columns(dataframe)
    return [
        column
        for column in dataframe.columns
        if column != target_column and column not in identifier_columns
    ]


# 선택한 Target에 서로 다른 값이 몇 개인지 계산
def get_target_unique_count(dataframe: pd.DataFrame, target_column: str) -> int:
    return int(dataframe[target_column].nunique())


# 그래프에 사용할 숫자형 Column만 반환
def get_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include="number").columns.tolist()


# 숫자형 Column의 대표적인 기술 통계를 표 형태로 계산
def get_numeric_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in get_numeric_columns(dataframe)
        if not is_identifier_column(column)
    ]
    columns = ["Column", "Count", "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max"]
    if not numeric_columns:
        return pd.DataFrame(columns=columns)

    numeric = dataframe[numeric_columns]
    rows = []
    for column in numeric_columns:
        values = numeric[column].dropna()
        rows.append({
            "Column": str(column),
            "Count": int(values.count()),
            "Mean": values.mean(),
            "Median": values.median(),
            "Std": values.std(),
            "Min": values.min(),
            "Q1": values.quantile(0.25),
            "Q3": values.quantile(0.75),
            "Max": values.max(),
        })
    return pd.DataFrame(rows, columns=columns)


# 결측 Sample, 중복 Sample과 Column 종류를 요약
def get_data_quality_summary(dataframe: pd.DataFrame) -> dict[str, int]:
    numeric_count = len(get_numeric_columns(dataframe))
    return {
        "missing_rows": int(dataframe.isna().any(axis=1).sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numeric_columns": numeric_count,
        "categorical_columns": int(len(dataframe.columns) - numeric_count),
    }


# 선택한 Target의 값별 개수와 전체 대비 비율을 계산
def get_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> pd.DataFrame:
    columns = ["Target Value", "Count", "Ratio (%)"]
    if target_column is None or target_column not in dataframe.columns:
        return pd.DataFrame(columns=columns)

    display_target = dataframe[target_column].astype(object).where(
        dataframe[target_column].notna(),
        "Missing",
    )
    counts = display_target.value_counts(dropna=False)
    rows = [
        {
            "Target Value": str(value),
            "Count": int(count),
            "Ratio (%)": float(count / len(dataframe) * 100) if len(dataframe) else 0.0,
        }
        for value, count in counts.items()
    ]
    return pd.DataFrame(rows, columns=columns)
