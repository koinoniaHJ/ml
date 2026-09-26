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
