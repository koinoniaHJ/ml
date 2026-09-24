# 데이터의 기본 통계와 Target,Feature, Column 정보를 계산
import pandas as pd


# DataFrame의 기본 정보를 계산
def get_data_summary(dataframe: pd.DataFrame) -> dict:
    sample_count, column_count = dataframe.shape

    # isna(): 각 값이 결측값인지 확인
    # sum(): True인 값의 개수를 합산
    missing_count = int(dataframe.isna().sum().sum())

    # dtypes: 각 Column의 데이터 타입 확인
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

    # endswith(): 문자열이 지정한 문자로 끝나는지 확인
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
    # nunique(): 서로 다른 값의 개수를 계산하는 메서드
    return int(dataframe[target_column].nunique())


# 그래프에 사용할 숫자형 Column만 반환
def get_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    # select_dtypes(): 지정한 데이터 타입에 해당하는 Column만 선택
    return dataframe.select_dtypes(include="number").columns.tolist()
