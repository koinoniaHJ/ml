# Train/Test 분리와 전처리 실습에 사용하는 계산을 UI와 분리해 제공
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


def split_dataset(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    test_size: float,
    random_state: int,
    use_stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Feature와 Target을 같은 기준으로 Train/Test Data로 나눈다."""
    features = dataframe.loc[:, list(feature_columns)].copy()
    target = dataframe.loc[:, target_column].copy()
    stratify_target = target if use_stratify else None

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def get_class_ratios(target: pd.Series) -> dict[str, float]:
    """Target의 Class별 비율을 백분율로 계산한다."""
    ratios = target.value_counts(normalize=True, dropna=False).sort_index() * 100
    return {str(class_name): float(ratio) for class_name, ratio in ratios.items()}


def impute_feature(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    column: str,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Train Data에서 결측치 처리 기준을 학습해 Train/Test Data에 적용한다."""
    imputer = SimpleImputer(strategy=strategy)
    train_result = train_data.copy()
    test_result = test_data.copy()

    train_result[column] = imputer.fit_transform(train_data[[column]]).ravel()
    test_result[column] = imputer.transform(test_data[[column]]).ravel()

    return train_result, test_result, imputer.statistics_[0]


def encode_target(
    train_target: pd.Series,
    test_target: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Train Target의 문자열 Class를 학습해 Train/Test Target을 숫자로 변환한다."""
    encoder = LabelEncoder()
    train_values = encoder.fit_transform(train_target)
    test_values = encoder.transform(test_target)
    mapping = {str(class_name): int(index) for index, class_name in enumerate(encoder.classes_)}

    train_result = pd.Series(train_values, index=train_target.index, name=train_target.name)
    test_result = pd.Series(test_values, index=test_target.index, name=test_target.name)
    return train_result, test_result, mapping


def one_hot_encode_feature(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train Data의 범주를 기준으로 범주형 Feature를 One-Hot Encoding한다."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_values = encoder.fit_transform(train_data[[column]])
    test_values = encoder.transform(test_data[[column]])
    encoded_columns = encoder.get_feature_names_out([column]).tolist()

    train_encoded = pd.DataFrame(train_values, index=train_data.index, columns=encoded_columns)
    test_encoded = pd.DataFrame(test_values, index=test_data.index, columns=encoded_columns)

    train_result = pd.concat([train_data.drop(columns=column), train_encoded], axis=1)
    test_result = pd.concat([test_data.drop(columns=column), test_encoded], axis=1)
    return train_result, test_result, encoded_columns


def scale_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    columns: Sequence[str],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | MinMaxScaler | None]:
    """Train Data에서 Scaling 기준을 학습해 Train/Test Data에 적용한다."""
    train_result = train_data.copy()
    test_result = test_data.copy()

    if method not in {"none", "standard", "minmax"}:
        raise ValueError(f"지원하지 않는 Scaling 방법입니다: {method}")

    if method == "none" or not columns:
        return train_result, test_result, None

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    selected_columns = list(columns)
    train_result[selected_columns] = scaler.fit_transform(train_data[selected_columns])
    test_result[selected_columns] = scaler.transform(test_data[selected_columns])
    return train_result, test_result, scaler


def compare_scaler_fit(
    train_values: pd.Series,
    test_values: pd.Series,
) -> dict[str, float]:
    """Train 전용 fit과 전체 데이터 fit에서 생기는 StandardScaler 기준 차이를 계산한다."""
    train_frame = train_values.to_numpy(dtype=float).reshape(-1, 1)
    test_frame = test_values.to_numpy(dtype=float).reshape(-1, 1)
    full_frame = np.vstack([train_frame, test_frame])

    train_scaler = StandardScaler().fit(train_frame)
    full_scaler = StandardScaler().fit(full_frame)

    return {
        "train_mean": float(train_scaler.mean_[0]),
        "full_mean": float(full_scaler.mean_[0]),
        "train_scale": float(train_scaler.scale_[0]),
        "full_scale": float(full_scaler.scale_[0]),
        "safe_test_first": float(train_scaler.transform(test_frame)[0, 0]),
        "leaked_test_first": float(full_scaler.transform(test_frame)[0, 0]),
    }
