# Train/Test 분리와 전처리 실습에 사용하는 계산을 UI와 분리해 제공
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass
class PreprocessingArtifacts:
    """Train Data에서 학습한 전처리 기준을 새로운 데이터에 재사용한다."""

    input_features: tuple[str, ...] = ()
    output_features: tuple[str, ...] = ()
    feature_imputers: dict[str, SimpleImputer] = field(default_factory=dict)
    feature_encoders: dict[str, OneHotEncoder] = field(default_factory=dict)
    target_encoder: LabelEncoder | None = None
    scaler: StandardScaler | MinMaxScaler | None = None
    scaler_columns: tuple[str, ...] = ()
    preprocessor: ColumnTransformer | None = None

    def transform_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """저장된 Train 전처리 기준을 새로운 Feature 데이터에 적용한다."""
        missing_columns = [
            column for column in self.input_features if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                "새로운 데이터에 필요한 Feature가 없습니다: "
                + ", ".join(missing_columns)
            )

        source = dataframe.loc[:, list(self.input_features)].copy()
        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(source)
            if isinstance(transformed, pd.DataFrame):
                return transformed
            return pd.DataFrame(
                transformed,
                index=source.index,
                columns=list(self.output_features),
            )

        result = source

        for column, imputer in self.feature_imputers.items():
            result[column] = imputer.transform(result[[column]]).ravel()

        for column, encoder in self.feature_encoders.items():
            if column not in result.columns:
                continue
            encoded_columns = encoder.get_feature_names_out([column]).tolist()
            encoded_values = encoder.transform(result[[column]])
            encoded = pd.DataFrame(
                encoded_values,
                index=result.index,
                columns=encoded_columns,
            )
            result = pd.concat([result.drop(columns=column), encoded], axis=1)

        if self.scaler is not None and self.scaler_columns:
            columns = list(self.scaler_columns)
            result[columns] = self.scaler.transform(result[columns])

        if self.output_features:
            result = result.reindex(columns=list(self.output_features), fill_value=0.0)
        return result

    def transform_target(self, target: pd.Series) -> pd.Series:
        """저장된 LabelEncoder가 있으면 새로운 Target에 같은 Class 번호를 적용한다."""
        if self.target_encoder is None:
            return target.copy()
        values = self.target_encoder.transform(target)
        return pd.Series(values, index=target.index, name=target.name)


@dataclass(frozen=True)
class PreprocessingResult:
    """다음 학습 Stage로 전달하는 현재 Train/Test 데이터와 전처리 기준이다."""

    dataset_name: str
    target_column: str | None
    task_type: str
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    artifacts: PreprocessingArtifacts = field(default_factory=PreprocessingArtifacts)
    raw_x_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_x_test: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_y_train: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    raw_y_test: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))


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
    missing_target_count = int(target.isna().sum())
    if missing_target_count:
        raise ValueError(
            f"Target에 결측치가 {missing_target_count}개 있습니다. "
            "Target 값을 확인하거나 해당 Sample을 제외해 주세요."
        )
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
) -> tuple[pd.DataFrame, pd.DataFrame, object, SimpleImputer]:
    """Train Data에서 결측치 처리 기준을 학습해 Train/Test Data에 적용한다."""
    imputer = SimpleImputer(strategy=strategy)
    train_result = train_data.copy()
    test_result = test_data.copy()

    train_result[column] = imputer.fit_transform(train_data[[column]]).ravel()
    test_result[column] = imputer.transform(test_data[[column]]).ravel()

    return train_result, test_result, imputer.statistics_[0], imputer


def encode_target(
    train_target: pd.Series,
    test_target: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, int], LabelEncoder]:
    """Train Target의 문자열 Class를 학습해 Train/Test Target을 숫자로 변환한다."""
    encoder = LabelEncoder()
    train_values = encoder.fit_transform(train_target)
    test_values = encoder.transform(test_target)
    mapping = {str(class_name): int(index) for index, class_name in enumerate(encoder.classes_)}

    train_result = pd.Series(train_values, index=train_target.index, name=train_target.name)
    test_result = pd.Series(test_values, index=test_target.index, name=test_target.name)
    return train_result, test_result, mapping, encoder


def one_hot_encode_feature(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], OneHotEncoder]:
    """Train Data의 범주를 기준으로 범주형 Feature를 One-Hot Encoding한다."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_values = encoder.fit_transform(train_data[[column]])
    test_values = encoder.transform(test_data[[column]])
    encoded_columns = encoder.get_feature_names_out([column]).tolist()

    train_encoded = pd.DataFrame(train_values, index=train_data.index, columns=encoded_columns)
    test_encoded = pd.DataFrame(test_values, index=test_data.index, columns=encoded_columns)

    train_result = pd.concat([train_data.drop(columns=column), train_encoded], axis=1)
    test_result = pd.concat([test_data.drop(columns=column), test_encoded], axis=1)
    return train_result, test_result, encoded_columns, encoder


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


def create_preprocessing_pipeline(
    artifacts: PreprocessingArtifacts,
) -> ColumnTransformer:
    """현재 단계에서 선택한 변환을 Column별 Pipeline으로 구성한다."""
    transformers = []
    for index, column in enumerate(artifacts.input_features):
        steps = []
        if column in artifacts.feature_imputers:
            steps.append(("imputer", clone(artifacts.feature_imputers[column])))
        if column in artifacts.feature_encoders:
            steps.append(("encoder", clone(artifacts.feature_encoders[column])))
        if artifacts.scaler is not None and column in artifacts.scaler_columns:
            steps.append(("scaler", clone(artifacts.scaler)))
        if steps:
            transformers.append((f"feature_{index}", Pipeline(steps), [column]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def apply_preprocessing_pipeline(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Train Data에서 Pipeline을 학습하고 Test Data에는 같은 기준을 적용한다."""
    input_columns = list(artifacts.input_features)
    train_source = train_data.loc[:, input_columns].copy()
    test_source = test_data.loc[:, input_columns].copy()
    preprocessor = create_preprocessing_pipeline(artifacts)
    train_result = preprocessor.fit_transform(train_source)
    test_result = preprocessor.transform(test_source)
    train_result.index = train_source.index
    test_result.index = test_source.index
    return train_result, test_result, preprocessor


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
