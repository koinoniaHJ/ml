from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

from common.paths import DATASETS_DIR


BUILTIN_DATASETS = {
    "Student Basic": DATASETS_DIR / "student_basic.csv",
    "Student Dirty": DATASETS_DIR / "student_dirty.csv",
    "Regression Sample": DATASETS_DIR / "regression_sample.csv",
    "Classification Sample": DATASETS_DIR / "classification_sample.csv",
    "Clustering Sample": DATASETS_DIR / "clustering_sample.csv",
    "Final Challenge": DATASETS_DIR / "final_challenge.csv",
}

BUILTIN_TARGETS = {
    "Student Basic": "passed",
    "Student Dirty": "passed",
    "Regression Sample": "price",
    "Classification Sample": "purchased",
    "Clustering Sample": None,
    "Final Challenge": "injury_risk",
}

SKLEARN_LOADERS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
    "Diabetes": load_diabetes,
}

SKLEARN_TARGETS = {
    "Iris": "target",
    "Wine": "target",
    "Breast Cancer": "target",
    "Diabetes": "target",
}


# 프로그램에 포함된 CSV Dataset을 불러옴
def load_builtin_dataset(name: str) -> pd.DataFrame:
    return pd.read_csv(BUILTIN_DATASETS[name])


# Built-in Dataset에 미리 정의된 Target을 반환
def get_builtin_target(name: str) -> str | None:
    return BUILTIN_TARGETS[name]


# scikit-learn Sample Dataset을 DataFrame으로 변환
def load_sklearn_dataset(name: str) -> pd.DataFrame:
    data = SKLEARN_LOADERS[name](as_frame=True)
    return data.frame


# scikit-learn Dataset에 미리 정의된 Target을 반환
def get_sklearn_target(name: str) -> str:
    return SKLEARN_TARGETS[name]


# scikit-learn 분류 Dataset의 Target 숫자와 실제 Class 이름을 연결
def get_sklearn_target_names(name: str) -> list[str] | None:
    data = SKLEARN_LOADERS[name]()

    if not hasattr(data, "target_names"):
        return None

    return [str(target_name) for target_name in data.target_names]


# 사용자가 선택한 Local CSV를 DataFrame으로 불러옴
def load_local_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)