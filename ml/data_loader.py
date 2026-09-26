# 학습에 사용할 scikit-learn 기본 데이터와 CSV 데이터를 불러옴
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.utils import Bunch

from common.paths import DATASETS_DIR


BUILTIN_DATASETS = {
    "Student Basic": {
        "path": DATASETS_DIR / "student_basic.csv",
        "target": "passed",
        "task_type": "classification",
        "class_names": ["미통과", "통과"],
    },
    "Student Dirty": {
        "path": DATASETS_DIR / "student_dirty.csv",
        "target": "passed",
        "task_type": "classification",
        "class_names": ["미통과", "통과"],
    },
    "Preprocessing Sample": {
        "path": DATASETS_DIR / "preprocessing_sample.csv",
        "target": "result",
        "task_type": "classification",
    },
    "Regression Sample": {
        "path": DATASETS_DIR / "regression_sample.csv",
        "target": "price",
        "task_type": "regression",
    },
    "Classification Sample": {
        "path": DATASETS_DIR / "classification_sample.csv",
        "target": "purchased",
        "task_type": "classification",
        "class_names": ["미구매", "구매"],
    },
    "Clustering Sample": {
        "path": DATASETS_DIR / "clustering_sample.csv",
        "target": None,
        "task_type": "clustering",
    },
    "Final Challenge": {
        "path": DATASETS_DIR / "final_challenge.csv",
        "target": "injury_risk",
        "task_type": "classification",
        "class_names": ["낮음", "높음"],
    },
}

SKLEARN_DATASETS = {
    "Iris": {"loader": load_iris, "target": "target", "task_type": "classification"},
    "Wine": {"loader": load_wine, "target": "target", "task_type": "classification"},
    "Breast Cancer": {"loader": load_breast_cancer, "target": "target", "task_type": "classification"},
    "Diabetes": {"loader": load_diabetes, "target": "target", "task_type": "regression"},
}


@lru_cache(maxsize=None)
def _load_sklearn_bunch(name: str) -> Bunch:
    return SKLEARN_DATASETS[name]["loader"](as_frame=True)


# 프로그램에 포함된 CSV Dataset을 불러옴
def load_builtin_dataset(name: str) -> pd.DataFrame:
    return pd.read_csv(BUILTIN_DATASETS[name]["path"])


# Built-in Dataset에 미리 정의된 Target을 반환
def get_builtin_target(name: str) -> str | None:
    return BUILTIN_DATASETS[name]["target"]


# Built-in Dataset의 학습 유형을 반환
def get_builtin_task_type(name: str) -> str:
    return BUILTIN_DATASETS[name]["task_type"]


# Built-in 분류 Dataset의 Class 이름을 반환
def get_builtin_class_names(name: str) -> list[str] | None:
    return BUILTIN_DATASETS[name].get("class_names")


# scikit-learn Sample Dataset을 DataFrame으로 변환
def load_sklearn_dataset(name: str) -> pd.DataFrame:
    return _load_sklearn_bunch(name).frame.copy()


# scikit-learn Dataset에 미리 정의된 Target을 반환
def get_sklearn_target(name: str) -> str:
    return SKLEARN_DATASETS[name]["target"]


# scikit-learn Dataset의 학습 유형을 반환
def get_sklearn_task_type(name: str) -> str:
    return SKLEARN_DATASETS[name]["task_type"]


# scikit-learn 분류 Dataset의 Target 숫자와 실제 Class 이름을 연결
def get_sklearn_target_names(name: str) -> list[str] | None:
    data = _load_sklearn_bunch(name)

    if not hasattr(data, "target_names"):
        return None

    return [str(target_name) for target_name in data.target_names]


# 사용자가 선택한 Local CSV를 DataFrame으로 불러옴
def load_local_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
