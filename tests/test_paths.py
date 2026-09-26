from common.paths import DATASETS_DIR


# 데이터셋 경로가 사용할 수 있는 디렉터리인지 확인
def test_datasets_directory_exists() -> None:
    assert DATASETS_DIR.exists()
    assert DATASETS_DIR.is_dir()
