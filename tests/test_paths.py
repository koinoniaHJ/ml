from common.paths import DATASETS_DIR, MODELS_DIR

# assert: 조건이 True인지 검사, False면 Test를 실패시킨다.
def test_datasets_directory_exists():
    # 경로가 실제로 존재하는지 확인
    assert DATASETS_DIR.exists()
    # 경로가 디렉터리인지 확인
    assert DATASETS_DIR.is_dir()


def test_models_directory_exists():
    assert MODELS_DIR.exists()
    assert MODELS_DIR.is_dir()