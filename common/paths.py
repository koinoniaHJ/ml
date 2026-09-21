from pathlib import Path

# __file__: 현재 실행 중인 Python 파일의 위치
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"