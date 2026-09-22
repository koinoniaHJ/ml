from pathlib import Path

# __file__: 현재 실행 중인 Python 파일의 위치
# 현재 파일 기준 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

FONT_PATH = FONTS_DIR / "Mona12.ttf"