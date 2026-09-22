from pathlib import Path

# __file__: 현재 실행 중인 Python 파일의 위치
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
FONT_PATH = FONTS_DIR / "Mona12.ttf"