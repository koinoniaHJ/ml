# 프로젝트에서 사용하는 파일과 디렉터리 경로를 관리
from pathlib import Path

# __file__: 현재 실행 중인 Python 파일의 위치
# 현재 파일 기준 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
DATASETS_DIR = PROJECT_ROOT / "datasets"

FONT_PATHS = (
    FONTS_DIR / "SUIT-Regular.ttf",
    FONTS_DIR / "SUIT-SemiBold.ttf",
    FONTS_DIR / "SUIT-Bold.ttf",
)
