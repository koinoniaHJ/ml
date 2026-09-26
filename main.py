# 애플리케이션의 글꼴·색상·스타일을 설정하고 메인 창을 실행
import sys

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from common.paths import FONT_PATHS
from common.styles import APP_STYLE
from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_SECONDARY,
    FONT_FAMILY, FONT_SIZE_SM,
)
from ui.main_window import MainWindow


# 프로젝트에서 사용할 글꼴을 등록
def load_font(app: QApplication) -> None:
    font_loaded = False

    for font_path in FONT_PATHS:
        font_id = QFontDatabase.addApplicationFont(str(font_path))

        if font_id == -1:
            print(f"[font] Failed to load: {font_path}", file=sys.stderr)
            continue

        font_loaded = True

    if font_loaded:
        font = QFont(FONT_FAMILY)
        font.setPixelSize(FONT_SIZE_SM)
        app.setFont(font)


# Windows 테마 영향을 받지 않도록 기본 팔레트를 설정
def set_app_palette(app: QApplication) -> None:
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLOR_OFF_WHITE))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLOR_OFF_BLACK))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_OFF_WHITE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_OFF_WHITE))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_OFF_BLACK))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLOR_OFF_WHITE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLOR_OFF_BLACK))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_SECONDARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(COLOR_OFF_BLACK))
    app.setPalette(palette)


# PySide6 애플리케이션을 실행
def main() -> None:
    app = QApplication(sys.argv)

    load_font(app)
    set_app_palette(app)
    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
