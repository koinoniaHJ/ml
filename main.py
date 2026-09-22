import sys

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from common.paths import FONT_PATH
from common.styles import APP_STYLE
from common.theme import COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_SECONDARY
from ui.main_window import MainWindow


# 프로젝트에서 사용할 Font를 등록
def load_font(app: QApplication):
    font_id = QFontDatabase.addApplicationFont(str(FONT_PATH))

    if font_id == -1:
        return

    font_families = QFontDatabase.applicationFontFamilies(font_id)

    if font_families:
        app.setFont(QFont(font_families[0], 10))


# Windows Theme 영향을 받지 않도록 기본 Palette를 설정
def set_app_palette(app: QApplication):
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


# PySide6 Application을 실행
def main():
    app = QApplication(sys.argv)

    load_font(app)
    set_app_palette(app)
    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()