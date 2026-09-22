import sys

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from common.paths import FONT_PATH
from common.styles import APP_STYLE
from ui.main_window import MainWindow


def load_font(app: QApplication):
    font_id = QFontDatabase.addApplicationFont(
        str(FONT_PATH)
    )

    if font_id == -1:
        return

    font_families = QFontDatabase.applicationFontFamilies(
        font_id
    )

    if font_families:
        app.setFont(
            QFont(font_families[0])
        )


def main():
    app = QApplication(sys.argv)

    load_font(app)

    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()