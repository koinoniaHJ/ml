from common.theme import (
    BORDER_RADIUS,
    COLOR_BACKGROUND,
    COLOR_SELECTED,
    COLOR_SELECTED_TEXT,
    COLOR_TEXT,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_HEADER,
    FONT_SIZE_SMALL,
    FONT_SIZE_TABLE,
    SCROLLBAR_RADIUS,
    SCROLLBAR_SIZE,
    SPACE_8,
    SPACE_12,
)


# QSS: PySide6/Qt 위젯의 색상, 글꼴, 여백, 테두리 같은 UI 스타일을 지정하는 문법
# f""": 여러 줄 문자열, 문자열 안에 {변수} 값을 넣을 수 있게 한다.
APP_STYLE = f"""
QWidget {{
    background-color: {COLOR_BACKGROUND};
    color: {COLOR_TEXT};
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_BASE}px;
}}


/* Header */

QLabel#headerLabel {{
    font-size: {FONT_SIZE_HEADER}px;
}}


/* Main Container */

QFrame#navigationFrame,
QFrame#pageFrame {{
    border: 1px solid {COLOR_SELECTED};
    border-radius: {BORDER_RADIUS}px;
}}


/* Navigation */

QPushButton#navigationButton {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: none;
    padding: {SPACE_12}px;
    text-align: left;
}}

QPushButton#navigationButton:checked {{
    background-color: {COLOR_SELECTED};
    color: {COLOR_SELECTED_TEXT};
}}


/* Home Dataset Button */

QPushButton#datasetButton {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: none;
    padding: {SPACE_12}px;
}}

QPushButton#datasetButton:hover {{
    background-color: {COLOR_SELECTED};
    color: {COLOR_SELECTED_TEXT};
}}


/* Data Lab Text */

QLabel#smallText {{
    font-size: {FONT_SIZE_SMALL}px;
}}


/* Data Lab ComboBox */

QComboBox#dataControl {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_SELECTED};
    border-radius: {BORDER_RADIUS}px;
    padding: {SPACE_8}px;
    font-size: {FONT_SIZE_SMALL}px;
}}

QComboBox#dataControl QAbstractItemView {{
    background-color: {COLOR_BACKGROUND};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_SELECTED};
    selection-background-color: {COLOR_SELECTED};
    selection-color: {COLOR_SELECTED_TEXT};
}}


/* Data Lab Button */

QPushButton#dataButton,
QPushButton#conceptButton {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_SELECTED};
    border-radius: {BORDER_RADIUS}px;
    padding: {SPACE_8}px {SPACE_12}px;
    font-size: {FONT_SIZE_SMALL}px;
}}

QPushButton#dataButton:hover,
QPushButton#conceptButton:hover {{
    background-color: {COLOR_SELECTED};
    color: {COLOR_SELECTED_TEXT};
}}


/* Data Preview */

QTableWidget#dataPreviewTable {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_SELECTED};
    gridline-color: {COLOR_SELECTED};
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_TABLE}px;
}}

QTableWidget#dataPreviewTable QHeaderView::section {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: none;
    border-bottom: 1px solid {COLOR_SELECTED};
    padding: {SPACE_8}px;
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_TABLE}px;
}}

QTableWidget#dataPreviewTable::item {{
    padding: {SPACE_8}px;
}}


/* Python Code */

QPlainTextEdit#codeView {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_SELECTED};
    border-radius: {BORDER_RADIUS}px;
    font-size: {FONT_SIZE_SMALL}px;
}}


/* Vertical Scrollbar */

QScrollBar:vertical {{
    background: transparent;
    width: {SCROLLBAR_SIZE}px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_SELECTED};
    border-radius: {SCROLLBAR_RADIUS}px;
    min-height: 32px;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}


/* Horizontal Scrollbar */

QScrollBar:horizontal {{
    background: transparent;
    height: {SCROLLBAR_SIZE}px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background: {COLOR_SELECTED};
    border-radius: {SCROLLBAR_RADIUS}px;
    min-width: 32px;
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
}}
"""