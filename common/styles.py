from common.theme import (
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY, FONT_FAMILY,
    FONT_SIZE_BASE, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_SECTION, FONT_SIZE_SMALL, FONT_SIZE_TABLE,
    RADIUS_MD, RADIUS_SM, SCROLLBAR_RADIUS, SCROLLBAR_SIZE, SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
)


# QSS: PySide6/Qt 위젯의 색상, 글꼴, 여백, 테두리 같은 UI 스타일을 지정하는 문법
# f""": 여러 줄 문자열, 문자열 안에 {변수} 값을 넣을 수 있게 한다.
APP_STYLE = f"""
QWidget {{
    color: {COLOR_OFF_BLACK};
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_BASE}pt;
}}

QMainWindow#mainWindow,
QWidget#centralWidget {{
    background-color: {COLOR_PRIMARY};
}}


/* Header */

QLabel#headerLabel {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    border: none;
    margin: 0px;
    padding: 0px;
    font-size: {FONT_SIZE_HEADER}pt;
}}


/* Main Container */

QFrame#navigationFrame,
QFrame#pageFrame {{
    background-color: {COLOR_OFF_WHITE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QStackedWidget#pageStack {{
    background-color: transparent;
}}

QLabel {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
}}


/* Page Title */

QLabel#pageTitle {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    font-size: {FONT_SIZE_LARGE}pt;
}}


/* Section Title */

QLabel#sectionTitle {{
    color: {COLOR_OFF_BLACK};
    font-size: {FONT_SIZE_SECTION}pt;
}}

QLabel#smallText {{
    font-size: {FONT_SIZE_SMALL}pt;
}}


/* Navigation */

QPushButton#navigationButton {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px;
    text-align: left;
    font-size: {FONT_SIZE_SMALL}pt;
}}

QPushButton#navigationButton:checked {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
}}


/* Home */

QWidget#homePage,
QWidget#placeholderPage {{
    background-color: transparent;
}}

QLabel#homeGuideLabel,
QLabel#placeholderTitle {{
    font-size: {FONT_SIZE_LARGE}pt;
}}

QPushButton#datasetButton {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_MD}px;
    padding: {SPACE_MD}px {SPACE_SM}px;
    font-size: {FONT_SIZE_SMALL}pt;
}}

QPushButton#datasetButton:hover,
QPushButton#datasetButton:pressed {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
}}


/* Data Lab */

QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}


/* ComboBox */

QComboBox#dataControl {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_LG}px {SPACE_XS}px {SPACE_XS}px;
    font-size: {FONT_SIZE_SMALL}pt;
}}

QComboBox#dataControl:disabled {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
}}

QComboBox#dataControl::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: {SPACE_LG}px;
    background-color: transparent;
    border: none;
}}

QComboBox#dataControl::down-arrow {{
    image: none;
    width: 0px;
    height: 0px;
}}

QComboBox#dataControl QAbstractItemView {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    padding: 4px;
    outline: none;
    selection-background-color: {COLOR_SECONDARY};
    selection-color: {COLOR_OFF_BLACK};
}}

QComboBox#dataControl QAbstractItemView::item {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: 4px;
    padding: {SPACE_XS}px;
    min-height: 24px;
}}

QComboBox#dataControl QAbstractItemView::item:selected {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
}}


/* Data Lab Button */

QPushButton#dataButton,
QPushButton#conceptButton {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    font-size: {FONT_SIZE_SMALL}pt;
}}

QPushButton#dataButton:hover,
QPushButton#conceptButton:hover {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
}}

QPushButton#conceptButton:checked {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_SECONDARY};
}}


/* Data Preview */

QTableWidget#dataPreviewTable {{
    background-color: {COLOR_OFF_WHITE};
    alternate-background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    gridline-color: {COLOR_PRIMARY};
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_TABLE}pt;
}}

QTableWidget#dataPreviewTable QHeaderView {{
    background-color: {COLOR_OFF_WHITE};
}}

QTableWidget#dataPreviewTable QHeaderView::section {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-bottom: 1px solid {COLOR_PRIMARY};
    padding: {SPACE_XS}px;
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_TABLE}pt;
}}

QTableWidget#dataPreviewTable::item {{
    background-color: {COLOR_OFF_WHITE};
    padding: {SPACE_XS}px;
}}

QTableWidget#dataPreviewTable::item:selected {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
}}


/* Python Code */

QPlainTextEdit#codeView {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    font-size: {FONT_SIZE_SMALL}pt;
}}


/* Scrollbar */

QScrollBar:vertical {{
    background: transparent;
    width: {SCROLLBAR_SIZE}px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_SECONDARY};
    border-radius: {SCROLLBAR_RADIUS}px;
    min-height: {SPACE_LG}px;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: {SCROLLBAR_SIZE}px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background: {COLOR_SECONDARY};
    border-radius: {SCROLLBAR_RADIUS}px;
    min-width: {SPACE_LG}px;
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