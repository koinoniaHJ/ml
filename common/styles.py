from common.theme import (
    BORDER_RADIUS,
    COLOR_BACKGROUND,
    COLOR_BORDER,
    COLOR_SELECTED_BACKGROUND,
    COLOR_SELECTED_TEXT,
    COLOR_TEXT,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_HEADER,
    SPACE_12,
)

# f""": 여러 줄 문자열, 문자열 안에 {변수} 값을 넣을 수 있게 한다.
APP_STYLE = f"""
QWidget {{
    background-color: {COLOR_BACKGROUND};
    color: {COLOR_TEXT};
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_BASE}px;
}}

QLabel#headerLabel {{
    font-size: {FONT_SIZE_HEADER}px;
}}

QFrame#navigationFrame,
QFrame#pageFrame {{
    border: 1px solid {COLOR_BORDER};
    border-radius: {BORDER_RADIUS}px;
}}

QPushButton#navigationButton {{
    background-color: transparent;
    color: {COLOR_TEXT};

    border: none;

    padding: {SPACE_12}px;

    text-align: left;
}}

QPushButton#navigationButton:checked {{
    background-color: {COLOR_SELECTED_BACKGROUND};
    color: {COLOR_SELECTED_TEXT};
}}

QPushButton#datasetButton {{
    background-color: transparent;
    color: {COLOR_TEXT};

    border: none;

    padding: {SPACE_12}px;
}}

QPushButton#datasetButton:hover {{
    background-color: {COLOR_SELECTED_BACKGROUND};
    color: {COLOR_SELECTED_TEXT};
}}
"""