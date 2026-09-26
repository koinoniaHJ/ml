# 애플리케이션 전체에 적용할 Qt 스타일시트
from common.theme import (
    COLOR_BLUE, COLOR_LIGHT_BLUE,
    COLOR_OFF_BLACK, COLOR_OFF_WHITE, COLOR_PRIMARY, COLOR_SECONDARY, FONT_FAMILY,
    FONT_SIZE_2XL, FONT_SIZE_LG, FONT_SIZE_MD, FONT_SIZE_SM, FONT_SIZE_XL,
    FONT_WEIGHT_BOLD, FONT_WEIGHT_REGULAR,
    RADIUS_MD, RADIUS_SM, SCROLLBAR_RADIUS, SCROLLBAR_SIZE, SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
)


# QSS는 PySide6 위젯의 색상, 글꼴, 여백, 테두리 같은 화면 스타일을 지정한다.
# `#이름`은 위젯의 setObjectName("이름")과 일치해야 한다.
# f""" 문자열을 사용해 디자인 변수의 값을 QSS에 넣는다.
APP_STYLE = f"""
QWidget {{
    color: {COLOR_OFF_BLACK};
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_MD}px;
    font-weight: {FONT_WEIGHT_REGULAR};
}}

QMainWindow#mainWindow,
QWidget#centralWidget {{
    background-color: {COLOR_PRIMARY};
}}


/* 상단 프로그램 제목 */

QLabel#headerLabel {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    border: none;
    margin: 0px;
    padding: 0px;
    font-size: {FONT_SIZE_2XL}px;
}}


/* 사이드 메뉴와 메인 페이지 영역 */

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


/* 각 페이지의 제목 */

QLabel#pageTitle {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    font-size: {FONT_SIZE_XL}px;
}}


/* 섹션 제목, 본문 설명, 작은 안내 문구 */

QLabel#sectionTitle {{
    color: {COLOR_OFF_BLACK};
    font-size: {FONT_SIZE_LG}px;
}}

QLabel#stepTitle,
QLabel#encodingTitle {{
    color: {COLOR_OFF_BLACK};
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
}}

QLabel#sectionBadge {{
    background-color: {COLOR_PRIMARY};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    font-size: {FONT_SIZE_LG}px;
}}

QLabel#bodyText {{
    font-size: {FONT_SIZE_MD}px;
}}

QLabel#smallText {{
    font-size: {FONT_SIZE_SM}px;
}}


/* 왼쪽 사이드 메뉴 */

QPushButton#navigationButton {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: 0px;
    font-size: {FONT_SIZE_SM}px;
}}

QLabel#navigationButtonText {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    font-size: {FONT_SIZE_SM}px;
}}

QPushButton#navigationButton:checked {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
}}


/* Home 페이지 기본 영역 */

QWidget#homePage,
QWidget#placeholderPage {{
    background-color: transparent;
}}

QLabel#homeGuideLabel {{
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
}}

QLabel#placeholderTitle {{
    font-size: {FONT_SIZE_XL}px;
}}

/* AI·ML·DL 관계 다이어그램 */

QFrame#machineLearningDiagram,
QFrame#mlLayer {{
    background-color: {COLOR_LIGHT_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QFrame#aiLayer,
QFrame#dlLayer {{
    background-color: {COLOR_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QLabel#diagramTitle {{
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
}}

QLabel#diagramText {{
    font-size: {FONT_SIZE_MD}px;
}}

QLabel#diagramSummary {{
    font-size: {FONT_SIZE_MD}px;
}}

/* 일반 프로그램과 머신러닝 비교 */

QFrame#programComparisonSection {{
    background-color: {COLOR_LIGHT_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QFrame#programComparisonCard {{
    background-color: {COLOR_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QFrame#programComparisonContentCard {{
    background-color: {COLOR_OFF_WHITE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QLabel#programComparisonTitle {{
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
}}

QLabel#programComparisonText,
QLabel#programComparisonDefinitions {{
    font-size: {FONT_SIZE_MD}px;
}}

/* 지도·비지도·강화학습 안내 */

QFrame#learningMethodsSection {{
    background-color: {COLOR_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QFrame#learningIntroCard,
QFrame#learningDetailCard,
QFrame#learningMethodCard {{
    background-color: {COLOR_LIGHT_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QLabel#learningIntroText,
QLabel#learningMethodText,
QLabel#learningDetailText {{
    font-size: {FONT_SIZE_MD}px;
}}

QLabel#learningMethodTitle,
QLabel#learningSectionTitle {{
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
}}

QPushButton#learningDetailsToggle {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    border: none;
    padding: {SPACE_XS}px;
    font-size: {FONT_SIZE_MD}px;
}}

/* Dataset 선택 카드 */

QFrame#datasetSelectionSection {{
    background-color: {COLOR_LIGHT_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QLabel#datasetIntroductionText {{
    font-size: {FONT_SIZE_MD}px;
}}

QFrame#datasetIntroductionCard {{
    background-color: {COLOR_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QFrame#datasetCard {{
    background-color: {COLOR_BLUE};
    border: none;
    border-radius: {RADIUS_MD}px;
}}

QLabel#datasetTitle {{
    color: {COLOR_OFF_BLACK};
    font-size: {FONT_SIZE_LG}px;
    font-weight: {FONT_WEIGHT_BOLD};
}}

QLabel#datasetText {{
    color: {COLOR_OFF_BLACK};
    font-size: {FONT_SIZE_MD}px;
    font-weight: {FONT_WEIGHT_REGULAR};
}}

QPushButton#datasetSelectButton {{
    background-color: {COLOR_LIGHT_BLUE};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px;
    font-size: {FONT_SIZE_MD}px;
    font-weight: {FONT_WEIGHT_REGULAR};
}}

/* Data Lab 페이지 기본 영역 */

QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}


/* Dataset과 열을 선택하는 콤보박스 */

QComboBox#dataControl {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_LG}px {SPACE_XS}px {SPACE_XS}px;
    font-size: {FONT_SIZE_MD}px;
}}

QSpinBox#dataControl,
QDoubleSpinBox#dataControl {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_LG}px {SPACE_XS}px {SPACE_XS}px;
    font-size: {FONT_SIZE_MD}px;
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
    padding: {SPACE_XS}px;
    outline: none;
    selection-background-color: {COLOR_SECONDARY};
    selection-color: {COLOR_OFF_BLACK};
}}

QComboBox#dataControl QAbstractItemView::item {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px;
    min-height: {SPACE_MD}px;
}}

QComboBox#dataControl QAbstractItemView::item:selected {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
}}


/* Data Lab 실행 버튼과 개념 선택 버튼 */

QPushButton#dataButton,
QPushButton#conceptButton {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    font-size: {FONT_SIZE_MD}px;
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

/* Preprocessing 단계와 결과 Card */

QPushButton#stepButton {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
    padding: {SPACE_XS}px {SPACE_SM}px;
    font-size: {FONT_SIZE_MD}px;
}}

QPushButton#stepButton:hover,
QPushButton#stepButton:checked {{
    background-color: {COLOR_SECONDARY};
    border: 1px solid {COLOR_SECONDARY};
}}

QPushButton#stepButton:disabled {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_PRIMARY};
    border: 1px solid {COLOR_PRIMARY};
}}

QCheckBox#dataCheckBox {{
    spacing: {SPACE_XS}px;
    padding: {SPACE_XS}px 0px;
    font-size: {FONT_SIZE_MD}px;
}}

QCheckBox#dataCheckBox::indicator {{
    width: {SPACE_SM}px;
    height: {SPACE_SM}px;
}}

QFrame#preprocessingCard,
QLabel#preprocessingInfoCard {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
}}

QLabel#preprocessingInfoCard,
QLabel#preprocessingResultCard,
QLabel#codeBlock {{
    padding: {SPACE_SM}px;
    font-size: {FONT_SIZE_MD}px;
}}

QLabel#preprocessingResultCard {{
    background-color: {COLOR_BLUE};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
}}

QFrame#codeBlockCard {{
    background-color: {COLOR_BLUE};
    border: none;
    border-radius: {RADIUS_SM}px;
}}

QLabel#codeBlockTitle {{
    color: {COLOR_OFF_BLACK};
    font-size: {FONT_SIZE_MD}px;
    font-weight: 700;
}}

QLabel#codeBlock {{
    background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: none;
    border-radius: {RADIUS_SM}px;
}}

QFrame#conceptDetailCard,
QLabel#graphDescription {{
    background-color: transparent;
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    border-radius: {RADIUS_SM}px;
}}

QLabel#graphDescription {{
    padding: {SPACE_SM}px;
    font-size: {FONT_SIZE_MD}px;
}}


/* 데이터 미리보기 표 */

QTableWidget#dataPreviewTable {{
    background-color: {COLOR_OFF_WHITE};
    alternate-background-color: {COLOR_OFF_WHITE};
    color: {COLOR_OFF_BLACK};
    border: 1px solid {COLOR_PRIMARY};
    gridline-color: {COLOR_PRIMARY};
    outline: none;
    font-family: "{FONT_FAMILY}";
    font-size: {FONT_SIZE_MD}px;
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
    font-size: {FONT_SIZE_MD}px;
}}

QTableWidget#dataPreviewTable::item {{
    background-color: {COLOR_OFF_WHITE};
    padding: {SPACE_XS}px;
}}

QTableWidget#dataPreviewTable::item:selected {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_OFF_BLACK};
    border: none;
}}


/* 세로·가로 스크롤바 */

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
