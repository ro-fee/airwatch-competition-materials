"""Application-wide visual theme and plot styling."""

from PyQt5.QtGui import QFont


APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0b1220;
    color: #dbe7f5;
    font-family: "Microsoft YaHei UI";
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #24344d;
    border-radius: 8px;
    background: #0f1929;
    top: -1px;
}
QTabBar::tab {
    background: #111d30;
    color: #91a4bd;
    padding: 10px 24px;
    margin-right: 4px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}
QTabBar::tab:selected { background: #176b87; color: white; }
QTabBar::tab:hover:!selected { background: #1a2940; color: #e5eef9; }
QGroupBox {
    background: #111d30;
    border: 1px solid #2a3b55;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #65d4e8;
}
QGroupBox#bearingDiagnosisPanel {
    background: #101d31;
    border: 1px solid #2f5874;
    margin-top: 12px;
}
QLabel#bearingSubtitle {
    color: #8fa6c0;
    font-size: 12px;
}
QLabel#bearingSectionLabel {
    color: #dbe7f5;
    font-weight: 700;
    min-width: 58px;
}
QFrame#bearingSummaryCard {
    background: #0b1727;
    border: 1px solid #2b4965;
    border-radius: 7px;
    min-height: 68px;
}
QLabel#bearingCardCaption {
    color: #8fa6c0;
    font-size: 11px;
}
QLineEdit#bearingFilePath {
    background: #091321;
}
QPushButton#bearingStartButton {
    background: #16816f;
    border-color: #32b89f;
}
QPushButton#bearingStartButton:hover {
    background: #20a189;
}
QFrame#waterfallPanel {
    background: #0b1727;
    border: 1px solid #28445f;
    border-radius: 7px;
}
QFrame#waterfallPanel QLabel#waterfallProgress {
    color: #65d4e8;
    font-family: "Cascadia Mono", "Microsoft YaHei UI";
    min-width: 80px;
}
QFrame#waterfallPanel QSpinBox {
    min-height: 30px;
    padding: 0 6px;
    color: #e9f2fb;
    background: #091321;
    border: 1px solid #344963;
    border-radius: 5px;
}
QPushButton#waterfallPlayButton { background: #16816f; border-color: #32b89f; }
QPushButton#waterfallPlayButton:hover { background: #20a189; }
QPushButton#waterfallStopButton { background: #73394b; border-color: #a65369; }
QPushButton#waterfallStopButton:hover { background: #91465c; }
QPushButton {
    min-height: 32px;
    padding: 0 16px;
    border: 1px solid #2389a8;
    border-radius: 6px;
    background: #176b87;
    color: white;
    font-weight: 600;
}
QPushButton:hover { background: #2085a4; border-color: #65d4e8; }
QPushButton:pressed { background: #11566e; }
QPushButton:disabled { background: #202c3d; border-color: #334155; color: #718096; }
QPushButton#bearingCancelButton { background: #73394b; border-color: #a65369; }
QPushButton#bearingCancelButton:hover { background: #91465c; }
QLabel#bearingQualityStatus {
    padding: 4px 10px;
    border-radius: 5px;
    font-weight: 700;
}
QLabel#bearingQualityStatus[qualityState="idle"] {
    color: #a8b7c9;
    background: #1b293b;
}
QLabel#bearingQualityStatus[qualityState="accepted"] {
    color: #b8f4d7;
    background: #14513f;
}
QLabel#bearingQualityStatus[qualityState="caution"] {
    color: #ffe5a6;
    background: #634a1c;
}
QLabel#bearingQualityStatus[qualityState="rejected"] {
    color: #ffd0d8;
    background: #642d3c;
}
QLabel#bearingTaskState { color: #65d4e8; font-weight: 600; }
QLabel#bearingDiagnosisResult, QLabel#bearingConfidence {
    color: #e9f2fb;
    font-weight: 700;
}
QLabel#bearingQualityMessage, QLabel#bearingDetails { color: #a8b7c9; }
QProgressBar#bearingProgress {
    min-height: 18px;
    border: 1px solid #344963;
    border-radius: 5px;
    background: #091321;
    text-align: center;
    color: #e9f2fb;
}
QProgressBar#bearingProgress::chunk {
    border-radius: 4px;
    background: #176b87;
}
QLineEdit, QComboBox, QTextEdit {
    background: #091321;
    border: 1px solid #344963;
    border-radius: 5px;
    color: #e9f2fb;
    selection-background-color: #176b87;
}
QLineEdit, QComboBox { min-height: 30px; padding: 0 8px; }
QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border-color: #42bfd7; }
QComboBox::drop-down { border: 0; width: 24px; }
QComboBox QAbstractItemView { background: #111d30; color: #e9f2fb; selection-background-color: #176b87; }
QTextEdit { padding: 7px; font-family: "Cascadia Mono", "Microsoft YaHei UI"; font-size: 12px; }
QRadioButton { spacing: 7px; }
QRadioButton::indicator { width: 16px; height: 16px; }
QRadioButton::indicator:checked { background: #42bfd7; border: 4px solid #102033; border-radius: 8px; }
QRadioButton::indicator:unchecked { background: #0b1422; border: 1px solid #60738c; border-radius: 8px; }
QLabel { background: transparent; }
QStatusBar { background: #09111e; color: #8fa6c0; border-top: 1px solid #24344d; }
QToolTip { background: #1a2940; color: white; border: 1px solid #42bfd7; padding: 5px; }
QScrollBar:vertical { background: #0d1726; width: 11px; }
QScrollBar::handle:vertical { background: #344963; min-height: 28px; border-radius: 5px; }
"""


def apply_window_theme(window):
    """Apply consistent typography and pyqtgraph colors to a window."""
    window.setStyleSheet(APP_STYLESHEET)
    window.setFont(QFont("Microsoft YaHei UI", 10))
    for name in ("label_signalshow_1", "label_signal", "label_featuremap", "label_genesig"):
        plot = getattr(window, name, None)
        if plot is None:
            continue
        plot.setBackground("#08111e")
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.getPlotItem().getAxis("left").setPen("#8094ad")
        plot.getPlotItem().getAxis("bottom").setPen("#8094ad")
