"""Canvas-first home layout. Presentation only; existing analysis callers stay intact."""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QStackedWidget,
    QScrollArea, QPushButton, QRadioButton, QButtonGroup, QComboBox, QSizePolicy,
)
from .plots.plot_registry import get_signal_view_spec


class ResultImage(QLabel):
    """Keep the full-resolution image; never stretch scientific axes independently."""
    changed = pyqtSignal()

    def __init__(self):
        super().__init__('尚无本次分析结果')
        self.source = QPixmap()
        self.fit = True
        self.scroll = None
        self.setAlignment(Qt.AlignCenter)

    def setScaledContents(self, enabled):
        # Legacy callers request stretching; this canvas instead preserves aspect ratio.
        super().setScaledContents(False)

    def setPixmap(self, pixmap):
        self.source = QPixmap(pixmap)
        self.update_size()
        self.changed.emit()

    def setText(self, text):
        self.source = QPixmap()
        super().setText(text)
        self.adjustSize()
        self.changed.emit()

    def clear(self):
        self.source = QPixmap()
        super().clear()

    def update_size(self):
        if self.source.isNull():
            return
        image = self.source
        if self.fit and self.scroll is not None:
            size = self.scroll.viewport().size()
            image = image.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(image)
        self.resize(image.size())


class ImageScroll(QScrollArea):
    def __init__(self, image):
        super().__init__()
        self.image = image
        image.scroll = self
        self.setWidget(image)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.NoFrame)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.image.update_size()


class SignalWorkbench(QWidget):
    """Rehouse home widgets, without copying input, inference or plotting logic."""
    expert_requested = pyqtSignal(str)

    def __init__(self, window):
        super().__init__(window.tab1)
        self.waterfall = window.waterfall_plotter
        self.setObjectName('signalWorkbench')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('''
            QWidget#signalWorkbench { background: #101214; }
            QFrame#commandStrip { background: #121719; border: 1px solid #725064;
                border-radius: 10px; }
            QLabel#canvasHeading { color: #f2f4ee; font-size: 22px; }
            QLabel#canvasHint { color: #9aadb7; }
            QRadioButton { padding: 7px; }
            QRadioButton:checked { color: #70e0d6; }
            QPushButton#expertToggle { background: #171b1e; border-color: #725064; color: #ffb4de; }
        ''')
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        heading = QLabel('信号观察工作台')
        heading.setObjectName('canvasHeading')
        root.addWidget(heading)
        intro = QLabel('先看信号，再展开细节  /  离线分析 · 不自动生成识别结论')
        intro.setObjectName('canvasHint')
        root.addWidget(intro)
        body = QHBoxLayout()
        root.addLayout(body, 1)
        left = QVBoxLayout()
        body.addLayout(left, 1)
        self.stack = QStackedWidget()
        self.stack.setMinimumSize(500, 300)
        self.stack.addWidget(window.label_signalshow_1)
        window.label_signalshow_1.setBackground('#0c1012')
        left.addWidget(self.stack, 1)

        self.images = []
        for name in ('label_featureshow_1', 'label_featureshow_2'):
            old = getattr(window, name)
            old.hide()
            old.deleteLater()
            image = ResultImage()
            image.setObjectName(name)
            setattr(window, name, image)
            page = ImageScroll(image)
            self.stack.addWidget(page)
            image.changed.connect(lambda page=page: self.show_result(page))
            self.images.append(image)

        navigation = QHBoxLayout()
        left.addLayout(navigation)
        # Keep the original radio objects and their signal wiring, explicitly grouped
        # because reparenting would otherwise break Qt's sibling auto-exclusivity.
        self.radio_group = QButtonGroup(self)
        window.radioButton_spectrum_1 = QRadioButton('频谱')
        radios = (window.radioButton_time_1, window.radioButton_spectrum_1,
                  window.radioButton_TF_1, window.radioButton_waterfall)
        for radio, title, key in zip(radios, ('波形', '频谱', '时频', '动态瀑布图'),
                                     ('waveform', 'spectrum', 'spectrogram', 'waterfall')):
            radio.setText(title)
            radio.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            radio.setToolTip(get_signal_view_spec(key).plain_explanation)
            self.radio_group.addButton(radio)
            navigation.addWidget(radio)
            radio.toggled.connect(lambda checked, key=key: checked and self.show_signal(key))
        navigation.addStretch()
        self.return_button = QPushButton('返回信号图')
        self.return_button.clicked.connect(
            lambda _checked=False: window._cancel_home_feature()
        )
        self.return_button.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        navigation.addWidget(self.return_button)
        self.zoom_button = QPushButton('原尺寸 / 适应')
        self.zoom_button.setToolTip('分析图片按原尺寸滚动查看，或等比例适应画布；信号图使用鼠标缩放。')
        self.zoom_button.clicked.connect(self.toggle_zoom)
        navigation.addWidget(self.zoom_button)

        self.expert_toggle = QPushButton('展开进阶分析 +')
        self.expert_toggle.setObjectName('expertToggle')
        self.expert_toggle.setCheckable(True)
        left.addWidget(self.expert_toggle)
        self.expert_panel = QWidget()
        expert = QHBoxLayout(self.expert_panel)
        expert.setContentsMargins(0, 0, 0, 0)
        self.expert_combo = QComboBox()
        for key, action in (('wavelet', '小波特征'), ('bispectrum', '双谱特征'),
                            ('jr', 'J、R特征'), ('hht', 'HHT特征')):
            spec = get_signal_view_spec(key)
            self.expert_combo.addItem(spec.title, action)
            self.expert_combo.setItemData(self.expert_combo.count()-1,
                                          spec.plain_explanation, Qt.ToolTipRole)
        expert.addWidget(self.expert_combo)
        self.analyse_button = QPushButton('分析当前选区')
        self.analyse_button.clicked.connect(lambda: self.expert_requested.emit(self.expert_combo.currentData()))
        expert.addWidget(self.analyse_button)
        for i in range(2):
            button = QPushButton(f'结果 {i+1}')
            button.setToolTip('两个交替更新的分析结果槽；不是类别或准确率。')
            button.clicked.connect(lambda checked=False, i=i: self.show_result(self.stack.widget(i+1)))
            expert.addWidget(button)
        left.addWidget(self.expert_panel)
        self.expert_panel.hide()
        self.expert_toggle.toggled.connect(self.expert_panel.setVisible)
        self.expert_toggle.toggled.connect(lambda on: self.expert_toggle.setText('收起进阶分析 −' if on else '展开进阶分析 +'))

        # Compact command strip below the primary canvas.
        command = QFrame()
        command.setObjectName('commandStrip')
        commands = QVBoxLayout(command)
        row = QHBoxLayout()
        commands.addLayout(row)
        for widget in (window.pushButton_openfile, window.label_filename_1,
                       window.label_Fs_1, window.lineEdit_Fs_1,
                       window.label_windowlength_1, window.lineEdit_windowlength_1,
                       window.pushButton_reload_TF):
            row.addWidget(widget)
        window.label_filename_1.setText('未选择文件')
        window.label_filename_1.setMaximumWidth(200)
        window.label_filename_1.setMinimumWidth(50)
        window.label_filename_1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.setStretch(1, 1)
        window.label_Fs_1.setText('采样率 Hz')
        window.label_windowlength_1.setText('窗长')
        for edit in (window.lineEdit_Fs_1, window.lineEdit_windowlength_1):
            edit.setMaximumWidth(100)
        commands.addWidget(window.waterfall_panel)
        left.addWidget(command)

        side = QWidget(self)
        sidebar = QVBoxLayout(side)
        body.addWidget(side)
        side.setFixedWidth(250)
        sidebar.setContentsMargins(8, 0, 0, 0)
        sidebar.addWidget(QLabel('图形说明'))
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        sidebar.addWidget(self.explanation)
        note = QLabel('进阶图读取主页实信号及采样率。\n波形模式使用当前可见选区，其他模式使用整段。\n\n星座图在生成页；模型特征与 t-SNE 在识别页。尚未接入此画布，避免混用数据。')
        note.setWordWrap(True)
        sidebar.addWidget(note)
        sidebar.addWidget(QLabel('操作记录 / 非识别结果'))
        sidebar.addWidget(window.textEdit_log_1, 1)
        self.stack.currentChanged.connect(self._page_changed)
        self.show_signal('spectrogram')

    def _page_changed(self, index):
        self.zoom_button.setEnabled(index != 0)
        self.return_button.setEnabled(index != 0)

    def show_signal(self, key):
        self.explanation.setText(get_signal_view_spec(key).plain_explanation)
        self.stack.setCurrentIndex(0)
        self._page_changed(0)

    def show_result(self, page):
        if self.waterfall.is_running:
            self.waterfall.pause()
        self.stack.setCurrentWidget(page)
        self.explanation.setText('分析结果为辅助观察，不是故障分类结论。可用“原尺寸 / 适应”查看细节；结果 1、2 交替更新。')

    def toggle_zoom(self):
        page = self.stack.currentWidget()
        if isinstance(page, ImageScroll):
            page.image.fit = not page.image.fit
            page.image.update_size()

    def reset_results(self):
        for image in self.images:
            image.setText('数据已更换，请重新分析。')
        self.stack.setCurrentIndex(0)


def install_signal_workbench(window):
    """Keep generated layouts for traceability, but move live widgets into one canvas."""
    # Remove layout items, not widgets. The old parameter containers remain owned by
    # tab1 (hidden) so historical attributes stay valid during incremental migration.
    while window.gridLayout_4.count():
        window.gridLayout_4.takeAt(0)
    workbench = SignalWorkbench(window)
    window.groupBox_fileshow.hide()
    window.groupBox_paraset.hide()
    window.gridLayout_4.addWidget(workbench, 0, 0)
    return workbench
