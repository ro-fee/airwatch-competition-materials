"""Presentation-only recognition workspace, reusing live legacy widgets and workers."""
from PyQt5.QtCore import Qt
from .uav_plot_panel import UAVPlotPanel
from .bearing_plot_panel import BearingPlotPanel
from PyQt5.QtGui import QIcon, QPainter, QPen, QPixmap, QColor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QStackedWidget, QFrame, QScrollArea, QTextEdit, QSizePolicy, QProgressBar)

from airwatch.runtime_resources import model_slot


def module_icon(kind):
    """Small original vector marks; no external fonts, assets or filesystem paths."""
    pixmap = QPixmap(28, 28)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor('#ff83c7'), 1.5))
    if kind == 'uav':
        painter.drawLine(6, 6, 22, 22); painter.drawLine(6, 22, 22, 6)
        for x,y in ((2,2),(18,2),(2,18),(18,18)):
            painter.drawEllipse(x,y,8,8)
    elif kind == 'bearing':
        painter.drawEllipse(3,3,22,22); painter.drawEllipse(10,10,8,8)
        for x,y in ((12,5),(5,12),(19,12),(12,19)):
            painter.drawEllipse(x,y,3,3)
    else:
        for x,y in ((4,10),(9,5),(14,2),(19,7),(24,11)):
            painter.drawLine(x,y,x,28-y)
    painter.end()
    return QIcon(pixmap)


def text_label(text):
    label = QLabel(text)
    label.setWordWrap(True)
    return label


class RecognitionWorkbench(QWidget):
    def __init__(self, window):
        super().__init__(window.tab_3)
        self.window = window
        self.setObjectName('recognitionWorkbench')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('''
            QWidget#recognitionWorkbench { background: #101214; }
            QFrame#recognitionCommand { background: #121719; border: 1px solid #725064; border-radius: 9px; }
            QPushButton#moduleChoice { background: #171c20; color: #9aadb7; padding: 8px 14px; }
            QPushButton#moduleChoice:checked { background: #193537; color: #eff8f2; border: 1px solid #ff83c7; }
            QLabel#recognitionTitle { font-size: 21px; color: #f2f4ee; }
            QFrame#uavCanvas { background: #0c1012; border: 1px solid #35505a; border-radius: 10px; }
            QFrame#uavSidePanel { background: #121719; border: 1px solid #725064; border-radius: 10px; }
            QLabel#uavPlotPlaceholder { color: #8097a0; background: #10181b; border: 1px dashed #d26b9e; border-radius: 8px; }
            QLabel#uavStatus, QLabel#uavResultCard { color: #d7e2df; background: #111c2d; border: 1px solid #28506a; border-radius: 6px; padding: 10px; }
            QLabel#uavStatus[quality="accepted"] { color: #70e0d6; border-color: #2d9c95; background: #102c2c; }
            QLabel#uavStatus[quality="caution"] { color: #ffe08a; border-color: #b88732; background: #302812; }
            QLabel#uavStatus[quality="rejected"] { color: #ff83c7; border-color: #a94d78; background: #321827; }

        ''')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12,8,12,8)
        top = QHBoxLayout()
        layout.addLayout(top)
        self.modules = QButtonGroup(self)
        self.module_buttons = {}
        self.pages = QStackedWidget()
        for index,(key,title) in enumerate((('uav','无人机识别'),('general','通用信号识别'),('bearing','轴承检测'))):
            button = QPushButton(module_icon(key),title)
            button.setObjectName('moduleChoice')
            button.setCheckable(True)
            self.modules.addButton(button,index)
            self.module_buttons[key] = button
            top.addWidget(button)
        top.addStretch()
        self.status = QLabel()
        top.addWidget(self.status)
        layout.addWidget(self.pages,1)
        self.pages.addWidget(self._uav_page())
        self.pages.addWidget(self._general_page())
        self.pages.addWidget(self._bearing_page())
        self.modules.idClicked.connect(self.select_module)
        self.select_module(1)

    def _uav_page(self):
        # Presentation-only contract shell. It deliberately has no model or fake result.
        page = QWidget()
        body = QHBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        canvas = QFrame()
        canvas.setObjectName('uavCanvas')
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(18, 16, 18, 16)
        title = text_label('无人机识别 · 频谱态势工作区')
        title.setObjectName('recognitionTitle')
        canvas_layout.addWidget(title)
        self.uav_plots = UAVPlotPanel(self)
        self.uav_plot_placeholder = self.uav_plots.plot
        self.uav_view_hint = self.uav_plots.caption
        canvas_layout.addWidget(self.uav_plots, 1)
        self.uav_choose = QPushButton('选择 NPY 文件')
        self.uav_choose.clicked.connect(self.window.load_uav_input)
        canvas_layout.addWidget(self.uav_choose, 0, Qt.AlignLeft)
        body.addWidget(canvas, 1)

        side = QFrame()
        side.setObjectName('uavSidePanel')
        side.setFixedWidth(285)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.addWidget(text_label('任务控制'))
        self.uav_input_hint = text_label('输入文件：尚未选择\n预览格式：NPY')
        self.uav_input_hint.setTextFormat(Qt.PlainText)
        self.uav_input_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.uav_plots.input_changed.connect(self.uav_input_hint.setText)
        side_layout.addWidget(self.uav_input_hint)
        self.uav_quality_label = text_label('信号质量：待检测')
        self.uav_quality_label.setObjectName('uavStatus')
        self.uav_plots.quality_changed.connect(self._set_uav_quality)
        side_layout.addWidget(self.uav_quality_label)
        self.uav_result_label = text_label('识别结果：未输出\n类别：—\n已知 / 未知：未评估\n置信度：—')
        self.uav_result_label.setObjectName('uavResultCard')
        side_layout.addWidget(self.uav_result_label)
        action_row = QHBoxLayout()
        self.uav_start = QPushButton('开始后台识别')
        self.uav_start.setEnabled(False)
        self.uav_start.clicked.connect(self.window.start_uav_recognition)
        action_row.addWidget(self.uav_start)
        self.uav_cancel = QPushButton('取消')
        self.uav_cancel.setEnabled(False)
        self.uav_cancel.clicked.connect(self.window.cancel_uav_recognition)
        action_row.addWidget(self.uav_cancel)
        side_layout.addLayout(action_row)
        self.uav_progress = QProgressBar()
        self.uav_progress.setRange(0, 1)
        self.uav_progress.setValue(0)
        self.uav_progress.setFormat('未开始')
        side_layout.addWidget(self.uav_progress)
        side_layout.addStretch(1)
        known_source = model_slot('uav.known_source')
        open_set = model_slot('uav.open_set')
        side_layout.addWidget(text_label(
            f'说明：{known_source.display_name}已接入，发布级别为开发冻结，'
            '不使用历史通用或轴承模型。\n'
            f'{open_set.limitation}当前只显示“未评估”，'
            '不会把低置信强制解释为未知无人机。'))
        body.addWidget(side)
        return page

    def _set_uav_quality(self, status, message):
        labels = {'accepted': '通过', 'caution': '谨慎', 'rejected': '拒绝'}
        self.uav_quality_label.setProperty('quality', status)
        self.uav_quality_label.setText(f'信号质量：{labels.get(status, "待检测")}\n{message}')
        self.uav_quality_label.style().unpolish(self.uav_quality_label)
        self.uav_quality_label.style().polish(self.uav_quality_label)

    def _general_page(self):
        w = self.window
        page = QWidget()
        body = QHBoxLayout(page)
        left = QVBoxLayout()
        body.addLayout(left,1)
        self.plot_caption = QLabel('输入信号 · 请先选择文件')
        left.addWidget(self.plot_caption)
        self.plot_stack = QStackedWidget()
        self.plot_stack.setMinimumSize(480,300)
        self.plot_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        for plot in (w.label_signal,w.label_featuremap):
            plot.setBackground('#0c1012')
            self.plot_stack.addWidget(plot)
        left.addWidget(self.plot_stack,1)
        row = QHBoxLayout()
        left.addLayout(row)
        self.radio_group = QButtonGroup(self)
        for radio,title in ((w.radioButton_time_3,'波形'),(w.radioButton_PP_3,'频谱'),(w.radioButton_TF_3,'时频')):
            radio.setText(title)
            radio.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Fixed)
            self.radio_group.addButton(radio)
            row.addWidget(radio)
            radio.clicked.connect(self.show_signal)
        row.addWidget(QLabel('当前文件'))
        row.addWidget(w.comboBox,1)
        self.signal_button = QPushButton('看输入信号')
        self.signal_button.clicked.connect(self.show_signal)
        row.addWidget(self.signal_button)
        self.feature_button = QPushButton('看模型特征')
        self.feature_button.clicked.connect(self.show_features)
        row.addWidget(self.feature_button)

        self.details_toggle = QPushButton('展开图形参数 / 专家选项 +')
        self.details_toggle.setCheckable(True)
        left.addWidget(self.details_toggle)
        self.details = QWidget()
        detail = QHBoxLayout(self.details)
        detail.setContentsMargins(0,0,0,0)
        for title,edit in (('采样率 Hz',w.lineEdit_Fs_2),('窗长',w.lineEdit_windowlength_3),('运行次数',w.lineEdit_windowlength_13)):
            detail.addWidget(QLabel(title)); edit.setMaximumWidth(90); detail.addWidget(edit)
        detail.addWidget(w.comboBox_choose_feature)
        self.feature_refresh = QPushButton('更新特征图')
        self.feature_refresh.clicked.connect(w.choose_feature_signal)
        self.feature_refresh.clicked.connect(self.show_features)
        detail.addWidget(self.feature_refresh)
        left.addWidget(self.details)
        self.details.hide()
        self.details_toggle.toggled.connect(self.details.setVisible)
        self.details_toggle.toggled.connect(lambda on: self.details_toggle.setText(
            '收起图形参数 / 专家选项 −' if on else '展开图形参数 / 专家选项 +'))

        command = QFrame()
        command.setObjectName('recognitionCommand')
        commands = QVBoxLayout(command)
        row = QHBoxLayout()
        commands.addLayout(row)
        self.general_cancel = QPushButton('取消识别')
        self.general_cancel.setEnabled(False)
        self.general_cancel.clicked.connect(w.cancel_general_recognition)
        for widget in (QLabel('任务'),w.comboBox_task,QLabel('数据模式'),w.comboBox_mod,w.pushButton_path,w.lineEdit_path,w.pushButton_recognition,self.general_cancel):
            row.addWidget(widget)
        w.comboBox_task.setMaximumWidth(145)
        w.comboBox_mod.setMaximumWidth(125)
        w.lineEdit_path.setMinimumWidth(50)
        w.lineEdit_path.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
        row.setStretch(5,1)
        self.model_hint = text_label('模型由所选任务自动匹配，尚未提供独立模型切换；沿用历史权重。')
        commands.addWidget(self.model_hint)
        self.general_progress = QProgressBar()
        self.general_progress.setRange(0, 1)
        self.general_progress.setFormat('未开始')
        self.general_progress.setMaximumHeight(1)
        self.general_progress.setTextVisible(False)
        commands.addWidget(self.general_progress)
        self.general_result_card = text_label('识别结果：—\n状态：未开始\n文件：—\n模型：—\n窗口：—\n耗时：—')
        self.general_result_card.setObjectName('generalResultCard')
        self.general_result_card.setStyleSheet('background:#111c2d; border:1px solid #28506a; border-radius:6px; padding:8px;')
        left.addWidget(command)

        side = QWidget()
        side.setFixedWidth(265)
        body.addWidget(side)
        sidebar = QVBoxLayout(side)
        sidebar.setContentsMargins(8,0,0,0)
        self.general_context = text_label('通用信号识别 · 历史射频任务\n输入波形不等于模型特征；t-SNE 按来源文件着色，不代表准确率。')
        sidebar.addWidget(self.general_context)
        sidebar.addWidget(text_label('结果与操作记录\n单文件和多文件识别已使用后台任务；重复运行和 t-SNE 仍沿用原流程。'))
        sidebar.addWidget(self.general_result_card)
        sidebar.addWidget(w.textEdit_log_3,1)
        w.comboBox_task.currentIndexChanged.connect(self.show_signal)
        w.comboBox_task.currentIndexChanged.connect(lambda: self.model_hint.setText(
            '模型由所选任务自动匹配，待识别时加载；不提供任意权重混用。'))
        w.pushButton_path.clicked.connect(self.show_signal)
        w.comboBox_choose_feature.currentIndexChanged.connect(self.show_features)
        return page

    def _bearing_page(self):
        w = self.window
        page = QWidget()
        body = QHBoxLayout(page)
        left = QVBoxLayout()
        body.addLayout(left,1)
        self.bearing_plots = BearingPlotPanel(self)
        left.addWidget(self.bearing_plots,1)
        command = QFrame()
        command.setObjectName('recognitionCommand')
        controls = QVBoxLayout(command)
        file_row = QHBoxLayout()
        controls.addLayout(file_row)
        for widget in (w.bearing_file_edit,w.bearing_select_button,w.bearing_start_button,w.bearing_cancel_button):
            file_row.addWidget(widget)
        controls.addWidget(w.bearing_task_state_label)
        controls.addWidget(w.bearing_progress)
        left.addWidget(command)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(290)
        body.addWidget(scroll)
        side = QWidget()
        sidebar = QVBoxLayout(side)
        scroll.setWidget(side)
        for widget in (w.bearing_subtitle_label,w.bearing_quality_card,w.bearing_result_card,
                       w.bearing_quality_message_label,w.bearing_details_label):
            sidebar.addWidget(widget)
        w.bearing_log = QTextEdit()
        w.bearing_log.setReadOnly(True)
        w.bearing_log.setMinimumHeight(100)
        sidebar.addWidget(w.bearing_log,1)
        sidebar.addStretch()
        w.bearing_panel.hide()
        return page

    def select_module(self,index):
        previous = self.pages.currentIndex()
        if previous == 1 and index != 1:
            if self.window.tsne_task.busy:
                self.window._cancel_tsne()
            else:
                self.window._invalidate_tsne()
        self.pages.setCurrentIndex(index)
        self.modules.button(index).setChecked(True)
        self.status.setText(('开发模型 · 已知源分类','历史射频 · 当前可用','技术验证 · 冻结模型')[index])
        # Bearing task deliberately survives navigation; results stay in its own page.

    def show_signal(self,*args):
        self.plot_stack.setCurrentIndex(0)
        self.plot_caption.setText('输入信号 · 使用当前文件与采样参数')

    def show_features(self,*args):
        self.plot_stack.setCurrentIndex(1)
        self.plot_caption.setText('模型特征 · ' + self.window.comboBox_choose_feature.currentText() + '（非原始信号）')


def install_recognition_workbench(window):
    while window.gridLayout_7.count():
        window.gridLayout_7.takeAt(0)
    view = RecognitionWorkbench(window)
    for widget in (window.groupBox_7,window.label_2,window.label_3,window.label_4,
                   window.label_41,window.label_mod):
        widget.hide()
    window.gridLayout_7.addWidget(view,0,0)
    window.tabWidget.setTabText(window.tabWidget.indexOf(window.tab_3),'智能识别')
    return view

