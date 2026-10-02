import os
import time
import sys
from pathlib import Path

# This project is built with PyQt5. Pin pyqtgraph's binding before it is
# imported, otherwise environments that also contain PySide6 may mix types.
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
os.environ['QT_API'] = 'pyqt5'

import torch
import datetime

# Windows OpenMP retains an intra-op pool for every short-lived Qt caller.
# Model loading alone reproduced +13 native threads per job at the default 14.
# Keep the desktop's CPU intra-op work single-threaded; CUDA kernels and the
# separate training entry points are unchanged. Do this before any Qt job starts.
if sys.platform == 'win32':
    torch.set_num_threads(1)

# CPU/GPU device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
from scipy.signal import hilbert
from PyQt5.QtGui import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from newwindow_ import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QListView, QAbstractItemView, QTreeView,\
    QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QProgressBar, QGroupBox, QRadioButton, QComboBox, QSpinBox, QPushButton, QFrame, QSizePolicy
from ui_theme import APP_STYLESHEET, apply_window_theme
from airwatch.ui.plots.waterfall import WaterfallPlotter
from airwatch.analysis.waterfall import prepare_waterfall_signal
from airwatch.analysis.generated_views import select_generated_sample
from airwatch.analysis.feature_views import select_feature_vector, compute_feature_spectrogram
from airwatch.ui.plots.feature_renderer import render_feature_vector, render_feature_spectrogram
from airwatch.analysis.signal_transforms import compute_constellation
from airwatch.ui.plots.pyqtgraph_renderer import render_constellation
from airwatch.inference.bearing_contract import DEFAULT_FROZEN_BEARING_CONTRACT
from airwatch.ui import BearingInferenceTask, build_bearing_diagnosis_presentation
from airwatch.ui.tsne_background import TsneTask
from airwatch.ui.recognition_background import RecognitionTask
from airwatch.ui.uav_background import UAVRecognitionTask
from airwatch.ui.generation_background import GenerationTask
from airwatch.ui.historical_comparison_background import HistoricalComparisonTask
from airwatch.inference.general_models import load_general_model
from airwatch.general_recognition_contract import task_spec
from airwatch.data.recognition_input import (
    discover_recognition_files,
    load_recognition_files,
)
from airwatch.data.home_input import load_home_signal
from airwatch.workflows.general_recognition import predict_one
from airwatch.workflows.ku_leuven_recognition import LazyKULeuvenRecognitionBackend
from airwatch.workflows.historical_generation import (
    ALL_CLASSES as GENERATION_ALL_CLASSES,
    CONSISTENCY_LIMITATION,
    GenerationRequest,
    HistoricalGenerationWorkflow,
    MAX_SAMPLES_PER_CLASS,
)
from airwatch.data.historical_comparison import (
    PERFORMANCE_LIMITATION as HISTORICAL_COMPARISON_LIMITATION,
    default_historical_comparison_manifest,
)
from airwatch.workflows.historical_model_comparison import (
    ComparisonRunRequest,
    HistoricalModelComparisonWorkflow,
)
from airwatch.analysis.tsne_views import prepare_tsne
from airwatch.analysis.home_features import prepare_home_feature_request
from airwatch.ui.home_feature_background import HomeFeatureTask
from airwatch.ui.plots.home_feature_renderer import render_home_feature
from airwatch.ui.signal_workbench import install_signal_workbench
from airwatch.ui.recognition_workbench import install_recognition_workbench
from airwatch.ui.plots.feature_renderer import render_tsne
from PyQt5.QtWidgets import QProgressDialog
from airwatch.ui.plots.pyqtgraph_renderer import render_spectrogram, render_spectrum, render_waveform
from airwatch.analysis.signal_transforms import compute_spectrogram
from airwatch.runtime_paths import ensure_user_directories, resource_path
from airwatch.runtime_resources import (
    HISTORICAL_GENERATOR_SLOT_ID,
    HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID,
    general_task_slot_id,
    model_slot,
    model_slot_resource_path,
)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.initUI()
        self.filename = []
        self.data = None
        self.home_data = None
        self.home_file = ''
        self.general_input = None
        self.label = None
        self.feature = []
        self.feature_row_counts = None
        self.feature_map = []
        self.signals = []
        self.gen_signals = None
        self.gen_labels = []
        self.generation_result = None
        self.save_path = ''
        self.home_feature_task = HomeFeatureTask(self)
        self.home_feature_target = None
        self.home_feature_progress = QProgressDialog(
            '准备主页专家分析', '取消', 0, 0, self
        )
        self.home_feature_progress.setWindowTitle('主页专家分析')
        self.home_feature_progress.setWindowModality(Qt.NonModal)
        self.home_feature_progress.setAutoClose(False)
        # Stop the constructor auto-show timer; hide() alone leaves it armed.
        self.home_feature_progress.reset()
        self.home_feature_progress.hide()
        self.home_feature_progress.canceled.connect(self._cancel_home_feature)
        self.home_feature_task.completed.connect(self._on_home_feature_result)
        self.home_feature_task.failed.connect(self._on_home_feature_error)
        self.home_feature_task.phase.connect(
            self.home_feature_progress.setLabelText
        )
        self.home_feature_task.busy_changed.connect(self._on_home_feature_busy)
        self.tsne_task = TsneTask(self)
        self.tsne_progress = QProgressDialog('准备 t-SNE', '取消', 0, 0, self)
        self.tsne_progress.setWindowTitle('特征分布后台计算')
        self.tsne_progress.setWindowModality(Qt.NonModal)
        self.tsne_progress.setAutoClose(False)
        # Stop the constructor auto-show timer; hide() alone leaves it armed.
        self.tsne_progress.reset()
        self.tsne_progress.hide()
        self.tsne_progress.canceled.connect(self._cancel_tsne)
        self.tsne_task.completed.connect(self._on_tsne_result)
        self.tsne_task.failed.connect(self._on_tsne_error)
        self.tsne_task.phase.connect(self.tsne_progress.setLabelText)
        self.tsne_task.busy_changed.connect(self._on_tsne_busy)
        self.generation_task = GenerationTask(
            HistoricalGenerationWorkflow(device=device), self
        )
        self.generation_progress = QProgressDialog(
            '准备历史信号生成', '取消', 0, 1, self
        )
        self.generation_progress.setWindowTitle('历史信号后台生成')
        self.generation_progress.setWindowModality(Qt.NonModal)
        self.generation_progress.setAutoClose(False)
        self.generation_progress.setAutoReset(False)
        self.generation_progress.reset()
        self.generation_progress.hide()
        self.generation_progress.canceled.connect(self.cancel_generation)
        self.generation_task.completed.connect(self._on_generation_completed)
        self.generation_task.failed.connect(self._on_generation_failed)
        self.generation_task.cancelled.connect(self._on_generation_cancelled)
        self.generation_task.busy_changed.connect(self._on_generation_busy)
        self.generation_task.phase.connect(self._on_generation_phase)
        self.generation_task.progress.connect(self._on_generation_progress)
        self.light_manifest_path = default_historical_comparison_manifest()
        self.light_comparison_result = None
        self.historical_comparison_task = HistoricalComparisonTask(
            HistoricalModelComparisonWorkflow(device=device), self
        )
        self.historical_comparison_progress = QProgressDialog(
            '准备历史模型无标签对比', '取消', 0, 1, self
        )
        self.historical_comparison_progress.setWindowTitle('历史模型后台对比')
        self.historical_comparison_progress.setWindowModality(Qt.NonModal)
        self.historical_comparison_progress.setAutoClose(False)
        self.historical_comparison_progress.setAutoReset(False)
        self.historical_comparison_progress.reset()
        self.historical_comparison_progress.hide()
        self.historical_comparison_progress.canceled.connect(
            self.cancel_historical_comparison
        )
        self.historical_comparison_task.completed.connect(
            self._on_historical_comparison_completed
        )
        self.historical_comparison_task.failed.connect(
            self._on_historical_comparison_failed
        )
        self.historical_comparison_task.cancelled.connect(
            self._on_historical_comparison_cancelled
        )
        self.historical_comparison_task.busy_changed.connect(
            self._on_historical_comparison_busy
        )
        self.historical_comparison_task.phase.connect(
            self._on_historical_comparison_phase
        )
        self.historical_comparison_task.progress.connect(
            self._on_historical_comparison_progress
        )
        self._configure_ui()
        self._configure_historical_model_availability()
        self.general_recognition_task = RecognitionTask(
            lambda task: load_general_model(task, device),
            predict_one,
            self,
        )
        self.general_recognition_task.completed.connect(self._on_general_recognition_completed)
        self.general_recognition_task.failed.connect(self._on_general_recognition_failed)
        self.general_recognition_task.cancelled.connect(self._on_general_recognition_cancelled)
        self.general_recognition_task.busy_changed.connect(self._on_general_recognition_busy)
        self.general_recognition_task.phase.connect(self._on_general_recognition_phase)
        self.general_recognition_task.progress.connect(self._on_general_recognition_progress)
        self.general_recognition_result = None
        self.uav_backend = LazyKULeuvenRecognitionBackend(
            device="cpu", batch_size=8
        )
        self.uav_task = UAVRecognitionTask(self.uav_backend, self)
        self.uav_task.completed.connect(self._on_uav_recognition_completed)
        self.uav_task.failed.connect(self._on_uav_recognition_failed)
        self.uav_task.cancelled.connect(self._on_uav_recognition_cancelled)
        self.uav_task.busy_changed.connect(self._on_uav_recognition_busy)
        self.uav_task.phase.connect(self._on_uav_recognition_phase)
        self.uav_task.progress.connect(self._on_uav_recognition_progress)
        self.recognition_workbench.uav_plots.recognition_ready_changed.connect(
            self._on_uav_input_readiness_changed
        )

        self.bearing_task = BearingInferenceTask(self)
        self.bearing_task.progress.connect(self._on_bearing_progress)
        self.bearing_task.completed.connect(self._on_bearing_completed)
        self.bearing_task.cancelled.connect(self._on_bearing_cancelled)
        self.bearing_task.failed.connect(self._on_bearing_failed)
        self.bearing_task.state_changed.connect(self._on_bearing_task_state_changed)

        ## ------------------编辑信号槽--------------------
        # ------------------Tab1控件------------------
        self.str2_fs = self.lineEdit_Fs_1.text()  # 获得用户输入采样率
        self.wlength = self.lineEdit_windowlength_1.text()  # 获得用户输入窗口长度

        self.pushButton_openfile.clicked.connect(self.openfile)  # 打开数据文件,保存到变量self.data
        self.pushButton_reload_TF.clicked.connect(self._refresh_home_view)
        self.label_signalshow_1.setMouseEnabled(x=True, y=False)  # 禁用轴操作
        self.radioButton_time_1.toggled.connect(
            lambda checked: self._on_home_view_toggled('time', checked)
        )
        self.radioButton_spectrum_1.toggled.connect(
            lambda checked: self._on_home_view_toggled('spectrum', checked))
        self.radioButton_TF_1.toggled.connect(
            lambda checked: self._on_home_view_toggled('spectrogram', checked)
        )
        self.radioButton_waterfall.toggled.connect(
            lambda checked: self._on_home_view_toggled('waterfall', checked)
        )
        self.pushButton_waterfall_play.clicked.connect(self.toggle_waterfall_playback)
        self.pushButton_waterfall_stop.clicked.connect(self.stop_waterfall_playback)
        self.comboBox_waterfall_fft.currentIndexChanged.connect(self._on_waterfall_settings_changed)
        self.spinBox_waterfall_history.valueChanged.connect(self._on_waterfall_settings_changed)
        self.comboBox_waterfall_speed.currentIndexChanged.connect(self._on_waterfall_settings_changed)
        self.tabWidget.currentChanged.connect(self._on_tab_changed)
        # self.comboBox_features.currentIndexChanged.connect(self.features)  # 时频图按钮
        # ------------------Tab1控件------------------

        # ------------------Tab2控件------------------
        self.pushButton_path.clicked.connect(self.load_data)  # 触发生成数据函数
        self.label_signal.setMouseEnabled(x=True, y=False)  # 禁用轴操作
        self.radioButton_time_3.toggled.connect(lambda checked: checked and self.change_original_signal())
        self.radioButton_TF_3.toggled.connect(lambda checked: checked and self.change_original_signal())
        self.radioButton_PP_3.toggled.connect(lambda checked: checked and self.change_original_signal())
        self.radioButton_time_3.toggled.connect(lambda checked: checked and self.plotsig_featuremap(self.feature_map,
                                                                                self.label_featuremap))  # 加载文件按钮

        self.radioButton_TF_3.toggled.connect(lambda checked: checked and self.plotspec_featuremap(
            self.feature_map, self.label_featuremap, self.lineEdit_Fs_2.text(), 16))  # 加载文件按钮
        self.comboBox_task.currentIndexChanged.connect(self.log_task)  # 时频图按钮
        self.comboBox_task.currentIndexChanged.connect(self._on_task_changed)
        self.pushButton_recognition.clicked.connect(self.start_general_recognition)  # 单文件后台识别
        self.comboBox.currentIndexChanged.connect(self.change_original_signal)  # 更换文件时频图按钮
        self.comboBox.currentIndexChanged.connect(self.change_feature_signal)  # 更换文件时频图按钮
        self.comboBox_choose_feature.currentIndexChanged.connect(self.choose_feature_signal)  # 更换文件时频图按钮
        # ------------------Tab2控件------------------

        # ------------------Tab3控件---------- --------
        self.pushButton_savepath.clicked.connect(self.select_save_path)  # 选择文件保存路径
        self.pushButton_generate.clicked.connect(self.generate_data)  # 触发生成数据函数
        self.comboBox_showclass.currentIndexChanged.connect(lambda: self.switch_gene_sig(self.signals,
                                                                                         self.label_genesig))
        self.pushButton_switchsignal.clicked.connect(lambda: self.switch_single_sig(self.signals,
                                                                                         self.label_genesig))
        self.radioButton_TF_2.toggled.connect(
            lambda checked: checked and self.plot_gene_sig(self.signals, self.label_genesig))
        self.radioButton_2.setText('星座图（实信号解析兼容）')
        self.radioButton_2.setToolTip('生成器输出实信号时，用 Hilbert 推导虚部；不是实测双路 I/Q。')
        self.radioButton_2.toggled.connect(
            lambda checked: checked and self.plot_gene_sig(self.signals, self.label_genesig))
        self.action_6.triggered.connect(lambda: self.features('小波特征'))
        self.actions.triggered.connect(lambda: self.features('双谱特征'))
        self.actionJR.triggered.connect(lambda: self.features('J、R特征'))
        self.actionHHT.triggered.connect(lambda: self.features('HHT特征'))
        self.actionHTT.triggered.connect(lambda: self.features('HHT特征'))
        # ------------------Tab3控件------------------

        # ------------------Tab4控件------------------
        self.pushButton_4.clicked.connect(self.show_lightweight_model_info)
        self.pushButton_qianhou.clicked.connect(self.qianhou)
        self.comboBox_task_2.currentIndexChanged.connect(self._on_light_task_changed)
        self.pushButton_path_2.clicked.connect(self.load_data_light)
        self.pushButton_recognition_2.clicked.connect(self.start_historical_comparison)
        # ------------------Tab4控件------------------
        ## ------------------编辑信号槽--------------------

        self.num = 0
        self.textEdit_log_1.append('初始化成功，欢迎进入智能信号处理原型验证系统软件！')
        self.task_name = self.comboBox_task.currentText()
        self.currentIndex = self.comboBox.currentIndex()
        self.feature_map = []
    def initUI(self):
        self.setupUi(self)
        self.setWindowTitle('空域电波哨兵 · 智能信号处理平台')

    def _configure_ui(self):
        """Set safe initial states, validation, hints and the visual theme."""
        self._create_waterfall_controls()
        self.waterfall_plotter = WaterfallPlotter(self.label_signalshow_1)
        self.waterfall_plotter.state_changed.connect(self._on_waterfall_state_changed)
        self.waterfall_plotter.progress_changed.connect(self._on_waterfall_progress)
        self.waterfall_plotter.playback_finished.connect(self._on_waterfall_finished)
        apply_window_theme(self)
        self.signal_workbench = install_signal_workbench(self)
        self.signal_workbench.expert_requested.connect(self.features)
        self._create_bearing_diagnosis_panel()
        self.recognition_workbench = install_recognition_workbench(self)
        # Create only user-writable directories here.  In a packaged build the
        # application resources may live below a read-only install directory.
        # Source runs still resolve to the repository root for compatibility.
        self.runtime_directories = ensure_user_directories()
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), '主页')
        self.setMinimumSize(1180, 760)
        self.resize(1440, 920)
        int_validator = QIntValidator(1, 1000000000, self)
        for edit in (self.lineEdit_Fs_1, self.lineEdit_Fs_2,
                     self.lineEdit_windowlength_1, self.lineEdit_windowlength_3,
                     self.lineEdit_windowlength_13,
                     self.lineEdit_switchsignal):
            edit.setValidator(int_validator)
        self.lineEdit_samplenum.setValidator(
            QIntValidator(2, MAX_SAMPLES_PER_CLASS, self)
        )
        self.lineEdit_samplenum.setToolTip(
            '每类 2–10,000 个；单次总量最多 30,000 个。全选 15 类时每类最多 2,000 个。'
        )
        self.lineEdit_path.setReadOnly(True)
        self.lineEdit_path_2.setReadOnly(True)
        self.lineEdit_savepath.setReadOnly(True)
        self.lineEdit_as.setReadOnly(True)
        self.lineEdit_score.setReadOnly(True)
        self.groupBox_5.setTitle('批内信号统计偏离（仅辅助定位异常样本）')
        self.label_39.setText('平均偏离：')
        self.label_44.setText('当前样本：')
        self.groupBox_5.setToolTip(CONSISTENCY_LIMITATION)
        self.pushButton_reload_TF.setEnabled(False)
        self.pushButton_recognition.setEnabled(False)
        self.pushButton_recognition_2.setEnabled(False)
        self.pushButton_generate.setEnabled(False)
        self.pushButton_switchsignal.setEnabled(False)
        self.statusbar.showMessage(
            f"就绪  |  计算设备: {'GPU / CUDA' if device.type == 'cuda' else 'CPU'}"
        )
        self.textEdit_log_1.append('提示：HHT 使用可选 EMD-signal 后端；点击分析时检查可用性。')
        self.comboBox_task_2.blockSignals(True)
        self.comboBox_task_2.clear()
        self.comboBox_task_2.addItem('信号个体识别')
        self.comboBox_task_2.blockSignals(False)
        self.groupBox_11.setTitle('历史模型无标签对比（非准确率）')
        self.pushButton_4.setText('模型槽位说明')
        self.pushButton_qianhou.setText('扩充评测证据说明')
        self.pushButton_path_2.setText('选择比较清单')
        self.pushButton_recognition_2.setText('开始无标签对比')
        self.lineEdit_path_2.setText(self.light_manifest_path.name)
        self.lineEdit_path_2.setToolTip(str(self.light_manifest_path))
        self.textEdit_6.setPlainText(
            '历史 TCN 对照模型\n等待开始；软件演示数据不提供真实标签。'
        )
        self.textEdit_7.setPlainText(
            '历史 10 层轻量模型\n等待开始；不发布 OA、宏平均精度、F1 或时延。'
        )
        self.groupBox_11.setToolTip(HISTORICAL_COMPARISON_LIMITATION)

    def _configure_historical_model_availability(self):
        """Reflect registered historical model availability in the live UI."""

        generator_slot = model_slot(HISTORICAL_GENERATOR_SLOT_ID)
        try:
            model_slot_resource_path(HISTORICAL_GENERATOR_SLOT_ID)
            self.generator_model_available = True
            self.pushButton_generate.setToolTip(
                f'已登记：{generator_slot.display_name}；选择保存目录后可开始生成。'
            )
        except Exception as exc:
            self.generator_model_available = False
            self.pushButton_generate.setEnabled(False)
            self.pushButton_generate.setToolTip(f'生成模型不可用：{exc}')

        reference_slot = model_slot(HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID)
        individual_slot_id = general_task_slot_id('信号个体识别')
        try:
            model_slot_resource_path(HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID)
            model_slot_resource_path(individual_slot_id)
            self.lightweight_models_available = True
            self.pushButton_recognition_2.setToolTip(
                f'已登记：{reference_slot.display_name}与历史轻量模型；'
                '通过校验清单后执行无标签软件路径对比，不发布准确率。'
            )
            self.pushButton_recognition_2.setEnabled(
                self.light_manifest_path.is_file()
            )
        except Exception as exc:
            self.lightweight_models_available = False
            self.pushButton_recognition_2.setEnabled(False)
            self.pushButton_recognition_2.setToolTip(f'轻量化对比模型不完整：{exc}')

    def _create_bearing_diagnosis_panel(self):
        """Create the independent bearing entry without touching generated UI code.

        This is deliberately presentation-only.  The panel keeps the existing
        worker/controller and frozen-contract interfaces intact while making the
        user path legible: choose data -> start -> watch progress -> read result.
        """
        self.bearing_panel = QGroupBox('轴承故障诊断', self.tab_3)
        self.bearing_panel.setObjectName('bearingDiagnosisPanel')
        self.bearing_panel.setToolTip(
            '使用已冻结的抗噪声模型进行后台诊断；不会训练模型，也不会替换历史权重。'
        )
        self.bearing_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        panel_layout = QVBoxLayout(self.bearing_panel)
        panel_layout.setContentsMargins(14, 10, 14, 12)
        panel_layout.setSpacing(8)

        self.bearing_subtitle_label = QLabel(
            '冻结抗噪声模型 · CWRU .mat 数据 · 后台推理（不训练、不覆盖历史权重）',
            self.bearing_panel,
        )
        self.bearing_subtitle_label.setObjectName('bearingSubtitle')
        self.bearing_subtitle_label.setWordWrap(True)
        panel_layout.addWidget(self.bearing_subtitle_label)

        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        file_label = QLabel('数据文件', self.bearing_panel)
        file_label.setObjectName('bearingSectionLabel')
        file_row.addWidget(file_label)
        self.bearing_file_edit = QLineEdit(self.bearing_panel)
        self.bearing_file_edit.setObjectName('bearingFilePath')
        self.bearing_file_edit.setReadOnly(True)
        self.bearing_file_edit.setPlaceholderText('请选择 CWRU 轴承 .mat 文件')
        self.bearing_file_edit.setToolTip('当前选中的 CWRU 轴承振动数据文件')
        file_row.addWidget(self.bearing_file_edit, 1)
        self.bearing_select_button = QPushButton('选择文件', self.bearing_panel)
        self.bearing_select_button.setObjectName('bearingSelectButton')
        self.bearing_select_button.setToolTip('选择一个 CWRU .mat 轴承振动数据文件')
        self.bearing_select_button.clicked.connect(self.select_bearing_file)
        file_row.addWidget(self.bearing_select_button)
        panel_layout.addLayout(file_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.bearing_start_button = QPushButton('开始后台诊断', self.bearing_panel)
        self.bearing_start_button.setObjectName('bearingStartButton')
        self.bearing_start_button.setToolTip('使用冻结契约在后台处理当前文件')
        self.bearing_start_button.setEnabled(False)
        self.bearing_start_button.clicked.connect(self.start_bearing_diagnosis)
        action_row.addWidget(self.bearing_start_button)
        self.bearing_cancel_button = QPushButton('取消', self.bearing_panel)
        self.bearing_cancel_button.setObjectName('bearingCancelButton')
        self.bearing_cancel_button.setToolTip('安全取消当前任务，不保留本次部分结果')
        self.bearing_cancel_button.setEnabled(False)
        self.bearing_cancel_button.clicked.connect(self.cancel_bearing_diagnosis)
        action_row.addWidget(self.bearing_cancel_button)
        action_row.addStretch(1)
        self.bearing_task_state_label = QLabel('尚未开始', self.bearing_panel)
        self.bearing_task_state_label.setObjectName('bearingTaskState')
        self.bearing_task_state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        action_row.addWidget(self.bearing_task_state_label)
        panel_layout.addLayout(action_row)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        progress_label = QLabel('处理进度', self.bearing_panel)
        progress_label.setObjectName('bearingSectionLabel')
        progress_row.addWidget(progress_label)
        self.bearing_progress = QProgressBar(self.bearing_panel)
        self.bearing_progress.setObjectName('bearingProgress')
        self.bearing_progress.setRange(0, 1)
        self.bearing_progress.setValue(0)
        self.bearing_progress.setFormat('未开始')
        progress_row.addWidget(self.bearing_progress, 1)
        panel_layout.addLayout(progress_row)

        result_row = QHBoxLayout()
        result_row.setSpacing(10)
        self.bearing_quality_card = QFrame(self.bearing_panel)
        self.bearing_quality_card.setObjectName('bearingSummaryCard')
        quality_card_layout = QVBoxLayout(self.bearing_quality_card)
        quality_card_layout.setContentsMargins(12, 8, 12, 8)
        quality_card_layout.setSpacing(3)
        quality_caption = QLabel('信号质量', self.bearing_quality_card)
        quality_caption.setObjectName('bearingCardCaption')
        quality_card_layout.addWidget(quality_caption)
        self.bearing_quality_status_label = QLabel('尚未诊断', self.bearing_panel)
        self.bearing_quality_status_label.setObjectName('bearingQualityStatus')
        quality_card_layout.addWidget(self.bearing_quality_status_label)
        result_row.addWidget(self.bearing_quality_card, 1)

        self.bearing_result_card = QFrame(self.bearing_panel)
        self.bearing_result_card.setObjectName('bearingSummaryCard')
        result_card_layout = QVBoxLayout(self.bearing_result_card)
        result_card_layout.setContentsMargins(12, 8, 12, 8)
        result_card_layout.setSpacing(3)
        result_caption = QLabel('诊断结论', self.bearing_result_card)
        result_caption.setObjectName('bearingCardCaption')
        result_card_layout.addWidget(result_caption)
        self.bearing_diagnosis_label = QLabel('—', self.bearing_panel)
        self.bearing_diagnosis_label.setObjectName('bearingDiagnosisResult')
        self.bearing_diagnosis_label.setWordWrap(True)
        result_card_layout.addWidget(self.bearing_diagnosis_label)
        self.bearing_confidence_label = QLabel('—', self.bearing_panel)
        self.bearing_confidence_label.setObjectName('bearingConfidence')
        result_card_layout.addWidget(self.bearing_confidence_label)
        result_row.addWidget(self.bearing_result_card, 1)
        panel_layout.addLayout(result_row)

        self.bearing_quality_message_label = QLabel('选择文件后开始诊断。', self.bearing_panel)
        self.bearing_quality_message_label.setObjectName('bearingQualityMessage')
        self.bearing_quality_message_label.setWordWrap(True)
        panel_layout.addWidget(self.bearing_quality_message_label)

        self.bearing_details_label = QLabel('', self.bearing_panel)
        self.bearing_details_label.setObjectName('bearingDetails')
        self.bearing_details_label.setWordWrap(True)
        panel_layout.addWidget(self.bearing_details_label)

        # ``verticalLayout_2`` is the maintained log area on the recognition
        # page.  Inserting the panel here keeps the generated UI untouched and
        # keeps the new workflow visually separate from the five legacy tasks.
        self.verticalLayout_2.insertWidget(1, self.bearing_panel)
        self._set_bearing_quality_state('idle', '尚未诊断')

    def _set_bearing_quality_state(self, state, title):
        self.bearing_quality_status_label.setText(title)
        self.bearing_quality_status_label.setProperty('qualityState', state)
        self.bearing_quality_card.setProperty('qualityState', state)
        self.bearing_quality_status_label.style().unpolish(self.bearing_quality_status_label)
        self.bearing_quality_status_label.style().polish(self.bearing_quality_status_label)
        self.bearing_quality_card.style().unpolish(self.bearing_quality_card)
        self.bearing_quality_card.style().polish(self.bearing_quality_card)

    def select_bearing_file(self):
        if hasattr(self, 'bearing_task') and self.bearing_task.is_running:
            return
        default_dir = resource_path('datasets', 'bearing', 'cwru', 'raw')
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择轴承振动数据',
            str(default_dir if default_dir.is_dir() else Path.cwd()),
            'MAT 数据 (*.mat);;所有文件 (*)',
        )
        if not file_path:
            return
        self.bearing_file_edit.setText(str(Path(file_path).resolve()))
        self._reset_bearing_result()
        self.bearing_start_button.setEnabled(True)
        self.statusbar.showMessage(f'已选择轴承数据：{Path(file_path).name}')
        self.recognition_workbench.bearing_plots.select_file(file_path)

    def start_bearing_diagnosis(self):
        if self.bearing_task.is_running:
            return
        input_path = self.bearing_file_edit.text().strip()
        if not input_path:
            QMessageBox.information(self, '请先选择数据', '请先选择一个轴承 .mat 文件。')
            return
        path = Path(input_path)
        if not path.is_file():
            QMessageBox.warning(self, '数据不存在', f'找不到所选数据文件：\n{path}')
            self._reset_bearing_result()
            return
        self._reset_bearing_result()
        self.bearing_task_state_label.setText('准备中…')
        self.bearing_progress.setRange(0, 0)
        self.bearing_progress.setFormat('准备中…')
        self._set_bearing_quality_state('idle', '正在检查')
        self.bearing_diagnosis_label.setText('等待诊断结果')
        self.bearing_quality_message_label.setText(
            '正在后台核对冻结契约并加载模型，界面仍可操作。'
        )
        self.statusbar.showMessage(
            '轴承诊断后台处理中；不会训练模型，也不会修改历史权重。'
        )
        try:
            self.bearing_task.start(
                input_path=path,
                contract_path=DEFAULT_FROZEN_BEARING_CONTRACT,
                device=str(device),
                batch_size=32,
            )
        except Exception as exc:
            self._on_bearing_failed({
                'title': '轴承诊断失败',
                'message': f'无法启动后台诊断：{exc}',
                'technical_detail': repr(exc),
            })

    def cancel_bearing_diagnosis(self):
        if not self.bearing_task.cancel():
            return
        self.bearing_task_state_label.setText('正在安全取消…')
        self.bearing_quality_message_label.setText(
            '已发出取消请求；当前模型批次结束后安全退出，不保留本次部分结果。'
        )
        self.statusbar.showMessage('正在安全取消轴承诊断…')

    def _on_bearing_progress(self, current, total, message):
        if total <= 0:
            self.bearing_progress.setRange(0, 0)
            self.bearing_progress.setFormat('处理中…')
            self.bearing_task_state_label.setText('后台处理中')
        else:
            self.bearing_progress.setRange(0, total)
            self.bearing_progress.setValue(min(max(int(current), 0), int(total)))
            self.bearing_progress.setFormat(f'{int(current)} / {int(total)} 个窗口')
            self.bearing_task_state_label.setText('后台处理中')
        self.bearing_quality_message_label.setText(str(message))
        self.statusbar.showMessage(f'轴承诊断：{message}')

    def _on_bearing_completed(self, result):
        try:
            presentation = build_bearing_diagnosis_presentation(result)
        except Exception as exc:
            self._on_bearing_failed({
                'title': '结果展示失败',
                'message': f'后台诊断已完成，但结果格式不符合冻结展示规则：{exc}',
                'technical_detail': repr(exc),
            })
            return
        self._set_bearing_quality_state(
            presentation.quality_status, presentation.state_title
        )
        self.bearing_task_state_label.setText('已完成')
        self.bearing_diagnosis_label.setText(presentation.diagnosis_text)
        self.bearing_confidence_label.setText(presentation.confidence_text)
        self.bearing_quality_message_label.setText(presentation.quality_message)
        self.bearing_details_label.setText(presentation.details_text)
        self.bearing_details_label.setToolTip(
            '结果来自冻结模型契约；模型文件、归一化、窗口数和耗时均为本次后台返回值。')
        window_count = int(result.get('window_count', 0))
        self.bearing_progress.setRange(0, max(window_count, 1))
        self.bearing_progress.setValue(window_count)
        self.bearing_progress.setFormat(f'{window_count} / {window_count} 个窗口')
        self.statusbar.showMessage(
            f'轴承诊断完成：{presentation.state_title}（{window_count} 个窗口）'
        )
        self.bearing_log.append(
            f'轴承后台诊断完成：{presentation.state_title}；{presentation.diagnosis_text}；'
            f'{presentation.confidence_text}'
        )

    def _on_bearing_cancelled(self, message):
        self._reset_bearing_result()
        self.bearing_task_state_label.setText('已取消')
        self.bearing_quality_message_label.setText(
            str(message) or '轴承诊断已取消，未生成新的诊断结果。'
        )
        self.bearing_progress.setRange(0, 1)
        self.bearing_progress.setValue(0)
        self.bearing_progress.setFormat('已取消')
        self.statusbar.showMessage('轴承诊断已取消，未保留部分结果。')
        self.bearing_log.append('轴承后台诊断已取消，未保留部分结果。')

    def _on_bearing_failed(self, error):
        self._reset_bearing_result()
        self.bearing_task_state_label.setText('失败')
        message = str(error.get('message') or '后台轴承诊断失败。')
        detail = str(error.get('technical_detail') or '')
        self.bearing_quality_message_label.setText(
            '本次诊断未完成，未保留旧结果。'
        )
        self.bearing_details_label.setText(detail)
        self.statusbar.showMessage('轴承诊断失败')
        self.bearing_log.append(f'轴承后台诊断失败：{message}')
        QMessageBox.critical(self, str(error.get('title') or '轴承诊断失败'), message)

    def _on_bearing_task_state_changed(self, state):
        running = state in {'running', 'cancelling'}
        self.bearing_select_button.setEnabled(not running)
        self.bearing_start_button.setEnabled(not running and bool(self.bearing_file_edit.text().strip()))
        self.bearing_cancel_button.setEnabled(state == 'running')
        if state == 'running':
            self.bearing_task_state_label.setText('后台处理中')
            self.bearing_start_button.setText('诊断进行中…')
        elif state == 'cancelling':
            self.bearing_task_state_label.setText('正在安全取消…')
            self.bearing_start_button.setText('等待取消…')
        elif state == 'idle':
            self.bearing_start_button.setText('开始后台诊断')
            if self.bearing_task_state_label.text() in {'后台处理中', '正在安全取消…'}:
                self.bearing_task_state_label.setText('已结束')
        else:
            self.bearing_start_button.setText('开始后台诊断')

    def _reset_bearing_result(self):
        self._set_bearing_quality_state('idle', '尚未诊断')
        self.bearing_diagnosis_label.setText('—')
        self.bearing_confidence_label.setText('—')
        self.bearing_quality_message_label.setText('选择文件后开始诊断。')
        self.bearing_details_label.setText('')
        self.bearing_progress.setRange(0, 1)
        self.bearing_progress.setValue(0)
        self.bearing_progress.setFormat('未开始')

    def _create_waterfall_controls(self):
        # The generated UI creates this label but forgets to add it to the row.
        self.label_windowlength_1.setObjectName('label_windowlength_1')
        self.horizontalLayout_2.insertWidget(2, self.label_windowlength_1)
        self.radioButton_waterfall = QRadioButton('动态瀑布图', self.groupBox_paraset)
        self.radioButton_waterfall.setToolTip('按真实时间比例动态回放信号频谱')
        reload_index = max(0, self.horizontalLayout_2.count() - 1)
        self.horizontalLayout_2.insertWidget(reload_index, self.radioButton_waterfall)
        self.pushButton_reload_TF.setText('刷新当前图')

        self.waterfall_panel = QFrame(self.groupBox_paraset)
        self.waterfall_panel.setObjectName('waterfallPanel')
        panel_layout = QHBoxLayout(self.waterfall_panel)
        panel_layout.setContentsMargins(10, 6, 10, 6)
        panel_layout.setSpacing(8)

        panel_layout.addWidget(QLabel('瀑布设置'))
        panel_layout.addWidget(QLabel('FFT 点数'))
        self.comboBox_waterfall_fft = QComboBox(self.waterfall_panel)
        self.comboBox_waterfall_fft.addItems(['128', '256', '512', '1024', '2048'])
        self.comboBox_waterfall_fft.setCurrentText('256')
        self.comboBox_waterfall_fft.setToolTip('点数越大，频率分辨率越高，但时间分辨率越低')
        panel_layout.addWidget(self.comboBox_waterfall_fft)

        panel_layout.addWidget(QLabel('历史行数'))
        self.spinBox_waterfall_history = QSpinBox(self.waterfall_panel)
        self.spinBox_waterfall_history.setRange(50, 1000)
        self.spinBox_waterfall_history.setSingleStep(50)
        self.spinBox_waterfall_history.setValue(300)
        self.spinBox_waterfall_history.setToolTip('屏幕中保留的历史频谱行数')
        panel_layout.addWidget(self.spinBox_waterfall_history)

        panel_layout.addWidget(QLabel('播放速度'))
        self.comboBox_waterfall_speed = QComboBox(self.waterfall_panel)
        for text, speed in [('0.5×', 0.5), ('1×', 1.0), ('2×', 2.0), ('4×', 4.0)]:
            self.comboBox_waterfall_speed.addItem(text, speed)
        self.comboBox_waterfall_speed.setCurrentIndex(1)
        panel_layout.addWidget(self.comboBox_waterfall_speed)

        self.label_waterfall_progress = QLabel('0 / 0 帧', self.waterfall_panel)
        self.label_waterfall_progress.setObjectName('waterfallProgress')
        panel_layout.addWidget(self.label_waterfall_progress)
        panel_layout.addStretch(1)

        self.pushButton_waterfall_play = QPushButton('播放', self.waterfall_panel)
        self.pushButton_waterfall_play.setObjectName('waterfallPlayButton')
        self.pushButton_waterfall_play.setEnabled(False)
        panel_layout.addWidget(self.pushButton_waterfall_play)
        self.pushButton_waterfall_stop = QPushButton('停止', self.waterfall_panel)
        self.pushButton_waterfall_stop.setObjectName('waterfallStopButton')
        self.pushButton_waterfall_stop.setEnabled(False)
        panel_layout.addWidget(self.pushButton_waterfall_stop)

        self.verticalLayout_7.addWidget(self.waterfall_panel)
        self.waterfall_panel.setVisible(False)

    def _on_home_view_toggled(self, mode, checked):
        if not checked:
            return
        self._cancel_home_feature(message=None)
        self.signal_workbench.show_signal({'time': 'waveform'}.get(mode, mode))
        is_waterfall = mode == 'waterfall'
        self.waterfall_panel.setVisible(is_waterfall)
        if is_waterfall:
            self.pushButton_waterfall_play.setEnabled(self.home_data is not None)
            if self.home_data is None:
                self.label_waterfall_progress.setText('等待数据')
                self.textEdit_log_1.append('瀑布图：请先打开一个信号文件。')
                return
            self.start_waterfall_playback()
            return

        self.waterfall_plotter.stop(clear=True)
        if self.home_data is None:
            return
        if mode == 'time':
            self.plotsig(self.home_data, self.label_signalshow_1)
        elif mode == 'spectrum':
            self.plotspectrum(self.home_data, self.label_signalshow_1)
        else:
            self.plotspec(
                self.home_data,
                self.label_signalshow_1,
            )

    def _refresh_home_view(self):
        if self.home_data is None:
            QMessageBox.information(self, '请先加载', '请先打开一个信号文件。')
            return
        if self.radioButton_waterfall.isChecked():
            self.start_waterfall_playback()
        elif self.radioButton_time_1.isChecked():
            self._on_home_view_toggled('time', True)
        elif self.radioButton_spectrum_1.isChecked():
            self._on_home_view_toggled('spectrum', True)
        else:
            self._on_home_view_toggled('spectrogram', True)

    def _waterfall_signal(self):
        return prepare_waterfall_signal(self.home_data)

    def start_waterfall_playback(self):
        if self.home_data is None:
            self.waterfall_plotter.stop(clear=True)
            QMessageBox.information(self, '请先加载', '请先打开一个信号文件。')
            return False
        try:
            fs = float(self.lineEdit_Fs_1.text())
            n_fft = int(self.comboBox_waterfall_fft.currentText())
            history = self.spinBox_waterfall_history.value()
            speed = float(self.comboBox_waterfall_speed.currentData())
            signal = self._waterfall_signal()
            self.waterfall_plotter.start(
                signal,
                fs,
                n_fft,
                history_lines=history,
                overlap_ratio=0.5,
                playback_speed=speed,
            )
            self.textEdit_log_1.append(
                f'瀑布图开始播放：采样率 {fs} Hz，FFT {n_fft} 点，'
                f'历史 {history} 行，速度 {speed:g}×。'
            )
            self.statusbar.showMessage(
                f'瀑布图播放中  |  FFT {n_fft}  |  {speed:g}×'
            )
            return True
        except Exception as e:
            self.waterfall_plotter.stop(clear=True)
            self.label_waterfall_progress.setText('无法播放')
            message = f'请检查信号形状、有效数值、采样率及 FFT 点数。详情：{e}'
            self.statusbar.showMessage(f'瀑布图无法播放：{message}')
            QMessageBox.warning(self, '无法播放瀑布图', message)
            return False

    def toggle_waterfall_playback(self):
        if self.waterfall_plotter.is_running:
            self.waterfall_plotter.pause()
        elif self.waterfall_plotter.is_paused:
            self.waterfall_plotter.resume()
        else:
            self.start_waterfall_playback()

    def stop_waterfall_playback(self):
        self.waterfall_plotter.stop(clear=False)
        self.label_waterfall_progress.setText('已停止')
        self.statusbar.showMessage('瀑布图已停止')

    def _on_waterfall_settings_changed(self, *_):
        if not hasattr(self, 'waterfall_plotter'):
            return
        if self.radioButton_waterfall.isChecked() and self.waterfall_plotter.state != 'stopped':
            self.waterfall_plotter.stop(clear=False)
            self.label_waterfall_progress.setText('参数已更新')
            self.pushButton_waterfall_play.setText('重新播放')

    def _on_waterfall_state_changed(self, state):
        labels = {
            'running': '暂停',
            'paused': '继续',
            'finished': '重新播放',
            'stopped': '播放',
        }
        self.pushButton_waterfall_play.setText(labels.get(state, '播放'))
        has_data = self.home_data is not None
        self.pushButton_waterfall_play.setEnabled(has_data and self.radioButton_waterfall.isChecked())
        self.pushButton_waterfall_stop.setEnabled(
            has_data and self.radioButton_waterfall.isChecked() and state != 'stopped'
        )

    def _on_waterfall_progress(self, current, total):
        self.label_waterfall_progress.setText(f'{current} / {total} 帧')

    def _on_waterfall_finished(self):
        self.statusbar.showMessage('瀑布图播放完成，可点击“重新播放”再次查看')
        self.textEdit_log_1.append('瀑布图播放完成。')

    def _on_tab_changed(self, index):
        if index != self.tabWidget.indexOf(self.tab1) and self.waterfall_plotter.is_running:
            self.waterfall_plotter.pause()

    def closeEvent(self, event):
        if self.historical_comparison_task.busy:
            self.historical_comparison_task.cancel(invalidate=True)
            if not self.historical_comparison_task.wait(5000):
                self.statusbar.showMessage('历史模型比较正在安全收尾，请稍后再次关闭。')
                event.ignore()
                return
        if self.generation_task.busy:
            self.generation_task.cancel(invalidate=True)
            if not self.generation_task.wait(5000):
                self.statusbar.showMessage('历史信号生成正在安全收尾，请稍后再次关闭。')
                event.ignore()
                return
        self.recognition_workbench.uav_plots.cancel()
        if not self.recognition_workbench.uav_plots.wait(100):
            self.statusbar.showMessage('无人机图预览正在安全收尾，请稍后再次关闭。')
            event.ignore()
            return
        if self.uav_task.busy:
            self.uav_task.cancel(invalidate=True)
            if not self.uav_task.wait(5000):
                self.statusbar.showMessage('无人机识别正在安全收尾，请稍后再次关闭。')
                event.ignore()
                return
        self.recognition_workbench.bearing_plots.cancel()
        if not self.recognition_workbench.bearing_plots.wait(100):
            self.statusbar.showMessage('轴承图预览正在安全收尾，请稍后再次关闭。')
            event.ignore()
            return
        self._cancel_home_feature(message=None)
        if not self.home_feature_task.wait(100):
            self.statusbar.showMessage('主页专家分析正在安全收尾，请稍后再次关闭。')
            event.ignore()
            return
        if self.general_recognition_task.busy:
            self.general_recognition_task.cancel(invalidate=True)
            if not self.general_recognition_task.wait(5000):
                self.statusbar.showMessage('通用识别正在安全收尾，请稍后再次关闭。')
                event.ignore()
                return
        self._invalidate_tsne()
        if not self.tsne_task.wait(100):
            self.statusbar.showMessage('t-SNE 正在安全收尾，请稍后再次关闭窗口。')
            event.ignore()
            return
        if hasattr(self, 'bearing_task') and self.bearing_task.is_running:
            self.bearing_task.cancel()
            if not self.bearing_task.wait(5000):
                QMessageBox.warning(
                    self,
                    '后台任务仍在运行',
                    '轴承诊断正在安全退出，请稍后再次关闭窗口。',
                )
                event.ignore()
                return
        if hasattr(self, 'waterfall_plotter'):
            self.waterfall_plotter.stop(clear=True)
        super().closeEvent(event)

    def _on_task_changed(self):
        if self.general_recognition_task.busy:
            self.general_recognition_task.cancel(invalidate=True)
        self.general_recognition_result = None
        self.recognition_workbench.general_result_card.setText('识别结果：—\n状态：待开始\n文件：—\n模型：—\n窗口：—\n耗时：—')
        self.recognition_workbench.general_progress.setRange(0,1); self.recognition_workbench.general_progress.setValue(0); self.recognition_workbench.general_progress.setFormat('未开始')
        self._invalidate_tsne()
        self.feature = []
        self.feature_map = []
        self.feature_row_counts = None
        self.label_featuremap.clear()
        self.general_input = None
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        self.comboBox.blockSignals(False)
        self.lineEdit_path.clear()
        self.label_signal.clear()
        self.pushButton_recognition.setEnabled(False)
        self.recognition_workbench.model_hint.setText(
            '任务已变更，请重新选择与当前输入契约匹配的文件。'
        )
        self.log_task()

    def _on_light_task_changed(self):
        self.light_comparison_result = None
        self.pushButton_recognition_2.setEnabled(
            self.light_manifest_path is not None
            and self.light_manifest_path.is_file()
            and self.lightweight_models_available
            and not self.historical_comparison_task.busy
        )
        self.textEdit_6.append('当前只开放经验证的历史个体识别双模型比较。')

    def show_lightweight_model_info(self):
        """Show registered runtime slots without loading checkpoints on the UI thread."""

        reference = model_slot(HISTORICAL_INDIVIDUAL_REFERENCE_SLOT_ID)
        lightweight = model_slot(general_task_slot_id('信号个体识别'))
        future = model_slot('edge.student')
        QMessageBox.information(
            self,
            '模型槽位说明',
            f'当前对照：{reference.display_name}\n'
            f'当前轻量：{lightweight.display_name}\n\n'
            '两者均为历史软件演示槽位，不代表竞赛端侧性能。\n\n'
            f'后续预留：{future.display_name}\n{future.limitation}',
        )

    def load_data_light(self):
        """Select a verified comparison manifest; file decoding stays in the Data Module."""

        manifest, _ = QFileDialog.getOpenFileName(
            self,
            '选择历史模型比较清单',
            str(self.light_manifest_path.parent),
            'JSON 清单 (*.json)',
        )
        if not manifest:
            return
        self.light_manifest_path = Path(manifest).resolve()
        self._reset_historical_comparison_result('已选择新清单，尚未运行比较。')
        self.lineEdit_path_2.setText(self.light_manifest_path.name)
        self.lineEdit_path_2.setToolTip(str(self.light_manifest_path))
        self.pushButton_recognition_2.setEnabled(
            self.light_manifest_path.is_file()
            and self.lightweight_models_available
            and not self.historical_comparison_task.busy
        )
        self.statusbar.showMessage('已选择比较清单；开始后将在后台校验文件与 SHA-256。')

    def start_historical_comparison(self):
        if self.historical_comparison_task.busy:
            return
        if self.light_manifest_path is None or not self.light_manifest_path.is_file():
            QMessageBox.warning(self, '缺少清单', '请选择存在的历史模型比较清单。')
            return
        try:
            request = ComparisonRunRequest(
                manifest_path=self.light_manifest_path,
                output_root=self.runtime_directories['comparison'],
            )
            self._reset_historical_comparison_result('历史模型比较正在后台运行；尚未发布本轮结果。')
            self.historical_comparison_progress.setRange(0, 1)
            self.historical_comparison_progress.setValue(0)
            self.historical_comparison_progress.setLabelText('准备校验比较清单…')
            self.historical_comparison_progress.show()
            self.historical_comparison_task.start(request)
        except Exception as exc:
            self.historical_comparison_progress.reset()
            self.historical_comparison_progress.hide()
            QMessageBox.critical(self, '无法开始历史模型比较', str(exc))

    def cancel_historical_comparison(self):
        if self.historical_comparison_task.cancel():
            self.historical_comparison_progress.setLabelText(
                '正在取消；不会发布部分 Evidence…'
            )

    def _reset_historical_comparison_result(self, message):
        self.light_comparison_result = None
        self.textEdit_6.setPlainText(message)
        self.textEdit_7.setPlainText(message)

    def _on_historical_comparison_busy(self, busy):
        self.comboBox_task_2.setEnabled(not busy)
        self.pushButton_path_2.setEnabled(not busy)
        self.pushButton_4.setEnabled(not busy)
        self.pushButton_recognition_2.setEnabled(
            not busy
            and self.lightweight_models_available
            and self.light_manifest_path is not None
            and self.light_manifest_path.is_file()
        )
        if not busy:
            self.historical_comparison_progress.reset()
            self.historical_comparison_progress.hide()

    def _on_historical_comparison_phase(self, message):
        self.historical_comparison_progress.setLabelText(message)
        self.statusbar.showMessage(message)

    def _on_historical_comparison_progress(self, current, total):
        self.historical_comparison_progress.setRange(0, max(1, total))
        self.historical_comparison_progress.setValue(current)

    @staticmethod
    def _comparison_prediction_text(predictions):
        return '\n'.join(
            f'文件 {index}: {label}'
            for index, label in enumerate(predictions, start=1)
        )

    def _on_historical_comparison_completed(self, result):
        self.light_comparison_result = result
        reference = result.reference
        lightweight = result.lightweight
        reduction = (
            1 - lightweight.parameter_count / reference.parameter_count
        ) * 100
        shared = (
            f'\n\n输入：{result.recording_count} 个记录 / {result.total_windows} 个窗口\n'
            f'两模型窗口输出一致：{result.agreement_count}/{result.total_windows} '
            f'({result.agreement_rate:.2%})；这不是准确率。\n'
            '准确率、宏平均精度、F1、端侧时延：未发布\n'
            f'Evidence：{result.evidence_file}\n'
            f'限制：{result.limitation}'
        )
        self.textEdit_6.setPlainText(
            f'{reference.role}\n'
            f'模型：{reference.model_name}\n'
            f'权重：{reference.checkpoint_name} ({reference.checkpoint_bytes:,} 字节)\n'
            f'参数量：{reference.parameter_count:,}\n'
            f'{self._comparison_prediction_text(reference.file_predictions)}'
            + shared
        )
        self.textEdit_7.setPlainText(
            f'{lightweight.role}\n'
            f'模型：{lightweight.model_name}\n'
            f'权重：{lightweight.checkpoint_name} ({lightweight.checkpoint_bytes:,} 字节)\n'
            f'参数量：{lightweight.parameter_count:,}\n'
            f'相对对照参数量减少：{reduction:.2f}%\n'
            f'{self._comparison_prediction_text(lightweight.file_predictions)}'
            + shared
        )
        self.statusbar.showMessage('历史模型无标签比较完成；Evidence 已原子写入。')

    def _on_historical_comparison_failed(self, message):
        self._reset_historical_comparison_result(f'比较失败，未发布本轮结果：{message}')
        self.statusbar.showMessage('历史模型比较失败，未发布部分 Evidence。')
        QMessageBox.critical(self, '历史模型比较失败', message)

    def _on_historical_comparison_cancelled(self):
        self._reset_historical_comparison_result('比较已取消，未发布本轮结果。')
        self.statusbar.showMessage('历史模型比较已取消，未发布部分 Evidence。')


    #  自定义信号
    def openfile(self):
        try:
            dic = {'modulation': '短波', 'yewu': '短波', 'individual': '卫星', 'tonglian': '超短波', 'bianma': ''}
            self.str2_fs = self.lineEdit_Fs_1.text()  # 获得用户输入采样率
            self.wlength = self.lineEdit_windowlength_1.text()  # 获得用户输入窗口长度
            data_file, _ = QFileDialog.getOpenFileName(self, '打开数据文件', './data/', 'NumPy 数据 (*.npy)')
            if not data_file:
                return
            data = load_home_signal(data_file)
            self._cancel_home_feature(message=None)
            class_name = Path(data_file).parent.name
            self.label_filename_1.setText(Path(data_file).name)
            self.home_data = data
            self.home_file = data_file
            self.signal_workbench.reset_results()
            self.label_filename_1.setToolTip(str(data_file))
            self._refresh_home_view()
            self.pushButton_reload_TF.setEnabled(True)
            self.textEdit_log_1.append(
                f'成功打开{dic.get(class_name, "信号")}文件：{self.home_file}，数据形状：{data.shape}；保留原始幅值。'
            )
        except Exception as e:
            self.statusbar.showMessage(f'打开失败，未替换当前信号：{e}')
            QMessageBox.critical(self, '打开失败', str(e))

    def plotspec(self, data, signalshow, fs=None, wlength=None):
        """Legacy page adapter; analysis and image rendering live outside the window."""
        try:
            if data is None:
                signalshow.clear()
                self.statusbar.showMessage('时频图：请先加载信号。')
                return
            is_home_plot = signalshow is self.label_signalshow_1
            if not is_home_plot and isinstance(data, (tuple, list)):
                data = data[max(0, self.comboBox.currentIndex())]
            if fs is None:
                fs = (self.lineEdit_Fs_1 if is_home_plot else self.lineEdit_Fs_2).text()
            if wlength is None:
                wlength = (self.lineEdit_windowlength_1 if is_home_plot
                           else self.lineEdit_windowlength_3).text()
            # Parse widget text here; numerical API retains strict integer validation.
            if isinstance(wlength, str):
                wlength = int(wlength.strip())
            data = np.asarray(data)
            if data.ndim == 2 and data.shape[0] == 1:
                data = data[0]
            view = compute_spectrogram(data, sample_rate_hz=fs, nperseg=wlength)
            render_spectrogram(signalshow, view)
            self.statusbar.showMessage('时频图已更新：Hann 窗、50% 重叠；颜色仅比较本记录内强弱，非校准功率。')
        except Exception as exc:
            signalshow.clear()
            self.statusbar.showMessage(
                f'时频图无法显示：采样率须为正数，窗长须为不超过信号长度的正整数，数据须有效。详情：{exc}'
            )

    def plotsig(self, data, signalshow):
        try:
            if data is None:
                signalshow.clear()
                self.statusbar.showMessage('时域波形：请先加载信号。')
                return
            is_home_plot = signalshow is self.label_signalshow_1
            self.currentIndex = 0 if is_home_plot else self.comboBox.currentIndex()
            if not is_home_plot and isinstance(data, (tuple, list)):
                data = data[0 if self.currentIndex < 0 else self.currentIndex]
            data = np.asarray(data)
            if data.ndim == 2 and data.shape[0] == 1:
                data = data[0]
            render_waveform(signalshow, data)
            self.statusbar.showMessage('时域波形已更新；横轴为采样点，幅值未经归一化。')
        except Exception as exc:
            signalshow.clear()
            # Keep this legacy slot non-fatal, but expose a useful reason in
            # the status bar rather than silently hiding a bad input.
            self.statusbar.showMessage(f'时域波形暂时无法显示：{exc}')
            # QMessageBox.about(self, '错误！', '请打开文件和输入采样率！')
            # QMessageBox.about(self, '错误！', traceback.print_exc())

    def plotspectrum(self, data, signalshow):
        try:
            if data is None:
                signalshow.clear()
                self.statusbar.showMessage('频谱图：请先加载信号。')
                return
            is_home = signalshow is self.label_signalshow_1
            if not is_home and isinstance(data, (tuple, list)):
                data = data[max(0, self.comboBox.currentIndex())]
            rate_edit = self.lineEdit_Fs_1 if is_home else self.lineEdit_Fs_2
            try:
                sample_rate = float(rate_edit.text().strip())
            except ValueError:
                raise ValueError('请在当前页面填写有效的采样率（Hz）。') from None
            if not np.isfinite(sample_rate) or sample_rate <= 0:
                raise ValueError('采样率必须是大于零的有限数值（Hz）。')
            data = np.asarray(data)
            # The legacy real-signal loader returns a single channel row.
            if data.ndim == 2 and data.shape[0] == 1:
                data = data[0]
            render_spectrum(signalshow, data, sample_rate_hz=sample_rate)
            self.statusbar.showMessage('频谱图已更新；幅度为 |FFT|/N，非校准功率。')
        except Exception as exc:
            signalshow.clear()
            self.statusbar.showMessage(
                f'频谱图无法显示：请检查信号形状、空数据或非有限值。详情：{exc}'
            )

    def xzt(self, data, signalshow):
        """Render explicit input only; never substitute an unrelated demo file."""
        try:
            view = compute_constellation(data)  # Real data requires explicit compatibility elsewhere.
            render_constellation(signalshow, view)
            self.statusbar.showMessage(f'I/Q 星座图：显示 {view.point_count} 个样本点，未做符号同步。')
        except Exception as exc:
            signalshow.clear()
            self.statusbar.showMessage(f'星座图无法显示：需要有效 I/Q 数据。详情：{exc}')

    def plotspec_featuremap(self, data, signalshow, fs, wlength):
        """Compatibility signature: fs is intentionally unused for model features."""
        self._render_model_feature(data, signalshow, window=wlength)

    def _render_model_feature(self, data, signalshow, window=None):
        if self.comboBox_choose_feature.currentText() == 'tsne':
            return
        try:
            if self.general_recognition_result is not None:
                file_count = len(self.general_recognition_result.filenames)
            elif self.general_input is not None:
                file_count = self.general_input.file_count
            else:
                file_count = len(self.filename)
            view = select_feature_vector(data, file_index=self.comboBox.currentIndex(),
                file_count=file_count, row_counts=self.feature_row_counts)
            if window is None:
                render_feature_vector(signalshow, view)
            else:
                spectrum = compute_feature_spectrogram(view, window)
                render_feature_spectrogram(signalshow, spectrum, view.rows_in_file)
        except Exception as exc:
            signalshow.clear()
            # Empty features before inference are normal, not a raw-signal plotting failure.
            signalshow.plotItem.setTitle(f'特征暂不可用：{exc}')

    def plotsig_featuremap(self, data, signalshow):
        self._render_model_feature(data, signalshow)

    def plot_gene_sig(self, data, signalshow):
        # Selection uses session batches/labels, not a guessed batch length or disk file.
        try:
            self._display_generated_sample(signalshow)
        except Exception as exc:
            signalshow.clear()
            self.statusbar.showMessage(f'生成信号无法显示：{exc}')

    def _display_generated_sample(self, signalshow):
        class_name = self.comboBox_showclass.currentText()
        try:
            sample_number = int(self.lineEdit_switchsignal.text() or '1')
            class_name, sample = select_generated_sample(
                self.signals, getattr(self, 'gen_labels', []), class_name, sample_number)
            if self.radioButton_2.isChecked():
                view = compute_constellation(sample, analytic_from_real=True)
                render_constellation(signalshow, view)
                detail = ('Hilbert 解析信号兼容显示，非真实双路采集'
                          if view.source_kind == 'analytic_real' else 'I/Q 样本散点，未做符号同步')
            else:
                real_source = not np.iscomplexobj(sample)
                render_waveform(signalshow, hilbert(sample) if real_source else sample)
                detail = '解析信号波形（Q 由 Hilbert 推导）' if real_source else 'I/Q 波形'
                signalshow.plotItem.setTitle(detail)
            if (
                self.generation_result is not None
                and class_name in self.generation_result.labels
            ):
                score = self.generation_result.score_for(class_name, sample_number)
                self.lineEdit_score.setText(f'{score:.4f}')
            self.statusbar.showMessage(f'{class_name} 第 {sample_number} 个生成样本：{detail}')
        except Exception:
            signalshow.clear()
            raise
        self.textEdit_log_2.append(f'显示 {class_name} 第 {sample_number} 个生成信号。')

    def switch_gene_sig(self, data, signalshow):
        try:
            self._display_generated_sample(signalshow)
        except Exception as e:
            QMessageBox.warning(self, '无法切换类别', str(e))

    def switch_single_sig(self, data, signalshow):
        try:
            self._display_generated_sample(signalshow)
        except Exception as e:
            QMessageBox.warning(self, '无法切换样本', str(e))


    def _feature_selection_bounds(self, data):
        array = np.asarray(data)
        data_length = (
            max(array.shape)
            if array.ndim == 2 and 1 in array.shape
            else len(array)
        )
        if data_length == 0:
            return 0, 0
        if self.radioButton_waterfall.isChecked():
            if self.waterfall_plotter.is_running:
                self.waterfall_plotter.pause()
            return 0, data_length
        if not self.radioButton_time_1.isChecked():
            return 0, data_length
        start, end = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0]
        start_index = max(0, int(np.floor(start)))
        end_index = min(data_length, int(np.floor(end)) + 1)
        if end_index <= start_index:
            return 0, data_length
        return start_index, end_index

    def features(self, feature_name=None):
        names = {
            '小波特征': 'wavelet',
            '双谱特征': 'bispectrum',
            'J、R特征': 'jr',
            'HHT特征': 'hht',
        }
        name = feature_name or '小波特征'
        kind = names.get(name)
        if kind is None:
            self.statusbar.showMessage(f'未登记的特征入口：{name}')
            return
        if self.home_feature_task.busy:
            self.statusbar.showMessage(
                '上一项主页专家分析仍在计算或收尾；可先取消，待收尾后重试。'
            )
            return
        self._start_home_feature(kind)

    def _start_home_feature(self, kind):
        target = (
            self.label_featureshow_1
            if self.num % 2 == 0
            else self.label_featureshow_2
        )
        target.clear()
        target.setToolTip('')
        self.home_feature_target = target
        try:
            if self.home_data is None:
                raise ValueError('请先在主页加载实信号。')
            start, end = self._feature_selection_bounds(self.home_data)
            request = prepare_home_feature_request(
                kind,
                self.home_data,
                self.lineEdit_Fs_1.text(),
                start_sample=start,
                end_sample=end,
            )
            self.home_feature_task.start(request)
            labels = {
                'wavelet': '小波',
                'bispectrum': '双谱',
                'jr': 'J/R',
                'hht': 'HHT',
            }
            target.setText(f'{labels[kind]}后台计算中…')
        except Exception as exc:
            self._on_home_feature_error(str(exc))

    def _cancel_home_feature(self, message='主页专家分析已取消；后台计算可能仍在收尾。'):
        self.home_feature_task.cancel()
        self.home_feature_progress.hide()
        if self.home_feature_target is not None:
            if message is None:
                self.home_feature_target.clear()
            else:
                self.home_feature_target.setText(message)
            self.home_feature_target = None

    def _on_home_feature_busy(self, busy):
        if busy:
            self.home_feature_progress.setLabelText(
                '正在准备主页专家分析（无百分比进度）'
            )
            self.home_feature_progress.show()
        else:
            self.home_feature_progress.hide()

    def _on_home_feature_error(self, message):
        if self.home_feature_target is not None:
            self.home_feature_target.setText(f'主页专家分析未完成：{message}')
            self.home_feature_target = None
        self.statusbar.showMessage(f'主页专家分析未完成：{message}')

    def _on_home_feature_result(self, result):
        if self.home_feature_target is None:
            return
        try:
            presentation = render_home_feature(result)
            self.home_feature_target.setPixmap(presentation.pixmap)
            self.home_feature_target.setScaledContents(True)
            self.home_feature_target.setToolTip(presentation.tooltip)
            self.num += 1
            self.home_feature_target = None
            self.statusbar.showMessage(presentation.status)
        except Exception as exc:
            self._on_home_feature_error(str(exc))
    def select_save_path(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            '选择生成会话的保存根目录',
            str(self.runtime_directories['generation']),
        )
        if not selected:
            return
        self.save_path = str(Path(selected).resolve())
        self.lineEdit_savepath.clear()
        self.lineEdit_savepath.setText(self.save_path)
        self.textEdit_log_2.append('生成会话保存根目录：' + self.save_path)
        self.pushButton_generate.setEnabled(
            self.generator_model_available and not self.generation_task.busy
        )

    def generate_data(self):
        try:
            if self.generation_task.busy:
                raise RuntimeError('上一项生成任务仍在运行或收尾。')
            output_root = self.lineEdit_savepath.text().strip()
            if not output_root:
                raise ValueError('请先选择生成会话的保存根目录。')
            request = GenerationRequest(
                output_root=Path(output_root),
                class_name=self.comboBox_class.currentText(),
                samples_per_class=int(self.lineEdit_samplenum.text()),
            )
            # A new accepted request supersedes only the displayed session.
            # Previously committed session directories remain untouched.
            self.generation_result = None
            self.signals = []
            self.gen_labels = []
            self.gen_signals = None
            self.label_genesig.clear()
            self.comboBox_showclass.blockSignals(True)
            self.comboBox_showclass.clear()
            self.comboBox_showclass.blockSignals(False)
            self.pushButton_switchsignal.setEnabled(False)
            self.lineEdit_as.setText('—')
            self.lineEdit_score.setText('—')
            self.generation_progress.reset()
            self.generation_progress.setRange(0, request.total_samples)
            self.generation_progress.setValue(0)
            self.generation_progress.setLabelText('正在准备历史信号生成…')
            self.textEdit_log_2.append(
                f'[{datetime.datetime.now()}] 后台生成开始：{request.class_name}；'
                f'每类 {request.samples_per_class} 个，共 {request.total_samples} 个。'
            )
            self.generation_task.start(request)
        except Exception as exc:
            QMessageBox.critical(self, '无法开始生成', str(exc))

    def cancel_generation(self):
        if self.generation_task.cancel():
            self.textEdit_log_2.append(
                f'[{datetime.datetime.now()}] 已请求取消生成；未提交的临时会话将清理。'
            )

    def _on_generation_phase(self, message):
        self.generation_progress.setLabelText(str(message))
        self.statusbar.showMessage(str(message))

    def _on_generation_progress(self, current, total):
        self.generation_progress.setRange(0, max(int(total), 1))
        self.generation_progress.setValue(int(current))

    def _on_generation_busy(self, busy):
        self.pushButton_savepath.setEnabled(not busy)
        self.comboBox_class.setEnabled(not busy)
        self.lineEdit_samplenum.setEnabled(not busy)
        self.pushButton_generate.setEnabled(
            not busy and self.generator_model_available and bool(self.lineEdit_savepath.text())
        )
        if busy:
            self.generation_progress.show()
        else:
            self.generation_progress.hide()

    def _on_generation_cancelled(self):
        self.textEdit_log_2.append(
            f'[{datetime.datetime.now()}] 生成已取消；未保留部分输出。'
        )
        self.statusbar.showMessage('历史信号生成已取消，未保留部分输出。')

    def _on_generation_failed(self, message):
        self.textEdit_log_2.append(
            f'[{datetime.datetime.now()}] 生成失败：{message}；未提交部分输出。'
        )
        self.statusbar.showMessage(f'历史信号生成失败：{message}')
        QMessageBox.critical(self, '历史信号生成失败', str(message))

    def _on_generation_completed(self, result):
        self.generation_result = result
        self.signals = list(result.batches)
        self.gen_labels = list(result.labels)
        self.gen_signals = result.batches[-1]
        self.samplenum = result.classes[0].samples.shape[0]
        self.comboBox_showclass.blockSignals(True)
        self.comboBox_showclass.clear()
        if len(result.labels) > 1:
            self.comboBox_showclass.addItem(GENERATION_ALL_CLASSES)
        self.comboBox_showclass.addItems(result.labels)
        self.comboBox_showclass.setCurrentText(
            GENERATION_ALL_CLASSES if len(result.labels) > 1 else result.labels[0]
        )
        self.comboBox_showclass.blockSignals(False)
        self.lineEdit_switchsignal.setText('1')
        self.lineEdit_as.setText(f'{result.mean_deviation:.4f}')
        self.pushButton_switchsignal.setEnabled(True)
        self._display_generated_sample(self.label_genesig)
        for item in result.classes:
            self.textEdit_log_2.append(
                f'{item.label}：{item.samples.shape[0]} 个样本，形状 '
                f'{list(item.samples.shape)}，耗时 {item.elapsed_seconds:.3f}s；'
                f'文件 {item.output_file}。'
            )
        self.textEdit_log_2.append(
            f'[{datetime.datetime.now()}] 原子生成会话已完成：{result.total_samples} 个样本；'
            f'总耗时 {result.elapsed_seconds:.3f}s；模型 {result.checkpoint_name}；'
            f'会话目录 {result.session_directory}。{result.limitation}'
        )
        self.statusbar.showMessage(
            f'生成完成：{result.total_samples} 个样本；会话目录 {result.session_directory}'
        )

    def load_data(self):
        """Select and decode historical-recognition files through one Data Interface."""

        dialog = None
        try:
            task_name = self.comboBox_task.currentText()
            spec = task_spec(task_name)
            dialog = QFileDialog(self)
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)
            demo_directory = resource_path('data', spec.demo_subdir)
            if demo_directory.is_dir():
                dialog.setDirectory(str(demo_directory))
            if self.comboBox_mod.currentText() == '单脉冲识别':
                dialog.setFileMode(QFileDialog.ExistingFiles)
                dialog.setNameFilter('原始信号 (*.dat *.bin *.raw);;所有文件 (*)')
            else:
                dialog.setFileMode(QFileDialog.DirectoryOnly)
            list_view = dialog.findChild(QListView, 'listView')
            if list_view:
                list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
            tree_view = dialog.findChild(QTreeView, 'treeView')
            if tree_view:
                tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
            if not dialog.exec_():
                return
            selected = dialog.selectedFiles()
            paths = (
                selected
                if self.comboBox_mod.currentText() == '单脉冲识别'
                else discover_recognition_files(selected)
            )
            loaded = load_recognition_files(task_name, paths)
        except Exception as exc:
            QMessageBox.critical(self, '加载识别数据失败', str(exc))
            return
        finally:
            # A parent-owned non-native selector retains QFileSystemModel's
            # watcher workers even after exec_() returns, including cancellation.
            if dialog is not None:
                dialog.deleteLater()

        if self.general_recognition_task.busy:
            self.general_recognition_task.cancel(invalidate=True)
        self.general_input = loaded
        self.general_recognition_result = None
        self._invalidate_tsne()
        self.feature = []
        self.feature_map = []
        self.feature_row_counts = None
        self.label_featuremap.clear()
        # Keep legacy attributes populated only as compatibility mirrors.  The
        # maintained recognition path reads ``general_input`` exclusively.
        self.filename = list(loaded.filenames)
        self.data = loaded.arrays[0] if loaded.file_count == 1 else list(loaded.arrays)
        self.label = (
            loaded.inferred_directory_labels[0]
            if loaded.file_count == 1
            else list(loaded.inferred_directory_labels)
        )
        self.data_file = loaded.filenames[0] if loaded.file_count == 1 else list(loaded.filenames)
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        self.comboBox.addItems(loaded.display_names)
        self.comboBox.setCurrentIndex(0)
        self.comboBox.blockSignals(False)
        self.lineEdit_path.setText('；'.join(loaded.display_names))
        self.lineEdit_path.setToolTip('\n'.join(loaded.filenames))
        self.recognition_workbench.general_result_card.setText(
            '识别结果：—\n状态：待开始\n'
            f'文件：{loaded.file_count} 个\n模型：待加载\n窗口：待计算\n耗时：—'
        )
        self.recognition_workbench.model_hint.setText(
            f'{task_name}：{spec.input_description}，{spec.channels} 通道，'
            f'窗长 {spec.window_size}；模型将在后台按任务自动加载。'
        )
        self.pushButton_recognition.setEnabled(True)
        self.change_original_signal()
        self.textEdit_log_3.append(
            f'[{datetime.datetime.now()}] 已按 {spec.input_description} 解码 '
            f'{loaded.file_count} 个文件：{"；".join(loaded.display_names)}'
        )
        self.statusbar.showMessage(
            f'识别数据已加载  |  {task_name}  |  文件数：{loaded.file_count}'
        )



    def qianhou(self,):
        QMessageBox.information(
            self,
            '需要验证集',
            '该对比需要同一验证集上的扩充前、扩充后两个真实模型。当前项目未提供这两组结果，已取消原先的演示随机等待和硬编码精度。'
        )


    def load_uav_input(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择无人机信号文件', '', 'NumPy 信号 (*.npy)')
        if not path:
            return
        self.recognition_workbench.uav_plots.select_file(path)
        self.recognition_workbench.uav_result_label.setText(
            '识别结果：待运行\n类别：—\n已知 / 未知：未评估\n置信度：—'
        )
        self.recognition_workbench.uav_progress.setRange(0, 1)
        self.recognition_workbench.uav_progress.setValue(0)
        self.recognition_workbench.uav_progress.setFormat('未开始')
        self.recognition_workbench.uav_start.setEnabled(False)

    def start_uav_recognition(self):
        if self.uav_task.busy:
            return
        plots = self.recognition_workbench.uav_plots
        prepared = plots.prepared_input
        if prepared is None or not plots.is_recognition_ready:
            QMessageBox.information(
                self,
                '输入尚未就绪',
                '请等待完整记录预览与质量检查完成，并填写真实采样率。',
            )
            return
        self.recognition_workbench.uav_result_label.setText(
            '识别结果：后台处理中\n类别：—\n已知 / 未知：未评估\n置信度：—'
        )
        self.recognition_workbench.uav_progress.setRange(0, 0)
        self.recognition_workbench.uav_progress.setFormat('准备中…')
        try:
            self.uav_task.start(prepared)
        except Exception as exc:
            self._on_uav_recognition_failed(str(exc))

    def cancel_uav_recognition(self):
        if self.uav_task.cancel():
            self.recognition_workbench.uav_progress.setFormat('正在取消…')
            self.statusbar.showMessage('正在安全取消无人机识别…')

    def _on_uav_recognition_progress(self, current, total):
        bar = self.recognition_workbench.uav_progress
        bar.setRange(0, max(int(total), 1))
        bar.setValue(min(int(current), max(int(total), 1)))
        bar.setFormat(f'{int(current)} / {int(total)} 个窗口')

    def _on_uav_recognition_phase(self, message):
        self.statusbar.showMessage(str(message))

    def _on_uav_input_readiness_changed(self, ready):
        if hasattr(self, 'uav_task') and self.uav_task.busy:
            return
        self.recognition_workbench.uav_start.setEnabled(bool(ready))

    def _on_uav_recognition_busy(self, busy):
        workbench = self.recognition_workbench
        workbench.uav_choose.setEnabled(not busy)
        workbench.uav_start.setEnabled(
            not busy and workbench.uav_plots.is_recognition_ready
        )
        workbench.uav_cancel.setEnabled(busy)

    def _on_uav_recognition_completed(self, result):
        workbench = self.recognition_workbench
        if not result.can_publish_prediction:
            workbench.uav_result_label.setText(
                '识别结果：拒绝判断\n类别：—\n已知 / 未知：未评估\n'
                f'置信度：—\n模型：{result.model_version or "—"}'
            )
            workbench._set_uav_quality(
                result.quality_status.value, result.quality_message
            )
            workbench.uav_progress.setRange(0, 1)
            workbench.uav_progress.setValue(0)
            workbench.uav_progress.setFormat('未运行')
            self.statusbar.showMessage(
                f'无人机识别未运行：{result.quality_message}'
            )
            return
        known_text = (
            '未评估（阈值尚未冻结）'
            if result.open_set_status == 'not_evaluated'
            else '未发布'
        )
        confidence = '—' if result.confidence is None else f'{result.confidence:.2%}'
        workbench.uav_result_label.setText(
            f'识别结果：已完成（开发版）\n类别：{result.label or "—"}\n'
            f'已知 / 未知：{known_text}\n置信度：{confidence}\n'
            f'模型：{result.model_version}\n契约：{result.contract_id or "—"}'
        )
        workbench._set_uav_quality(
            result.quality_status.value, result.quality_message
        )
        if result.window_count is None:
            workbench.uav_progress.setRange(0, 1)
            workbench.uav_progress.setValue(1)
            workbench.uav_progress.setFormat('已完成')
        else:
            workbench.uav_progress.setRange(0, result.window_count)
            workbench.uav_progress.setValue(result.window_count)
            workbench.uav_progress.setFormat(
                f'{result.window_count} / {result.window_count} 个窗口'
            )
        self.statusbar.showMessage(
            f'无人机开发版已知源识别完成：{result.label}；未知拒识未评估。'
        )

    def _on_uav_recognition_failed(self, message):
        self.recognition_workbench.uav_result_label.setText(
            f'识别结果：失败\n类别：—\n已知 / 未知：未评估\n置信度：—\n原因：{message}'
        )
        self.recognition_workbench.uav_progress.setRange(0, 1)
        self.recognition_workbench.uav_progress.setValue(0)
        self.recognition_workbench.uav_progress.setFormat('失败')
        self.statusbar.showMessage(f'无人机识别失败：{message}')

    def _on_uav_recognition_cancelled(self):
        self.recognition_workbench.uav_result_label.setText(
            '识别结果：已取消\n类别：—\n已知 / 未知：未评估\n置信度：—'
        )
        self.recognition_workbench.uav_progress.setRange(0, 1)
        self.recognition_workbench.uav_progress.setValue(0)
        self.recognition_workbench.uav_progress.setFormat('已取消')
        self.statusbar.showMessage('无人机识别已取消，未发布部分结果。')

    def _make_general_snapshot(self):
        if self.general_input is None:
            raise ValueError('请先选择通用信号数据文件。')
        task = self.comboBox_task.currentText()
        if self.general_input.task_name != task:
            raise ValueError('当前文件属于其他识别任务，请重新选择数据。')
        repeat = int(self.lineEdit_windowlength_13.text() or 1)
        return self.general_input.snapshot(repeat)

    def start_general_recognition(self):
        if self.general_recognition_task.busy:
            return
        try:
            snapshot = self._make_general_snapshot()
            self.general_recognition_result = None
            self._invalidate_tsne()
            self.feature = []
            self.feature_map = []
            self.feature_row_counts = None
            self.label_featuremap.clear()
            self.recognition_workbench.general_result_card.setText('识别结果：处理中…\n状态：后台运行\n文件：%s\n模型：加载中…\n窗口：处理中…\n耗时：—' % '；'.join(Path(x).name for x in snapshot.filenames))
            self.textEdit_log_3.append(f'[{datetime.datetime.now()}] 后台识别开始：{snapshot.task_name}；文件 {Path(snapshot.filenames[0]).name}')
            self.recognition_workbench.model_hint.setText('后台识别中…界面仍可操作；结果只对应本次输入快照。')
            self.recognition_workbench.general_progress.setRange(0, 1); self.recognition_workbench.general_progress.setValue(0); self.recognition_workbench.general_progress.setFormat('准备中…')
            self.recognition_workbench.general_result_card.setText('识别结果：处理中…\n状态：后台运行\n文件：%s\n模型：加载中…\n窗口：处理中…\n耗时：—' % '；'.join(Path(x).name for x in snapshot.filenames))
            self.pushButton_recognition.setEnabled(False)
            self.general_recognition_task.start(snapshot)
        except Exception as exc:
            QMessageBox.warning(self, '无法开始识别', str(exc))

    def cancel_general_recognition(self):
        if self.general_recognition_task.cancel():
            self.textEdit_log_3.append(f'[{datetime.datetime.now()}] 已请求取消通用信号识别。')

    def _on_general_recognition_progress(self, current, total):
        bar = self.recognition_workbench.general_progress
        bar.setRange(0, max(int(total), 1)); bar.setValue(int(current)); bar.setFormat(f'{int(current)} / {int(total)} 个窗口')

    def _on_general_recognition_cancelled(self):
        self.general_recognition_result = None
        self.recognition_workbench.general_progress.setRange(0, 1)
        self.recognition_workbench.general_progress.setValue(0)
        self.recognition_workbench.general_progress.setFormat('已取消')
        self.recognition_workbench.general_result_card.setText('识别结果：—\n状态：已取消\n文件：本次输入\n模型：未发布结果\n窗口：未完成\n耗时：—')
        self.textEdit_log_3.append(f'[{datetime.datetime.now()}] 通用信号识别已取消，未保留部分结果。')
        self.statusbar.showMessage('通用信号识别已取消')

    def _on_general_recognition_phase(self, message):
        self.recognition_workbench.model_hint.setText(str(message))
        self.statusbar.showMessage(str(message))

    def _on_general_recognition_busy(self, busy):
        self.recognition_workbench.general_cancel.setEnabled(busy)
        self.pushButton_recognition.setEnabled(not busy and self.general_input is not None)
        if not busy:
            self.recognition_workbench.model_hint.setText('模型由所选任务自动匹配；最近一次后台结果只对应其输入快照。')

    @staticmethod
    def _format_general_result_card(result):
        # Keep one compact row per input file so multi-file runs remain readable.
        rows = []
        repeat_labels = getattr(result, 'repeat_labels', ())
        for index, (filename, label, windows) in enumerate(zip(
                result.filenames, result.predicted_labels, result.feature_row_counts)):
            rounds = ''
            if repeat_labels:
                per_file = [round_labels[index] for round_labels in repeat_labels
                            if index < len(round_labels)]
                if per_file:
                    rounds = '；各轮：' + ' / '.join(per_file)
            rows.append(f'{index + 1}. {Path(filename).name} → {label}（{windows} 窗口{rounds}）')
        return ('识别结果：\n' + '\n'.join(rows) +
                f'\n状态：已完成\n模型：{result.checkpoint_name}'
                f'\n运行次数：{result.repeat_count}\n总耗时：{result.elapsed_seconds:.4f} 秒')

    def _on_general_recognition_completed(self, result):
        self.general_recognition_result = result
        self.feature = result.features
        self.feature_map = result.features
        self.feature_row_counts = list(result.feature_row_counts)
        self.label_featuremap.clear()
        self.recognition_workbench.general_result_card.setText(self._format_general_result_card(result))
        self.textEdit_log_3.append(
            f'[{datetime.datetime.now()}] 后台识别完成：{len(result.filenames)} 个文件（{"；".join(Path(x).name for x in result.filenames)}）→ {"；".join(result.predicted_labels)}；'
            f'窗口 {sum(result.feature_row_counts)}；运行 {result.repeat_count} 次；耗时 {result.elapsed_seconds:.4f}s；模型 {result.checkpoint_name}'
        )
        self.statusbar.showMessage(f'后台识别完成：{result.predicted_labels[0]}')

    def _on_general_recognition_failed(self, message):
        self.general_recognition_result = None
        self.recognition_workbench.general_progress.setRange(0,1); self.recognition_workbench.general_progress.setValue(0); self.recognition_workbench.general_progress.setFormat('失败')
        self.recognition_workbench.general_result_card.setText('识别结果：未完成\n状态：失败\n原因：%s' % message)
        self.feature = []
        self.feature_map = []
        self.feature_row_counts = None
        self.label_featuremap.clear()
        self.textEdit_log_3.append(f'[{datetime.datetime.now()}] 通用信号识别失败：{message}')
        self.statusbar.showMessage(f'通用识别失败：{message}')
        QMessageBox.critical(self, '通用信号识别失败', str(message))



    def change_original_signal(self):
        try:
            self.currentIndex = max(0, self.comboBox.currentIndex())
            source = (
                self.general_input.selected(self.currentIndex)
                if self.general_input is not None
                else self.data
            )
            if self.radioButton_TF_3.isChecked():
                self.plotspec(source, self.label_signal)
            elif self.radioButton_time_3.isChecked():
                self.plotsig(source, self.label_signal)
            elif self.radioButton_PP_3.isChecked():
                self.plotspectrum(source, self.label_signal)
        except Exception as exc:
            self.label_signal.clear()
            self.statusbar.showMessage(f'通用识别输入无法显示：{exc}')

    def change_feature_signal(self):
        try:
            if self.comboBox_choose_feature.currentText() == '中间层特征':
                self.currentIndex = self.comboBox.currentIndex()
                if self.radioButton_TF_3.isChecked():
                    self.plotspec_featuremap(self.feature, self.label_featuremap, None,
                                             16)
                elif self.radioButton_time_3.isChecked():
                    self.plotsig_featuremap(self.feature, self.label_featuremap)

            else:
                pass
        except Exception as e:
            QMessageBox.about(self, 'change_feature_signal', str(e))
            pass

    def choose_feature_signal(self):
        self._invalidate_tsne()
        choice = self.comboBox_choose_feature.currentText()
        if choice == '中间层特征':
            self.label_featuremap.clear()
            if self.radioButton_TF_3.isChecked():
                self.plotspec_featuremap(
                    self.feature_map, self.label_featuremap, None, 16
                )
            elif self.radioButton_time_3.isChecked():
                self.plotsig_featuremap(self.feature_map, self.label_featuremap)
            return
        if choice != 'tsne':
            self.label_featuremap.clear()
            self.label_featuremap.setTitle(f'未登记的特征视图：{choice}')
            self.statusbar.showMessage(f'未登记的特征视图：{choice}')
            return
        self.label_featuremap.clear()
        try:
            filenames = (
                self.general_recognition_result.filenames
                if self.general_recognition_result is not None
                else self.general_input.filenames if self.general_input is not None else self.filename
            )
            source = prepare_tsne(self.feature_map, filenames, self.feature_row_counts)
            self.tsne_task.start(source)
            self.label_featuremap.setTitle('t-SNE 后台计算中…')
        except Exception as exc:
            self._on_tsne_error(str(exc))

    def _invalidate_tsne(self):
        self.tsne_task.cancel()
        self.tsne_progress.hide()

    def _cancel_tsne(self):
        self._invalidate_tsne()
        self.label_featuremap.clear()
        self.label_featuremap.setTitle('t-SNE 已取消显示；后台收尾后可切换选项重试')
        self.statusbar.showMessage('已取消 t-SNE 显示；后台计算可能仍在安全收尾。')

    def _on_tsne_busy(self, busy):
        if busy:
            self.tsne_progress.show()
        else:
            self.tsne_progress.hide()

    def _on_tsne_result(self, view):
        if self.comboBox_choose_feature.currentText() == 'tsne':
            render_tsne(self.label_featuremap, view)
            self.statusbar.showMessage(
                f't-SNE 完成：{len(view.coordinates)} 个窗口；颜色按来源文件，种子 {view.seed}，perplexity={view.perplexity:g}；不代表准确率。')

    def _on_tsne_error(self, message):
        self.label_featuremap.clear()
        self.label_featuremap.setTitle('t-SNE 未完成，请查看状态栏')
        self.statusbar.showMessage(f't-SNE 无法显示：{message}')

    def log_task(self):
        self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务
        self.textEdit_log_3.append('['+str(datetime.datetime.now()) + ']' + f'选择 {self.task_name} 任务')



def main():
    """Start the desktop application."""

    os.chdir(Path(__file__).resolve().parent)
    app = QApplication(sys.argv)
    icon_path = resource_path('assets', 'airwatch.ico')
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    ensure_user_directories()
    app.setStyle('Fusion')
    app.setStyleSheet(APP_STYLESHEET)
    myWin = MyWindow()
    myWin.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())

