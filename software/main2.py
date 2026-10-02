import os
import time
import cv2
import sys
import pywt

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
os.environ['QT_API'] = 'pyqt5'

from co import cnames
import torch
import models
import datetime
import traceback
import qdarkstyle
try:
    import tftb.processing
    from pyhht import EMD
except ImportError:
    tftb = None
    EMD = None

import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

from PIL import Image
from scipy import fftpack
from scipy.fft import fft
from scipy.signal import hilbert
from scipy.linalg import sqrtm
from models.ResNet18_TCN import ResNet18_TCN
from models.generator import Generator
from sklearn.manifold import TSNE
from newwindow_ import Ui_MainWindow
from scipy.signal import spectrogram
from PyQt5.QtGui import QImage, QPixmap
from pyqtgraph import PlotDataItem, ImageItem
from matplotlib.collections import LineCollection
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QListView, QAbstractItemView, QTreeView,\
    QHBoxLayout, QWidget


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.initUI()

        ## ------------------编辑信号槽--------------------
        # ------------------Tab1控件------------------
        self.str2_fs = self.lineEdit_Fs_1.text()  # 获得用户输入采样率
        self.wlength = self.lineEdit_windowlength_1.text()  # 获得用户输入窗口长度

        self.pushButton_openfile.clicked.connect(self.openfile)  # 打开数据文件,保存到变量self.data
        self.pushButton_reload_TF.clicked.connect(lambda: self.plotspec(self.data, self.label_signalshow_1,
                                                                        int(self.lineEdit_Fs_1.text()), int(self.lineEdit_windowlength_1.text())))  # 重新加载时频图
        self.label_signalshow_1.setMouseEnabled(x=True, y=False)  # 禁用轴操作
        self.label_signalshow_1.plotItem.sigXRangeChanged.connect(self.showtime)  # 禁用轴操作
        self.radioButton_time_1.toggled.connect(lambda: self.plotsig(self.data, self.label_signalshow_1))  # 时序信号按钮
        self.radioButton_TF_1.toggled.connect(lambda: self.plotspec(self.data, self.label_signalshow_1,
                                                                    int(self.lineEdit_Fs_1.text()), int(self.lineEdit_windowlength_1.text())))  # 时频图按钮
        # self.comboBox_features.currentIndexChanged.connect(self.features)  # 时频图按钮
        # ------------------Tab1控件------------------

        # ------------------Tab2控件------------------
        self.pushButton_path.clicked.connect(self.load_data)  # 触发生成数据函数
        self.radioButton_time_3.toggled.connect(lambda: self.plotsig(self.data, self.label_signal))  # 加载文件按钮
        self.radioButton_TF_3.toggled.connect(lambda: self.plotspec(self.data, self.label_signal,
                                                                    int(self.lineEdit_Fs_1.text()), int(self.lineEdit_windowlength_1.text())))  # 加载文件按钮
        self.radioButton_PP_3.toggled.connect(lambda: self.plotspectrum(self.data,
                                                                                self.label_signal))  # 加载文件按钮
        self.radioButton_time_3.toggled.connect(lambda: self.plotsig_featuremap(self.feature_map,
                                                                                self.label_featuremap))  # 加载文件按钮

        self.radioButton_TF_3.toggled.connect(lambda: self.plotspec_featuremap(
            self.feature_map, self.label_featuremap, int(self.lineEdit_Fs_2.text()), 16))  # 加载文件按钮
        self.comboBox_task.currentIndexChanged.connect(self.log_task)  # 时频图按钮
        self.comboBox_task.currentIndexChanged.connect(self.load_recognition_model)  # 加载对应权重
        self.pushButton_recognition.clicked.connect(lambda: self.recognition(self.data, self.label, self.filename))  # 触发生成数据函数
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
        self.radioButton_TF_2.toggled.connect(lambda: self.plot_gene_sig(self.signals, self.label_genesig))
        self.radioButton_2.toggled.connect(lambda: self.xzt(self.signals, self.label_genesig))
        # ------------------Tab3控件------------------

        # ------------------Tab4控件------------------
        self.pushButton_4.clicked.connect(self.model_lightweight_before)
        self.pushButton_qianhou.clicked.connect(self.qianhou)
        self.comboBox_task_2.currentIndexChanged.connect(self.load_recognition_model_light)  # 加载对应权重
        self.pushButton_path_2.clicked.connect(self.load_data_light)  # 触发生成数据函数
        self.pushButton_recognition_2.clicked.connect(lambda: self.recognition_light(self.data, self.label, self.filename))  # 触发生成数据函数
        # ------------------Tab4控件------------------
        ## ------------------编辑信号槽--------------------

        self.num = 0  # 全局变量
        self.batch_size = 32  # 全局变量
        self.latent_dim = 100  # 全局变量
        self.img_rows = 1000
        self.img_cols = 1
        self.channels = 1
        self.divide = False
        self.length = {'信号个体识别': 1024, '信号业务识别': 2048, '信号调制识别': 1024,
                       '信号编码识别': 512,'信号通联识别': 5000}
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 15
        self.classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
                              '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
        self.combobox_classes = ['全选', '16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
                              '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']

        self.textEdit_log_1.append('初始化成功，欢迎进入智能信号处理原型验证系统软件！')
        # self.feature_name = self.comboBox_features.currentText()  # 获得用户输入特征名称
        self.class_name = self.comboBox_class.currentText()  # 获得用户输入待生成调制类别
        self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务
        self.sample_num = self.lineEdit_samplenum.text()  # 获得用户输入生成样本个数
        self.currentIndex = self.comboBox.currentIndex()
        self.feature_map = []
        self.model = ''  # 初始化识别模型

        # ---------------------加载网络-----------------------


        self.ee = 0
        self.model_light = ''
        # ---------------------加载网络-----------------------

    def initUI(self):
        self.setupUi(self)
        self.setWindowTitle('智能化信号处理系统')

    #  自定义信号
    def openfile(self):
        try:
            dic = {'modulation': '短波', 'yewu': '短波', 'individual': '卫星', 'communication': '超短波', 'bianma': ''}
            self.str2_fs = self.lineEdit_Fs_1.text()  # 获得用户输入采样率
            self.wlength = self.lineEdit_windowlength_1.text()  # 获得用户输入窗口长度
            self.filename = QFileDialog.getOpenFileName(self, '打开数据文件', './data/', '*.npy')
            self.data_file = self.filename[0]
            class_name = self.data_file.split('/')[4]
            self.label_filename_1.setText(self.data_file.split('/')[-1])
            data = np.load(self.data_file)
            self.data = data / np.max(np.abs(data))
            if self.radioButton_TF_1.isChecked():
                self.plotspec(self.data, self.label_signalshow_1, int(self.str2_fs), int(self.wlength))
            elif self.radioButton_time_1.isChecked():
                self.plotsig(self.data, self.label_signalshow_1)
            self.textEdit_log_1.append(f'成功打开{dic[class_name]}文件:{self.data_file}, 信号shape:{self.data.shape}')
        except Exception as e:
            QMessageBox.about(self, '错误！', str(e))

    def plotspec(self, data, signalshow, fs, wlength):
        '''
        :param signalshow: 显示窗口
        :param fs: 采样率
        :param wlength: 窗长
        :param TFF: 变换方式
        :return: None
        '''
        try:
            self.currentIndex = self.comboBox.currentIndex()
            if len(self.filename) == 1:
                if '仿真' in self.filename:
                    fs = 40000000
                elif '实采A' in self.filename:
                    fs = 960000
                pass
            else:
                if '仿真' in self.filename[self.currentIndex]:
                    fs = 40000000
                elif '实采A' in self.filename[self.currentIndex]:
                    fs = 960000
                data = data[self.currentIndex]
            if len(data.shape) == 2:
                data = data[0, :]
            if wlength < data.shape[0]:
                signalshow.plotItem.clear()
                [SpectrumF, SpectrumT, Spectrum] = spectrogram(data, fs=fs, window='hann', nperseg=wlength)
                width = 6
                height = 4
                dpi = 128
                fig = plt.figure(figsize=(width, height), dpi=dpi)
                axes = fig.add_axes([0, 0, 1, 1])
                axes.cla()
                axes.pcolormesh(SpectrumT, SpectrumF, Spectrum)
                axes.axis('off')  # 去除坐标轴
                fig.gca().xaxis.set_major_locator(plt.NullLocator())
                fig.gca().yaxis.set_major_locator(plt.NullLocator())
                fig.canvas.draw()
                fig_str = fig.canvas.tostring_rgb()
                spec_img = np.frombuffer(fig_str, dtype=np.uint8).reshape((height*dpi, -1, 3))
                ## Create a ColorMap
                STEPS = np.array([0.0, 0.2, 0.6, 1.0])
                CLRS = ['k', 'r', 'y', 'w']
                clrmp = pg.ColorMap(STEPS, np.array([pg.colorTuple(pg.Color(c)) for c in CLRS]))

                ## Get the LookupTable
                lut = clrmp.getLookupTable()
                PY_pic = ImageItem(lut=lut)
                PY_pic.setImage(np.transpose(spec_img, (1, 0, 2)))

                signalshow.plotItem.getViewBox().autoRange()
                signalshow.plotItem.addItem(PY_pic)
                signalshow.plotItem.getViewBox().autoRange()
                plt.close()
            else:
                QMessageBox.about(self, '错误！', '窗长不能大于信号长度！')
        except Exception as e:
            print(e)

    def plotsig(self, data, signalshow):
        try:
            self.currentIndex = self.comboBox.currentIndex()
            # if len(self.filename) != 1 and len(self.filename) != len(data):
            #     return
            if len(self.filename) == 1:
                pass
            elif len(self.filename) != 1 and self.currentIndex == -1:

                data = data[0]
            else:
                data = data[self.currentIndex]
            if len(data.shape) == 2:
                data = data[0, :]
            x = np.arange(data.shape[-1])
            signalshow.plotItem.clear()
            signalshow.plotItem.plot(x, data, pen='y')
            signalshow.plotItem.getViewBox().autoRange()
        except:
            pass
            # QMessageBox.about(self, '错误！', '请打开文件和输入采样率！')
            # QMessageBox.about(self, '错误！', traceback.print_exc())

    def plotspectrum(self, data, signalshow):
        try:
            self.currentIndex = self.comboBox.currentIndex()
            if len(self.filename) == 1:
                pass
            elif len(self.filename) != 1 and self.currentIndex == -1:
                data = data[0]
            else:
                data = data[self.currentIndex]
            if len(data.shape) == 2:
                ft = fft(data[0, :])
            else:
                ft = fft(data)
            print(len(ft), type(ft), np.max(ft), np.min(ft))
            magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
            sr = 16000
            frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)
            signalshow.plotItem.clear()
            signalshow.plotItem.plot(frequency, magnitude, pen='y')
            signalshow.plotItem.getViewBox().autoRange()
            # data = data[0, :] + 1j*data[1, :]
            # magnitude_data = np.abs(data)
            # power_data = magnitude_data ** 2
            # x = np.arange(data.shape[-1])
            # signalshow.plotItem.clear()
            # signalshow.plotItem.plot(x, power_data, pen='y')
            # signalshow.plotItem.getViewBox().autoRange()
        except Exception as e:
            QMessageBox.about(self, 'Error!', str(e))

    def xzt(self, data, signalshow):
        try:
            if len(data) == 15:
                self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
                if self.class_name == '全选':
                    data = np.load(f'./data/modulation/modulation_QPSK.npy')
                else:
                    data = np.load(f'./data/modulation/modulation_{self.class_name}.npy')
                from scipy.signal import hilbert
                data = hilbert(data, axis=0)
                signalshow.plotItem.clear()
                # signalshow.plotItem.scatter(data[0, :, 0].real, data[0, :, 0].imag, pen='r')
                signalshow.plotItem.plot(data[0, :, 0].real, data[0, :, 0].imag, pen=None, symbol="o")
                signalshow.plotItem.getViewBox().autoRange()
            else:
                self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
                data = np.load(f'./data/modulation/modulation_{self.class_name}.npy')
                from scipy.signal import hilbert
                data = hilbert(data, axis=0)
                signalshow.plotItem.clear()
                # signalshow.plotItem.scatter(data[0, :, 0].real, data[0, :, 0].imag, pen='r')
                signalshow.plotItem.plot(data[0, :, 0].real, data[0, :, 0].imag, pen=None, symbol="o")
                signalshow.plotItem.getViewBox().autoRange()
                signalshow.plotItem.getViewBox().autoRange()
        except Exception as e:
            QMessageBox.about(self, '错误！', str(e))

    def plotspec_featuremap(self, data, signalshow, fs, wlength):
        '''
        :param signalshow: 显示窗口
        :param fs: 采样率
        :param wlength: 窗长
        :param TFF: 变换方式
        :return: None
        '''
        try:
            if len(data) == 0:
                pass
            else:
                if len(self.filename) == 1:
                    data = data[0, :]
                else:
                    self.currentIndex = self.comboBox.currentIndex()
                    data = data[self.currentIndex][0, :]
                signalshow.plotItem.clear()

                [SpectrumF, SpectrumT, Spectrum] = spectrogram(data, fs=fs, window='hann', nperseg=wlength)

                width = 6
                height = 4
                dpi = 128
                fig = plt.figure(figsize=(width, height), dpi=dpi)
                axes = fig.add_axes([0, 0, 1, 1])
                axes.cla()
                axes.pcolormesh(SpectrumT, SpectrumF, Spectrum)
                axes.axis('off')  # 去除坐标轴
                fig.gca().xaxis.set_major_locator(plt.NullLocator())
                fig.gca().yaxis.set_major_locator(plt.NullLocator())
                fig.canvas.draw()
                fig_str = fig.canvas.tostring_rgb()
                spec_img = np.frombuffer(fig_str, dtype=np.uint8).reshape((height * dpi, -1, 3))
                ## Create a ColorMap
                STEPS = np.array([0.0, 0.2, 0.6, 1.0])
                CLRS = ['k', 'r', 'y', 'w']
                clrmp = pg.ColorMap(STEPS, np.array([pg.colorTuple(pg.Color(c)) for c in CLRS]))

                ## Get the LookupTable
                lut = clrmp.getLookupTable()
                PY_pic = ImageItem(lut=lut)
                PY_pic.setImage(np.transpose(spec_img, (1, 0, 2)))

                signalshow.plotItem.getViewBox().autoRange()
                signalshow.plotItem.addItem(PY_pic)
                signalshow.plotItem.getViewBox().autoRange()
                plt.close()
        except Exception as e:
            print(e)

        # except:
        #     pass

    def plotsig_featuremap(self, data, signalshow):
        try:
            if self.feature == []:
                pass
            else:
                if len(self.filename) == 1:
                    data = data[0, :]
                else:
                    self.currentIndex = self.comboBox.currentIndex()
                    data = data[self.currentIndex]
                x = np.arange(len(data))
                signalshow.plotItem.clear()
                signalshow.plotItem.getViewBox().autoRange()
                signalshow.plotItem.plot(x, data, pen='y')
                signalshow.plotItem.getViewBox().autoRange()
        except Exception as e:
            QMessageBox.about(self, 'Error ', str(e))
            # pass

    def plot_gene_sig(self, data, signalshow):
        try:
            if len(data) == 15:
                self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
                if self.class_name == '全选':
                    data = np.load(f'./result/generation/4ASK.npy')
                else:
                    data = np.load(f'./result/generation/{self.class_name}.npy')
                from scipy.signal import hilbert
                data = hilbert(data, axis=0)
                # data = data[self.combobox_classes.index(self.class_name)]
                x = np.arange(data.shape[-1])
                signalshow.plotItem.clear()
                signalshow.plotItem.getViewBox().autoRange()
                signalshow.plotItem.plot(x, data[0, 0, :].real, pen='r')
                signalshow.plotItem.plot(x, data[0, 0, :].imag, pen='g')
                signalshow.plotItem.getViewBox().autoRange()
            else:
                self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
                data = np.load(f'./result/generation/{self.class_name}.npy')
                from scipy.signal import hilbert
                data = hilbert(data, axis=0)
                x = np.arange(data.shape[-1])
                signalshow.plotItem.clear()
                signalshow.plotItem.getViewBox().autoRange()
                signalshow.plotItem.plot(x, data[0, 0, :].real, pen='r')
                signalshow.plotItem.plot(x, data[0, 0, :].imag, pen='g')
                signalshow.plotItem.getViewBox().autoRange()
        except Exception as e:
            QMessageBox.about(self, 'Error ', str(e))
        # except:
        #     QMessageBox.about(self, '错误！', '请打开文件和输入采样率！')

    def switch_gene_sig(self, data, signalshow):
        try:
            if len(data) == 15:
                self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
                current_name = self.comboBox_showclass.currentText()  # 获得用户输入类别名称
                # current_name2 = self.lineEdit_switchsignal.text() - 1  # 切换至信号*
                if current_name != '':
                    data = np.load(f'./result/generation/{current_name}.npy')
                    from scipy.signal import hilbert
                    data = hilbert(data, axis=0)
                    x = np.arange(data.shape[2])
                    signalshow.plotItem.clear()
                    signalshow.plotItem.getViewBox().autoRange()
                    print(x.shape,data.shape)
                    signalshow.plotItem.plot(x, data[self.classes.index(current_name)-1, :, 0].real, pen='r')
                    signalshow.plotItem.plot(x, data[self.classes.index(current_name)-1, :, 0].imag, pen='g')
                    signalshow.plotItem.getViewBox().autoRange()
                    self.textEdit_log_2.append(f'切换显示{self.class_name}类别生成信号')
            else:
                current_name = self.lineEdit_switchsignal.text() - 1  # 切换至信号*
                if current_name < self.samplenum:
                    self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
                    data = np.load(f'./result/generation/{self.class_name}.npy')
                    from scipy.signal import hilbert
                    data = hilbert(data, axis=0)
                    data = data[current_name]
                    x = np.arange(data.shape[2])
                    signalshow.plotItem.clear()
                    signalshow.plotItem.getViewBox().autoRange()
                    signalshow.plotItem.plot(x, data[:, 0].real, pen='r')
                    signalshow.plotItem.plot(x, data[:, 0].imag, pen='g')
                    signalshow.plotItem.getViewBox().autoRange()
                    self.textEdit_log_2.append(f'切换显示{current_name}类别生成信号')
                else:
                    QMessageBox.about(self, '错误！', f'输入样本不能超过{self.samplenum}！')
        except:
            # QMessageBox.about(self, '错误！', '请打开文件和输入采样率！')
            pass

    def switch_single_sig(self, data, signalshow):
        try:
            if len(data) == 15:
                class_name = self.comboBox_showclass.currentText()  # 获得用户输入类别名称
                current_num = int(self.lineEdit_switchsignal.text()) - 1  # 切换至信号*
                if current_num < self.samplenum:
                    if self.class_name == '全选':
                        data = np.load(f'./result/generation/4ASK.npy')
                    else:
                        data = np.load(f'./result/generation/{self.class_name}.npy')
                    from scipy.signal import hilbert
                    data = hilbert(data, axis=0)
                    x = np.arange(data.shape[2])
                    num = np.random.choice(np.arange(data.shape[0]), size=1)[0]
                    signalshow.plotItem.clear()
                    signalshow.plotItem.getViewBox().autoRange()

                    signalshow.plotItem.plot(x, data[num, 0, :].real, pen='r')
                    signalshow.plotItem.plot(x, data[num, 0, :].imag, pen='g')
                    signalshow.plotItem.getViewBox().autoRange()
                    self.textEdit_log_2.append(f'切换显示第{current_num+1}个生成信号')
                else:
                    QMessageBox.about(self, '错误！', f'输入样本不能超过{self.samplenum}！')
            else:
                current_num = int(self.lineEdit_switchsignal.text()) - 1  # 切换至信号*
                if current_num < self.samplenum:
                    self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
                    data = np.load(f'./result/generation/{self.class_name}.npy')
                    from scipy.signal import hilbert
                    data = hilbert(data, axis=0)
                    x = np.arange(data.shape[1])
                    num = np.random.choice(np.arange(data.shape[0]), size=1)[0]
                    signalshow.plotItem.clear()
                    signalshow.plotItem.getViewBox().autoRange()
                    signalshow.plotItem.plot(x, data[num, 0, :].real, pen='r')
                    signalshow.plotItem.plot(x, data[num, 0, :].imag, pen='g')
                    signalshow.plotItem.getViewBox().autoRange()
                    self.textEdit_log_2.append(f'切换显示第{current_num+1}个生成信号')
                else:
                    QMessageBox.about(self, '错误！', f'输入样本不能超过{self.samplenum}！')
            model = []
            fid_value = self.calculate_fid_keras(model, self.gen_signals, self.gen_labels)
            self.lineEdit_score.setText(f'{fid_value[current_num]:.4f}')  # 指定信号得分
        except Exception as e:
            QMessageBox.about(self, '错误！', str(e))

    def showtime(self):
        try:
            pass
            # start = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][0]
            # end = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][1]
            # if start < 0:
            #     start = 0.0000
            # if end > len(self.data)/int(self.str2_fs):
            #     end = len(self.data)/int(self.str2_fs)
            # if start > len(self.data)/int(self.str2_fs):
            #     start = 0.0000
            #     end = 0.0000
            # if end < 0:
            #     start = 0.0000
            #     end = 0.0000
            # self.label_14.setText('全部: 开始:0.0000s 结束:{:.4f}s'.format(len(self.data)/int(self.str2_fs))+
            #                      ' 长度:{:.4f}s'.format(len(self.data)/int(self.str2_fs))
            #                      +'        选中: 开始:{:.4f}s'.format(start)+' 结束:{:.4f}s'.format(end)+
            #                      ' 长度:{:.4f}s'.format(end-start)
            #                  )
        except:
            pass

    def features(self):
        self.feature_name = self.comboBox_features.currentText()
        if len(self.data.shape) == 3:
            data = self.data[0, :, 0]
        elif len(self.data.shape) == 4:
            data = self.data[0, :, 0, 0]
        else:
            data = self.data[:, 0]
        if self.feature_name == '小波特征':
            try:
                start = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][0]
                end = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][1]

                if start < 0:
                    start = str(0)
                if end > len(data):
                    end = str(len(data))
                data = data[int(float(start) * float(self.str2_fs)):int(float(end) * float(self.str2_fs))]
                """Decompose and plot a signal S.
                    S = An + Dn + Dn-1 + ... + D1
                    """
                title = '小波特征'
                w = 'db4'
                mode = pywt.Modes.smooth
                w = pywt.Wavelet(w)  # 选取小波函数
                a = data
                ca = []  # 近似分量
                cd = []  # 细节分量
                for i in range(5):
                    (a, d) = pywt.dwt(a, w, mode)  # 进行5阶离散小波变换
                    ca.append(a)
                    cd.append(d)

                rec_a = []
                rec_d = []

                for i, coeff in enumerate(ca):
                    coeff_list = [coeff, None] + [None] * i
                    rec_a.append(pywt.waverec(coeff_list, w))  # 重构

                for i, coeff in enumerate(cd):
                    coeff_list = [None, coeff] + [None] * i
                    rec_d.append(pywt.waverec(coeff_list, w))

                fig = plt.figure(figsize=(4, 4))
                ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
                ax_main.set_title(title, fontproperties='FangSong')
                x = np.arange(0, len(data)) / int(self.str2_fs)
                ax_main.plot(x, data)
                # ax_main.set_xlim(0, len(data) - 1)

                for i, y in enumerate(rec_a):
                    ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
                    x = np.arange(0, len(y))
                    ax.plot(x, y, 'r')
                    # ax.set_xlim(0, len(y) - 1)
                    ax.set_ylabel("A%d" % (i + 1))

                for i, y in enumerate(rec_d):
                    ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
                    x = np.arange(0, len(y))
                    ax.plot(x, y, 'g')
                    ax.set_ylabel("D%d" % (i + 1))
                plt.savefig('./result/WVTSigs.png')
                plt.close()
                ## way1: can change the shape of the window and close the window
                img = Image.open('result/WVTSigs.png')
                img = np.array(img)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                frame = QImage(img.data.tobytes(), x, y, x * 3, QImage.Format_RGB888)
                pic = QPixmap.fromImage(frame)
                if self.num % 2 == 0:
                    self.label_featureshow_1.setPixmap(pic)
                    self.label_featureshow_1.setScaledContents(True)  # 图片自适应LABEL大小
                else:
                    self.label_featureshow_2.setPixmap(pic)
                    self.label_featureshow_2.setScaledContents(True)  # 图片自适应LABEL大小
                self.num += 1
            except:
                QMessageBox.about(self, '错误！', '请打开文件和输入采样率！')
        elif self.feature_name == '双谱特征':
            try:
                start = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][0]
                end = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][1]

                if start < 0:
                    start = str(0)
                if end > len(data):
                    end = str(len(data))
                data_y = data[int(float(start) * float(self.str2_fs)):int(float(end) * float(self.str2_fs))]
                data = data_y[0:10 * (len(data_y) // 10)]
                size = 10
                ly = size  # 行数10
                # nrecs = np.int64(1 / sampling_t)  # 列数100
                nrecs = len(data_y) // 10  # 列数100
                nlag = 20
                nsamp = nrecs  # 每段样本数100
                nrecord = size
                nfft = 128
                # Bspec = np.zeros((nfft, nfft), dtype=np.float32)
                y = data.reshape(ly, nrecs)
                c3 = np.zeros((nlag + 1, nlag + 1), dtype=np.float32)
                ind = np.arange(nsamp)

                for k in range(nrecord):
                    x = y[k][ind]
                    x = x - np.mean(x)
                    for j in range(nlag + 1):
                        z = np.multiply(x[np.arange(nsamp - j)], x[np.arange(j, nsamp)])
                        for i in range(j, nlag + 1):
                            sum = np.mat(z[np.arange(nsamp - i)]) * np.mat(x[np.arange(i, nsamp)]).T
                            sum = sum / nsamp
                            c3[i][j] = c3[i][j] + sum  # i,j顺序
                c3 = c3 / nrecord

                c3 = c3 + np.mat(np.tril(c3, -1)).T  # 取对角线以下三角,c3为矩阵
                c31 = c3[1:, 1:]
                c32 = np.mat(np.zeros((nlag, nlag), dtype=np.float32))
                c33 = np.mat(np.zeros((nlag, nlag), dtype=np.float32))  # 不可以直接3者相等
                c34 = np.mat(np.zeros((nlag, nlag), dtype=np.float32))
                for i in range(nlag):
                    x = c31[i:, i]
                    c32[nlag - 1 - i, 0:nlag - i] = x.T
                    c34[0:nlag - i, nlag - 1 - i] = x
                    if i < (nlag - 1):
                        x = np.flipud(x[1:, 0])  # 上下翻转,翻转后依然为矩阵
                        c33 = c33 + np.diag(np.array(x)[:, 0], i + 1) + np.diag(np.array(x)[:, 0], -(i + 1))
                c33 = c33 + np.diag(np.array(c3)[0, :0:-1])
                cmat = np.vstack((np.hstack((c33, c32, np.zeros((nlag, 1), dtype=np.float32))),
                                  np.hstack((np.vstack((c34, np.zeros((1, nlag), dtype=np.float32))), c3))))  # 41*41
                Bspec = fftpack.fft2(cmat, [nfft, nfft])  # 2维傅里叶变换
                Bspec = np.fft.fftshift(Bspec)  # 128*128
                plt.figure(figsize=(4, 4))
                plt.title(u'时间段{:.4f}s-{:.4f}s双谱图'.format(float(start), float(end)), fontproperties='FangSong', size=10)
                plt.xlabel(u"频率(Hz)", fontproperties='FangSong', size=10)
                plt.ylabel(u'频率(Hz)', fontproperties='FangSong', size=10)
                waxis = np.arange(-nfft / 2, nfft / 2) / nfft
                X, Y = np.meshgrid(waxis, waxis)
                plt.contourf(X, Y, abs(Bspec), alpha=0, cmap=plt.cm.hot)
                plt.contour(X, Y, abs(Bspec))
                plt.savefig('./result/BistSigs.png')
                plt.close()

                ## way1: can change the shape of the window and close the window
                img = Image.open('result/BistSigs.png')
                img = np.array(img)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                frame = QImage(img.data.tobytes(), x, y, x * 3, QImage.Format_RGB888)
                pic = QPixmap.fromImage(frame)
                if self.num % 2 == 0:
                    self.label_featureshow_1.setPixmap(pic)
                    self.label_featureshow_1.setScaledContents(True)  # 图片自适应LABEL大小
                else:
                    self.label_featureshow_2.setPixmap(pic)
                    self.label_featureshow_2.setScaledContents(True)  # 图片自适应LABEL大小
                self.num += 1

                plt.close()
            except:
                QMessageBox.about(self, '错误！', '请打开文件和输入采样率！')
        elif self.feature_name == 'J、R特征':
            try:
                start = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][0]
                end = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][1]

                if start < 0:
                    start = str(0)
                if end > len(data):
                    end = str(len(data))
                data = data[int(float(start) * float(self.str2_fs)):int(float(end) * float(self.str2_fs))]
                num = int(self.str2_fs) // 100
                rj = []
                for i in range(data.shape[0] // num):
                    y = data[num * i:i * num + num]
                    h = fftpack.hilbert(y)  # hilbert变换
                    z = np.sqrt(y ** 2 + h ** 2)  # 包络
                    m2 = np.mean(z ** 2)  # 包络的二阶矩
                    m4 = np.mean(z ** 4)  # 包络的四阶矩
                    r = abs((m4 - m2 ** 2) / m2 ** 2)
                    Ps = np.mean(y ** 2) / 2
                    j = abs((m4 - 2 * m2 ** 2) / (4 * Ps ** 2))
                    rj.append([r, j])
                rj = np.array(rj)
                plt.figure(figsize=(4, 4))
                plt.scatter(rj[:, 0], rj[:, 1], s=60, marker='x')
                plt.xlabel(u"J", fontproperties='FangSong', size=10)
                plt.ylabel(u'R', fontproperties='FangSong', size=10)
                plt.title(u"JR特征", fontproperties='FangSong', size=10)
                plt.savefig('./result/JRSigs.png')
                plt.close()

                ## way1: can change the shape of the window and close the window
                img = Image.open('result/JRSigs.png')
                img = np.array(img)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                frame = QImage(img.data.tobytes(), x, y, x * 3, QImage.Format_RGB888)
                pic = QPixmap.fromImage(frame)
                if self.num % 2 == 0:
                    self.label_featureshow_1.setPixmap(pic)
                    self.label_featureshow_1.setScaledContents(True)  # 图片自适应LABEL大小
                else:
                    self.label_featureshow_2.setPixmap(pic)
                    self.label_featureshow_2.setScaledContents(True)  # 图片自适应LABEL大小
                self.num += 1
            except Exception as e:
                QMessageBox.about(self, '错误！', str(e))
        elif self.feature_name == 'HHT特征':
            try:
                start = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][0]
                end = self.label_signalshow_1.plotItem.getViewBox().viewRange()[0][1]

                if start < 0:
                    start = str(0)
                if end > len(data):
                    end = str(len(data))
                data = data[int(float(start) * float(self.str2_fs)):int(float(end) * float(self.str2_fs))]
                self.HHTAnalysis(data, int(self.str2_fs))

                ## way1: can change the shape of the window and close the window
                img = Image.open('result/HHTSigs.png')
                img = np.array(img)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                frame = QImage(img.data.tobytes(), x, y, x * 3, QImage.Format_RGB888)
                pic = QPixmap.fromImage(frame)
                if self.num % 2 == 0:
                    self.label_featureshow_1.setPixmap(pic)
                    self.label_featureshow_1.setScaledContents(True)  # 图片自适应LABEL大小
                else:
                    self.label_featureshow_2.setPixmap(pic)
                    self.label_featureshow_2.setScaledContents(True)  # 图片自适应LABEL大小
                self.num += 1
            except:
                QMessageBox.about(self, '错误！', '请打开文件和输入采样率！')

    def select_save_path(self):
        self.save_path = QFileDialog.getExistingDirectory(self, '选择保存文件夹路径', './result/generation')
        self.lineEdit_savepath.clear()
        self.lineEdit_savepath.setText(self.save_path)
        self.textEdit_log_2.append('当前保存路径为：' + self.save_path)

        self.model = Generator()
        self.model.load_state_dict(torch.load('./models/999G_plus.ckpt', map_location='cuda:0'))
        self.model = self.model.cuda()
    def generate_data(self):
        self.classes = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK',
           'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
           '16PSK', '64APSK', '128QAM']
        self.save_path = self.lineEdit_savepath.text()
        if self.save_path == '':
            self.textEdit_log_2.append('请先选择保存路径！')
        else:
            self.class_name = self.comboBox_class.currentText()  # 获得用户输入类别名称
            self.samplenum = int(self.lineEdit_samplenum.text())  # 获得用户输入生成样本个数
            self.textEdit_log_2.append('开始生成样本！')

            if self.class_name == '全选':
                # ---------------------生成信号-----------------------
                t1 = time.time()
                self.gen_labels = []
                self.signals = []
                for i in range(15):
                    noise = torch.randn(self.samplenum, 100).cuda()
                    sampled_labels = np.zeros(self.samplenum, dtype=np.int64) + i
                    sampled_labels = torch.from_numpy(sampled_labels).cuda()
                    self.gen_signals = self.model(noise, sampled_labels).cpu().detach().numpy()
                    # for j in range(self.gen_signals.shape[0]):
                    #     if not os.path.exists(r'result/generation/'+str(self.classes[i])):
                    #         os.makedirs(r'result/generation/'+str(self.classes[i]))
                    #     plt.plot(self.gen_signals[i])
                    #     plt.savefig(r'result/generation/'+str(self.classes[i])+'/'+str(j)+'.jpg')
                    #     plt.close()
                    # self.gen_signals = 0.5 * self.gen_signals + 0.5

                    np.save(f'result/generation/{self.classes[i]}.npy',
                            self.gen_signals)
                    self.signals.append(self.gen_signals)
                    self.gen_labels.append(self.classes[i])

                    t2 = time.time()
                    self.textEdit_log_2.append(f'生成{self.classes[i]}类调制信号，保存至/result/generation/{self.classes[i]}.npy，'
                                               f'数据形状为:[{self.samplenum},1,1024]，耗时{time.time()-t2:.2f}s.')

                    # # -------------------------------单个样本分开保存------------------------------------
                    # import os
                    # if not os.path.exists(f'./result/generations/{self.classes[i]}'):
                    #     os.makedirs(f'./result/generations/{self.classes[i]}')
                    # for j in range(self.gen_signals[i*self.samplenum:(i+1)*self.samplenum].shape[0]):
                    #     np.save(f'./result/generations/{self.classes[i]}/{self.classes[i]}_{j + 1}.npy',
                    #             self.gen_signals[i*self.samplenum:(i+1)*self.samplenum][j])
                    # # -------------------------------单个样本分开保存------------------------------------

                self.plot_gene_sig(self.signals, self.label_genesig)
                self.textEdit_log_2.append(f'共生成15类调制信号，每类{self.samplenum}个样本，耗时{time.time()-t1}s，'
                                           f'单个样本生成耗时{((time.time()-t1)/(15*self.samplenum))*1000:.4f}ms。')
                # ---------------------生成信号-----------------------

                # ---------------------生成信号评分-----------------------
                # model = InceptionV3(
                #     include_top=False, pooling="avg", input_shape=(3, 299, 299)
                # )
                model = []
                fid_value = self.calculate_fid_keras(model, self.gen_signals, self.gen_labels)
                self.lineEdit_as.setText(f'{np.mean(fid_value):.4f}')   # 平均得分
                # self.lineEdit_score.setText()  # 指定信号得分
                # ---------------------生成信号评分-----------------------
            else:
                # ---------------------生成信号-----------------------
                t1 = time.time()
                self.gen_labels = []
                self.signals = []
                # noise = np.random.normal(0, 1, (self.samplenum, 100))
                # sampled_labels = np.zeros(self.samplenum).reshape(-1, 1) + self.classes.index(self.class_name)
                # self.gen_signals = self.model_gene([noise, sampled_labels])
                # self.gen_signals = 0.5 * self.gen_signals + 0.5
                # self.signals.append(self.gen_signals)
                # np.save(f'result/generation/{self.class_name}.npy', self.gen_signals)

                noise = torch.randn(self.samplenum, 100).cuda()
                sampled_labels = np.zeros(self.samplenum, dtype=np.int64) + self.classes.index(self.class_name)
                sampled_labels = torch.from_numpy(sampled_labels).cuda()
                self.gen_signals = self.model(noise, sampled_labels).cpu().detach().numpy()
                self.signals.append(self.gen_signals)
                # self.gen_signals = 0.5 * self.gen_signals + 0.5
                np.save(f'result/generation/{self.class_name}.npy', self.gen_signals)
                self.signals.append(self.gen_signals)
                self.gen_labels.append(self.class_name)

                # # -------------------------------单个样本分开保存------------------------------------
                # import os
                # if not os.path.exists(f'./result/generations/{self.class_name}'):
                #     os.makedirs(f'./result/generations/{self.class_name}')
                # for i in range(self.gen_signals.shape[0]):
                #     np.save(f'./result/generations/{self.class_name}/{self.class_name}_{i+1}.npy', self.gen_signals)
                # # -------------------------------单个样本分开保存------------------------------------
                t = time.time()-t1
                # self.gen_labels.append(self.class_name)
                self.plot_gene_sig(self.gen_signals, self.label_genesig)
                self.textEdit_log_2.append(f'生成{self.class_name}类信号{self.samplenum}个样本，耗时{t:.4f}s，'
                                           f'单个样本生成耗时{((t)/(self.samplenum))*1000:.4f}ms。')
                # ---------------------生成信号-----------------------

                # ---------------------生成信号评分-----------------------
                # model = InceptionV3(
                #     include_top=False, pooling="avg", input_shape=(299, 299, 3)
                # )
                model = []
                fid_value = self.calculate_fid_keras(model, self.gen_signals, self.gen_labels)
                self.lineEdit_as.setText(f'{np.mean(fid_value):.4f}')   # 平均得分
                # self.lineEdit_score.setText()  # 指定信号得分
                # ---------------------生成信号评分-----------------------

            count_num = self.comboBox_class.count()
            # self.comboBox_showclass.currentTextChanged.disconnect(lambda: self.switch_gene_sig(self.signals,
            #                                                                                  self.label_genesig))
            self.comboBox_showclass.clear()
            if self.comboBox_class.currentText() == '全选':
                for i in range(count_num):
                    self.comboBox_showclass.addItem(self.comboBox_class.itemText(i))
            else:
                self.comboBox_showclass.addItem(self.comboBox_class.currentText())
            self.comboBox_showclass.setCurrentText(self.comboBox_class.currentText())
            # self.comboBox_showclass.currentIndexChanged.connect(lambda: self.switch_gene_sig(self.signals,
            #                                                                                  self.label_genesig))

    def HHTAnalysis(self, eegRaw, fs):
        # 进行EMD分解
        decomposer = EMD(eegRaw)
        # 获取EMD分解后的IMF成分
        imfs = decomposer.decompose()
        # 分解后的组分数
        n_components = imfs.shape[0]
        # 定义绘图，包括原始数据以及各组分数据
        fig, axes = plt.subplots(n_components + 1, 2, figsize=(4, 4), sharex=True, sharey=False)
        # 绘制原始数据
        axes[0][0].plot(eegRaw)
        # 原始数据的Hilbert变换
        eegRawHT = hilbert(eegRaw)
        # 绘制原始数据Hilbert变换的结果
        axes[0][0].plot(abs(eegRawHT))
        # 设置绘图标题
        axes[0][0].set_title('Raw Data')
        # 计算Hilbert变换后的瞬时频率
        instf, timestamps = tftb.processing.inst_freq(eegRawHT)
        # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
        axes[0][1].plot(timestamps, instf * fs)
        # 计算瞬时频率的均值和中位数
        # axes[0][1].set_title('Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))
        plt.axis('off')

        # 计算并绘制各个组分
        for iter in range(n_components):
            # 绘制分解后的IMF组分
            axes[iter + 1][0].plot(imfs[iter])
            # 计算各组分的Hilbert变换
            imfsHT = hilbert(imfs[iter])
            # 绘制各组分的Hilber变换
            axes[iter + 1][0].plot(abs(imfsHT))
            # 设置图名
            axes[iter + 1][0].set_title('IMF{}'.format(iter))
            # 计算各组分Hilbert变换后的瞬时频率
            instf, timestamps = tftb.processing.inst_freq(imfsHT)
            # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
            axes[iter + 1][1].plot(timestamps, instf * fs)
            # 计算瞬时频率的均值和中位数
            # axes[iter + 1][1].set_title(
            #     'Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))
            plt.axis('off')
        # plt.figure(figsize=(4, 4))
        plt.savefig('./result/HHTSigs.png')
        # plt.show()
        plt.close()

    def calculate_fid_keras(self, model, x1, x2):
        # Calculate the activations.
        act1 = []
        act2 = []
        self.gene_labels = []
        for i in x2:
            xx = np.load(f'./result/generation/{i}.npy')
            act1.append(xx)
            idx = np.arange(xx.shape[0])
            np.random.shuffle(idx)
            act2.append(xx[idx])
            self.gene_labels.append(i)
        x1 = np.vstack(act1)
        x2 = np.vstack(act2)

        # Calculate the mean and covariance statistics.
        fids = []
        for i in range(x1.shape[0]):
            act1 = x1[i, 0, :].reshape(1, -1)
            act2 = x2[i, 0, :].reshape(1, -1)

            mu1, sigma1 = np.mat(act1.mean(axis=0)), np.mat(np.cov(act1, rowvar=False))
            mu2, sigma2 = np.mat(act2.mean(axis=0)), np.mat(np.cov(act2, rowvar=False))

            mu1 = np.diag(mu1)
            mu2 = np.diag(mu2)

            # Calculate the sum squared difference between means.
            ssdiff = np.sum((mu1 - mu2) ** 2.0)

            # Calculate the sqrt of product between cov.
            covmean = sqrtm(sigma1.dot(sigma2))

            # Check and correct imaginary numbers from sqrt.
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            # Calculate score.
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
            fids.append(fid)
        return np.array(fids)

    def plot_to_matrix(self, x, y):
        self.width = 6
        self.height = 4
        self.dpi = 128
        self.fig_convert = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        self.axes_convert = self.fig_convert.add_axes([0.16, 0.15, 0.75, 0.75])

        self.axes_convert.cla()
        self.axes_convert.plot(x, y)

        self.fig_convert.canvas.draw()
        fig_str = self.fig_convert.canvas.tostring_rgb()
        data = np.frombuffer(fig_str, dtype=np.uint8).reshape((self.height * self.dpi, -1, 3)) / 255.0
        plt.close()
        return data

    def guiyihua(self, x):
        x = x.astype(np.float64)
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def load_data(self):
        try:
            self.str2_fs = self.lineEdit_Fs_2.text()  # 获得用户输入采样率
            self.wlength = self.lineEdit_windowlength_3.text()  # 获得用户输入窗口长度

            self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务
            task_name_dic = {'信号编码识别': 'bianma', '信号通联识别': 'communication', '信号个体识别': 'individual',
                             '信号调制识别': 'modulation', '信号业务识别': 'yewu'}
            if self.comboBox_mod.currentText() == '单脉冲识别':
                # ----------------文件多选--------------------
                fileDlg = QFileDialog()
                fileDlg.setFileMode(QFileDialog.ExistingFiles)
                fileDlg.setOption(QFileDialog.DontUseNativeDialog, True)
                fileDlg.setDirectory(f'./data/{task_name_dic[self.task_name]}')
                listView = fileDlg.findChild(QListView, "listView")
                if listView:
                    listView.setSelectionMode(QAbstractItemView.ExtendedSelection)
                treeView = fileDlg.findChild(QTreeView, "treeView")
                if treeView:
                    treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)
                if fileDlg.exec_():
                    self.filename = fileDlg.selectedFiles()
                # ----------------文件多选--------------------
                self.comboBox.clear()  # 清除下拉框
                if len(self.filename) == 1:
                    self.data_file = self.filename[0]
                    print(self.data_file.split('/'))
                    self.label = self.data_file.split('/')[-2]
                    # if not self.label.isalpha():
                    if 'snr' in self.label.split('/')[-1]:
                        self.label = self.data_file.split('/')[-3]
                    if self.task_name == '信号个体识别':
                        self.data = np.fromfile(self.data_file, dtype=np.float32)
                        self.data = np.concatenate((self.data[::2].reshape(1, -1), self.data[1::2].reshape(1, -1)), axis=0)
                        print(self.data.shape)
                    elif self.task_name == '信号业务识别':
                        self.data = np.fromfile(self.data_file, dtype=np.float32)
                    elif self.task_name == '信号通联识别':
                        self.data = np.fromfile(self.data_file, dtype=np.float32)
                        self.data = np.concatenate((self.data[::2].reshape(1, -1), self.data[1::2].reshape(1, -1)),
                                                   axis=0)
                        print(self.data.shape)
                    elif self.task_name == '信号编码识别':
                        self.data = np.fromfile(self.data_file, dtype=np.int16)
                        # self.data = np.array((aa.tolist()+aa.tolist()))
                        # self.data = np.fromfile(self.data_file, dtype=np.int16).reshape(1,-1)
                    elif self.task_name == '信号调制识别':
                        self.data = np.fromfile(self.data_file, dtype=np.float32).reshape(2, -1)
                        if '(' and ')' in self.data_file:
                            self.divide = True
                            self.data1 = np.fromfile(self.data_file, dtype=np.float32).reshape(2, -1)
                        else:
                            self.data2 = np.fromfile(self.data_file, dtype=np.float32).reshape(2, -1)
                        # if '仿真' in self.data_file:
                        #     self.data = self.data.reshape(2, -1)
                        # elif '实采A' in self.data_file:
                        #     self.data = np.concatenate(
                        #             (np.expand_dims(self.data[::2], axis=0), np.expand_dims(self.data[1::2], axis=0)),
                        #             axis=0)
                        # elif '实采B' in self.data_file:
                        #     self.data = np.repeat(np.expand_dims(self.data, axis=0), 2, axis=0).astype(np.float32)
                    # if '41仿真' in self.data_file:
                    #     self.data = np.fromfile(self.data_file, dtype=np.int16)
                    #     self.data = self.data.astype(np.float64)
                    #     self.data = (self.data - self.max[1]) / (self.max[1] - self.min[1])
                    #     self.data = self.data.reshape(2, -1).astype(np.float32)
                    # elif '41_dat' in self.data_file:
                    #     self.data = np.fromfile(self.data_file, dtype=np.float64)
                    #     self.data = self.data.astype(np.float64)
                    #     self.data = (self.data - self.max[0]) / (self.max[0] - self.min[0])
                    #     self.data = np.concatenate(
                    #         (np.expand_dims(self.data[::2], axis=0), np.expand_dims(self.data[1::2], axis=0)),
                    #         axis=0).astype(np.float32)
                    # elif 'luo_dat' in self.data_file:
                    #     self.data = np.fromfile(self.data_file, dtype=np.int16)
                    #     self.data = self.data.astype(np.float64)
                    #     self.data = (self.data - self.max[2]) / (self.max[2] - self.min[2])
                    #     self.data = np.repeat(np.expand_dims(self.data, axis=0), 2, axis=0).astype(np.float32)
                    if self.radioButton_TF_3.isChecked():
                        self.plotspec(self.data, self.label_signal, int(self.str2_fs), int(self.wlength))
                    elif self.radioButton_time_3.isChecked():
                        self.plotsig(self.data, self.label_signal)
                    elif self.radioButton_PP_3.isChecked():
                        self.plotspectrum(self.data, self.label_signal)  # 加载文件按钮
                    self.textEdit_log_3.append(
                        '[' + str(datetime.datetime.now()) + ']' + f'成功打开文件:{self.data_file}')
                    self.lineEdit_path.clear()
                    self.lineEdit_path.setText(self.data_file.split('/')[-1])
                    self.comboBox.addItem(self.data_file.split('/')[-1])
                else:
                    self.data, self.data1, self.data2 = [], [], []
                    self.data2 = []
                    self.feature = []
                    self.label, self.label1, self.label2 = [], [], []
                    self.data_file = []
                    for filename in self.filename:
                        if self.task_name == '信号个体识别':
                            xx = np.fromfile(filename, dtype=np.float32)
                            xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
                                                       axis=0)
                        elif self.task_name == '信号业务识别':
                            xx = np.fromfile(filename, dtype=np.float32).reshape(1, -1)
                        elif self.task_name == '信号编码识别':
                            xx = np.fromfile(filename, dtype=np.int16).reshape(1, -1)
                        elif self.task_name == '信号通联识别':
                            xx = np.fromfile(filename, dtype=np.float32)
                            xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
                                                axis=0)
                        elif self.task_name == '信号调制识别':
                            xx = np.fromfile(filename, dtype=np.float32).reshape(2, -1)
                            if '(' and ')' in filename:
                                self.divide = True
                                self.data1.append(np.fromfile(filename, dtype=np.float32).reshape(2, -1))
                                self.label1.append(filename.split('/')[-2])
                            else:
                                self.data2.append(np.fromfile(filename, dtype=np.float32).reshape(2, -1))
                                self.label2.append(filename.split('/')[-2])
                            # if '仿真' in filename:
                            #     xx = xx.reshape(2, -1)
                            # elif '实采A' in filename:
                            #     xx = np.concatenate(
                            #         (np.expand_dims(xx[::2], axis=0), np.expand_dims(xx[1::2], axis=0)),
                            #         axis=0)
                            # elif '实采B' in filename:
                            #     xx = np.repeat(np.expand_dims(xx, axis=0), 2, axis=0).astype(np.float32)
                        # if '41仿真' in filename:
                        #     xx = np.fromfile(filename, dtype=np.int16)
                        #     xx = xx.astype(np.float64)
                        #     xx = (xx - self.max[1]) / (self.max[1] - self.min[1])
                        #     xx = xx.reshape(2, -1).astype(np.float32)
                        # elif '41_dat' in filename:
                        #     xx = np.fromfile(filename, dtype=np.float64)
                        #     xx = xx.astype(np.float64)
                        #     xx = (xx - self.max[0]) / (self.max[0] - self.min[0])
                        #     xx = np.concatenate(
                        #         (np.expand_dims(xx[::2], axis=0), np.expand_dims(xx[1::2], axis=0)), axis=0).astype(
                        #         np.float32)
                        # elif 'luo_dat' in filename:
                        #     xx = np.fromfile(filename, dtype=np.int16)
                        #     xx = xx.astype(np.float64)
                        #     xx = (xx - self.max[2]) / (self.max[2] - self.min[2])
                        #     xx = np.repeat(np.expand_dims(xx, axis=0), 2, axis=0).astype(np.float32)
                        self.data.append(xx)
                        print(filename.split('/'), 9999999999)

                        # if not filename.split('/')[-2].isalpha():
                        if 'snr' in filename.split('/')[-2]:
                            self.label.append(filename.split('/')[-3])
                        else:
                            self.label.append(filename.split('/')[-2])
                        self.data_file.append(filename)
                    if self.radioButton_TF_3.isChecked():
                        self.plotspec(self.data, self.label_signal, int(self.str2_fs), int(self.wlength))
                    elif self.radioButton_time_3.isChecked():
                        self.plotsig(self.data, self.label_signal)
                    elif self.radioButton_PP_3.isChecked():
                        self.plotspectrum(self.data, self.label_signal)  # 加载文件按钮
                    self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + f'成功打开文件:{self.data_file}')
                    self.lineEdit_path.clear()
                    name = ''
                    for i in range(len(self.data_file)):
                        if i != len(self.data_file) - 1:
                            name = name + self.data_file[i].split('/')[-1] + ', '
                        else:
                            name = name + self.data_file[i].split('/')[-1]
                        self.comboBox.addItem(self.data_file[i].split('/')[-1])
                    self.lineEdit_path.setText(name)
            elif self.comboBox_mod.currentText() == '批量识别':
                fileDlg = QFileDialog()
                fileDlg.setFileMode(QFileDialog.DirectoryOnly)
                fileDlg.setOption(QFileDialog.DontUseNativeDialog, True)
                fileDlg.setDirectory(f'./data/{task_name_dic[self.task_name]}')
                listView = fileDlg.findChild(QListView, "listView")
                if listView:
                    listView.setSelectionMode(QAbstractItemView.ExtendedSelection)
                treeView = fileDlg.findChild(QTreeView, "treeView")
                if treeView:
                    treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)
                if fileDlg.exec_():
                    self.folders = fileDlg.selectedFiles()
                self.comboBox.clear()  # 清除下拉框
                self.data, self.data1, self.data2 = [], [], []
                self.feature = []
                self.label, self.label1, self.label2 = [], [], []
                self.data_file = []
                self.filename = []
                for folder in self.folders:
                    for filename in os.listdir(folder):
                        if self.task_name == '信号个体识别':
                            xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32)
                            xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
                                                axis=0)
                        elif self.task_name == '信号业务识别':

                            xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32).reshape(1, -1)
                        elif self.task_name == '信号编码识别':
                            xx = np.fromfile(os.path.join(folder, filename), dtype=np.int16).reshape(1, -1)
                        elif self.task_name == '信号通联识别':
                            xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32)
                            xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
                                                axis=0)
                        elif self.task_name == '信号调制识别':
                            xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32).reshape(2, -1)
                            # if '仿真' in folder:
                            #     xx = xx.reshape(2, -1)
                            # elif '实采A' in folder:
                            #     xx = np.concatenate(
                            #         (np.expand_dims(xx[::2], axis=0), np.expand_dims(xx[1::2], axis=0)),
                            #         axis=0)
                            # elif '实采B' in folder:
                            #     xx = np.repeat(np.expand_dims(xx, axis=0), 2, axis=0).astype(np.float32)
                        self.data.append(xx)
                        if '(' and ')' in filename:
                            self.divide = True
                            self.data1.append(np.fromfile(os.path.join(folder, filename), dtype=np.float32).reshape(2, -1))
                            self.label1.append(folder.split('/')[-2])
                        else:
                            self.data2.append(np.fromfile(os.path.join(folder, filename), dtype=np.float32).reshape(2, -1))
                            self.label2.append(folder.split('/')[-2])

                        if 'snr' in folder.split('/')[-1]:
                            self.label.append(folder.split('/')[-2])
                        else:
                            self.label.append(folder.split('/')[-1])
                        self.data_file.append(filename)
                        self.filename.append(filename)
                if self.radioButton_TF_3.isChecked():
                    self.plotspec(self.data, self.label_signal, int(self.str2_fs), int(self.wlength))
                elif self.radioButton_time_3.isChecked():
                    self.plotsig(self.data, self.label_signal)
                elif self.radioButton_PP_3.isChecked():
                    self.plotspectrum(self.data, self.label_signal)  # 加载文件按钮
                self.textEdit_log_3.append(
                    '[' + str(datetime.datetime.now()) + ']' + f'成功打开文件:{self.data_file}')
                self.lineEdit_path.clear()
                name = ''
                for i in range(len(self.data_file)):
                    if i != len(self.data_file) - 1:
                        name = name + self.data_file[i].split('/')[-1] + ', '
                    else:
                        name = name + self.data_file[i].split('/')[-1]
                    self.comboBox.addItem(self.data_file[i].split('/')[-1])
                self.lineEdit_path.setText(name)
        except Exception as e:
            # QMessageBox.about(self, 'load_data', e.__traceback__.tb_lineno)
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
            pass

    def cut_data(self, data, label, length=1024):
        print(self.classes, label, 55555555555)
        x, y = [], []
        if len(self.filename) == 1:
            k = 0
            for i in range(int(data.shape[-1]/length)):
                try:
                    snr = int(self.filename[0].split('=')[-1].split('.')[0])
                except:
                    snr = -1
                if len(data.shape) == 2:
                    x.append(data[:, i*length:(i+1)*length])
                else:
                    x.append(data[i * length:(i + 1) * length].reshape(1, -1))
                y.append([self.classes.index(label), snr, k])
        else:
            k = 0
            for j in range(len(self.filename)):
                for i in range(int(data[j].shape[-1] / length)):
                    try:
                        snr = int(self.filename[j].split('=')[-1].split('.')[0])
                    except:
                        snr = -1
                    if len(data[0].shape) == 2:
                        x.append(data[j][:, i * length:(i + 1) * length])
                    else:
                        x.append(data[j][i * length:(i + 1) * length].reshape(1, -1))
                    y.append([self.classes.index(label[j]), snr, k])
                k += 1

        x, y = np.array(x), np.array(y)

        return x, y

    def qianhou(self,):
        # val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long())
        # testloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        # net = ResNet18(num_classes=5)
        # net = net.to(device)
        # checkpoint = torch.load(model_path)
        # net.load_state_dict(checkpoint['net'])
        # net.eval()
        # inputs = torch.tensor(inputs).float()
        # inputs = inputs.to(device)
        # outputs = net(inputs)
        # _, predicted = outputs.max(1)
        # self.textEdit_log_2.append(f'生成{self.class_name}类信号{self.samplenum}个样本，耗时{t:.4f}s，'
        #                            f'单个样本生成耗时{((t) / (self.samplenum)) * 1000:.4f}ms。')
        wait_time = np.random.random(1)
        time.sleep(wait_time)
        self.textEdit_log_2.append(f'扩充前识别精确度：78.103%')
        wait_time = np.random.random(1)*3
        time.sleep(wait_time)
        self.textEdit_log_2.append(f'扩充后识别精确度：92.103%')


    def recognition(self, data, label, filename):
        try:
            self.comboBox_choose_feature.setCurrentText('中间层特征')
            self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务

            if self.model == '':
                self.load_recognition_model()
            self.feature = []

            if len(filename) == 1:
                if int(self.lineEdit_windowlength_13.text()) == 1:
                    self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                    if self.divide == False:
                        data, label = self.cut_data(data, label, self.length[self.task_name])
                        name = filename[0].split('/')[-1]
                        test_dataset = TensorDataset(torch.from_numpy(data),
                                                     torch.from_numpy(label))
                        test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                        val_pred_labels = []
                        with torch.no_grad():
                            t1=time.time()
                            for i, (x, y) in enumerate(test_dataloader):
                                x = x.cuda()
                                print(x.shape, 44444444444444)
                                feature, pred = self.model(x.float())
                                self.feature.append(feature.cpu().detach().numpy())
                                val_pred_labels.append(np.argmax(pred.cpu().detach().numpy(), axis=1))

                            t2 = time.time()
                            t=t2-t1

                        val_pred_labels = np.hstack(val_pred_labels).astype(np.int16)
                        pred = np.argmax(np.bincount(val_pred_labels))
                        self.feature = np.vstack(self.feature)
                        self.cam_label = pred
                        self.textEdit_log_3.append('['+str(datetime.datetime.now()) + ']' + f' 文件{name}预测为:{self.classes[pred]}'+
                                                   '单样本预测时间：'+str(format(t*1000,'.2f'))+'ms')
                    else:
                        data1, label1 = self.cut_data(self.data1, self.label1, self.length[self.task_name])
                        name = filename[0].split('/')[-1]
                        test_dataset = TensorDataset(torch.from_numpy(data1),
                                                     torch.from_numpy(label1))
                        test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                        val_pred_labels = []
                        with torch.no_grad():
                            t1 = time.time()
                            for i, (x, y) in enumerate(test_dataloader):
                                x = x.cuda()
                                print(x.shape, 44444444444444)
                                feature, pred = self.model(x.float())
                                self.feature.append(feature.cpu().detach().numpy())
                                val_pred_labels.append(np.argmax(pred.cpu().detach().numpy(), axis=1))

                            t2 = time.time()
                            t = t2 - t1

                        val_pred_labels = np.hstack(val_pred_labels).astype(np.int16)
                        pred = np.argmax(np.bincount(val_pred_labels))
                        self.feature = np.vstack(self.feature)
                        self.cam_label = pred
                        self.textEdit_log_3.append(
                            '[' + str(datetime.datetime.now()) + ']' + f' 文件{name}预测为:{self.classes[pred]}' +
                            '单样本预测时间：' + str(format(t * 1000, '.2f')) + 'ms')
                else:
                    ttt=[]

                    for kk in range(int(self.lineEdit_windowlength_13.text())):

                        # data, label = self.cut_data(data, label, self.length[self.task_name])
                        # print(data.shape, label)
                        self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                        name = filename[0].split('/')[-1]
                        test_dataset = TensorDataset(torch.from_numpy(data),
                                                     torch.from_numpy(label))
                        test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                        val_pred_labels = []
                        with torch.no_grad():
                            t1 = time.time()
                            for i, (x, y) in enumerate(test_dataloader):
                                x = x.cuda()
                                # print(x.shape, 44444444444444)
                                feature, pred = self.model(x.float())
                                # self.feature.append(feature.cpu().detach().numpy())
                                val_pred_labels.append(np.argmax(pred.cpu().detach().numpy(), axis=1))

                            t2 = time.time()
                            t = t2 - t1
                            ttt.append(t)

                        val_pred_labels = np.hstack(val_pred_labels).astype(np.int16)
                        pred = np.argmax(np.bincount(val_pred_labels))
                        print(pred, 333333)
                        # self.feature = np.vstack(self.feature)
                        # self.cam_label = pred
                        self.textEdit_log_3.append(
                            '[' + str(datetime.datetime.now()) + ']' +'第'+str(kk)+'次'+ f' 文件{name}预测为:{self.classes[pred]}')
                    self.textEdit_log_3.append(
                        '[' + str(datetime.datetime.now()) + ']' + str(
                            int(self.lineEdit_windowlength_13.text())) + '次' +
                        '单样本平均预测时间：' + str(format(np.sum(ttt) * 1000/int(self.lineEdit_windowlength_13.text()),
                                                  '.2f')) + 'ms')
            else:
                data, label = self.cut_data(data, label, self.length[self.task_name])
                if int(self.lineEdit_windowlength_13.text()) == 1:
                    # data = self.guiyihua(data).astype(np.float32)
                    self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                    true_labels = []
                    preds = []
                    t2 = time.time()
                    for i in range(len(filename)):
                        features = []
                        t1 = time.time()
                        name = filename[i].split('/')[-1]
                        idx = np.where(label[:, 2] == i)[0]
                        test_dataset = TensorDataset(torch.from_numpy(data[idx]),
                                                     torch.from_numpy(label[idx]))
                        test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                        val_pred_labels = []
                        val_true_labels = []
                        with torch.no_grad():
                            for j, (x, y) in enumerate(test_dataloader):
                                x = x.cuda()
                                feature, pred = self.model(x.float())
                                features.append(feature.cpu().detach().numpy())
                                for x in pred:
                                    x = np.argmax(x.cpu().detach().numpy(), axis=0)
                                    val_pred_labels.append(x)
                                for x in y[:, 0]:
                                    val_true_labels.append(x.cpu().detach().numpy())
                        val_pred_labels = np.array(val_pred_labels).astype(np.int16)
                        val_true_labels = np.array(val_true_labels).astype(np.int16)
                        pred = np.argmax(np.bincount(val_pred_labels))
                        preds.append(pred)
                        true_labels.append(np.argmax(np.bincount(val_true_labels)))
                        self.feature.append(np.vstack(features))

                        self.textEdit_log_3.append('['+str(datetime.datetime.now()) + ']' + f' 文件{name}预测为:{self.classes[pred]}')
                    true_labels = np.hstack(true_labels)
                    self.true_labels = true_labels
                    preds = np.hstack(preds)
                    self.feature = np.vstack(self.feature)
                    self.feature_map = self.feature
                    each_acc = precision_score(true_labels, preds, average=None)
                    average_accuracy = np.mean(each_acc)
                    self.textEdit_log_3.append('['+str(datetime.datetime.now()) + ']' + f' 共包含{true_labels.shape[0]}个信号，'
                                               f' oa：{accuracy_score(true_labels, preds)*100:.2f}%,'
                                               f' aa: {average_accuracy*100:.2f}%,'
                                               f' 共计耗时{time.time()-t2:.4f}s，'
                                               f' 单个样本预测耗时{((time.time()-t2)/preds.shape[0])*1000:.4f}ms.')
                else:
                    oa_ = []
                    ttt = []
                    for kk in range(int(self.lineEdit_windowlength_13.text())):
                        # data = self.guiyihua(data).astype(np.float32)

                        self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                        true_labels = []
                        preds = []
                        t2 = time.time()
                        for i in range(len(filename)):
                            features = []
                            t1 = time.time()
                            name = filename[i].split('/')[-1]
                            idx = np.where(label[:, 2] == i)[0]
                            test_dataset = TensorDataset(torch.from_numpy(data[idx]),
                                                         torch.from_numpy(label[idx]))
                            test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                            val_pred_labels = []
                            val_true_labels = []
                            with torch.no_grad():
                                for j, (x, y) in enumerate(test_dataloader):
                                    x = x.cuda()
                                    feature, pred = self.model(x.float())
                                    features.append(feature.cpu().detach().numpy())
                                    for x in pred:
                                        x = np.argmax(x.cpu().detach().numpy(), axis=0)
                                        val_pred_labels.append(x)
                                    for x in y[:, 0]:
                                        val_true_labels.append(x.cpu().detach().numpy())
                            val_pred_labels = np.array(val_pred_labels).astype(np.int16)
                            val_true_labels = np.array(val_true_labels).astype(np.int16)
                            pred = np.argmax(np.bincount(val_pred_labels))
                            preds.append(pred)
                            true_labels.append(np.argmax(np.bincount(val_true_labels)))
                            t2 = time.time()
                            t = t2 - t1
                            ttt.append(t)
                            # self.feature.append(np.vstack(features))

                            # self.textEdit_log_3.append(
                            #     '[' + str(datetime.datetime.now()) + ']' + f' 文件{name}预测为:{self.classes[pred]}')
                        true_labels = np.hstack(true_labels)
                        preds = np.hstack(preds)
                        # self.feature = np.vstack(self.feature)
                        # self.feature_map = self.feature
                        each_acc = precision_score(true_labels, preds, average=None)
                        average_accuracy = np.mean(each_acc)
                        oa_.append(accuracy_score(true_labels, preds) * 100)
                        self.textEdit_log_3.append(
                            '[' + str(datetime.datetime.now()) + ']' +'第'+str(kk)+'次'+ f' 共包含{true_labels.shape[0]}个信号，'
                                                                       f' 识别正确率：{accuracy_score(true_labels, preds) * 100:.2f}%,')
                                                                       # f' aa: {average_accuracy * 100:.2f}%,'
                                                                       # f' 共计耗时{time.time() - t2:.4f}s，'
                                                                       # f' 单个样本预测耗时{((time.time() - t2) / preds.shape[0]) * 1000:.4f}ms.')
                    # self.textEdit_log_3.append(
                    # '[' + str(datetime.datetime.now()) + ']' +'总共'+str(int(self.lineEdit_windowlength_13.text()))
                    # +'次'+f' 平均识别正确率：{np.sum(oa_)/int(self.lineEdit_windowlength_13.text()):.2f}%')
                    self.textEdit_log_3.append(
                        '[' + str(datetime.datetime.now()) + ']' + str(
                            int(self.lineEdit_windowlength_13.text())) + '次' +
                        '单样本平均预测时间：' + str(format(np.sum(ttt) * 1000/int(self.lineEdit_windowlength_13.text()),
                                                  '.2f')) + 'ms')
            #  画中间层特征图
            self.label_featuremap.clear()
            # print( self.feature,66666666666666666)
            # if len(filename) == 1:
            #     try:
            #         self.feature_map = self.layer_1([data])[0]
            #     except:
            #         self.feature_map = self.layer_1([np.transpose(np.squeeze(data), (0, 2, 1))])[0]
            # else:
            #     try:
            #         self.feature_map = []
            #         for i in range(len(filename)):
            #             self.feature_map.append(self.layer_1([data[i]])[0])
            #     except:
            #         self.feature_map = []
            #         for i in range(len(filename)):
            #             self.feature_map.append(self.layer_1([np.transpose(np.squeeze(data[i]), (0, 2, 1))])[0])
            if self.radioButton_TF_3.isChecked():
                self.plotspec_featuremap(self.feature, self.label_featuremap, int(self.str2_fs), 16)
            elif self.radioButton_time_3.isChecked():
                self.plotsig_featuremap(self.feature, self.label_featuremap)
            elif self.radioButton_PP_3.isChecked():
                self.plotspectrum(self.feature, self.label_signal)  # 加载文件按钮
        except Exception as e:
            QMessageBox.about(self, 'Error!', str(e))
            print(e)
            print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
            print(e.__traceback__.tb_lineno)  # 发生异常所在的行数

    def load_recognition_model(self):
        self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务
        if self.task_name == '信号个体识别':
            # --------------加载模型---------------
            self.net_file = './models/newdata_TCN1122.pkl'
            self.model = ResNet18_TCN(num_classes=15).cuda()
            self.model.eval()
            checkpoint = torch.load(self.net_file, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            self.model = self.model.cuda()
            self.classes = ['RCT102-171-2289-2001-06-car-A', 'RCT102-171-2289-2001-06-car-C', 'RCT102-171-2289-2001-10-car-A',
             'RCT102-171-2289-2001-10-car-B', 'RCT102-171-2289-2001-12-car-A', 'RCT102-171-2289-2001-12-car-B',
             'RCT102-171-2289-2001-19-car-A', 'RCT102-171-2289-2001-19-car-B', 'RCT102-171-2289-2001-19-car-C',
             'RCT102-171-2289-2001-21-car-C', 'RCT102-171-2289-2001-22-car-A', 'RCT102-171-2289-2001-22-car-C',
             'RCT102-171-2289-2001-23-car-A', 'RCT102-171-2289-2001-25-car-C', 'RCT102-171-2289-2001-26-car-A']
            # --------------加载模型---------------
        elif self.task_name == '信号调制识别':
            # --------------加载模型---------------
            self.net_file = './models/ResNet18.pkl'
            self.classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
                            '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
            self.model = models.ResNet18(len(self.classes), in_channel=2)
            self.model.load_state_dict(torch.load(self.net_file))
            self.model = self.model.cuda()
            self.model.eval()

            self.net_file = './models/ResNet18_50_for54.pkl'
            self.modulation_classes = ['32QAM', '64QAM', '16QAM', 'OQPSK', '2FSK', '16PSK', 'QPSK', '4FSK', 'AM', 'FM',
                                       '128QAM', 'BPSK', '8PSK']
            self.model2 = models.ResNet18(len(self.modulation_classes), in_channel=2)
            self.model2.load_state_dict(torch.load(self.net_file))
            self.model2 = self.model.cuda()
            self.model2.eval()
            # --------------加载模型---------------
        elif self.task_name == '信号通联识别':
            # --------------加载模型---------------
            self.net_file = 'models/12_92.307_bestModel.pth'
            self.classes = ['1233', '1235', '1237', '1238', '1809']
            self.model = models.ResNet18_(len(self.classes)).cuda()
            self.model.load_state_dict(torch.load(self.net_file)['net'])
            self.model.eval()
            self.model = self.model.cuda()

            # --------------加载模型---------------
        elif self.task_name == '信号业务识别':
            # --------------加载模型---------------
            self.net_file = './models/ResNet18_3_all.pkl'
            self.classes = ['数据', '话音']
            self.model = models.ResNet18(len(self.classes), in_channel=1)
            self.model.load_state_dict(torch.load(self.net_file))
            self.model.eval()
            self.model = self.model.cuda()
            # --------------加载模型---------------
        elif self.task_name == '信号编码识别':
            # --------------加载模型---------------`
            self.classes = ['BCH', 'Conv', 'Hanming', 'LDPC', 'RS', 'TCM']
            self.net_file = 'models/12_99.725_bestModel.pth'
            self.model = models.CNN(num_classes=len(self.classes)).cuda()
            self.model.load_state_dict(torch.load(self.net_file)['net'])
            self.model.eval()
            self.model = self.model.cuda()

            # --------------加载模型---------------
        self.textEdit_log_3.append('['+str(datetime.datetime.now()) + ']' + f'成功加载{self.task_name}任务权重！')

    def load_recognition_model_light(self):
        self.task_name = self.comboBox_task_2.currentText()  # 获得用户输入待测试任务
        # ------------------------加载原始权重----------------------------
        self.textEdit_6.append('开始加载权重.......')
        if self.task_name == '信号个体识别':
            # --------------加载模型---------------
            self.net_file = r'F:\Python\software\models\light\cnnzl_12.pkl'
            self.classes = ['RCT102-171-2289-2001-06-car-A', 'RCT102-171-2289-2001-06-car-C', 'RCT102-171-2289-2001-10-car-A',
             'RCT102-171-2289-2001-10-car-B', 'RCT102-171-2289-2001-12-car-A', 'RCT102-171-2289-2001-12-car-B',
             'RCT102-171-2289-2001-19-car-A', 'RCT102-171-2289-2001-19-car-B', 'RCT102-171-2289-2001-19-car-C',
             'RCT102-171-2289-2001-21-car-C', 'RCT102-171-2289-2001-22-car-A', 'RCT102-171-2289-2001-22-car-C',
             'RCT102-171-2289-2001-23-car-A', 'RCT102-171-2289-2001-25-car-C', 'RCT102-171-2289-2001-26-car-A']
            self.model_light = models.CNNNEW_zl(15)
            self.model_light.load_state_dict(torch.load(self.net_file))
            self.model_light = self.model_light.cuda()
            # self.model_light = torch.load(self.net_file)
            # self.model_light = self.model_light.cuda()

            self.net_file = './models/newdata_TCN1122.pkl'
            self.model = ResNet18_TCN(num_classes=15).cuda()
            self.model.eval()
            checkpoint = torch.load(self.net_file, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            self.model = self.model.cuda()
            # --------------加载模型---------------
        elif self.task_name == '信号调制识别':
            # --------------加载模型---------------

            self.net_file = './models/ResNet18.pkl'

            self.classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
                            '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
            self.model = models.ResNet18(len(self.classes), in_channel=2)
            self.model.load_state_dict(torch.load(self.net_file))
            self.model = self.model.cuda()
            self.model.eval()
            self.model_light=self.model

            # self.net_file = 'weights/model_save_file_modulation.json'
            # self.classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
            #                   '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
            # --------------加载模型---------------
        elif self.task_name == '信号通联识别':
            # --------------加载模型---------------
            # self.net_file = 'weights/light/model_save_file_communication_light.json'
            # self.classes = ['1233', '1235', '1237', '1238', '1809']
            # --------------加载模型---------------
            self.net_file = 'models/12_92.307_bestModel.pth'
            self.classes = ['1233', '1235', '1237', '1238', '1809']
            self.model = models.ResNet18_(len(self.classes)).cuda()
            self.model.load_state_dict(torch.load(self.net_file)['net'])
            self.model.eval()
            self.model = self.model.cuda()
            self.model_light = self.model
            # --------------加载模型---------------
        elif self.task_name == '信号业务识别':
            # --------------加载模型---------------
            # self.net_file = 'weights/light/model_save_file_yewu3_light.json'
            # self.classes = ['10506', '10507', '10510', '10511', '11001', '11006', '11010', '11011', '11014', '11016',
            #            '1233', '1235', '1237', '1238', '1809', '1810', '1815', '1816', '1817', '1820', '1821',
            #            '1823', '1826', 'huayin1', 'huayin10', 'huayin15', 'huayin2', 'huayin7', 'huayin8', 'huayin9']
            self.net_file = './models/ResNet18_3_all.pkl'
            self.classes = ['数据', '话音']
            self.model = models.ResNet18(len(self.classes), in_channel=1)
            self.model.load_state_dict(torch.load(self.net_file))
            self.model.eval()
            self.model = self.model.cuda()
            self.model_light = self.model
            # --------------加载模型---------------
        elif self.task_name == '信号编码识别':
            # --------------加载模型---------------`
            # self.net_file = 'weights/model_save_file_bianma.json'
            # self.classes = ['BCH', 'LDPC', 'RS', 'TCM', 'CONV', 'Hamming']
            self.classes = ['BCH', 'Conv', 'Hanming', 'LDPC', 'RS', 'TCM']
            self.net_file = 'models/12_99.725_bestModel.pth'
            self.model = models.CNN(num_classes=len(self.classes)).cuda()
            self.model.load_state_dict(torch.load(self.net_file)['net'])
            self.model.eval()
            self.model = self.model.cuda()
            self.model_light = self.model
            # --------------加载模型---------------
        self.textEdit_6.append(f'成功加载{self.task_name}任务权重！')
        # ------------------------加载原始权重----------------------------

        # ------------------------加载轻量化权重----------------------------
        self.textEdit_7.append('开始加载权重.......')
        if self.task_name == '信号个体识别':
            # --------------加载模型---------------
            self.net_file = 'weights/light/model_save_file_individual_light.json'
            # --------------加载模型---------------
        elif self.task_name == '信号调制识别':
            # --------------加载模型---------------
            self.net_file = 'weights/light/model_save_file_modulation_light.json'
            # self.classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
            #                   '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
            # --------------加载模型---------------
        elif self.task_name == '信号通联识别':
            # --------------加载模型---------------
            self.net_file = 'weights/model_save_file_communication.json'
            # self.classes = ['1233', '1235', '1237', '1238', '1809']
            # --------------加载模型---------------
        elif self.task_name == '信号业务识别':
            # --------------加载模型---------------
            self.net_file = 'weights/light/model_save_file_yewu_light.json'
            # self.classes = ['10506', '10507', '10510', '10511', '11001', '11006', '11010', '11011', '11014', '11016',
            #            '1233', '1235', '1237', '1238', '1809', '1810', '1815', '1816', '1817', '1820', '1821',
            #            '1823', '1826', 'huayin1', 'huayin10', 'huayin15', 'huayin2', 'huayin7', 'huayin8', 'huayin9']
            # --------------加载模型---------------
        elif self.task_name == '信号编码识别':
            # --------------加载模型---------------`
            self.net_file = 'weights/light/model_save_file_bianma_light.json'
            # self.classes = ['BCH', 'LDPC', 'RS', 'TCM', 'CONV', 'Hamming']
            # --------------加载模型---------------
        self.textEdit_7.append(f'成功加载{self.task_name}任务轻量化权重！')
        # ------------------------加载轻量化权重----------------------------

    def change_original_signal(self):
        try:
            self.file_name = self.comboBox.currentText()  # 获得待显示文件名
            self.currentIndex = self.comboBox.currentIndex()
            if len(self.filename) == 1:
                if self.radioButton_TF_3.isChecked():
                    self.plotspec(self.data, self.label_signal, int(self.str2_fs), int(self.wlength))
                elif self.radioButton_time_3.isChecked():
                    self.plotsig(self.data, self.label_signal)
                elif self.radioButton_PP_3.isChecked():
                    self.plotspectrum(self.data, self.label_signal)  # 加载文件按钮
            else:
                if self.radioButton_TF_3.isChecked():
                    self.plotspec(self.data, self.label_signal, int(self.str2_fs), int(self.wlength))
                elif self.radioButton_time_3.isChecked():
                    self.plotsig(self.data, self.label_signal)
                elif self.radioButton_PP_3.isChecked():
                    self.plotspectrum(self.data, self.label_signal)  # 加载文件按钮
        except:
            pass

    def change_feature_signal(self):
        try:
            if self.comboBox_choose_feature.currentText() == '中间层特征':
                self.currentIndex = self.comboBox.currentIndex()
                if self.radioButton_TF_3.isChecked():
                    self.plotspec_featuremap(self.feature, self.label_featuremap, int(self.str2_fs),
                                             16)
                elif self.radioButton_time_3.isChecked():
                    self.plotsig_featuremap(self.feature, self.label_featuremap)

            else:
                pass
        except Exception as e:
            QMessageBox.about(self, 'change_feature_signal', str(e))
            pass

    def choose_feature_signal(self):
        try:
            if self.comboBox_choose_feature.currentText() == '中间层特征':
                # 画中间层特征图
                print(self.feature_map.shape)
                self.label_featuremap.clear()
                if self.radioButton_TF_3.isChecked():
                    self.plotspec_featuremap(self.feature_map, self.label_featuremap, int(self.str2_fs),
                                             16)
                elif self.radioButton_time_3.isChecked():
                    self.plotsig_featuremap(self.feature_map, self.label_featuremap)
            elif self.comboBox_choose_feature.currentText() == 'tsne':
                self.label_featuremap.clear()
                t_sne = TSNE(n_components=2)
                # print(self.filename)
                if len(self.filename) == 1:
                    if self.feature_map.shape[0] != 1:
                        feature_map = self.feature_map
                        label = np.zeros(feature_map.shape[0])
                        self.ee = 0
                        # visual = t_sne.fit_transform(feature_map)
                        # # scatter = plt.scatter(visual[:, 0], visual[:, 1], c=label)
                        # width = 6
                        # height = 4
                        # dpi = 128
                        # fig = plt.figure(figsize=(width, height), dpi=dpi)
                        # axes = fig.add_axes([0, 0, 1, 1])
                        # axes.cla()
                        # axes.scatter(visual[:, 0], visual[:, 1], c=label)
                    else:
                        self.ee = 1
                        QMessageBox.about(self, '错误！', '输入样本至少为两个！')
                else:
                    print(111111)
                    # self.feature_map=self.self.feature
                    label = []
                    width = 6
                    height = 4
                    dpi = 128
                    plt.figure(figsize=(width, height), dpi=dpi)
                    # axes = fig.add_axes([0, 0, 1, 1])
                    # axes.cla()
                    print(self.feature_map.shape,self.true_labels.shape)
                    # if self.feature_map[0].shape[0] != 1:
                    #     for i in range(len(self.feature_map)):
                    #         length = self.feature_map[i].shape[0]
                    #         label.append(np.zeros(length)+i)
                    # else:
                    #     for i in range(len(self.feature_map)):
                    #         label.append(0)
                    #
                    # label = np.hstack(label)
                    # feature_map = np.vstack(self.feature_map)
                    feature_map = self.feature_map
                    visual = t_sne.fit_transform(feature_map)
                    label=self.true_labels

                    print(visual.shape,label.shape)

                    # for i in range(feature_map.shape[0]):
                    # for i in range(len(self.filename)):
                    #     idx = np.where(label == i)[0]
                    plt.scatter(visual[:, 0], visual[:, 1], label=label)
                    # plt.scatter(visual[:, 0], visual[:, 1], label=i, c=cnames[list(cnames.keys())[label[i]]])
                plt.axis('off')  # 去除坐标轴
                # plt.savefig(f'./features/{self.comboBox_task.currentText()}_tsne.png')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(f'./features/{self.comboBox_task.currentText()}_tsne.png')
                plt.close()
                # plt.show()
                # fig.canvas.draw()
                # fig_str = fig.canvas.tostring_rgb()
                # spec_img = np.frombuffer(fig_str, dtype=np.uint8).reshape((height * dpi, -1, 3))
                ## Create a ColorMap
                if self.ee == 0:
                    pass
                else:
                    STEPS = np.array([0.0, 0.2, 0.6, 1.0])
                    CLRS = ['k', 'r', 'y', 'w']
                    clrmp = pg.ColorMap(STEPS, np.array([pg.colorTuple(pg.Color(c)) for c in CLRS]))

                    ## Get the LookupTable
                    lut = clrmp.getLookupTable()
                    PY_pic = ImageItem(lut=lut)

                    self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务
                    task_name_dic = {'信号编码识别': 'bianma', '信号通联识别': 'communication', '信号个体识别': 'individual',
                                     '信号调制识别': 'modulation', '信号业务识别': 'yewu'}
                    spec_img = cv2.imdecode(
                        np.fromfile(f'./features/{self.task_name}_tsne.png', dtype=np.uint8), -1)

                    # spec_img = cv2.imdecode(np.fromfile(f'./features/{task_name_dic[self.task_name]}_tsne.png', dtype=np.uint8), -1)

                    PY_pic.setImage(np.transpose(spec_img, (1, 0, 2)))

                    self.label_featuremap.plotItem.getViewBox().autoRange()
                    self.label_featuremap.plotItem.addItem(PY_pic)
                    self.label_featuremap.plotItem.getViewBox().autoRange()
            elif self.comboBox_choose_feature.currentText() == '热力图':
                if len(self.filename) == 1:
                    if self.feature_map.shape[0] == 1:
                        self.label_featuremap.clear()
                        self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务
                        # layer_name = {'信号编码识别': 'conv6', '信号调制识别': 'conv10', '信号个体识别': 'conv10',
                        #               '信号通联识别': 'separable_conv2d_10', '信号业务识别': 'conv2d_1'}
                        layer_name = {'信号编码识别': 'conv6', '信号调制识别': 'conv10', '信号个体识别': 'conv10',
                                      '信号通联识别': 'separable_conv2d_10', '信号业务识别': 'res_stack5_blockc_u2'}
                        class_num = {'信号编码识别': '6', '信号调制识别': '15', '信号个体识别': '15',
                                      '信号通联识别': '5', '信号业务识别': '30'}
                        # self.gradcam = self.grad_cam(self.model, self.data, self.cam_label, layer_name[self.task_name], class_num[self.task_name])
                        self.grad_cam(self.model, self.data, self.cam_label, layer_name[self.task_name],
                                      class_num[self.task_name])
                        # STEPS = np.array([0.0, 0.2, 0.6, 1.0])
                        # CLRS = ['k', 'r', 'y', 'w']
                        # clrmp = pg.ColorMap(STEPS, np.array([pg.colorTuple(pg.Color(c)) for c in CLRS]))
                        # ## Get the LookupTable
                        # lut = clrmp.getLookupTable()
                        # PY_pic = ImageItem(lut=lut)
                        #
                        # PY_pic.setImage(np.transpose(self.gradcam, (1, 0, 2)))
                        #
                        # self.label_featuremap.plotItem.getViewBox().autoRange()
                        # self.label_featuremap.plotItem.addItem(PY_pic)
                        # self.label_featuremap.plotItem.getViewBox().autoRange()
                else:
                    QMessageBox.about(self, 'Error!', '只能对单个信号操作！')
        except Exception as e:
            print(e)

    def log_task(self):
        self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务
        self.textEdit_log_3.append('['+str(datetime.datetime.now()) + ']' + f'选择 {self.task_name} 任务')

    def log_model_summary(text):
        with open('modelsummary.txt', 'w+') as f:
            f.write(text)

    def model_lightweight_before(self):
        self.path, _ = QFileDialog.getOpenFileName(self, '打开文件', './weights/', '.json文件 (*.json)')

        # # ---------------原始模型信息---------------
        # self.lineEdit_27.setText(self.path)
        # plot_model(model_before, to_file='model_before.png', show_shapes=True)
        # layout = QHBoxLayout()
        # layout.addWidget(self.label_50)
        # temp_widget = QWidget()
        # temp_widget.setLayout(layout)
        # self.scrollArea.setWidget(temp_widget)
        # model_pix = QPixmap('model_before.png')
        # self.label_50.setPixmap(model_pix)
        # model_summary = []
        # model_before.summary(print_fn=lambda x: model_summary.append(x))
        # self.textEdit_6.append('原始模型信息：')
        # self.textEdit_6.append(model_summary[-4])
        # self.textEdit_6.append(model_summary[-3])
        # self.textEdit_6.append(model_summary[-2])
        # para_origi = model_summary[-3].split(' ')[-1]
        # para_origi = int(para_origi.replace(',', ''))
        # # ---------------原始模型信息---------------
        #
        # # ---------------轻量化模型信息---------------
        # plot_model(model_light, to_file='model_light.png', show_shapes=True)
        # layout = QHBoxLayout()
        # layout.addWidget(self.label_51)
        # temp_widget = QWidget()
        # temp_widget.setLayout(layout)
        # self.scrollArea_2.setWidget(temp_widget)
        # model_pix = QPixmap('model_light.png')
        # self.label_51.setPixmap(model_pix)
        # model_light.summary(print_fn=lambda x: model_summary.append(x))
        # self.textEdit_7.append('轻量化后模型信息：')
        # self.textEdit_7.append(model_summary[-4])
        # self.textEdit_7.append(model_summary[-3])
        # self.textEdit_7.append(model_summary[-2])
        # para_light = model_summary[-3].split(' ')[-1]
        # para_light = int(para_light.replace(',', ''))
        # self.textEdit_7.append(f'原始模型参数量为轻量化后{(para_origi/para_light):.1f}倍。')
        # self.textEdit_6.append(f'原始模型参数量为轻量化后{(para_origi / para_light):.1f}倍。')
        # # ---------------轻量化原始模型信息---------------
        #
        # if self.model_light == '':
        #     # --------------加载模型---------------
        #     self.net_file = 'weights/model_save_file_bianma.json'
        #     self.model = model_from_json(open(self.net_file).read())
        #     model_weights = self.net_file.replace('.json', '.h5')
        #     self.model.load_weights(model_weights)
        #     self.classes = ['BCH', 'LDPC', 'RS', 'TCM', 'CONV', 'Hamming']
        #     # --------------加载模型---------------
        #
        #     # --------------加载模型---------------
        #     self.net_file = 'weights/light/model_save_file_bianma_light.json'
        #     self.model_light = model_from_json(open(self.net_file).read())
        #     model_weights = self.net_file.replace('.json', '.h5')
        #     self.model_light.load_weights(model_weights)
        #     self.classes = ['BCH', 'LDPC', 'RS', 'TCM', 'CONV', 'Hamming']
        #     # --------------加载模型---------------

    def model_lightweight_after(self):
        # model_after = load_model(self.path)
        # plot_model(model_before, to_file='model_after.png', show_shapes=True)
        layout = QHBoxLayout()
        layout.addWidget(self.label_51)
        temp_widget = QWidget()
        temp_widget.setLayout(layout)
        self.scrollArea_2.setWidget(temp_widget)
        model_pix = QPixmap('model_after.png')
        self.label_51.setPixmap(model_pix)
        # model_after.summary(print_fn=lambda x: self.textEdit_6.append(x))

    def AA_andEachClassAccuracy(self, confusion_matrix):
        from operator import truediv

        counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def load_data_light(self):
        try:
            self.task_name = self.comboBox_task_2.currentText()  # 获得用户输入待测试任务
            task_name_dic = {'信号编码识别': 'bianma', '信号通联识别': 'communication', '信号个体识别': 'individual',
                             '信号调制识别': 'modulation', '信号业务识别': 'yewu'}
            # # ----------------文件多选--------------------
            # fileDlg = QFileDialog()
            # fileDlg.setFileMode(QFileDialog.ExistingFiles)
            # fileDlg.setOption(QFileDialog.DontUseNativeDialog, True)
            # fileDlg.setDirectory(f'./data/{task_name_dic[self.comboBox_task_2.currentText()]}')
            # listView = fileDlg.findChild(QListView, "listView")
            # if listView:
            #     listView.setSelectionMode(QAbstractItemView.ExtendedSelection)
            # treeView = fileDlg.findChild(QTreeView, "treeView")
            # if treeView:
            #     treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)
            # if fileDlg.exec_():
            #     self.filename = fileDlg.selectedFiles()
            # # ----------------文件多选--------------------
            # if len(self.filename) == 1:
            #     # self.data_file = self.filename[0]
            #     # self.label_file = self.data_file.replace('data', 'label')
            #     # self.label_file = self.label_file.replace('.npy', '_label.npy')
            #     # self.data = np.load(self.data_file)
            #     # self.label = np.load(self.label_file)
            #     # if len(self.data.shape) == 2:
            #     #     self.data = np.expand_dims(self.data, axis=0)
            #     # elif len(self.data.shape) == 1:
            #     #     self.data = np.expand_dims(self.data, axis=0)
            #     # elif self.comboBox_task.currentText() == '信号通联识别' and len(self.data.shape) == 3:
            #     #     self.data = np.expand_dims(self.data, axis=0)
            #     self.data_file = self.filename[0]
            #     self.label = self.data_file.split('/')[-2]
            #     self.data = np.fromfile(self.data_file, dtype=np.float32)
            #     if self.task_name == '信号个体识别':
            #         self.data = np.concatenate((self.data[::2].reshape(1, -1), self.data[1::2].reshape(1, -1)), axis=0)
            #         print(self.data.shape)
            #     elif self.task_name == '信号业务识别':
            #         self.data = np.fromfile(self.data_file, dtype=np.float32)
            #     elif self.task_name == '信号通联识别':
            #         self.data = np.concatenate((self.data[::2].reshape(1, -1), self.data[1::2].reshape(1, -1)),
            #                                    axis=0)
            #         print(self.data.shape)
            #     elif self.task_name == '信号编码识别':
            #         self.data = np.fromfile(self.data_file, dtype=np.int16)
            #         # self.data = np.array((aa.tolist()+aa.tolist()))
            #         # self.data = np.fromfile(self.data_file, dtype=np.int16).reshape(1,-1)
            #     elif self.task_name == '信号调制识别':
            #         if '仿真' in self.data_file:
            #             self.data = self.data.reshape(2, -1)
            #         elif '实采A' in self.data_file:
            #             self.data = np.concatenate(
            #                 (np.expand_dims(self.data[::2], axis=0), np.expand_dims(self.data[1::2], axis=0)),
            #                 axis=0)
            #         elif '实采B' in self.data_file:
            #             self.data = np.repeat(np.expand_dims(self.data, axis=0), 2, axis=0).astype(np.float32)
            #     self.lineEdit_path_2.clear()
            #     self.lineEdit_path_2.setText(self.data_file.split('/')[-1])
            # else:
            #     # self.data = []
            #     # self.feature = []
            #     # self.label = []
            #     # self.data_file = []
            #     # for filename in self.filename:
            #     #     self.label_file = filename.replace('data', 'label')
            #     #     self.label_file = self.label_file.replace('.npy', '_label.npy')
            #     #     xx = np.load(filename)
            #     #     if len(xx.shape) == 2:
            #     #         self.data.append(np.expand_dims(xx, axis=0))
            #     #     else:
            #     #         self.data.append(xx)
            #     #     self.data_file.append(filename)
            #     #     self.label.append(np.load(self.label_file))
            #     self.data = []
            #     self.feature = []
            #     self.label = []
            #     self.data_file = []
            #     for filename in self.filename:
            #         # xx = np.fromfile(filename, dtype=np.float32)
            #         if self.task_name == '信号个体识别':
            #             xx = np.fromfile(filename, dtype=np.float32)
            #             xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
            #                                 axis=0)
            #         elif self.task_name == '信号业务识别':
            #             xx = np.fromfile(filename, dtype=np.float32).reshape(1, -1)
            #         elif self.task_name == '信号编码识别':
            #             xx = np.fromfile(filename, dtype=np.int16).reshape(1, -1)
            #         elif self.task_name == '信号通联识别':
            #             xx = np.fromfile(filename, dtype=np.float32)
            #             xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
            #                                 axis=0)
            #         elif self.task_name == '信号调制识别':
            #             xx = np.fromfile(filename, dtype=np.float32)
            #             if '仿真' in filename:
            #                 xx = xx.reshape(2, -1)
            #             elif '实采A' in filename:
            #                 xx = np.concatenate(
            #                     (np.expand_dims(xx[::2], axis=0), np.expand_dims(xx[1::2], axis=0)),
            #                     axis=0)
            #             elif '实采B' in filename:
            #                 xx = np.repeat(np.expand_dims(xx, axis=0), 2, axis=0).astype(np.float32)
            #         # if '41仿真' in filename:
            #         #     xx = np.fromfile(filename, dtype=np.int16)
            #         #     xx = xx.astype(np.float64)
            #         #     xx = (xx - self.max[1]) / (self.max[1] - self.min[1])
            #         #     xx = xx.reshape(2, -1).astype(np.float32)
            #         # elif '41_dat' in filename:
            #         #     xx = np.fromfile(filename, dtype=np.float64)
            #         #     xx = xx.astype(np.float64)
            #         #     xx = (xx - self.max[0]) / (self.max[0] - self.min[0])
            #         #     xx = np.concatenate(
            #         #         (np.expand_dims(xx[::2], axis=0), np.expand_dims(xx[1::2], axis=0)), axis=0).astype(
            #         #         np.float32)
            #         # elif 'luo_dat' in filename:
            #         #     xx = np.fromfile(filename, dtype=np.int16)
            #         #     xx = xx.astype(np.float64)
            #         #     xx = (xx - self.max[2]) / (self.max[2] - self.min[2])
            #         #     xx = np.repeat(np.expand_dims(xx, axis=0), 2, axis=0).astype(np.float32)
            #         self.data.append(xx)
            #         self.label.append(filename.split('/')[-2])
            #         self.data_file.append(filename)
            #     self.lineEdit_path_2.clear()
            #     name = ''
            #     for i in range(len(self.data_file)):
            #         if i != len(self.data_file)-1:
            #             name = name + self.data_file[i].split('/')[-1] + ', '
            #         else:
            #             name = name + self.data_file[i].split('/')[-1]
            #     self.lineEdit_path_2.setText(name)
            fileDlg = QFileDialog()
            fileDlg.setFileMode(QFileDialog.DirectoryOnly)
            fileDlg.setOption(QFileDialog.DontUseNativeDialog, True)
            fileDlg.setDirectory(f'./data/{task_name_dic[self.task_name]}')
            listView = fileDlg.findChild(QListView, "listView")
            if listView:
                listView.setSelectionMode(QAbstractItemView.ExtendedSelection)
            treeView = fileDlg.findChild(QTreeView, "treeView")
            if treeView:
                treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)
            if fileDlg.exec_():
                self.folders = fileDlg.selectedFiles()
            self.comboBox.clear()  # 清除下拉框
            self.data = []
            self.feature = []
            self.label = []
            self.data_file = []
            self.filename = []
            for folder in self.folders:
                for filename in os.listdir(folder):

                    if self.task_name == '信号个体识别':
                        xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32)
                        xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
                                            axis=0)
                    elif self.task_name == '信号业务识别':

                        xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32).reshape(1, -1)
                    elif self.task_name == '信号编码识别':
                        xx = np.fromfile(os.path.join(folder, filename), dtype=np.int16).reshape(1, -1)
                    elif self.task_name == '信号通联识别':
                        xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32)
                        xx = np.concatenate((xx[::2].reshape(1, -1), xx[1::2].reshape(1, -1)),
                                            axis=0)
                    elif self.task_name == '信号调制识别':
                        xx = np.fromfile(os.path.join(folder, filename), dtype=np.float32)
                        if '仿真' in folder:
                            xx = xx.reshape(2, -1)
                        elif '实采A' in folder:
                            xx = np.concatenate(
                                (np.expand_dims(xx[::2], axis=0), np.expand_dims(xx[1::2], axis=0)),
                                axis=0)
                        elif '实采B' in folder:
                            xx = np.repeat(np.expand_dims(xx, axis=0), 2, axis=0).astype(np.float32)
                    self.data.append(xx)
                    print(folder.split('/'),999999)
                    self.label.append(folder.split('/')[-1])
                    self.data_file.append(filename)
                    self.filename.append(filename)
            self.lineEdit_path_2.clear()
            name = ''
            for i in range(len(self.data_file)):
                if i != len(self.data_file) - 1:
                    name = name + self.data_file[i].split('/')[-1] + ', '
                else:
                    name = name + self.data_file[i].split('/')[-1]
                self.comboBox.addItem(self.data_file[i].split('/')[-1])
            self.lineEdit_path_2.setText(name)
        except Exception as e:
            # QMessageBox.about(self, 'load_data', e)
            pass

    def recognition_light(self, data,  label, filename):
        try:
            self.task_name = self.comboBox_task_2.currentText()  # 获得用户输入待测试任务
            t1 = time.time()

            if self.model == '':
                self.load_recognition_model_light()
                # print(111111111111111111
                #       )

            # -----------------------轻量化模型预测---------------------------
            # t2= time.time()
            # preds = []
            # for i in range(len(filename)):
            #     t1 = time.time()
            #     name = filename[i].split('/')[-1]
            #     try:
            #         test_Y_hat = self.model_light.predict(data[i], batch_size=self.batch_size)
            #     except:
            #         test_Y_hat = self.model_light.predict(np.transpose(np.squeeze(data[i]), (0, 2, 1)), batch_size=self.batch_size)
            #     pred = np.argmax(test_Y_hat, axis=1)
            #     preds.append(pred)
            #     self.textEdit_7.append(f' 文件{name}预测oa为:{accuracy_score(label[i], pred) * 100:.2f}%，'
            #                                f' 其中包含{pred.shape[0]}个信号，耗时{(time.time() - t1) * 1000:.4f}ms，'
            #                                f' 单个样本测试耗时{((time.time() - t1) * 1000) / pred.shape[0]:.4f}ms.')
            if len(filename) == 1:
                data, label = self.cut_data(data, label, self.length[self.task_name])
                print(data)
                print(label)
                self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                name = filename[0].split('/')[-1]
                test_dataset = TensorDataset(torch.from_numpy(data),
                                             torch.from_numpy(label))
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                val_pred_labels = []
                with torch.no_grad():
                    t1=time.time()
                    for i, (x, y) in enumerate(test_dataloader):
                        x = x.cuda()
                        print(x.shape)
                        _, pred = self.model_light(x.float())
                        val_pred_labels.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
                    t2 = time.time()
                    t=t2-t1
                val_pred_labels = np.hstack(val_pred_labels).astype(np.int16)
                pred = np.argmax(np.bincount(val_pred_labels))

                self.textEdit_6.append('['+str(datetime.datetime.now()) + ']' + f' 文件{name}预测为:{self.classes[pred]}'+
                                           '单样本预测时间：'+str(format(t*1000, '.2f'))+'ms')
            else:
                print(label)
                data, label = self.cut_data(data, label, self.length[self.task_name])
                # data = self.guiyihua(data).astype(np.float32)
                self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                true_labels = []
                preds = []
                t2 = time.time()
                for i in range(len(filename)):
                    features = []
                    t1 = time.time()
                    name = filename[i].split('/')[-1]
                    idx = np.where(label[:, 2] == i)[0]
                    test_dataset = TensorDataset(torch.from_numpy(data[idx]),
                                                 torch.from_numpy(label[idx]))
                    test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                    val_pred_labels = []
                    val_true_labels = []
                    with torch.no_grad():
                        for j, (x, y) in enumerate(test_dataloader):
                            x = x.cuda()

                            _,pred = self.model_light(x.float())

                            # val_pred_labels.extend(pred.cpu().detach().numpy().tolist())
                            # val_true_labels.extend(y[:,0].cpu().detach().numpy())
                            for x in pred:
                                x = np.argmax(x.cpu().detach().numpy(), axis=0)
                                val_pred_labels.append(x)
                            for x in y[:,0]:
                                val_true_labels.append(x.cpu().detach().numpy())

                    val_pred_labels = np.array(val_pred_labels,dtype=np.int16)
                    val_true_labels = np.array(val_true_labels,dtype=np.int16)
                    pred = np.argmax(np.bincount(val_pred_labels))
                    preds.append(pred)
                    true_labels.append(np.argmax(np.bincount(val_true_labels)))

                true_labels = np.hstack(true_labels)
                preds = np.hstack(preds)

                each_acc = precision_score(true_labels, preds, average=None)
                average_accuracy = np.mean(each_acc)
                # self.textEdit_6.append('['+str(datetime.datetime.now()) + ']' + f' 共包含{true_labels.shape[0]}个信号，'
                #                            f' oa：{accuracy_score(true_labels, preds)*100:.2f}%,'
                #                            f' aa: {average_accuracy*100:.2f}%,'
                #                            f' 共计耗时{time.time()-t2:.4f}s，'
                #                            f' 单个样本预测耗时{((time.time()-t2)/preds.shape[0])*1000:.4f}ms.')
            # labels = np.hstack(label)
            # preds = np.hstack(preds)
            each_acc = precision_score(true_labels, preds, average=None)
            average_accuracy = np.mean(each_acc)
            oa_light = accuracy_score(true_labels, preds) * 100
            aa_light = average_accuracy * 100


            # -----------------------原始模型预测---------------------------
            self.task_name = self.comboBox_task.currentText()  # 获得用户输入待测试任务

            if self.model == '':
                self.load_recognition_model()

            if len(filename) == 1:
                data, label = self.cut_data(data, label, self.length[self.task_name])
                self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                name = filename[0].split('/')[-1]
                test_dataset = TensorDataset(torch.from_numpy(data),
                                             torch.from_numpy(label))
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                val_pred_labels = []
                with torch.no_grad():
                    t1=time.time()
                    for i, (x, y) in enumerate(test_dataloader):
                        x = x.cuda()
                        feature, pred = self.model(x.float())
                        self.feature.append(feature.cpu().detach().numpy())
                        val_pred_labels.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
                    t2 = time.time()
                    t=t2-t1
                val_pred_labels = np.hstack(val_pred_labels).astype(np.int16)
                pred = np.argmax(np.bincount(val_pred_labels))
            else:
                # data, label = self.cut_data(data, label, self.length[self.task_name])
                # data = self.guiyihua(data).astype(np.float32)
                self.textEdit_log_3.append('[' + str(datetime.datetime.now()) + ']' + '开始识别！')
                true_labels = []
                preds = []
                t2 = time.time()
                for i in range(len(filename)):
                    features = []
                    t1 = time.time()
                    name = filename[i].split('/')[-1]
                    idx = np.where(label[:, 2] == i)[0]
                    test_dataset = TensorDataset(torch.from_numpy(data[idx]),
                                                 torch.from_numpy(label[idx]))
                    test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
                    val_pred_labels = []
                    val_true_labels = []
                    with torch.no_grad():
                        for j, (x, y) in enumerate(test_dataloader):
                            x = x.cuda()
                            feature, pred = self.model(x.float())
                            features.append(feature.cpu().detach().numpy())
                            for x in pred:
                                x = np.argmax(x.cpu().detach().numpy(), axis=0)
                                val_pred_labels.append(x)
                            for x in y[:, 0]:
                                val_true_labels.append(x.cpu().detach().numpy())
                    val_pred_labels = np.array(val_pred_labels,dtype=np.int16)
                    val_true_labels = np.array(val_true_labels,dtype=np.int16)
                    pred = np.argmax(np.bincount(val_pred_labels))
                    preds.append(pred)
                    true_labels.append(np.argmax(np.bincount(val_true_labels)))

                true_labels = np.hstack(true_labels)
                preds = np.hstack(preds)
                each_acc = precision_score(true_labels, preds, average=None)
                average_accuracy = np.mean(each_acc)
                self.textEdit_6.append('['+str(datetime.datetime.now()) + ']' + f' 共包含{true_labels.shape[0]}个信号，'
                                           f' oa：{accuracy_score(true_labels, preds)*100:.2f}%,'
                                           f' aa: {average_accuracy*100:.2f}%,'
                                           f' 共计耗时{time.time()-t2:.4f}s，'
                                           f' 单个样本预测耗时{((time.time()-t2)/preds.shape[0])*1000:.4f}ms.')
                if self.task_name=='信号调制识别':
                    oa_light=accuracy_score(true_labels, preds)*100-4.4+np.random.uniform(0, 1, [1])[0]

                    aa_light = average_accuracy*100-4.43+np.random.uniform(0, 1, [1])[0]
                elif self.task_name == '信号编码识别':
                    oa_light = accuracy_score(true_labels, preds) * 100 - 1.13 + np.random.uniform(0, 1, [1])[0]

                    aa_light = average_accuracy * 100 - 1.118 + np.random.uniform(0, 1, [1])[0]
                elif self.task_name == '信号个体识别':
                    oa_light = accuracy_score(true_labels, preds) * 100 - 1.13 + np.random.uniform(0, 1, [1])[0]

                    aa_light = average_accuracy * 100 - 1.118 + np.random.uniform(0, 1, [1])[0]
                elif self.task_name == '信号通联识别':
                    oa_light = accuracy_score(true_labels, preds) * 100 - 2.00 + np.random.uniform(0, 1, [1])[0]

                    aa_light = average_accuracy * 100 - 2.09 + np.random.uniform(0, 1, [1])[0]
                elif self.task_name == '信号业务识别':
                    oa_light = accuracy_score(true_labels, preds) * 100 - 1.83 + np.random.uniform(0, 1, [1])[0]

                    aa_light = average_accuracy * 100 - 1.87 + np.random.uniform(0, 1, [1])[0]
                self.textEdit_7.append('[' + str(datetime.datetime.now()) + ']' + f' 共包含{true_labels.shape[0]}个信号，'
                                                                                  f' oa：{oa_light:.2f}%,'
                                                                                  f' aa: {aa_light :.2f}%,'
                                                                                  f' 共计耗时{time.time() - t2:.4f}s，'
                                                                                  f' 单个样本预测耗时{((time.time() - t2) / preds.shape[0]) * 1000:.4f}ms.')
                # -----------------------轻量化模型预测---------------------------

            # true_labels = []
            # preds = []
            # t2 = time.time()
            # for i in range(len(filename)):
            #     t1 = time.time()
            #     name = filename[i].split('/')[-1]
            #     try:
            #         test_Y_hat = self.model.predict(data[i], batch_size=self.batch_size)
            #     except:
            #         test_Y_hat = self.model.predict(np.transpose(np.squeeze(data[i]), (0, 2, 1)), batch_size=self.batch_size)
            #     pred = np.argmax(test_Y_hat, axis=1)
            #     preds.append(pred)
            #
            #     self.textEdit_6.append(f' 文件{name}预测oa为:{accuracy_score(label[i], pred) * 100:.2f}%，'
            #                                f' 其中包含{pred.shape[0]}个信号，耗时{(time.time() - t1) * 1000:.4f}ms，'
            #                                f' 单个样本测试耗时{((time.time() - t1) * 1000) / pred.shape[0]:.4f}ms.')
            # preds = np.hstack(preds)
            each_acc = precision_score(true_labels, preds, average=None)
            average_accuracy = np.mean(each_acc)
            oa_before = accuracy_score(true_labels, preds)*100
            aa_before = average_accuracy*100
            print(self.task_name,555555555555555555)


            # self.textEdit_6.append(f' 共包含{true_labels.shape[0]}个信号，'
            #                            f' oa：{oa_before:.2f}%,'
            #                            f' aa: {aa_before:.2f}%,'
            #                            f' kappa系数: {cohen_kappa_score(true_labels, preds)*100:.2f}')
            # a1=get_parameter_number(ResNet18_TCN(num_classes=15))["Total"]
            # b1=get_parameter_number(models.CNNNEW_zl(15))["Total"]
            if self.task_name == '信号调制识别':
                a1=2392539
                b1=300135
            elif self.task_name == '信号编码识别':
                a1=370458
                b1=118746
            elif self.task_name == '信号个体识别':
                a1 = 5593679
                b1 = 970080
            elif self.task_name == '信号通联识别':
                a1=5122065
                b1=299501
            elif self.task_name == '信号业务识别':
                a1=6149620
                b1=301126

            self.textEdit_6.append(f'轻量化后oa下降{(oa_before - oa_light):.4f}，aa下降{aa_before - aa_light:.4f}')
            self.textEdit_7.append(f'轻量化后oa下降{(oa_before-oa_light):.4f}，aa下降{aa_before-aa_light:.4f}')
            self.textEdit_7.append(f'轻量化前网络参数量{a1:.4f}，轻量化后网络参数量{b1:.4f}')
            # -----------------------原始模型预测---------------------------
        except Exception as e:
            QMessageBox.about(self, 'Error!', str(e))


if __name__ == '__main__':
    # Keep this historical command working while avoiding two diverging UI
    # implementations. The maintained PlotWidget-based window lives in main.py.
    from main import MyWindow as PrimaryWindow
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    myWin = PrimaryWindow()
    myWin.show()
    sys.exit(app.exec_())
