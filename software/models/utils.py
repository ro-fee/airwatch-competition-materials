import math
import os

import pywt
import time
import wave
import models
import traceback

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(task, length, data_set, mod):
    '''
    :param task: 识别任务名称
    :return: 数据
    数据类型包括：.dat、.wav、.pcm
    '''
    import wave
    if task == 'Modulation':
        modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8psk',
                              '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
        if mod == 'test':
            modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
                                  '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
            if data_set != 'all':
                X_test, Y_test = [], []
                k = 0
                if data_set == '41fangzhen':
                    test_path = '../data/41仿真/test/'
                    X_test, Y_test = [], []
                    k = 0
                    for folder in os.listdir(test_path):
                        for filename in os.listdir(test_path + folder):
                            label = modulation_classes.index(folder)
                            snr = int(filename.split('=')[-1].split('.')[0])
                            data = np.fromfile(test_path + folder + '/' + filename, dtype=np.int16)
                            data = data.reshape(2, -1)
                            data = np.concatenate(
                                (np.expand_dims(data[0, :], axis=0), np.expand_dims(data[1, :], axis=0)),
                                axis=0)
                            x, y = cut_data(data, [label, snr, k], 1024)
                            X_test.append(x)
                            Y_test.append(y)
                            k += 1
                    X_test = guiyihua(np.vstack(X_test)).astype(np.float32)
                    Y_test = np.vstack(Y_test)
                elif data_set == '41':
                    data_set = '41_dat'
                    test_path = f'../data/{data_set}/test/'
                    for folder in os.listdir(test_path):
                        for filename in os.listdir(test_path + folder):
                            label = modulation_classes.index(folder)
                            snr = int(filename.split('=')[-1].split('_')[0])
                            data = np.fromfile(test_path + folder + '/' + filename, dtype=np.float64)
                            data = np.concatenate(
                                (np.expand_dims(data[::2], axis=0), np.expand_dims(data[1::2], axis=0)),
                                axis=0)
                            x, y = cut_data(data, [label, snr, k], 1024)
                            X_test.append(x)
                            Y_test.append(y)
                            k += 1
                    X_test = guiyihua(np.vstack(X_test)).astype(np.float32)
                    Y_test = np.vstack(Y_test)
                elif data_set == 'luo':
                    data_set = 'luo_dat'
                    test_path = f'../data/{data_set}/test/'
                    for folder in os.listdir(test_path):
                        for filename in os.listdir(test_path + folder):
                            label = modulation_classes.index(folder)
                            data = np.fromfile(test_path + folder + '/' + filename, dtype=np.int16)
                            x, y = cut_data(data, [label, -1, k], 1024)
                            X_test.append(np.repeat(np.expand_dims(x, axis=1), 2, axis=1))
                            Y_test.append(y)
                            k += 1
                    X_test = guiyihua(np.vstack(X_test)).astype(np.float32)
                    Y_test = np.vstack(Y_test)
            else:
                X_test_41, Y_test_41 = [], []
                k = 0
                modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK',
                                      '8PSK', '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
                test_path = f'./data/41_dat/test/'
                for folder in os.listdir(test_path):
                    for filename in os.listdir(test_path+folder):
                        label = modulation_classes.index(folder)
                        snr = int(filename.split('=')[-1].split('_')[0])
                        data = np.fromfile(test_path + folder + '/' + filename, dtype=np.float64)
                        data = np.concatenate((np.expand_dims(data[::2], axis=0), np.expand_dims(data[1::2], axis=0)),
                                              axis=0)
                        x, y = cut_data(data, [label, snr, k], 1024)
                        X_test_41.append(x)
                        Y_test_41.append(y)
                        k += 1
                    # print('41', folder, k)
                X_test_41 = guiyihua(np.vstack(X_test_41)).astype(np.float32)
                Y_test_41 = np.vstack(Y_test_41)


                X_test_41fz, Y_test_41fz = [], []
                modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK',
                                      '8PSK', '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
                test_path = f'./data/41仿真/test/'
                for folder in os.listdir(test_path):
                    for filename in os.listdir(test_path + folder):
                        label = modulation_classes.index(folder)
                        snr = int(filename.split('=')[-1].split('.')[0])
                        data = np.fromfile(test_path + folder + '/' + filename, dtype=np.int16)
                        data = data.reshape(2, -1)
                        data = np.concatenate((np.expand_dims(data[0, :], axis=0), np.expand_dims(data[1, :], axis=0)),
                                              axis=0)
                        x, y = cut_data(data, [label, snr, k], 1024)
                        X_test_41fz.append(x)
                        Y_test_41fz.append(y)
                        k += 1
                    # print('仿真', folder, k)
                X_test_41fz = guiyihua(np.vstack(X_test_41fz)).astype(np.float32)
                Y_test_41fz = np.vstack(Y_test_41fz)


                X_test_luo, Y_test_luo = [], []
                modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK',
                                      '8psk', '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
                test_path = f'./data/luo_dat/test/'
                for folder in os.listdir(test_path):
                    for filename in os.listdir(test_path+folder):
                        label = modulation_classes.index(folder)
                        data = np.fromfile(test_path + folder + '/' + filename, dtype=np.int16)
                        x, y = cut_data(data, [label, -1, k], 1024)
                        X_test_luo.append(np.repeat(np.expand_dims(x, axis=1), 2, axis=1))
                        Y_test_luo.append(y)
                        k += 1
                    # print('luo', folder, k)
                X_test_luo = guiyihua(np.vstack(X_test_luo)).astype(np.float32)
                Y_test_luo = np.vstack(Y_test_luo)


                X_test = np.concatenate((X_test_41, X_test_41fz, X_test_luo), axis=0)
                Y_test = np.concatenate((Y_test_41, Y_test_41fz, Y_test_luo), axis=0)
                del X_test_41, X_test_41fz, X_test_luo

            return X_test, Y_test
        else:
            if data_set == 'luo':
                modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8psk',
                                      '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
                train_path = './data/luo_dat/train/'
                test_path = '../data/luo_dat/test/'

                X_train, Y_train = [], []
                for folder in os.listdir(train_path):
                    for filename in os.listdir(train_path+folder):
                        label = modulation_classes.index(folder)
                        data = np.fromfile(train_path + folder + '/' + filename, dtype=np.int16)
                        if data.shape[-1] < 1024:
                            pass
                        else:
                            x, y = cut_data(data, label, 1024)
                            X_train.append(np.repeat(np.expand_dims(x, axis=1), 2, axis=1))
                            Y_train.append(y)
                X_train = guiyihua(np.vstack(X_train)).astype(np.float32)
                Y_train = np.hstack(Y_train)
                X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)

                X_test, Y_test = [], []
                for folder in os.listdir(test_path):
                    for filename in os.listdir(test_path+folder):
                        label = modulation_classes.index(folder)
                        data = np.fromfile(test_path + folder + '/' + filename, dtype=np.int16)
                        x, y = cut_data(data, label, 1024)
                        X_test.append(np.repeat(np.expand_dims(x, axis=1), 2, axis=1))
                        Y_test.append(y)
                X_test = guiyihua(np.vstack(X_test)).astype(np.float32)
                Y_test = np.hstack(Y_test)

                return X_train, X_val, X_test, y_train, y_val, Y_test
            elif data_set == '41':
                modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK',
                                      '8PSK', '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
                train_path = './data/41_dat/train/'
                X_train, Y_train = [], []
                for folder in os.listdir(train_path):
                    for filename in os.listdir(train_path+folder):
                        label = modulation_classes.index(folder)
                        snr = int(filename.split('=')[-1].split('_')[0])
                        data = np.fromfile(train_path + folder + '/' + filename, dtype=np.float64)
                        data = np.concatenate((np.expand_dims(data[::2], axis=0), np.expand_dims(data[1::2], axis=0)),
                                              axis=0)
                        if data.shape[-1] < 1024:
                            pass
                        else:
                            x, y = cut_data(data, [label, snr], 1024)
                            X_train.append(x)
                            Y_train.append(y)
                X_train = guiyihua(np.vstack(X_train)).astype(np.float32)
                Y_train = np.vstack(Y_train)
                X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)

                test_path = '../data/41_dat/test/'
                X_test, Y_test = [], []
                for folder in os.listdir(test_path):
                    for filename in os.listdir(test_path+folder):
                        label = modulation_classes.index(folder)
                        snr = int(filename.split('=')[-1].split('_')[0])
                        data = np.fromfile(test_path + folder + '/' + filename, dtype=np.float64)
                        data = np.concatenate((np.expand_dims(data[::2], axis=0), np.expand_dims(data[1::2], axis=0)),
                                              axis=0)
                        x, y = cut_data(data, [label, snr], 1024)
                        X_test.append(x)
                        Y_test.append(y)
                X_test = guiyihua(np.vstack(X_test)).astype(np.float32)
                Y_test = np.vstack(Y_test)

                return X_train, X_val, X_test, y_train, y_val, Y_test
            # elif data_set == '41':
            #     data = np.load('/media/yc/rushb/project/2022/wangjun/第二次验收/西电样本数据/41_data/x_train.npy')[:, :, :1024]
            #     Y = np.load('/media/yc/rushb/project/2022/wangjun/第二次验收/西电样本数据/41_data/y_train.npy').astype(np.int16)
            #     train_ratio = .8
            #
            #     X_train, X_val, y_train, y_val = train_test_split(data, Y, train_size=train_ratio)
            #     X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5)
            #
            #     return X_train, X_val, X_test, y_train, y_val, y_test
            elif data_set == '41fangzhen':
                modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK',
                                      '8PSK', '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
                train_path = '../data/41仿真/train/'
                X_train, Y_train = [], []
                for folder in os.listdir(train_path):
                    for filename in os.listdir(train_path + folder):
                        label = modulation_classes.index(folder)
                        snr = int(filename.split('=')[-1].split('.')[0])
                        data = np.fromfile(train_path + folder + '/' + filename, dtype=np.int16)
                        data = data.reshape(2, -1)
                        data = np.concatenate((np.expand_dims(data[0, :], axis=0), np.expand_dims(data[1, :], axis=0)),
                                              axis=0)
                        if data.shape[-1] < 1024:
                            pass
                        else:
                            x, y = cut_data(data, [label, snr], 1024)
                            X_train.append(x)
                            Y_train.append(y)
                X_train = guiyihua(np.vstack(X_train)).astype(np.float32)
                Y_train = np.vstack(Y_train)
                X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)

                test_path = '../data/41仿真/test/'
                X_test, Y_test = [], []
                for folder in os.listdir(test_path):
                    for filename in os.listdir(test_path + folder):
                        label = modulation_classes.index(folder)
                        snr = int(filename.split('=')[-1].split('.')[0])
                        data = np.fromfile(test_path + folder + '/' + filename, dtype=np.int16)
                        data = data.reshape(2, -1)
                        data = np.concatenate((np.expand_dims(data[0, :], axis=0), np.expand_dims(data[1, :], axis=0)),
                                              axis=0)
                        x, y = cut_data(data, [label, snr], 1024)
                        X_test.append(x)
                        Y_test.append(y)
                X_test = guiyihua(np.vstack(X_test)).astype(np.float32)
                Y_test = np.vstack(Y_test)

                return X_train, X_val, X_test, y_train, y_val, Y_test
            elif data_set == 'gnu':
                base_path = '/media/yc/rushb/project/2022/wangjun/第二次验收/西电样本数据/gnu_data'

                train_ratio = .6
                val_ratio = .2

                files = list(filter(lambda x: x.startswith('x'), os.listdir(base_path)))
                files.sort()
                train_data = []
                val_data = []
                test_data = []
                train_y = []
                val_y = []
                test_y = []
                for file in files:
                    data = np.load(os.path.join(base_path, file))
                    Y = np.load(os.path.join(base_path, 'y' + file[1:]))
                    label = Y[:, 0]
                    for c in np.unique(label):
                        idx = label == c
                        size = np.sum(idx)
                        train_idx = np.random.choice(range(size), size=int(size * train_ratio), replace=False)
                        val_idx = np.random.choice(list(set(range(size)) - set(train_idx)), size=int(size * val_ratio),
                                                   replace=False)
                        test_idx = list(set(range(size)) - set(train_idx) - set(val_idx))
                        train_data.append(data[idx][train_idx])

                        val_data.append(data[idx][val_idx])

                        test_data.append(data[idx][test_idx])

                        train_y.append(label[idx][train_idx])
                        val_y.append(label[idx][val_idx])
                        test_y.append(label[idx][test_idx])
                train_data = guiyihua(np.vstack(train_data)).astype(np.float32)
                val_data = guiyihua(np.vstack(val_data)).astype(np.float32)
                test_data = guiyihua(np.vstack(test_data)).astype(np.float32)
                train_y = np.hstack(train_y)
                val_y = np.hstack(val_y)
                test_y = np.hstack(test_y)

                return train_data, val_data, test_data, train_y, val_y, test_y
            elif data_set == 'all':
                X_train_gnu, X_val_gnu, X_test_gnu, y_train_gnu, y_val_gnu, y_test_gnu = load_data('Modulation',
                                                                                                         1024,
                                                                                                         '41fangzhen', 'wuhu')
                X_train_41, X_val_41, X_test_41, y_train_41, y_val_41, y_test_41 = load_data('Modulation',
                                                                                                   1024, '41',
                                                                                                   'wuhu')
                X_train_luo, X_val_luo, X_test_luo, y_train_luo, y_val_luo, y_test_luo = load_data('Modulation',
                                                                                                         1024,
                                                                                                         'luo', 'wuhu')
                # # 平衡类别数据量
                # idx1, idx2, idx3 = [], [], []
                # print('-------------平衡前-----------------')
                # for i in range(len(modulation_classes)):
                #     idx_gnu = np.where(y_train_gnu[:, 0] == i)[0]
                #     idx_41 = np.where(y_train_41[:, 0] == i)[0]
                #     idx_luo = np.where(y_train_luo == i)[0]
                #     print(modulation_classes[i], 'gnu:', idx_gnu.shape[0], '41:', idx_41.shape[0], 'luo:', idx_luo.shape[0])
                #     idx = np.array([idx_gnu.shape[0], idx_41.shape[0], idx_luo.shape[0]])
                #     length = min([idx[j] for j in np.where(idx != 0)[0]])
                #     print(length)
                #     if idx_gnu.shape[0] != 0:
                #         idx1.append(idx_gnu[:length])
                #     if idx_41.shape[0] != 0:
                #         idx2.append(idx_41[:length])
                #     if idx_luo.shape[0] != 0:
                #         idx3.append(idx_luo[:length])
                # idx1, idx2, idx3 = np.hstack(idx1), np.hstack(idx2), np.hstack(idx3)
                # X_train_gnu, y_train_gnu = X_train_gnu[idx1], y_train_gnu[idx1]
                # X_train_41, y_train_41 = X_train_41[idx2], y_train_41[idx2]
                # X_train_luo, y_train_luo = X_train_luo[idx3], y_train_luo[idx3]
                # print('-------------平衡后-----------------')
                # for i in range(len(modulation_classes)):
                #     idx_gnu = np.where(y_train_gnu[:, 0] == i)[0]
                #     idx_41 = np.where(y_train_41[:, 0] == i)[0]
                #     idx_luo = np.where(y_train_luo == i)[0]
                #     print(modulation_classes[i], 'gnu:', idx_gnu.shape[0], '41:', idx_41.shape[0], 'luo:',
                #           idx_luo.shape[0])
                # 扩展为IQ数据
                # X_train_luo = np.repeat(np.expand_dims(X_train_luo, axis=1), 2, axis=1)
                # X_val_luo = np.repeat(np.expand_dims(X_val_luo, axis=1), 2, axis=1)
                # X_test_luo = np.repeat(np.expand_dims(X_test_luo, axis=1), 2, axis=1)

                X_train = np.concatenate((X_train_gnu, X_train_41, X_train_luo), axis=0)
                del X_train_gnu, X_train_41, X_train_luo
                y_train = np.concatenate((y_train_gnu[:, 0], y_train_41[:, 0], y_train_luo), axis=0)

                X_val = np.concatenate((X_val_gnu, X_val_41, X_val_luo), axis=0)
                del X_val_gnu, X_val_41, X_val_luo
                y_val = np.concatenate((y_val_gnu[:, 0], y_val_41[:, 0], y_val_luo), axis=0)

                X_test = np.concatenate((X_test_gnu, X_test_41, X_test_luo), axis=0)
                del X_test_gnu, X_test_41, X_test_luo
                y_test = np.concatenate((y_test_gnu[:, 0], y_test_41[:, 0], y_test_luo), axis=0)

                return X_train, X_val, X_test, y_train, y_val, y_test

                # length = min(X_train_gnu.shape[0], X_train_41.shape[0], X_train_luo.shape[0])
                # X_train = np.concatenate((X_train_gnu[:length], X_train_41[:length], X_train_luo[:length]), axis=0)
                # del X_train_gnu, X_train_41, X_train_luo
                # y_train = np.concatenate((y_train_gnu[:length, 0], y_train_41[:length, 0], y_train_luo[:length]), axis=0)
                # length = min(X_val_gnu.shape[0], X_val_41.shape[0], X_val_luo.shape[0])
                # X_val = np.concatenate((X_val_gnu[:length], X_val_41[:length], X_val_luo[:length]), axis=0)
                # del X_val_gnu, X_val_41, X_val_luo
                # y_val = np.concatenate((y_val_gnu[:length, 0], y_val_41[:length, 0], y_val_luo[:length]), axis=0)
                # length = min(X_test_gnu.shape[0], X_test_41.shape[0], X_test_luo.shape[0])
                # X_test = np.concatenate((X_test_gnu[:length], X_test_41[:length], X_test_luo[:length]), axis=0)
                # del X_test_gnu, X_test_41, X_test_luo
                # y_test = np.concatenate((y_test_gnu[:length, 0], y_test_41[:length, 0], y_test_luo[:length]), axis=0)
                # return X_train, X_val, X_test, y_train, y_val, y_test


    elif task == 'Individual':
        path = '/media/yc/rushb/project/2022/wangjun/第二次验收/西电样本数据/Individual/train/'

        data, label = [], []
        for filename in os.listdir(path):
            print(filename)
            try:
                label_name = int(filename[2:])-1
                for file in os.listdir(path + filename):
                    try:
                        if '.dat' in file or '.pcm' in file or '.wav' in file:
                            x = np.fromfile(os.path.join(path + filename, file), dtype=np.int16)
                            x, y = cut_data(x, label_name, length)
                            data.append(x)
                            label.append(y)
                        # elif '.wav' in file:
                        #     # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
                        #     f = wave.open(os.path.join(path + filename, file), "rb")
                        #     # 读取格式信息
                        #     # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
                        #     params = f.getparams()
                        #     nchannels, sampwidth, framerate, nframes = params[:4]
                        #     # 读取波形数据
                        #     # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
                        #     str_data = f.readframes(nframes)
                        #     f.close()
                        #     # 将波形数据转换成数组
                        #     # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
                        #     wave_data = np.frombuffer(str_data, dtype=np.int16)
                        #     x, y = cut_data(wave_data, label_name, length)
                        #     data.append(x)
                        #     label.append(y)
                    except Exception as e:
                        print(file, e)
            except Exception as e:
                print(e)
        data = np.expand_dims(np.vstack(data), axis=1)
        data = guiyihua(data)
        label = np.hstack(label)

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.8)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5)

        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        print('任务名称错误！')


def load_model(model_name, class_num=100):
    if model_name == 'VGG11' or model_name == 'VGG13' or model_name == 'VGG16' or model_name == 'VGG19':
        model = models.VGG(model_name, class_num).cuda()
    elif model_name == 'Xception':
        model = models.Xception(class_num).cuda()
    elif model_name == 'ResNet18':
        model = models.ResNet18(class_num).cuda()
    elif model_name == 'ResNet34':
        model = models.ResNet34(class_num).cuda()
    elif model_name == 'ComplexNet':
        model = models.ComplexNet(class_num).cuda()
    else:
        raise ValueError("Invalid model name!")

    return model


def plot_snr_acc(train_snr_labels, val_snr_labels, train_true_labels, train_pre_labels, val_true_labels, val_pred_labels, modulation_classes, writer):
    train_oa, val_oa = [], []
    train_true_labels, train_pre_labels, val_true_labels, val_pred_labels = np.array(train_true_labels),\
                                                                            np.array(train_pre_labels),\
                                                                            np.array(val_true_labels),\
                                                                            np.array(val_pred_labels),
    for j in np.unique(train_true_labels):
        idx_1 = np.where(train_true_labels == j)[0]
        idx_2 = np.where(val_true_labels == j)[0]
        oa1, oa2 = [], []
        for i in np.unique(train_snr_labels):  # 计算不同信噪比的准确率
            idx1 = np.where(train_snr_labels[idx_1] == i)[0]
            idx2 = np.where(val_snr_labels[idx_2] == i)[0]
            oa1.append(accuracy_score(train_true_labels[idx_1][idx1], train_pre_labels[idx_1][idx1]))
            oa2.append(accuracy_score(val_true_labels[idx_2][idx2], val_pred_labels[idx_2][idx2]))
        train_oa.append(oa1)
        val_oa.append(oa2)

    # 不同信噪比下准确率的折线图
    fig = plt.figure()
    x = range(np.unique(train_snr_labels).shape[0])
    k = 0
    for i in np.unique(train_true_labels):
        plt.plot(x, train_oa[k], mec='r', mfc='w', label=f'{modulation_classes[i]}')
        k += 1
    plt.ylim(0, 1)
    plt.legend()  # 让图例生效
    plt.xticks(x, np.unique(train_snr_labels), rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("SNRS")  # X轴标签
    plt.ylabel("Accuracy")  # Y轴标签
    for i in range(np.unique(train_snr_labels).shape[0]):
        for a, b in zip(x, train_oa[i]):
            plt.text(a, b, round(b, 2))
    writer.add_figure(f'train_snr_acc', fig)
    plt.close()

    fig2 = plt.figure()
    # 不同信噪比下准确率的折线图
    x = range(np.unique(train_snr_labels).shape[0])
    k = 0
    for i in np.unique(train_true_labels):
        plt.plot(x, val_oa[k], mec='r', mfc='w', label=f'{modulation_classes[i]}')
        k += 1
    plt.ylim(0, 1)
    plt.legend()  # 让图例生效
    plt.xticks(x, np.unique(train_snr_labels), rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("SNRS")  # X轴标签
    plt.ylabel("Accuracy")  # Y轴标签
    for i in range(np.unique(train_snr_labels).shape[0]):
        for a, b in zip(x, val_oa[i]):
            plt.text(a, b, round(b, 2))
    writer.add_figure(f'val_snr_acc', fig2)
    plt.close()


# 封装成函数
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0


def wavelet_noising(data):
    data = list(data)  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym8')  #选择sym8小波基
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 5层小波分解

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))#固定阈值计算
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象

    #软硬阈值折中的方法
    a = 0.5

    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs


def plot_confusion_matrix(y_true, y_pred, labels, mod, writer):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.figure(figsize=(10, 8), dpi=120)
    fig = plt.figure()
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=10, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')

    writer.add_figure(f'confusion_matrix_{mod}', fig)

    plt.close()


# 检测脉冲
def detect_mc(x, threshold=.7):
    print('开始检测脉冲......')
    t1 = time.time()
    # X = wavelet_noising(X)  # 调用函数进行小波阈值去噪
    x = x.astype(np.float64)
    X = guiyihua(x)
    print(np.max(X), np.min(X))
    print('去噪用时：', time.time()-t1)
    t1 = time.time()
    # plt.plot(X[:100000])
    # plt.show()
    # plt.close()
    idx = np.where(X > threshold)[0]  # 模大于threshold的点为突发信号所在点
    start = [idx[0]]
    end = []
    for i in range(idx.shape[0]):
        try:
            if idx[i] + 5000 < idx[i + 1]:
                start.append(idx[i + 1])
                end.append(idx[i])
        except:
            continue
    end.append(idx[-1])
    X = []
    for k in range(len(start)):
        if end[k] - start[k] > 1000:
            X.append(x[start[k]:end[k]])
    X = np.hstack(X)
    print('检测用时：', time.time() - t1)

    return X


def cut_data(x, y, length):
    data, label = [], []
    for i in range(int(x.shape[-1]/length)):
        if x.shape[0] != 2:
            data.append(x[i*length:(i+1)*length])
        else:
            data.append(x[:, i * length:(i + 1) * length])
        label.append(y)
    data = np.array(data)
    label = np.array(label)


    return data, label


def guiyihua(x):
    x = x.astype(np.float64)
    return (x-np.min(x))/(np.max(x)-np.min(x))


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = load_data('Modulation', 1024, 'all', 'wuhu')