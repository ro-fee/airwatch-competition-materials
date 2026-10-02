# -*- coding: UTF-8 -*
import os
import time
import h5py
import models
import torch
import utils
import wave
import argparse
import datetime

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score


def plot_confusion_matrix(y_true, y_pred, labels, task):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    if not os.path.exists('./result'):
        os.makedirs('./result')

    cmap = plt.cm.binary
    # cm = confusion_matrix(y_true, y_pred)
    xx1 = np.unique(y_true)
    xx2 = np.unique(y_pred)
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
    if (intFlag):
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
    plt.savefig(f'./result/confusion_{task}.png')

    plt.close()


def plot_snr(val_snr_labels, val_true_labels, val_pred_labels, modulation_classes, data_set):
    train_oa, val_oa = [], []
    val_true_labels, val_pred_labels = np.array(val_true_labels), np.array(val_pred_labels)
    for j in np.unique(val_true_labels):
        idx = np.where(val_true_labels == j)[0]
        oa = []
        for i in np.unique(val_snr_labels):  # 计算不同信噪比的准确率
            idx1 = np.where(val_snr_labels[idx] == i)[0]
            oa.append(accuracy_score(val_true_labels[idx][idx1], val_pred_labels[idx][idx1]))
        val_oa.append(oa)

    # 不同信噪比下准确率的折线图
    x = range(np.unique(val_snr_labels).shape[0])
    k = 0
    for i in np.unique(val_true_labels):
        plt.plot(x, val_oa[k], mec='r', mfc='w', label=f'{modulation_classes[i]}')
        k += 1
    plt.ylim(0, 1)
    plt.legend()  # 让图例生效
    plt.xticks(x, np.unique(val_snr_labels), rotation=45)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("SNRS")  # X轴标签
    plt.ylabel("Accuracy")  # Y轴标签
    for i in range(np.unique(val_true_labels).shape[0]):
        for a, b in zip(x, val_oa[i]):
            plt.text(a, b, round(b, 2))
    # plt.show()
    with open(f'./result/{data_set}_snr.txt', 'w') as f:
        for i in range(np.unique(val_true_labels).shape[0]):
            f.write(modulation_classes[i] + '\n')
            snrs = np.unique(val_snr_labels)
            for j in range(snrs.shape[0]):
                f.write(str(snrs[j]) + ': ' + str(val_oa[i][j]) + '  ')
            f.write('\n')
    plt.savefig(f'./result/{data_set}_snr.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-d', '--data_set', default='41fangzhen', choices=['gnu', '41', 'luo', '41fangzhen', 'all'])
    parser.add_argument('-l', '--length', default='1024', type=int)
    parser.add_argument('-m', '--model', default='ResNet18', choices=['ComplexNet', 'Xception', 'ResNet18', 'ResNet34'])
    parser.add_argument('-t', '--task', default='Modulation', choices=['Modulation', 'Individual', 'Encoder', 'Service'])
    args = parser.parse_args()

    # -----------------load data-----------------
    modulation_classes = ['16QAM', '32QAM', '64QAM', '128QAM', 'BFSK', '4FSK', 'CPFSK', 'BPSK', 'QPSK', '8PSK',
                          '16PSK', 'OQPSK', 'FM', 'AM', 'PAM4']
    X_test, y_test = utils.load_data(args.task, args.length, args.data_set, 'test')
    y_test = y_test.astype(np.int16)
    for i in np.unique(y_test[:, 0]):
        idx = np.where(y_test[:, 0] == i)[0]
        idx1 = np.where(y_test[idx, 1] == 30)[0]
        k = 0
        for j in idx1:
            plt.title(modulation_classes[i])
            plt.scatter(X_test[idx[j], 0, :], X_test[idx[j], 1, :])
            plt.show()
            plt.close()
            k += 1
            if k == 5:
                break
    xx = np.unique(y_test[:, 1])
    # X_train, X_val, X_test, y_train, y_val, y_test = utils.load_data(args.task, args.length, args.data_set, 'wuhu')
    class_num = len(modulation_classes)

    # -----------------load model-----------------
    model = utils.load_model(args.model, class_num)
    # model.load_state_dict(torch.load(f'./logs/weights/{args.task}/{args.data_set}/{args.model}_6.pkl'))
    model.load_state_dict(torch.load(f'../models/{args.model}.pkl'))
    model = model.cuda()
    model.eval()
    # -----------------load model-----------------

    test_dataset = TensorDataset(torch.from_numpy(X_test),
                                 torch.from_numpy(y_test))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    val_pred_labels = []
    val_true_labels = []
    val_file_labels = []
    val_snr_labels = []
    t1 = time.time()
    with torch.no_grad():
        for x, y in tqdm(test_dataloader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x.float())
            for x in pred:
                label = np.argmax(x.cpu().detach().numpy(), axis=0)
                val_pred_labels.append(label)
            for x in y[:, 0]:
                val_true_labels.append(x.cpu().detach().numpy())
            for x in y[:, 1]:
                val_snr_labels.append(x.cpu().detach().numpy())
            for x in y[:, 2]:
                val_file_labels.append(x.cpu().detach().numpy())
    val_pred_labels = np.array(val_pred_labels).astype(np.int16)
    val_file_labels = np.array(val_file_labels).astype(np.int16)
    val_true_labels = np.array(val_true_labels).astype(np.int16)
    val_snr_labels = np.array(val_snr_labels).astype(np.int16)
    t2 = time.time()
    print('耗时：', t2-t1, ' s, ', '单个样本耗时：', 1000*((t2-t1)/np.unique(val_file_labels).shape[0]), ' ms')
    pred_labels = []
    true_labels = []
    for i in np.unique(val_file_labels):
        idx = np.where(val_file_labels == i)[0]
        pred_labels.append(np.argmax(np.bincount(val_pred_labels[idx])))
        true_labels.append(np.argmax(np.bincount(val_true_labels[idx])))
    acc_for_each_class = precision_score(val_true_labels, val_pred_labels, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    print('oa', accuracy_score(pred_labels, true_labels), 'aa', average_accuracy)
    plot_snr(val_snr_labels, val_true_labels, val_pred_labels, modulation_classes, args.data_set)
    # xx = np.bincount(val_pred_labels)
    # label = np.argmax(np.bincount(val_pred_labels))
    # print(filelist[j], '预测为：', modulation_classes[label])
    # for j in range(len(filelist)):
    #     test_dataset = TensorDataset(torch.from_numpy(X_test[j]),
    #                                  torch.from_numpy(Y_test[j]))
    #     test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    #     val_pred_labels = []
    #     val_true_labels = []
    #     with torch.no_grad():
    #         for x, y in tqdm(test_dataloader):
    #             x = x.cuda()
    #             y = y.cuda()
    #             pred = model(x.float())
    #             for x in pred:
    #                 label = np.argmax(x.cpu().detach().numpy(), axis=0)
    #                 val_pred_labels.append(label)
    #             for x in y:
    #                 val_true_labels.append(x.cpu().detach().numpy())
    #     xx = np.bincount(val_pred_labels)
    #     label = np.argmax(np.bincount(val_pred_labels))
    #     print(filelist[j], '预测为：', modulation_classes[label])
    # val_true_labels = np.array(val_true_labels)
    # val_pred_labels = np.array(val_pred_labels)
    # print('Test accuracy: ', accuracy_score(val_true_labels, val_pred_labels))
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    # idx = np.where(true_labels == 10)[0]
    # idx_false = true_labels[idx] == pred_labels[idx]
    # idx_false = np.where(idx_false == False)[0]
    # idx_false = np.random.choice(idx_false, 19, replace=False)
    # import shutil
    # for i in idx_false:
    #     src_path = '/media/yc/rushb/project/2022/wangjun/第二次验收/classification/调制数据/41仿真/test/16PSK/'
    #     dst_path = '/media/yc/rushb/project/2022/wangjun/第二次验收/classification/调制数据/41仿真/16PSK/'
    #     shutil.move(src_path+filelist[idx[i]], dst_path+filelist[idx[i]])
    xx1 = np.unique(true_labels)
    xx2 = np.unique(pred_labels)

    plot_confusion_matrix(true_labels, pred_labels, modulation_classes, args.data_set)