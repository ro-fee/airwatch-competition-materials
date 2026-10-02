import torch.nn as nn
import torch
import torch.nn.functional as F


class CNNNEW_zl(nn.Module):
    def __init__(self, num_classes):
        super(CNNNEW_zl, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3), nn.BatchNorm1d(8), nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=15), nn.BatchNorm1d(512), nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # self.conv8 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7), nn.BatchNorm1d(32), nn.ReLU()
        # )
        #
        # self.conv9 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5), nn.BatchNorm1d(32), nn.ReLU()
        # )
        #
        # self.conv10 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3), nn.BatchNorm1d(32), nn.ReLU()
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 128))
        self.fc2 = nn.Sequential(
            nn.Linear(128, num_classes))

    def forward(self, x):
        out = self.conv1(x)
        # print('conv1: ', out.shape)
        out = self.conv2(out)
        # print('conv2: ', out.shape)
        out = self.conv3(out)
        # print('conv3: ', out.shape)
        out = self.conv4(out)
        # print('conv4: ', out.shape)
        out = self.conv5(out)
        # print('conv5: ', out.shape)
        out = self.conv6(out)
        # print('conv6: ', out.shape)
        out = self.conv7(out)
        out = F.adaptive_avg_pool1d(out, 1)
        # print('conv7: ', out.shape)
        # out = self.conv8(out)
        # out = self.conv9(out)
        # feature = self.conv10(out)
        # print('conv10: ', out.shape)

        feature = out.reshape(out.size(0), -1)
        # print(feature.shape)
        out = self.fc1(feature)
        # print('fc1: ', out.shape)
        out = self.fc2(out)
        # print('fc2: ', out.shape)
        return feature, out