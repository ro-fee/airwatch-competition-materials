import torch.nn as nn
import torch
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,padding=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride,padding),
            # nn.BatchNorm1d(out_planes),
            nn.ReLU()
        )


class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv1 =  ConvBNReLU(1, 32, 3,1,1)
        self.conv2 =  ConvBNReLU(32, 32, 3,1,1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv3 =  ConvBNReLU(32, 64, 5,1,2)
        self.conv4 =  ConvBNReLU(64, 64, 5,1,2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv5 =  ConvBNReLU(64, 128, 7,1,3)
        self.conv6 =  nn.Conv1d(128, 128, 7,1,3)
        self.pool3 = nn.MaxPool1d(2)

        self.conv7 = ConvBNReLU(128, 256, 9,1,4)
        self.conv8 = ConvBNReLU(256, 256, 9,1,4)
        self.pool4 = nn.MaxPool1d(2)




        self.fc1 = nn.Sequential(
            nn.Linear(8192, 20),
            nn.BatchNorm1d(20,eps=1e-06),
            # nn.Dropout(0.2),
            # nn.BatchNorm1d(20)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(20, num_classes),
            # nn.BatchNorm1d(6),
        )

    def forward(self, out):
        # print(out.shape)
        out = self.conv1(out)
        # print(out.shape,1)
        out = self.conv2(out)
        # print(out.shape,2)
        out = self.pool1(out)
        out = self.conv3(out)
        # print(out.shape, 3)
        out = self.conv4(out)
        out = self.pool2(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool3(out)
        # out = self.conv7(out)
        # out = self.conv8(out)
        # out = self.pool4(out)
        # print(out.shape, 4)
        feature = out.reshape(out.size(0), -1)
        # print(out.shape)
        out = self.fc1(feature)
        out = self.fc2(out)

        return feature, out
# def test():
#     net = CNN(num_classes=3)
#     y = net(torch.randn(1, 1, 512))
#     print(y.size())
# test()