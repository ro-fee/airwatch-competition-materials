import torch.nn as nn
import torch
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    input and output sizes will be the same
    """
    def __init__(self, in_size, out_size, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size,padding=self.pad,dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]
        return x


class TCNResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inputC, outputC, kernal_size=3):
        super(TCNResidualBlock,self).__init__()
        self.dilatedCausalConv1 = nn.Sequential(
            CausalConv1d(inputC,inputC,kernel_size=kernal_size,dilation=1),
            nn.BatchNorm1d(inputC),
            nn.ReLU()
        )
        self.dilatedCausalConv2 = nn.Sequential(
            CausalConv1d(inputC, inputC, kernel_size=kernal_size, dilation=2),
            nn.BatchNorm1d(inputC),
            nn.ReLU()
        )
        self.dilatedCausalConv3 = nn.Sequential(
            CausalConv1d(inputC, outputC, kernel_size=kernal_size, dilation=5),
            nn.BatchNorm1d(outputC),
            nn.ReLU()
        )
        self.Conv1 = nn.Sequential(
            nn.Conv1d(inputC, outputC, kernel_size=1),
            nn.BatchNorm1d(outputC),
        )

    def forward(self, x):
        x1 = self.dilatedCausalConv1(x)
        x1 = self.dilatedCausalConv2(x1)
        x1 = self.dilatedCausalConv3(x1)
        x = self.Conv1(x)

        return F.relu(x1 + x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=30, in_channel=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3])
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for stride in range(num_blocks):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes * block.expansion
        layers.append(nn.Conv1d(planes, planes, kernel_size=3,stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):  # target
        x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool1d(out, 1)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        out = self.dequant(out)
        return feature, out


def ResNet18_TCN(num_classes=11):
    return ResNet(TCNResidualBlock, [1, 1, 2, 2], num_classes, in_channel=2)


if __name__ == '__main__':
    input = torch.randn((1, 1, 2048))
    net = ResNet18_TCN(num_classes=18)
    out = net(input)

