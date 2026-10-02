import torch
from torch import nn

# 生成器结构
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(15, 15)

        self.model = nn.Sequential(

            nn.Linear(115, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 1024),
            nn.Tanh()
        )

    def forward(self, noise, label):
        # print(self.label_emb(label).shape,noise.shape)
        out = torch.cat((noise, self.label_emb(label)), -1)
        # print(out.shape)
        img = self.model(out)     # torch.Size([64, 784])
        img = img.view(img.size(0), 1, 1024)     # torch.Size([64, 1, 32, 32])
        return img