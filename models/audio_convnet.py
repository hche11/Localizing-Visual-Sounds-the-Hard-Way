import torch
import os
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


class AudioConvNet(nn.Module):

    def __init__(self):
        super(AudioConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.cnn1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bat10 = nn.BatchNorm2d(64)
        self.bat11 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bat20 = nn.BatchNorm2d(128)
        self.bat21 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bat30 = nn.BatchNorm2d(256)
        self.bat31 = nn.BatchNorm2d(256)

        self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bat40 = nn.BatchNorm2d(512)
        self.bat41 = nn.BatchNorm2d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #  nn.init.normal_(m.weight, mean=1, std=0.02)
                #  nn.init.normal_(m.weight, mean=0, std=0.2)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        c = F.relu(self.bat10(self.cnn1(inp)))
        c = F.relu(self.bat11(self.cnn2(c)))
        c = self.pool(c)

        c = F.relu(self.bat20(self.cnn3(c)))
        c = F.relu(self.bat21(self.cnn4(c)))
        c = self.pool(c)

        c = F.relu(self.bat30(self.cnn5(c)))
        c = F.relu(self.bat31(self.cnn6(c)))
        c = self.pool(c)

        c = F.relu(self.bat40(self.cnn7(c)))
        c = F.relu(self.bat41(self.cnn8(c)))

        return c

    # Dummy function, just to test if feedforward is working or not
    def loss(self, output):
        return (output.mean())**2


if __name__ == "__main__":

    model = AudioConvNet().cuda()
    print("Model loaded.")
    image = Variable(torch.rand(2, 1, 257, 200)).cuda()
    print("Image loaded.")

    # Run some sample epochs to see if everything's working
    optim = SGD(model.parameters(), lr=1e-4)
    for i in range(100):
        optim.zero_grad()
        c = model(image)
        print(c.shape)
        print(image.shape)
        loss = model.loss(c)
        loss.backward()
        optim.step()
        print(loss.data[0])
