## torch
import torch 
import torch.nn as nn

class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.up0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x0):
        x1 = self.up0(x0)
        x0 = self.conv0(x0)
        x0 = self.relu(x0)
        x0 = self.conv1(x0)
        x0 = self.relu(x0)
        x0 = self.conv2(x0)
        x0 = self.relu(x0)
        x0 = self.conv3(x0)
        x0 = self.relu(x0)
        x0 = self.up1(x0)
        x0 = self.conv4(x0)
        x0 = self.relu(x0)
        out = x1 + x0

        return out

## tensorflow
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D
from tensorflow.keras import Model 
class TFNet(Model):
    def __init__(self):
        super(TFNet, self).__init__()

        self.conv0 = Conv2D(64, 3, padding='same', activation='relu', data_format='channels_first')
        self.conv1 = Conv2D(64, 3, padding='same', activation='relu', data_format='channels_first')
        self.conv2 = Conv2D(64, 3, padding='same', activation='relu', data_format='channels_first')
        self.conv3 = Conv2D(64, 3, padding='same', activation='relu', data_format='channels_first')
        self.conv4 = Conv2D(3, 3, padding='same', activation='relu', data_format='channels_first')

        self.up0 = UpSampling2D(size=(2,2), interpolation='nearest', data_format='channels_first')
        self.up1 = UpSampling2D(size=(2,2), interpolation='nearest', data_format='channels_first')

    def call(self, x):
        x1 = self.up0(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = self.up1(x)
        x = self.conv4(x)
        out = x1 + x
        return out

## mxnet
