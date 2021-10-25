import torch.nn as nn
from hw_asr.base import BaseModel


class Cell(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, dilation=1, activation=True):
        super(Cell, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size, stride=stride, dilation=dilation, groups=input_channels, padding = dilation * kernel_size // 2),
            nn.Conv1d(input_channels, output_channels, 1),
            nn.BatchNorm1d(output_channels)
        )
        if activation:
            self.layers.add_module('relu', nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class BBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, block_size=5):
        super(BBlock, self).__init__()
        self.layers = []

        for index in range(block_size):
            input_channels_modified = output_channels
            activation = True
            if index == 0:
                input_channels_modified = input_channels
            if index == block_size - 1:
                activation = False
            self.layers.append(Cell(input_channels_modified, output_channels, kernel_size, activation=activation))

        self.layers = nn.Sequential(*self.layers)
        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 1),
            nn.BatchNorm1d(output_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layers(x) + self.skip_connection(x))


class BGroup(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, group_size=3):
        super(BGroup, self).__init__()
        self.layers = []

        for index in range(group_size):
            input_channels_modified = output_channels
            if index == 0:
                input_channels_modified = input_channels
            self.layers.append(BBlock(input_channels_modified, output_channels, kernel_size))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.layers = nn.Sequential(
            Cell(n_feats, 256, kernel_size=33, stride=2),  # C1
            BGroup(256, 256, kernel_size=39),
            BGroup(256, 256, kernel_size=33),
            BGroup(256, 512, kernel_size=51),
            BGroup(512, 512, kernel_size=63),
            BGroup(512, 512, kernel_size=75),
            Cell(512, 512, kernel_size=87),  # C2
            Cell(512, 1024, kernel_size=1),  # C3
            nn.Conv1d(1024, n_class, kernel_size=1, dilation=2)  # C4
        )

    def forward(self, spectrogram, *args, **kwargs):
        return self.layers(spectrogram.permute(0, 2, 1)).permute(0, 2, 1)

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
