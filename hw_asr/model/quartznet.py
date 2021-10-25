import torch.nn as nn
from hw_asr.base import BaseModel


class CBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, dilation=1):
        super(CBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding = dilation * kernel_size // 2),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class BCell(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, dilation=1, activation=True):
        super(BCell, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, stride=stride,
                        dilation=dilation, groups=input_channels, padding = dilation * kernel_size // 2),
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, dilation=dilation),
            nn.BatchNorm1d(output_channels)
        )
        if activation:
            self.layers.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class BBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, block_size):
        super(BBlock, self).__init__()
        layers = []
        for i in range(block_size):
            in_channels = input_channels if i == 0 else output_channels
            activation = False if i == block_size - 1 else True
            layers.append(BCell(in_channels, output_channels, kernel_size=kernel_size, activation=activation))

        self.layers = nn.Sequential(*layers)

        self.skip_connection = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 1),
            nn.BatchNorm1d(output_channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        skip_connection = self.skip_connection(x)
        x = self.layers(x)
        return self.relu(x + skip_connection)


class BGroup(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, group_size=1, block_size=1):
        super(BGroup, self).__init__()
        layers = [BBlock(input_channels, output_channels, kernel_size=kernel_size, block_size=block_size) for _ in range(group_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.layers = nn.Sequential(
            CBlock(n_feats, 256, kernel_size=33, stride=2), # C1
            BGroup(256, 256, kernel_size=33),
            BGroup(256, 256, kernel_size=39),
            BGroup(256, 512, kernel_size=51),
            BGroup(512, 512, kernel_size=63),
            BGroup(512, 512, kernel_size=75),
            CBlock(512, 512, kernel_size=87), # C2
            CBlock(512, 1024, kernel_size=1), # C3
            nn.Conv1d(1024, n_class, kernel_size=1, dilation=2) # C4
        )

    def forward(self, spectrogram, *args, **kwargs):
        return self.layers(spectrogram.permute(0, 2, 1)).permute(0, 2, 1)

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2