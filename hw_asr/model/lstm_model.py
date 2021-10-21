from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel


class LSTM(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, n_layers=1, bidirectional=False, *args, **kwargs):
        super().__init__(n_feats, n_class, hidden_size=hidden_size,
                         n_layers=n_layers, bidirectional=bidirectional, *args, **kwargs)

        self.lstm = nn.LSTM(n_feats, hidden_size, n_layers, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, n_class)

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        x, _ = self.lstm(spectrogram)
        x = self.linear(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
