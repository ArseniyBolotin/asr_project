from torch import nn
import torch.nn.functional as F

from hw_asr.base import BaseModel


class CheckpointModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, n_layers=3, *args, **kwargs):
        super().__init__(n_feats, n_class, hidden_size=512, n_layers=3, *args, **kwargs)

        self.lstm = nn.LSTM(n_feats, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, n_class)

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        x, _ = self.lstm(spectrogram)
        x = self.linear(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
