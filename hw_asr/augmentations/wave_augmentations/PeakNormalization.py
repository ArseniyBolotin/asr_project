import torch_audiomentations
from torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase


class PeakNormalization(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PeakNormalization(apply_to="only_too_loud_sounds", *args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < 0.2:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        return data
