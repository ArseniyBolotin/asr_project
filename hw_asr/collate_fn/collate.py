import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def last_dim_padding(dataset_items, key):
    items_with_padding = []
    max_size = max(list(item[key].size(-1) for item in dataset_items))
    for item in dataset_items:
        items_with_padding.append(F.pad(item[key], (0, max_size - item[key].size(-1))))
    return torch.cat(tuple(items_with_padding), dim=0)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    for key in ('audio', 'spectrogram', 'text_encoded'):
        result_batch[key] = last_dim_padding(dataset_items, key)

    for key in ('duration', 'text', 'audio_path'):
        result_batch[key] = [item[key] for item in dataset_items]

    result_batch['spectrogram'] = result_batch['spectrogram'].permute(0, 2, 1)
    result_batch['text_encoded_length'] = torch.tensor(list(item['text_encoded'].size(-1) for item in dataset_items), dtype=torch.int32)
    result_batch['spectrogram_length'] = torch.tensor(list(item['spectrogram'].size(-1) for item in dataset_items), dtype=torch.int32)
    return result_batch
