import math
import random
import torch
from torch.utils.data import Sampler


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
        super().__init__(data_source)
        self.sorted_index = data_source._index
        self.batch_size = batch_size
        self.batches_per_group = batches_per_group
        self.group_size = self.batch_size * self.batches_per_group
        self.groups = math.ceil(len(self.sorted_index) / self.group_size)

    def __iter__(self):
        group_index = random.randint(0, self.groups - 1)
        idx = random.sample(range(group_index * self.group_size, (group_index + 1) * self.group_size), k=self.batch_size)
        yield torch.tensor(idx)

    def __len__(self):
        return math.ceil(len(self.sorted_index) / self.batch_size)
