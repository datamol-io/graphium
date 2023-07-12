import torch.utils.data as data_utils
import numpy as np


class CustomSampler(data_utils.Sampler):
    def __init__(self, dataset, sampler_task_dict):
        self.dataset = dataset
        self.sampler_task_dict = sampler_task_dict
        self.task_indices = {}
        for i in range(len(dataset)):
            # the dataset[i]["labels"].keys() are not deterministic, need to sort by key length
            task_name = sorted(dataset[i]["labels"].keys(), key=len)[-1]
            if task_name not in self.task_indices:
                self.task_indices[task_name] = []
            self.task_indices[task_name].append(i)

    def __iter__(self):
        indices = []
        for task_name in self.task_indices.keys():
            task_size = int(len(self.task_indices[task_name]) * self.sampler_task_dict[task_name])
            indices += np.random.choice(self.task_indices[task_name], task_size, replace=False).tolist()
            indices_set = set(indices)
            self.total_size = len(indices_set)
        return iter(indices_set)

    def __len__(self):
        return self.total_size
