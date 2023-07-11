import torch.utils.data as data_utils
import numpy as np


class CustomSampler(data_utils.Sampler):
    def __init__(self, dataset, sampler_task_dict):
        self.dataset = dataset
        self.sampler_task_dict = sampler_task_dict
        self.task_indices = {}
        for i in range(len(dataset)):
            task_name = dataset[i]["label"].name()
            if task_name not in self.task_indices:
                self.task_indices[task_name] = []
            self.task_indices[task_name].append(i)

    def __iter__(self):
        indices = []
        for task_name in self.sampler_task_dict.keys():
            task_size = int(len(self.task_indices[task_name]) * self.sampler_task_dict[task_name])
            indices += np.random.choice(self.task_indices[task_name], task_size, replace=False).tolist()
        return iter(indices)

    def __len__(self):
        total_size = 0
        for task_name in self.sampler_task_dict.keys():
            total_size += int(len(self.task_indices[task_name]) * self.sampler_task_dict[task_name])
        return total_size
