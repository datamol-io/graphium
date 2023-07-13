import torch.utils.data as data_utils
import numpy as np
import time
import os
import torch


class CustomSampler(data_utils.Sampler):
    def __init__(self, dataset, sampler_task_dict, data_path, data_hash):
        self.dataset = dataset
        self.sampler_task_dict = sampler_task_dict
        self.task_indices = {}
        now = time.time()
        path_with_hash = os.path.join(data_path, data_hash)
        os.makedirs(path_with_hash, exist_ok=True)
        filename = os.path.join(path_with_hash, "dataset_indices.pkl")
        # if file dataset_indices.pkl exists, load indices from file.
        if os.path.isfile(filename):
            print(f"Sampler--loading dataset indices from disk.")
            self.task_indices = torch.load(filename)
        # if not, iterate through all data items in dataset and save indices to disk
        else:
            for i in range(len(dataset)):
                # the dataset[i]["labels"].keys() are not deterministic, need to sort by key length
                task_name = sorted(dataset[i]["labels"].keys(), key=len)[-1]
                if task_name not in self.task_indices:
                    self.task_indices[task_name] = []
                self.task_indices[task_name].append(i)
            elapsed = round(time.time() - now)
            print(f"Sampler-->time spent on getting indices: {elapsed}.")
            torch.save(self.task_indices, filename, pickle_protocol=4)

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
