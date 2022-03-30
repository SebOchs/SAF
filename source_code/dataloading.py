import numpy as np
import torch
from torch.utils.data import Dataset


# data loader for a given data set
class T5Dataset(Dataset):
    def __init__(self, filename):
        self.data = np.load(filename, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        input_text, input_attn, output, label = data[0], data[1], data[2], data[3][:4]
        return torch.tensor(input_text).long(), torch.tensor(input_attn).long(), \
               torch.tensor(output).long(), torch.tensor(label).long()
