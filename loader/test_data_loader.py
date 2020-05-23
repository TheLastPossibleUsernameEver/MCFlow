import torch
from torch import nn

import util


class TestDataLoader(nn.Module):

    def __init__(self, path='physionet_test'):
        self.path = path

        matrix = util.path_to_matrix(path)

        self.matrix, self.maxs, self.mins = util.preprocess(matrix)

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return torch.Tensor(self.matrix[idx])
