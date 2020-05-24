import torch
import numpy as np
from torch import nn
import util


class TestDataLoader(nn.Module):

    MODE_TEST = 1

    def __init__(self, path='physionet_test', mode=MODE_TEST, seed=0):
        self.path = path

        matrix = util.path_to_matrix(path)

        self.matrix, self.maxs, self.mins = util.preprocess(matrix)

        self.mask = util.make_just_mask(self.matrix)

        self.original_tr, self.original_te = util.create_k_fold(self.matrix, seed)
        self.unique_values = []
        self.mask_tr, self.mask_te = util.create_k_fold_mask(seed, self.mask)

        trans = np.transpose(self.matrix)

        for r_idx, rows in enumerate(trans):
            row = []
            for c_idx, element in enumerate(rows):
                if self.mask[c_idx][r_idx] == 0:
                    row.append(element)
            self.unique_values.append(np.asarray(row))

        self.train, self.test = util.fill_missingness(self.matrix, self.mask, self.unique_values, self.path, seed)
        self.mode = mode

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return torch.Tensor(self.test[idx]), (torch.Tensor(self.original_te[idx]), torch.Tensor(self.mask_te[idx]))
