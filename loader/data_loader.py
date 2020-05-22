import numpy as np
import torch
from torch import nn
import sys
import util


class DataLoader(nn.Module):
    """Data loader module for training MCFlow
    Args:
        mode (int): Determines if we are loading training or testing data
        seed (int): Used to determine the fold for cross validation experiments or reproducibility consideration if not
        path (str): Determines the dataset to load
        drp_percent (float): Determines the binomial success rate for observing a feature
    """
    MODE_TRAIN = 0
    MODE_TEST = 1

    def __init__(self, mode=MODE_TRAIN, seed=0, path='physionet_train', drp_percent=0.5):

        self.path = path

        matrix = util.path_to_matrix(path)

        print("Initialized matrix")

        self.matrix, self.maxs, self.mins = util.preprocess(matrix)  # Preprocess according to the paper cited above

        print("Matrix preprocessed")

        np.random.shuffle(self.matrix)
        np.random.seed(seed)

        # check if the mask is there or not in this function
        self.mask = util.make_static_mask(drp_percent, seed, path, self.matrix)

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

    @classmethod
    def test_data_loader(cls, path='physionettest', mode=3):
        cls.path = path

        cls.matrix = util.path_to_matrix(path)

    def reset_imputed_values(self, nn_model, nf_model, args):

        random_mat = np.clip(util.inference_imputation_networks(nn_model, nf_model, self.train, args), 0, 1)
        self.train = (1 - self.mask_tr) * self.original_tr + self.mask_tr * random_mat
        random_mat = np.clip(util.inference_imputation_networks(nn_model, nf_model, self.test, args), 0, 1)
        self.test = (1 - self.mask_te) * self.original_te + self.mask_te * random_mat

    def __len__(self):
        if self.mode == 0:
            return len(self.train)
        elif self.mode == 1:
            return len(self.test)
        else:
            print("Data loader mode error -- acceptable modes are 0,1,2")
            sys.exit()

    def __getitem__(self, idx):
        if self.mode == 0:
            return torch.Tensor(self.train[idx]), (torch.Tensor(self.original_tr[idx]), torch.Tensor(self.mask_tr[idx]))
        elif self.mode == 1:
            return torch.Tensor(self.test[idx]), (torch.Tensor(self.original_te[idx]), torch.Tensor(self.mask_te[idx]))
        else:
            print("Data loader mode error -- acceptable modes are 0,1,2")
            sys.exit()
