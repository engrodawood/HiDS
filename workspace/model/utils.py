import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data#, DataLoader
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import pickle
from copy import deepcopy
from numpy.random import randn
import time
from scipy.stats import spearmanr
import os
import pandas as pd
import matplotlib.pyplot as plt


USE_CUDA = torch.cuda.is_available()
device = {True: 'cuda:3', False: 'cpu'}[USE_CUDA]

def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))


def toNumpy(v):
    if type(v) is not torch.Tensor:
        return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()


def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return torch.load(f)


def toGeometric(Gb, y, tt=1e-3):
    return Data(x=Gb.X, edge_index=(Gb.getW() > tt).nonzero().t().contiguous(), y=y)


def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)


def toGeometricWW(X, W, y, tt=0):
    return Data(x=toTensor(X, requires_grad=False), edge_index=(toTensor(W, requires_grad=False) > tt).nonzero().t().contiguous(), y=toTensor([y], dtype=torch.float, requires_grad=False))


def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

class StratifiedSampler(Sampler):
    """Stratified Sampling
         return a stratified batch
    """

    def __init__(self, class_vector, batch_size=10):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        """
        self.batch_size = batch_size
        self.n_splits = int(class_vector.size(0) / self.batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        YY = self.class_vector.numpy()
        idx = np.arange(len(YY))
        # return array of arrays of indices in each batch
        return [tidx for _, tidx in skf.split(idx, YY)]

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def calc_roc_auc(target, prediction):
    return roc_auc_score(toNumpy(target), toNumpy(prediction))


def calc_pr(target, prediction):
    return average_precision_score(toNumpy(target), toNumpy(prediction))