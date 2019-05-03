import numpy  as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch as th
from torch.autograd import Variable as Var


def adjust_lr(optimizer, alpha, lr):
    lr /= alpha
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def binarize(arr):
    arr[arr >= 0.5] = 1.
    arr[arr < 0.5] = 0.

    return arr


def as_float_cpu(arr, numpy=False):
    if isinstance(arr, Var):
        arr = arr.data

    if arr.__class__.__module__.startswith('torch'):
        arr = arr.cpu().type(th.FloatTensor)

        if numpy:
            arr = arr.numpy()

    return arr


def accuracy_score(y_true, y_pred):
    y_true = as_float_cpu(y_true)
    y_pred = as_float_cpu(y_pred)

    y_true = binarize(y_true)
    y_pred = binarize(y_pred)

    k = y_true.shape[0]
    acc = np.mean(y_true.reshape(k, -1) == y_pred.reshape(k, -1), axis=1).mean()

    return acc


def roc_auc_score(y_true, y_pred):
    y_true = as_float_cpu(y_true)
    y_pred = as_float_cpu(y_pred)

    y_true = binarize(y_true)

    k = y_true.shape[0]
    auc = sklearn.metrics.roc_auc_score(y_true.reshape(k, -1).T, y_pred.reshape(k, -1).T)

    return auc
