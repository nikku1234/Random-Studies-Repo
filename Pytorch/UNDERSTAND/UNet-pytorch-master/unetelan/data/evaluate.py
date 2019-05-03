import numpy as np
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from torch import nn
import torch as th
from torch.nn import functional as F


def plot_prediction(img, pred):
    """ Overlaps predictions with the image. Also plots the raw image for comparison

    Parameters
    ----------
    img : uint8[:, :, :]
        raw image in rgb format
    pred : bool[:, :]
        0-1 predictions
    """

    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)

    mask  = pred.astype(np.bool)
    img = np.copy(img)
    img[mask, :] = [0, 0, 126]

    fig.add_subplot(1, 2, 2)
    plt.imshow(img)


def plot_confusion_matrix(cm, normalize=False):
    """  Print and plot the confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        confusion matrix to plot
    normalize : bool
        If we should normalize numbers to 0-1
    """

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ['Present', 'Absent']

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def evaluate(true_masks, pred_raw):
    """ Evaluate the performance of the model on a set of ground truths and predictions

    Parameters
    ----------
    true_masks : List
        writeme
    pred_raw : List
        writeme

    """

    true_masks = th.from_numpy(np.stack(img.ravel() for img in true_masks))
    pred_raw   = th.from_numpy(np.stack(p.ravel() for p in pred_raw))
    pred_mask  = th.round(pred_raw)

    bce = F.binary_cross_entropy(V(pred_raw), V(true_masks)).data[0]
    acc = th.mean((pred_mask == true_masks).type_as(pred_raw).mean(1))
    l2  = th.mean((pred_raw - true_masks)**2)

    msg = f"""
Binary Cross Entropy: {bce}
Accuracy: {acc}
Euclidean Distance: {l2}
"""
    print(msg)

    confmat = confusion_matrix(true_masks.numpy().ravel(), pred_mask.numpy().ravel())
    plot_confusion_matrix(confmat)
