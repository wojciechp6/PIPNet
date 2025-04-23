import itertools

import numpy as np
import torch
from matplotlib import pyplot as plt

def plot_batch(tensor: torch.Tensor, samples = 8, title: str = None):
    batch_size = tensor.shape[0]
    if batch_size > samples:
        indices = torch.linspace(0, batch_size - 1, steps=samples).long()
        sample_tensor = tensor[indices]
    else:
        sample_tensor = tensor

    stats = [
        ("Max over batch", torch.max(tensor, dim=0).values),
        ("Min over batch", torch.min(tensor, dim=0).values),
        ("Mean over batch", torch.mean(tensor, dim=0))
    ]

    fig = plt.figure(figsize=(12, 10))
    if title:
        fig.suptitle(title, fontsize=16)

    ax = plt.subplot(2, 2, 1)
    for s in sample_tensor:
        plot_tensor(s, subplot=True)
    ax.set_title("Samples")

    for i, (stat_title, stat_tensor) in enumerate(stats, start=2):
        ax = plt.subplot(2, 2, i)
        plot_tensor(stat_tensor, subplot=True)
        ax.set_title(stat_title)

    plt.tight_layout()
    return fig


def plot_tensor(tensor: torch.Tensor, title: str = None, samples: int = 50, subplot: bool = False):
    array = tensor.cpu().numpy()
    array = smooth_array(array, samples)
    if not subplot:
        plt.figure()
    plt.plot(array)
    if title is not None:
        plt.title(title)
    return plt.gcf()

def smooth_array(array: np.ndarray, samples: int = 50):
    window_size = len(array) // samples
    kernel = np.ones(window_size) / window_size
    smoothed_tensor = np.convolve(array, kernel, mode='valid')
    return smoothed_tensor

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure