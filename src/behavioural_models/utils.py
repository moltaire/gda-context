#!usr/bin/python
import numpy as np


def softmax(x):
    """
    Robust softmax transform for two-dimensional input along axis 1.
    """
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    out = e_x / np.sum(e_x, axis=1, keepdims=True)
    return out


def choose(choice_probabilities):
    """
    Make row-wise choices of columns according to choice probabilities.
    Basically a row-wise matrix implementation of np.random.choice
    Source: https://stackoverflow.com/a/34190035
    """
    s = choice_probabilities.cumsum(axis=1)
    r = np.random.rand(choice_probabilities.shape[0], 1)
    k = (s < r).sum(axis=1)
    return k
