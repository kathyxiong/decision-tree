"""
This is rev0 implementation of a decision tree algorithm.
Rev0 scope:
- binary classification tasks only
- binary features only
- split criterion is entropy
"""
import numpy as np

def calc_entropy(y):
    """
    Calculates entropy give a binary class vector
    """
    entropy = 0
    if len(y) > 0:
        p = sum(y)/len(y)
        if p != 0 and p != 1:
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return entropy


