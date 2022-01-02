from dataloader import Dataset
from network import NN
from optimizers import Optimizer
from visualization import Visual

import numpy as np


def concatenate(datasets, axis=0):
    dataset = Dataset()
    for set in datasets:
        if dataset.getX() is None or dataset.getX() is None:
            dataset.add(set.getX(), set.gety())
        else:
            dataset.concatenate(set, axis=axis)
    dataset.distribute()

    return dataset
