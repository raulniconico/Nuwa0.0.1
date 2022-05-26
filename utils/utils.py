import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

dtype = "float32"
epsilon = 1e-8
device = 'cpu'


def getdtype():
    return dtype


def getepsilon():
    return epsilon


def getdevice():
    return device


def getsupportdevice():
    DEVICE = {
        "cpu",
        "gpu",
        "cuda",
    }
    return DEVICE


def getsupportdtype():
    DTYPE = {
        "float32",
        "float64",
    }
