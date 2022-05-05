from Ottergrad.autograd import Func, checktensor
import Ottergrad.otternumpy as on
from Ottergrad.utils import getdtype


class _Sigmoid(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        Sigmoid function
        """
        tensor = getdtype()(1) / (getdtype()(1) + on.exp(-x))
        self.setroot(tensor)
        return tensor


@checktensor
def Sigmoid(x):
    func = _Sigmoid()
    return func(x)


class _ReLU(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        :param x: ndarray,
        :return:
        """
        tensor = on.maximum(0, x)
        self.setroot(tensor)
        return tensor


@checktensor
def ReLU(x):
    func = _ReLU()
    return func(x)


class _LeakyReLU(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        tensor = on.where(x > 0, x, x * 0.01)
        self.setroot(tensor)
        return tensor


@checktensor
def LeakyReLU(x):
    func = _LeakyReLU()
    return func(x)


class _tanh(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        tensor = on.tanh(x)
        self.setroot(tensor)
        return tensor


@checktensor
def tanh(x):
    func = _tanh()
    return func(x)


@checktensor
def none(x):
    return x
