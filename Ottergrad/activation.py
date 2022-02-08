
import Ottergrad.autograd as ag
from Ottergrad.autograd import Func
import Ottergrad.otternumpy as otnp
from Ottergrad.autograd import checktensor


class _Sigmoid(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        Sigmoid function
        """
        func = 1.0 / (1.0 + otnp.exp(-x))
        self.setroot(func)
        return func


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
        func = ag.Func(otnp.maximum(0, x))
        self.setroot(func)
        return func


@checktensor
def ReLU(x):
    func = _ReLU()
    return func(x)


class _LeakyReLU(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        func = ag.Func(otnp.where(x > 0, x, x * 0.01))
        self.setroot(func)
        return func


def LeakyReLU(x):
    func = _LeakyReLU()
    return func(x)


class _tanh(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        func = ag.Func(otnp.tanh(x))
        self.setroot(func)
        return func


@checktensor
def tanh(x):
    func = _tanh()
    return func(x)


@checktensor
def none(x):
    return x
