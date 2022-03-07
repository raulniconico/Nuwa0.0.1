from Ottergrad.autograd import Tensor, Func, checktensor
import Ottergrad.otternumpy as on
from Ottergrad.utils import getepsilon


class _norm(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor):
        sigma_2 = on.var(x)
        sigma_2.type = "sigma_2"
        mu = on.mean(x)
        mu.type = "mu"
        x_bar = (x - mu) / on.sqrt(sigma_2 + getepsilon())
        x_bar.type = "x_bar"
        return x_bar


@checktensor
def norm(x):
    func = _norm()
    return func(x)


class _Sigmoid(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        Sigmoid function
        """
        func = 1.0 / (1.0 + on.exp(-x))
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
        func = Func(on.maximum(0, x))
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
        func = Func(on.where(x > 0, x, x * 0.01))
        self.setroot(func)
        return func


@checktensor
def LeakyReLU(x):
    func = _LeakyReLU()
    return func(x)


class _tanh(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        func = Func(on.tanh(x))
        self.setroot(func)
        return func


@checktensor
def tanh(x):
    func = _tanh()
    return func(x)


@checktensor
def none(x):
    return x
