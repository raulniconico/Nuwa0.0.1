import copy

import numpy as np

from Ottergrad.utils import getdtype
from Ottergrad.autograd import Tensor, Func, checktensor
from Ottergrad.utils import checkgradisnone


class _dot(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        tensor = Tensor()
        tensor.type = np.dot
        tensor.left = x
        tensor.right = y
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        self.root = tensor
        self.input = tensor
        return tensor

    @staticmethod
    def _forwardfunc(node: Tensor):
        node.setdata(np.dot(node.getleft().getdata(), node.getright().getdata()))

    @checkgradisnone
    def gradient(self):
        self.root.getleft().setgrad(self.root.getgrad() + np.dot(self.root.getgrad(), self.root.getright().getdata().T))
        self.root.getright().setgrad(self.root.getgrad() + np.dot(self.root.getleft().getdata().T, self.root.getgrad()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + np.dot(node.getgrad(), node.getright().getdata().T))
        node.getright().setgrad(node.getright().getgrad() + np.dot(node.getleft().getdata().T, node.getgrad()))


@checktensor
def dot(x, y):
    func = _dot()
    return func(x, y)


class _sum(Func):
    def __init__(self, x: Tensor = None, axis=0, **kwargs):
        super().__init__()
        self.x = x
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, x: Tensor, axis=0, **kwargs):
        self.x = x
        self.axis = axis
        self.kwargs = kwargs

        tensor = Tensor()
        tensor.left = x
        tensor.type = np.sum
        tensor.setargs({axis})
        tensor.setkwargs(kwargs)
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)

        return tensor

    @staticmethod
    def _forwardfunc(node: Tensor):
        if node.getkwargs() != []:

            node.setdata(np.sum(node.getleft().getdata(), *node.getargs(), **node.getkwargs()))
        else:
            node.setdata(np.sum(node.getleft().getdata(), *node.getargs()))

    def gradient(self):
        self.root.getleft().setgrad(np.sum(self.root.getgrad(), *self.root.getargs()))

    @staticmethod
    @checkgradisnone
    def _gradient(node: Tensor):
        node.getleft().setgrad(node.getleft().getgrad() +
                               np.tile(np.sum(node.getgrad(), *node.getargs()), node.getleft().getdata().shape))


def sum(x: Tensor, axis: int = 0, **kwargs):
    func = _sum()
    return func(x, axis, **kwargs)


class _ones(Func):
    def __init__(self, shape=None, dtype=getdtype()):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def __call__(self, shape, dtype=getdtype()):
        tensor = Tensor()
        tensor.type = np.ones
        tensor.isgrad = False
        tensor.setargs({shape, dtype})
        tensor.setgradfunc(self._gradient)
        self.root = tensor
        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.ones(*node.gettype()))

    def gradient(self):
        pass

    @staticmethod
    def _gradient(node):
        pass


def ones(shape, dtype=getdtype()):
    func = _ones()
    return func(shape, dtype)


class _shape(Func):
    def __init__(self, x=None):
        super().__init__()
        self.x = x

    def __call__(self, x):
        self.x = x
        return np.shape(x.getdata())

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.shape(node.getdata()))

    def gradient(self):
        pass

    @staticmethod
    def _gradient(node):
        pass


@checktensor
def shape(x):
    func = _shape()
    return func(x)


class _exp(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x, **kwargs):
        tensor = Tensor()
        tensor.left = x
        tensor.type = np.exp
        tensor.setkwargs(kwargs)
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        self.root = tensor
        return tensor

    @staticmethod
    def _forwardfunc(node):
        if node.getkwargs() != []:
            node.setdata(np.exp(node.getleft().getdata(), **node.getkwargs()))
        else:
            node.setdata(np.exp(node.getleft().getdata()))

    def gradient(self):
        self.root.getleft().setgrad(self.root.getleft().getgrad() + self.root.getgrad())

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + np.dot(node.getgrad(), node.getdata()))


def exp(x, **kwargs):
    func = _exp()
    return func(x, **kwargs)


class _maximum(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y, **kwargs):
        tensor = Tensor()
        tensor.left = x
        tensor.right = y
        tensor.setkwargs(kwargs)
        tensor.type = np.maximum
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        self.root = tensor
        return tensor

    @staticmethod
    def _forwardfunc(node):
        if node.getkwargs() != []:
            node.setdata(np.maximum(node.getleft().getdata(), node.getright().getdata(), **node.getkwargs()))
        else:
            node.setdata(np.maximum(node.getleft().getdata(), node.getright().getdata()))

    def gradient(self):
        #     grad = copy.deepcopy(node.getgrad())
        #     grad[grad <= 0] = 0
        #     node.getleft().grad = grad
        self.root.getleft().setgrad(np.minimum(self.root.getleft().getdata(), self.root.getright().getdata()))
        self.root.getright().setgrad(np.minimum(self.root.getright().getdata(), self.root.getleft().getdata()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(
            node.getleft().getgrad() + np.minimum(node.getleft().getdata(), node.getright().getdata()))
        node.getright().setgrad(
            node.getleft().getgrad() + np.minimum(node.getright().getdata(), node.getleft().getdata()))


def maximum(x: [Tensor, int, float, np.ndarray], y: [Tensor, int, float, np.ndarray], **kwargs):
    func = _maximum()
    return func(x, y, **kwargs)


class _tanh(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        tensor = Tensor()
        tensor.setleft(x)
        tensor.type = np.tanh
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.tanh(node.getleft().getdata()))

    def gradient(self):
        self.getroot().getleft().setgrad(self.getroot().getleft().getgrad() +
                                         np.multiply((np.ones(self.getroot().getgrad().shape) -
                                                      np.tanh(self.getroot().getgrad()) ** 2),
                                                     self.getroot().getgrad()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() +
                               np.multiply((np.ones(node.getgrad().shape) - np.tanh(node.getgrad()) ** 2),
                                           node.getgrad()))


@checktensor
def tanh(x):
    func = _tanh()
    return func(x)


class _sin(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        tensor = Tensor()
        tensor.setleft(x)
        tensor.type = np.sin
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.sin(node.getleft().getdata()))

    def gradient(self):
        self.getroot().getleft().setgrad(
            np.dot(self.getroot().getleft().getgrad() + self.root.getgrad().T, np.cos(self.root.getleft().getdata())))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + np.dot(node.getgrad().T, np.cos(node.getleft().getdata())))


@checktensor
def sin(x):
    func = _sin()
    return func(x)


class _cos(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        tensor = Tensor()
        tensor.setleft(x)
        tensor.type = np.cos
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.cos(node.getleft().getdata()))

    def gradient(self):
        self.getroot().getleft().setgrad(self.getroot().getleft().getgrad() +
                                         np.dot(self.root.getgrad().T, -np.sin(self.root.getleft().getdata())))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + np.dot(node.getgrad().T, -np.sin(node.getleft().getdata())))


@checktensor
def cos(x):
    func = _cos()
    return func(x)


class _tan(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        tensor = Tensor()
        tensor.setleft(x)
        tensor.type = np.tan
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.tan(node.getleft().getdata()))

    def gradient(self):
        self.getroot().getleft().setgrad(self.getroot().getleft().getgrad() +
                                         np.dot(self.root.getgrad().T, 1 - np.tanh(self.root.getleft().getdata())**2))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() +
                               np.dot(node.getgrad().T, 1 - np.tanh(node.getleft().getdata())**2))


@checktensor
def tan(x):
    func = _tan()
    return func(x)


class _where(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, condition: Tensor, x: Tensor = None, y: Tensor = None):
        tensor = Tensor()
        tensor.setleft(Func(condition).getinput())
        tensor.type = np.where
        tensor.setargs([Func(condition), Func(x), Func(y)])
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        return tensor

    @staticmethod
    def _forwardfunc(node):
        args = []
        for arg in node.getargs():
            arg.forward()
            args.append(arg.getroot().getdata())
        node.setdata(np.where(*args))

    def gradient(self):
        self.root.getleft().setgrad()

    @staticmethod
    def _gradient(node):
        node.getargs()[1].getroot().setgrad(node.getgrad())
        node.getargs()[2].getroot().setgrad(node.getgrad())

        x = copy.deepcopy(node.getargs()[1])
        y = copy.deepcopy(node.getargs()[2])
        x.backward()
        y.backward()
        node.getleft().setgrad(np.where(node.getargs()[0].getroot().getdata(),
                                        x.getinput().getgrad(), y.getinput().getgrad()))


@checktensor
def where(contition, x, y):
    func = _where()
    return func(contition, x, y)
