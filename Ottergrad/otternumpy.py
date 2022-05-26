import copy

import numpy as np

from utils.utils import getdtype
from Ottergrad.autograd import Tensor, Func, checkTensor
from Ottergrad.utils import *


def ndot(a, b):
    res = np.dot(a, b)
    return res


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

        res = ndot(node.getleft().getdata(), node.getright().getdata())
        node.setdata(res)

    @checkgradisnone
    def gradient(self):
        grad_l = ndot(self.root.getgrad(), self.root.getright().getdata().T)
        grad_r = ndot(self.root.getleft().getdata().T, self.root.getgrad())
        self.root.getleft().setgrad(self.root.getgrad() + grad_l)
        self.root.getright().setgrad(self.root.getgrad() + grad_r)

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + ndot(node.getgrad(), node.getright().getdata().T))
        node.getright().setgrad(node.getright().getgrad() + ndot(node.getleft().getdata().T, node.getgrad()))


@checkTensor
def dot(x, y):
    func = _dot()
    return func(x, y)


class _abs(Func):
    def __init__(self, x=None):
        super().__init__()
        self.x = x

    def __call__(self, x):
        self.x = x
        tensor = Tensor()
        tensor.left = x
        tensor.type = np.abs
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)

        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.abs(node.getleft().getdata(), dtype=getdtype()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        condlist = [node.getleft().getdata() < 0, node.getleft().getdata() > 0]
        choicelist = [-1, 1]
        node.getleft().setgrad(node.getleft().getgrad() + np.select(condlist, choicelist))


def nabs(x):
    res = np.abs(x)
    return res


@checkTensor
def abs(x):
    func = _abs()
    return func(x)


class _sum(Func):
    def __init__(self, x: Tensor = None, axis=None):
        super().__init__()
        self.x = x
        self.axis = axis

    def __call__(self, x: Tensor, axis=None):
        self.x = x
        self.axis = axis

        tensor = Tensor()
        tensor.left = x
        tensor.type = np.sum
        tensor.setkwargs({'axis': axis})
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)

        return tensor

    @staticmethod
    def _forwardfunc(node: Tensor):
        node.setdata(nsum(node.getleft().getdata(), **node.getkwargs()))

    def gradient(self):
        self.root.getleft().setgrad(self.root.getleft().getgrad() + nsum(self.root.getgrad(), *self.root.getargs()))

    @staticmethod
    @checkgradisnone
    def _gradient(node: Tensor):
        node.getleft().setgrad(node.getleft().getgrad()
                               + node.getgrad() * np.ones(node.getleft().getdata().shape))


def nsum(x, axis):
    res = np.sum(x, axis)
    return res


def sum(x: Tensor, axis: int = None):
    func = _sum()
    return func(x, axis)


class _concatenate(Func):
    def __init__(self, sets=None, axix=0):
        super().__init__()

    def __call__(self, sets: tuple, axis=0):
        assert len(sets) > 1, "must have two or more sets"
        isalldata = True
        for set in sets:
            assert type(set) is Tensor, "set type must be Tensor"
            isalldata = isalldata and (set.getdata() is not None)

        if isalldata:
            result = sets[0]
            for set in sets:
                nconcatenate((result, set), axis)
            return result

        else:
            result = sets[0]
            for set in sets[1:]:
                tensor = Tensor()
                tensor.left = result
                tensor.right = set
                tensor.type = np.concatenate
                tensor.setkwargs({'axis', axis})
                tensor.setforwardfunc(self._forwardfunc)
                tensor.setgradfunc(self._gradient)
                result = tensor
            return tensor

    @staticmethod
    def _forwardfunc(node: Tensor):
        node.setdata(nconcatenate((node.getleft().getdata(), node.getright().getdata()), **node.getkwargs()))

    @staticmethod
    @checkgradisnone
    def _gradient(node: Tensor):
        pass


# @pyjit
def nconcatenate(x, y, **kwargs):
    return np.concatenate(x, y, **kwargs)


def concatenate(sets: tuple, axis=0):
    for set in sets:
        assert type(set) is Tensor, "set type must be Tensor"
    func = _concatenate()
    return func(sets, axis)


class _split(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor, indices, axis=0):
        if x.getdata() is not None:
            return np.split(x.getdata(), indices, axis)
        else:
            tensor = Tensor()
            tensor.left = x
            tensor.type = np.split
            tensor.setkwargs({"indices_or_sections": indices, 'axis': axis})
            tensor.setforwardfunc(self._forwardfunc)
            tensor.setgradfunc(self._gradient)
            return tensor

    @staticmethod
    def _forwardfunc(node: Tensor):
        node.setdata(nsplit(node.getleft().getdata(), **node.getkwargs()))

    @staticmethod
    @checkgradisnone
    def _gradient(node: Tensor):
        pass


# @pyjit
def nsplit(x, indices, axis=0):
    return np.split(x, indices, axis=axis)


def split(x, indices, axis=0):
    func = _split()
    return func(x, indices, axis)


class _ones(Func):
    def __init__(self, shape=None, dtype=getdtype()):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def __call__(self, shape, dtype=getdtype()):
        tensor = Tensor()
        tensor.type = np.ones
        tensor.isgrad = False
        tensor.setargs({"shape": shape, "dtype": dtype})
        tensor.setforwardfunc(self._forwardfunc)
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
        if x.getdata() is not None:
            return np.shape(x.getdata())
        else:
            tensor = Tensor()
            tensor.setleft(x)
            tensor.type = np.shape
            tensor.isgrad = False
            tensor.setforwardfunc(self._forwardfunc)
            tensor.setgradfunc(self._gradient)
            return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.shape(node.getleft().getdata()))

    def gradient(self):
        pass

    @staticmethod
    def _gradient(*args):
        pass


@checkTensor
def shape(x):
    func = _shape()
    return func(x)


class _take(Func):
    def __init__(self, x=None, choosen=None, axis=0):
        super().__init__()
        self.x = x
        self.choosen = choosen
        self.axis = axis

    def __call__(self, x, indices=None, axis=0):
        self.x = x
        self.indices = indices
        self.axis = axis

        if x.getdata() is not None:
            return np.take(x.getdata(), indices, axis)
        else:
            tensor = Tensor()
            tensor.setleft(x)
            tensor.type = np.take
            tensor.isgrad = False
            tensor.setkwargs({"indices": indices, "axis": axis})
            tensor.setforwardfunc(self._forwardfunc)
            tensor.setgradfunc(self._gradient)
            return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(ntake(node.getleft().getdata(), **node.getkwargs()))

    def gradient(self):
        pass

    @staticmethod
    def _gradient(*args):
        pass


# @pyjit
def ntake(x, indices=None, axis=0):
    return np.take(x, indices=indices, axis=axis)


def take(x: Tensor, choosen, axis=0):
    func = _take()
    return func(x, choosen, axis)


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
            node.setdata(nexp(node.getleft().getdata(), **node.getkwargs()))
        else:
            node.setdata(nexp(node.getleft().getdata()))

    def gradient(self):
        self.root.getleft().setgrad(self.root.getleft().getgrad() + self.root.getgrad())

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + (node.getgrad() * nexp(node.getleft().getdata())))

def nexp(x):
    return np.exp(x)


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
            node.setdata(nmaximum(node.getleft().getdata(), node.getright().getdata(), **node.getkwargs()))
        else:
            node.setdata(nmaximum(node.getleft().getdata(), node.getright().getdata()))

    def gradient(self):
        #     grad = copy.deepcopy(node.getgrad())
        #     grad[grad <= 0] = 0
        #     node.getleft().grad = grad
        self.root.getleft().setgrad(nminimum(self.root.getleft().getdata(), self.root.getright().getdata()))
        self.root.getright().setgrad(nminimum(self.root.getright().getdata(), self.root.getleft().getdata()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(
            node.getleft().getgrad() + nminimum(node.getleft().getdata(), node.getright().getdata()))
        node.getright().setgrad(
            node.getright().getgrad() + nminimum(node.getright().getdata(), node.getleft().getdata()))


# @pyjit
def nmaximum(x, y):
    return np.maximum(x, y)


# @pyjit
def nminimum(x, y):
    return np.minimum(x, y)


def maximum(x: [Tensor, int, float, np.ndarray], y: [Tensor, int, float, np.ndarray], **kwargs):
    if type(x) is not Tensor:
        x = Tensor(x)
        x.isgrad = False
    if type(y) is not Tensor:
        y = Tensor(y)
        y.isgrad = False

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
                                                      ntanh(self.getroot().getgrad()) ** 2),
                                                     self.getroot().getgrad()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() +
                               np.multiply((np.ones(node.getgrad().shape) - ntanh(node.getgrad()) ** 2),
                                           node.getgrad()))


def ntanh(x):
    return np.tanh(x)


@checkTensor
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
        self.getroot().getleft().setgrad(self.getroot().getleft().getgrad() +
                                         np.multiply(self.root.getgrad(), np.cos(self.root.getleft().getdata())))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + np.multiply(node.getgrad(), np.cos(node.getleft().getdata())))


def nsin(x):
    return np.sin(x)


@checkTensor
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
                                         np.multiply(self.root.getgrad(), -np.sin(self.root.getleft().getdata())))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() +
                               np.multiply(node.getgrad(), -np.sin(node.getleft().getdata())))


def ncos(x):
    return np.cos(x)


@checkTensor
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
                                         np.multiply(self.root.getgrad(), 1 -
                                                     np.tanh(self.root.getleft().getdata()) ** 2))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() +
                               np.multiply(node.getgrad(), 1 - np.tanh(node.getleft().getdata()) ** 2))


def ntan(x):
    return np.tan(x)


@checkTensor
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

    @checkgradisnone
    @staticmethod
    def _gradient(node):
        node.getargs()[1].getroot().setgrad(node.getgrad())
        node.getargs()[2].getroot().setgrad(node.getgrad())

        x = copy.deepcopy(node.getargs()[1])
        y = copy.deepcopy(node.getargs()[2])

        x.backward()
        y.backward()

        node.getleft().setgrad(np.where(node.getargs()[0].getinput().getdata(),
                                        x.getinput().getgrad(), y.getinput().getgrad()))


def nwhere(contition, x, y):
    return np.where(contition, x, y)


@checkTensor
def where(contition, x, y):
    func = _where()
    return func(contition, x, y)


class _mean(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor = None, axis=0):
        tensor = Tensor()
        tensor.setleft(x)
        tensor.type = np.mean
        tensor.setkwargs({"axis": axis})
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.mean(node.getleft().getdata(), **node.getkwargs()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        shape = np.shape(node.getleft().getdata())
        node.getleft().setgrad(node.getleft().getgrad() + 1 / shape[0] * np.ones(shape) * node.getgrad())


def nmean(x, axis=0):
    return np.mean(x, axis=axis)


def mean(x: Tensor, axis=0):
    func = _mean()
    return func(x, axis)


class _var(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor = None, axis=0):
        tensor = 1 / shape(x)[0] * (x - mean(x, axis)) ** 2
        return tensor


@checkTensor
def var(x, axis=0):
    func = _var()
    return func(x, axis)


class _sqrt(Func):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor):
        self.x = x
        tensor = Tensor()
        tensor.setleft(x)
        tensor.type = np.sqrt
        tensor.setforwardfunc(self._forwardfunc)
        tensor.setgradfunc(self._gradient)
        return tensor

    @staticmethod
    def _forwardfunc(node):
        node.setdata(np.sqrt(node.getleft().getdata()))

    @staticmethod
    @checkgradisnone
    def _gradient(node):
        node.getleft().setgrad(node.getleft().getgrad() + 1 / 2 * node.getleft().getdata() ** (-1 / 2))


@checkTensor
def sqrt(x):
    func = _sqrt()
    return func(x)
