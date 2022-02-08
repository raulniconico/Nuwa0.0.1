import copy
from operator import __matmul__
import numpy as np
from numpy import ndarray
from inspect import signature

import Ottergrad


def checktensor(operator):
    def check(*args):
        arg_list = []
        for arg in args:
            if type(arg) is not Tensor:
                arg_list.append(Tensor(arg))
            else:
                arg_list.append(arg)
        return operator(*arg_list)

    return check


from Ottergrad.utils import checkgradisnone


class Tensor:
    OVERLOADABLE_OPERATORS = {
        # Binary.
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__div__",
        "__rdiv__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__mod__",
        "__rmod__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__ne__",
        "__eq__",
        "__and__",
        "__rand__",
        "__or__",
        "__ror__",
        "__xor__",
        "__rxor__",
        "__getitem__",
        "__pow__",
        "__rpow__",
        # Unary.
        "__invert__",
        "__neg__",
        "__abs__",
        "__matmul__",
        "__rmatmul__"
    }

    def __init__(self, data=None, isgrad=True, *args, **kwargs):
        self.data = None
        self.grad = None
        self.left = None
        self.right = None
        self.type = "weight"
        self.momentum = None
        self.isgrad = isgrad
        self.isconst = False
        self.args = None
        self.kwargs = None
        self.gradfunc = None
        self.forwardfunc = None

        self.setargs(list(args))
        self.setkwargs(list(kwargs))
        self.setdata(data)

    @checktensor
    def __add__(self, other):
        def _forwardfunc(node):
            node.setdata(np.add(node.getleft().getdata(), node.getright().getdata()))

        @checkgradisnone
        def _gradient(node):
            if node.getleft().getisconst():
                node.getleft().setgrad(node.getleft().getgrad() + np.sum(node.getgrad(), axis=0))
            else:
                node.getleft().setgrad(node.getleft().getgrad() + node.getgrad())

            if node.getright().getisconst():
                node.getright().setgrad(node.getright().getgrad() + np.sum(node.getgrad(), axis=0))
            else:
                node.getright().setgrad(node.getright().getgrad() + node.getgrad())

        tensor = Tensor()
        tensor.type = np.add
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checktensor
    def __radd__(self, other):
        def _forwardfunc(node):
            node.setdata(np.add(node.getleft().getdata(), node.getright().getdata()))

        @checkgradisnone
        def _gradient(node):
            if node.getleft().getisconst():
                node.getleft().setgrad(node.getleft().getgrad() + np.sum(node.getgrad(), axis=0))
            else:
                node.getleft().setgrad(node.getleft().getgrad() + node.getgrad())
            if node.getright().getisconst():
                node.getright().setgrad(np.sum(node.getright().getgrad() + node.getgrad(), axis=0))
            else:
                node.getright().setgrad(node.getright().getgrad() + node.getgrad())

        tensor = Tensor()
        tensor.type = ndarray.__radd__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checktensor
    def __sub__(self, other):
        def _forwardfunc(node):
            node.setdata(np.subtract(node.getleft().getdata(), node.getright().getdata()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(node.getleft().getgrad() + node.getgrad())
            node.getright().setgrad(node.getright().getgrad() + -node.getgrad())

        tensor = Tensor()
        tensor.type = ndarray.__sub__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checktensor
    def __rsub__(self, other):
        def _forwardfunc(node):
            node.setdata(np.subtract(node.getright().getdata(), node.getleft().getdata()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(node.getleft().getgrad() + -node.getgrad())
            node.getright().setgrad(node.getright().getgrad() + node.getgrad())

        tensor = Tensor()
        tensor.type = ndarray.__rsub__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checktensor
    def __mul__(self, other):
        def _forwardfunc(node):
            node.setdata(np.multiply(node.getleft().getdata(), node.getright().getdata()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(node.getleft().getgrad() + np.multiply(node.getright().getdata(), node.getgrad()))
            node.getright().setgrad(node.getright().getgrad() + np.multiply(node.getleft().getdata(), node.getgrad()))

        tensor = Tensor()
        tensor.type = ndarray.__mul__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checktensor
    def __rmul__(self, other):
        def _forwardfunc(node):
            node.setdata(np.multiply(node.getleft().getdata(), node.getright().getdata()))

        def _gradient(node):
            node.getleft().setgrad(node.getleft().getgrad() + np.multiply(node.getright().getdata(), node.getgrad()))
            node.getright().setgrad(node.getright().getgrad() + np.multiply(node.getleft().getdata(), node.getgrad()))

        tensor = Tensor()
        tensor.type = ndarray.__rmul__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checktensor
    def __truediv__(self, other):
        def _forwardfunc(node):
            node.setdata(np.divide(node.getleft().getdata(), node.getright().getdata()))

        @checkgradisnone
        def _gradient(node: Tensor):
            node.getleft().setgrad(node.getleft().getgrad() +
                                   np.dot(np.divide(1, node.getright().getdata()), node.getgrad()))
            node.getright().setgrad(
                node.getright().getgrad() + np.dot(node.getleft().getdata(), -node.getright().getdata() ** -2))

        tensor = Tensor()
        tensor.type = ndarray.__truediv__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setforwardfunc(_forwardfunc)
        tensor.setgradfunc(_gradient)
        return tensor

    @checktensor
    def __rtruediv__(self, other):
        def _forwardfunc(node):
            node.setdata(np.divide(node.getright().getdata(), node.getleft().getdata()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(
                node.getleft().getgrad() + np.dot(node.getleft().getdata(), -node.getright().getdata() ** -2))
            node.getright().setgrad(
                node.getright().getgrad() + np.dot(np.divide(1, node.getright().getdata()), node.getgrad()))

        tensor = Tensor()
        tensor.type = ndarray.__rtruediv__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checktensor
    def __pow__(self, power, modulo=None):
        def _forward(node):
            node.setdata(np.power(node.getleft().getdata(), node.getright().getdata()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(node.getleft().getgrad() +
                                   np.dot(node.getgrad().T, np.dot(node.getright().getdata(),
                                                                   node.getleft().getdata() ** (
                                                                           node.getright().getdata() - 1))))
            node.getright().setgrad(node.getright().getgrad() +
                                    np.dot(node.getgrad()).T, np.dot(node.getgrad(), np.log(node.getleft().getdata())))

        tensor = Tensor()
        tensor.type = ndarray.__pow__
        tensor.setleft(self)
        tensor.setright(power)
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checktensor
    def __rpow__(self, other):
        def _forward(node):
            node.setdata(np.power(node.getright().getdata(), node.getleft().getdata()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(node.getleft().getgrad() +
                                   np.dot(node.getgrad()).T, np.dot(node.getgrad(), np.log(node.getleft().getdata())))

            node.getright().setgrad(node.getright().getgrad() +
                                    np.dot(node.getgrad().T, np.dot(node.getright().getdata(),
                                                                    node.getleft().getdata() ** (
                                                                            node.getright().getdata() - 1))))

        tensor = Tensor()
        tensor.setleft(self)
        tensor.setright(other)
        tensor.type = ndarray.__rpow__
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checktensor
    def __neg__(self):
        def _forward(node):
            node.setdata(-node.getleft().getdata())

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(node.getleft().getgrad() - node.getgrad())

        tensor = Tensor()
        tensor.type = ndarray.__neg__
        tensor.setleft(self)
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checktensor
    def __matmul__(self, other):

        return Ottergrad.otternumpy.dot(self, other)

    @checktensor
    def __gt__(self, other):
        def _forward(node):
            node.setdata(np.greater(node.getleft().getdata(), node.getright().getdata()))

        tensor = Tensor()
        tensor.type = np.greater
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setforwardfunc(_forward)
        return tensor

    @checktensor
    def __lt__(self, other):
        def _forward(node):
            node.setdata(np.less(node.getleft().getdata(), node.getright().getdata()))

        tensor = Tensor()
        tensor.type = np.less
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setforwardfunc(_forward)
        return tensor

    def getdata(self):
        return self.data

    def setdata(self, data):
        self.data = data

    def getgrad(self):
        return self.grad

    def setgrad(self, grad):
        self.grad = grad

    def getleft(self):
        return self.left

    def setleft(self, left):
        self.left = left

    def getright(self):
        return self.right

    def setright(self, right):
        self.right = right

    def gettype(self):
        return self.type

    def getmomentum(self):
        return self.momentum

    def getisgrad(self):
        return self.isgrad

    def getisconst(self):
        return self.isconst

    def getargs(self):
        return self.args

    def setargs(self, args):
        if args == []:
            self.args = None
        else:
            self.args = args

    def getkwargs(self):
        return self.kwargs

    def setkwargs(self, kwargs):
        if kwargs == []:
            self.kwargs = None
        else:
            self.kwargs = kwargs

    def getgradfunc(self):
        return self.gradfunc

    def setgradfunc(self, gradfunc):
        self.gradfunc = gradfunc

    def getforwardfunc(self):
        return self.forwardfunc

    def setforwardfunc(self, func):
        self.forwardfunc = func


class Graph:
    def __init__(self, root=None, *args, **kwargs):
        self.root = root
        self.leaf = None

    def join(self, other):
        other.leaf.left = self
        return other.root

    def get_input(self):
        node = self.root
        while True:
            if node.getleft() is None:
                return node
            else:
                node = node.getleft()

    def getroot(self):
        return self.root

    def backpropagation(self):

        def backward(node):
            epsilon = Ottergrad.utils.getepsilon()

            if node.getleft() is not None and node.getgradfunc() is not None:
                gradfunc = node.getgradfunc()
                gradfunc(node)
                backward(node.getleft())
                if node.getright() is not None:
                    backward(node.getright())
            else:
                return

        if self.getroot().getgrad() is None:
            self.getroot().setgrad(np.ones(np.shape(self.getroot().getdata())))

        backward(self.getroot())
        return self.root

    def forwardpropagation(self):
        def forward(node: Tensor):

            # if node.getargs() is not None:
            #     try:
            #         args = []
            #         for arg in node.getargs():
            #             if type(arg) is Tensor:
            #                 forward(arg)
            #                 args.append(arg.getdata())
            #             else:
            #                 args.append(arg)
            #         node.args = args
            #
            #     except:
            #
            #         if type(node.getargs()) is Tensor:
            #             forward(node.getargs())
            #             node.args = node.getargs().getdata()
            #
            # if node.getkwargs() is not None:
            #     try:
            #         kwargs = []
            #         for kwarg in node.getkwargs():
            #             if type(kwarg) is Tensor:
            #                 forward(kwarg)
            #                 kwargs.append(kwarg.getdata())
            #             else:
            #                 kwargs.append(kwarg)
            #         node.kwargs = kwargs
            #
            #     except:
            #         if type(node.getkwargs()) is Tensor:
            #             forward(node.getkwargs())
            #             node.kwargs = node.getkwargs().getdata()
            #
            # if type(node.getdata()) is Tensor:
            #     forward(node.getdata())
            #     node.data = node.getdata().getdata()

            if node.getleft() is not None:
                if node.getleft().getdata() is None:
                    forward(node.getleft())

            if node.getright() is not None:
                if node.getright().getdata() is None:
                    forward(node.getright())

            func = node.getforwardfunc()
            if func is not None:
                func(node)
            else:
                print(str(node.getdata()) + " has no forward function")
            return

        forward(self.getroot())
        return self.getroot()


class Func:
    def __init__(self, root=None):
        self.graph = None
        self.root = None
        self.input = None
        self.weight_list = None

        self.setroot(root)

    @classmethod
    def fromfunction(cls, func):
        param_list = []
        for param in signature(func).parameters.keys():
            locals()[param] = Tensor()
            param_list.append(locals()[param])
        func = func(*param_list)
        func_tensor = cls(func)
        func_tensor.weight_list = param_list
        return func_tensor

    def getinput(self):
        if self.input is None:
            try:
                return self.getgraph().get_input()
            except:
                raise ValueError

    def setinput(self, input):
        self.input = input
        self.graph = Graph(input)

    def getroot(self):
        return self.root

    def setroot(self, root):
        self.root = root
        if self.root is not None:
            self.graph = Graph(self.root)

    def forward(self):
        result = self.graph.forwardpropagation()
        return result.getdata()

    def backward(self):
        root = self.graph.backpropagation()
        return root

    def getgraph(self):
        return self.graph


class Optim:
    def __init__(self, func: Func = None, optimizer=None, batch_size=32, epoch=20000, lr=0.0001,
                 decay_rate=0, pretrained=None):
        self.graph = Graph()
        self.func = func
        self.weight_list = None
        self.gradient_list = None
        self.loss_list = None
        self.passer_list = None

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr

        self.decay_rate = decay_rate
        self.pretrained = pretrained

    # def forward(self):
