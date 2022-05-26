from inspect import signature

import numpy as np
import skcuda.misc
from numpy import ndarray

import Ottergrad
import shell.shell
from Ottergrad.utils import checkgradisnone

from shell.utils import getcudadevice, checkDataDevice

import pycuda.gpuarray as gpuarray
from utils.utils import getepsilon, getdtype, getdevice, getsupportdevice

from skcuda import cublas
import skcuda.linalg as linalg
skcuda.linalg.init()

# set device the same as Nuwa
device = getdevice()
DEVICE = getsupportdevice()
dtype = getdtype()
if dtype == "float32":
    np_dtype = np.float32
    cuda_dtype = "single"
elif dtype == "float64":
    np_dtype = np.float64
    cuda_dtype = "double"
elif dtype == "float16":
    np.dtype = np.float16
    cuda_dtype = "half"


def checkTensor(operator):
    def check(*args):
        arg_list = []
        for arg in args:
            if type(arg) is not Tensor:
                arg_list.append(Tensor(arg))
            else:
                arg_list.append(arg)
        return operator(*arg_list)

    return check


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

    def __init__(self, data=None, isgrad=True, dtype=np_dtype, device="cpu", *args, **kwargs):
        assert device in DEVICE, "device must in Tensor.DEVICE"
        self.data = None
        self.dtype = dtype
        self.device = device
        self.grad = None
        self.momentum = None
        self.left = None
        self.right = None
        self.type = None
        self.momentum = None
        self.isgrad = isgrad
        self.isconst = False
        self.args = None
        self.kwargs = None
        self.gradfunc = None
        self.forwardfunc = None

        self.setdevice(device)
        self.setargs(list(args))
        self.setkwargs(list(kwargs))
        self.setdata(data)

    def __getitem__(self, item):
        forwardfunc = self.forwardfunc
        index = item

        def _forwardfunc(node: Tensor):
            forwardfunc(node)
            node.setdata(node.getdata()[index])

        def _gradient(*args):
            pass

        if self.data is not None:
            tensor = Tensor(self.data[item])
        else:
            tensor = Tensor()
        tensor.left = self
        tensor.type = self.type
        tensor.dtype = self.dtype
        tensor.device = self.device
        tensor.isgrad = False
        tensor.forwardfunc = _forwardfunc
        tensor.gradfunc = _gradient
        return tensor

    @checkTensor
    def __add__(self, other):

        def _forwardfunc(node: Tensor):
            node.setdata(np.add(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))

        @checkDataDevice
        def _forwardfunc_cuda(node: Tensor):
            node.setdata(node.getleft().getdata() + node.getright().getdata())

        @checkgradisnone
        def _gradient(node):
            if node.getleft().shape() != node.getright().shape():
                node.getleft().setgrad(node.getleft().getgrad() + np.sum(node.getgrad(), axis=0, dtype=getdtype()))
                node.getright().setgrad(node.getright().getgrad() + np.sum(node.getgrad(), axis=0, dtype=getdtype()))

            else:
                node.getleft().setgrad(np.add(node.getleft().getgrad(), node.getgrad(), dtype=getdtype()))
                node.getright().setgrad(np.add(node.getright().getgrad(), node.getgrad(), dtype=getdtype()))

        @checkgradisnone
        def _gradient_cuda(node):
            if node.getleft().shape() != node.getright().shape():
                node.getleft().setgrad(node.getleft().getgrad() + skcuda.misc.sum(node.getgrad(), axis=0))
                node.getright().setgrad(node.getright().getgrad() + skcuda.misc.sum(node.getgrad(), axis=0))

            else:
                node.getleft().setgrad(node.getleft().getgrad() + node.getgrad())
                node.getright().setgrad(node.getright().getgrad() + node.getgrad())

        tensor = Tensor()
        tensor.type = "add"
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setdevice(self.device)

        if self.device == "gpu" or self.device == "cuda":
            tensor.setforwardfunc(_forwardfunc_cuda)
            tensor.setgradfunc(_gradient_cuda)
        else:
            tensor.setforwardfunc(_forwardfunc)
            tensor.setgradfunc(_gradient)
        return tensor

    @checkTensor
    def __radd__(self, other):
        @checkDataDevice
        def _forwardfunc(node):
            node.setdata(np.add(node.getleft().getdata(), node.getright().getdata()))

        @checkgradisnone
        def _gradient(node):
            if node.getleft().getisconst():
                node.getleft().setgrad(node.getleft().getgrad() + np.sum(node.getgrad(), axis=0, dtype=getdtype()))
            else:
                node.getleft().setgrad(node.getleft().getgrad() + node.getgrad())
            if node.getright().getisconst():
                node.getright().setgrad(node.getright().getgrad() + np.sum(node.getgrad(), axis=0, dtype=getdtype()))
            else:
                node.getright().setgrad(node.getright().getgrad() + node.getgrad())

        tensor = Tensor()
        tensor.type = ndarray.__radd__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checkTensor
    def __sub__(self, other):
        def _forwardfunc(node):
            node.setdata(np.subtract(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(np.add(node.getleft().getgrad(), node.getgrad(), dtype=getdtype()))
            node.getright().setgrad(np.add(node.getright().getgrad(), -node.getgrad(), dtype=getdtype()))

        tensor = Tensor()
        tensor.type = ndarray.__sub__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checkTensor
    def __rsub__(self, other):
        def _forwardfunc(node):
            node.setdata(np.subtract(node.getright().getdata(), node.getleft().getdata(), dtype=getdtype()))

        @checkgradisnone
        def _gradient(node):
            node.getleft().setgrad(np.add(node.getleft().getgrad(), -node.getgrad(), dtype=getdtype()))
            node.getright().setgrad(np.add(node.getright().getgrad(), node.getgrad(), dtype=getdtype()))

        tensor = Tensor()
        tensor.type = ndarray.__rsub__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checkTensor
    def __mul__(self, other):
        @checkDataDevice
        def _forwardfunc(node):
            if node.getdevice() in ["cpu"]:
                node.setdata(np.multiply(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))
            elif node.getdevice() in ["gpu", "cuda"]:
                node.setdata(skcuda.misc.multiply(node.getleft().getdata(), node.getright().getdata()))
            else:
                raise Exception

        @checkgradisnone
        def _gradient(node):
            if node.getdevice() in ["cpu"]:
                if node.getleft().getdata().shape == node.getright().getdata().shape:
                    node.getleft().setgrad(node.getleft().getgrad()
                                           + np.multiply(node.getright().getdata(), node.getgrad(), dtype=getdtype()))
                else:
                    node.getleft().setgrad(node.getleft().getgrad()
                                           + np.sum(np.multiply(node.getright().getdata(),
                                                                node.getgrad()), axis=0, dtype=getdtype()))
                node.getright().setgrad(node.getright().getgrad() +
                                        np.multiply(node.getleft().getdata(), node.getgrad(), dtype=getdtype()))

            elif node.getdevice() in ["gpu", "cuda"]:
                if node.getleft().getdata().shape == node.getright().getdata().shape:
                    node.getleft().setgrad(node.getleft().getgrad()
                                           + skcuda.misc.multiply(node.getright().getdata(), node.getgrad()))
                else:
                    node.getleft().setgrad(node.getleft().getgrad()
                                           + skcuda.misc.sum(skcuda.misc.multiply(node.getright().getdata(),
                                                                                  node.getgrad()), axis=0))
                node.getright().setgrad(node.getright().getgrad() +
                                        skcuda.misc.multiply(node.getleft().getdata(), node.getgrad()))

        tensor = Tensor()
        tensor.type = ndarray.__mul__
        tensor.setleft(self)
        tensor.device = self.device
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checkTensor
    def __rmul__(self, other):
        def _forwardfunc(node) -> None:
            node.setdata(np.multiply(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))

        @checkgradisnone
        def _gradient(node) -> None:
            node.getleft().setgrad(node.getleft().getgrad()
                                   + np.multiply(node.getright().getdata(), node.getgrad(), dtype=getdtype()))
            node.getright().setgrad(node.getright().getgrad()
                                    + np.multiply(node.getleft().getdata(), node.getgrad(), dtype=getdtype()))

        tensor = Tensor()
        tensor.type = ndarray.__rmul__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checkTensor
    def __truediv__(self, other):
        def _forwardfunc(node) -> None:
            node.setdata(np.divide(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))

        @checkgradisnone
        def _gradient(node: Tensor) -> None:
            node.getleft().setgrad(
                node.getleft().getgrad() +
                np.multiply(np.divide(1, node.getright().getdata()), node.getgrad(), dtype=getdtype()))
            node.getright().setgrad(
                node.getright().getgrad()
                + np.multiply(node.getleft().getdata(), -node.getright().getdata() ** -2, dtype=getdtype()))

        tensor = Tensor()
        tensor.type = ndarray.__truediv__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setforwardfunc(_forwardfunc)
        tensor.setgradfunc(_gradient)
        return tensor

    @checkTensor
    def __rtruediv__(self, other):
        def _forwardfunc(node) -> None:
            node.setdata(np.divide(node.getright().getdata(), node.getleft().getdata(), dtype=getdtype()))

        @checkgradisnone
        def _gradient(node) -> None:
            node.getleft().setgrad(
                node.getleft().getgrad()
                + np.dot(node.getleft().getdata(), -node.getright().getdata() ** -2))
            node.getright().setgrad(
                node.getright().getgrad()
                + np.dot(np.divide(1, node.getright().getdata()), node.getgrad()))

        tensor = Tensor()
        tensor.type = ndarray.__rtruediv__
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setgradfunc(_gradient)
        tensor.setforwardfunc(_forwardfunc)
        return tensor

    @checkTensor
    def __pow__(self, power, modulo=None):
        def _forward(node) -> None:
            node.setdata(np.power(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))

        @checkgradisnone
        def _gradient(node) -> None:
            epsilon = getepsilon()
            node.getleft().setgrad(
                node.getleft().getgrad() +
                np.multiply(node.getgrad(),
                            np.multiply(node.getright().
                                        getdata(),
                                        node.getleft().getdata() ** (node.getright().getdata() - 1), dtype=getdtype())))

            node.getleft().data = np.where(np.abs(node.getleft().getdata()) > epsilon, node.getleft().getdata(),
                                           np.sign(node.getleft().getdata()) * epsilon)

            node.getleft().data = np.where(node.getleft().getdata() == 0., epsilon, node.getleft().getdata())

            node.getright().setgrad(node.getright().getgrad() +
                                    np.multiply(node.getgrad(),
                                                np.multiply(node.getleft().getdata(),
                                                            np.log(node.getleft().getdata(),
                                                                   dtype=getdtype()), dtype=getdtype()),
                                                dtype=getdtype()))

        tensor = Tensor()
        tensor.type = ndarray.__pow__
        tensor.setleft(self)
        tensor.setright(power)
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checkTensor
    def __rpow__(self, other):
        def _forward(node) -> None:
            node.setdata(np.power(node.getright().getdata(), node.getleft().getdata()))

        @checkgradisnone
        def _gradient(node) -> None:
            node.getleft().setgrad(node.getleft().getgrad() +
                                   np.dot(node.getgrad().T,
                                          np.dot(node.getgrad(),
                                                 np.log(node.getleft().getdata()), dtype=getdtype()), dtype=getdtype()))

            node.getright().setgrad(node.getright().getgrad() +
                                    np.dot(node.getgrad().T, np.dot(node.getright().getdata(),
                                                                    node.getleft().getdata() ** (
                                                                            node.getright().getdata() - 1),
                                                                    dtype=getdtype()), dtype=getdtype()))

        tensor = Tensor()
        tensor.setleft(self)
        tensor.setright(other)
        tensor.type = ndarray.__rpow__
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checkTensor
    def __neg__(self):
        @checkDataDevice
        def _forward(node) -> None:
            node.setdata(-node.getleft().getdata())

        @checkgradisnone
        def _gradient(node) -> None:
            node.getleft().setgrad(node.getleft().getgrad() - node.getgrad())

        tensor = Tensor()
        tensor.type = ndarray.__neg__
        tensor.setleft(self)
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checkTensor
    def __matmul__(self, other):
        @checkDataDevice
        def _forward(node: Tensor):
            if node.getdevice() in ["cpu"]:
                node.setdata(np.dot(node.getleft().getdata(), node.getright().getdata()).astype(node.dtype))

            elif node.getdevice() in ["gpu", "cuda"]:
                node.setdata(linalg.dot(node.getleft().getdata(), node.getright().getdata()).astype(node.dtype))

        @checkgradisnone
        def _gradient(node):
            if node.getdevice() in ["cpu"]:

                node.getleft().setgrad(node.getleft().getgrad() + np.dot(node.getgrad(), node.getright().getdata().T))
                node.getright().setgrad(node.getright().getgrad() + np.dot(node.getleft().getdata().T, node.getgrad()))

            elif node.getdevice() in ["gpu", "cuda"]:

                res_left = skcuda.linalg.dot(node.getgrad(), node.getright().getdata())
                if res_left.shape == ():
                    res_left = gpuarray.ones_like(node.getleft().getgrad()).fill(res_left.get())
                node.getleft().setgrad(node.getleft().getgrad() + res_left)

                trans = skcuda.linalg.transpose(node.getleft().getdata())
                res_right = skcuda.linalg.dot(trans, node.getgrad())
                if res_right.shape == ():
                    res_right = gpuarray.ones_like(node.getright().getgrad()).fill(res_right.get())
                node.getright().setgrad(node.getright().getgrad() + res_right)

            else:
                raise Exception

        tensor = Tensor()
        tensor.type = "dot"
        tensor.setleft(self)
        tensor.setright(other)
        tensor.device = self.device
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checkTensor
    def __gt__(self, other):
        def _forward(node) -> None:
            node.setdata(np.greater(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))

        def _gradient(*args) -> None:
            pass

        tensor = Tensor()
        tensor.type = np.greater
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    @checkTensor
    def __lt__(self, other):
        def _forward(node) -> None:
            node.setdata(np.less(node.getleft().getdata(), node.getright().getdata(), dtype=getdtype()))

        def _gradient(*args) -> None:
            pass

        tensor = Tensor()
        tensor.type = np.less
        tensor.setleft(self)
        tensor.setright(other)
        tensor.setforwardfunc(_forward)
        tensor.setgradfunc(_gradient)
        return tensor

    def getdata(self):
        return self.data

    def setdata(self, data):
        if data is not None:
            if self.device in ["gpu", "cuda"]:
                if type(data) == gpuarray.GPUArray:
                    self.data = data.astype(self.dtype)
                elif type(data) == np.ndarray:
                    self.data = data.astype(self.dtype)
                    self.data = gpuarray.to_gpu(self.data)
                else:
                    self.data = np.array(data, dtype=self.dtype)
                    self.data = gpuarray.to_gpu(self.data)
            elif self.device in ["cpu"]:
                if type(data) == gpuarray.GPUArray:
                    self.data = data.get().astype(self.dtype)
                elif type(data) == np.ndarray:
                    self.data = data.astype(self.dtype)
                else:
                    self.data = np.array(data, dtype=self.dtype)
            else:
                raise Exception

    def setdevice(self, device):
        self.device = device


    def getdtype(self):
        return self.dtype

    def setdtype(self, dtype: np.dtype):
        self.dtype = dtype

    def getdevice(self):
        return self.device

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

    def setmomentum(self, momentum):
        self.momentum = momentum

    def getmomentum(self):
        return self.momentum

    def getisgrad(self):
        return self.isgrad

    def getisconst(self):
        return self.isconst

    def getargs(self):
        return self.args

    def setargs(self, args):
        try:
            _ = (arg for arg in args)
            if args == []:
                self.args = None
            else:
                self.args = args
        except:
            args = {args}
            self.args = args

    def getkwargs(self):
        return self.kwargs

    def setkwargs(self, kwargs):
        try:
            _ = (kwarg for kwarg in kwargs)
            if kwargs == []:
                self.kwargs = None
            else:
                self.kwargs = kwargs
        except:
            kwargs = {kwargs}
            self.kwargs = kwargs

    def getgradfunc(self):
        return self.gradfunc

    def setgradfunc(self, gradfunc):
        self.gradfunc = gradfunc

    def getforwardfunc(self):
        return self.forwardfunc

    def setforwardfunc(self, func):
        self.forwardfunc = func

    def shape(self):
        def _forwardfunc(node):
            node.setdata(np.shape(node.getleft().getdata()))

        def _gradient(*args):
            pass

        if self.data is not None:
            return np.shape(self.data)
        else:
            tensor = Tensor()
            tensor.setleft(self)
            tensor.type = np.shape
            tensor.isgrad = False
            tensor.setforwardfunc(_forwardfunc)
            tensor.setgradfunc(_gradient)
            return tensor

    def T(self):
        return np.transpose(self.data)

    def toDevice(self):
        assert self.data is not None, "Tensor's data is None, add data first"
        assert self.device != "cpu", "Tensor's device must not be cpu"
        self.device = "gpu"
        self.data = gpuarray.to_gpu(self.data)


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

    def backpropagation(self) -> Tensor:

        def backward(node: Tensor):
            if node.getleft() is not None and node.getgradfunc() is not None:
                gradfunc = node.getgradfunc()
                gradfunc(node)
                backward(node.getleft())
                if node.getright() is not None:
                    backward(node.getright())
                    return
            else:
                return

        if self.getroot().getgrad() is None:
            if self.getroot().getdevice() in ["cuda", "gpu"]:
                self.getroot().setgrad(gpuarray.ones_like(self.getroot().getdata(),dtype=self.getroot().dtype))
            elif self.getroot().getdevice() in ["cpu"]:
                self.getroot().setgrad(np.ones(self.getroot().getdata().shape, dtype=self.getroot().dtype))

        backward(self.getroot())
        return self.root

    def forwardpropagation(self):
        def forward(node: Tensor):
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
                pass
            return

        forward(self.getroot())
        return self.getroot()

    def through_graph(self, func=None):
        def through(node: Tensor):

            if node.getleft() is not None:
                through(node.getleft())

                if node.getright() is not None:
                    through(node.getright())

            if func is not None:
                func(node)
            else:
                pass
            return

        through(self.getroot())
        return self.getroot()


class Func:
    def __init__(self, root=None):
        self.graph = None
        self.root = None
        self.input = None
        self.weight_list = None

        self.setroot(root)

    def __call__(self, *args, **kwargs):
        self.setroot(args)
        # if type(x) is Tensor:
        #     self.getinput().setleft(x)
        # elif type(x) is Func:
        #     self.getinput().setleft(x.getroot())

        # if y is not None:
        #     if type(y) is Tensor:
        #         self.getinput().setright(y)
        #     elif type(y) is Func:
        #         self.getinput().setright(x.getroot())
        return self

    def __add__(self, other):
        tensor = self.getroot() + other.getroot()
        func = Func(tensor)
        if self.getroot().getdata() is not None and other.getroot().getdata() is not None:
            func.forward()
        return func

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
        assert self.root is not None, "No root set"
        result = self.graph.forwardpropagation()
        return result.getdata()

    def backward(self) -> Tensor:
        root = self.graph.backpropagation()
        return root

    def getgraph(self):
        assert self.root is not None, "no root been set"
        self.setroot(self.root)
        return self.graph
