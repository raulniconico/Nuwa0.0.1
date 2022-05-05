import numpy as np


dtype = np.float32
epsilon = 1e-8
device = 'cuda'


def getdtype():
    return dtype


def getepsilon():
    return epsilon


def getdevice():
    return device


def checkndarray(operator):
    def check(*args):
        arg_list = []
        for arg in args:
            if type(arg) is not np.ndarray:
                arg_list.append(np.ndarray(arg))
            else:
                arg_list.append(arg)
        return operator(*arg_list)
    return check


def checkgradisnone(operator):
    def check(*args):
        arg_list = []
        for arg in args:
            if arg.getleft() is not None:
                if arg.getleft().getgrad() is None:
                    arg.getleft().setgrad(np.zeros_like(arg.getleft().getdata()))
            if arg.getright() is not None:
                if arg.getright().getgrad() is None:
                    arg.getright().setgrad(np.zeros_like(arg.getright().getdata()))

            arg_list.append(arg)
        return operator(*arg_list)

    return check


