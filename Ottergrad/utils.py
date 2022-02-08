import numpy
import numpy as np


dtype = numpy.float16
epsilon = 1e-8


def getdtype():
    return dtype


def getepsilon():
    return epsilon


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
                    arg.getleft().setgrad(0)
            if arg.getright() is not None:
                if arg.getright().getgrad() is None:
                    arg.getright().setgrad(0)

            arg_list.append(arg)
        return operator(*arg_list)

    return check
