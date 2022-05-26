import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray


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
            if arg.device == "gpu" or arg.device == "cuda":
                assert type(arg.grad) == pycuda.gpuarray.GPUArray or arg.grad is None, \
                    "Ottergrad error: arg's grad must be GPUArray type. @checkGRadDeviceIsNone"
                if arg.getgrad() is None:
                    arg.setgrad(gpuarray.ones_like(arg.getdata()))

                if arg.left is not None:
                    assert type(arg.left.grad) == pycuda.gpuarray.GPUArray or arg.left.grad is None, \
                        "Ottergrad error: arg's grad must be GPUArray type. @checkGRadDeviceIsNone"
                    if arg.getleft().getgrad() is None:
                        arg.getleft().setgrad(gpuarray.zeros_like(arg.getleft().getdata()))

                if arg.right is not None:
                    assert type(arg.right.grad) == pycuda.gpuarray.GPUArray or arg.right.grad is None, \
                        "shell error: arg's right node grad type must be GPUArray type. @checkDevice"
                    if arg.getright().getgrad() is None:
                        arg.getright().setgrad(gpuarray.zeros_like(arg.getright().getdata()))

            elif arg.device == "cpu":
                if arg.getgrad() is None:
                    arg.setgrad(np.ones(arg.getdata().shape))

                if arg.getleft() is not None:
                    if arg.getleft().getgrad() is None:
                        arg.getleft().setgrad(np.zeros(arg.getleft().getdata().shape))
                if arg.getright() is not None:
                    if arg.getright().getgrad() is None:
                        arg.getright().setgrad(np.zeros(arg.getright().getdata().shape))

            arg_list.append(arg)
        return operator(*arg_list)
    return check
