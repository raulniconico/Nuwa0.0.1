import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit


############### cuda ###############
from pycuda import gpuarray


def getcudadevice():
    """
    initial shell device, if found return device, else return None
    :return:
    """
    devices = []
    for i in range(cuda.Device.count()):
        gpu_device = cuda.Device(i)
        devices.append(gpu_device)
        # print("gpu_device name: ", gpu_device.name())
        compute_capability = float('%d.%d' % gpu_device.compute_capability())
        # print("compute_capability: ", compute_capability)
        # print("total memory: ", gpu_device.total_memory() // (1024 ** 2))
    return devices


def getcudadeviceattributes(gpu_device):
    device_attributes_tuples = gpu_device.get_attributes().items()
    # for item, count in device_attributes_tuples:
    #     print(item, count)
    return device_attributes_tuples


def hasdevice():
    device = pycuda.autoinit.device
    device_count = device.count()
    if device_count >= 1:
        return True
    else:
        return False


def checkDataDevice(operator):
    def check(*args):
        arg_list = []
        for arg in args:
            if arg.device == "gpu" or arg.device == "cuda":
                # assert arg.data is not None, "shell error: Tensor's data must not be None. @checkDevice"
                # assert type(arg.data) == pycuda.gpuarray.GPUArray, "shell error: data must be GPUArray type. " \
                #                                                    "@checkDevice"
                if arg.left is not None:
                    assert type(arg.left.data) == pycuda.gpuarray.GPUArray, \
                        "shell error: arg's left's data type must be GPUArray type. @checkDevice"
                if arg.right is not None:
                    assert type(arg.right.data) == pycuda.gpuarray.GPUArray, \
                        "shell error: arg's right's data type must be GPUArray type. @checkDevice"
            elif arg.device == "cpu":
                pass
            else:
                raise Exception
            arg_list.append(arg)
        return operator(*arg_list)
    return check


def checkGradDevice(operator):
    def check(*args):
        arg_list = []
        for arg in args:
            if arg.device == "gpu" or arg.device == "cuda":
                assert type(arg.grad) == pycuda.gpuarray.GPUArray or arg.grad is None, \
                    "shell error: arg's grad must be GPUArray type. @checkDevice"
            if arg.left is not None:
                assert type(arg.left.grad) == pycuda.gpuarray.GPUArray or arg.left.grad is None, \
                    "shell error: arg's left's data type must be GPUArray type. @checkDevice"
            if arg.right is not None:
                assert type(arg.right.grad) == pycuda.gpuarray.GPUArray or arg.right.grad is None, \
                    "shell error: arg's right node grad type must be GPUArray type. @checkDevice"
            arg_list.append(arg)
        return operator(*arg_list)
    return check


def _hosttodevice(data_host):
    """
    transform host data to device
    :param data_host:
    :return:
    """
    assert hasdevice(), "host To Device error, must have device number >= 1"
    data_device = cuda.mem_alloc(data_host.nbytes)
    cuda.memcpy_htod(data_device, data_host)
    return data_device


def _devicetohost(data_device):
    data_host = np.empty_like(1, dtype=np.float32)
    cuda.memcpy_dtoh(data_host, data_device)
    return data_host


