import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel

# add
mod = SourceModule("""
    __global__ void add_ew(float *a, float *x)
    {
    int idx = threadIdx.x;
    
    x[idx] = x[idx] + a[idx];
    }
    
    __global__ void sum0(float *x, float *b)
    {
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    
    b[idx] = x[idx];
}
  """)


def add(a: [pycuda.gpuarray.GPUArray], b: [pycuda.gpuarray.GPUArray]):
    assert type(a) == type(b) == pycuda.gpuarray.GPUArray, "shell error: argument must be GPUArray type"
    result = (a + b).get()
    return result


def radd(a, b):
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    result = (a_gpu + b_gpu).get()
    return result


elementwise_multiple = ElementwiseKernel(
        "float a, float *x, float *z",
        "z[i] = a*x[i]",
        "elementwise_multiple")
