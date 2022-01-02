import os
import re
import shutil

import numpy as np
from utils.node import Node


class WeightIni:
    """
    Provide weight initial functions. util class
    """

    @staticmethod
    def init_linear_weight(input_dim, output_dim):
        return np.random.uniform(-1, 1, (input_dim, output_dim))

    @staticmethod
    def init_BN_weight(dim):

        return np.ones((1, dim)), np.ones((1, dim), dtype="float32")

    @staticmethod
    def init_conv2D_kernel(shape):
        """
        :param shape: Union[tuple, int, float] shape of kernel
        :return:
        """
        return np.random.random(shape)

    @staticmethod
    def initial_weight_list(layer_list):
        """
        @Staticmethod. Given layer list and return respected initiall weight list

        :param layer_list: list, layer list
        :return: list, list of weight in Node class
        """
        weight_list = []
        # initial weights in weight list by their type
        layer_num = len(layer_list)
        for i in range(layer_num):
            # linear weight operation
            if layer_list[i].gettype() == "Linear":
                weight_list.append(Node(
                    WeightIni.init_linear_weight(layer_list[i].getinputdim(),
                                                 layer_list[i].getoutputdim()), "weight"))
            elif layer_list[i].gettype() == "BN":
                dim = layer_list[i - 1].getoutputdim()
                gamma, beta = WeightIni.init_BN_weight(dim)
                weight_list.append(Node(gamma, "weight"))
                weight_list.append(Node(beta, "weight"))
                layer_list[i].input_dim = dim
                layer_list[i].output_dim = dim
            # kernel parse operation
            elif layer_list[i].gettype() == "Conv2D":
                weight_list.append(
                    Node(WeightIni.init_conv2D_kernel(layer_list[i].getkernelsize()), "weight"))
            else:
                return NameError
            # check if you need BN init
            if layer_list[i].getBN():
                dim = layer_list[i].getoutputdim()
                gamma, beta = WeightIni.init_BN_weight(dim)
                weight_list.append(Node(gamma, "weight"))
                weight_list.append(Node(beta, "weight"))

        return weight_list


def saveweight(weight, path='runs/train'):
    if not os.path.isdir(path):
        os.makedirs(path)

    i = 0
    while True:
        filepath = os.path.join(path, "weight" + str(i))
        if not os.path.isfile(filepath):
            np.save(filepath, weight)
            break
        else:
            pass
        i += 1
    return filepath


def readweight(path):
    if os.path.isfile(path):
        return np.load(path)
    else:
        raise NameError
