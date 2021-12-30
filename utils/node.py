import numpy as np


class Node:
    def __init__(self, data: np.ndarray, type: str):
        """
        Node class, is the node of binary tree which has two child node: left and right.
        It can also be presented as weight. Every passer during the back propagation is saved as
        a node class contains data, type, back and cache for calculation

        :param data: ndarray, value given during forward propagation
        :param type: str, the type of node, it can be "weight", "data" or calculation like "@", "+" etc
        :param back: ndarray, value updated during back propagation
        :param cache: array_like stock forward propagation's detail and middle value for the convenient of back propagation
        """
        self.left = None
        self.right = None
        self.data = data
        self.type = type
        self.back = None
        self.cache = None
        self.momentum = None

    def getleft(self):
        return self.left

    def getright(self):
        return self.right

    def gettype(self):
        return self.type

    def getdata(self):
        return self.data

    def getback(self):
        return self.back

    def getmomentum(self):
        return self.momentum
