import numpy as np


class NN:
    import numpy as np
    def __init__(self, dataset):
        """
        :param dataset:
        """
        # self.input = input
        self.dataset = dataset
        self.layer_list = []

    def addlayers(self, layers):
        self.layer_list = layers

    def getlayers(self):
        return self.layer_list

    # activation functions
    class ActivationFunc:
        @staticmethod
        def sigmoid(x):
            """
            Sigmoid function
            """
            return 1.0 / (1.0 + np.exp(-x))

        @staticmethod
        def ReLU(x):
            return np.maximum(0, x)

        @staticmethod
        def LeakyReLU(x):
            return np.where(x > 0, x, x * 0.01)

        @staticmethod
        def tanh(x):
            return np.tanh(x)

        @staticmethod
        def none(x):
            return x

    # Layer class
    class Layer:
        def __init__(self, type, input_dim, output_dim, activation):
            """
            :param type: Layer type, choose 'Linear', 'Conv' etc
            :param input_dim: input dim or previous layer's output
            :param output_dim: output dim of this layer
            :param activation: activation function, 'none' for no activation
            """
            self.type = type
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.activation = activation

        def getinputdim(self):
            return self.input_dim

        def getoutputdim(self):
            return self.output_dim

        def gettype(self):
            return self.type

        def getact(self, x):
            func_name = "NN.ActivationFunc." + self.activation
            func = eval(func_name)
            return func(x)

        def getactname(self):
            return self.activation

    class LinearLayer(Layer):
        def __init__(self, input_dim, output_dim):
            self.type = "Linear"
            self.input_dim = input_dim
            self.output_dim = output_dim

    class Conv2DLayer(Layer):
        def __init__(self, image_size, kernel_size, stride, padding):
            self.type = "Conv2D"
            self.image_size = image_size
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

        def getimagesize(self):
            return self.image_size

        def getkernelsize(self):
            return self.kernel_size

        def getstride(self):
            return self.stride

        def getpadding(self):
            return self.padding

    class BN(Layer):
        def __init__(self):
            self.type = "BN"
            self.activation = "none"
