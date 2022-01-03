from dataloader import Dataset
import numpy as np


class NN:
    import numpy as np
    def __init__(self, layer_list=None):
        """
        This class contains Activation function util class, Layer class for construct networks, it contains several extend classes like LinearLayer, Conv2D etc.

        Examples:

        layer_list = [NN.Layer('Linear',3,10,'sigmoid',BN=True), NN.Layer('Linear',10,100,'sigmoid',BN=True),
              NN.Layer('Linear',100,10,'sigmoid',BN=True),NN.Layer('Linear',10,3,'none') ]

        dataset = Dataset(X, y, mini_batch= 64)

        nn = NN(dataset)

        layer_list is a list has 4 layers all are Layer class. Note that here we don't use LinearLayer,
        to use LinearLayer, replace NN.Layer('Linear',3,10,'sigmoid',BN=True) as NN.LinearLayer(,3,10,'sigmoid',BN=True) or
        NN.LinearLayer(3,10,'sigmoid'), NN.BN()

        :param dataset: Dataset class
        """
        # self.input = input
        self.layer_list = layer_list

    def addlayers(self, layer_list):
        self.layer_list = layer_list

    def getlayers(self):
        return self.layer_list

    # activation functions
    class ActivationFunc:
        """
        ActivationFunc is an util class with different types of activation function.
        it can
        """

        @staticmethod
        def sigmoid(x):
            """
            Sigmoid function
            """
            return 1.0 / (1.0 + np.exp(-x))

        @staticmethod
        def ReLU(x):
            """
            :param x: ndarray,
            :return:
            """
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
        def __init__(self, type, input_dim, output_dim, activation, BN=False):
            """
            Define a layer contains activation function or other normalization.

            :param type: Layer type, choose 'Linear', 'Conv' etc
            :param input_dim: input dim or previous layer's output
            :param output_dim: output dim of this layer
            :param activation: activation function, it now support "sigmoid", "ReLU", "LeakyReLU", "tanh" and "none" for no activation function
            :param BN, batch normalization , Default False

            Examples:

            A linear layer with input dim = 3 and output dim = 10, following batch normalization and a sigmoid activation function
            NN.Layer('Linear',3,10,'sigmoid',BN=True)

            """
            self.type = type
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.activation = activation
            self.BN = BN

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

        def getBN(self):
            return self.BN

    class LinearLayer(Layer):
        """
        Define a linear layer

        As same as Layer except no need to clarify type
        """

        def __init__(self, input_dim, output_dim):
            self.type = "Linear"
            self.input_dim = input_dim
            self.output_dim = output_dim

    class Conv2DLayer(Layer):
        """
        Define a 2D convolutional layer_
        """

        def __init__(self, input_size, kernel_size, stride, padding):
            """
            initialize 2D conv layer

            :param input_size: Union[tuple, ndarray]  layer's input size
            :param kernel_size:  Union[tuple, ndarray] layer's kernel size
            :param stride: Int
            :param padding: Int
            """
            self.type = "Conv2D"
            self.input_size = input_size
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

        def getinputsize(self):
            return self.input_size

        def getkernelsize(self):
            return self.kernel_size

        def getstride(self):
            return self.stride

        def getpadding(self):
            return self.padding

    class BN(Layer):
        def __init__(self):
            """
            Define a batch normalization layer
            """
            self.type = "BN"
            self.activation = "none"
