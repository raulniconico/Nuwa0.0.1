import numpy as np
from Ottergrad.autograd import Tensor, Func
from Ottergrad.nn import norm
from Nuwa.utils import getdtype
from Nuwa.activation import *


class NN:
    def __init__(self, layer_list: [list, np.ndarray, iter] = None, BN: bool = False, pretrained=None):

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
        """
        # self.input = input
        self.layer_list = layer_list
        self.BN = BN
        self.input = None
        self.pretrained = pretrained
        self.func = None
        self.graphe = None
        self.weights = None

    def __call__(self, input: Tensor) -> Tensor:
        self.input = input
        self.forword()
        return self.func.getroot()

    def setlayers(self, *layer_list):
        for layer in layer_list:
            self.layer_list.append(layer)

    def getlayers(self):
        return self.layer_list

    def addlayer(self, *layer_list):
        pass

    def setpretrained(self, pretrained):
        self.pretrained = pretrained

    def getpretrained(self):
        return self.pretrained

    def getfunc(self):
        return self.func

    def forword(self):
        self.func = self.input

        for layer in self.layer_list:
            self.func = layer(self.func)

        self.func = Func(self.func)
        self.func.forward()

        return self.func.getroot().getdata()

    def getweights(self):
        nn_weight_list = {}
        for layer in self.layer_list:
            nn_weight_list.update(layer.getweights())
        return nn_weight_list.values()


# Layer class
class Layer:
    ACTIVATION = {
        "sigmoid": Sigmoid,
        "ReLU": ReLU,
        "LeakyReLU": LeakyReLU,
        "tanh": tanh,
        "none": none
    }

    def __init__(self, name: str = None, input_dim=None, output_dim=None,
                 activation: str = "none", BN: bool = False, pretrained=None):
        """
        Define a layer contains activation function or other normalization.

        :param name: Layer type, choose 'Linear', 'Conv' etc
        :param input_dim: input dim or previous layer's output
        :param output_dim: output dim of this layer
        :param activation: activation function, it now supports "sigmoid", "ReLU", "LeakyReLU", "tanh" and "none" for
         no activation function
        :param BN, batch normalization , Default False

        Examples:

        A linear layer with input dim = 3 and output dim = 10, following batch normalization and a sigmoid
        activation function

        NN.Layer('Linear',3,10,'sigmoid',BN=True)

        """
        assert activation in Layer.ACTIVATION, "invalid activation function, use Layer.ACTIVATION to check " \
                                               "all available activations"
        self.input = None
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.BN = BN
        self.weights = None
        self.pretrained = pretrained
        self.passer_list = None

    def getinputdim(self):
        return self.input_dim

    def getoutputdim(self):
        return self.output_dim

    def getname(self):
        return self.name

    def getact(self, x):
        func_name = "NN.ActivationFunc." + self.activation
        func = eval(func_name)
        return func(x)

    def getactname(self):
        return self.activation

    def getBN(self):
        return self.BN

    def getweights(self):
        return self.weights


class Linear(Layer):
    """
    Define a linear layer

    As same as Layer except no need to clarify type
    """

    def __init__(self, input_dim=None, output_dim=None, activation="none", BN: bool = False, pretrained=None):
        super().__init__("Linear", input_dim, output_dim, activation, BN, pretrained)

    def __call__(self, x: [Tensor, Func]):
        # init weight
        if self.weights is None:
            if self.pretrained is None:
                self.init_weight()
            else:
                self.weights = self.pretrained
        self.passer_list = []

        # linear weight
        self.passer_list.append(x)
        W = self.weights["W"]
        self.passer_list.append(x @ W)

        # activation function
        self.passer_list.append(Layer.ACTIVATION[self.activation](self.passer_list[-1]))

        # BN layer
        if self.BN is True:
            gamma = self.weights["gamma"]
            beta = self.weights["beta"]
            normal = norm(self.passer_list[-1])
            normal.type = "normal"
            self.passer_list.append(gamma * normal + beta)

        return self.passer_list[-1]

    def init_weight(self):
        weight = Tensor(getdtype()(np.random.uniform(-1, 1, (self.input_dim, self.output_dim))))
        weight.type = "linear weight"
        self.weights = {"W": weight}
        if self.BN is True:
            gamma = Tensor(np.ones((self.output_dim,)), getdtype())
            gamma.type = "gamma"
            beta = Tensor(np.ones((self.output_dim,), getdtype()))
            beta.type = "beta"
            self.weights["gamma"] = gamma
            self.weights["beta"] = beta
        return self.weights


class Conv2D(Layer):
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
        super().__init__()
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
