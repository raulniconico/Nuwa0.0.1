import os

import numpy as np
import copy
import time
from network import NN
import warnings

warnings.filterwarnings("ignore")


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


class LossFunc:
    class Loss:
        def __init__(self, y_true, y_pred):
            self.y_true = y_true
            self.y_pred = y_pred
            self.name = ""

        def data(self, y_true, y_pred):
            self.y_true = y_true
            self.y_pred = y_pred

        def getname(self):
            return self.name

    class Logarithmic(Loss):
        def __init__(self, y_true, y_pred, eps=1e-16):
            """
            Loss function we would like to optimize (minimize)
            We are using Logarithmic Loss
            http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
            """
            super().__init__(y_true, y_pred)
            self.eps = eps
            self.name = "Logarithmic"

        def loss(self):
            self.y_pred = np.maximum(self.y_pred, self.eps)
            self.y_pred = np.minimum(self.y_pred, (1 - self.eps))
            return -(np.sum(self.y_true * np.log(self.y_pred)) + np.sum(
                (1 - self.y_true) * np.log(1 - self.y_pred))) / len(self.y_true)

    class Quadratic(Loss):
        def __init__(self, y_true=None, y_pred=None, norm=0):
            self.y_true = y_true
            self.y_pred = y_pred
            self.norm = norm
            self.name = "Quadratic"

        def loss(self):
            return 1 / self.y_true.shape[0] * 0.5 * np.sum((self.y_pred - self.y_true) ** 2)

        def diff(self):
            return 2 * (self.y_pred - self.y_true)

    class MSE(Loss):
        def __init__(self, y_true=None, y_pred=None, x=None):
            self.y_true = y_true
            self.y_pred = y_pred
            self.x = x
            self.name = "MSE"

        def data(self, y_true, y_pred, x):
            self.y_true = y_true
            self.y_pred = y_pred
            self.x = x

        def loss(self):
            return 1 / np.shape(self.y_true)[0] * np.sum((self.y_pred - self.y_true) ** 2)

        def diff(self):
            return 2 / np.shape(self.y_true)[0] * np.sum(self.x @ (self.y_pred - self.y_true))

    class MSEContinue(Loss):
        def __init__(self, initial, boundary, inner):
            """

            :param initial:
            :param boundary:
            :param inner: array like, could be latent function h or its derivative h_x or its secound order h_xx
            """
            self.initial = initial
            self.boundary = boundary
            self.inner = inner
            self.name = "MSEContinue"

            self.loss_initial = LossFunc.MSE(self.boundary[0], self.boundary[1], self.boundary[2])
            self.loss_bound = []
            self.loss_in = LossFunc.MSE(self.inner[0], self.inner[1], self.inner[2])

            for func in self.boundary:
                self.loss_bound.append(LossFunc.MSE(func[0], func[1], func[2]))

        def loss_init(self):
            return self.loss_initial.loss()

        def loss_inner(self):
            return self.loss_in.loss()

        def loss(self):
            return self.loss_boundary() + self.loss_inner() + self.loss_f()

        def diff_boundary(self):
            return self.loss_bound.diff()

        def diff_in(self):
            return self.loss_in.diff()

        def diff_f(self):
            return self.loss_func.diff()

        def diff(self):
            return self.diff_boundary() + self.diff_in() + self.diff_f()


def GD(root: Node, weight_list, lr=0.001):
    """
    Gradient descent, do the back propagation and update weight list

    :param lr:
    :param root: Node, the root of passer binary tree
    :param weight_list: list, weight list
    :return: list, updated weight list
    """
    Optimizer.backpropagation(root)
    gradient_list = []

    for node in weight_list:
        node.data = node.data - lr * node.back
        gradient_list.append(node.back)
    return weight_list, gradient_list


def SGD(dataset, weight_list, passer_list, lr=0.001):
    # we resume mini-batch equals 1 each time
    """
    Stochastic gradient descent. It takes weight list and passer list as inputs, it will
    :param weight_list:
    :param passer_list:
    :return:
    """

    def init_random_node(node, random_num_list, mini_weight_list):
        node.data = node.data[random_num_list, :]
        node.back = None
        if node.getright() is not None:
            mini_weight_list.append(node.getright())
        if node.getleft() is not None:
            init_random_node(node.getleft(), random_num_list, mini_weight_list)
        else:
            return

    # obs = observation number = output layer's dim 0
    num_obs = dataset.gettrainset().getX().shape[0]
    mini_passer_list = copy.deepcopy(passer_list)
    root = mini_passer_list[-1]
    gradient_list = []

    # randomly pick observations from original obs
    random_num_list = np.random.randint(0, num_obs, num_obs)

    # initial random node
    mini_weight_list = []
    init_random_node(root, random_num_list, mini_weight_list)

    # back propagation
    root.back = 2 * (- dataset.gettrainset().gety()[random_num_list] + root.getdata()[random_num_list])
    Optimizer.backpropagation(root)

    i = 0
    # update weight list
    for weight in weight_list:
        weight.data = weight.data - lr * mini_weight_list[-i - 1].back
        gradient_list.append(mini_weight_list[-i - 1].back)
        i = i + 1

    return weight_list, gradient_list


def momentumgd(root: Node, weight_list, lr=0.001, beta=0.2):
    """

    :param root: Node, the root of passer binary tree
    :param weight_list: list, weight list
    :param beta: momentum conservation rate
    :return: list, updated weight list
    """
    Optimizer.backpropagation(root)
    gradient_list = []

    for node in weight_list:
        if node.getmomentum() is None:
            node.momentum = (1 - beta) * node.getback()
        else:
            node.momentum = beta * node.getmomentum() + (1 - beta) * node.getback()
        node.data = node.getdata() - lr * (1 - beta) * node.getback()
        gradient_list.append(node.back)
    return weight_list, gradient_list


def RMSprop(root: Node, weight_list, lr=0.001, beta=0.2, eps=1e-10):
    Optimizer.backpropagation(root)
    gradient_list = []

    for node in weight_list:
        if node.getmomentum() is None:
            node.momentum = (1 - beta) * node.getback() ** 2
        else:
            node.momentum = beta * node.getmomentum() + (1 - beta) * node.getback() ** 2

        node.data = node.getdata() - lr * node.getback() / (np.sqrt(node.getmomentum()) + eps)
        gradient_list.append(node.back)
    return weight_list, gradient_list


def Adam(root: Node, weight_list, lr=0.001, beta_mom=0.2, beta_rms=0.2, eps=1e-10):
    """
    Adam optimizer
    :param root:
    :param weight_list:
    :param beta_mom:
    :param beta_rms:
    :param eps:
    :return:
    """
    Optimizer.backpropagation(root)
    gradient_list = []

    for node in weight_list:
        if node.getmomentum() is None:
            node.momentum = [(1 - beta_mom) * node.getback(), (1 - beta_rms) * node.getback() ** 2]
        else:
            node.momentum[0] = (beta_mom * node.getmomentum()[0] + (1 - beta_mom) * node.getback()) / (1 - beta_mom)
            node.momentum[1] = (beta_rms * node.getmomentum()[1] + (1 - beta_rms) * node.getback() ** 2) / (
                    1 - beta_rms)

        node.data = node.getdata() - lr * node.getmomentum()[0] / (np.sqrt(node.getmomentum()[1]) + eps)
        gradient_list.append(node.back)
    return weight_list, gradient_list


class Optimizer:
    def __init__(self, dataset=None, nn=None, optimizer=None, loss_function=None, batch_size=32, epoch=20000, lr=0.0001,
                 decay_rate=0, pretrained=None):
        """
        :param nn: input an NN class
        :param optimizer: optimizer as "GD", "SGD" etc
        :param batch_size: batch size for mini batch optimization
        :param epoch: epoch number
        :param lr: learning rate
        :param decay_rate: float, learning rate decay rate by default is 0
        """

        self.dataset = dataset
        self.nn = nn
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.weight_list = None
        self.gradient_list = None
        self.loss_list = None
        self.passer_list = None
        self.decay_rate = decay_rate
        self.pretrained = pretrained
        self.savepath = None

    def adddataset(self, dataset):
        self.dataset = dataset

    def addnn(self, nn):
        self.nn = nn

    def addoptimizer(self, optimizer):
        self.optimizer = optimizer

    def loss_function(self, loss_function):
        self.loss_function = loss_function

    def getgradientlist(self):
        return self.gradient_list

    def getlosslist(self):
        return self.loss_list

    def getweightlist(self):
        return self.weight_list

    def getpretrained(self):
        return self.pretrained

    def saveweight(self, path='runs/train/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        i = 0
        while True:
            filepath = os.path.join(path, "exp" + str(i))
            if not os.path.isdir(filepath):
                np.save(filepath, self.weight_list[-1])
                break
            else:
                pass
            i += 1

        return os.path.join(os.path.join(filepath, "weight.npy"))

    def lrdecay(self, iter):
        """
        Learning rate decay function. Given iteration, modify learning rate

        :param iter: int, iteration count
        """
        self.lr = 1 / (1 + self.decay_rate * iter) * self.lr

    @staticmethod
    def forword(passer, weight_list, layer_list):
        layer_num = len(layer_list)
        passer_list = [Node(passer, "data")]
        # Every layer not necessarily has only one weight, like BN has 2 weights in a single layer
        weight_count = 0

        for i in range(layer_num):
            if layer_list[i].gettype() == 'Linear':
                passer = passer @ weight_list[weight_count].getdata()
                # append binary tree after inner product of weight and previous layer
                node = Node(passer, "@")
                node.left = passer_list[-1]
                node.right = weight_list[weight_count]
                passer_list.append(node)

                weight_count += 1

                if layer_list[i].getBN():
                    node_cache = [passer, np.var(passer, axis=0), np.mean(passer, axis=0)]

                    passer = (passer - np.mean(passer, axis=0)) / np.sqrt(np.var(passer, axis=0))
                    node = Node(passer, "normalization")
                    node.cache = node_cache
                    node.left = passer_list[-1]
                    passer_list.append(node)

                    node = Node(passer, "*scalar")
                    node.left = passer_list[-1]
                    node.right = weight_list[weight_count]
                    passer_list.append(node)

                    passer = passer + weight_list[weight_count + 1].getdata()
                    node = Node(passer, "+scalar")
                    node.left = passer_list[-1]
                    node.right = weight_list[weight_count + 1]
                    passer_list.append(node)

                    weight_count += 2

                passer = layer_list[i].getact(passer)
                # append binary tree after activation function
                node = Node(passer, layer_list[i].getactname())
                node.left = passer_list[-1]
                passer_list.append(node)

            # elif layer_list[j].gettype() == "Conv2D":
            else:
                raise NameError

        return passer_list

    @staticmethod
    def backpropagation(node):
        epsilon = 1e-8
        if node.getleft() is not None:
            if node.gettype() == "@":
                node.getleft().back = node.getback() @ node.getright().getdata().T
                node.getright().back = node.getleft().getdata().T @ node.getback()
            elif node.gettype() == "sigmoid":
                node.getleft().back = np.multiply(node.getback(), np.multiply(NN.ActivationFunc.sigmoid(node.getback()),
                                                                              1 - NN.ActivationFunc.sigmoid(
                                                                                  node.getback())))
            elif node.gettype() == "ReLU":
                back = copy.deepcopy(node.getback())
                back[back <= 0] = 0
                node.getleft().back = back
            elif node.gettype() == "LeakyReLU":
                back = copy.deepcopy(node.getback())
                back[back < 0] = 0.01 * back[back < 0]
                node.getleft().back = back
            elif node.gettype() == "tanh":
                node.getleft().back = np.multiply(
                    (np.ones(node.getback().shape) - NN.ActivationFunc.tanh(node.getback()) ** 2),
                    node.getback())
            elif node.gettype() == "+":
                node.getleft().back = node.getback()
                node.getright().back = node.getback()
            elif node.gettype() == "-":
                node.getleft().back = node.getback()
                node.getright().back = -node.getback()
            elif node.gettype() == "+scalar":
                node.getleft().back = node.getback()
                node.getright().back = np.sum(node.getback(), axis=0)
            elif node.gettype() == "*scalar":
                node.getleft().back = node.getright().getdata() * node.getback()
                node.getright().back = np.sum(node.getleft().getdata().T, axis=0) @ node.getback()
            elif node.gettype() == "none":
                node.getleft().back = node.getback()
            elif node.gettype() == "normalization":
                # cache = [x,  sigma_beta^2, mu_beta]

                # dx = 1/N / std * (N * dx_norm -
                #       dx_norm.sum(axis=0) -
                #       x_norm * (dx_norm * x_norm).sum(axis=0))

                x = node.cache[0]
                sigma2 = node.cache[1]
                mu = node.cache[2]

                dl_dx_hat = node.getback()
                dl_dsigma2 = np.sum(dl_dx_hat, axis=0) * (x - mu) * -0.5 * (sigma2 + epsilon) ** -3 / 2
                dl_dmu = np.sum(dl_dx_hat, axis=0) * -1 / np.sqrt(sigma2 + epsilon) + dl_dsigma2 * np.sum(-2 * (x - mu),
                                                                                                          axis=0) / \
                         x.shape[0]
                dl_dx = dl_dx_hat * 1 / np.sqrt(sigma2 + epsilon) + dl_dsigma2 * 2 * (x - mu) / x.shape[0] + dl_dmu / \
                        x.shape[0]
                node.getleft().back = dl_dx

            Optimizer.backpropagation(node.getleft())
        else:
            return

    def train(self):
        """
        train process, it will first initial weight, loss, gradient and passer list, then, optimize weights by given optimizer.
        In the end, calculate loss and step to the next epoch.

        It will finally stock all the weight, loss, gradient and passer during the training process
        """
        layer_list = self.nn.getlayers()

        # initial weight, loss and gradient list
        self.weight_list = [[] for i in range(self.epoch + 1)]
        if self.pretrained is None:
            self.weight_list[0] = WeightIni.initial_weight_list(layer_list)
        else:
            self.weight_list[0] = self.pretrained
        self.loss_list = np.zeros(self.epoch)
        self.gradient_list = [[] for i in range(self.epoch)]
        self.passer_list = [[] for i in range(self.epoch)]

        # mini batch type gradient descent
        for i in range(self.epoch):
            start_time = time.time()

            # get mini batch
            minisets = self.dataset.gettrainset().getminiset()
            epoch_weight_list = [copy.deepcopy(self.weight_list[i])]
            epoch_loss_list = np.zeros(len(minisets))

            # GD for every mini batch
            for j in range(len(minisets)):
                X_bar = minisets[j]
                self.passer_list[i].append(Optimizer.forword(X_bar.getX(), epoch_weight_list[j], layer_list))

                root = self.passer_list[i][j][-1]
                loss_func = self.loss_function(X_bar.gety(), root.getdata())

                epoch_loss_list[j] = loss_func.loss()
                root.back = loss_func.diff()
                root.momentum = root.getback()

                if self.optimizer == "minibatchgd":
                    weight, gradient = GD(root, epoch_weight_list[j], lr=self.lr)
                elif self.optimizer == "momentumgd":
                    weight, gradient = momentumgd(root, epoch_weight_list[j], lr=self.lr)
                elif self.optimizer == "RMSprop":
                    weight, gradient = RMSprop(root, epoch_weight_list[j], lr=self.lr)
                elif self.optimizer == "Adam":
                    weight, gradient = Adam(root, epoch_weight_list[j], lr=self.lr)
                else:
                    raise NameError
                epoch_weight_list.append(weight)

            self.weight_list[i + 1] = epoch_weight_list[-1]
            self.gradient_list[i] = gradient

            self.loss_list[i] = sum(epoch_loss_list) / len(epoch_loss_list)

            # learning rate decay

            self.lrdecay(i)
            # every epoch shuffle the dataset
            self.dataset.distribute()

            if (i + 1) % 10 == 0:
                used_time = time.time() - start_time
                print("epoch " + str(i + 1) + ', Training time: %.4f' % used_time + ', Training loss: %.6f' %
                      self.loss_list[i])

    def test(self):
        """
        Use trained weight on testset for the evaluation of the model
        :return: model prediction and loss on the testset
        """
        weight = self.weight_list[-1]
        layer_list = self.nn.getlayers()
        testset = self.dataset.gettestset()
        passer = testset.getX()

        passer_list = self.forword(passer, weight, layer_list)
        predicted = passer_list[-1].getdata()

        loss = self.loss_function.loss(testset.gety(), predicted)
        return predicted, loss

    def predict(self, X):
        """
        Use trained weight on X and output prediction
        :param X: ndarray, feature data wish to be predicted
        :return: model's prediction by using trained data
        """
        passer = X
        weight = self.weight_list[-1]
        passer_list = self.forword(passer, weight, self.nn.getlayers())
        return passer_list


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


def saveweight(weight, path='runs/train/'):
    if not os.path.isdir(path):
        os.makedirs(path)

    i = 0
    while True:
        filepath = os.path.join(path, "exp" + str(i))
        if not os.path.isdir(filepath):
            np.save(filepath, weight)
            break
        else:
            pass
        i += 1

    return os.path.join(os.path.join(filepath, "weight.npy"))


def readweight(path):
    if os.path.isfile(path):
        return np.load(path)
    else:
        raise NameError
