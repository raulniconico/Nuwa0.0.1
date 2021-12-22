import numpy as np
import copy
import time
import network
from network import NN
import warnings
warnings.filterwarnings("ignore")


class Optimizer:
    def __init__(self, nn, optimizer, loss_function="quadra_loss", batch_size=8, epoch=20000, lr=0.0001, decay_rate=0):
        """
        :param nn: input an NN class
        :param optimizer: optimizer as "GD", "SGD" etc
        :param batch_size: batch size for mini batch optimization
        :param epoch: epoch number
        :param lr: learning rate
        :param decay_rate: float, learning rate decay rate by default is 0
        """

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

    class LossFunc:
        @staticmethod
        def log_loss(y_true, y_pred, eps=1e-16):
            """
            Loss function we would like to optimize (minimize)
            We are using Logarithmic Loss
            http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
            """
            y_pred = np.maximum(y_pred, eps)
            y_pred = np.minimum(y_pred, (1 - eps))
            return -(np.sum(y_true * np.log(y_pred)) + np.sum((1 - y_true) * np.log(1 - y_pred))) / len(y_true)

        @staticmethod
        def quadra_loss(y_true, y_pred):
            return 1 / y_true.shape[0] * 0.5 * np.sum((y_pred - y_true) ** 2)

    class Node:
        def __init__(self, data: np.ndarray, type: str):
            """
            Node class, is the node of binary tree which has two child node: left and right.
            It can also be presented as weight. Every passer during the back propagation is saved as
            a node class contains data, type, back and cache for calculation

            :param data: ndarray, value given during forward propagation :param type: str, the type of node,
            it can be "weight", "data" or calculation like "@", "+" etc :param back: ndarray, value updated during
            back propagation :param cache: array_like stock forward propagation's detail and middle value for the
            convenient of back propagation
            """
            self.left = None
            self.right = None
            self.data = data
            self.type = type
            self.back = None
            self.cache = None

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

    def getgradientlist(self):
        return self.gradient_list

    def getlosslist(self):
        return self.loss_list

    def getweightlist(self):
        return self.weight_list

    class WeightIni:
        """
        Provide weight initial functions. util class
        """

        @staticmethod
        def init_linear_weight(input_dim, output_dim):
            return np.random.uniform(-1, 1, (input_dim, output_dim))

        @staticmethod
        def init_BN_weight(dim):

            return np.ones((1, dim)), np.ones((1, dim))

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
                    weight_list.append(Optimizer.Node(
                        Optimizer.WeightIni.init_linear_weight(layer_list[i].getinputdim(),
                                                               layer_list[i].getoutputdim()), "weight"))
                elif layer_list[i].gettype() == "BN":
                    dim = layer_list[i - 1].getoutputdim()
                    gamma, beta = Optimizer.WeightIni.init_BN_weight(dim)
                    weight_list.append(Optimizer.Node(gamma, "weight"))
                    weight_list.append(Optimizer.Node(beta, "weight"))
                    layer_list[i].input_dim = dim
                    layer_list[i].output_dim = dim
                # kernel parse operation
                elif layer_list[i].gettype() == "Conv2D":
                    weight_list.append(
                        Optimizer.Node(Optimizer.WeightIni.init_conv2D_kernel(layer_list[i].getkernelsize()), "weight"))
                else:
                    return NameError
                # check if you need BN init
                if layer_list[i].getBN():
                    dim = layer_list[i].getoutputdim()
                    gamma, beta = Optimizer.WeightIni.init_BN_weight(dim)
                    weight_list.append(Optimizer.Node(gamma, "weight"))
                    weight_list.append(Optimizer.Node(beta, "weight"))

            return weight_list

    @staticmethod
    def forword(passer, weight_list, layer_list):
        layer_num = len(layer_list)
        passer_list = [Optimizer.Node(passer, "data")]
        # Every layer not necessarily has only one weight, like BN has 2 weights in a single layer
        weight_count = 0

        for i in range(layer_num):
            if layer_list[i].gettype() == 'Linear':
                passer = passer @ weight_list[weight_count].getdata()
                # append binary tree after inner product of weight and previous layer
                node = Optimizer.Node(passer, "@")
                node.left = passer_list[-1]
                node.right = weight_list[weight_count]
                passer_list.append(node)

                weight_count += 1

                if layer_list[i].getBN():
                    node_cache = [passer, np.var(passer, axis=0), np.mean(passer, axis=0)]

                    passer = (passer - np.mean(passer, axis=0)) / np.sqrt(np.var(passer, axis=0))
                    node = Optimizer.Node(passer, "normalization")
                    node.cache = node_cache
                    node.left = passer_list[-1]
                    passer_list.append(node)

                    node = Optimizer.Node(passer, "*scalar")
                    node.left = passer_list[-1]
                    node.right = weight_list[weight_count]
                    passer_list.append(node)

                    passer = passer + weight_list[weight_count + 1].getdata()
                    node = Optimizer.Node(passer, "+scalar")
                    node.left = passer_list[-1]
                    node.right = weight_list[weight_count + 1]
                    passer_list.append(node)

                    weight_count += 2

                passer = layer_list[i].getact(passer)
                # append binary tree after activation function
                node = Optimizer.Node(passer, layer_list[i].getactname())
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

    def lrdecay(self, iter):
        """
        Learning rate decay function. Given iteration, modify learning rate

        :param iter: int, iteration count
        """
        self.lr = 1 / (1 + self.decay_rate * iter) * self.lr

    def GD(self, root: Node, weight_list):
        """
        Gradient descent, do the back propagation and update weight list

        :param root: Node, the root of passer binary tree
        :param weight_list: list, weight list
        :return: list, updated weight list
        """
        Optimizer.backpropagation(root)
        gradient_list = []

        for node in weight_list:
            node.data = node.data - self.lr * node.back
            gradient_list.append(node.back)
        return weight_list, gradient_list

    def SGD(self, weight_list, passer_list):
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
        num_obs = self.nn.dataset.gettrainset().getX().shape[0]
        mini_passer_list = copy.deepcopy(passer_list)
        root = mini_passer_list[-1]
        gradient_list = []

        # randomly pick observations from original obs
        random_num_list = np.random.randint(0, num_obs, num_obs)

        # initial random node
        mini_weight_list = []
        init_random_node(root, random_num_list, mini_weight_list)

        # back propagation
        root.back = 2 * (- self.nn.dataset.gettrainset().gety()[random_num_list] + root.getdata()[random_num_list])
        Optimizer.backpropagation(root)

        i = 0
        # update weight list
        for weight in weight_list:
            weight.data = weight.data - self.lr * mini_weight_list[-i - 1].back
            gradient_list.append(mini_weight_list[-i - 1].back)
            i = i + 1

        return weight_list, gradient_list

    def train(self):
        """
        train process, it will first initial weight, loss, gradient and passer list, then, optimize weights by given optimizer.
        In the end, calculate loss and step to the next epoch.

        It will finally stock all the weight, loss, gradient and passer during the training process
        """
        start_time = time.time()
        layer_list = self.nn.getlayers()

        # initial weight, loss and gradient list
        self.weight_list = [[] for i in range(self.epoch + 1)]
        self.weight_list[0] = Optimizer.WeightIni.initial_weight_list(layer_list)
        self.loss_list = np.zeros(self.epoch)
        self.gradient_list = [[] for i in range(self.epoch)]
        self.passer_list = [[] for i in range(self.epoch)]

        # for GD and SGD, they use full dataset, so need only read X and y once
        if self.optimizer == "GD" or self.optimizer == "SGD":
            X = self.nn.dataset.gettrainset().getX()
            X = Optimizer.Node(X, "data")
            for i in range(self.epoch):
                # forward propagation
                self.passer_list[i] = Optimizer.forword(X.getdata(), self.weight_list[i], layer_list)
                root = self.passer_list[i][-1]

                # calculate loss by using: loss2 * (-self.nn.dataset.gettrainset().gety() + root.getdata())
                self.loss_list[i] = Optimizer.LossFunc.quadra_loss(self.nn.dataset.gettrainset().gety(), root.getdata())
                root.back = 2 * (-self.nn.dataset.gettrainset().gety() + root.getdata())
                # upgrade gradient by selected optimizer
                if self.optimizer == "GD":
                    self.weight_list[i + 1], self.gradient_list[i] = Optimizer.GD(self, root, self.weight_list[i])

                elif self.optimizer == "SGD":
                    self.weight_list[i + 1], self.gradient_list[i] = Optimizer.SGD(self, self.weight_list[i],
                                                                                   self.passer_list[i])

        # mini batch type gradient descent
        elif self.optimizer == "minibatchgd" or self.optimizer == "Adam":
            for i in range(self.epoch):
                # get mini batch
                minisets = self.nn.dataset.gettrainset().getminiset()
                epoch_weight_list = [copy.deepcopy(self.weight_list[i])]
                epoch_loss_list = np.zeros(len(minisets))

                # GD for every mini batch
                for j in range(len(minisets)):
                    X_bar = minisets[j]
                    self.passer_list[i].append(Optimizer.forword(X_bar.getX(), epoch_weight_list[j], layer_list))

                    root = self.passer_list[i][j][-1]

                    root.back = 2 * (-X_bar.gety() + root.getdata())
                    epoch_loss_list[j] = Optimizer.LossFunc.quadra_loss(X_bar.gety(), root.getdata())

                    weight, gradient = Optimizer.GD(self, root, epoch_weight_list[j])
                    epoch_weight_list.append(weight)

                self.weight_list[i + 1] = epoch_weight_list[-1]
                self.gradient_list[i] = gradient

                self.loss_list[i] = sum(epoch_loss_list) / len(epoch_loss_list)

                # learnign rate decay

                self.lrdecay(i)
                # every epoch shuffle the dataset
                self.nn.dataset.distribute()

        used_time = time.time() - start_time
        print('Training time: %.4f' % (used_time))
        print('Training loss: %.6f' % self.loss_list[-1])

    def test(self):
        """
        Use trained weight on testset for the evaluation of the model
        :return: model prediction and loss on the testset
        """
        weight = self.weight_list[-1]
        layer_list = self.nn.getlayers()
        testset = self.nn.dataset.gettestset()
        passer = testset.getX()

        passer_list = self.forword(passer, weight, layer_list)
        predicted = passer_list[-1].getdata()
        loss = Optimizer.LossFunc.quadra_loss(testset.gety(), predicted)
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
