import numpy as np
import copy
import network
from network import NN

class Optimizer:
    def __init__(self,nn,optimizer,batch_size=8,epoch=20000,lr=0.0001):
        """
        :param nn: input an NN class
        :param optimizer: optimizer as "GD", "SGD" etc
        :param batch_size: batch size for mini batch optimization
        :param epoch: epoch number
        :param lr: learning rate
        """

        self.nn = nn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.weight_list = None
        self.gradient_list = None
        self.loss_list = None
        self.passer_list = None

    class LossFunc:
        @staticmethod
        def log_loss(y_true, y_pred, eps=1e-16):
            """
            Loss function we would like to optimize (minimize)
            We are using Logarithmic Loss
            http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
            """
            y_pred = np.maximum(y_pred,eps)
            y_pred = np.minimum(y_pred,(1-eps))
            return -(np.sum(y_true * np.log(y_pred)) + np.sum((1-y_true)*np.log(1-y_pred)))/len(y_true)

        @staticmethod
        def quadra_loss(y_true, y_pred):
            return 1/y_true.shape[0] * 0.5*np.sum((y_pred-y_true)**2)

    class Node:
        def __init__(self, data,type):
            self.left = None
            self.right = None
            self.data = data
            self.type = type
            self.back = None

        def getleft(self):
            return self.left

        def getright(self):
            return self.right

        def gettype(self):
            return self.type

        def getdata(self):
            return  self.data

        def getback(self):
            return  self.back

    def getgradientlist(self):
        return self.gradient_list

    def getlosslist(self):
        return self.loss_list

    def getweightlist(self):
        return self.weight_list

    class WeightIni:
        @staticmethod
        def init_linear_weight(input_dim, output_dim):
            return np.random.uniform(-1, 1, (input_dim, output_dim))

        @staticmethod
        def init_BN_weight(x):
            gamma = np.sqrt(np.var(x))
            beta = np.mean(x)
            return gamma, beta

        @staticmethod
        def init_conv2D_kernel(shape):
            """
            :param shape: shape of kernel
            :return:
            """
            return np.random.random(shape)

        @staticmethod
        def initial_weight_list(layer_list):
            weight_list = []
            # initial weights in weight list by their type
            for layer in layer_list:
                # linear weight operation
                if layer.gettype() == "Linear":
                    weight_list.append(Optimizer.Node(Optimizer.WeightIni.init_linear_weight(layer.getinputdim(), layer.getoutputdim()),"weight"))
                # elif layer.gettype() == "BN":
                #     gamma, beta = Optimizer.WeightIni.init_BN_weight(X.getdata())
                #     weight_list.append(Optimizer.Node((gamma, beta),"weight"))
                # kernel parse operation
                elif layer.gettype() == "Conv2D":
                    weight_list.append(Optimizer.Node(Optimizer.WeightIni.init_conv2D_kernel(layer.getkernelsize()),"weight"))
                else:
                    return  NameError
            return weight_list

    @staticmethod
    def forword(passer, weight_list, layer_list):
        layer_num = len(layer_list)
        passer_list = [Optimizer.Node(passer, "data")]
        for i in range(layer_num):
            if layer_list[i].gettype() =='Linear':

                passer = passer@weight_list[i].getdata()
                # append binary tree after inner product of weight and previous layer
                node = Optimizer.Node(passer,"inner_product")
                node.left = passer_list[-1]
                node.right = weight_list[i]
                passer_list.append(node)

                passer = layer_list[i].getact(passer)
                #append binary tree after activation function
                node = Optimizer.Node(passer,layer_list[i].getactname())
                node.left = passer_list[-1]
                passer_list.append(node)

            elif layer_list[i].gettype() == "BN":
                # batch_mean = np.mean(passer)
                # batch_var = np.var(passer)
                passer = np.linalg.norm(passer)

                passer = weight_list[i].getdata()[0] * passer
                node = Optimizer.Node(passer,"inner_product")
                node.left = passer_list[-1]
                passer_list.append(node)

                passer = passer + weight_list[i].getdata()[1]
                node = Optimizer.Node(passer,"plus")
                node.left = passer_list[-1]
                passer_list.append(node)
            # elif layer_list[j].gettype() == "Conv2D":
            else: raise NameError
        return passer_list

    @staticmethod
    def backpropagation(node):
        if node.getleft() is not None:
            if node.gettype() == "inner_product":
                node.getleft().back = node.getback()@node.getright().getdata().T
                node.getright().back = node.getleft().getdata().T@node.getback()
            elif node.gettype() == "sigmoid":
                # node.getleft().back = np.multiply(node.getback(),np.multiply(NN.ActivationFunc.sigmoid(node.getback()),
                #                                                              (np.ones(node.getback().shape))-NN.ActivationFunc.sigmoid(node.getback())))
                node.getleft().back = np.multiply(node.getback(),np.multiply(NN.ActivationFunc.sigmoid(node.getback()),
                                                                             1-NN.ActivationFunc.sigmoid(node.getback())))
            elif node.gettype() == "ReLU":
                back = copy.deepcopy(node.getback())
                back[back<=0] = 0
                node.getleft().back = back
            elif node.gettype() == "LeakyReLU":
                back = copy.deepcopy(node.getback())
                back[back<0] = 0.01*back[back<0]
                node.getleft().back = back
            elif node.gettype() == "tanh":
                node.getleft().back = np.multiply((np.ones(node.getback().shape)-NN.ActivationFunc.tanh(node.getback())**2),
                                                  node.getback())
                # node.getleft().back = (np.ones(node.getback().shape)-NN.ActivationFunc.tanh(node.getback())**2)
            elif node.gettype() == "plus":
                node.getleft().back = node.getback()
                node.getright().back = node.getback()
            elif node.gettype() == "minus":
                node.getleft().back = node.getback()
                node.getright().back = -node.getback()
            elif node.gettype() == "none":
                node.getleft().back = node.getback()

            Optimizer.backpropagation(node.getleft())
        else:
            return

    def GD(self, root, weight_list):

        Optimizer.backpropagation(root)
        gradient_list = []

        for node in weight_list:
            node.data = node.data - self.lr * node.back
            gradient_list.append(node.back)
        return weight_list, gradient_list

    def SGD(self, weight_list, passer_list):
        # we resume mini-batch equals 1 each time
        """
        :param root:
        :param weight_list:
        :param passer_list:
        :return:
        """
        def init_random_node(node, random_num_list, mini_weight_list):
            node.data = node.data[random_num_list,:]
            node.back = None
            if node.getright() is not None:
                mini_weight_list.append(node.getright())
            if node.getleft() is not None:
                init_random_node(node.getleft(), random_num_list, mini_weight_list)
            else: return

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
            weight.data = weight.data - self.lr * mini_weight_list[-i-1].back
            gradient_list.append(mini_weight_list[-i-1].back)
            i = i + 1

        return weight_list, gradient_list

    def train(self):
        layer_list = self.nn.getlayers()
        layer_num = len(layer_list)

        # initial weight, loss and gradient list
        self.weight_list = [[] for i in range(self.epoch+1)]
        self.weight_list[0] = Optimizer.WeightIni.initial_weight_list(layer_list)
        self.loss_list = np.zeros(self.epoch)
        self.gradient_list = [[] for i in range(self.epoch)]
        self.passer_list = [[] for i in range(self.epoch)]

        # for GD and SGD, they use full dataset, so need only read X and y once
        if self.optimizer =="GD" or self.optimizer == "SGD":
            X = self.nn.dataset.gettrainset().getX()
            X = Optimizer.Node(X, "data")
            for i in range(self.epoch):
                # forward propagation
                self.passer_list[i] = Optimizer.forword(X.getdata(), self.weight_list[i],layer_list)
                root = self.passer_list[i][-1]

                # calculate loss by using: loss2 * (-self.nn.dataset.gettrainset().gety() + root.getdata())
                self.loss_list[i] = Optimizer.LossFunc.quadra_loss(self.nn.dataset.gettrainset().gety(),root.getdata())
                root.back = 2 * (-self.nn.dataset.gettrainset().gety() + root.getdata())
                # upgrade gradient by selected optimizer
                if self.optimizer =="GD":
                    self.weight_list[i+1], self.gradient_list[i] = Optimizer.GD(self, root, self.weight_list[i])

                elif self.optimizer =="SGD":
                    self.weight_list[i+1], self.gradient_list[i]  = Optimizer.SGD(self, self.weight_list[i], self.passer_list[i])

        elif self.optimizer =="minibatchgd":
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
                    epoch_loss_list[j] = Optimizer.LossFunc.quadra_loss(X_bar.gety(),root.getdata())


                    weight, gradient = Optimizer.GD(self, root, epoch_weight_list[j])
                    epoch_weight_list.append(weight)

                self.weight_list[i+1]= epoch_weight_list[-1]
                self.gradient_list[i] = gradient

                self.loss_list[i] = sum(epoch_loss_list)/len(epoch_loss_list)

                # every epoch shuffle the dataset
                self.nn.dataset.distribute()
