import os
import time

from network import NN
from optimizers import Optimizer


class Session:
    def __init__(self, datasets=None, layer_list=None, optimizer=None, loss_func=None, epoch=100, mini_epoch=1, lr=1e-3,
                 decay_rate=0, mini_batch=64, pretrained=None):
        self.datasets = datasets
        self.layer_list = layer_list
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.epoch = epoch
        self.mini_epoch = mini_epoch
        self.lr = lr
        self.decay_rate = decay_rate
        self.mini_batch = mini_batch
        self.pretrained = pretrained

        self.nn = None
        self.optim = None
        self.weight_list = None
        self.visual = None
        self.savepath = None

        self.loss_list = []

    def addset(self, datasets):
        self.datasets = datasets

    def addnn(self, nn):
        self.nn = nn

    def addlayer(self, layer_list):
        self.layer_list = layer_list

    def addoptimizor(self, optimizer):
        self.optimizer = optimizer

    def addlossfunc(self, loss_function):
        self.loss_function = loss_function

    def addepoch(self, epoch):
        self.epoch = epoch

    def addlr(self, lr):
        self.lr = lr

    def adddecay(self, decay_rate):
        self.decay_rate = decay_rate

    def addmini_batch(self, mini_batch):
        self.mini_batch = mini_batch

    def initexp(self):
        """
        initial experiment folder, the folder will contains the result of the experiment

        :return: the folder name
        """
        if not os.path.isdir('runs/train'):
            os.makedirs("runs/train")

        i = 0

        while True:
            filepath = "runs/train/exp" + str(i)
            if os.path.isdir(filepath):
                i += 1
                pass
            else:
                os.makedirs(filepath)
                break
        return filepath

    def start(self):
        self.savepath = self.initexp()

        try:
            iter(self.datasets)
        except:
            self.datasets = [self.datasets]

        for i in range(self.epoch):
            start_time = time.time()
            for dataset in self.datasets:
                dataset.mini_batch = self.mini_batch
                self.nn = NN(self.layer_list)
                self.optim = Optimizer(dataset, self.nn, self.optimizer, self.loss_func, epoch=self.mini_epoch,
                                       lr=self.lr, decay_rate=self.decay_rate, pretrained=self.pretrained)

                self.optim.train()
                self.loss_list.append(self.optim.loss_list[-1])
                # visual = Visual(self.optim)
                # visual.plotloss()
                # visual.plotgradientnorm()

            self.pretrained = self.optim.getweightlist()[-1]

            if (i + 1) % 1 == 0:
                used_time = time.time() - start_time
                print("epoch " + str(i + 1) + ', Training time: %.4f' % used_time + ', Training loss: %.6f' %
                      self.loss_list[i])
