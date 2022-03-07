import os
import time

from Nuwa.dataloader import Dataset
from Nuwa.network import NN
from Nuwa.optimizers import *
from Nuwa.utils import plotloss
from Ottergrad.autograd import Tensor, Func
from Nuwa.optimizers import SGD, Adam, _Optim, lrdecay


class Session:
    def __init__(self, dataset: Dataset = None, nn: NN = None, optim=None, loss_func: Func = None,
                 epoch=100, mini_epoch=1, lr=1e-3, decay_rate=0, mini_batch=64, pretrained=None):

        self.dataset = dataset
        self.nn = nn
        self.optim = optim
        self.loss_func = loss_func
        self.epoch = epoch
        self.mini_epoch = mini_epoch
        self.lr = lr
        self.decay_rate = decay_rate
        self.mini_batch = mini_batch
        self.pretrained = pretrained

        self.savepath = None

        self.loss_list = []

    def set(self, dataset):
        self.dataset = dataset

    def optimizor(self, optim):
        self.optim = optim

    def lossfunc(self, loss_func):
        self.loss_func = loss_func

    def epoch(self, epoch):
        self.epoch = epoch

    def lr(self, lr):
        self.lr = lr

    def decay(self, decay_rate):
        self.decay_rate = decay_rate

    def mini_batch(self, mini_batch):
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
        criterion = self.loss_func
        loss_list = []

        for epoch in range(self.epoch):
            print("################# begin " + str(epoch) + " epoch #################")
            # get mini batch
            minisets = self.dataset.gettrainset().getminiset()
            epoch_loss_list = []

            for miniset in minisets:
                y_pred = self.nn(miniset.getftr())
                loss = criterion(y_pred, miniset.getlabel())
                epoch_loss_list.append(loss.getroot().getdata())
                lr = lrdecay(self.lr, epoch, decay_rate=self.decay_rate)
                loss.backward()
                self.optim(self.nn.getweights(), lr=lr)
                self.optim.zero_grad(loss)

            loss_list.append(np.mean(epoch_loss_list))
            if epoch % 1 == 0:
                print("epoch: ", epoch, ",loss: ", loss_list[-1], ", lr: ", lr)

        plotloss(loss_list)

