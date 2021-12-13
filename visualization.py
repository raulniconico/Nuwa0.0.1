import numpy as np
import matplotlib.pyplot as plt


class Visual:
    def __init__(self, optim):
        self.optim = optim

    def plotloss(self):
        """
        :return: plot loss flow during the training
        """
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(self.optim.loss_list, label='loss')
        ax.legend(loc='upper right')
        ax.set_ylabel('Loss during the training')

    def plotgradientnorm(self):
        plt.style.use('seaborn-whitegrid')
        fig, axs = plt.subplots(len(self.optim.getgradientlist()[0]))
        for i in range(len(self.optim.getgradientlist()[0])):
            gradient_norm_list = []
            for j in range(len(self.optim.getgradientlist())):
                gradient_norm_list.append(np.linalg.norm(self.optim.getgradientlist()[j][i]))
            axs[i].plot(gradient_norm_list, label='norm 2')
            axs[i].legend(loc='upper right')
            axs[i].set_ylabel('W' + str(i) + " norm")
