import numpy as np
import matplotlib.pyplot as plt

dtype = np.float32
epsilon = 1e-8
devise = 'cuda'


def getdtype():
    return dtype


def getepsilon():
    return epsilon


def getdevice():
    return devise


def plotloss(loss_list):
    """
    :return: plot loss flow during the training
    """
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(loss_list, label='loss')
    ax.legend(loc='upper right')
    ax.set_ylabel('Loss during the training')


def plotgradientnorm(grad_list):
    plt.style.use('seaborn-whitegrid')
    fig, axs = plt.subplots(len(grad_list))
    for i in range(len(grad_list)):
        gradient_norm_list = []
        for j in range(len(grad_list)):
            gradient_norm_list.append(np.linalg.norm(grad_list[j][i]))
        axs[i].plot(gradient_norm_list, label='norm 2')
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel('W' + str(i) + " norm")
