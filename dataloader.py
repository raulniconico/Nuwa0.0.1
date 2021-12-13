import numpy as np
import random


class Dataset:
    def __init__(self, X, y, proportion=0.8, shuffle=True, mini_batch=0):
        """
        :param X: ndarray
                    feature data
        :param y: labels
        :param proportion: number between 0 and 1
        :param shuffle: boolean, whether apply data shuffle
        """
        self.X = X
        self.y = y
        self.trainset = None
        self.testset = None
        self.validationset = None
        self.proportion = proportion
        self.shuffle = shuffle
        self.mini_batch = mini_batch
        self.allset = np.concatenate((X, y), axis=1)
        self.minisets = []

        if self.shuffle:
            # automatic distribution
            self.distribute()

        # generate subsets respect mini batch
        if self.mini_batch != 0:
            self.getminiset()

    # @classmethod
    # def imageset(cls, path, proportion = 0.8, shuffle = None):
    #     pass

    def distribute(self):
        """
        This function will automatically distribute train and test dataset
        """
        n = np.shape(self.X)[0]
        samples = np.concatenate((self.X, self.y), axis=1)
        random.shuffle(samples)
        # sample train and test dataset
        self.trainset = samples[0:round(n * self.proportion), :]
        self.testset = samples[round(n * self.proportion) + 1:, :]

    def getX(self):
        return self.X

    def gety(self):
        return self.y

    def getminibatch(self):
        return self.mini_batch

    def gettrainset(self):
        """
        :return: return train dataset with respect of proportion
        """
        return Dataset(self.trainset[:, 0:self.X.shape[1]], self.trainset[:, self.X.shape[1]:])

    def gettestset(self):
        """
        :return: test dataset with respect of proportion
        """
        return Dataset(self.testset[:, 0:self.X.shape[1]], self.testset[:, self.X.shape[1]:])

    def getminiset(self):
        spilit_list = np.arange(self.mini_batch, self.allset.shape[0], self.mini_batch)
        minisets = np.split(self.allset, spilit_list)
        for i in range(len(minisets)):
            self.minisets.append(
                Dataset(minisets[i][:, 0:self.X.shape[1]], minisets[i][:, self.X.shape[1]:], shuffle=False))
