import numpy as np
import random


class Dataset:
    def __init__(self, X=None, y=None, proportion=0.8, shuffle=False, mini_batch=0):
        """
        Dataset class provide tools to manage dataset

        :param X: ndarray, features, highly recommand ndarray
        :param y: ndarray, labels
        :param proportion: number between 0 and 1, the proportion of train dataset and test dataset
        :param shuffle: boolean,
        :param mini_batch mini batch size, 0 by default, in this case no mini batch size dataset will be generated
        """
        self.X = X
        self.y = y
        self.trainset = None
        self.testset = None
        self.validationset = None
        self.proportion = proportion
        self.shuffle = shuffle
        self.mini_batch = mini_batch
        self.allset = None
        self.minisets = []

        # automatic distribution
        if self.X is not None and self.y is not None:
            self.distribute()

    # @classmethod
    # def imageset(cls, path, proportion = 0.8, shuffle = None):
    #     pass

    def add(self, X, y):
        self.X = X
        self.y = y
        self.distribute()

    def distribute(self):
        """
        This function will automatically distribute train and test dataset
        call this function to reshuffle all the dataset and also generate new train and test set
        """
        n = np.shape(self.X)[0]
        samples = np.concatenate((self.X, self.y), axis=1)
        if self.shuffle:
            random.shuffle(samples)
        # sample train and test dataset
        self.trainset = samples[0:round(n * self.proportion), :]
        self.testset = samples[round(n * self.proportion) + 1:, :]
        self.allset = np.concatenate((self.X, self.y), axis=1)

    def getX(self):
        return self.X

    def gety(self):
        return self.y

    def getallset(self):
        return self.allset

    def getminibatch(self):
        return self.mini_batch

    def gettrainset(self):
        """
        :return: return train dataset with respect of proportion
        """
        return Dataset(self.trainset[:, 0:self.X.shape[1]], self.trainset[:, self.X.shape[1]:],
                       mini_batch=self.mini_batch)

    def gettestset(self):
        """
        :return: test dataset with respect of proportion
        """
        return Dataset(self.testset[:, 0:self.X.shape[1]], self.testset[:, self.X.shape[1]:],
                       mini_batch=self.mini_batch)

    def getminiset(self):
        """
        get mini sets with mini batch size
        :return: Dataset array
        """
        spilit_list = np.arange(self.mini_batch, self.allset.shape[0], self.mini_batch)
        minisets = np.split(self.allset, spilit_list)
        for i in range(len(minisets)):
            self.minisets.append(
                Dataset(minisets[i][:, 0:self.X.shape[1]], minisets[i][:, self.X.shape[1]:], shuffle=False,
                        mini_batch=self.mini_batch))
        return self.minisets

    def concatenate(self, dataset, axis=0):
        self.X = np.concatenate((self.getX(), dataset.getX()), axis=axis)
        self.y = np.concatenate((self.gety(), dataset.gety()), axis=axis)
        self.allset = np.concatenate((self.X, self.y), axis=1)
        self.distribute()


def concatenate(datasets, axis=0):
    dataset = Dataset()
    for set in datasets:
        if dataset.getX() is None or dataset.getX() is None:
            dataset.add(set.getX(), set.gety())
        else:
            dataset.concatenate(set, axis=axis)
    dataset.distribute()

    return dataset
