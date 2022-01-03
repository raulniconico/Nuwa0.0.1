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


class DatasetPI2D:
    def __init__(self, h, x, t, sample_size=None, proportion=0.8, mini_batch=32):
        """

        :param h:
        :param x:
        :param t:
        :param sample_size: array_like, [N_0, N_b, N_f]
        :param proportion:
        :param mini_batch:
        """
        self.h = h
        self.x = x
        self.t = t
        self.sample_size = sample_size
        self.proportion = proportion
        self.mini_batch = mini_batch

        self.u = np.real(h)
        self.v = np.imag(h)
        self.h_norm = np.sqrt(self.u ** 2 + self.v ** 2)

        self.initset = None
        self.innerset = None
        self.boundaryset = None

        self.trainset = None
        self.testset = None

        self.distribute()

    def getinitset(self, sample_size=1):
        # initial data
        t0 = np.zeros(self.x.shape[0])
        u0 = self.u[:, 0]
        v0 = self.v[:, 0]
        return Dataset(np.concatenate((self.x, t0[:, None]), axis=1),
                       np.concatenate((u0[:, None], v0[:, None]), axis=1), proportion=sample_size,
                       mini_batch=self.mini_batch)

    def getupboundaryset(self, sample_size=1):
        x_ub = np.repeat(self.x[0], self.t.shape[0])
        u_ub = self.u[0, :]
        v_ub = self.v[0, :]
        return Dataset(np.concatenate((x_ub[:, None], self.t), axis=1),
                       np.concatenate((u_ub[:, None], v_ub[:, None]), axis=1), proportion=sample_size,
                       mini_batch=self.mini_batch)

    def getlowboundaryset(self, sample_size=1):
        x_lb = np.repeat(self.x[-1], self.t.shape[0])
        u_lb = self.u[-1, :]
        v_lb = self.v[-1, :]
        return Dataset(np.concatenate((x_lb[:, None], self.t), axis=1),
                       np.concatenate((u_lb[:, None], v_lb[:, None]), axis=1), proportion=sample_size,
                       mini_batch=self.mini_batch)

    def getboundaryset(self, sample_size=1):
        x_b = np.repeat(self.x[[0, -1]], self.t.shape[0])
        u_b = np.concatenate((self.u[0, :], self.u[-1, :]), axis=0)
        v_b = np.concatenate((self.v[0, :], self.v[-1, :]), axis=0)
        return Dataset(np.concatenate((x_b[:, None], np.tile(self.t, 2).flatten()[:, None]), axis=1),
                       np.concatenate((u_b[:, None], v_b[:, None]), axis=1), proportion=sample_size,
                       mini_batch=self.mini_batch)

    def getinnerset(self, sample_size=1):
        u_in = self.u[1: -1, 1: -1].flatten()
        v_in = self.v[1: -1, 1: -1].flatten()
        x_in = np.tile(self.x[1: -1], self.t.shape[0] - 2).flatten()
        t_in = np.repeat(self.t[1: -1], self.x.shape[0] - 2)
        return Dataset(np.concatenate((x_in[:, None], t_in[:, None]), axis=1),
                       np.concatenate((u_in[:, None], v_in[:, None]), axis=1), proportion=sample_size,
                       mini_batch=self.mini_batch)

    def distribute(self):
        self.initset = self.getinitset(sample_size=self.sample_size[0])
        self.boundaryset = self.getboundaryset(sample_size=self.sample_size[1])
        self.innerset = self.getinnerset(sample_size=self.sample_size[2])

        self.trainset = [self.initset.gettrainset(), self.boundaryset.gettrainset(), self.innerset.gettrainset()]
        self.testset = [self.initset.gettestset(), self.boundaryset.gettestset(), self.innerset.gettestset()]

    def gettrainset(self):
        return self.trainset

    def gettestset(self):
        return self.testset
