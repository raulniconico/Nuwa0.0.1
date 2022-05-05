import numpy as np

import Nuwa
from Ottergrad.autograd import Tensor
import Ottergrad.otternumpy as on
from Ottergrad.utils import getdtype


class Dataset:
    def __init__(self, ftr=None, label=None, proportion=0.8, shuffle: bool = False, mini_batch=1):
        """
        Dataset class provide tools to manage dataset

        :param ftr: ndarray, features, highly recommend ndarray
        :param label: ndarray, labels
        :param proportion: number between 0 and 1, the proportion of train dataset and test dataset
        :param shuffle: boolean,
        :param mini_batch mini batch size, 0 by default, in this case no mini batch size dataset will be generated

        ## Usage
        dataset = Dataset(ftr, label)

        """
        assert 0 <= proportion <= 1, "proportion must be number >= 0 and <= 1"
        self.ftr = ftr
        self.label = label
        self.proportion = proportion
        self.shuffle = shuffle
        self.mini_batch = mini_batch

        self.shuffledset = None
        self.trainset = None
        self.testset = None
        self.validationset = None

        self.subsets = []

        if self.label is not None and self.ftr is not None:
            self.distribute()

    def __iter__(self):
        """ Returns the Iterator object """
        return Dataset(self)

    def _shuffledata(self):

        assert self.ftr is not None, "No feature data"
        assert self.label is not None, "No label data"
        assert np.shape(self.ftr)[0] == np.shape(self.label)[0], 'shape 0 of the first position' + str(
            np.shape(self.ftr)[0]) + 'is not same as the second position ' + str(np.shape(self.label)[0])

        n = self.ftr.shape[0]
        self.shuffledset = Dataset(mini_batch=self.mini_batch)
        # random shuffle the data
        seq_num = np.arange(n, dtype=int)

        self.shuffledset.setftr(np.take(self.ftr, seq_num, axis=0))
        self.shuffledset.setlabel(np.take(self.label, seq_num, axis=0))

    def distribute(self) -> None:
        """
        This function will automatically distribute train and test dataset
        call this function to reshuffle all the dataset and also generate new train and test set

        """
        assert self.ftr is not None, "No feature data"
        assert self.label is not None, "No label data"

        n = np.shape(self.ftr)[0]
        if self.shuffle:
            self._shuffledata()

        self.trainset = Dataset(mini_batch=self.mini_batch)
        self.testset = Dataset(mini_batch=self.mini_batch)
        self.validationset = Dataset(mini_batch=self.mini_batch)

        if self.shuffle:
            # distribute trainset
            self.trainset.setftr(self.shuffledset.ftr[0: round(n * self.proportion), :])
            self.trainset.setlabel(self.shuffledset.label[0: round(n * self.proportion), :])

            # sample test dataset
            self.testset.setftr(self.shuffledset.ftr[round(n * self.proportion) + 1:, :])
            self.testset.setlabel(self.shuffledset.label[round(n * self.proportion) + 1:, :])
        else:
            self.trainset.setftr(self.ftr[0: round(n * self.proportion), :])
            self.trainset.setlabel(self.label[0: round(n * self.proportion), :])

            # sample test dataset
            self.testset.setftr(self.ftr[round(n * self.proportion) + 1:, :])
            self.testset.setlabel(self.label[round(n * self.proportion) + 1:, :])

    def setftr(self, ftr):
        assert ftr is not None, "ftr can't be None type"
        self.ftr = ftr

    def setlabel(self, label):
        assert label is not None, "label can't be None type"
        self.label = label

    def adddata(self, ftr, label):
        assert ftr is not None and label is not None and on.shape(ftr)[0] == \
               on.shape(label)[0], "input feature has different shape 0 as input label's shape"
        assert on.shape(ftr)[1:] == on.shape(self.ftr)[1:], \
            "new feature data has different dimension as dataset's dimension"
        assert on.shape(label)[1:] == on.shape(self.label)[1:], \
            "new label data has different dimension as dataset's dimension"

        if self.ftr is not None:
            on.concatenate((self.ftr, ftr), axis=0)
        else:
            self.setftr(ftr)
        if self.label is not None:
            on.concatenate((self.label, label), axis=0)
        else:
            self.setlabel(label)

        self.distribute()

    def getftr(self) -> Tensor:
        assert self.ftr is not None, "Dataset has no feature data"
        return Tensor(getdtype()(self.ftr))

    def getlabel(self) -> Tensor:
        assert self.label is not None, "Dataset has no label data"
        return Tensor(getdtype()(self.label))

    def getminibatch(self):
        return self.mini_batch

    def gettrainset(self):
        """
        :return: return train dataset with respect of proportion
        """
        # return Dataset(self.trainset[:, 0:self.ftr.shape[1]], self.trainset[:, self.ftr.shape[1]:],
        #                mini_batch=self.mini_batch)
        assert self.ftr is not None and self.label is not None, \
            "Train dataset has no train set, check it is distributed"
        if self.trainset is None:
            self.distribute()
        return self.trainset

    def gettestset(self):
        """
        :return: test dataset with respect of proportion
        """
        assert self.testset is not None, "Test data set has no train set, check it is distributed"
        return self.testset

    def getminiset(self):
        """
        get mini sets with mini batch size
        :return: Dataset array
        """
        assert np.shape(self.ftr)[0] == np.shape(self.label)[0], \
            "ftr dimension != label dimension" + str(np.shape(self.ftr)[0]) + " != " + str(np.shape(self.label)[0])
        assert self.mini_batch > 0 and type(self.mini_batch) is int, \
            "mini batch less or equal 0 or mini batch is not int"

        if self.trainset is None or self.testset is None:
            self.distribute()
        # split feature and label into mini batch size subset
        spilit_list = np.arange(self.mini_batch, np.shape(self.ftr)[0], self.mini_batch)
        miniftr = np.split(self.ftr, spilit_list)
        minilabel = np.split(self.label, spilit_list)

        for i in range(len(miniftr)):
            self.subsets.append(
                Dataset(miniftr[i], minilabel[i], shuffle=False, mini_batch=self.mini_batch))
        return self.subsets

    def concatenate(self, dataset, axis=0):
        assert dataset is not None, "dataset must not be None type"

        self.ftr = np.concatenate((self.getftr(), dataset.getftr()), axis=axis)
        self.label = np.concatenate((self.getlabel(), dataset.getlabel()), axis=axis)
        self.distribute()


def concatenate(datasets: Dataset, axis=0):
    assert type(datasets) is Dataset, "datasets type must be Dataset"
    dataset = Dataset()
    for set in datasets:
        if dataset.getftr() is None or dataset.getftr() is None:
            dataset.adddata(set.getftr(), set.getlabel())
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

        # self.distribute()

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
