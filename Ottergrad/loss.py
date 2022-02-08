from Ottergrad.autograd import Func
import Ottergrad.otternumpy as otnp
import numpy as np


class Quadratic(Func):
    def __init__(self, root, y_true=None, y_pred=None, norm=0):
        super().__init__(root)
        self.y_true = y_true
        self.y_pred = y_pred
        self.norm = norm
        self.name = "Quadratic"

    def loss(self):
        return Func(1 / self.y_true.shape[0] * 0.5 * otnp.sum((self.y_pred - self.y_true) ** 2))


class MSE(Func):
        def __init__(self, y_true=None, y_pred=None, x=None):
            super().__init__()
            self.y_true = y_true
            self.y_pred = y_pred
            self.x = x
            self.name = "MSE"

        def data(self, y_true, y_pred, x):
            self.y_true = y_true
            self.y_pred = y_pred
            self.x = x

        def loss(self):
            return 1 / otnp.shape(self.y_true)[0] * np.sum((self.y_pred - self.y_true) ** 2)


class MSEContinue(Func):
        def __init__(self, initial, boundary, inner):
            """

            :param initial:
            :param boundary:
            :param inner: array like, could be latent function h or its derivative h_x or its secound order h_xx
            """
            super().__init__()
            self.initial = initial
            self.boundary = boundary
            self.inner = inner
            self.name = "MSEContinue"

            self.loss_initial = MSE(self.boundary[0], self.boundary[1], self.boundary[2])
            self.loss_bound = []
            self.loss_in = MSE(self.inner[0], self.inner[1], self.inner[2])


