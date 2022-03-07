from Ottergrad.autograd import Tensor, Func, checktensor
import Ottergrad.otternumpy as on
import numpy as np


class Quadratic(Func):
    def __init__(self, root=None, y_pred=None, y_true=None, norm=0):
        super().__init__(root)
        self.y_true = y_true
        self.y_pred = y_pred
        self.norm = norm
        self.name = "Quadratic"

    def __call__(self, y_pred: Tensor, y_true: Tensor, norm=0):
        self.y_pred = y_pred
        self.y_true = y_true
        self.norm = norm
        assert type(y_pred) is Tensor and type(y_true) is Tensor, "y_pred and y_true must be Tensor type"
        func = Func(1 / self.y_true.shape()[0] * 0.5 * on.sum((self.y_pred - self.y_true) ** 2))
        self.setroot(func.getroot())
        if y_pred.getdata() is not None and y_true.getdata() is not None:
            func.forward()
        return func


class Logarithmic(Func):
    def __init__(self, y_true, y_pred, eps=1e-16):
        """
        Loss function we would like to optimize (minimize)
        We are using Logarithmic Loss
        http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
        """
        super().__init__(y_true, y_pred)
        self.eps = eps
        self.name = "Logarithmic"

    def loss(self):
        self.y_pred = np.maximum(self.y_pred, self.eps)
        self.y_pred = np.minimum(self.y_pred, (1 - self.eps))
        return -(np.sum(self.y_true * np.log(self.y_pred)) + np.sum(
            (1 - self.y_true) * np.log(1 - self.y_pred))) / len(self.y_true)


class MSE(Func):

    def __init__(self, rppt=None, y_true=None, y_pred=None):
        super().__init__(rppt)
        self.y_true = y_true
        self.y_pred = y_pred
        self.name = "MSE"

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Func:
        self.y_pred = y_pred
        self.y_true = y_true
        assert type(y_pred) is Tensor and type(y_true) is Tensor, "y_pred and y_true must be Tensor type"
        func = Func(1 / on.shape(self.y_true)[0] * on.sum((self.y_pred - self.y_true) ** 2))
        self.setroot(func.getroot())
        if y_pred.getdata() is not None and y_true.getdata() is not None:
            func.forward()
        return func


class _MSEContinue(Func):
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

