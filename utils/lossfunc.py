import numpy as np
class LossFunc:
    class Loss:
        def __init__(self, y_true, y_pred):
            self.y_true = y_true
            self.y_pred = y_pred
            self.name = ""

        def data(self, y_true, y_pred):
            self.y_true = y_true
            self.y_pred = y_pred

        def getname(self):
            return self.name

    class Logarithmic(Loss):
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

    class Quadratic(Loss):
        def __init__(self, y_true=None, y_pred=None, norm=0):
            self.y_true = y_true
            self.y_pred = y_pred
            self.norm = norm
            self.name = "Quadratic"

        def loss(self):
            return 1 / self.y_true.shape[0] * 0.5 * np.sum((self.y_pred - self.y_true) ** 2)

        def diff(self):
            return 2 * (self.y_pred - self.y_true)

    class MSE(Loss):
        def __init__(self, y_true=None, y_pred=None, x=None):
            self.y_true = y_true
            self.y_pred = y_pred
            self.x = x
            self.name = "MSE"

        def data(self, y_true, y_pred, x):
            self.y_true = y_true
            self.y_pred = y_pred
            self.x = x

        def loss(self):
            return 1 / np.shape(self.y_true)[0] * np.sum((self.y_pred - self.y_true) ** 2)

        def diff(self):
            return 2 / np.shape(self.y_true)[0] * np.sum(self.x @ (self.y_pred - self.y_true))

    class MSEContinue(Loss):
        def __init__(self, initial, boundary, inner):
            """

            :param initial:
            :param boundary:
            :param inner: array like, could be latent function h or its derivative h_x or its secound order h_xx
            """
            self.initial = initial
            self.boundary = boundary
            self.inner = inner
            self.name = "MSEContinue"

            self.loss_initial = LossFunc.MSE(self.boundary[0], self.boundary[1], self.boundary[2])
            self.loss_bound = []
            self.loss_in = LossFunc.MSE(self.inner[0], self.inner[1], self.inner[2])

            for func in self.boundary:
                self.loss_bound.append(LossFunc.MSE(func[0], func[1], func[2]))

        def loss_init(self):
            return self.loss_initial.loss()

        def loss_inner(self):
            return self.loss_in.loss()

        def loss(self):
            return self.loss_boundary() + self.loss_inner() + self.loss_f()

        def diff_boundary(self):
            return self.loss_bound.diff()

        def diff_in(self):
            return self.loss_in.diff()

        def diff_f(self):
            return self.loss_func.diff()

        def diff(self):
            return self.diff_boundary() + self.diff_in() + self.diff_f()