from typing import Text
import numpy as np
import warning
from number import Number


class _LogisticRegression():

    def __init__(self,
                 penalty: Text = "l2",
                 C: float = 1.0,
                 random_state: int = None,
                 verbose: int = 0,
                 n_jobs: int = None,
                 l1_ratio: int = None):

        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.multi_class = True

    def _check_multi_class(self):

        num_classes = len(self.classes_)

        if num_classes <= 1:
            raise ValueError(
                'multi_class should be at least 2 classes or above. Right now the number of classes is %s'
                % num_classes
            )
        elif num_classes == 2:
            self.multi_class = False
        else:
            self.multi_class = True

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def gradients(X, y, y_hat):

        m = X.shape[0]

        # gradient of loss w.r.t. weights
        dW = (1/m) * np.dot(X.T, (y_hat - y))

        # gradient of loss w.r.t. bias
        db = (1/m) * np.sum((y_hat - y))

        return dW, db

    def loss(y, y_hat):
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
        return loss



    def fit(self, X, y, sample_weight=None):

        if not isinstance(self.C, Number) or self.C < 0:
            raise ValueError(
                "Penalty term must be positive; got (C=%r)" % self.C)

        if self.l1_ratio is not None:
            warning.warn(
                "l1_ratio parameter is only used when penalty is 'elasticnet'. Got "
                "(penalty=%s)" % self.penalty
            )
        elif self.penalty is "elasticnet":
            if (
                not isinstance(self.l1_ratio, Number)
                or self.l1_ratio < 0
                or self.l1_ratio > 1
            ):
                raise ValueError(
                    "l1_ratio must be between 0 and 1; got (l1_ratio=%s)" % self.l1_ratio
                )

        if self.penalty is "none":
            if self.C is not 1.0:
                warning.warn(
                    "Setting penalty='none' will ignore the C and l1_ratio parameters"
                )
                # Note that check for l1_ratio is done right above
            C_ = np.inf
            penalty = "l2"
        else:
            C_ = self.C
            penalty = self.penalty

        self.classes_ = np.unique(y)

        # check multi_class
        self._check_multi_class()

        if self.multi_class is True:
            # run softmax function

        else:
            # run sigmoid function
