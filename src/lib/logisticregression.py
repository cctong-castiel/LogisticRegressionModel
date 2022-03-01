from typing import Text, Union
import numpy as np
import warning
from number import Number


class _LogisticRegression():

    def __init__(self,
                 penalty: Text = "l2",
                 C: float = 1.0,
                 random_state: int = None,
                 max_iter: int = 500,
                 batch_size: int = 100,
                 verbose: int = 0,
                 n_jobs: int = None,
                 l1_ratio: int = None,
                 lr: float = 0.001):

        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.lr = lr
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

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:

        m, n = X.shape

        # normalizing all the n features of X
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        return X

    def lambda_penalty(self) -> Union[float, None]:
        
        lambda_1, lambda_2 = None, None

        if self.penalty is "l1":
            lambda_1 = 1 / self.C
        elif self.penalty is "l2":
            lambda_2 = 1 / self.C
        else:
            lambda_1 = (1 / self.C) * (1 - self.l1_ratio)
            lambda_2 = (1 / self.C) * self.l1_ratio

        return lambda_1, lambda_2

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z: np.ndarray):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def regularization(self, W: np.ndarray,
                       lambda_1: Union[float, None],
                       lambda_2: Union[float, None]) -> np.array:

        regu_term = None
        if self.penalty is "elastic":
            regu_term = lambda_1 * np.abs(W) + lambda_2 * np.square(W)
        elif self.penalty is "l1":
            regu_term = lambda_1 * np.abs(W)
        else:
            regu_term = lambda_2 * np.square(W)

        return regu_term

    @staticmethod
    def gradients(X: np.ndarray, 
                  y: np.array,
                  y_hat: np.array,
                  regu_term: np.array):

        m = X.shape[0]

        # gradient of loss w.r.t. weights
        dW = (1/m) * (np.dot(X.T, (y_hat - y)) + regu_term)

        # gradient of loss w.r.t. bias
        db = (1/m) * np.sum((y_hat - y))

        return dW, db

    @staticmethod
    def loss(y: np.array,
             y_hat: np.array) -> np.ndarray:

        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))

        return loss

    def fit(self, 
            X: np.ndarray,
            y: np.array,
            sample_weight: Union[float, None]=None) -> np.ndarray:

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

        # determine lambda with C and penalty
        lambda_1, lambda_2 = self.lambda_penalty()

        # determine sampling weight

        # get observation and feature number 
        m, n = X.shape

        # initializing weights and bias to zeros
        W = np.zeros((n, 1))
        b = 0

        # normalize the inputs
        X = self.normalize(X)

        if self.multi_class is True:
            # run softmax function

            for iter_ in range(self.max_iter):
                
                for i in range((m-1) // self.batch_size + 1):

                    # defining batches 
                    start_i = i * self.batch_size
                    end_i = start_i + self.batch_size
                    xb, yb = X[start_i:end_i], y[start_i:end_i]

                    # softmax function
                    y_hat = self.softmax(np.dot(xb, W) + yb)

                    # regularization
                    if self.penalty:
                        regu_term = self.regularization(W, lambda_1, lambda_2)

                    # getting the gradient of loss w.r.t parameters
                    dW, db = self.gradients(xb, yb, y_hat, regu_term)

                    # updating the parameters
                    W -= self.lr * dW
                    b -= self.lr * db

        else:
            # run sigmoid function

            for iter_ in range(self.max_iter):
                
                for i in range((m-1) // self.batch_size + 1):

                    # defining batches 
                    start_i = i * self.batch_size
                    end_i = start_i + self.batch_size
                    xb, yb = X[start_i:end_i], y[start_i:end_i]

                    # sigmoid function
                    y_hat = self.sigmoid(np.dot(xb, W) + yb)

                    # regularization
                    if self.penalty:
                        lambda_1, lambda_2 = self.lambda_penalty()
                        regu_term = self.regularization(W, lambda_1, lambda_2)

                    # getting the gradient of loss w.r.t parameters
                    dW, db = self.gradients(xb, yb, y_hat, regu_term)

                    # updating the parameters
                    W -= self.lr * dW
                    b -= self.lr * db
            
        return W, b
