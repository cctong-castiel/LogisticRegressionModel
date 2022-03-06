from typing import Text, Union, List
import numpy as np
import warnings
from numbers import Number
import multiprocessing
from joblib import Parallel, delayed

cpu = multiprocessing.cpu_count() - 1

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
        self.id_2_class = None
        

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
            d_id_class = {}
            for index, y_ in enumerate(self.classes_):
                d_id_class[index] = y_
            self.id_2_class = d_id_class

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
            regu_term = 0.5 * (lambda_1 * np.abs(W) + lambda_2 * np.square(W))
        elif self.penalty is "l1":
            regu_term = 0.5 * lambda_1 * np.abs(W)
        else:
            regu_term = 0.5 * lambda_2 * np.square(W)

        return regu_term

    def loss_beta_term(self, weights: np.array,
                       lambda_1: Union[float, None],
                       lambda_2: Union[float, None]) -> np.array:

        if self.penalty is "elastic":
            regu_loss = 0.5 * (lambda_1 * np.abs(weights) + lambda_2 * np.square(weights))
        elif self.penalty is "l1":
            regu_loss = 0.5 * lambda_1 * np.abs(weights)
        else:
            regu_loss = 0.5 * lambda_2 * np.square(weights)

        return regu_loss

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
             y_hat: np.array,
             regu_loss: np.array) -> np.ndarray:

        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat)) + regu_loss

        return loss

    def train_softmax(self,
                      X: np.ndarray,
                      y: np.array,
                      index: int,
                      i: int,
                      lambda_1: Union[float, None],
                      lambda_2: Union[float, None]):

        # defining batches 
        start_i = i * self.batch_size
        end_i = start_i + self.batch_size
        xb, yb = X[start_i:end_i], y[start_i:end_i]

        # softmax function
        y_hat = self.softmax(np.dot(xb, self.W[index]) + yb)

        # regularization
        if self.penalty:
            regu_term = self.regularization(self.W[index], lambda_1, lambda_2)

        # getting the gradient of loss w.r.t parameters
        dW, db = self.gradients(xb, yb, y_hat, regu_term)

        # updating the parameters
        print(f"softmax path shape of dW: {dW.shape}")
        print(f"softmax path shape of db: {db.shape}")
        
        yield dW, db, regu_term

    def train_sigmoid(self,
                      X: np.ndarray,
                      y: np.array,
                      i: int,
                      lambda_1: Union[float, None],
                      lambda_2: Union[float, None]):
        
        # defining batches 
        start_i = i * self.batch_size
        end_i = start_i + self.batch_size
        xb, yb = X[start_i:end_i], y[start_i:end_i]

        # sigmoid function
        y_hat = self.sigmoid(np.dot(xb, self.W) + yb)

        # regularization
        if self.penalty:
            regu_term = self.regularization(self.W, lambda_1, lambda_2)

        # getting the gradient of loss w.r.t parameters
        dW, db = self.gradients(xb, yb, y_hat, regu_term)

        # updating the parameters
        print(f"sigmoid path shape of dW: {dW.shape}")
        print(f"sigmoid path shape of db: {db.shape}")

        yield dW, db, regu_term

    def softmax_pipe(self,
                     iter_: int,
                     m: int,
                     X: np.ndarray,
                     y: np.array,
                     lambda_1: Union[float, None],
                     lambda_2: Union[float, None],
                     losses: List[float]):

        for index, y_idx in enumerate(self.classes_):

            X_ = X[y == y_idx, :]
            y_ = y[y == y_idx]

            print(f"X_: {X_}\n y_: {y_}")

            for i in range((m-1) // self.batch_size + 1):
                
                dW, db, regu_term = next(self.train_softmax(
                    X_, y_, index, i, lambda_1, lambda_2
                ))

                self.W[index] -= self.lr * dW
                self.b[index] -= self.lr * db
                
            loss = self.loss(y, self.softmax(np.dot(X, self.W[index]) + self.b[index]), regu_term)
            losses.append(loss)
            print(f"loss in {iter_} is: {loss}")

    def sigmoid_pipe(self, 
                     iter_: int,
                     m: int,
                     X: np.ndarray,
                     y: np.array,
                     lambda_1: Union[float, None],
                     lambda_2: Union[float, None],
                     losses: List[float]):

        for i in range((m-1) // self.batch_size + 1):

            dW, db, regu_term = next(self.train_sigmoid(
                X, y, i, lambda_1, lambda_2
            ))

            self.W -= self.lr * dW
            self.b -= self.lr * db
    
        loss = self.loss(y, self.sigmoid(np.dot(X, self.W) + self.b), regu_term)
        losses.append(loss)
        print(f"loss in {iter_} is: {loss}")


    def fit(self, 
            X: np.ndarray,
            y: np.array) -> np.ndarray:

        if not isinstance(self.C, Number) or self.C < 0:
            raise ValueError(
                "Penalty term must be positive; got (C=%r)" % self.C)

        if self.l1_ratio is not None:
            warnings.warn(
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
                warnings.warn(
                    "Setting penalty='none' will ignore the C and l1_ratio parameters"
                )
                # Note that check for l1_ratio is done right above
            C_ = np.inf
            penalty = "l2"
        else:
            C_ = self.C
            penalty = self.penalty

        # set variables
        d_id_class = {}
        m, n = X.shape
        self.classes_ = np.unique(y)
        y_classes = len(self.classes_)

        for index, y_ in enumerate(self.classes_):
            d_id_class[index] = y_
        self.id_2_class = d_id_class

        # check multi_class
        self._check_multi_class()

        # determine lambda with C and penalty
        lambda_1, lambda_2 = self.lambda_penalty()

        # determine sampling weight

        # initializing weights and bias to zeros
        if self.multi_class:
            self.W = np.zeros((y_classes, n))
            self.b = np.zeros(y_classes)
        else:
            self.W = np.zeros((n))
            self.b = 0

        # normalize the inputs
        X = self.normalize(X)

        # losses and weights
        losses = []

        if self.multi_class is True:
            # run softmax function

            Parallel(
                n_jobs=cpu,
                backend="threading"
            )(delayed(self.softmax_pipe)(
                    iter_, m, X, y, lambda_1, lambda_2, losses
                )
            for iter_ in range(self.max_iter))  

        else:
            # run sigmoid function

            Parallel(
                n_jobs=cpu,
                backend="threading"
            )(delayed(self.sigmoid_pipe)(
                    iter_, m, X, y, lambda_1, lambda_2, losses
                )
            for iter_ in range(self.max_iter))

        return self.W, self.b

    def predict_proba(self, X):

        # normalizing
        X = self.normalize(X)

        if self.multi_class:
            arr_prop = np.zeros((len(self.classes_), X.shape[0]))

            for k, v in self.id_2_class.items():
                arr_prop[k] = self.softmax(np.dot(X, self.W[k]) + self.b[k])
                print(f"arr_prop now: {arr_prop}")
            
            arr_prop = arr_prop.T

        else:
            arr_prop = self.sigmoid(np.dot(X, self.W) + self.b)

        return arr_prop

    def predict(self, X):

        pred = np.ndarray(X.shape[0])

        preds = self.predict_proba(X)

        if self.multi_class:
            preds = preds.T
            max_prop = np.argmax(preds, axis=0)
            
            for index in self.id_2_class:
                pred[max_prop == index] = self.id_2_class[index]                             
        else:
            pred = np.where(preds > 0.5, 1, 0)
        
        return pred
