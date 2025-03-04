import numpy as np
import pandas as pd
import random

from conn2res.readout import Readout

class MemoryCapacity:
    """
    This class provides all the necessary steps in constructing the data, training on the data and validating performance based on the memory task.

    The code is strongly based on the work of Suárez et al. (2024) and is built to be compatible with Conn2res.
    Therefore, it contains the same functions as the readout.py module in Conn2res.
    
    Parameters
    ----------
    tau: int
        the number of lags calculated of y on x, by default 20
    n: int
        the length of the output signal y, later this is split in training and test data
    """
    def __init__(self, tau = 20, n = 1000):
        self.tau = tau
        self.n = n

    def fetch_data(self, train = 0.5, low=-1, high=1, seed=None, input_gain = 1):
        """
        Function generating a randomly sampled input u of length n + tau and an output y of length n

        Parameters
        ----------
        train: float, optional
            percentage of the total values n that you want to keep for training
        low: int, optional
            lower boundary of the input signal, by default -1
        high: int, optional
            upper boundary of the input signal, by default 1
        seed: int, optional
            seed used for random number generator, by default 0
        input_gain: int or float, optional
            factor to multiply the input signal with, the output is not multiplied by this value, by default 1

        Returns
        -------
        u, y: input (n + tau), output (n)
        """
        #generate an input u of length n + tau by randomly sampling from a uniform distribution
        rng = np.random.default_rng(seed=seed)
        
        u_train = rng.uniform(low=low, high=high, size=(int(self.n * train) + self.tau))[:, np.newaxis]
        #generate the output signal of length n, containing the first n values of u
        y_train = u_train[:-self.tau]

        u_test = rng.uniform(low=low, high=high, size=(int(self.n * (1 - train)) + self.tau))[:, np.newaxis]
        y_test = u_test[:-self.tau]

        #take into account possible input gain
        u_train *= input_gain
        u_test *= input_gain

        return u_train, u_test, y_train, y_test

class ReadoutMemory(Readout):
    """
    Class based on Readout that works with this specific Memory Capacity task

    Parameters
    ----------
    estimator: readout model, optional
        the model that uses the reservoir states to predict output y, by default None
    y: target output, optional
        the target output which is used to determine the necessary estimator
    """
    def __init__(self, estimator=None, y=None, tau=20):
        super().__init__(estimator=estimator, y=y)
        self.tau = tau

    def run_task(self, X, y, metric=None):
        """
        Parameters
        ----------
        X: numpy.ndarray
            array (rs_train, rs_test) containing both the training and the testing reservoir states
        y: numpy.ndarray
            array (y_train, y_test) containing both the training and the testing target outputs
        metric: str, optional
            the metric(s) that is (are) used to evaluate the performance of the reservoir

        Returns
        -------
        df_scores: pandas DataFrame
            DataFrame containing the scores for each of the different tau values
        """
        (x_train, x_test), (y_train, y_test) = X, y
        MC_train = []
        MC_test = []
        e_train = []
        e_test = []
        taus = []
        for tau in range(1, self.tau):
            self.train(x_train[tau:-self.tau + tau], y_train)

            y_pred = self._model.predict(x_train[tau:-self.tau + tau])
            if np.std(y_train) == 0 or np.std(y_pred) == 0:
                score = np.nan
            else:
                score = (np.corrcoef(y_train.flatten(), y_pred)[0][1])**2
            MC_train.append(score)
            e_train.append((y_train.flatten() - y_pred)**2)

            y_pred = self._model.predict(x_test[tau:-self.tau + tau])
            if np.std(y_test) == 0 or np.std(y_pred) == 0:
                score = np.nan
            else:
                score = (np.corrcoef(y_test.flatten(), y_pred)[0][1])**2
            MC_test.append(score)
            e_test.append((y_test.flatten() - y_pred)**2)
        return np.sum(np.array(MC_train)), np.sum(np.array(MC_test)), np.sum(np.array(e_train)), np.sum(np.array(e_test))
        # return MC_test
