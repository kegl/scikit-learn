"""Calibration estimators."""

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD 3 clause

from math import log
import numpy as np

from scipy.optimize import fmin_bfgs

from .base import BaseEstimator
from .isotonic import IsotonicRegression
from .naive_bayes import GaussianNB


class IsotonicCalibrator(BaseEstimator):
    """Probability calibration with Isotonic Regression

    Parameters
    ----------
    estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs.

    Notes
    -----
    References:
    Obtaining calibrated probability estimates from decision trees
    and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    Transforming Classifier Scores into Accurate Multiclass
    Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)
    """
    def __init__(self, estimator=GaussianNB()):
        self.estimator = estimator

    def fit(self, X, y, X_oob, y_oob):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        X_oob : array-like, shape (n_samples, n_features)
            Out of bag training data used for calibration.

        y_oob : array-like, shape (n_samples,)
            Out of bag targets used for calibration.

        Returns
        -------
        self : object
            returns an instance of self.
        """
        self.estimator.fit(X, y)
        if hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X_oob)
        else:
            df = self.estimator.predict_proba(X_oob)[:, 1:]
        if df.ndim > 1 and df.shape[1] > 1:
            raise ValueError('IsotonicCalibrator only support binary '
                             'classification.')
        df = df.ravel()
        self._ir = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip')
        self._ir.fit(df, y_oob)
        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, 2)
            The predicted probas.
        """
        if hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X)
        else:
            df = self.estimator.predict_proba(X)[:, 1:]
        df = df.ravel()
        proba = self._ir.predict(df)
        proba = np.c_[1. - proba, proba]
        return proba

    def predict(self, X):
        """Predit the target of new samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        prob = self.predict_proba(X)[:, 1]
        return prob > 0.5


def sigmoid_calibration(df, y):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function for the samples.
    y : ndarray, shape (n_samples,)
        The targets.

    Returns
    -------
    P : ndarray, shape (n_samples,)
        The probas of being 1.

    Notes
    -----
    Reference: Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    F = df  # F follows Platt's notations
    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        return -(np.dot(T, np.log(P + tiny))
                 + np.dot(T1, np.log(1. - P + tiny)))

    def grad(AB):
        # gradient of the objective function
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        TEP_minus_T1P = P * (T * E - T1)
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    prob = 1. / (1. + np.exp(AB_[0] * F + AB_[1]))
    return prob
