"""Calibration estimators."""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style

import numpy as np

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
        X : array-like, shape = [n_samples, n_features]
            Training data.

        y : array-like, shape = [n_samples]
            Target values.

        X_oob : array-like, shape = [n_samples, n_features]
            Out of bag training data used for calibration.

        y_oob : array-like, shape = [n_samples]
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
        self._ir = IsotonicRegression(y_min=0., y_max=1.)
        self._ir.fit(df, y_oob)
        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, 2]
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
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            The predicted class.
        """
        prob = self.predict_proba(X)[:, 1]
        return prob > 0.5
