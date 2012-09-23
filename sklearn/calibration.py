"""Calibration estimators."""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style

import numpy as np

from .base import BaseEstimator
from .linear_model import IsotonicRegression


class IsotonicCalibrator(BaseEstimator):
    """Probability calibration with Isotonic Regression

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator whose output decision function needs to be calibrated.

    Notes
    -----
    References:
    Obtaining calibrated probability estimates from decision trees
    and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    Transforming Classiﬁer Scores into Accurate Multiclass
    Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)
    """
    def __init__(self, estimator):
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
        df = self.estimator.decision_function(X_oob)
        if df.ndim > 1 and df.shape[1] > 1:
            raise ValueError('IsotonicCalibrator only support binary '
                             'classification.')
        df = df.ravel()
        order = np.argsort(df)
        df = df[order]
        self._ir = IsotonicRegression(y_min=0., y_max=1.)
        self._ir.fit(df, y_oob[order])
        return self

    def predict_proba(self, X):
        """
        This function return posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, 2]
            The predicted probas.
        """
        proba = self._ir.predict(self.estimator.decision_function(X).ravel())
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
