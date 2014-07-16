"""Calibration estimators."""

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Balazs Kegl <balazs.kegl@gmail.com>
#
# License: BSD 3 clause

from math import log
import numpy as np

from scipy.optimize import fmin_bfgs

from .base import BaseEstimator, RegressorMixin, clone
from .isotonic import IsotonicRegression
from .naive_bayes import GaussianNB
from .cross_validation import check_cv


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
    def __init__(self, estimator=GaussianNB(), cv=2):
        self.estimator = estimator
        self.cv = cv

    def fit(self, X, y):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        cv : a cross-validation object

        Returns
        -------
        self : object
            returns an instance of self.
        """
        cv = check_cv(self.cv, X, y)
        self.models_ = []
        for train, test in cv:
            this_estimator = clone(self.estimator)
            this_estimator.fit(X[train], y[train])
            if hasattr(this_estimator, "decision_function"):
                df = this_estimator.decision_function(X[test])
            else:
                df = this_estimator.predict_proba(X[test])[:, 1:]
            if df.ndim > 1 and df.shape[1] > 1:
                raise ValueError('IsotonicCalibrator only support binary '
                                 'classification.')
            df = df.ravel()
            this_ir = IsotonicRegression(y_min=0., y_max=1.)
            this_ir.fit(df, y[test])
            self.models_.append((this_estimator, this_ir))
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
        log_proba = np.zeros(X.shape[0])
        for this_estimator, this_ir in self.models_:
            if hasattr(this_estimator, "decision_function"):
                df = this_estimator.decision_function(X)
            else:
                df = this_estimator.predict_proba(X)[:, 1:]
            df = df.ravel()
            log_proba += np.log(this_ir.predict(df))

        log_proba /= len(self.models_)
        proba = np.exp(log_proba)
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
    a : float
        The slope.
    b : float
        The intercept.

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
    return AB_


class SigmoidCalibration(BaseEstimator, RegressorMixin):
    """Sigmoid regression model.

    Attributes
    ----------
    `a_` : float
        The slope.

    `b_` : float
        The intercept.
    """
    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape=(n_samples,)
            Training data.

        y : array-like, shape=(n_samples,)
            Training target.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self.a_, self.b_ = sigmoid_calibration(X, y)
        return self

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        `T_` : array, shape (n_samples,)
            The predicted data.
        """
        return 1. / (1. + np.exp(self.a_ * T + self.b_))
