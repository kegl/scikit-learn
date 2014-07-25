"""Calibration estimators."""

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Balazs Kegl <balazs.kegl@gmail.com>
#
# License: BSD 3 clause

from math import log
import numpy as np

from scipy.optimize import fmin_bfgs

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from .preprocessing import LabelBinarizer
from .utils import check_array, indexable, column_or_1d
from .isotonic import IsotonicRegression
from .naive_bayes import GaussianNB
from .cross_validation import _check_cv


class ProbabilityCalibrator(BaseEstimator, ClassifierMixin):
    """Probability calibration with Isotonic Regression or sigmoid

    Parameters
    ----------
    estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs.

    method : 'sigmoid' | 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parameteric approach.

    cv : integer or cross-validation generator, optional
        If an integer is passed, it is the number of folds (default 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    Notes
    -----
    References:
    Obtaining calibrated probability estimates from decision trees
    and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    Transforming Classifier Scores into Accurate Multiclass
    Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    def __init__(self, estimator=GaussianNB(), method='sigmoid', cv=3):
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def _preproc(self, estimator, X):
        n_classes = len(self.classes_)
        if hasattr(estimator, "decision_function"):
            df = estimator.decision_function(X)
            if df.ndim == 1:
                df = df[:, np.newaxis]
        elif hasattr(estimator, "predict_proba"):
            df = estimator.predict_proba(X)
            if n_classes == 2:
                df = df[:, 1:]
        else:
            raise RuntimeError('classifier has no decision_function or '
                               'predict_proba method.')

        idx_pos_class = np.arange(df.shape[1])

        return df, idx_pos_class

    def fit(self, X, y):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            returns an instance of self.
        """
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo'])
        y = column_or_1d(y)
        X, y = indexable(X, y)

        cv = _check_cv(self.cv, X, y, classifier=True)
        lb = LabelBinarizer()
        Y = lb.fit_transform(y)
        self.classes_ = lb.classes_
        self.models_ = []

        for train, test in cv:
            this_estimator = clone(self.estimator)
            this_estimator.fit(X[train], y[train])

            df, idx_pos_class = self._preproc(this_estimator, X[test])
            this_calibrators = []

            for k, this_df in zip(idx_pos_class, df.T):
                if self.method == 'isotonic':
                    this_calibrator = IsotonicRegression(y_min=0., y_max=1.,
                                                         out_of_bounds='clip')
                    this_calibrator.fit(this_df, Y[test, k])
                elif self.method == 'sigmoid':
                    this_calibrator = _SigmoidCalibration()
                    this_calibrator.fit(this_df, Y[test, k])
                else:
                    raise ValueError('method should be "sigmoid" or '
                                     '"isotonic". Got %s.' % self.method)
                this_calibrators.append(this_calibrator)
            self.models_.append((this_estimator, this_calibrators))

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
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo'])
        n_classes = len(self.classes_)
        log_proba = np.zeros((X.shape[0], n_classes))
        tiny = np.finfo(np.float).tiny

        for this_estimator, this_calibrators in self.models_:
            df, idx_pos_class = self._preproc(this_estimator, X)

            for k, this_df, this_calibrator in \
                    zip(idx_pos_class, df.T, this_calibrators):
                if n_classes == 2:
                    k += 1
                proba = this_calibrator.predict(this_df)
                proba = np.maximum(proba, tiny)  # to avoid log of 0 warning
                log_proba[:, k] += np.log(proba)

        log_proba /= len(self.models_)
        proba = np.exp(log_proba)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        return proba

    def predict(self, X):
        """Predict the target of new samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def sigmoid_calibration(df, y):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray, shape (n_samples,)
        The targets.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

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


class _SigmoidCalibration(BaseEstimator, RegressorMixin):
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
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        if len(X.shape) != 1:
            raise ValueError("X should be a 1d array")

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
        T = column_or_1d(T)
        return 1. / (1. + np.exp(self.a_ * T + self.b_))
