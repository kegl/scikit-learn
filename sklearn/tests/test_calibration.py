import numpy as np

from sklearn.utils.testing import assert_array_almost_equal
from nose.tools import assert_true

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score
from sklearn.calibration import IsotonicCalibrator
from sklearn.calibration import sigmoid_calibration, SigmoidCalibration


def test_isotonic_calibration():
    n_samples = 500
    X, y = make_classification(n_samples=2 * n_samples, n_features=6,
                               random_state=42)

    # split train, test and OOB for calibration
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Logistic Regression
    clf = LogisticRegression(C=1., intercept_scaling=100.)
    clf.fit(X_train, y_train)
    prob_pos_lr = clf.predict_proba(X_test)[:, 1]

    # Naive Bayes with isotonic calibration
    ir_clf = IsotonicCalibrator(clf)
    ir_clf.fit(X_train, y_train)
    prob_pos_lr_ir = ir_clf.predict_proba(X_test)[:, 1]

    assert_true(brier_score(y_test, prob_pos_lr) >
                brier_score(y_test, prob_pos_lr_ir))


def test_sigmoid_calibration():
    exF = np.array([5, -4, 1.0])
    exY = np.array([1, -1, -1])
    # computed from my python port of the C++ code in LibSVM
    AB_lin_libsvm = np.array([-0.20261354391187855, 0.65236314980010512])
    assert_array_almost_equal(AB_lin_libsvm, sigmoid_calibration(exF, exY), 3)
    lin_prob = 1. / (1. + np.exp(AB_lin_libsvm[0] * exF + AB_lin_libsvm[1]))
    assert_array_almost_equal(lin_prob, SigmoidCalibration().fit(exF, exY).predict(exF), 6)
