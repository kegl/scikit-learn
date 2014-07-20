import numpy as np
from scipy import sparse

from sklearn.utils.testing import assert_array_almost_equal
from nose.tools import assert_true

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import brier_score
from sklearn.calibration import ProbabilityCalibrator
from sklearn.calibration import sigmoid_calibration, _SigmoidCalibration


def test_calibration():
    """Test calibration objects with isotonic and sigmoid"""
    n_samples = 500
    X, y = make_classification(n_samples=2 * n_samples, n_features=6,
                               random_state=42)

    X[X < 0] = 0

    # split train, test and OOB for calibration
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Gaussian Nayes-Bayes
    # clf = GaussianNB()  # XXX : GaussianNB does not support sparse data !!!
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    prob_pos_lr = clf.predict_proba(X_test)[:, 1]

    # Gaussian Naive Bayes with isotonic calibration
    for this_X_train, this_X_test in [(X_train, X_test),
                                      (sparse.csr_matrix(X_train),
                                       sparse.csr_matrix(X_test))]:
        for method in ['isotonic', 'sigmoid']:
            ir_clf = ProbabilityCalibrator(clf, method=method)
            ir_clf.fit(this_X_train, y_train)
            prob_pos_lr_ir = ir_clf.predict_proba(this_X_test)[:, 1]

            assert_true(brier_score(y_test, prob_pos_lr) >
                        brier_score(y_test, prob_pos_lr_ir))


def test_sigmoid_calibration():
    """Test calibration values with Platt sigmoid model"""
    exF = np.array([5, -4, 1.0])
    exY = np.array([1, -1, -1])
    # computed from my python port of the C++ code in LibSVM
    AB_lin_libsvm = np.array([-0.20261354391187855, 0.65236314980010512])
    assert_array_almost_equal(AB_lin_libsvm, sigmoid_calibration(exF, exY), 3)
    lin_prob = 1. / (1. + np.exp(AB_lin_libsvm[0] * exF + AB_lin_libsvm[1]))
    sk_prob = _SigmoidCalibration().fit(exF, exY).predict(exF)
    assert_array_almost_equal(lin_prob, sk_prob, 6)
