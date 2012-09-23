from nose.tools import assert_true

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score
from sklearn.calibration import IsotonicCalibrator


def test_isotonic_calibration():
    n_samples = 500
    X, y = make_classification(n_samples=2 * n_samples, n_features=6,
                               random_state=42)

    # split train, test and OOB for calibration
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_train_oob, y_train_oob = X[:n_samples / 2], y[:n_samples / 2]
    X_oob, y_oob = X[n_samples / 2:], y[n_samples / 2:]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Logistic Regression
    clf = LogisticRegression(C=1., intercept_scaling=100.)
    clf.fit(X_train, y_train)
    prob_pos_lr = clf.predict_proba(X_test)[:, 1]

    # Logistic Regression with isotonic calibration
    ir_clf = IsotonicCalibrator(LogisticRegression(C=1., intercept_scaling=100.))
    ir_clf.fit(X_train_oob, y_train_oob, X_oob, y_oob)
    prob_pos_lr_ir = ir_clf.predict_proba(X_test)[:, 1]

    assert_true(brier_score(y_test, prob_pos_lr) > brier_score(y_test, prob_pos_lr_ir))
