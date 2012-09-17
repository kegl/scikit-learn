"""
================
Calibration plot
================

"""
print __doc__

# Author: Mathieu Blondel <mathieu@mblondel.org>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score, calibration_plot
from sklearn.linear_model import IsotonicRegression
from sklearn.svm import SVC

# X, y = make_classification(n_samples=5000, random_state=42)

rng = np.random.RandomState(42)
n_samples = 1000
std = 1.0
X1 = std * rng.randn(n_samples, 2)
X2 = np.r_[np.array([2, 2]) + std * rng.randn(n_samples / 2, 2),
           np.array([5, 5]) + std * rng.randn(n_samples / 2, 2)]
X = np.concatenate([X1, X2], axis=0)
y = np.r_[np.zeros(n_samples), np.ones(n_samples)].ravel()

order = np.arange(len(X))
rng.shuffle(order)
X = X[order]
y = y[order]
del order

# Add some outliers and reshuffle
outliers_idx = rng.randint(n_samples * 2, size=100)
y[outliers_idx] = 1 - y[outliers_idx]

# split train, test and OOB for calibration
X_train, y_train = X[:n_samples], y[:n_samples]
X_train_oob, y_train_oob = X[:n_samples / 2], y[:n_samples / 2]
X_oob, y_oob = X[n_samples / 2:], y[n_samples / 2:]
X_test, y_test = X[n_samples:], y[n_samples:]
n_bins = 8  # for calibration_plot

# Logistic Regression
clf = LogisticRegression(C=1., intercept_scaling=100.)
clf.fit(X_train, y_train)
prob_pos_lr = clf.predict_proba(X_test)[:, 1]

# SVM with Platt
clf = SVC(C=1., probability=True)
clf.fit(X_train, y_train)
prob_pos_svc = clf.predict_proba(X_test)[:, 1]

# Ref:
# Obtaining calibrated probability estimates from decision trees
# and naive Bayesian classifiers, ICML 2001

# Logistic Regression with isotonic calibration
clf = LogisticRegression(C=1., intercept_scaling=100.)
clf.fit(X_train_oob, y_train_oob)

# compute binning
bins = np.linspace(0, 1.0, n_bins)
binids = np.digitize(clf.predict_proba(X_oob)[:, 1], bins) - 1
bin_proba = list()
for k in xrange(n_bins - 1):
    y_k = y_oob[binids == k]
    if y_k.size:
        bin_proba.append(np.mean(y_k))

bin_proba = np.array(bin_proba)

ir = IsotonicRegression()
prob_oob = np.sort(clf.predict_proba(X_oob)[:, 1])
prob_calibrated_oob = bin_proba[np.searchsorted(bins, prob_oob) - 1]

ir.fit(prob_oob, prob_calibrated_oob)
prob_pos_lr_ir = ir.predict(clf.predict_proba(X_test)[:, 1])

# SGD Log
clf = SGDClassifier(loss="log", seed=0, penalty='l2', n_iter=200)
clf.fit(X_train, y_train)
prob_pos_sgd_log = clf.predict_proba(X_test)[:, 1]

# SGD Huber
clf = SGDClassifier(loss="modified_huber", seed=0, penalty='l2', n_iter=200)
clf.fit(X_train, y_train)
prob_pos_sgd_huber = clf.predict_proba(X_test)[:, 1]

print "Brier scores: (the smaller the better)"
print "LogisticRegression: %1.3f" % brier_score(y_test, prob_pos_lr)
pt_lr, pp_lr = calibration_plot(y_test, prob_pos_lr, bins=n_bins)

print "SVC + Platt: %1.3f" % brier_score(y_test, prob_pos_svc)
pt_lr, pp_lr = calibration_plot(y_test, prob_pos_svc, bins=n_bins)

print "LogisticRegression (IR): %1.3f" % brier_score(y_test, prob_pos_lr_ir)
pt_lr_ir, pp_lr_ir = calibration_plot(y_test, prob_pos_lr_ir, bins=n_bins)

print "SGD (log): %1.3f" % brier_score(y_test, prob_pos_sgd_log)
pt_sgd, pp_sgd = calibration_plot(y_test, prob_pos_sgd_log, bins=n_bins)

print "SGD (modified_huber): %1.3f" % brier_score(y_test, prob_pos_sgd_huber)
pt_sgd2, pp_sgd2 = calibration_plot(y_test, prob_pos_sgd_huber, bins=n_bins)

###############################################################################
# Plot calibration plots

import pylab as pl

pl.close('all')

pl.figure()
pl.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='r')
pl.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='g')

pl.figure()
order = np.lexsort((y_test, prob_pos_lr))
pl.plot(prob_pos_lr[order], 'xr', label='prob(y_)')
pl.plot(prob_pos_lr_ir[order], 'b', linewidth=3, label='prob(y_) (IR)')
pl.plot(y_test[order], 'sg', linewidth=3, label='y')
pl.ylim([-0.05, 1.05])
pl.legend(loc="center left")
pl.show()

pl.figure()
pl.xlabel("Predicted probability")
pl.ylabel("True probability")
pl.plot([0, 1], [0, 1], "b", label="Perfectly calibrated")
pl.plot(pp_lr, pt_lr, "ms-", label="LogisticRegression")
pl.plot(pp_lr, pt_lr, "cs-", label="SVC + Platt")
pl.plot(pp_lr_ir, pt_lr_ir, "ks-", label="LogisticRegression (IR)")
pl.plot(pp_sgd, pt_sgd, "rs-", label="SGDClassifier (log)")
pl.plot(pp_sgd2, pt_sgd2, "gs-", label="SGDClassifier (huber)")
pl.legend(loc="lower right")
pl.ylim([-0.05, 1.05])
pl.show()
