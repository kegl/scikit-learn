"""
============================
Probability Calibration plot
============================

When performing classfication you often want to predict, not only
the class label, but also the associated probability. This probability
gives you some kind of confidence on the prediction. This example
demonstrates the use of various estimators that can predict probabilities
as well as some estimators that can be used to calibrate the
output score into a probability. The accuracy is estimated
with Brier's score (see http://en.wikipedia.org/wiki/Brier_score).

"""
print __doc__

# Author: Mathieu Blondel <mathieu@mblondel.org>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD Style.

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score, calibration_plot
from sklearn.svm import SVC
from sklearn.calibration import IsotonicCalibrator
from sklearn.cross_validation import train_test_split

n_samples = 4000
n_bins = 8  # for calibration_plot
X, y = make_classification(n_samples=n_samples, n_features=6,
                           random_state=42)

# split train, test and OOB for calibration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=42)
X_train_oob, X_oob, y_train_oob, y_oob = train_test_split(X_train, y_train,
                                                test_size=0.5, random_state=42)

# Logistic Regression
clf = LogisticRegression(C=1., intercept_scaling=100.)
clf.fit(X_train, y_train)
prob_pos_lr = clf.predict_proba(X_test)[:, 1]

# SVM with Platt
clf = SVC(C=1., probability=True)
clf.fit(X_train, y_train)
prob_pos_svc = clf.predict_proba(X_test)[:, 1]

# Logistic Regression with isotonic calibration
ir_clf = IsotonicCalibrator(LogisticRegression(C=1., intercept_scaling=100.))
ir_clf.fit(X_train_oob, y_train_oob, X_oob, y_oob)
prob_pos_lr_ir = ir_clf.predict_proba(X_test)[:, 1]

# SGD Huber
clf = SGDClassifier(loss="modified_huber", seed=0, penalty='l2', n_iter=200)
clf.fit(X_train, y_train)
prob_pos_sgd_huber = clf.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes
ir_clf = IsotonicCalibrator(GaussianNB())
ir_clf.fit(X_train_oob, y_train_oob, X_oob, y_oob)
prob_pos_nb_ir = ir_clf.predict_proba(X_test)[:, 1]

# Random Forests
ir_clf = IsotonicCalibrator(RandomForestClassifier())
ir_clf.fit(X_train_oob, y_train_oob, X_oob, y_oob)
prob_pos_rf_ir = ir_clf.predict_proba(X_test)[:, 1]

print "Brier scores: (the smaller the better)"
print "LogisticRegression: %1.3f" % brier_score(y_test, prob_pos_lr)
pt_lr, pp_lr = calibration_plot(y_test, prob_pos_lr, bins=n_bins)

print "SVC + Platt: %1.3f" % brier_score(y_test, prob_pos_svc)
pt_svc, pp_svc = calibration_plot(y_test, prob_pos_svc, bins=n_bins)

print "LogisticRegression (IR): %1.3f" % brier_score(y_test, prob_pos_lr_ir)
pt_lr_ir, pp_lr_ir = calibration_plot(y_test, prob_pos_lr_ir, bins=n_bins)

print "SGD (modified_huber): %1.3f" % brier_score(y_test, prob_pos_sgd_huber)
pt_sgd, pp_sgd = calibration_plot(y_test, prob_pos_sgd_huber, bins=n_bins)

print "Gaussian Naive Bayes (IR): %1.3f" % brier_score(y_test, prob_pos_nb_ir)
pt_nb_ir, pp_nb_ir = calibration_plot(y_test, prob_pos_nb_ir, bins=n_bins)

print "Random Forests (IR): %1.3f" % brier_score(y_test, prob_pos_rf_ir)
pt_rf_ir, pp_rf_ir = calibration_plot(y_test, prob_pos_rf_ir, bins=n_bins)

###############################################################################
# Plot calibration plots

import pylab as pl
pl.close('all')

pl.figure()
order = np.lexsort((y_test, prob_pos_lr))
pl.plot(prob_pos_lr[order], 'xr', label='Logistic prob(y_)')
pl.plot(prob_pos_lr_ir[order], 'b', linewidth=3,
        label='Isotonic Logistic prob(y_)')
pl.plot(y_test[order], 'sg', linewidth=3, label='y')
pl.ylim([-0.05, 1.05])
pl.legend(loc="center left")
pl.show()

pl.figure()
pl.xlabel("Predicted probability")
pl.ylabel("True probability")
pl.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
pl.plot(pp_lr, pt_lr, "rs-", label="LogisticRegression")
pl.plot(pp_svc, pt_svc, "gs-", label="SVC + Platt")
pl.plot(pp_lr_ir, pt_lr_ir, "bs-", label="LogisticRegression (IR)")
pl.plot(pp_sgd, pt_sgd, "cs-", label="SGDClassifier (huber)")
pl.plot(pp_nb_ir, pt_nb_ir, "ms-", label="GaussianNB (IR)")
pl.plot(pp_rf_ir, pt_rf_ir, "ks-", label="Random Forests (IR)")
pl.legend(loc="lower right")
pl.ylim([-0.05, 1.05])
pl.show()
