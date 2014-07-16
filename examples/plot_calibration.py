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
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD Style.

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score, calibration_plot
from sklearn.svm import SVC
from sklearn.calibration import IsotonicCalibrator
from sklearn.cross_validation import train_test_split

n_samples = 6000
n_bins = 8  # for calibration_plot
X, y = make_classification(n_samples=n_samples, n_features=6, n_informative=4,
                           random_state=42, n_clusters_per_class=1)

# split train, test and OOB for calibration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=42)
X_train_oob, X_oob, y_train_oob, y_oob = train_test_split(X_train, y_train,
                                                test_size=0.25, random_state=42)

# SVM with Platt
svc = SVC(C=1., kernel='linear', probability=True)
svc.fit(X_train, y_train)
prob_pos_svc = svc.predict_proba(X_test)[:, 1]

# Logistic Regression
lr = LogisticRegression(C=1., intercept_scaling=100.)
lr.fit(X_train, y_train)
prob_pos_lr = lr.predict_proba(X_test)[:, 1]

# Logistic Regression with isotonic calibration
lr_ir = IsotonicCalibrator(lr)
lr_ir.fit(X_train_oob, y_train_oob, X_oob, y_oob)
prob_pos_lr_ir = lr_ir.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
prob_pos_nb = nb.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with isotonic calibration
nb_ir = IsotonicCalibrator(nb)
nb_ir.fit(X_train_oob, y_train_oob, X_oob, y_oob)
prob_pos_nb_ir = nb_ir.predict_proba(X_test)[:, 1]

# Random Forests
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
prob_pos_rf = rf.predict_proba(X_test)[:, 1]

# Random Forests with isotonic calibration
rf_ir = IsotonicCalibrator(RandomForestClassifier())
rf_ir.fit(X_train_oob, y_train_oob, X_oob, y_oob)
prob_pos_rf_ir = rf_ir.predict_proba(X_test)[:, 1]

print "Brier scores: (the smaller the better)"

svc_score = brier_score(y_test, prob_pos_svc)
print "SVC + Platt: %1.3f" % svc_score
pt_svc, pp_svc = calibration_plot(y_test, prob_pos_svc, bins=n_bins)

lr_score = brier_score(y_test, prob_pos_lr)
print "LogisticRegression: %1.3f" % lr_score
pt_lr, pp_lr = calibration_plot(y_test, prob_pos_lr, bins=n_bins)

lr_ir_score = brier_score(y_test, prob_pos_lr_ir)
print "LogisticRegression (IR): %1.3f" % lr_ir_score
pt_lr_ir, pp_lr_ir = calibration_plot(y_test, prob_pos_lr_ir, bins=n_bins)

nb_score = brier_score(y_test, prob_pos_nb)
print "Gaussian Naive Bayes: %1.3f" % nb_score
pt_nb, pp_nb = calibration_plot(y_test, prob_pos_nb, bins=n_bins)

nb_ir_score = brier_score(y_test, prob_pos_nb_ir)
print "Gaussian Naive Bayes (IR): %1.3f" % nb_ir_score
pt_nb_ir, pp_nb_ir = calibration_plot(y_test, prob_pos_nb_ir, bins=n_bins)

rf_score = brier_score(y_test, prob_pos_rf)
print "Random Forests: %1.3f" % rf_score
pt_rf, pp_rf = calibration_plot(y_test, prob_pos_rf, bins=n_bins)

rf_ir_score = brier_score(y_test, prob_pos_rf_ir)
print "Random Forests (IR): %1.3f" % rf_ir_score
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
pl.plot(pp_svc, pt_svc, "gs-", label="SVC + Platt (%0.3f)" % svc_score)
pl.plot(pp_lr, pt_lr, "bs-", label="Log. Reg. (%0.3f)" % lr_score)
pl.plot(pp_lr_ir, pt_lr_ir, "bs--",
        label="Log. Reg. + IR (%0.3f)" % lr_ir_score)
pl.plot(pp_nb, pt_nb, "rs-",
        label="GaussianNB (%0.3f)" % nb_score)
pl.plot(pp_nb_ir, pt_nb_ir, "rs--",
        label="GaussianNB + IR (%0.3f)" % nb_ir_score)
pl.plot(pp_rf, pt_rf, "ks-", label="Random Forests (%0.3f)" % rf_score)
pl.plot(pp_rf_ir, pt_rf_ir, "ks--",
        label="Random Forests + IR (%0.3f)" % rf_ir_score)
pl.legend(loc="lower right")
pl.ylim([-0.05, 1.05])
pl.title('Calibration plots')
pl.show()
