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

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score, calibration_plot
from sklearn.calibration import ProbabilityCalibrator
from sklearn.cross_validation import train_test_split

make_blobs
n_samples = 50000
n_bins = 10  # for calibration_plot

X, y = make_blobs(n_samples=n_samples, n_features=2, centers=3, cluster_std=1.0,
                  center_box=(-10.0, 10.0), shuffle=False, random_state=42)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

# split train, test and OOB for calibration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,
                                                    random_state=42)

cv = 4
method = 'isotonic'
# method = 'sigmoid'

# Model with no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# Model with calibration
# Logistic Regression with isotonic calibration
clf_pc = ProbabilityCalibrator(clf, cv=cv, method=method)
clf_pc.fit(X_train, y_train)
prob_pos_clf_pc = clf_pc.predict_proba(X_test)[:, 1]

print "Brier scores: (the smaller the better)"

clf_score = brier_score(y_test, prob_pos_clf)
print "No calibration: %1.3f" % clf_score
pt_clf, pp_clf = calibration_plot(y_test, prob_pos_clf, n_bins=n_bins)

clf_pc_score = brier_score(y_test, prob_pos_clf_pc)
print "With calibration: %1.3f" % clf_pc_score
pt_clf_pc, pp_clf_pc = calibration_plot(y_test, prob_pos_clf_pc, n_bins=n_bins)

###############################################################################
# Plot calibration plots

import matplotlib.pyplot as plt
plt.close('all')

plt.figure()
for this_y in np.unique(y):
    this_X = X[y == this_y]
    plt.plot(this_X[:, 0], this_X[:, 1], 'x')

plt.figure()
order = np.lexsort((y_test, prob_pos_clf))
plt.plot(prob_pos_clf[order], 'xr', label='No calibration (%1.3f)' % clf_score)
plt.plot(prob_pos_clf_pc[order], 'b', linewidth=3,
        label='With calibration (%1.3f)' % clf_pc_score)
plt.plot(y_test[order], 'sg', linewidth=3, label='y')
plt.ylim([-0.05, 1.05])
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.xlabel("Predicted probability")
plt.ylabel("True probability")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.plot(pp_clf, pt_clf, "gs-", label="No calibration (%1.3f)" % clf_score)
plt.plot(pp_clf_pc, pt_clf_pc, "bs-",
        label="With calibration (%1.3f)" % clf_pc_score)
plt.legend(loc="upper left")
plt.ylim([-0.05, 1.05])
plt.title('Calibration plots')
plt.show()
