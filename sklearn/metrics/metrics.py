import warnings
warnings.warn("sklearn.metrics.metrics is deprecated and will be remove in "
             "0.18. Please import from sklearn.metrics",
             DeprecationWarning)


from .ranking import auc
from .ranking import average_precision_score
from .ranking import label_ranking_average_precision_score
from .ranking import log_loss
from .ranking import precision_recall_curve
from .ranking import roc_auc_score
from .ranking import roc_curve
from .ranking import hinge_loss

from .classification import accuracy_score
from .classification import classification_report
from .classification import confusion_matrix
from .classification import f1_score
from .classification import fbeta_score
from .classification import hamming_loss
from .classification import jaccard_similarity_score
from .classification import matthews_corrcoef
from .classification import precision_recall_fscore_support
from .classification import precision_score
from .classification import recall_score
from .classification import zero_one_loss

from .regression import explained_variance_score
from .regression import mean_absolute_error
from .regression import mean_squared_error
from .regression import r2_score

# Deprecated in 0.16
from .ranking import auc_score


def _check_and_normalize(y_true, y_prob):
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    labels = np.unique(y_true)

    if len(labels) != 2:
        raise ValueError("Only binary classification is supported.")

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")

    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")

    y_true = y_true.copy()
    y_true[y_true == labels[0]] = 0
    y_true[y_true == labels[1]] = 1

    return y_true


def brier_score(y_true, y_prob):
    """Compute the Brier score

    The smaller the Brier score, the better.

    Across all items in a set N predictions, the Brier score measures the
    mean squared difference between (1) the predicted probability assigned
    to the possible outcomes for item i, and (2) the actual outcome.
    Therefore, the lower the Brier score is for a set of predictions, the
    better the predictions are calibrated. Note that the Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1).

    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another).

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets.

    y_prob : array, shape = [n_samples]
        Probabilities of the positive class.

    Returns
    -------
    score : float
        Brier score

    References
    ----------
    http://en.wikipedia.org/wiki/Brier_score
    """
    y_true = _check_and_normalize(y_true, y_prob)
    return np.mean((y_true - y_prob) ** 2)


def calibration_plot(y_true, y_prob, n_bins=5):
    """Compute true and predicted probabilities for a calibration plot

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    n_bins : int
        Number of bins. A bigger number requires more data.

    Returns
    -------
    prob_true : array, shape (n_bins,)
        The true probability in each bin.

    prob_pred : array, shape (n_bins,)
        The predicted probability in each bin.
    """
    y_true = _check_and_normalize(y_true, y_prob)

    # Adaptive binning
    bins = np.linspace(0, y_true.size - 1, n_bins + 1).astype(np.int)
    bins = np.sort(y_prob)[bins]
    bins[0] = 0.
    bins[-1] = 1.

    # # Fixed binning
    # bins = np.linspace(0., 1., n_bins + 1)

    binids = np.digitize(y_prob, bins) - 1
    ids = np.arange(len(y_true))

    prob_true = []
    prob_pred = []

    for binid in np.unique(binids):
        sel = ids[binids == binid]
        prob_true.append(np.mean(y_true[sel]))
        prob_pred.append(np.mean(y_prob[sel]))

    return np.array(prob_true), np.array(prob_pred)
