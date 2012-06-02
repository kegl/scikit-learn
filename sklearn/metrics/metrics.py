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
    """Compute the Brier score.

    The smaller the Brier score, the better.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets.

    y_prob : array, shape = [n_samples]
        Probabilities of the positive class.

    Returns
    -------
    float : Brier score

    References
    ----------
    http://en.wikipedia.org/wiki/Brier_score
    """
    y_true = _check_and_normalize(y_true, y_prob)
    return np.mean((y_true - y_prob) ** 2)

