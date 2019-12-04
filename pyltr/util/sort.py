"""

Utilities for sorting query entries.

"""

import numpy as np
import sklearn.utils


def get_sorted_y_positions(y, y_pred, check=True):
    if check:
        y = sklearn.utils.validation.column_or_1d(y)
        y_pred = sklearn.utils.validation.column_or_1d(y_pred)
        sklearn.utils.validation.check_consistent_length(y, y_pred)
    return np.array(np.lexsort((y, -y_pred)))


def get_sorted_y(y, y_pred, check=True):
    """Returns a copy of `y` sorted by position in `y_pred`.

    Parameters
    ----------
    y : array_like of shape = [n_samples_in_query]
        List of sample scores for a query.
    y_pred : array_like of shape = [n_samples_in_query]
        List of predicted scores for a query.

    Returns
    -------
    y_sorted : array_like of shape = [n_samples_in_query]
        Copy of `y` sorted by descending order of `y_pred`.
        Ties are broken in ascending order of `y`.

    """
    # return [y[i] for i in get_sorted_y_positions(y, y_pred, check=check)]

    if check:
        y = sklearn.utils.validation.column_or_1d(y)
        y_pred = sklearn.utils.validation.column_or_1d(y_pred)
        sklearn.utils.validation.check_consistent_length(y, y_pred)
    return [t[0] for t in sorted(list(zip(y, y_pred)), key=lambda t : -t[1])]
