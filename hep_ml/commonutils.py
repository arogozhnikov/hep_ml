"""
**hep_ml.commonutils** contains some helpful functions and classes
which are often used (by other modules)
"""

from __future__ import print_function, division, absolute_import

from multiprocessing.pool import ThreadPool
import numbers
import itertools

import numpy
import pandas
from scipy.special import expit
import sklearn.cross_validation
from sklearn.neighbors.unsupervised import NearestNeighbors

__author__ = "Alex Rogozhnikov"


def _threads_wrapper(func_and_args):
    func = func_and_args[0]
    args = func_and_args[1:]
    return func(*args)


def map_on_cluster(parallel_profile, *args, **kw_args):
    """
    The same as map, but the first argument is ipc_profile. Distributes the task over IPython cluster.

    :param parallel_profile: the IPython cluster profile to use.
    :type parallel_profile: None or str
    :param list args: function, arguments
    :param dict kw_args: kwargs for LoadBalacedView.map_sync

    (function copied from REP)

    :return: the result of mapping
    """
    if parallel_profile is None:
        return map(*args)
    elif str.startswith(parallel_profile, 'threads-'):
        n_threads = int(parallel_profile[len('threads-'):])
        pool = ThreadPool(processes=n_threads)
        func, params = args[0], args[1:]
        return pool.map(_threads_wrapper, zip(itertools.cycle([func]), *params))
    else:
        from IPython.parallel import Client
        return Client(profile=parallel_profile).load_balanced_view().map_sync(*args, **kw_args)


def sigmoid_function(x, width):
    """ Sigmoid function is smoothing of Heaviside function,
    the less width, the closer we are to Heaviside function
    :type x: array-like with floats, arbitrary shape
    :type width: float, if width == 0, this is simply Heaviside function
    """
    assert width >= 0, 'the width should be non-negative'
    if width > 0.0001:
        return expit(x / width)
    else:
        return (x > 0) * 1.0


def generate_sample(n_samples, n_features, distance=2.0):
    """Generates some test distribution,
    signal and background distributions are gaussian with same dispersion and different centers,
    all variables are independent (gaussian correlation matrix is identity).

    This function is frequently used in tests. """
    from sklearn.datasets import make_blobs

    centers = numpy.zeros((2, n_features))
    centers[0, :] = - distance / 2
    centers[1, :] = distance / 2

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
    columns = ["column" + str(x) for x in range(n_features)]
    X = pandas.DataFrame(X, columns=columns)
    return X, y


def check_uniform_label(uniform_label):
    """Convert uniform label to numpy.array

    :param uniform_label: label or list of labels (examples: 0, 1, [0], [1], [0, 1])
    :return: numpy.array (with [0], [1] or [0, 1])
    """
    if isinstance(uniform_label, numbers.Number):
        return numpy.array([uniform_label])
    else:
        return numpy.array(uniform_label)


def train_test_split(*arrays, **kw_args):
    """Does the same thing as train_test_split, but preserves columns in DataFrames.
    Uses the same parameters: test_size, train_size, random_state, and has the same interface

    :type list[numpy.array|pandas.DataFrame] arrays: arrays to split
    """
    assert len(arrays) > 0, "at least one array should be passed"
    length = len(arrays[0])
    for array in arrays:
        assert len(array) == length, "different size"
    train_indices, test_indices = sklearn.cross_validation.train_test_split(range(length), **kw_args)
    result = []
    for array in arrays:
        if isinstance(array, pandas.DataFrame):
            result.append(array.iloc[train_indices, :])
            result.append(array.iloc[test_indices, :])
        else:
            result.append(array[train_indices])
            result.append(array[test_indices])
    return result


def weighted_quantile(array, quantiles, sample_weight=None, array_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param array: numpy.array with data
    :param quantiles: array-like with many percentiles
    :param sample_weight: array-like of the same length as `array`
    :param array_sorted: bool, if True, then will avoid sorting
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed percentiles.
    """
    array = numpy.array(array)
    quantiles = numpy.array(quantiles)
    sample_weight = check_sample_weight(array, sample_weight)
    assert numpy.all(quantiles >= 0) and numpy.all(quantiles <= 1), 'Percentiles should be in [0, 1]'

    if not array_sorted:
        sorter = numpy.argsort(array)
        array, sample_weight = array[sorter], sample_weight[sorter]

    weighted_quantiles = numpy.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(quantiles, weighted_quantiles, array)


def build_normalizer(signal, sample_weight=None):
    """Prepares normalization function for some set of values
    transforms it to uniform distribution from [0, 1]. Example of usage:
    >>>normalizer = build_normalizer(signal)
    >>>pylab.hist(normalizer(background))
    >>># this one should be uniform in [0,1]
    >>>pylab.hist(normalizer(signal))

    :param numpy.array signal: shape = [n_samples] with floats
    :param numpy.array sample_weight: shape = [n_samples], non-negative weights associated to events.
    """
    sample_weight = check_sample_weight(signal, sample_weight)
    assert numpy.all(sample_weight >= 0.), 'sample weight must be non-negative'
    sorter = numpy.argsort(signal)
    signal, sample_weight = signal[sorter], sample_weight[sorter]
    predictions = numpy.cumsum(sample_weight) / numpy.sum(sample_weight)

    def normalizing_function(data):
        return numpy.interp(data, signal, predictions)

    return normalizing_function


def compute_cut_for_efficiency(efficiency, mask, y_pred, sample_weight=None):
    """ Computes such cut(s), that provide given target global efficiency(ies).
    Example:
    >>> p = classifier.predict_proba(X)
    >>> threshold = compute_cut_for_efficiency(0.5, mask=y == 1, y_pred=p[:, 1])

    :type efficiency: float or numpy.array with target efficiencies, shape = [n_effs]
    :type mask: array-like, shape = [n_samples], True for needed samples
    :type y_pred: array-like, shape = [n_samples], predictions or scores (float)
    :type sample_weight: None | array-like, shape = [n_samples]
    :return: float or numpy.array, shape = [n_effs]
    """
    sample_weight = check_sample_weight(mask, sample_weight)
    assert len(mask) == len(y_pred), 'lengths are different'
    efficiency = numpy.array(efficiency)
    is_signal = mask > 0.5
    y_pred, sample_weight = y_pred[is_signal], sample_weight[is_signal]
    return weighted_quantile(y_pred, 1. - efficiency, sample_weight=sample_weight)


# region Knn-related functions


def compute_knn_indices_of_signal(X, is_signal, n_neighbours=50):
    """For each event returns the knn closest signal(!) events. No matter of what class the event is.

    :type X: numpy.array, shape = [n_samples, n_features] the distance is measured over these variables
    :type is_signal: numpy.array, shape = [n_samples] with booleans
    :rtype numpy.array, shape [len(dataframe), knn], each row contains indices of closest signal events
    """
    assert len(X) == len(is_signal), "Different lengths"
    signal_indices = numpy.where(is_signal)[0]
    X_signal = numpy.array(X)[numpy.array(is_signal)]
    neighbours = NearestNeighbors(n_neighbors=n_neighbours, algorithm='kd_tree').fit(X_signal)
    _, knn_signal_indices = neighbours.kneighbors(X)
    return numpy.take(signal_indices, knn_signal_indices)


def compute_knn_indices_of_same_class(X, y, n_neighbours=50):
    """
    For each event returns the knn closest events of the same class.
    :type X: numpy.array, the distance is measured over these variables
    :type y: numpy.array, shape = [n_samples] with booleans
    :rtype numpy.array, shape [len(dataframe), knn], each row contains indices of closest signal events
    """
    assert len(X) == len(y), "different size"
    result = numpy.zeros([len(X), n_neighbours], dtype=numpy.int)
    for label in set(y):
        is_signal = y == label
        label_knn = compute_knn_indices_of_signal(X, is_signal, n_neighbours)
        result[is_signal, :] = label_knn[is_signal, :]
    return result


# endregion

def indices_of_values(array):
    """For each value in array returns indices with this value
    :param array: numpy.array with 1-dimensional initial data
    :return: sequence of tuples (value, indices_with_this_value), sequence is ordered by value
    """
    indices = numpy.argsort(array)
    sorted_array = array[indices]
    diff = numpy.nonzero(numpy.ediff1d(sorted_array))[0]
    limits = [0] + list(diff + 1) + [len(array)]
    for i in range(len(limits) - 1):
        yield sorted_array[limits[i]], indices[limits[i]: limits[i + 1]]


def take_features(X, features):
    """
    Takes features from dataset.
    NOTE: may return view to original data!

    :param X: numpy.array or pandas.DataFrame
    :param features: list of strings (if pandas.DataFrame) or list of ints
    :return: pandas.DataFrame or numpy.array with the same length.
    """
    from numbers import Number

    are_strings = all([isinstance(feature, str) for feature in features])
    are_numbers = all([isinstance(feature, Number) for feature in features])
    if are_strings and isinstance(X, pandas.DataFrame):
        return X.ix[:, features]
    elif are_numbers:
        return numpy.array(X)[:, features]
    else:
        raise NotImplementedError("Can't take features {} from object of type {}".format(features, type(X)))


def check_sample_weight(y_true, sample_weight, normalize=False, normalize_by_class=False):
    """Checks the weights, returns normalized version

    :param y_true: numpy.array of shape [n_samples]
    :param sample_weight: array-like of shape [n_samples] or None
    :param normalize: bool, if True, will scale everything to mean = 1.
    :param normalize_by_class: bool, if set, will set equal weight = 1 for each value of y_true.
        Better to use normalize if normalize_by_class is used.
    :returns: numpy.array with weights of shape [n_samples]"""
    if sample_weight is None:
        sample_weight = numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert numpy.ndim(sample_weight) == 1, 'weights vector should be 1-dimensional'
        assert len(y_true) == len(sample_weight), \
            "The length of weights is different: not {0}, but {1}".format(len(y_true), len(sample_weight))

    if normalize_by_class:
        sample_weight = numpy.copy(sample_weight)
        for value in numpy.unique(y_true):
            sample_weight[y_true == value] /= numpy.sum(sample_weight[y_true == value])

    if normalize:
        sample_weight = sample_weight / numpy.mean(sample_weight)

    return sample_weight


def check_xyw(X, y, sample_weight=None, classification=False, allow_multiple_outputs=False):
    """Checks parameters of classifier / loss / metrics.

    :param X: array-like of shape [n_samples, n_features] (numpy.array or pandas.DataFrame)
    :param y: array-like of shape [n_samples]
    :param sample_weight: None or array-like of shape [n_samples]
    :return: normalized 3-tuple (X, y, sample_weight)
    """

    y = numpy.array(y)
    if not allow_multiple_outputs:
        assert numpy.ndim(y) == 1, 'y should be one-dimensional'
    sample_weight = check_sample_weight(y, sample_weight=sample_weight)

    # only pandas.DataFrame and numpy.array are allowed. No checks on sparsity here.
    if not (isinstance(X, pandas.DataFrame) or isinstance(X, numpy.ndarray)):
        X = numpy.array(X)
    if classification:
        y = numpy.array(y, dtype=int)

    assert len(X) == len(y), 'lengths are different: {} and {}'.format(len(X), len(y))
    assert numpy.ndim(X) == 2, 'X should have 2 dimensions'

    return X, y, sample_weight


def score_to_proba(score):
    """Compute class probability estimates from decision scores.
    Uses logistic function.

    :param score: numpy.array of shape [n_samples]
    :return: probabilities, numpy.array of shape [n_samples, 2]
    """
    proba = numpy.zeros((score.shape[0], 2), dtype=numpy.float)
    proba[:, 1] = expit(score)
    proba[:, 0] = 1.0 - proba[:, 1]
    return proba


def take_last(sequence):
    """
    Returns the last element in sequence or raises an error
    """
    empty = True
    for element in sequence:
        empty = False
    if empty:
        raise IndexError('The sequence is empty.')
    else:
        return element


def to_pandas_dataframe(X):
    """
    Convert 2-dimensional array to DataFrame. If input was a DataFrame, returns itself.
    """
    if isinstance(X, pandas.DataFrame):
        return X
    else:
        return pandas.DataFrame(X, columns=['Feature{}'.format(i) for i in range(X.shape[1])])
