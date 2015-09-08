from __future__ import division, print_function, absolute_import

import numpy
from sklearn.utils.validation import column_or_1d

from .commonutils import check_sample_weight, sigmoid_function


__author__ = 'Alex Rogozhnikov'


def prepare_distribution(data, weights):
    """Prepares the distribution to be used later in KS and CvM,
    merges equal data, computes (summed) weights and cumulative distribution.
    All output arrays are of same length and correspond to each other.

    :param data: array of shape [n_samples]
    :param weights: array of shape [n_samples]
    :return: tuple with (prepared_data, prepared_weights, prepared_cdf),
        components are three parallel arrays of shape [n_unique_values]
    """
    weights = weights / numpy.sum(weights)
    prepared_data, indices = numpy.unique(data, return_inverse=True)
    prepared_weights = numpy.bincount(indices, weights=weights)
    prepared_cdf = compute_cdf(prepared_weights)
    return prepared_data, prepared_weights, prepared_cdf


# region Helpful functions to work with bins and groups

"""
There are two basic approaches to handle are bins and knn.
Here they are represented as bins and (!) groups.

The difference between bins and groups: each event belongs to one and only one bin,
in the case of groups each event may belong to several groups.
Knn is one particular case of groups, bins can be reduced to groups either

Bin_indices is an array, where for each event it's bin is written:
bin_indices = [0, 0, 1, 2, 2, 4]

Group_indices is list, each item is indices of events in some group
group_indices = [[0,1], [2], [3,4], [5]]

Group matrix is another way to write group_indices,
this is sparse matrix of shape [n_groups, n_samples],
group_matrix[group_id, sample_id] = 1, if event belong to cell, 0 otherwise


While bin indices are computed for all the events together, group indices
are typically computed only for events of some particular class.
"""


def compute_bin_indices(X_part, bin_limits=None, n_bins=20):
    """For arbitrary number of variables computes the indices of data,
    the indices are unique numbers of bin from zero to \prod_j (len(bin_limits[j])+1)

    :param X_part: columns along which binning is done
    :param bin_limits: array of edges between bins.
        If bin_limits is not provided, they are computed using data.
    :type X_part: numpy.ndarray
    """
    if bin_limits is None:
        bin_limits = []
        for variable_index in range(X_part.shape[1]):
            variable_data = X_part[:, variable_index]
            bin_limits.append(numpy.linspace(numpy.min(variable_data), numpy.max(variable_data), n_bins + 1)[1: -1])

    bin_indices = numpy.zeros(len(X_part), dtype=numpy.int)
    for axis, bin_limits_axis in enumerate(bin_limits):
        bin_indices *= (len(bin_limits_axis) + 1)
        bin_indices += numpy.searchsorted(bin_limits_axis, X_part[:, axis])

    return bin_indices


def bin_to_group_indices(bin_indices, mask):
    """ Transforms bin_indices into group indices, skips empty bins
    :type bin_indices: numpy.array, each element in index of bin this event belongs, shape = [n_samples]
    :type mask: numpy.array, boolean mask of indices to split into bins, shape = [n_samples]
    :rtype: list(numpy.array), each element is indices of elements in some bin
    """
    assert len(bin_indices) == len(mask), "Different length"
    bins_id = numpy.unique(bin_indices)
    result = list()
    for bin_id in bins_id:
        result.append(numpy.where(mask & (bin_indices == bin_id))[0])
    return result


def group_indices_to_groups_matrix(group_indices, n_events):
    """
    :param group_indices: array, each component corresponds to group
        (element = list with indices of events belonging to group)
    :return: sparse matrix of shape [n_groups, n_samples],
        one if particular event belongs to particular category, 0 otherwise
    """
    from scipy import sparse

    groups_matrix = sparse.lil_matrix((len(group_indices), n_events))
    for group_id, events_in_group in enumerate(group_indices):
        groups_matrix[group_id, events_in_group] = 1
    return sparse.csr_matrix(groups_matrix)


# endregion


# region Supplementary uniformity-related functions (to measure flatness of predictions)

def compute_cdf(ordered_weights):
    """Computes cumulative distribution function (CDF) by ordered weights,
    be sure that sum(ordered_weights) == 1.
    Minor difference: using symmetrized version
    F(x) = 1/2 (F(x-0) + F(x+0))
    """
    return numpy.cumsum(ordered_weights) - 0.5 * ordered_weights


def compute_bin_weights(bin_indices, sample_weight):
    assert len(bin_indices) == len(sample_weight), 'Different lengths of array'
    result = numpy.bincount(bin_indices, weights=sample_weight)
    return result / numpy.sum(result)


def compute_divided_weight(group_matrix, sample_weight):
    """Divided weight takes into account that different events
    are met different number of times """
    occurences = numpy.array(group_matrix.sum(axis=0)).flatten()
    return sample_weight / numpy.maximum(occurences, 1)


def compute_group_weights(group_matrix, sample_weight):
    """
    Group weight = sum of divided weights of indices inside that group.
    """
    divided_weight = compute_divided_weight(group_matrix=group_matrix, sample_weight=sample_weight)
    result = group_matrix.dot(divided_weight)
    return result / numpy.sum(result)


def compute_bin_efficiencies(y_score, bin_indices, cut, sample_weight, minlength=None):
    """Efficiency of bin = total weight of (signal) events that passed the cut
    in the bin / total weight of signal events in the bin.
    Returns small negative number for empty bins"""
    y_score = column_or_1d(y_score)
    assert len(y_score) == len(sample_weight) == len(bin_indices), "different size"
    if minlength is None:
        minlength = numpy.max(bin_indices) + 1

    bin_total = numpy.bincount(bin_indices, weights=sample_weight, minlength=minlength)
    passed_cut = y_score > cut
    bin_passed_cut = numpy.bincount(bin_indices[passed_cut],
                                    weights=sample_weight[passed_cut], minlength=minlength)
    return bin_passed_cut / numpy.maximum(bin_total, 1)


def compute_group_efficiencies_by_indices(y_score, groups_indices, cut, divided_weight=None, smoothing=0.0):
    """ Provided cut, computes efficiencies inside each bin.
    :param divided_weight: weight for each event, divided by the number of it's occurences """
    y_score = column_or_1d(y_score)
    divided_weight = check_sample_weight(y_score, sample_weight=divided_weight)
    # with smoothing=0, this is 0 or 1, latter for passed events.
    passed_cut = sigmoid_function(y_score - cut, width=smoothing)

    if isinstance(groups_indices, numpy.ndarray) and numpy.ndim(groups_indices) == 2:
        # this speedup is specially for knn
        result = numpy.average(numpy.take(passed_cut, groups_indices),
                               weights=numpy.take(divided_weight, groups_indices),
                               axis=1)
    else:
        result = numpy.zeros(len(groups_indices))
        for i, group in enumerate(groups_indices):
            result[i] = numpy.average(passed_cut[group], weights=divided_weight[group])
    return result


def compute_group_efficiencies(y_score, groups_matrix, cut, divided_weight=None, smoothing=0.0):
    """ Provided cut, computes efficiencies inside each bin.
    :param divided_weight: weight for each event, divided by the number of it's occurences """
    y_score = column_or_1d(y_score)
    divided_weight = check_sample_weight(y_score, sample_weight=divided_weight)
    # with smoothing=0, this is 0 or 1, latter for passed events.
    passed_cut = sigmoid_function(y_score - cut, width=smoothing)
    passed_weight = groups_matrix.dot(divided_weight * passed_cut)
    total_weight = groups_matrix.dot(divided_weight)
    return passed_weight / numpy.maximum(total_weight, 1e-10)


def weighted_deviation(a, weights, power=2.):
    """ sum weight * |x - x_mean|^power, measures deviation from mean """
    mean = numpy.average(a, weights=weights)
    return numpy.average(numpy.abs(mean - a) ** power, weights=weights)


# endregion


# region Special methods for uniformity metrics


def theil(x, weights):
    """Theil index of array with regularization"""
    assert numpy.all(x >= 0), "negative numbers can't be used in Theil"
    x_mean = numpy.average(x, weights=weights)
    normed = x / x_mean
    # to avoid problems with log of negative number.
    normed[normed < 1e-20] = 1e-20
    return numpy.average(normed * numpy.log(normed), weights=weights)


def _ks_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, cdf1):
    """Pay attention - prepared data should not only be sorted,
    but equal items should be merged (by summing weights),
    data2 should not have elements larger then max(prepared_data1) """
    indices = numpy.searchsorted(prepared_data1, data2)
    weights2 /= numpy.sum(weights2)
    prepared_weights2 = numpy.bincount(indices, weights=weights2, minlength=len(prepared_data1))
    cdf2 = compute_cdf(prepared_weights2)
    return numpy.max(numpy.abs(cdf1 - cdf2))


def ks_2samp_weighted(data1, data2, weights1, weights2):
    """Kolmogorov-Smirnov distance, almost the same as ks2samp from scipy.stats, but this version supports weights.

    :param data1: array-like of shape [n_samples1]
    :param data2: array-like of shape [n_samples2]
    :param weights1: None or array-like of shape [n_samples1]
    :param weights2: None or array-like of shape [n_samples2]

    :return: float, Kolmogorov-Smirnov distance.
    """
    x = numpy.unique(numpy.concatenate([data1, data2]))
    weights1 = weights1 / numpy.sum(weights1) * 1.
    weights2 = weights2 / numpy.sum(weights2) * 1.
    inds1 = numpy.searchsorted(x, data1)
    inds2 = numpy.searchsorted(x, data2)
    w1 = numpy.bincount(inds1, weights=weights1, minlength=len(x))
    w2 = numpy.bincount(inds2, weights=weights2, minlength=len(x))
    F1 = compute_cdf(w1)
    F2 = compute_cdf(w2)
    return numpy.max(numpy.abs(F1 - F2))


def _cvm_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, cdf1, power=2.):
    """Pay attention - prepared data should not only be sorted,
    but equal items should be merged (by summing weights) """
    indices = numpy.searchsorted(prepared_data1, data2)
    weights2 /= numpy.sum(weights2)
    prepared_weights2 = numpy.bincount(indices, weights=weights2, minlength=len(prepared_data1))
    cdf2 = compute_cdf(prepared_weights2)
    return numpy.average(numpy.abs(cdf1 - cdf2) ** power, weights=prepared_weights1)


# endregion


